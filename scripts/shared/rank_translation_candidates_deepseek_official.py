#!/usr/bin/env python
"""Rank translation candidates with the official DeepSeek API.

The source JSON is never modified. Results are appended to model-specific
ranking JSON files so interrupted runs can be resumed safely.

Examples:
  python scripts/shared/rank_translation_candidates_deepseek_official.py --direction ec
  python scripts/shared/rank_translation_candidates_deepseek_official.py --direction all --models deepseek-v4-flash

Set DEEPSEEK_API_KEY in the environment, or pass --api-key.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_API_KEY = "sk-71ffb35e193e46d8acbb2817617e9401"  # Optional: paste your official DeepSeek API key here.
DEFAULT_PROMPT = Path("prompts/rank_candidates_prompt.txt")
DEFAULT_MODELS = "deepseek-v4-flash"
CANDIDATE_FIELDS = ["candidate_A", "candidate_B", "candidate_C"]
RESULT_FIELD = "rank"

CONFIGS = {
    "ec": {
        "input": Path("ec/datasets/ffn_200ec.with_candidates.shuffled.json"),
        "output_dir": Path("ec/results/model_based_metrics/rankings/json"),
    },
    "ce": {
        "input": Path("ce/datasets/ecpcfe_200ce.with_candidates.shuffled.json"),
        "output_dir": Path("ce/results/model_based_metrics/rankings/json"),
    },
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON array")
        return data

    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no} is not valid JSON: {exc}") from exc
    return rows


def write_json(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_json(path: Path, row: dict[str, Any]) -> None:
    rows = read_json(path)
    rows.append(row)
    write_json(path, rows)


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("id")) for row in read_json(path) if row.get("id")}


def safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model).strip("._-") or "model"


def default_output_path(input_path: Path, output_dir: Path, model: str) -> Path:
    return output_dir / f"{input_path.stem}.ranked.deepseek__{safe_model_name(model)}.json"


def build_prompt(template: str, row: dict[str, Any]) -> str:
    sample = {
        "id": row.get("id"),
        "direction": row.get("direction"),
        "source_lang": row.get("source_lang"),
        "target_lang": row.get("target_lang"),
        "source_text": row.get("source_text"),
        "candidates": {
            "candidate_A": row.get("candidate_A"),
            "candidate_B": row.get("candidate_B"),
            "candidate_C": row.get("candidate_C"),
        },
    }
    return (
        template.strip()
        + "\n\n"
        + "Sample to rank follows. Return JSON only.\n"
        + json.dumps(sample, ensure_ascii=False, indent=2)
    )


def extract_json_object(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    starts = [idx for idx, char in enumerate(text) if char == "{"]
    ends = [idx for idx, char in enumerate(text) if char == "}"]
    for start in reversed(starts):
        for end in reversed(ends):
            if end <= start:
                continue
            try:
                data = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
    raise ValueError(f"model did not return a JSON object: {text[:800]}")


def normalize_candidate_name(value: str) -> str:
    aliases = {
        "a": "candidate_A",
        "candidate a": "candidate_A",
        "candidate_a": "candidate_A",
        "candidate_A": "candidate_A",
        "b": "candidate_B",
        "candidate b": "candidate_B",
        "candidate_b": "candidate_B",
        "candidate_B": "candidate_B",
        "c": "candidate_C",
        "candidate c": "candidate_C",
        "candidate_c": "candidate_C",
        "candidate_C": "candidate_C",
    }
    key = value.strip()
    return aliases.get(key.lower(), key)


def normalize_ranking(result: dict[str, Any]) -> dict[str, int]:
    ranking = result.get("ranking") or result.get("rank") or result.get("ordered_candidates")
    if ranking is None and all(normalize_candidate_name(str(key)) in CANDIDATE_FIELDS for key in result):
        ranking = result

    if isinstance(ranking, dict):
        rank = {}
        for key, value in ranking.items():
            candidate = normalize_candidate_name(str(key))
            rank[candidate[-1]] = int(value)
        if sorted(rank) != ["A", "B", "C"] or sorted(rank.values()) != [1, 2, 3]:
            raise ValueError(f"rank must map A/B/C to 1/2/3 exactly; got {rank}")
        return rank

    if isinstance(ranking, str):
        ranking = [part.strip() for part in re.split(r"[,>，\s]+", ranking) if part.strip()]
    if not isinstance(ranking, list):
        raise ValueError(f"missing ranking list in result: {result}")

    normalized = [normalize_candidate_name(str(item)) for item in ranking]
    if sorted(normalized) != sorted(CANDIDATE_FIELDS):
        raise ValueError(f"ranking must contain exactly {CANDIDATE_FIELDS}; got {normalized}")
    return {candidate[-1]: rank for rank, candidate in enumerate(normalized, start=1)}


def response_content(data: dict[str, Any]) -> str:
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"unexpected DeepSeek response: {data}") from exc

    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                pieces.append(item["text"])
            elif isinstance(item, str):
                pieces.append(item)
        if pieces:
            return "\n".join(pieces)
    raise ValueError(f"DeepSeek response message has no text content: {message}")


def call_deepseek(
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    timeout: int,
    max_tokens: int,
    temperature: float,
    use_response_format: bool,
    thinking: str,
) -> dict[str, int]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a careful financial translation evaluator."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if use_response_format:
        payload["response_format"] = {"type": "json_object"}
    if thinking != "omit":
        payload["thinking"] = {"type": thinking}

    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return normalize_ranking(extract_json_object(response_content(data)))


def call_with_retries(
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    timeout: int,
    max_tokens: int,
    temperature: float,
    retries: int,
    use_response_format: bool,
    thinking: str,
) -> dict[str, int]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return call_deepseek(
                api_key=api_key,
                base_url=base_url,
                model=model,
                prompt=prompt,
                timeout=timeout,
                max_tokens=max_tokens,
                temperature=temperature,
                use_response_format=use_response_format,
                thinking=thinking,
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {exc.code} {exc.reason}: {detail}")
        except (TimeoutError, urllib.error.URLError, json.JSONDecodeError, KeyError, ValueError) as exc:
            last_error = exc

        if attempt < retries:
            wait_s = min(30 * attempt, 180) if "429" in str(last_error) else min(2**attempt, 30)
            print(f"  attempt {attempt} failed: {last_error}; retrying in {wait_s}s")
            time.sleep(wait_s)
    raise RuntimeError(f"failed after {retries} attempts: {last_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", choices=["ec", "ce", "all"], default="ec")
    parser.add_argument("--input", type=Path, default=None, help="Override input JSON; only valid for one direction.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory; only valid for one direction.")
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--models", default=DEFAULT_MODELS, help="Comma-separated official DeepSeek model ids.")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=os.getenv("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--thinking", choices=["omit", "disabled", "enabled"], default="omit")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--no-response-format", action="store_true")
    return parser.parse_args()


def run_direction(args: argparse.Namespace, direction: str, model: str, prompt_template: str, api_key: str) -> None:
    cfg = CONFIGS[direction]
    input_path = args.input or cfg["input"]
    output_dir = args.output_dir or cfg["output_dir"]
    rows = read_json(input_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    output_path = default_output_path(input_path, output_dir, model)
    error_path = output_path.with_suffix(".errors.json")
    if args.restart:
        for path in [output_path, error_path]:
            if path.exists():
                path.unlink()

    completed = done_ids(output_path)
    print(f"\ndirection: {direction}")
    print(f"model: {model}")
    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"rows: {len(rows)}, already done: {len(completed)}")

    ok = 0
    failed = 0
    skipped = 0
    for index, row in enumerate(rows, start=1):
        sample_id = str(row.get("id"))
        if sample_id in completed:
            skipped += 1
            continue

        missing = [field for field in CANDIDATE_FIELDS if field not in row]
        if missing:
            failed += 1
            append_json(error_path, {"id": sample_id, "error": f"missing candidate fields: {missing}", "timestamp": now()})
            continue

        print(f"[{index}/{len(rows)}] {sample_id}")
        try:
            ranking = call_with_retries(
                api_key=api_key,
                base_url=args.base_url,
                model=model,
                prompt=build_prompt(prompt_template, row),
                timeout=args.timeout,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                retries=args.retries,
                use_response_format=not args.no_response_format,
                thinking=args.thinking,
            )
            append_json(output_path, {**row, RESULT_FIELD: ranking})
            completed.add(sample_id)
            ok += 1
        except Exception as exc:  # noqa: BLE001 - batch should continue.
            failed += 1
            append_json(error_path, {"id": sample_id, "error": str(exc), "timestamp": now()})
            print(f"  failed: {exc}")

    print(f"done for {direction}/{model}: ok={ok}, skipped={skipped}, failed={failed}")


def main() -> int:
    args = parse_args()
    if args.direction == "all" and (args.input is not None or args.output_dir is not None):
        raise ValueError("--input/--output-dir overrides can only be used with --direction ec or --direction ce")

    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY") or DEEPSEEK_API_KEY
    if not api_key:
        api_key = getpass.getpass("Enter DEEPSEEK_API_KEY: ").strip()
    if not api_key:
        print("No DeepSeek API key provided.")
        return 2

    prompt_template = args.prompt_file.read_text(encoding="utf-8")
    models = [model.strip() for model in args.models.split(",") if model.strip()]
    if not models:
        raise ValueError("--models is empty")

    directions = ["ec", "ce"] if args.direction == "all" else [args.direction]
    for model in models:
        for direction in directions:
            run_direction(args, direction, model, prompt_template, api_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
