#!/usr/bin/env python
"""Rank three translation candidates with one or more OpenRouter models.

The source JSONL is never modified. Each model writes to its own output JSONL.
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


OPENROUTER_API_KEY = ""  # Optional: paste your OpenRouter API key here.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_INPUT = Path("ffn_finance_200ec.with_variants.jsonl")
DEFAULT_PROMPT = Path("prompts/rank_candidates_prompt.txt")
DEFAULT_OUTPUT_DIR = Path("openrouter_rankings")
CANDIDATE_FIELDS = ["human_translation", "variant_1_better", "variant_2_slightly_weaker"]
RESULT_FIELD = "openrouter_ranking"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON: {exc}") from exc
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("id")) for row in read_jsonl(path) if row.get("id")}


def safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model).strip("._-") or "model"


def default_output_path(input_path: Path, output_dir: Path, model: str) -> Path:
    return output_dir / f"{input_path.stem}.ranked.{safe_model_name(model)}.jsonl"


def build_prompt(template: str, row: dict[str, Any]) -> str:
    sample = {
        "id": row.get("id"),
        "direction": row.get("direction"),
        "source_lang": row.get("source_lang"),
        "target_lang": row.get("target_lang"),
        "source_text": row.get("source_text"),
        "domain_subtype": row.get("domain_subtype"),
        "difficulty": row.get("difficulty"),
        "candidates": {
            "human_translation": row.get("human_translation"),
            "variant_1_better": row.get("variant_1_better"),
            "variant_2_slightly_weaker": row.get("variant_2_slightly_weaker"),
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

    starts = [i for i, c in enumerate(text) if c == "{"]
    ends = [i for i, c in enumerate(text) if c == "}"]
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


def normalize_ranking(result: dict[str, Any]) -> dict[str, Any]:
    ranking = result.get("ranking") or result.get("rank") or result.get("ordered_candidates")
    if isinstance(ranking, str):
        ranking = [part.strip() for part in re.split(r"[,>，\s]+", ranking) if part.strip()]
    if not isinstance(ranking, list):
        raise ValueError(f"missing ranking list in result: {result}")

    aliases = {
        "human": "human_translation",
        "reference": "human_translation",
        "human_translation": "human_translation",
        "variant_1": "variant_1_better",
        "variant_1_better": "variant_1_better",
        "better": "variant_1_better",
        "variant_2": "variant_2_slightly_weaker",
        "variant_2_slightly_weaker": "variant_2_slightly_weaker",
        "slightly_weaker": "variant_2_slightly_weaker",
    }
    normalized = []
    for item in ranking:
        key = str(item).strip()
        key = aliases.get(key, key)
        normalized.append(key)

    if sorted(normalized) != sorted(CANDIDATE_FIELDS):
        raise ValueError(f"ranking must contain exactly {CANDIDATE_FIELDS}; got {normalized}")

    return {
        "ranking": normalized,
        "best": normalized[0],
        "worst": normalized[-1],
        "reason": result.get("reason", ""),
        "raw_result": result,
    }


def call_openrouter(
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    timeout: int,
    max_tokens: int,
    temperature: float,
    use_response_format: bool,
) -> dict[str, Any]:
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

    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "finance-translation-ranking",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]
    return normalize_ranking(extract_json_object(content))


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
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return call_openrouter(
                api_key=api_key,
                base_url=base_url,
                model=model,
                prompt=prompt,
                timeout=timeout,
                max_tokens=max_tokens,
                temperature=temperature,
                use_response_format=use_response_format,
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {exc.code} {exc.reason}: {detail}")
        except (TimeoutError, urllib.error.URLError, json.JSONDecodeError, KeyError, ValueError) as exc:
            last_error = exc

        if attempt < retries:
            wait_s = min(2**attempt, 20)
            print(f"  attempt {attempt} failed: {last_error}; retrying in {wait_s}s")
            time.sleep(wait_s)
    raise RuntimeError(f"failed after {retries} attempts: {last_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--models", required=True, help="Comma-separated OpenRouter model ids.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--no-response-format", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = OPENROUTER_API_KEY or args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        api_key = getpass.getpass("Enter OPENROUTER_API_KEY: ").strip()
    if not api_key:
        print("No OpenRouter API key provided.")
        return 2

    rows = read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]

    prompt_template = args.prompt_file.read_text(encoding="utf-8")
    models = [model.strip() for model in args.models.split(",") if model.strip()]
    if not models:
        raise ValueError("--models is empty")

    for model in models:
        output_path = default_output_path(args.input, args.output_dir, model)
        error_path = output_path.with_suffix(".errors.jsonl")
        if args.restart:
            for path in [output_path, error_path]:
                if path.exists():
                    path.unlink()

        completed = done_ids(output_path)
        print(f"\nmodel: {model}")
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
                append_jsonl(error_path, {"id": sample_id, "error": f"missing candidate fields: {missing}", "timestamp": now()})
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
                )
                append_jsonl(
                    output_path,
                    {
                        **row,
                        RESULT_FIELD: {
                            "model": model,
                            "timestamp": now(),
                            **ranking,
                        },
                    },
                )
                completed.add(sample_id)
                ok += 1
            except Exception as exc:  # noqa: BLE001 - batch should continue.
                failed += 1
                append_jsonl(error_path, {"id": sample_id, "error": str(exc), "timestamp": now()})
                print(f"  failed: {exc}")

        print(f"done for {model}: ok={ok}, skipped={skipped}, failed={failed}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
