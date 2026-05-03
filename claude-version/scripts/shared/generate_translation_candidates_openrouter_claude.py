#!/usr/bin/env python
"""Generate two Claude translation candidates for EC and/or CE datasets."""

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


DEFAULT_MODEL = "anthropic/claude-sonnet-4.6"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
RESPONSE_FIELDS = ("variant_1_better", "variant_2_slightly_weaker")
CONFIGS = {
    "ec": {
        "input": Path("ec/datasets/ffn_200ec.json"),
        "output": Path("ec/datasets/ffn_200ec.with_variants.json"),
        "errors": Path("ec/results/logs/generation/generate_ffn_200ec_claude_errors.json"),
    },
    "ce": {
        "input": Path("ce/datasets/ecpcfe_200ce.json"),
        "output": Path("ce/datasets/ecpcfe_200ce.with_variants.json"),
        "errors": Path("ce/results/logs/generation/generate_ecpcfe_200ce_claude_errors.json"),
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
            raise ValueError(f"{path}: expected a JSON array")
        return data
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def append_json(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = read_json(path)
    rows.append(row)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def done_ids(path: Path) -> set[str]:
    return {str(row.get("id")) for row in read_json(path) if row.get("id")}


def build_prompt(row: dict[str, Any]) -> str:
    if row.get("direction") == "CE":
        target_style = (
            "Write both candidates in polished professional English suitable for financial news, "
            "economic commentary, or market analysis."
        )
        quote_rule = (
            "If a translation contains quotation marks inside a JSON string, escape every internal "
            'double quote as \\" so the returned object remains valid JSON.'
        )
    else:
        target_style = (
            "两个候选译文都必须使用自然、专业的中文财经新闻/市场评论文体；"
            "不要翻译腔，不要故意写得生硬。"
        )
        quote_rule = (
            "中文译文内部需要引号时，使用中文弯引号“”或单引号，不要在译文字符串内部使用未转义的英文双引号。"
        )

    return f"""You are a senior financial translator. Create two high-quality candidate translations for an LLM-as-a-Judge benchmark.

Metadata:
- source_lang: {row.get("source_lang")}
- target_lang: {row.get("target_lang")}
- direction: {row.get("direction")}
- domain_subtype: {row.get("domain_subtype")}
- difficulty: {row.get("difficulty")}

source_text:
{row.get("source_text")}

human_translation:
{row.get("human_translation")}

General quality requirement:
- {target_style}
- Use human_translation as a quality anchor. The two candidates should look like serious professional translations at a similar level.
- Each candidate must translate the whole source_text as one complete translation string.
- Do not output notes, explanations, labels, comments, analysis, Markdown, or extra JSON keys.
- Do not split one translation across multiple JSON fields.
- Escape quotation marks correctly inside JSON strings.
- {quote_rule}
- Neither candidate should contain intentional factual errors.
- Preserve all source information: numbers, percentages, dates, entities, market direction, causality, modality, negation, conditions, attribution, and scope.
- Do not omit important information and do not add unsupported information.

variant_1_better:
- Produce a translation that is slightly better than human_translation.
- The improvement should be professional but not dramatic: smoother financial/economic phrasing, better term choice, clearer sentence organization, or more idiomatic target-language style.
- Do not over-polish, embellish, summarize, or add context not found in source_text.
- It should still feel like a faithful translation of the same source, not a rewritten article.

variant_2_slightly_weaker:
- Produce an accurate translation that is only slightly weaker than human_translation.
- The weakness must be subtle and quality-adjacent, not an obvious defect.
- Acceptable weaknesses include: slightly less elegant sentence rhythm, a somewhat less idiomatic but still correct term, mildly flatter style, or less concise sentence organization.
- Do not create factual errors, terminology mistakes, mistranslated relationships, number changes, omissions, additions, broken grammar, awkward literalness, or any low-level signal that makes the candidate easy to reject.
- It should remain a plausible professional candidate and should not be clearly worse at a glance.

Return exactly this JSON object and nothing else:
{{
  "variant_1_better": "complete candidate translation here",
  "variant_2_slightly_weaker": "complete candidate translation here"
}}
"""


def extract_json_object(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    starts = [i for i, char in enumerate(text) if char == "{"]
    ends = [i for i, char in enumerate(text) if char == "}"]
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


def strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def trim_jsonish_value(value: str) -> str:
    value = value.strip()
    value = re.sub(r",?\s*$", "", value)
    if value.startswith('"'):
        value = value[1:]
    if value.endswith('"'):
        value = value[:-1]
    return value.strip()


def parse_jsonish_fields(text: str) -> dict[str, str]:
    """Recover Claude's JSON-shaped output when internal quotes are unescaped."""
    text = strip_markdown_fence(text)
    key1 = '"variant_1_better"'
    key2 = '"variant_2_slightly_weaker"'
    idx1 = text.find(key1)
    idx2 = text.find(key2)
    if idx1 < 0 or idx2 < 0 or idx2 <= idx1:
        raise ValueError(f"model did not return a JSON object: {text[:800]}")

    colon1 = text.find(":", idx1 + len(key1))
    colon2 = text.find(":", idx2 + len(key2))
    if colon1 < 0 or colon2 < 0:
        raise ValueError(f"model response missing field separators: {text[:800]}")

    raw1 = text[colon1 + 1 : idx2]
    raw2 = text[colon2 + 1 :]
    raw1 = re.sub(r'"\s*,\s*$', '"', raw1.strip(), flags=re.DOTALL)
    raw2 = re.sub(r"\s*}\s*$", "", raw2.strip(), flags=re.DOTALL)
    parsed = {
        "variant_1_better": trim_jsonish_value(raw1),
        "variant_2_slightly_weaker": trim_jsonish_value(raw2),
    }
    if not parsed["variant_1_better"] or not parsed["variant_2_slightly_weaker"]:
        raise ValueError(f"model response has empty recovered fields: {text[:800]}")
    return parsed


def parse_model_json(text: str) -> dict[str, str]:
    try:
        data = extract_json_object(text)
    except ValueError:
        return parse_jsonish_fields(text)
    missing = [field for field in RESPONSE_FIELDS if field not in data]
    if missing:
        raise ValueError(f"model response missing fields {missing}: {text[:800]}")
    extra = [key for key in data if key not in RESPONSE_FIELDS]
    if extra:
        raise ValueError(f"model response has extra fields {extra}: {text[:800]}")
    parsed = {}
    for field in RESPONSE_FIELDS:
        value = data[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"model response field is empty or not string: {field}")
        parsed[field] = value.strip()
    return parsed


def call_openrouter(
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    timeout: int,
    temperature: float,
    max_tokens: int,
) -> dict[str, str]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a professional financial translator."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "finance-translation-candidate-generation",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]
    return parse_model_json(content)


def call_with_retries(
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    timeout: int,
    temperature: float,
    max_tokens: int,
    retries: int,
) -> dict[str, str]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return call_openrouter(api_key, base_url, model, prompt, timeout, temperature, max_tokens)
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


def generate_direction(direction: str, args: argparse.Namespace, api_key: str) -> tuple[int, int, int]:
    config = CONFIGS[direction]
    input_path = args.input or config["input"]
    output_path = args.output or config["output"]
    errors_path = args.errors or config["errors"]
    if args.restart:
        for path in [output_path, errors_path]:
            if path.exists():
                path.unlink()

    rows = read_json(input_path)
    if args.limit is not None:
        rows = rows[: args.limit]
    completed = done_ids(output_path)
    print(f"[{direction.upper()}] input: {input_path}")
    print(f"[{direction.upper()}] output: {output_path}")
    print(f"[{direction.upper()}] rows: {len(rows)}, already done: {len(completed)}")

    ok = failed = skipped = 0
    for index, row in enumerate(rows, start=1):
        sample_id = str(row.get("id"))
        if sample_id in completed:
            skipped += 1
            continue
        print(f"[{direction.upper()} {index}/{len(rows)}] {sample_id}")
        try:
            generated = call_with_retries(
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                prompt=build_prompt(row),
                timeout=args.timeout,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                retries=args.retries,
            )
            append_json(
                output_path,
                {
                    **row,
                    **generated,
                    "generation_model": args.model,
                    "generation_provider": "openrouter",
                    "generation_timestamp": now(),
                },
            )
            completed.add(sample_id)
            ok += 1
        except Exception as exc:  # noqa: BLE001 - keep the batch going.
            failed += 1
            append_json(errors_path, {"id": sample_id, "error": str(exc), "timestamp": now()})
            print(f"  failed: {exc}")
    print(f"[{direction.upper()}] done: ok={ok}, skipped={skipped}, failed={failed}")
    return ok, skipped, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("direction", choices=["ec", "ce", "all"])
    parser.add_argument("--input", type=Path, default=None, help="Override input path for a single direction.")
    parser.add_argument("--output", type=Path, default=None, help="Override output path for a single direction.")
    parser.add_argument("--errors", type=Path, default=None, help="Override error path for a single direction.")
    parser.add_argument("--model", default=os.getenv("OPENROUTER_GENERATION_MODEL", DEFAULT_MODEL))
    parser.add_argument("--base-url", default=os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.direction == "all" and any([args.input, args.output, args.errors]):
        raise SystemExit("--input/--output/--errors can only be used with a single direction")

    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        api_key = getpass.getpass("Enter OPENROUTER_API_KEY: ").strip()
    if not api_key:
        print("No API key provided.")
        return 2

    directions = ["ec", "ce"] if args.direction == "all" else [args.direction]
    total_failed = 0
    for direction in directions:
        _ok, _skipped, failed = generate_direction(direction, args, api_key)
        total_failed += failed
    return 1 if total_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
