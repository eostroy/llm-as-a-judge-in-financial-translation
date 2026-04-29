#!/usr/bin/env python
"""Generate two extra translation candidates for each row in an FFN JSONL file.

Default input:
  ffn_finance_200ec.jsonl

Default output:
  ffn_finance_200ec.with_variants.jsonl
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL = "kimi-k2.5"
BASE_URL = "https://api.moonshot.cn/v1"
MOONSHOT_API_KEY = ""  # Optional: paste your API key here for local runs. Do not commit real keys.
RESPONSE_FIELDS = [
    "variant_1_better",
    "variant_2_slightly_weaker",
]


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


def build_prompt(row: dict[str, Any]) -> str:
    if row.get("direction") == "CE":
        target_style = (
            "Write both candidates in polished professional English suitable for financial news, "
            "earnings commentary, or market analysis."
        )
    else:
        target_style = (
            "两个候选译文都必须使用自然、专业的中文财经新闻/市场评论文体；"
            "不要翻译腔，不要故意写得生硬。"
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
- Neither candidate should contain intentional factual errors.
- Preserve all source information: numbers, percentages, dates, entities, market direction, causality, modality, negation, conditions, attribution, and scope.
- Do not omit important information and do not add unsupported information.

variant_1_better:
- Produce a translation that is slightly better than human_translation.
- The improvement should be professional but not dramatic: smoother financial-news phrasing, better term choice, clearer sentence organization, or more idiomatic target-language style.
- Do not over-polish, embellish, summarize, or add context not found in source_text.
- It should still feel like a faithful translation of the same source, not a rewritten article.

variant_2_slightly_weaker:
- Produce an accurate translation that is only slightly weaker than human_translation.
- The weakness must be subtle and quality-adjacent, not an obvious defect.
- Acceptable weaknesses include: slightly less elegant sentence rhythm, a somewhat less idiomatic but still correct financial term, mildly flatter style, or less concise sentence organization.
- Do not create factual errors, terminology mistakes, mistranslated relationships, number changes, omissions, additions, broken grammar, awkward literalness, or any low-level signal that makes the candidate easy to reject.
- It should remain a plausible professional candidate and should not be clearly worse at a glance.

Return exactly this JSON object and nothing else:
{{
  "variant_1_better": "complete candidate translation here",
  "variant_2_slightly_weaker": "complete candidate translation here"
}}
"""


def parse_model_json(text: str) -> dict[str, Any]:
    data = load_json_object(text)
    if not isinstance(data, dict):
        raise ValueError("model response is not a JSON object")
    data = merge_split_fields(data)
    missing = [field for field in RESPONSE_FIELDS if field not in data]
    if missing:
        preview = text[:1000].replace("\n", "\\n")
        raise ValueError(f"model response missing fields: {missing}; raw={preview}")
    extra = sorted(set(data) - set(RESPONSE_FIELDS))
    allowed_extra = {field + suffix for field in RESPONSE_FIELDS for suffix in ["_continued", "_conclusion"]}
    allowed_extra.add("")
    allowed_extra.add("\n  ")
    extra = [key for key in extra if key not in allowed_extra and str(key).strip()]
    if extra:
        preview = text[:1000].replace("\n", "\\n")
        raise ValueError(f"model response has extra fields: {extra}; raw={preview}")
    for field in RESPONSE_FIELDS:
        if not isinstance(data[field], str) or not data[field].strip():
            preview = text[:1000].replace("\n", "\\n")
            raise ValueError(f"model response field is empty or not string: {field}; raw={preview}")
    return {field: data[field].strip() for field in RESPONSE_FIELDS}


def load_json_object(text: str) -> dict[str, Any]:
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
            candidate = text[start : end + 1]
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and all(field in data for field in RESPONSE_FIELDS):
                return data
    raise json.JSONDecodeError("could not find a valid JSON object in model response", text, 0)


def merge_split_fields(data: dict[str, Any]) -> dict[str, Any]:
    merged = dict(data)
    for field in RESPONSE_FIELDS:
        pieces = []
        for suffix in ["", "_continued", "_conclusion"]:
            value = data.get(field + suffix)
            if isinstance(value, str) and value.strip():
                pieces.append(value.strip())
        if pieces:
            merged[field] = "".join(pieces)
    return merged


def call_kimi(api_key: str, base_url: str, model: str, prompt: str, timeout: int) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a professional financial translation evaluator and translator.",
            },
            {"role": "user", "content": prompt},
        ],
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
        "max_completion_tokens": 8192,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    content = result["choices"][0]["message"]["content"]
    return parse_model_json(content)


def call_with_retries(
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return call_kimi(api_key, base_url, model, prompt, timeout)
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            last_error = RuntimeError(f"HTTP {exc.code} {exc.reason}: {detail}")
            if attempt < retries:
                wait_s = min(2**attempt, 8)
                print(f"  attempt {attempt} failed: {last_error}; retrying in {wait_s}s")
                time.sleep(wait_s)
        except (TimeoutError, urllib.error.URLError, json.JSONDecodeError, KeyError, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                wait_s = min(2**attempt, 8)
                print(f"  attempt {attempt} failed: {exc}; retrying in {wait_s}s")
                time.sleep(wait_s)
    raise RuntimeError(f"failed after {retries} attempts: {last_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("ffn_finance_200ec.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("ffn_finance_200ec.with_variants.jsonl"))
    parser.add_argument("--errors", type=Path, default=Path("generation_errors.jsonl"))
    parser.add_argument("--base-url", default=os.getenv("MOONSHOT_BASE_URL", BASE_URL))
    parser.add_argument("--model", default=os.getenv("MOONSHOT_MODEL", MODEL))
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--restart", action="store_true", help="Delete output/error files and start from row 1.")
    parser.add_argument("--api-key", default=None, help="Moonshot API key. If omitted, env var or prompt is used.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = MOONSHOT_API_KEY or args.api_key or os.getenv("MOONSHOT_API_KEY")
    if not api_key:
        api_key = getpass.getpass("Enter MOONSHOT_API_KEY: ").strip()
    if not api_key:
        print("No API key provided.")
        return 2

    if args.restart:
        for path in [args.output, args.errors]:
            if path.exists():
                path.unlink()

    rows = read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]

    completed = done_ids(args.output)
    print(f"input: {args.input}")
    print(f"output: {args.output}")
    print(f"errors: {args.errors}")
    print(f"model: {args.model}")
    print(f"base_url: {args.base_url}")
    print(f"mode: {'restart' if args.restart else 'resume'}")
    print(f"rows: {len(rows)}, already done: {len(completed)}")

    ok = 0
    failed = 0
    skipped = 0
    for i, row in enumerate(rows, start=1):
        sample_id = str(row.get("id"))
        if sample_id in completed:
            skipped += 1
            continue

        print(f"[{i}/{len(rows)}] {sample_id}")
        try:
            generated = call_with_retries(
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                prompt=build_prompt(row),
                timeout=args.timeout,
                retries=args.retries,
            )
            append_jsonl(
                args.output,
                {
                    **row,
                    **generated,
                    "generation_model": args.model,
                    "generation_timestamp": now(),
                },
            )
            completed.add(sample_id)
            ok += 1
        except Exception as exc:  # noqa: BLE001 - keep the batch going.
            failed += 1
            append_jsonl(args.errors, {"id": sample_id, "error": str(exc), "timestamp": now()})
            print(f"  failed: {exc}")

    print(f"done: ok={ok}, skipped={skipped}, failed={failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
