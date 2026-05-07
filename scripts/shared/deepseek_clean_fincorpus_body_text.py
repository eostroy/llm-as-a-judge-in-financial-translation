#!/usr/bin/env python
"""Clean non-body boilerplate from DeepSeek-screened FinCorpus records.

This pass keeps already accepted records, calls DeepSeek once per row, and asks
it to remove only datelines, source labels, reporter/editor signatures, app
suffixes, image captions, and similar material outside the article body.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "data"
    / "raw"
    / "fincorpus"
    / "deepseek_screened"
    / "fin_news_deepseek_400.accepted.jsonl"
)
DEFAULT_OUT_DIR = ROOT / "data" / "raw" / "fincorpus" / "deepseek_screened"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


SYSTEM_PROMPT = """你是中文财经新闻正文清洗员。你的任务不是改写文章，而是删除正文之外的元信息。
只输出一个 JSON 对象，不要输出解释性正文。"""

USER_PROMPT_TEMPLATE = """请清洗下面的中文财经新闻文本，只删除不属于正文的部分。

需要删除的典型内容包括：
1. 开头电头/讯头/来源与客户端标记，例如“新华社北京X月X日电”“中新经纬客户端X月X日电”“上海证券报讯”“格隆汇X月X日丨”等。
2. 括号中的记者、作者、通讯员、编辑、责任编辑、来源说明，例如“（记者 张三）”“（中新经纬APP）”“责任编辑：李四”。
3. 图片说明、图表说明、截图来源、来源：Wind、资料图、视觉中国等非正文说明。
4. 文末下载 App、关注公众号、免责声明、投资建议提示、版权声明、推广语。
5. 与正文无关的孤立栏目名、标签、时间戳、网页残留。

必须保留：
1. 正文事实、引用、数字、公司名、机构名、市场表现、财务数据。
2. 正文中自然出现的“据Wind数据显示”“记者采访时表示”等内容，只要它是句子的一部分。
3. 原文语序和表达方式。不要润色，不要补写，不要翻译，不要概括。

如果文本已经基本干净，原样返回。
如果删除后正文过短或没有正文，返回 usable=false。

请输出 JSON，字段固定为：
{{
  "usable": true/false,
  "cleaned_text": "清洗后的正文",
  "removed": ["删除了什么类型的非正文内容"],
  "reason": "一句中文说明"
}}

待清洗文本：
<<<
{text}
>>>"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(text: Any) -> str:
    value = str(text or "")
    value = re.sub(r"<[^>]+>", "", value)
    value = (
        value.replace("&nbsp;", " ")
        .replace("\u00a0", " ")
        .replace("\u3000", " ")
        .replace("\\/", "/")
    )
    return re.sub(r"\s+", " ", value).strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def make_record_key(index: int, row: dict[str, Any]) -> str:
    record_id = str(row.get("id") or f"row_{index:04d}")
    text = normalize_text(row.get("text", ""))
    return f"{index:04d}|{record_id}|{text_hash(text)}"


def load_completed_keys(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_key = row.get("record_key")
            if record_key:
                completed.add(str(record_key))
    return completed


def response_content(data: dict[str, Any]) -> str:
    message = data["choices"][0]["message"]
    content = message.get("content")
    if isinstance(content, str):
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


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("DeepSeek response is not a JSON object")
    return data


def normalize_review(data: dict[str, Any], original_text: str) -> dict[str, Any]:
    usable = bool(data.get("usable", True))
    cleaned = normalize_text(data.get("cleaned_text", ""))
    if usable and not cleaned:
        cleaned = original_text
    removed = data.get("removed", [])
    if not isinstance(removed, list):
        removed = [str(removed)]
    if not usable:
        cleaned = ""
    return {
        "usable": usable,
        "cleaned_text": cleaned,
        "removed": [str(item) for item in removed],
        "reason": str(data.get("reason", "")),
    }


def call_deepseek(
    api_key: str,
    base_url: str,
    model: str,
    text: str,
    timeout: int,
    max_tokens: int,
    temperature: float,
    use_response_format: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)},
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
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return normalize_review(extract_json_object(response_content(data)), text)


def call_with_retries(
    api_key: str,
    base_url: str,
    model: str,
    text: str,
    timeout: int,
    max_tokens: int,
    temperature: float,
    retries: int,
    use_response_format: bool,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return call_deepseek(
                api_key=api_key,
                base_url=base_url,
                model=model,
                text=text,
                timeout=timeout,
                max_tokens=max_tokens,
                temperature=temperature,
                use_response_format=use_response_format,
            )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"DeepSeek cleaning failed after retries: {last_error}")


def run(args: argparse.Namespace, api_key: str) -> int:
    output_jsonl = args.out_dir / f"{args.prefix}.jsonl"
    output_txt = args.out_dir / f"{args.prefix}.txt"
    reviews_path = args.out_dir / f"{args.prefix}.reviews.jsonl"
    errors_path = args.out_dir / f"{args.prefix}.errors.jsonl"
    metadata_path = args.out_dir / f"{args.prefix}.metadata.json"
    if args.restart:
        for path in [output_jsonl, output_txt, reviews_path, errors_path, metadata_path]:
            if path.exists():
                path.unlink()

    rows = load_jsonl(args.input_jsonl)
    completed = load_completed_keys(reviews_path)
    stats: dict[str, Any] = {
        "started_at_utc": now(),
        "input_jsonl": str(args.input_jsonl),
        "total_input_rows": len(rows),
        "initial_completed": len(completed),
        "processed": 0,
        "skipped_completed": 0,
        "usable": 0,
        "unusable": 0,
        "errors": 0,
        "model": args.model,
    }

    for index, row in enumerate(rows, start=1):
        record_id = str(row.get("id") or f"row_{index:04d}")
        record_key = make_record_key(index, row)
        if record_key in completed:
            stats["skipped_completed"] += 1
            continue
        text = normalize_text(row.get("text", ""))
        if not text:
            stats["errors"] += 1
            append_jsonl(errors_path, {"id": record_id, "error": "empty text", "timestamp_utc": now()})
            continue
        print(f"[{index}/{len(rows)}] cleaning {record_id}", flush=True)
        try:
            if args.dry_run:
                review = {
                    "usable": True,
                    "cleaned_text": text,
                    "removed": [],
                    "reason": "dry run",
                }
            else:
                review = call_with_retries(
                    api_key=api_key,
                    base_url=args.base_url,
                    model=args.model,
                    text=text,
                    timeout=args.timeout,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    retries=args.retries,
                    use_response_format=not args.no_response_format,
                )
            review_record = {
                "id": record_id,
                "record_key": record_key,
                "input_row_number": index,
                "source_id": row.get("source_id"),
                "input_text_hash": text_hash(text),
                "original_char_count": len(text),
                "cleaned_char_count": len(review["cleaned_text"]),
                "review": {
                    "usable": review["usable"],
                    "removed": review["removed"],
                    "reason": review["reason"],
                },
                "timestamp_utc": now(),
            }
            append_jsonl(reviews_path, review_record)
            completed.add(record_key)
            stats["processed"] += 1
            if review["usable"]:
                cleaned_row = dict(row)
                cleaned_row["input_row_number"] = index
                cleaned_row["pre_body_clean_text"] = row.get("text", "")
                cleaned_row["text"] = review["cleaned_text"]
                cleaned_row["body_cleaning"] = {
                    "removed": review["removed"],
                    "reason": review["reason"],
                    "input_text_hash": text_hash(text),
                    "cleaned_text_hash": text_hash(review["cleaned_text"]),
                    "cleaned_at_utc": now(),
                }
                cleaned_row["char_count"] = len(review["cleaned_text"])
                append_jsonl(output_jsonl, cleaned_row)
                with output_txt.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(review["cleaned_text"] + "\n")
                stats["usable"] += 1
            else:
                stats["unusable"] += 1
        except Exception as exc:  # noqa: BLE001 - batch should continue.
            stats["errors"] += 1
            append_jsonl(
                errors_path,
                {
                    "id": record_id,
                    "record_key": record_key,
                    "input_row_number": index,
                    "source_id": row.get("source_id"),
                    "error": str(exc),
                    "text": text,
                    "timestamp_utc": now(),
                },
            )
            print(f"  failed: {exc}", flush=True)

    stats["finished_at_utc"] = now()
    stats["final_reviewed_rows"] = len(load_completed_keys(reviews_path))
    stats["output_files"] = {
        "cleaned_jsonl": str(output_jsonl),
        "cleaned_txt": str(output_txt),
        "reviews_jsonl": str(reviews_path),
        "errors_jsonl": str(errors_path),
    }
    metadata_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    return 0 if stats["errors"] == 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", default="fin_news_deepseek_400.body_cleaned")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=os.getenv("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL))
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-response-format", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key and not args.dry_run:
        api_key = getpass.getpass("Enter DEEPSEEK_API_KEY: ").strip()
    if not api_key and not args.dry_run:
        print("No DeepSeek API key provided.")
        return 2
    return run(args, api_key or "")


if __name__ == "__main__":
    raise SystemExit(main())
