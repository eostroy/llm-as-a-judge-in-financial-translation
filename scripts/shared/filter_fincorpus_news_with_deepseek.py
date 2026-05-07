#!/usr/bin/env python
"""Use DeepSeek to screen FinCorpus Chinese financial-news candidates.

The script can either review an existing JSONL candidate file or stream the
large FinCorpus gzip file directly. It calls the official DeepSeek chat API for
each coarse-filtered row, records every review decision, and writes accepted
texts to a compact JSONL/TXT pair. It is designed to be restartable: completed
text hashes in the review log are skipped on later runs.
"""

from __future__ import annotations

import argparse
import gzip
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
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URL = (
    "https://huggingface.co/datasets/Duxiaoman-DI/FinCorpus/resolve/main/"
    "data/fin_news_data_final.jsonl.gz"
)
DEFAULT_OUT_DIR = ROOT / "data" / "raw" / "fincorpus" / "deepseek_screened"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"

TEXT_KEYS = ("text", "content", "article", "body", "正文")
COARSE_EXCLUDE = [
    "投资者提问",
    "董秘回答",
    "免责声明",
    "不构成投资建议",
    "风险自担",
    "财富小精灵",
    "写稿机器人",
    "特此公告",
    "本公司及董事会",
    "公告编号",
    "操作建议",
    "止损",
    "止盈",
    "今日操作",
    "打板",
    "核按钮",
    "加盟",
    "什么品牌好",
    "商务合作",
    "微信号",
    "打赏",
]


SYSTEM_PROMPT = """你是中文财经新闻语料筛选员。你的任务是判断给定中文文本是否适合作为“中译英财经新闻翻译评测”的中文原文。

请只输出一个 JSON 对象，不要输出解释性正文。"""

USER_PROMPT_TEMPLATE = """请判断下面这段文本能否作为 CE（Chinese-to-English）财经新闻翻译评测的中文原文。

合格标准：
1. 必须是自然、连贯、原生中文财经新闻/财经资讯正文，适合翻译成英文。
2. 内容应偏宏观经济、金融市场、公司经营、产业经济、银行/保险/证券/基金、国际经贸、上市公司新闻等。
3. 文本应自足，不依赖图片、表格或上下文才能理解。
4. 长度大致适合翻译评测，不是标题或短讯碎片。

不合格标准：
1. 董秘问答、投资者互动、股评荐股、交易复盘、操作建议、广告软文、招商加盟、公众号导流、免责声明。
2. 纯公告模板、法律声明、证券代码/公告编号堆砌、表格残留严重。
3. 数字/股票代码/列表过密，翻译价值低。
4. 语句破碎、重复、乱码、明显拼接错误。
5. 内容不是财经新闻，或者过于口语化/营销化。

如果只有少量来源署名、原标题、记者名等尾巴，可以在 cleaned_text 中删除；但不要改写正文，不要补充信息。

请输出 JSON，字段固定为：
{{
  "usable": true/false,
  "score": 0-100,
  "category": "macro|market|company|industry|finance|trade|other|reject",
  "issues": ["..."],
  "reason": "一句中文理由",
  "cleaned_text": "若 usable=true，给出清理后的原文；若不合格，返回空字符串"
}}

待判断文本：
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
    value = re.sub(r"\s+", " ", value).strip()
    return value


def cjk_count(text: str) -> int:
    return sum("\u4e00" <= char <= "\u9fff" for char in text)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_text(row: dict[str, Any]) -> str:
    for key in TEXT_KEYS:
        if row.get(key):
            return normalize_text(row[key])
    return ""


def coarse_eligible(text: str, args: argparse.Namespace) -> bool:
    if not (args.min_chars <= len(text) <= args.max_chars):
        return False
    if any(term in text for term in COARSE_EXCLUDE):
        return False
    if cjk_count(text) / max(len(text), 1) < args.min_cjk_ratio:
        return False
    if len(re.findall(r"\d", text)) / max(len(text), 1) > args.max_digit_ratio:
        return False
    if text.count("。") + text.count("！") + text.count("？") < args.min_sentence_punct:
        return False
    return True


def iter_input_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "id" not in row:
                row["id"] = f"{path.stem}_{line_no:08d}"
            yield row


def iter_fincorpus_stream(url: str, timeout: int) -> Iterator[dict[str, Any]]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "llm-as-a-judge-deepseek-fincorpus-filter/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        with gzip.GzipFile(fileobj=response) as gz:
            for line_no, raw_line in enumerate(gz, start=1):
                try:
                    row = json.loads(raw_line.decode("utf-8"))
                except Exception:
                    continue
                if isinstance(row, dict):
                    row.setdefault("id", f"FINCORPUS_STREAM_{line_no:08d}")
                    row["_stream_row"] = line_no
                    yield row


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
    usable = bool(data.get("usable"))
    score = int(data.get("score", 0))
    score = max(0, min(100, score))
    category = str(data.get("category", "reject")).strip() or "reject"
    issues = data.get("issues", [])
    if not isinstance(issues, list):
        issues = [str(issues)]
    cleaned = normalize_text(data.get("cleaned_text", ""))
    if usable and not cleaned:
        cleaned = original_text
    if not usable:
        cleaned = ""
        category = "reject"
    return {
        "usable": usable,
        "score": score,
        "category": category,
        "issues": [str(item) for item in issues],
        "reason": str(data.get("reason", "")),
        "cleaned_text": cleaned,
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
    for attempt in range(1, retries + 1):
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


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_reviewed_hashes(path: Path) -> set[str]:
    reviewed: set[str] = set()
    if not path.exists():
        return reviewed
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("text_hash"):
                reviewed.add(str(row["text_hash"]))
    return reviewed


def load_accepted_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def iter_rows(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    if args.input_jsonl:
        return iter_input_jsonl(args.input_jsonl)
    return iter_fincorpus_stream(args.url, args.stream_timeout)


def run(args: argparse.Namespace, api_key: str) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = args.out_dir / f"{args.prefix}.accepted.jsonl"
    accepted_txt_path = args.out_dir / f"{args.prefix}.accepted.txt"
    reviews_path = args.out_dir / f"{args.prefix}.reviews.jsonl"
    errors_path = args.out_dir / f"{args.prefix}.errors.jsonl"
    metadata_path = args.out_dir / f"{args.prefix}.metadata.json"

    if args.restart:
        for path in [accepted_path, accepted_txt_path, reviews_path, errors_path, metadata_path]:
            if path.exists():
                path.unlink()

    reviewed_hashes = load_reviewed_hashes(reviews_path)
    accepted_count = load_accepted_count(accepted_path)

    stats = {
        "started_at_utc": now(),
        "input_jsonl": str(args.input_jsonl) if args.input_jsonl else None,
        "url": None if args.input_jsonl else args.url,
        "target": args.target,
        "initial_accepted": accepted_count,
        "initial_reviewed": len(reviewed_hashes),
        "scanned": 0,
        "coarse_rejected": 0,
        "skipped_reviewed": 0,
        "model_reviewed": 0,
        "model_accepted": 0,
        "model_rejected": 0,
        "errors": 0,
        "model": args.model,
    }

    print(f"accepted output: {accepted_path}")
    print(f"review log: {reviews_path}")
    print(f"starting accepted={accepted_count}, reviewed={len(reviewed_hashes)}")

    for row in iter_rows(args):
        if accepted_count >= args.target:
            break
        if args.max_scan is not None and stats["scanned"] >= args.max_scan:
            break

        stats["scanned"] += 1
        original_text = extract_text(row)
        if not coarse_eligible(original_text, args):
            stats["coarse_rejected"] += 1
            continue

        h = text_hash(original_text)
        if h in reviewed_hashes:
            stats["skipped_reviewed"] += 1
            continue

        source_id = str(row.get("id") or f"row_{stats['scanned']}")
        print(f"[scan={stats['scanned']} accepted={accepted_count}/{args.target}] {source_id}")
        try:
            if args.dry_run:
                review = {
                    "usable": True,
                    "score": 100,
                    "category": "other",
                    "issues": [],
                    "reason": "dry_run",
                    "cleaned_text": original_text,
                }
            else:
                review = call_with_retries(
                    api_key=api_key,
                    base_url=args.base_url,
                    model=args.model,
                    text=original_text,
                    timeout=args.timeout,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    retries=args.retries,
                    use_response_format=not args.no_response_format,
                )
            stats["model_reviewed"] += 1
            reviewed_hashes.add(h)
            review_record = {
                "source_id": source_id,
                "text_hash": h,
                "original_text": original_text,
                "original_char_count": len(original_text),
                "stream_row": row.get("_stream_row"),
                "review": review,
                "timestamp_utc": now(),
            }
            append_jsonl(reviews_path, review_record)

            if review["usable"] and review["score"] >= args.min_score:
                accepted_count += 1
                accepted_row = {
                    "id": f"FINCORPUS_DEEPSEEK_ZH_NEWS_{accepted_count:04d}",
                    "source_id": source_id,
                    "text": review["cleaned_text"],
                    "original_text": original_text,
                    "char_count": len(review["cleaned_text"]),
                    "cjk_count": cjk_count(review["cleaned_text"]),
                    "score": review["score"],
                    "category": review["category"],
                    "issues": review["issues"],
                    "reason": review["reason"],
                    "text_hash": h,
                    "source": (row.get("meta") or {}).get("source") if isinstance(row.get("meta"), dict) else row.get("source"),
                    "meta": row.get("meta") if isinstance(row.get("meta"), dict) else {},
                }
                append_jsonl(accepted_path, accepted_row)
                with accepted_txt_path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(accepted_row["text"] + "\n")
                stats["model_accepted"] += 1
            else:
                stats["model_rejected"] += 1
        except Exception as exc:  # noqa: BLE001 - batch should continue.
            stats["errors"] += 1
            append_jsonl(
                errors_path,
                {
                    "source_id": source_id,
                    "text_hash": h,
                    "error": str(exc),
                    "text": original_text,
                    "timestamp_utc": now(),
                },
            )
            print(f"  failed: {exc}")

    stats["finished_at_utc"] = now()
    stats["final_accepted"] = accepted_count
    stats["output_files"] = {
        "accepted_jsonl": str(accepted_path),
        "accepted_txt": str(accepted_txt_path),
        "reviews_jsonl": str(reviews_path),
        "errors_jsonl": str(errors_path),
    }
    metadata_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0 if accepted_count >= args.target else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=None, help="Existing candidate JSONL to review.")
    parser.add_argument("--url", default=DEFAULT_URL, help="FinCorpus .jsonl.gz URL used when --input-jsonl is omitted.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", default="fin_news_deepseek_screened")
    parser.add_argument("--target", type=int, default=400)
    parser.add_argument("--max-scan", type=int, default=100_000)
    parser.add_argument("--min-score", type=int, default=80)
    parser.add_argument("--min-chars", type=int, default=180)
    parser.add_argument("--max-chars", type=int, default=800)
    parser.add_argument("--min-cjk-ratio", type=float, default=0.50)
    parser.add_argument("--max-digit-ratio", type=float, default=0.35)
    parser.add_argument("--min-sentence-punct", type=int, default=2)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=os.getenv("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL))
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--stream-timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Do not call DeepSeek; accept coarse-filtered rows.")
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
