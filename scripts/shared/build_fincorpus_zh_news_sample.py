#!/usr/bin/env python
"""Stream-sample cleaner Chinese financial-news texts from FinCorpus.

The FinCorpus news file is large, so this script reads the gzip stream from
Hugging Face and stops once enough acceptable texts have been collected. It
does not save the full upstream archive.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URL = (
    "https://huggingface.co/datasets/Duxiaoman-DI/FinCorpus/resolve/main/"
    "data/fin_news_data_final.jsonl.gz"
)
DEFAULT_OUT_DIR = ROOT / "data" / "raw" / "fincorpus"

HARD_EXCLUDE_PATTERNS = [
    r"投资者提问",
    r"董秘回答",
    r"查看更多董秘问答",
    r"免责声明",
    r"不构成投资建议",
    r"风险自担",
    r"本站",
    r"下载.{0,8}APP",
    r"返回搜狐",
    r"查看更多",
    r"财富小精灵",
    r"写稿机器人",
    r"\[公告\]",
    r"公告编号",
    r"证券代码",
    r"证券简称",
    r"本公司及董事会",
    r"本公司董事会",
    r"特此公告",
    r"中财网",
    r"财华社讯",
    r"操作建议",
    r"止损",
    r"止盈",
    r"今日操作",
    r"打板",
    r"核按钮",
    r"龙虎榜",
    r"主力资金",
    r"涨停原因",
    r"历史涨停",
    r"Choice数据",
    r"短期上涨概率",
    r"低吸",
    r"追高",
    r"抄底",
    r"做T",
    r"反包",
    r"标的",
    r"介入良机",
    r"后续大概率",
    r"题材",
    r"研报机构",
    r"研报作者",
    r"报告类型",
    r"目标价",
    r"评级为",
    r"买入评级",
    r"本站不推荐任何股票",
    r"什么品牌好",
    r"创业商机",
    r"加盟",
    r"轻钢别墅",
    r"点作者头像关注",
    r"黑马猎人",
    r"立即查看",
    r"点击【?阅读原文",
    r"微信号",
    r"商务合作",
    r"传播不易",
    r"小编",
    r"欢迎给.*打赏",
    r"赏金不求多",
    r"扫描二维码",
    r"关注我们",
    r"投稿",
    r"导读\s",
]

SOFT_NOISE_PATTERNS = [
    r"\d{6}[）)]",
    r"[Ss][Zz]\d{6}",
    r"[Ss][Hh]\d{6}",
    r"\(\d{5}-[Hh][Kk]\)",
    r"行情\d{6}",
    r"诊股",
]

TEXT_KEYS = ("text", "content", "article", "body", "正文")


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


def sentence_punctuation_count(text: str) -> int:
    return text.count("。") + text.count("！") + text.count("？")


def extract_text(row: dict[str, Any]) -> str:
    for key in TEXT_KEYS:
        if row.get(key):
            return normalize_text(row[key])
    return ""


def reject_reason(
    text: str,
    min_chars: int,
    max_chars: int,
    min_cjk_ratio: float,
    max_digit_ratio: float,
    min_sentence_punct: int,
) -> str | None:
    if not (min_chars <= len(text) <= max_chars):
        return "length"
    if any(re.search(pattern, text) for pattern in HARD_EXCLUDE_PATTERNS):
        return "hard_noise"
    if cjk_count(text) / max(len(text), 1) < min_cjk_ratio:
        return "cjk_ratio"
    if len(re.findall(r"\d", text)) / max(len(text), 1) > max_digit_ratio:
        return "digit_ratio"
    if sentence_punctuation_count(text) < min_sentence_punct:
        return "sentence_count"
    if text.count("：") > 8 or text.count("【") > 4:
        return "punctuation_noise"
    soft_hits = sum(1 for pattern in SOFT_NOISE_PATTERNS if re.search(pattern, text))
    if soft_hits >= 2 and (text.count("股") > 6 or text.count("涨") > 6):
        return "stock_ticker_noise"
    if text.count("。") and len(text) / max(text.count("。"), 1) < 35:
        return "fragmented"
    return None


def stream_sample(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    seen_prefixes: set[str] = set()
    scanned = 0
    fields_seen: Counter[str] = Counter()
    reject_counts: Counter[str] = Counter()

    request = urllib.request.Request(
        args.url,
        headers={"User-Agent": "llm-as-a-judge-fincorpus-sampler/1.0"},
    )
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        with gzip.GzipFile(fileobj=response) as gz:
            for raw_line in gz:
                scanned += 1
                if scanned > args.max_scan or len(accepted) >= args.target:
                    break
                try:
                    row = json.loads(raw_line.decode("utf-8"))
                except Exception:
                    reject_counts["json_decode"] += 1
                    continue
                if isinstance(row, dict):
                    fields_seen.update(row.keys())
                else:
                    reject_counts["not_object"] += 1
                    continue

                text = extract_text(row)
                reason = reject_reason(
                    text,
                    args.min_chars,
                    args.max_chars,
                    args.min_cjk_ratio,
                    args.max_digit_ratio,
                    args.min_sentence_punct,
                )
                if reason:
                    reject_counts[reason] += 1
                    continue

                dedupe_key = re.sub(r"\W+", "", text)[:160]
                if dedupe_key in seen_prefixes:
                    reject_counts["duplicate_prefix"] += 1
                    continue
                seen_prefixes.add(dedupe_key)

                meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
                accepted.append(
                    {
                        "id": f"FINCORPUS_ZH_NEWS_{len(accepted) + 1:04d}",
                        "text": text,
                        "char_count": len(text),
                        "cjk_count": cjk_count(text),
                        "source": meta.get("source") or row.get("source") or "",
                        "meta": meta,
                    }
                )

    metadata = {
        "dataset": "Duxiaoman-DI/FinCorpus",
        "url": args.url,
        "streamed_only": True,
        "downloaded_full_file": False,
        "target": args.target,
        "accepted": len(accepted),
        "scanned_rows": scanned,
        "filters": {
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "min_cjk_ratio": args.min_cjk_ratio,
            "max_digit_ratio": args.max_digit_ratio,
            "min_sentence_punct": args.min_sentence_punct,
        },
        "top_level_fields_seen": fields_seen.most_common(30),
        "reject_counts": reject_counts.most_common(),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return accepted, metadata


def write_outputs(rows: list[dict[str, Any]], metadata: dict[str, Any], out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{prefix}.jsonl"
    txt_path = out_dir / f"{prefix}.txt"
    metadata_path = out_dir / f"{prefix}.metadata.json"

    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with txt_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(row["text"] + "\n")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(f"wrote {len(rows)} rows to {jsonl_path}")
    print(f"wrote clean text to {txt_path}")
    print(f"wrote metadata to {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--target", type=int, default=400)
    parser.add_argument("--max-scan", type=int, default=500_000)
    parser.add_argument("--min-chars", type=int, default=220)
    parser.add_argument("--max-chars", type=int, default=650)
    parser.add_argument("--min-cjk-ratio", type=float, default=0.58)
    parser.add_argument("--max-digit-ratio", type=float, default=0.22)
    parser.add_argument("--min-sentence-punct", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", default="fin_news_zh_clean_400")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, metadata = stream_sample(args)
    write_outputs(rows, metadata, args.out_dir, args.prefix)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    if len(rows) < args.target:
        print(f"warning: accepted {len(rows)} rows, below target {args.target}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
