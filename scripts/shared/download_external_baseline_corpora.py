#!/usr/bin/env python
"""Download small native finance text baselines for translationese calibration.

The script intentionally stores both row-level JSONL provenance and one-text-per-line
clean files. It uses a small Hugging Face-hosted CSV, so it does not require the
`datasets` package or large full-corpus downloads.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "data" / "external_baselines"

KENPACHE_DATASET = "Kenpache/multilingual-financial-sentiment"
KENPACHE_CSV_URL = (
    "https://huggingface.co/datasets/"
    "Kenpache/multilingual-financial-sentiment/resolve/main/all_languages_clean.csv"
)


def fetch_text(url: str, retries: int = 4, pause: float = 2.0) -> str:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "llm-as-a-judge-baseline-builder/1.0",
                    "Accept": "text/csv,text/plain,*/*",
                },
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read().decode("utf-8")
        except Exception as exc:  # pragma: no cover - network dependent.
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def cjk_count(text: str) -> int:
    return sum(1 for char in text if "\u4e00" <= char <= "\u9fff")


def repair_mojibake(text: str, expected_language: str) -> str:
    """Repair common HF dataset-card mojibake while keeping already-good text."""
    candidates = [text]
    for encoding in ("latin1", "cp1252", "gbk"):
        try:
            candidates.append(text.encode(encoding).decode("utf-8"))
        except UnicodeError:
            pass

    if expected_language == "zh":
        return max(candidates, key=lambda value: (cjk_count(value), -value.count("\ufffd")))

    return max(candidates, key=lambda value: (-value.count("\ufffd"), len(value)))


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_text_lines(path: Path, records: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            text = normalize_text(record["text"])
            if text:
                handle.write(text + "\n")
                count += 1
    return count


def collect_kenpache_language(
    language: str, limit: int, start_offset: int = 0
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    csv_text = fetch_text(KENPACHE_CSV_URL)
    scanned_rows = 0

    for row_idx, row in enumerate(csv.DictReader(io.StringIO(csv_text))):
        scanned_rows += 1
        if row_idx < start_offset:
            continue
        if row.get("language") != language:
            continue
        text = repair_mojibake(str(row.get("sentence", "")), language)
        text = normalize_text(text)
        if language == "zh" and cjk_count(text) < 10:
            continue
        if language == "en" and len(text.split()) < 8:
            continue
        collected.append(
            {
                "row_idx": row_idx,
                "dataset": KENPACHE_DATASET,
                "language": language,
                "source": row.get("source"),
                "label": row.get("label"),
                "text": text,
                "original_sentence": row.get("sentence"),
            }
        )
        if len(collected) >= limit:
            break

    meta = {
        "dataset": KENPACHE_DATASET,
        "dataset_url": f"https://huggingface.co/datasets/{KENPACHE_DATASET}",
        "source_file_url": KENPACHE_CSV_URL,
        "config": "default",
        "split": "train",
        "language_filter": language,
        "requested_limit": limit,
        "start_offset": start_offset,
        "collected": len(collected),
        "scanned_rows": scanned_rows,
        "text_field": "sentence",
        "notes": "Financial news/sentiment sentences; mojibake repaired with UTF-8 round-trip heuristics when needed.",
    }
    return collected, meta


def save_corpus(subdir: str, records: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    out_dir = OUT_ROOT / subdir
    raw_dir = out_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        **metadata,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records),
    }
    write_jsonl(raw_dir / "records.jsonl", records)
    write_text_lines(out_dir / "clean.txt", records)
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def save_readme() -> None:
    readme = """# External Native Finance Baselines

These corpora are saved for target-language baseline normalization of translationese-style features.

## zh_native_finance_kenpache

- Source: `Kenpache/multilingual-financial-sentiment`
- URL: https://huggingface.co/datasets/Kenpache/multilingual-financial-sentiment
- Selection: rows with `language == "zh"`
- Clean text: one financial-news sentence per line

## en_native_finance_kenpache

- Source: `Kenpache/multilingual-financial-sentiment`
- URL: https://huggingface.co/datasets/Kenpache/multilingual-financial-sentiment
- Selection: rows with `language == "en"`
- Clean text: one financial-news sentence per line

Each subdirectory contains:

- `clean.txt`: normalized one-text-per-line baseline input
- `raw/records.jsonl`: row-level provenance and retained source fields
- `metadata.json`: retrieval date, source, split, field choices, and counts
"""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "README.md").write_text(readme, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zh-limit", type=int, default=5000)
    parser.add_argument("--en-limit", type=int, default=5000)
    parser.add_argument(
        "--zh-start-offset",
        type=int,
        default=7900,
        help="Kenpache rows are grouped by language; zh begins around this offset.",
    )
    args = parser.parse_args()

    zh_records, zh_meta = collect_kenpache_language(
        "zh", args.zh_limit, args.zh_start_offset
    )
    en_records, en_meta = collect_kenpache_language("en", args.en_limit)

    save_corpus("zh_native_finance_kenpache", zh_records, zh_meta)
    save_corpus("en_native_finance_kenpache", en_records, en_meta)
    save_readme()

    print(f"Saved {len(zh_records)} zh records to {OUT_ROOT / 'zh_native_finance_kenpache'}")
    print(f"Saved {len(en_records)} en records to {OUT_ROOT / 'en_native_finance_kenpache'}")


if __name__ == "__main__":
    main()
