#!/usr/bin/env python
"""Validate final benchmark files and write a Markdown report."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_arabic_numbers(text: str) -> list[str]:
    return re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", text or "")


def count_duplicate_candidates(rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        candidates = [row.get("candidate_A"), row.get("candidate_B"), row.get("candidate_C")]
        if len(set(candidates)) < 3:
            count += 1
    return count


def count_numeric_anomalies(rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        source_numbers = set(extract_arabic_numbers(row.get("source_text") or ""))
        if not source_numbers:
            continue
        for label in ["A", "B", "C"]:
            candidate_numbers = set(extract_arabic_numbers(row.get(f"candidate_{label}") or ""))
            missing = source_numbers - candidate_numbers
            if len(missing) > 1 or len(missing) > max(1, len(source_numbers) // 3):
                count += 1
                break
    return count


def format_counter(counter: Counter[Any]) -> str:
    if not counter:
        return "- None\n"
    return "".join(f"- {key}: {value}\n" for key, value in sorted(counter.items(), key=lambda kv: str(kv[0])))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, default=Path("data/final/finance_laj_benchmark_full.jsonl"))
    parser.add_argument("--failures", type=Path, default=Path("data/interim/ffn_generation_failures.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("data/final/dataset_report.md"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.full)
    failures = read_jsonl(args.failures)

    position_counter = Counter()
    for row in rows:
        for label in ["A", "B", "C"]:
            if row.get(f"candidate_{label}"):
                position_counter[label] += 1

    report = [
        "# Finance LAJ Benchmark Dataset Report",
        "",
        f"- Total samples: {len(rows)}",
        f"- CE count: {sum(1 for row in rows if row.get('direction') == 'CE')}",
        f"- EC count: {sum(1 for row in rows if row.get('direction') == 'EC')}",
        f"- Failed generation samples: {len(failures)}",
        f"- Duplicate candidate rows: {count_duplicate_candidates(rows)}",
        f"- Numeric anomaly rows: {count_numeric_anomalies(rows)}",
        "",
        "## A/B/C Position Distribution",
        format_counter(position_counter).rstrip(),
        "",
        "## human_preferred_label Distribution",
        format_counter(Counter(row.get("human_preferred_label") for row in rows)).rstrip(),
        "",
        "## clean_weaker_label Distribution",
        format_counter(Counter(row.get("clean_weaker_label") for row in rows)).rstrip(),
        "",
        "## subtle_error_label Distribution",
        format_counter(Counter(row.get("subtle_error_label") for row in rows)).rstrip(),
        "",
        "## domain_subtype Distribution",
        format_counter(Counter(row.get("domain_subtype") for row in rows)).rstrip(),
        "",
        "## difficulty Distribution",
        format_counter(Counter(row.get("difficulty") for row in rows)).rstrip(),
        "",
        "## variant_2_error_type Distribution",
        format_counter(Counter(row.get("variant_2_error_type") for row in rows)).rstrip(),
        "",
    ]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
