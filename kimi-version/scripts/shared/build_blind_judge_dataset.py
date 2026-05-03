#!/usr/bin/env python
"""Build full and blind three-candidate judge datasets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


LABELS = ["A", "B", "C"]


def read_json(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path}: expected a JSON array")
        return data
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/interim/ffn_with_generated_variants.json"))
    parser.add_argument("--full-output", type=Path, default=Path("data/final/finance_laj_benchmark_full.json"))
    parser.add_argument("--blind-output", type=Path, default=Path("data/final/finance_laj_benchmark_blind.json"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    rows = read_json(args.input)
    full_rows: list[dict[str, Any]] = []
    blind_rows: list[dict[str, Any]] = []

    print(f"Loaded {len(rows)} generated rows from {args.input}")
    for row in rows:
        candidates = [
            ("human", row["human_translation"]),
            ("clean_weaker", row["variant_1_clean_weaker"]),
            ("subtle_error", row["variant_2_subtle_error"]),
        ]
        rng.shuffle(candidates)
        label_by_kind = {kind: LABELS[i] for i, (kind, _text) in enumerate(candidates)}
        candidate_by_label = {LABELS[i]: text for i, (_kind, text) in enumerate(candidates)}

        common = {
            "id": row.get("id"),
            "direction": row.get("direction"),
            "source_lang": row.get("source_lang"),
            "target_lang": row.get("target_lang"),
            "source_text": row.get("source_text"),
            "candidate_A": candidate_by_label["A"],
            "candidate_B": candidate_by_label["B"],
            "candidate_C": candidate_by_label["C"],
            "domain_subtype": row.get("domain_subtype"),
            "difficulty": row.get("difficulty"),
        }

        full_rows.append(
            {
                **common,
                "human_preferred_label": label_by_kind["human"],
                "clean_weaker_label": label_by_kind["clean_weaker"],
                "subtle_error_label": label_by_kind["subtle_error"],
                "human_translation_original": row.get("human_translation"),
                "variant_1_clean_weaker": row.get("variant_1_clean_weaker"),
                "variant_2_subtle_error": row.get("variant_2_subtle_error"),
                "variant_2_error_type": row.get("variant_2_error_type"),
                "variant_2_error_explanation": row.get("variant_2_error_explanation"),
                "corpus_source": row.get("corpus_source"),
                "source_sheet": row.get("source_sheet"),
                "source_row": row.get("source_row"),
            }
        )
        blind_rows.append(common)

    write_json(args.full_output, full_rows)
    write_json(args.blind_output, blind_rows)
    print(f"Wrote full dataset: {args.full_output}")
    print(f"Wrote blind dataset: {args.blind_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
