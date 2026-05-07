#!/usr/bin/env python
"""Extend the ECPCFE CE base dataset from 200 to 400 rows.

This script keeps the existing 200-row dataset unchanged, samples another
200 non-overlapping source spans from the same ECPCFE candidate pool, and
writes both the additional subset and the merged 400-row base dataset.
"""

from __future__ import annotations

import importlib.util
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ORIGINAL = Path("ce/datasets/ecpcfe_200ce.json")
ADDITIONAL_OUTPUT = Path("ce/datasets/ecpcfe_200ce_additional.json")
COMBINED_OUTPUT = Path("ce/datasets/ecpcfe_400ce.json")
SEED = 20260506
ADDITIONAL_SIZE = 200


def load_builder():
    script_path = Path(__file__).with_name("build_ecpcfe_200ce_dataset.py")
    spec = importlib.util.spec_from_file_location("build_ecpcfe_200ce_dataset", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (row["source_book"], int(row["source_segment_start"]), int(row["source_segment_end"]))


def candidate_key(item: dict[str, Any]) -> tuple[str, int, int]:
    return (item["book"], int(item["source_segment_start"]), int(item["source_segment_end"]))


def mark_segments(used_segments: dict[str, set[int]], book: str, start: int, end: int) -> None:
    for segment in range(start, end + 1):
        used_segments[book].add(segment)


def has_used_segment(used_segments: dict[str, set[int]], item: dict[str, Any]) -> bool:
    return any(
        segment in used_segments[item["book"]]
        for segment in range(int(item["source_segment_start"]), int(item["source_segment_end"]) + 1)
    )


def make_row(item: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "id": f"ECPCFE_CE_{index:04d}",
        "direction": "CE",
        "source_lang": "Chinese",
        "target_lang": "English",
        "source_text": item["source_text"],
        "human_translation": item["human_translation"],
        "domain_subtype": item["domain_subtype"],
        "difficulty": item["difficulty"],
        "corpus_source": "ECPCFE",
        "source_book": item["book"],
        "source_segment_start": item["source_segment_start"],
        "source_segment_end": item["source_segment_end"],
        "zh_char_count": item["zh_char_count"],
        "en_word_count": item["en_word_count"],
    }


def sample_additional(
    builder: Any,
    candidates: list[dict[str, Any]],
    original_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    original_keys = {source_key(row) for row in original_rows}
    used_segments: dict[str, set[int]] = defaultdict(set)
    for row in original_rows:
        mark_segments(
            used_segments,
            row["source_book"],
            int(row["source_segment_start"]),
            int(row["source_segment_end"]),
        )

    domain_targets = builder.target_counts(original_rows, "domain_subtype")
    difficulty_targets = builder.target_counts(original_rows, "difficulty")
    joint_targets = builder.allocate_joint_targets(domain_targets, difficulty_targets)

    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        if candidate_key(item) in original_keys:
            continue
        by_cell[(item["domain_subtype"], item["difficulty"])].append(item)
    for items in by_cell.values():
        rng.shuffle(items)

    selected: list[dict[str, Any]] = []
    for domain in builder.DOMAIN_ORDER:
        for difficulty in ["easy", "medium", "hard"]:
            need = joint_targets.get((domain, difficulty), 0)
            picked = 0
            for item in by_cell.get((domain, difficulty), []):
                if picked >= need:
                    break
                if has_used_segment(used_segments, item):
                    continue
                selected.append(item)
                picked += 1
                mark_segments(
                    used_segments,
                    item["book"],
                    int(item["source_segment_start"]),
                    int(item["source_segment_end"]),
                )
            if picked != need:
                raise RuntimeError(f"only picked {picked}/{need} for {(domain, difficulty)}")

    if len(selected) != ADDITIONAL_SIZE:
        raise RuntimeError(f"selected {len(selected)} rows, expected {ADDITIONAL_SIZE}")

    rng.shuffle(selected)
    start_index = len(original_rows) + 1
    return [make_row(item, index) for index, item in enumerate(selected, start=start_index)]


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    builder = load_builder()
    reference_rows = builder.read_json(builder.EC_REFERENCE)
    original_rows = builder.read_json(ORIGINAL)
    candidates = builder.build_candidates(reference_rows)
    additional_rows = sample_additional(builder, candidates, original_rows)
    combined_rows = original_rows + additional_rows

    if len(combined_rows) != 400:
        raise RuntimeError(f"combined dataset has {len(combined_rows)} rows, expected 400")
    if len({source_key(row) for row in combined_rows}) != 400:
        raise RuntimeError("combined dataset contains duplicate source spans")
    if len({row["id"] for row in combined_rows}) != 400:
        raise RuntimeError("combined dataset contains duplicate ids")

    write_json(ADDITIONAL_OUTPUT, additional_rows)
    write_json(COMBINED_OUTPUT, combined_rows)

    print(f"candidate pool: {len(candidates)}")
    print(f"wrote {len(additional_rows)} rows to {ADDITIONAL_OUTPUT}")
    print(f"wrote {len(combined_rows)} rows to {COMBINED_OUTPUT}")
    print("additional domain:", dict(Counter(row["domain_subtype"] for row in additional_rows)))
    print("additional difficulty:", dict(Counter(row["difficulty"] for row in additional_rows)))
    print("combined domain:", dict(Counter(row["domain_subtype"] for row in combined_rows)))
    print("combined difficulty:", dict(Counter(row["difficulty"] for row in combined_rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
