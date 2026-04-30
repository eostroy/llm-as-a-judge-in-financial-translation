#!/usr/bin/env python
"""Create a copy of the EC test set with candidate positions shuffled."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("benchmark/ffn_200ec.with_candidates.json")
DEFAULT_OUTPUT = Path("benchmark/ffn_200ec.with_candidates.shuffled.json")
DEFAULT_SEED = 20260429
CANDIDATES = ("A", "B", "C")


def read_json(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path}: expected a JSON array")
        return data
    rows = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def shuffle_rows(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    shuffled_rows = []
    for row in rows:
        original_values = {letter: row[f"candidate_{letter}"] for letter in CANDIDATES}
        original_letters = list(CANDIDATES)
        rng.shuffle(original_letters)
        new_row = {key: value for key, value in row.items() if key not in {f"candidate_{letter}" for letter in CANDIDATES}}
        position_map = {}
        for new_letter, original_letter in zip(CANDIDATES, original_letters):
            new_row[f"candidate_{new_letter}"] = original_values[original_letter]
            position_map[new_letter] = original_letter
        new_row["candidate_position_map"] = position_map
        shuffled_rows.append(new_row)
    return shuffled_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    rows = read_json(args.input)
    shuffled = shuffle_rows(rows, args.seed)
    write_json(args.output, shuffled)
    counts = {letter: 0 for letter in CANDIDATES}
    for row in shuffled:
        for new_letter, original_letter in row["candidate_position_map"].items():
            if new_letter == original_letter:
                counts[new_letter] += 1
    print(f"wrote {len(shuffled)} rows to {args.output}")
    print(f"same-position counts by displayed candidate: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
