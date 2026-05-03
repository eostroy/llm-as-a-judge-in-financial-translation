#!/usr/bin/env python
"""Build three-candidate datasets from generated variant files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULTS = {
    "ec": {
        "input": Path("ec/datasets/ffn_200ec.with_variants.json"),
        "output": Path("ec/datasets/ffn_200ec.with_candidates.json"),
    },
    "ce": {
        "input": Path("ce/datasets/ecpcfe_200ce.with_variants.json"),
        "output": Path("ce/datasets/ecpcfe_200ce.with_candidates.json"),
    },
}


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


def display_id(row_id: str, index: int) -> str:
    match = row_id.rsplit("_", 1)
    if len(match) == 2 and match[1].isdigit():
        return f"{int(match[1]):03d}"
    return f"{index:03d}"


def build_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    built = []
    for index, row in enumerate(rows, start=1):
        sample_id = display_id(str(row.get("id", "")), index)
        built.append(
            {
                "id": sample_id,
                "original_id": row.get("id"),
                "direction": row.get("direction"),
                "source_lang": row.get("source_lang"),
                "target_lang": row.get("target_lang"),
                "source_text": row.get("source_text"),
                "candidate_A": row.get("human_translation"),
                "candidate_B": row.get("variant_1_better"),
                "candidate_C": row.get("variant_2_slightly_weaker"),
                "candidate_source_map": {
                    "A": "human_translation",
                    "B": "claude_variant_1_better",
                    "C": "claude_variant_2_slightly_weaker",
                },
                "domain_subtype": row.get("domain_subtype"),
                "difficulty": row.get("difficulty"),
                "corpus_source": row.get("corpus_source"),
                "generation_model": row.get("generation_model"),
                "generation_provider": row.get("generation_provider"),
            }
        )
    return built


def sort_key(row: dict[str, Any]) -> tuple[int, str]:
    row_id = str(row.get("id", ""))
    suffix = row_id.rsplit("_", 1)[-1]
    if suffix.isdigit():
        return int(suffix), row_id
    return 10**9, row_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("direction", choices=["ec", "ce", "all"])
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def build_direction(direction: str, args: argparse.Namespace) -> None:
    input_path = args.input or DEFAULTS[direction]["input"]
    output_path = args.output or DEFAULTS[direction]["output"]
    rows = sorted(read_json(input_path), key=sort_key)
    built = build_rows(rows)
    write_json(output_path, built)
    print(f"[{direction.upper()}] wrote {len(built)} rows to {output_path}")


def main() -> int:
    args = parse_args()
    if args.direction == "all" and any([args.input, args.output]):
        raise SystemExit("--input/--output can only be used with a single direction")
    directions = ["ec", "ce"] if args.direction == "all" else [args.direction]
    for direction in directions:
        build_direction(direction, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
