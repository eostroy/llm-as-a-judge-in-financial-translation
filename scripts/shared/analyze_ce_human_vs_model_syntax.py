#!/usr/bin/env python
"""Compare CE model-selected translations with ECPCFE human translations syntactically."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import stanza


ROOT = Path(__file__).resolve().parents[2]
CE_SCRIPTS = ROOT / "ce" / "scripts"
if str(CE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(CE_SCRIPTS))

from analyze_ce_deep_features import CANDIDATES, MODEL_FILES, read_json, top_candidate, write_csv, write_json  # noqa: E402
from analyze_ce_syntax_info_features import FEATURE_KEYS, stanza_features  # noqa: E402


RAW_DATASET = ROOT / "ce/datasets/ecpcfe_200ce.json"
DATASET = ROOT / "ce/datasets/ecpcfe_200ce.with_candidates.shuffled.json"
CANDIDATE_SYNTAX = (
    ROOT / "ce/results/parser_derived_syntactic_metrics/analysis/pilot/json/ecpcfe_200ce.syntax_info_features.by_candidate.json"
)
OUT_DIR = ROOT / "ce/results/parser_derived_syntactic_metrics/analysis/human_reference"
OUT_JSON_DIR = OUT_DIR / "json"
OUT_CSV_DIR = OUT_DIR / "csv"
CACHE_PATH = OUT_JSON_DIR / "ecpcfe_200ce.human_translation.syntax_info_features.cache.json"


LOWER_IS_LESS_BURDEN = {
    "dependency_depth",
    "mean_dependency_distance",
    "max_dependency_distance",
    "normalized_dependency_distance",
    "sentence_compression_ratio",
    "passive_count",
}


def load_cache(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected object")
    return data


def save_cache(path: Path, cache: dict[str, dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def short_id(raw_id: str, fallback_index: int) -> str:
    if "_" in raw_id:
        suffix = raw_id.rsplit("_", 1)[-1]
        if suffix.isdigit():
            return str(int(suffix)).zfill(3)
    return str(fallback_index).zfill(3)


def load_human_rows() -> dict[str, dict[str, str]]:
    raw_rows = read_json(RAW_DATASET)
    dataset_rows = read_json(DATASET)
    dataset_by_id = {str(row["id"]): row for row in dataset_rows}
    human_by_id = {}
    for index, raw in enumerate(raw_rows, start=1):
        sample_id = short_id(str(raw.get("id", "")), index)
        if sample_id not in dataset_by_id:
            raise ValueError(f"raw row {raw.get('id')} maps to missing dataset id {sample_id}")
        dataset_source = str(dataset_by_id[sample_id]["source_text"]).strip()
        raw_source = str(raw["source_text"]).strip()
        if dataset_source != raw_source:
            raise ValueError(f"source mismatch for {sample_id}")
        human_by_id[sample_id] = {
            "id": sample_id,
            "raw_id": str(raw.get("id", "")),
            "source_text": raw["source_text"],
            "human_translation": raw["human_translation"],
        }
    return human_by_id


def compute_human_features(human_by_id: dict[str, dict[str, str]]) -> dict[str, dict[str, float]]:
    cache = load_cache(CACHE_PATH)
    missing = [sample_id for sample_id in sorted(human_by_id) if sample_id not in cache]
    if missing:
        nlp = stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse", verbose=False, use_gpu=False)
        for index, sample_id in enumerate(missing, start=1):
            row = human_by_id[sample_id]
            cache[sample_id] = stanza_features(nlp, row["source_text"], row["human_translation"])
            if index % 25 == 0 or index == len(missing):
                save_cache(CACHE_PATH, cache)
                print(f"parsed human translations {index}/{len(missing)}")
    return cache


def load_candidate_features() -> dict[str, dict[str, dict[str, float]]]:
    rows = read_json(CANDIDATE_SYNTAX)
    return {str(row["id"]): row["syntax_information_features"] for row in rows}


def load_rankings() -> dict[str, dict[str, dict[str, int]]]:
    rankings: dict[str, dict[str, dict[str, int]]] = {}
    for model, path in MODEL_FILES.items():
        rows = read_json(ROOT / path if not path.is_absolute() else path)
        rankings[model] = {str(row["id"]): {key: int(value) for key, value in row["rank"].items()} for row in rows}
    return rankings


def paired_rows(
    rankings: dict[str, dict[str, dict[str, int]]],
    human_features: dict[str, dict[str, float]],
    candidate_features: dict[str, dict[str, dict[str, float]]],
) -> list[dict[str, Any]]:
    rows = []
    for model, model_rows in sorted(rankings.items()):
        for sample_id, rank in sorted(model_rows.items()):
            top = top_candidate(rank)
            row = {"model": model, "id": sample_id, "top_candidate": top}
            for key in FEATURE_KEYS:
                model_value = float(candidate_features[sample_id][top][key])
                human_value = float(human_features[sample_id][key])
                row[f"human_{key}"] = round(human_value, 6)
                row[f"model_top1_{key}"] = round(model_value, 6)
                row[f"diff_model_minus_human_{key}"] = round(model_value - human_value, 6)
            rows.append(row)
    return rows


def summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    models = sorted({row["model"] for row in rows})
    summaries = []
    for model in models:
        model_rows = [row for row in rows if row["model"] == model]
        for key in FEATURE_KEYS:
            human_values = [float(row[f"human_{key}"]) for row in model_rows]
            model_values = [float(row[f"model_top1_{key}"]) for row in model_rows]
            diffs = [float(row[f"diff_model_minus_human_{key}"]) for row in model_rows]
            model_lower = sum(1 for diff in diffs if diff < 0)
            model_higher = sum(1 for diff in diffs if diff > 0)
            summaries.append(
                {
                    "model": model,
                    "feature": key,
                    "n": len(model_rows),
                    "avg_human": round(sum(human_values) / len(human_values), 6),
                    "avg_model_top1": round(sum(model_values) / len(model_values), 6),
                    "avg_diff_model_minus_human": round(sum(diffs) / len(diffs), 6),
                    "share_model_lower_than_human": round(model_lower / len(model_rows), 6),
                    "share_model_higher_than_human": round(model_higher / len(model_rows), 6),
                    "lower_means_less_burden": key in LOWER_IS_LESS_BURDEN,
                }
            )
    return summaries


def main() -> int:
    human_by_id = load_human_rows()
    human_features = compute_human_features(human_by_id)
    candidate_features = load_candidate_features()
    rankings = load_rankings()

    enriched_human = [
        {
            **human_by_id[sample_id],
            "human_syntax_information_features": human_features[sample_id],
        }
        for sample_id in sorted(human_by_id)
    ]
    pairs = paired_rows(rankings, human_features, candidate_features)
    summaries = summary_rows(pairs)

    write_json(OUT_JSON_DIR / "ecpcfe_200ce.human_translation.syntax_info_features.by_sample.json", enriched_human)
    write_json(OUT_JSON_DIR / "ecpcfe_200ce.model_top1_vs_human.syntax_info_features.by_sample.json", pairs)
    write_csv(
        OUT_CSV_DIR / "ecpcfe_200ce.model_top1_vs_human.syntax_info_features.summary.csv",
        summaries,
        [
            "model",
            "feature",
            "n",
            "avg_human",
            "avg_model_top1",
            "avg_diff_model_minus_human",
            "share_model_lower_than_human",
            "share_model_higher_than_human",
            "lower_means_less_burden",
        ],
    )
    print(f"wrote CE human-vs-model syntax comparison for {len(human_by_id)} samples and {len(rankings)} models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
