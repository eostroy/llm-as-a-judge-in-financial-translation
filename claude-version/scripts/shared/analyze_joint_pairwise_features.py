#!/usr/bin/env python
"""Fit joint pairwise preference models across model, NLI, syntax, and control features."""

from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any


CANDIDATES = ("A", "B", "C")
PAIRS = tuple(combinations(CANDIDATES, 2))

FEATURE_SPECS = [
    ("labse", "crosslingual_embedding_similarity"),
    ("nli", "bidirectional_entailment"),
    ("nli", "nli_boundary_risk"),
    ("nli", "nli_omission_risk"),
    ("nli", "nli_contradiction_risk"),
    ("syntax", "dependency_depth"),
    ("syntax", "mean_dependency_distance"),
    ("syntax", "normalized_dependency_distance"),
    ("syntax", "sentence_compression_ratio"),
    ("rule", "extra_number_ratio"),
    ("rule", "extra_entity_count"),
    ("rule", "register_score"),
    ("rule", "translationese_score"),
    ("target_lm", "target_lm_naturalness_score"),
]
FEATURE_KEYS = [name for _, name in FEATURE_SPECS]

CONFIGS = {
    "ec": {
        "out_dir": Path("ec/results/model_based_metrics/analysis/pilot"),
        "prefix": "ffn_200ec",
        "model_files": {
            "openai__gpt-5.2": Path(
                "ec/results/model_based_metrics/rankings/json/ffn_200ec.with_candidates.shuffled.ranked.openai__gpt-5.2.json"
            ),
            "google__gemini-3-flash-preview": Path(
                "ec/results/model_based_metrics/rankings/json/ffn_200ec.with_candidates.shuffled.ranked.google__gemini-3-flash-preview.json"
            ),
            "anthropic__claude-sonnet-4.6": Path(
                "ec/results/model_based_metrics/rankings/json/ffn_200ec.with_candidates.shuffled.ranked.anthropic__claude-sonnet-4.6.json"
            ),
            "moonshotai__kimi-k2.5": Path(
                "ec/results/model_based_metrics/rankings/json/ffn_200ec.with_candidates.shuffled.ranked.moonshotai__kimi-k2.5.json"
            ),
            "deepseek__deepseek-v4-flash": Path(
                "ec/results/model_based_metrics/rankings/json/ffn_200ec.with_candidates.shuffled.ranked.deepseek__deepseek-v4-flash.json"
            ),
        },
        "feature_files": {
            "labse": (
                Path("ec/results/model_based_metrics/analysis/pilot/json/ffn_200ec.local_embedding_features.by_candidate.json"),
                "local_embedding_features",
            ),
            "nli": (
                Path("ec/results/model_based_metrics/analysis/pilot/json/ffn_200ec.crosslingual_nli_features.by_candidate.json"),
                "crosslingual_nli_features",
            ),
            "syntax": (
                Path("ec/results/parser_derived_syntactic_metrics/analysis/pilot/json/ffn_200ec.syntax_info_features.by_candidate.json"),
                "syntax_information_features",
            ),
            "rule": (
                Path("ec/results/rule_based_proxy_features/analysis/pilot/json/ffn_200ec.deep_features.by_candidate.json"),
                "deep_candidate_features",
            ),
            "target_lm": (
                Path("ec/results/model_based_metrics/analysis/pilot/json/ffn_200ec.target_lm_features.by_candidate.json"),
                "target_lm_features",
            ),
        },
    },
    "ce": {
        "out_dir": Path("ce/results/model_based_metrics/analysis/pilot"),
        "prefix": "ecpcfe_200ce",
        "model_files": {
            "openai__gpt-5.2": Path(
                "ce/results/model_based_metrics/rankings/json/ecpcfe_200ce.with_candidates.shuffled.ranked.openai__gpt-5.2.json"
            ),
            "google__gemini-3-flash-preview": Path(
                "ce/results/model_based_metrics/rankings/json/ecpcfe_200ce.with_candidates.shuffled.ranked.google__gemini-3-flash-preview.json"
            ),
            "anthropic__claude-sonnet-4.6": Path(
                "ce/results/model_based_metrics/rankings/json/ecpcfe_200ce.with_candidates.shuffled.ranked.anthropic__claude-sonnet-4.6.json"
            ),
            "moonshotai__kimi-k2.5": Path(
                "ce/results/model_based_metrics/rankings/json/ecpcfe_200ce.with_candidates.shuffled.ranked.moonshotai__kimi-k2.5.json"
            ),
            "deepseek__deepseek-v4-flash": Path(
                "ce/results/model_based_metrics/rankings/json/ecpcfe_200ce.with_candidates.shuffled.ranked.deepseek__deepseek-v4-flash.json"
            ),
        },
        "feature_files": {
            "labse": (
                Path("ce/results/model_based_metrics/analysis/pilot/json/ecpcfe_200ce.local_embedding_features.by_candidate.json"),
                "local_embedding_features",
            ),
            "nli": (
                Path("ce/results/model_based_metrics/analysis/pilot/json/ecpcfe_200ce.crosslingual_nli_features.by_candidate.json"),
                "crosslingual_nli_features",
            ),
            "syntax": (
                Path("ce/results/parser_derived_syntactic_metrics/analysis/pilot/json/ecpcfe_200ce.syntax_info_features.by_candidate.json"),
                "syntax_information_features",
            ),
            "rule": (
                Path("ce/results/rule_based_proxy_features/analysis/pilot/json/ecpcfe_200ce.deep_features.by_candidate.json"),
                "deep_candidate_features",
            ),
            "target_lm": (
                Path("ce/results/model_based_metrics/analysis/pilot/json/ecpcfe_200ce.target_lm_features.by_candidate.json"),
                "target_lm_features",
            ),
        },
    },
}


def read_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected JSON array")
    return data


def write_json(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_rankings(model_files: dict[str, Path]) -> dict[str, dict[str, dict[str, int]]]:
    rankings = {}
    for model, path in model_files.items():
        rows = read_json(path)
        rankings[model] = {str(row["id"]): {letter: int(value) for letter, value in row["rank"].items()} for row in rows}
    return rankings


def load_feature_group(path: Path, group_key: str) -> dict[str, dict[str, dict[str, float]]]:
    rows = read_json(path)
    return {str(row["id"]): row[group_key] for row in rows}


def merge_features(cfg: dict[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    groups = {name: load_feature_group(path, key) for name, (path, key) in cfg["feature_files"].items()}
    sample_ids = set.intersection(*(set(group) for group in groups.values()))
    merged: dict[str, dict[str, dict[str, float]]] = {}
    for sample_id in sample_ids:
        merged[sample_id] = {}
        for letter in CANDIDATES:
            merged[sample_id][letter] = {}
            for group_name, feature_name in FEATURE_SPECS:
                merged[sample_id][letter][feature_name] = float(groups[group_name][sample_id][letter][feature_name])
    return merged


def winner_from_rank(rank: dict[str, int], a: str, b: str) -> str:
    return a if int(rank[a]) < int(rank[b]) else b


def standardize_matrix(matrix: list[list[float]]) -> list[list[float]]:
    cols = len(matrix[0])
    means = [sum(row[col] for row in matrix) / len(matrix) for col in range(cols)]
    stds = []
    for col in range(cols):
        variance = sum((row[col] - means[col]) ** 2 for row in matrix) / len(matrix)
        stds.append(math.sqrt(variance) or 1.0)
    return [[(value - means[col]) / stds[col] for col, value in enumerate(row)] for row in matrix]


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def fit_logistic(
    matrix: list[list[float]],
    labels: list[int],
    lr: float = 0.07,
    epochs: int = 3500,
    l2: float = 0.01,
) -> tuple[float, list[float], float]:
    x = standardize_matrix(matrix)
    weights = [0.0] * len(x[0])
    bias = 0.0
    n = len(labels)
    for _ in range(epochs):
        grad_b = 0.0
        grad_w = [0.0] * len(weights)
        for row, target in zip(x, labels):
            pred = sigmoid(bias + sum(weight * value for weight, value in zip(weights, row)))
            err = pred - target
            grad_b += err
            for idx, value in enumerate(row):
                grad_w[idx] += err * value
        bias -= lr * grad_b / n
        for idx in range(len(weights)):
            weights[idx] -= lr * ((grad_w[idx] / n) + l2 * weights[idx])
    correct = 0
    for row, target in zip(x, labels):
        pred = 1 if sigmoid(bias + sum(weight * value for weight, value in zip(weights, row))) >= 0.5 else 0
        correct += int(pred == target)
    return bias, weights, correct / n


def pairwise_rows(
    rankings: dict[str, dict[str, dict[str, int]]],
    features_by_id: dict[str, dict[str, dict[str, float]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    coefficient_rows = []
    observation_rows = []
    for model, model_rows in sorted(rankings.items()):
        x = []
        y = []
        for sample_id, rank in model_rows.items():
            if sample_id not in features_by_id:
                raise ValueError(f"missing features for sample {sample_id}")
            features = features_by_id[sample_id]
            for a, b in PAIRS:
                diff = [features[a][key] - features[b][key] for key in FEATURE_KEYS]
                target = 1 if winner_from_rank(rank, a, b) == a else 0
                x.append(diff)
                y.append(target)
                observation_rows.append(
                    {
                        "model": model,
                        "sample_id": sample_id,
                        "candidate_a": a,
                        "candidate_b": b,
                        "winner": a if target else b,
                        **{f"diff_{key}": round(value, 6) for key, value in zip(FEATURE_KEYS, diff)},
                    }
                )
        bias, weights, accuracy = fit_logistic(x, y)
        for key, weight in zip(FEATURE_KEYS, weights):
            coefficient_rows.append(
                {
                    "model": model,
                    "feature": key,
                    "standardized_coefficient": round(weight, 6),
                    "training_accuracy": round(accuracy, 6),
                    "intercept": round(bias, 6),
                    "n_pairwise_observations": len(y),
                }
            )
    return coefficient_rows, observation_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("direction", choices=sorted(CONFIGS))
    args = parser.parse_args()

    cfg = CONFIGS[args.direction]
    out_dir = cfg["out_dir"]
    out_json_dir = out_dir / "json"
    out_csv_dir = out_dir / "csv"
    prefix = cfg["prefix"]
    rankings = load_rankings(cfg["model_files"])
    features_by_id = merge_features(cfg)
    coefficient_rows, observation_rows = pairwise_rows(rankings, features_by_id)

    write_csv(
        out_csv_dir / f"{prefix}.joint_features.pairwise_logistic_preferences.csv",
        coefficient_rows,
        ["model", "feature", "standardized_coefficient", "training_accuracy", "intercept", "n_pairwise_observations"],
    )
    write_json(out_json_dir / f"{prefix}.joint_features.pairwise_observations.json", observation_rows)
    write_json(
        out_json_dir / f"{prefix}.joint_features.method_notes.json",
        [
            {
                "features": FEATURE_KEYS,
                "model": "standardized pairwise logistic regression per judge model",
                "target_lm_naturalness_score": "negative target-language LM log perplexity; positive coefficient means preference for lower perplexity after controlling other features",
                "l2": 0.01,
            }
        ],
    )
    print(f"wrote joint pairwise feature model for {args.direction} to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
