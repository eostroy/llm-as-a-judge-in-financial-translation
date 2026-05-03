#!/usr/bin/env python
"""Add cluster-robust standard errors to joint pairwise logistic models.

The original joint analysis treats every pairwise comparison as one row. For
each source sample there are three rows, so this script clusters the sandwich
covariance estimator by sample_id while keeping the original coefficient
estimation procedure unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


FEATURE_KEYS = [
    "crosslingual_embedding_similarity",
    "bidirectional_entailment",
    "nli_boundary_risk",
    "nli_omission_risk",
    "nli_contradiction_risk",
    "dependency_depth",
    "mean_dependency_distance",
    "normalized_dependency_distance",
    "sentence_compression_ratio",
    "extra_number_ratio",
    "extra_entity_count",
    "register_score",
    "translationese_score",
    "target_lm_naturalness_score",
]

VERSIONS = ("kimi-version", "claude-version")
DIRECTIONS = {
    "ec": "ffn_200ec",
    "ce": "ecpcfe_200ce",
}


def sigmoid_array(values: np.ndarray) -> np.ndarray:
    return np.where(values >= 0, 1 / (1 + np.exp(-values)), np.exp(values) / (1 + np.exp(values)))


def standardize_matrix(matrix: np.ndarray) -> np.ndarray:
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds[stds == 0] = 1.0
    return (matrix - means) / stds


def fit_logistic(
    matrix: np.ndarray,
    labels: np.ndarray,
    lr: float = 0.07,
    epochs: int = 3500,
    l2: float = 0.01,
) -> tuple[float, np.ndarray, float, np.ndarray]:
    x = standardize_matrix(matrix)
    weights = np.zeros(x.shape[1], dtype=float)
    bias = 0.0
    n = labels.shape[0]
    for _ in range(epochs):
        preds = sigmoid_array(bias + x @ weights)
        errors = preds - labels
        bias -= lr * float(errors.mean())
        weights -= lr * ((x.T @ errors / n) + l2 * weights)
    final_preds = sigmoid_array(bias + x @ weights)
    accuracy = float(((final_preds >= 0.5).astype(int) == labels).mean())
    return bias, weights, accuracy, x


def cluster_robust_covariance(
    design: np.ndarray,
    labels: np.ndarray,
    beta: np.ndarray,
    clusters: list[str],
    l2: float = 0.01,
) -> np.ndarray:
    preds = sigmoid_array(design @ beta)
    weights = preds * (1 - preds)
    hessian = design.T @ (design * weights[:, None])
    penalty = np.eye(design.shape[1]) * (len(labels) * l2)
    penalty[0, 0] = 0.0
    bread = np.linalg.pinv(hessian + penalty)

    score_by_cluster: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(design.shape[1], dtype=float))
    residuals = labels - preds
    for row, residual, cluster in zip(design, residuals, clusters):
        score_by_cluster[cluster] += row * residual

    meat = np.zeros((design.shape[1], design.shape[1]), dtype=float)
    for score in score_by_cluster.values():
        meat += np.outer(score, score)

    n = design.shape[0]
    k = design.shape[1]
    g = len(score_by_cluster)
    correction = (g / (g - 1)) * ((n - 1) / (n - k)) if g > 1 and n > k else 1.0
    return correction * bread @ meat @ bread


def p_value_from_z(z_value: float) -> float:
    return math.erfc(abs(z_value) / math.sqrt(2))


def read_observations(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array")
    return data


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def analyze_file(version_dir: Path, direction: str) -> tuple[Path, list[dict[str, Any]]]:
    prefix = DIRECTIONS[direction]
    input_path = (
        version_dir
        / direction
        / "results/model_based_metrics/analysis/pilot/json"
        / f"{prefix}.joint_features.pairwise_observations.json"
    )
    rows = read_observations(input_path)
    rows_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_model[str(row["model"])].append(row)

    output_rows: list[dict[str, Any]] = []
    for model, model_rows in sorted(rows_by_model.items()):
        matrix = np.array(
            [[float(row[f"diff_{feature}"]) for feature in FEATURE_KEYS] for row in model_rows],
            dtype=float,
        )
        labels = np.array(
            [1 if str(row["winner"]) == str(row["candidate_a"]) else 0 for row in model_rows],
            dtype=float,
        )
        clusters = [str(row["sample_id"]) for row in model_rows]
        intercept, weights, accuracy, x_std = fit_logistic(matrix, labels)
        design = np.column_stack([np.ones(labels.shape[0]), x_std])
        beta = np.concatenate([[intercept], weights])
        covariance = cluster_robust_covariance(design, labels, beta, clusters)
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0))

        for idx, feature in enumerate(["intercept", *FEATURE_KEYS]):
            coefficient = float(beta[idx])
            se = float(standard_errors[idx])
            z_value = coefficient / se if se else float("nan")
            p_value = p_value_from_z(z_value) if math.isfinite(z_value) else float("nan")
            output_rows.append(
                {
                    "model": model,
                    "feature": feature,
                    "standardized_coefficient": round(coefficient, 6),
                    "cluster_robust_se": round(se, 6),
                    "z_cluster": round(z_value, 6) if math.isfinite(z_value) else "",
                    "p_cluster": round(p_value, 6) if math.isfinite(p_value) else "",
                    "ci95_low": round(coefficient - 1.96 * se, 6),
                    "ci95_high": round(coefficient + 1.96 * se, 6),
                    "cluster_significant_05": bool(math.isfinite(p_value) and p_value < 0.05),
                    "training_accuracy": round(accuracy, 6),
                    "n_pairwise_observations": len(model_rows),
                    "n_clusters": len(set(clusters)),
                }
            )

    output_path = (
        version_dir
        / direction
        / "results/model_based_metrics/analysis/pilot/csv"
        / f"{prefix}.joint_features.cluster_robust_pairwise_logistic.csv"
    )
    write_csv(
        output_path,
        output_rows,
        [
            "model",
            "feature",
            "standardized_coefficient",
            "cluster_robust_se",
            "z_cluster",
            "p_cluster",
            "ci95_low",
            "ci95_high",
            "cluster_significant_05",
            "training_accuracy",
            "n_pairwise_observations",
            "n_clusters",
        ],
    )
    write_json(
        output_path.with_suffix(".method_notes.json"),
        {
            "input": str(input_path),
            "model": "same standardized L2 pairwise logistic model as the original joint analysis",
            "cluster": "sample_id",
            "standard_error": "cluster-robust sandwich covariance with finite-sample correction",
            "interpretation": "coefficients are unchanged in substance; cluster-robust SE corrects inference for three non-independent pairwise rows per sample",
            "features": FEATURE_KEYS,
            "l2": 0.01,
        },
    )
    return output_path, output_rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feature_rows = [row for row in rows if row["feature"] != "intercept"]
    significant = [row for row in feature_rows if row["cluster_significant_05"]]
    return {
        "feature_rows": len(feature_rows),
        "significant_rows_p_lt_05": len(significant),
        "significant_features_by_model": {
            row["model"]: sorted(
                feature_row["feature"]
                for feature_row in significant
                if feature_row["model"] == row["model"]
            )
            for row in feature_rows
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=VERSIONS)
    parser.add_argument("--direction", choices=sorted(DIRECTIONS))
    parser.add_argument("--all", action="store_true", help="Run all versions and directions.")
    args = parser.parse_args()

    if args.all:
        jobs = [(version, direction) for version in VERSIONS for direction in sorted(DIRECTIONS)]
    else:
        if not args.version or not args.direction:
            parser.error("use --all or provide both --version and --direction")
        jobs = [(args.version, args.direction)]

    for version, direction in jobs:
        output_path, output_rows = analyze_file(Path(version), direction)
        info = summarize(output_rows)
        print(
            f"wrote {output_path} "
            f"({info['significant_rows_p_lt_05']}/{info['feature_rows']} feature rows p<.05)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
