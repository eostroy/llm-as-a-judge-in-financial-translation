#!/usr/bin/env python
"""Add cluster-robust inference to pairwise feature-family logistic models.

Most ``*.pairwise_logistic_preferences.csv`` files are fitted from three
pairwise comparisons per source sample. Those rows are not independent, so this
script refits the same standardized L2 logistic models and clusters the
sandwich standard errors by sample id.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any


CANDIDATES = ("A", "B", "C")
PAIRS = tuple(combinations(CANDIDATES, 2))
VERSIONS = ("kimi-version", "claude-version")
DIRECTIONS = {"ec": "ffn_200ec", "ce": "ecpcfe_200ce"}

FAMILY_CONFIGS = {
    "local_embedding_features": {
        "result_root": "model_based_metrics",
        "json_key": "local_embedding_features",
        "json_family": "local_embedding_features",
    },
    "gemini_embedding_features": {
        "result_root": "model_based_metrics",
        "json_key": "gemini_embedding_features",
        "json_family": "gemini_embedding_features",
    },
    "target_lm_features": {
        "result_root": "model_based_metrics",
        "json_key": "target_lm_features",
        "json_family": "target_lm_features",
    },
    "crosslingual_nli_features": {
        "result_root": "model_based_metrics",
        "json_key": "crosslingual_nli_features",
        "json_family": "crosslingual_nli_features",
    },
    "syntax_info_features": {
        "result_root": "parser_derived_syntactic_metrics",
        "json_key": "syntax_information_features",
        "json_family": "syntax_info_features",
    },
    "deep_features": {
        "result_root": "rule_based_proxy_features",
        "json_key": "deep_candidate_features",
        "json_family": "deep_features",
    },
}


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def standardize_matrix(matrix: list[list[float]]) -> list[list[float]]:
    cols = len(matrix[0])
    means = [sum(row[col] for row in matrix) / len(matrix) for col in range(cols)]
    stds: list[float] = []
    for col in range(cols):
        variance = sum((row[col] - means[col]) ** 2 for row in matrix) / len(matrix)
        stds.append(math.sqrt(variance) or 1.0)
    return [[(value - means[col]) / stds[col] for col, value in enumerate(row)] for row in matrix]


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def invert_matrix(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    augmented = [
        [float(value) for value in row] + [1.0 if i == j else 0.0 for j in range(n)]
        for i, row in enumerate(matrix)
    ]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            augmented[col][col] += 1e-8
            pivot = col
        if pivot != col:
            augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        scale = augmented[col][col]
        if abs(scale) < 1e-15:
            raise ValueError("singular matrix in covariance calculation")
        augmented[col] = [value / scale for value in augmented[col]]
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor:
                augmented[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(augmented[row], augmented[col])
                ]
    return [row[n:] for row in augmented]


def matmul(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    right_t = list(map(list, zip(*right)))
    return [[dot(row, col) for col in right_t] for row in left]


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    return list(map(list, zip(*matrix)))


def fit_logistic(
    matrix: list[list[float]],
    labels: list[float],
    lr: float = 0.07,
    epochs: int = 3500,
    l2: float = 0.01,
) -> tuple[float, list[float], float, list[list[float]]]:
    x = standardize_matrix(matrix)
    weights = [0.0] * len(x[0])
    bias = 0.0
    n = len(labels)
    for _ in range(epochs):
        grad_b = 0.0
        grad_w = [0.0] * len(weights)
        for row, target in zip(x, labels):
            pred = sigmoid(bias + dot(row, weights))
            err = pred - target
            grad_b += err
            for idx, value in enumerate(row):
                grad_w[idx] += err * value
        bias -= lr * grad_b / n
        for idx in range(len(weights)):
            weights[idx] -= lr * ((grad_w[idx] / n) + l2 * weights[idx])
    correct = 0
    for row, target in zip(x, labels):
        pred = 1.0 if sigmoid(bias + dot(row, weights)) >= 0.5 else 0.0
        correct += int(pred == target)
    accuracy = correct / n
    return bias, weights, accuracy, x


def cluster_robust_covariance(
    design: list[list[float]],
    labels: list[float],
    beta: list[float],
    clusters: list[str],
    l2: float = 0.01,
) -> list[list[float]]:
    preds = [sigmoid(dot(row, beta)) for row in design]
    weights = [pred * (1 - pred) for pred in preds]
    k = len(design[0])
    hessian = [[0.0 for _ in range(k)] for _ in range(k)]
    for row, weight in zip(design, weights):
        for i in range(k):
            for j in range(k):
                hessian[i][j] += row[i] * row[j] * weight
    for idx in range(1, k):
        hessian[idx][idx] += len(labels) * l2
    bread = invert_matrix(hessian)

    score_by_cluster: dict[str, list[float]] = defaultdict(lambda: [0.0] * k)
    residuals = [target - pred for target, pred in zip(labels, preds)]
    for row, residual, cluster in zip(design, residuals, clusters):
        score = score_by_cluster[cluster]
        for idx, value in enumerate(row):
            score[idx] += value * residual

    meat = [[0.0 for _ in range(k)] for _ in range(k)]
    for score in score_by_cluster.values():
        for i in range(k):
            for j in range(k):
                meat[i][j] += score[i] * score[j]

    n = len(design)
    g = len(score_by_cluster)
    correction = (g / (g - 1)) * ((n - 1) / (n - k)) if g > 1 and n > k else 1.0
    covariance = matmul(matmul(bread, meat), bread)
    return [[correction * value for value in row] for row in covariance]


def p_value_from_z(z_value: float) -> float:
    return math.erfc(abs(z_value) / math.sqrt(2))


def read_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array")
    return data


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def feature_csv_paths(version_dir: Path, direction: str, prefix: str) -> list[Path]:
    paths: list[Path] = []
    for family, cfg in FAMILY_CONFIGS.items():
        csv_dir = version_dir / direction / "results" / cfg["result_root"] / "analysis/pilot/csv"
        paths.extend(sorted(csv_dir.glob(f"{prefix}.{family}.*pairwise_logistic_preferences.csv")))
    return paths


def family_from_csv(path: Path, prefix: str) -> str:
    name = path.name
    body = name.removeprefix(f"{prefix}.")
    for family in FAMILY_CONFIGS:
        if body.startswith(family + "."):
            return family
    raise ValueError(f"Cannot infer feature family from {path}")


def load_rankings(version_dir: Path, direction: str, prefix: str, models: set[str]) -> dict[str, dict[str, dict[str, int]]]:
    ranking_dir = version_dir / direction / "results/model_based_metrics/rankings/json"
    rankings: dict[str, dict[str, dict[str, int]]] = {}
    for model in sorted(models):
        path = ranking_dir / f"{prefix}.with_candidates.shuffled.ranked.{model}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing ranking file for {model}: {path}")
        rows = read_json(path)
        rankings[model] = {str(row["id"]): {letter: int(value) for letter, value in row["rank"].items()} for row in rows}
    return rankings


def load_feature_values(path: Path, json_key: str) -> dict[str, dict[str, dict[str, float]]]:
    rows = read_json(path)
    values: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        sample_id = str(row["id"])
        group = row[json_key]
        values[sample_id] = {
            letter: {feature: float(value) for feature, value in group[letter].items()}
            for letter in CANDIDATES
        }
    return values


def winner_from_rank(rank: dict[str, int], a: str, b: str) -> str:
    return a if int(rank[a]) < int(rank[b]) else b


def build_design(
    ranking_by_id: dict[str, dict[str, int]],
    features_by_id: dict[str, dict[str, dict[str, float]]],
    feature_keys: list[str],
) -> tuple[list[list[float]], list[float], list[str]]:
    matrix: list[list[float]] = []
    labels: list[float] = []
    clusters: list[str] = []
    for sample_id, rank in sorted(ranking_by_id.items()):
        if sample_id not in features_by_id:
            raise ValueError(f"Missing feature values for sample {sample_id}")
        features = features_by_id[sample_id]
        for a, b in PAIRS:
            matrix.append([features[a][key] - features[b][key] for key in feature_keys])
            labels.append(1.0 if winner_from_rank(rank, a, b) == a else 0.0)
            clusters.append(sample_id)
    return matrix, labels, clusters


def robust_rows_from_coefficients(
    model: str,
    ranking_by_id: dict[str, dict[str, int]],
    features_by_id: dict[str, dict[str, dict[str, float]]],
    feature_keys: list[str],
    model_type: str,
    intercept: float,
    coefficients: dict[str, float],
    training_accuracy: float,
) -> list[dict[str, Any]]:
    matrix, labels, clusters = build_design(ranking_by_id, features_by_id, feature_keys)
    x_std = standardize_matrix(matrix)
    design = [[1.0, *row] for row in x_std]
    beta = [intercept, *[coefficients[feature] for feature in feature_keys]]
    covariance = cluster_robust_covariance(design, labels, beta, clusters)
    standard_errors = [math.sqrt(max(covariance[idx][idx], 0.0)) for idx in range(len(covariance))]

    rows: list[dict[str, Any]] = []
    for idx, feature in enumerate(feature_keys, start=1):
        coefficient = float(beta[idx])
        se = float(standard_errors[idx])
        z_value = coefficient / se if se else float("nan")
        p_value = p_value_from_z(z_value) if math.isfinite(z_value) else float("nan")
        rows.append(
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
                "training_accuracy": round(training_accuracy, 6),
                "intercept": round(intercept, 6),
                "n_pairwise_observations": len(labels),
                "n_clusters": len(set(clusters)),
                "model_type": model_type,
            }
        )
    return rows


def analyze_csv(version_dir: Path, direction: str, prefix: str, csv_path: Path) -> tuple[Path, list[dict[str, Any]]]:
    family = family_from_csv(csv_path, prefix)
    cfg = FAMILY_CONFIGS[family]
    old_rows = [row for row in read_csv(csv_path) if row.get("feature")]
    models = {row["model"] for row in old_rows}
    feature_keys = list(dict.fromkeys(row["feature"] for row in old_rows))
    univariate = ".univariate_pairwise_logistic_preferences.csv" in csv_path.name

    json_path = (
        version_dir
        / direction
        / "results"
        / cfg["result_root"]
        / "analysis/pilot/json"
        / f"{prefix}.{cfg['json_family']}.by_candidate.json"
    )
    if not json_path.exists():
        raise FileNotFoundError(f"Missing feature JSON for {csv_path}: {json_path}")

    rankings = load_rankings(version_dir, direction, prefix, models)
    features_by_id = load_feature_values(json_path, cfg["json_key"])
    output_rows: list[dict[str, Any]] = []
    model_type = "univariate" if univariate else "feature_family_multivariate"
    rows_by_model_feature = {
        (row["model"], row["feature"]): row
        for row in old_rows
    }

    for model in sorted(models):
        if univariate:
            for feature in feature_keys:
                old_row = rows_by_model_feature[(model, feature)]
                output_rows.extend(
                    robust_rows_from_coefficients(
                        model,
                        rankings[model],
                        features_by_id,
                        [feature],
                        model_type,
                        float(old_row["intercept"]),
                        {feature: float(old_row["standardized_coefficient"])},
                        float(old_row["training_accuracy"]),
                    )
                )
        else:
            model_rows = [rows_by_model_feature[(model, feature)] for feature in feature_keys]
            output_rows.extend(
                robust_rows_from_coefficients(
                    model,
                    rankings[model],
                    features_by_id,
                    feature_keys,
                    model_type,
                    float(model_rows[0]["intercept"]),
                    {
                        row["feature"]: float(row["standardized_coefficient"])
                        for row in model_rows
                    },
                    float(model_rows[0]["training_accuracy"]),
                )
            )

    output_path = csv_path.with_name(csv_path.name.replace("pairwise_logistic_preferences", "cluster_robust_pairwise_logistic"))
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
            "intercept",
            "n_pairwise_observations",
            "n_clusters",
            "model_type",
        ],
    )
    write_json(
        output_path.with_suffix(".method_notes.json"),
        {
            "input_coefficients": str(csv_path),
            "input_features": str(json_path),
            "ranking_files": "same judge ranking JSON files used by the original analysis",
            "model": "standardized L2 pairwise logistic regression refit from candidate-level features",
            "model_type": model_type,
            "cluster": "sample_id",
            "standard_error": "cluster-robust sandwich covariance with finite-sample correction",
            "interpretation": "use p_cluster and ci95_low/high when discussing statistical support; coefficients remain descriptive associations, not causal effects",
            "l2": 0.01,
            "features": feature_keys,
        },
    )
    return output_path, output_rows


def analyze_job(version: str, direction: str) -> list[tuple[Path, int, int]]:
    version_dir = Path(version)
    prefix = DIRECTIONS[direction]
    summaries: list[tuple[Path, int, int]] = []
    for csv_path in feature_csv_paths(version_dir, direction, prefix):
        output_path, rows = analyze_csv(version_dir, direction, prefix, csv_path)
        significant = sum(1 for row in rows if row["cluster_significant_05"])
        summaries.append((output_path, significant, len(rows)))
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=VERSIONS)
    parser.add_argument("--direction", choices=sorted(DIRECTIONS))
    parser.add_argument("--all", action="store_true", help="Run every version/direction.")
    args = parser.parse_args()

    if args.all:
        jobs = [(version, direction) for version in VERSIONS for direction in sorted(DIRECTIONS)]
    else:
        if not args.version or not args.direction:
            parser.error("use --all or provide both --version and --direction")
        jobs = [(args.version, args.direction)]

    for version, direction in jobs:
        for output_path, significant, total in analyze_job(version, direction):
            print(f"wrote {output_path} ({significant}/{total} rows p_cluster<.05)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
