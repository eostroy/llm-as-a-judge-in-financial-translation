#!/usr/bin/env python
"""Analyze judge sets after excluding the candidate generator itself.

This script intentionally avoids the ambiguous historical four_models/five_models
labels. Output filenames state the exclusion rule directly:

    <dataset>.judges_excluding_candidate_generator_<model_slug>.<artifact>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any


CANDIDATES = ("A", "B", "C")
PAIRS = tuple(combinations(CANDIDATES, 2))

COMPOSITE_FEATURE_SPECS = [
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
COMPOSITE_FEATURE_KEYS = [name for _, name in COMPOSITE_FEATURE_SPECS]
TRANSLATIONESE_SUBFEATURE_KEYS = [
    "target_function_word_density",
    "target_passive_density",
    "target_pronoun_subject_ratio",
    "target_explicit_connective_density",
    "target_nominalization_suffix_density",
    "target_avg_sentence_length",
]

MODEL_FILES = {
    "ec": {
        "openai__gpt-5.2": Path("ec/results/model_based_metrics/rankings/json/ffn_200ec.with_candidates.shuffled.ranked.openai__gpt-5.2.json"),
        "google__gemini-3-flash-preview": Path("ec/results/model_based_metrics/rankings/json/ffn_200ec.with_candidates.shuffled.ranked.google__gemini-3-flash-preview.json"),
        "anthropic__claude-sonnet-4.6": Path("ec/results/model_based_metrics/rankings/json/ffn_200ec.with_candidates.shuffled.ranked.anthropic__claude-sonnet-4.6.json"),
        "moonshotai__kimi-k2.5": Path("ec/results/model_based_metrics/rankings/json/ffn_200ec.with_candidates.shuffled.ranked.moonshotai__kimi-k2.5.json"),
        "deepseek__deepseek-v4-flash": Path("ec/results/model_based_metrics/rankings/json/ffn_200ec.with_candidates.shuffled.ranked.deepseek__deepseek-v4-flash.json"),
    },
    "ce": {
        "openai__gpt-5.2": Path("ce/results/model_based_metrics/rankings/json/ecpcfe_200ce.with_candidates.shuffled.ranked.openai__gpt-5.2.json"),
        "google__gemini-3-flash-preview": Path("ce/results/model_based_metrics/rankings/json/ecpcfe_200ce.with_candidates.shuffled.ranked.google__gemini-3-flash-preview.json"),
        "anthropic__claude-sonnet-4.6": Path("ce/results/model_based_metrics/rankings/json/ecpcfe_200ce.with_candidates.shuffled.ranked.anthropic__claude-sonnet-4.6.json"),
        "moonshotai__kimi-k2.5": Path("ce/results/model_based_metrics/rankings/json/ecpcfe_200ce.with_candidates.shuffled.ranked.moonshotai__kimi-k2.5.json"),
        "deepseek__deepseek-v4-flash": Path("ce/results/model_based_metrics/rankings/json/ecpcfe_200ce.with_candidates.shuffled.ranked.deepseek__deepseek-v4-flash.json"),
    },
}

CONFIGS = {
    "ec": {
        "prefix": "ffn_200ec",
        "dataset": Path("ec/datasets/ffn_200ec.with_candidates.shuffled.json"),
        "feature_files": {
            "labse": (Path("ec/results/model_based_metrics/analysis/pilot/json/ffn_200ec.local_embedding_features.by_candidate.json"), "local_embedding_features"),
            "nli": (Path("ec/results/model_based_metrics/analysis/pilot/json/ffn_200ec.crosslingual_nli_features.by_candidate.json"), "crosslingual_nli_features"),
            "syntax": (Path("ec/results/parser_derived_syntactic_metrics/analysis/pilot/json/ffn_200ec.syntax_info_features.by_candidate.json"), "syntax_information_features"),
            "rule": (Path("ec/results/rule_based_proxy_features/analysis/pilot/json/ffn_200ec.deep_features.by_candidate.json"), "deep_candidate_features"),
            "target_lm": (Path("ec/results/model_based_metrics/analysis/pilot/json/ffn_200ec.target_lm_features.by_candidate.json"), "target_lm_features"),
        },
    },
    "ce": {
        "prefix": "ecpcfe_200ce",
        "dataset": Path("ce/datasets/ecpcfe_200ce.with_candidates.shuffled.json"),
        "feature_files": {
            "labse": (Path("ce/results/model_based_metrics/analysis/pilot/json/ecpcfe_200ce.local_embedding_features.by_candidate.json"), "local_embedding_features"),
            "nli": (Path("ce/results/model_based_metrics/analysis/pilot/json/ecpcfe_200ce.crosslingual_nli_features.by_candidate.json"), "crosslingual_nli_features"),
            "syntax": (Path("ce/results/parser_derived_syntactic_metrics/analysis/pilot/json/ecpcfe_200ce.syntax_info_features.by_candidate.json"), "syntax_information_features"),
            "rule": (Path("ce/results/rule_based_proxy_features/analysis/pilot/json/ecpcfe_200ce.deep_features.by_candidate.json"), "deep_candidate_features"),
            "target_lm": (Path("ce/results/model_based_metrics/analysis/pilot/json/ecpcfe_200ce.target_lm_features.by_candidate.json"), "target_lm_features"),
        },
    },
}

VERSION_EXCLUSIONS = {
    "kimi-version": {
        "excluded_model": "moonshotai__kimi-k2.5",
        "excluded_model_slug": "kimi_k2_5",
        "candidate_generator": "Kimi K2.5",
    },
    "claude-version": {
        "excluded_model": "anthropic__claude-sonnet-4.6",
        "excluded_model_slug": "claude_sonnet_4_6",
        "candidate_generator": "Claude Sonnet 4.6",
    },
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_rankings(version_dir: Path, direction: str, excluded_model: str) -> dict[str, dict[str, dict[str, int]]]:
    rankings = {}
    for model, relative_path in MODEL_FILES[direction].items():
        if model == excluded_model:
            continue
        rows = read_json(version_dir / relative_path)
        rankings[model] = {str(row["id"]): {letter: int(value) for letter, value in row["rank"].items()} for row in rows}
    return rankings


def load_feature_group(path: Path, group_key: str) -> dict[str, dict[str, dict[str, float]]]:
    rows = read_json(path)
    return {str(row["id"]): row[group_key] for row in rows}


def count_terms(text: str, terms: tuple[str, ...]) -> int:
    return sum(text.count(term) for term in terms)


def zh_chars(text: str) -> list[str]:
    return [char for char in text if "\u4e00" <= char <= "\u9fff"]


def zh_sentences(text: str) -> list[str]:
    pieces = [piece.strip() for piece in re.split(r"[。！？!?；;]", text) if piece.strip()]
    return pieces or [text.strip()]


def en_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)?", text)


def en_sentences(text: str) -> list[str]:
    pieces = [piece.strip() for piece in re.split(r"[.!?;]+", text) if piece.strip()]
    return pieces or [text.strip()]


def translationese_subfeatures(direction: str, candidate: str) -> dict[str, float]:
    if direction == "ec":
        chars = max(len(zh_chars(candidate)), 1)
        sentences = zh_sentences(candidate)
        pronouns = ("我", "我们", "你", "你们", "他", "他们", "她", "她们", "它", "它们", "本公司", "该公司")
        connectives = ("因为", "所以", "虽然", "但是", "如果", "那么", "由于", "因此", "然而", "此外", "同时", "从而", "并且")
        nominal_suffixes = ("性", "化", "度", "率")
        sentence_initial_pronouns = sum(1 for sentence in sentences if sentence.startswith(pronouns))
        return {
            "target_function_word_density": round(candidate.count("的") / chars, 6),
            "target_passive_density": round(count_terms(candidate, ("被",)) / chars * 100, 6),
            "target_pronoun_subject_ratio": round(sentence_initial_pronouns / len(sentences), 6),
            "target_explicit_connective_density": round(count_terms(candidate, connectives) / chars * 100, 6),
            "target_nominalization_suffix_density": round(count_terms(candidate, nominal_suffixes) / chars * 100, 6),
            "target_avg_sentence_length": round(chars / len(sentences), 6),
        }

    lower = candidate.lower()
    words = max(len(en_words(candidate)), 1)
    sentences = en_sentences(candidate)
    pronouns = ("i", "we", "you", "he", "she", "it", "they", "this", "these", "that", "those")
    connectives = ("because", "therefore", "however", "although", "though", "if", "while", "moreover", "furthermore", "thus")
    passive_aux = ("be", "is", "are", "was", "were", "been", "being")
    sentence_initial_pronouns = 0
    for sentence in sentences:
        words_in_sentence = en_words(sentence.lower())
        if words_in_sentence and words_in_sentence[0] in pronouns:
            sentence_initial_pronouns += 1
    nominalizations = re.findall(r"\b[A-Za-z]+(?:tion|ment|ness|ity|ance|ence|ization|isation)\b", lower)
    return {
        "target_function_word_density": round(len(re.findall(r"\bof\b", lower)) / words, 6),
        "target_passive_density": round(sum(1 for word in en_words(lower) if word in passive_aux) / words * 100, 6),
        "target_pronoun_subject_ratio": round(sentence_initial_pronouns / len(sentences), 6),
        "target_explicit_connective_density": round(sum(1 for word in en_words(lower) if word in connectives) / words * 100, 6),
        "target_nominalization_suffix_density": round(len(nominalizations) / words * 100, 6),
        "target_avg_sentence_length": round(words / len(sentences), 6),
    }


def feature_specs(use_translationese_subfeatures: bool) -> list[tuple[str, str]]:
    if not use_translationese_subfeatures:
        return COMPOSITE_FEATURE_SPECS
    return [spec for spec in COMPOSITE_FEATURE_SPECS if spec[1] != "translationese_score"]


def feature_keys(use_translationese_subfeatures: bool) -> list[str]:
    keys = [name for _, name in feature_specs(use_translationese_subfeatures)]
    if use_translationese_subfeatures:
        keys.extend(TRANSLATIONESE_SUBFEATURE_KEYS)
    return keys


def merge_features(
    version_dir: Path,
    direction: str,
    use_translationese_subfeatures: bool = False,
) -> dict[str, dict[str, dict[str, float]]]:
    groups = {
        name: load_feature_group(version_dir / path, key)
        for name, (path, key) in CONFIGS[direction]["feature_files"].items()
    }
    sample_ids = set.intersection(*(set(group) for group in groups.values()))
    dataset_by_id = {}
    if use_translationese_subfeatures:
        dataset_rows = read_json(version_dir / CONFIGS[direction]["dataset"])
        dataset_by_id = {str(row["id"]): row for row in dataset_rows}
    merged: dict[str, dict[str, dict[str, float]]] = {}
    for sample_id in sample_ids:
        merged[sample_id] = {}
        for letter in CANDIDATES:
            merged[sample_id][letter] = {}
            for group_name, feature_name in feature_specs(use_translationese_subfeatures):
                merged[sample_id][letter][feature_name] = float(groups[group_name][sample_id][letter][feature_name])
            if use_translationese_subfeatures:
                candidate = str(dataset_by_id[sample_id][f"candidate_{letter}"])
                merged[sample_id][letter].update(translationese_subfeatures(direction, candidate))
    return merged


def winner_from_rank(rank: dict[str, int], a: str, b: str) -> str:
    return a if int(rank[a]) < int(rank[b]) else b


def top_candidate(rank: dict[str, int]) -> str:
    return min(rank, key=lambda letter: (int(rank[letter]), letter))


def pairwise_judgment_rows(rankings: dict[str, dict[str, dict[str, int]]]) -> list[dict[str, Any]]:
    rows = []
    for model, model_rows in sorted(rankings.items()):
        for sample_id, rank in sorted(model_rows.items()):
            for a, b in PAIRS:
                rows.append(
                    {
                        "model": model,
                        "sample_id": sample_id,
                        "candidate_a": a,
                        "candidate_b": b,
                        "winner": winner_from_rank(rank, a, b),
                    }
                )
    return rows


def pairwise_agreement_rows(rankings: dict[str, dict[str, dict[str, int]]]) -> list[dict[str, Any]]:
    rows = []
    for model_a, model_b in combinations(sorted(rankings), 2):
        total = 0
        agree = 0
        common_ids = sorted(set(rankings[model_a]) & set(rankings[model_b]))
        for sample_id in common_ids:
            rank_a = rankings[model_a][sample_id]
            rank_b = rankings[model_b][sample_id]
            for left, right in PAIRS:
                total += 1
                agree += int(winner_from_rank(rank_a, left, right) == winner_from_rank(rank_b, left, right))
        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "pairwise_agreement": round(agree / total, 6) if total else "",
                "n_pairwise_comparisons": total,
            }
        )
    return rows


def consensus_rows(rankings: dict[str, dict[str, dict[str, int]]]) -> list[dict[str, Any]]:
    sample_ids = sorted(set.intersection(*(set(rows) for rows in rankings.values())))
    rows = []
    for sample_id in sample_ids:
        rank_sums = {letter: 0 for letter in CANDIDATES}
        for model_rows in rankings.values():
            for letter, rank_value in model_rows[sample_id].items():
                rank_sums[letter] += int(rank_value)
        consensus_order = sorted(CANDIDATES, key=lambda letter: (rank_sums[letter], letter))
        consensus_rank = {letter: index for index, letter in enumerate(consensus_order, start=1)}
        rows.append(
            {
                "id": sample_id,
                "consensus_method": "Borda rank-sum over self-excluded judge set",
                "consensus_order": ">".join(consensus_order),
                "consensus_rank": consensus_rank,
                "top1_candidate": consensus_order[0],
                **{f"rank_sum_{letter}": rank_sums[letter] for letter in CANDIDATES},
            }
        )
    return rows


def model_vs_consensus_rows(
    rankings: dict[str, dict[str, dict[str, int]]],
    consensus: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    consensus_by_id = {row["id"]: row["consensus_rank"] for row in consensus}
    rows = []
    for model, model_rows in sorted(rankings.items()):
        top1_same = 0
        exact_same = 0
        pairwise_same = 0
        pairwise_total = 0
        for sample_id, rank in model_rows.items():
            consensus_rank = consensus_by_id[sample_id]
            top1_same += int(top_candidate(rank) == top_candidate(consensus_rank))
            exact_same += int(rank == consensus_rank)
            for a, b in PAIRS:
                pairwise_total += 1
                pairwise_same += int(winner_from_rank(rank, a, b) == winner_from_rank(consensus_rank, a, b))
        n = len(model_rows)
        rows.append(
            {
                "model": model,
                "top1_matches_consensus": round(top1_same / n, 6),
                "exact_rank_matches_consensus": round(exact_same / n, 6),
                "pairwise_matches_consensus": round(pairwise_same / pairwise_total, 6),
                "n_samples": n,
            }
        )
    return rows


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def standardize_matrix(matrix: list[list[float]]) -> list[list[float]]:
    cols = len(matrix[0])
    means = [sum(row[col] for row in matrix) / len(matrix) for col in range(cols)]
    stds: list[float] = []
    for col in range(cols):
        variance = sum((row[col] - means[col]) ** 2 for row in matrix) / len(matrix)
        stds.append(math.sqrt(variance) or 1.0)
    return [[(value - means[col]) / stds[col] for col, value in enumerate(row)] for row in matrix]


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
                augmented[row] = [value - factor * pivot_value for value, pivot_value in zip(augmented[row], augmented[col])]
    return [row[n:] for row in augmented]


def matmul(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    right_t = list(map(list, zip(*right)))
    return [[dot(row, col) for col in right_t] for row in left]


def fit_logistic(matrix: list[list[float]], labels: list[float], lr: float = 0.07, epochs: int = 3500, l2: float = 0.01) -> tuple[float, list[float], float, list[list[float]]]:
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
    return bias, weights, correct / n, x


def cluster_robust_covariance(design: list[list[float]], labels: list[float], beta: list[float], clusters: list[str], l2: float = 0.01) -> list[list[float]]:
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


def joint_observation_rows(
    rankings: dict[str, dict[str, dict[str, int]]],
    features_by_id: dict[str, dict[str, dict[str, float]]],
    keys: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for model, model_rows in sorted(rankings.items()):
        for sample_id, rank in sorted(model_rows.items()):
            if sample_id not in features_by_id:
                raise ValueError(f"missing features for sample {sample_id}")
            features = features_by_id[sample_id]
            for a, b in PAIRS:
                diff = [features[a][key] - features[b][key] for key in keys]
                winner = winner_from_rank(rank, a, b)
                rows.append(
                    {
                        "model": model,
                        "sample_id": sample_id,
                        "candidate_a": a,
                        "candidate_b": b,
                        "winner": winner,
                        **{f"diff_{key}": round(value, 6) for key, value in zip(keys, diff)},
                    }
                )
    return rows


def cluster_robust_joint_rows(observations: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    rows_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        rows_by_model[str(row["model"])].append(row)

    output_rows = []
    for model, model_rows in sorted(rows_by_model.items()):
        matrix = [[float(row[f"diff_{feature}"]) for feature in keys] for row in model_rows]
        labels = [1.0 if str(row["winner"]) == str(row["candidate_a"]) else 0.0 for row in model_rows]
        clusters = [str(row["sample_id"]) for row in model_rows]
        intercept, weights, accuracy, x_std = fit_logistic(matrix, labels)
        design = [[1.0, *row] for row in x_std]
        beta = [intercept, *weights]
        covariance = cluster_robust_covariance(design, labels, beta, clusters)
        standard_errors = [math.sqrt(max(covariance[idx][idx], 0.0)) for idx in range(len(covariance))]

        for idx, feature in enumerate(["intercept", *keys]):
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
                    "in_sample_accuracy": round(accuracy, 6),
                    "n_pairwise_observations": len(model_rows),
                    "n_clusters": len(set(clusters)),
                }
            )
    return output_rows


def feature_tier_rows(all_joint_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_joint_rows:
        if row["feature"] == "intercept":
            continue
        groups[(row["direction"], row["feature"])].append(row)

    tier_rows = []
    for (direction, feature), rows in sorted(groups.items()):
        total = len(rows)
        significant = [row for row in rows if row["cluster_significant_05"]]
        positive = sum(1 for row in rows if float(row["standardized_coefficient"]) > 0)
        negative = sum(1 for row in rows if float(row["standardized_coefficient"]) < 0)
        dominant_sign = "positive" if positive >= negative else "negative"
        dominant_count = max(positive, negative)
        if len(significant) == total and dominant_count == total:
            tier = "no_exception"
        elif len(significant) >= math.ceil(total * 0.625) and dominant_count >= math.ceil(total * 0.75):
            tier = "mostly_supported_with_exceptions"
        else:
            tier = "no_stable_evidence"
        tier_rows.append(
            {
                "direction": direction.upper(),
                "feature": feature,
                "tier": tier,
                "dominant_sign": dominant_sign,
                "significant_models": len(significant),
                "total_models": total,
                "positive_coefficients": positive,
                "negative_coefficients": negative,
            }
        )
    return tier_rows


def run_job(version: str, direction: str) -> dict[str, list[dict[str, Any]]]:
    version_dir = Path(version)
    exclusion = VERSION_EXCLUSIONS[version]
    prefix = CONFIGS[direction]["prefix"]
    file_prefix = f"{prefix}.judges_excluding_candidate_generator_{exclusion['excluded_model_slug']}"
    out_dir = version_dir / direction / "results/model_based_metrics/analysis/pilot"
    out_json_dir = out_dir / "json"
    out_csv_dir = out_dir / "csv"

    rankings = load_rankings(version_dir, direction, exclusion["excluded_model"])
    composite_keys = feature_keys(False)
    subfeature_keys = feature_keys(True)
    features_by_id = merge_features(version_dir, direction, use_translationese_subfeatures=False)
    pairwise_rows = pairwise_judgment_rows(rankings)
    agreement_rows = pairwise_agreement_rows(rankings)
    consensus = consensus_rows(rankings)
    model_consensus = model_vs_consensus_rows(rankings, consensus)
    observations = joint_observation_rows(rankings, features_by_id, composite_keys)
    joint_rows = cluster_robust_joint_rows(observations, composite_keys)

    subfeatures_by_id = merge_features(version_dir, direction, use_translationese_subfeatures=True)
    subfeature_observations = joint_observation_rows(rankings, subfeatures_by_id, subfeature_keys)
    subfeature_joint_rows = cluster_robust_joint_rows(subfeature_observations, subfeature_keys)

    write_json(
        out_json_dir / f"{file_prefix}.judge_set_metadata.json",
        {
            "version": version,
            "direction": direction.upper(),
            "candidate_generator": exclusion["candidate_generator"],
            "excluded_judge_model": exclusion["excluded_model"],
            "included_judge_models": sorted(rankings),
            "n_included_judges": len(rankings),
            "filename_prefix": file_prefix,
        },
    )
    write_json(out_json_dir / f"{file_prefix}.pairwise_judgments.json", pairwise_rows)
    write_json(out_json_dir / f"{file_prefix}.consensus_borda.json", consensus)
    write_json(out_json_dir / f"{file_prefix}.joint_features.pairwise_observations.json", observations)

    write_csv(
        out_csv_dir / f"{file_prefix}.pairwise_agreement.csv",
        agreement_rows,
        ["model_a", "model_b", "pairwise_agreement", "n_pairwise_comparisons"],
    )
    write_csv(
        out_csv_dir / f"{file_prefix}.consensus_borda_summary.csv",
        [
            {
                "id": row["id"],
                "consensus_order": row["consensus_order"],
                "top1_candidate": row["top1_candidate"],
                "consensus_rank": json.dumps(row["consensus_rank"], ensure_ascii=False, sort_keys=True),
                "rank_sum_A": row["rank_sum_A"],
                "rank_sum_B": row["rank_sum_B"],
                "rank_sum_C": row["rank_sum_C"],
            }
            for row in consensus
        ],
        ["id", "consensus_order", "top1_candidate", "consensus_rank", "rank_sum_A", "rank_sum_B", "rank_sum_C"],
    )
    write_csv(
        out_csv_dir / f"{file_prefix}.model_vs_consensus_agreement.csv",
        model_consensus,
        ["model", "top1_matches_consensus", "exact_rank_matches_consensus", "pairwise_matches_consensus", "n_samples"],
    )
    write_csv(
        out_csv_dir / f"{file_prefix}.joint_features.cluster_robust_pairwise_logistic.csv",
        joint_rows,
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
            "in_sample_accuracy",
            "n_pairwise_observations",
            "n_clusters",
        ],
    )
    write_json(
        out_csv_dir / f"{file_prefix}.joint_features.cluster_robust_pairwise_logistic.method_notes.json",
        {
            "version": version,
            "direction": direction.upper(),
            "candidate_generator": exclusion["candidate_generator"],
            "excluded_judge_model": exclusion["excluded_model"],
            "included_judge_models": sorted(rankings),
            "model": "standardized L2 pairwise logistic model per included judge model",
            "cluster": "sample_id",
            "accuracy_field": "in-sample accuracy; this is explanatory fit, not held-out prediction accuracy",
            "features": composite_keys,
            "l2": 0.01,
        },
    )
    write_json(
        out_json_dir / f"{file_prefix}.joint_features_with_translationese_subfeatures.pairwise_observations.json",
        subfeature_observations,
    )
    write_csv(
        out_csv_dir / f"{file_prefix}.joint_features_with_translationese_subfeatures.cluster_robust_pairwise_logistic.csv",
        subfeature_joint_rows,
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
            "in_sample_accuracy",
            "n_pairwise_observations",
            "n_clusters",
        ],
    )
    write_json(
        out_csv_dir / f"{file_prefix}.joint_features_with_translationese_subfeatures.cluster_robust_pairwise_logistic.method_notes.json",
        {
            "version": version,
            "direction": direction.upper(),
            "candidate_generator": exclusion["candidate_generator"],
            "excluded_judge_model": exclusion["excluded_model"],
            "included_judge_models": sorted(rankings),
            "model": "same joint model as the composite analysis, but the weighted translationese_score is replaced by independent translationese diagnostic subfeatures",
            "cluster": "sample_id",
            "accuracy_field": "in-sample accuracy; this is explanatory fit, not held-out prediction accuracy",
            "translationese_subfeatures": TRANSLATIONESE_SUBFEATURE_KEYS,
            "features": subfeature_keys,
            "l2": 0.01,
        },
    )
    print(f"wrote {version}/{direction}: {file_prefix} ({len(rankings)} judges)")
    for row in joint_rows:
        row["version"] = version
        row["direction"] = direction
    for row in subfeature_joint_rows:
        row["version"] = version
        row["direction"] = direction
    return {"composite": joint_rows, "translationese_subfeatures": subfeature_joint_rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=sorted(VERSION_EXCLUSIONS))
    parser.add_argument("--direction", choices=sorted(CONFIGS))
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        jobs = [(version, direction) for version in sorted(VERSION_EXCLUSIONS) for direction in sorted(CONFIGS)]
    else:
        if not args.version or not args.direction:
            parser.error("use --all or provide both --version and --direction")
        jobs = [(args.version, args.direction)]

    all_composite_rows: list[dict[str, Any]] = []
    all_subfeature_rows: list[dict[str, Any]] = []
    for version, direction in jobs:
        job_rows = run_job(version, direction)
        all_composite_rows.extend(job_rows["composite"])
        all_subfeature_rows.extend(job_rows["translationese_subfeatures"])

    if len(jobs) > 1:
        tier_rows = feature_tier_rows(all_composite_rows)
        subfeature_tier_rows = feature_tier_rows(all_subfeature_rows)
        summary_dir = Path("data/analysis_summaries")
        write_csv(
            summary_dir / "self_excluded_judge_set_feature_stability_tiers.csv",
            tier_rows,
            [
                "direction",
                "feature",
                "tier",
                "dominant_sign",
                "significant_models",
                "total_models",
                "positive_coefficients",
                "negative_coefficients",
            ],
        )
        write_csv(
            summary_dir / "self_excluded_judge_set_translationese_subfeature_stability_tiers.csv",
            subfeature_tier_rows,
            [
                "direction",
                "feature",
                "tier",
                "dominant_sign",
                "significant_models",
                "total_models",
                "positive_coefficients",
                "negative_coefficients",
            ],
        )
        print(f"wrote {summary_dir / 'self_excluded_judge_set_feature_stability_tiers.csv'}")
        print(f"wrote {summary_dir / 'self_excluded_judge_set_translationese_subfeature_stability_tiers.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
