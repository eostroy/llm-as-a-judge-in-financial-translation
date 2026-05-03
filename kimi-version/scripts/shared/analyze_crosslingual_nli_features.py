#!/usr/bin/env python
"""Run cross-lingual NLI features for EC or CE ranking analysis.

The script computes bidirectional NLI scores for every source/candidate pair:

- forward: source_text -> candidate_translation
- backward: candidate_translation -> source_text

It then writes candidate-level features, model Top-1 feature averages, and
pairwise logistic preference coefficients.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


CANDIDATES = ("A", "B", "C")
PAIRS = tuple(combinations(CANDIDATES, 2))
DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

FEATURE_KEYS = [
    "forward_entailment",
    "forward_neutral",
    "forward_contradiction",
    "backward_entailment",
    "backward_neutral",
    "backward_contradiction",
    "bidirectional_entailment",
    "nli_boundary_risk",
    "nli_omission_risk",
    "nli_contradiction_risk",
    "semantic_fidelity_score",
]


CONFIGS = {
    "ec": {
        "dataset": Path("ec/datasets/ffn_200ec.with_candidates.shuffled.json"),
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
    },
    "ce": {
        "dataset": Path("ce/datasets/ecpcfe_200ce.with_candidates.shuffled.json"),
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
    rankings: dict[str, dict[str, dict[str, int]]] = {}
    for model, path in model_files.items():
        rows = read_json(path)
        rankings[model] = {str(row["id"]): {letter: int(value) for letter, value in row["rank"].items()} for row in rows}
    return rankings


def top_candidate(rank: dict[str, int]) -> str:
    return min(CANDIDATES, key=lambda letter: int(rank[letter]))


def winner_from_rank(rank: dict[str, int], a: str, b: str) -> str:
    return a if int(rank[a]) < int(rank[b]) else b


def label_map(model: Any) -> dict[str, int]:
    mapping = {}
    for idx, label in model.config.id2label.items():
        lowered = str(label).lower()
        if "entail" in lowered:
            mapping["entailment"] = int(idx)
        elif "neutral" in lowered:
            mapping["neutral"] = int(idx)
        elif "contrad" in lowered:
            mapping["contradiction"] = int(idx)
    missing = {"entailment", "neutral", "contradiction"} - set(mapping)
    if missing:
        raise ValueError(f"model label map missing {missing}: {model.config.id2label}")
    return mapping


def nli_scores(
    tokenizer: Any,
    model: Any,
    labels: dict[str, int],
    pairs: list[tuple[str, str]],
    batch_size: int,
    device: torch.device,
) -> list[dict[str, float]]:
    scores: list[dict[str, float]] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            premises = [item[0] for item in batch]
            hypotheses = [item[1] for item in batch]
            encoded = tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu()
            for prob in probs:
                scores.append(
                    {
                        "entailment": round(float(prob[labels["entailment"]]), 6),
                        "neutral": round(float(prob[labels["neutral"]]), 6),
                        "contradiction": round(float(prob[labels["contradiction"]]), 6),
                    }
                )
            print(f"NLI scored {min(start + batch_size, len(pairs))}/{len(pairs)} pairs")
    return scores


def load_cache(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected object")
    return data


def save_cache(path: Path, cache: dict[str, dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False) + "\n", encoding="utf-8")


def compute_nli_features(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    labels: dict[str, int],
    cache_path: Path,
    batch_size: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, float]]]]:
    cache = load_cache(cache_path)
    pending_keys: list[str] = []
    pending_pairs: list[tuple[str, str]] = []
    for row in rows:
        sample_id = str(row["id"])
        source = str(row["source_text"])
        for letter in CANDIDATES:
            candidate = str(row[f"candidate_{letter}"])
            forward_key = f"{sample_id}:{letter}:forward"
            backward_key = f"{sample_id}:{letter}:backward"
            if forward_key not in cache:
                pending_keys.append(forward_key)
                pending_pairs.append((source, candidate))
            if backward_key not in cache:
                pending_keys.append(backward_key)
                pending_pairs.append((candidate, source))

    if pending_pairs:
        scored = nli_scores(tokenizer, model, labels, pending_pairs, batch_size, device)
        for key, score in zip(pending_keys, scored):
            cache[key] = score
        save_cache(cache_path, cache)

    enriched = []
    features_by_id: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        sample_id = str(row["id"])
        row_features = {}
        for letter in CANDIDATES:
            forward = cache[f"{sample_id}:{letter}:forward"]
            backward = cache[f"{sample_id}:{letter}:backward"]
            bidirectional = min(forward["entailment"], backward["entailment"])
            boundary_risk = forward["neutral"] + forward["contradiction"]
            omission_risk = backward["neutral"] + backward["contradiction"]
            contradiction_risk = max(forward["contradiction"], backward["contradiction"])
            fidelity = 0.5 * (forward["entailment"] + backward["entailment"]) - contradiction_risk
            features = {
                "forward_entailment": forward["entailment"],
                "forward_neutral": forward["neutral"],
                "forward_contradiction": forward["contradiction"],
                "backward_entailment": backward["entailment"],
                "backward_neutral": backward["neutral"],
                "backward_contradiction": backward["contradiction"],
                "bidirectional_entailment": round(bidirectional, 6),
                "nli_boundary_risk": round(boundary_risk, 6),
                "nli_omission_risk": round(omission_risk, 6),
                "nli_contradiction_risk": round(contradiction_risk, 6),
                "semantic_fidelity_score": round(fidelity, 6),
            }
            row_features[letter] = features
        features_by_id[sample_id] = row_features
        enriched.append({**row, "crosslingual_nli_features": row_features})
    return enriched, features_by_id


def top1_average_rows(
    rankings: dict[str, dict[str, dict[str, int]]],
    features_by_id: dict[str, dict[str, dict[str, float]]],
) -> list[dict[str, Any]]:
    rows = []
    for model, model_rows in sorted(rankings.items()):
        totals = {key: 0.0 for key in FEATURE_KEYS}
        top_counts = {letter: 0 for letter in CANDIDATES}
        for sample_id, rank in model_rows.items():
            top = top_candidate(rank)
            top_counts[top] += 1
            for key in FEATURE_KEYS:
                totals[key] += features_by_id[sample_id][top][key]
        n = len(model_rows)
        row = {"model": model, "n": n, "top_A": top_counts["A"], "top_B": top_counts["B"], "top_C": top_counts["C"]}
        row.update({f"avg_{key}": round(totals[key] / n, 6) for key in FEATURE_KEYS})
        rows.append(row)
    return rows


def standardize_matrix(matrix: list[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    cols = len(matrix[0])
    means = [sum(row[col] for row in matrix) / len(matrix) for col in range(cols)]
    stds = []
    for col in range(cols):
        variance = sum((row[col] - means[col]) ** 2 for row in matrix) / len(matrix)
        stds.append(math.sqrt(variance) or 1.0)
    scaled = [[(value - means[col]) / stds[col] for col, value in enumerate(row)] for row in matrix]
    return scaled, means, stds


def fit_logistic(x: list[list[float]], y: list[int], epochs: int = 2500, lr: float = 0.08) -> tuple[float, list[float], float]:
    x_scaled, _, _ = standardize_matrix(x)
    weights = [0.0 for _ in x_scaled[0]]
    bias = 0.0
    l2 = 0.01
    n = len(y)
    for _ in range(epochs):
        grad_w = [0.0 for _ in weights]
        grad_b = 0.0
        for row, label in zip(x_scaled, y):
            z = bias + sum(weight * value for weight, value in zip(weights, row))
            pred = 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, z))))
            error = pred - label
            grad_b += error
            for i, value in enumerate(row):
                grad_w[i] += error * value
        bias -= lr * grad_b / n
        for i in range(len(weights)):
            weights[i] -= lr * ((grad_w[i] / n) + l2 * weights[i])
    correct = 0
    for row, label in zip(x_scaled, y):
        z = bias + sum(weight * value for weight, value in zip(weights, row))
        pred = 1 if z >= 0 else 0
        correct += pred == label
    return bias, weights, correct / n


def pairwise_preference_rows(
    rankings: dict[str, dict[str, dict[str, int]]],
    features_by_id: dict[str, dict[str, dict[str, float]]],
) -> list[dict[str, Any]]:
    rows = []
    for model, model_rows in sorted(rankings.items()):
        x = []
        y = []
        for sample_id, rank in model_rows.items():
            features = features_by_id[sample_id]
            for a, b in PAIRS:
                x.append([features[a][key] - features[b][key] for key in FEATURE_KEYS])
                y.append(1 if winner_from_rank(rank, a, b) == a else 0)
        bias, weights, accuracy = fit_logistic(x, y)
        for key, weight in zip(FEATURE_KEYS, weights):
            rows.append(
                {
                    "model": model,
                    "feature": key,
                    "standardized_coefficient": round(weight, 6),
                    "training_accuracy": round(accuracy, 6),
                    "intercept": round(bias, 6),
                    "n_pairwise_observations": len(y),
                }
            )
    return rows


def univariate_pairwise_preference_rows(
    rankings: dict[str, dict[str, dict[str, int]]],
    features_by_id: dict[str, dict[str, dict[str, float]]],
) -> list[dict[str, Any]]:
    rows = []
    for model, model_rows in sorted(rankings.items()):
        for key in FEATURE_KEYS:
            x = []
            y = []
            for sample_id, rank in model_rows.items():
                features = features_by_id[sample_id]
                for a, b in PAIRS:
                    x.append([features[a][key] - features[b][key]])
                    y.append(1 if winner_from_rank(rank, a, b) == a else 0)
            bias, weights, accuracy = fit_logistic(x, y)
            rows.append(
                {
                    "model": model,
                    "feature": key,
                    "standardized_coefficient": round(weights[0], 6),
                    "training_accuracy": round(accuracy, 6),
                    "intercept": round(bias, 6),
                    "n_pairwise_observations": len(y),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("direction", choices=sorted(CONFIGS))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    cfg = CONFIGS[args.direction]
    out_dir = cfg["out_dir"]
    out_json_dir = out_dir / "json"
    out_csv_dir = out_dir / "csv"
    prefix = cfg["prefix"]
    cache_path = out_json_dir / f"{prefix}.crosslingual_nli.cache.json"

    rows = read_json(cfg["dataset"])
    rankings = load_rankings(cfg["model_files"])
    expected_ids = {str(row["id"]) for row in rows}
    missing = {model: sorted(expected_ids - set(model_rows)) for model, model_rows in rankings.items()}
    missing = {model: ids for model, ids in missing.items() if ids}
    if missing:
        raise ValueError(f"ranking outputs missing ids: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=not args.allow_download)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        local_files_only=not args.allow_download,
    ).to(device)
    labels = label_map(model)

    enriched, features_by_id = compute_nli_features(
        rows,
        tokenizer,
        model,
        labels,
        cache_path,
        args.batch_size,
        device,
    )
    top1_rows = top1_average_rows(rankings, features_by_id)
    preference_rows = pairwise_preference_rows(rankings, features_by_id)
    univariate_preference_rows = univariate_pairwise_preference_rows(rankings, features_by_id)

    write_json(out_json_dir / f"{prefix}.crosslingual_nli_features.by_candidate.json", enriched)
    write_csv(
        out_csv_dir / f"{prefix}.crosslingual_nli_features.model_top1_averages.csv",
        top1_rows,
        ["model", "n", "top_A", "top_B", "top_C"] + [f"avg_{key}" for key in FEATURE_KEYS],
    )
    write_csv(
        out_csv_dir / f"{prefix}.crosslingual_nli_features.pairwise_logistic_preferences.csv",
        preference_rows,
        ["model", "feature", "standardized_coefficient", "training_accuracy", "intercept", "n_pairwise_observations"],
    )
    write_csv(
        out_csv_dir / f"{prefix}.crosslingual_nli_features.univariate_pairwise_logistic_preferences.csv",
        univariate_preference_rows,
        ["model", "feature", "standardized_coefficient", "training_accuracy", "intercept", "n_pairwise_observations"],
    )
    write_json(
        out_json_dir / f"{prefix}.crosslingual_nli_features.method_notes.json",
        [
            {
                "model": args.model,
                "device": str(device),
                "forward": "source_text as premise, candidate translation as hypothesis",
                "backward": "candidate translation as premise, source_text as hypothesis",
                "bidirectional_entailment": "min(forward_entailment, backward_entailment)",
                "nli_boundary_risk": "forward_neutral + forward_contradiction; proxy for source-not-entailing-candidate information",
                "nli_omission_risk": "backward_neutral + backward_contradiction; proxy for candidate-not-entailing-source omissions",
                "semantic_fidelity_score": "mean forward/backward entailment minus max contradiction risk",
                "local_files_only": not args.allow_download,
            }
        ],
    )
    print(f"wrote cross-lingual NLI features for {args.direction} to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
