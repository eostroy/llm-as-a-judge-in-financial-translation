#!/usr/bin/env python
"""Run local cross-lingual embedding similarity for EC or CE ranking analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from itertools import combinations
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer


CANDIDATES = ("A", "B", "C")
PAIRS = tuple(combinations(CANDIDATES, 2))
DEFAULT_MODEL = "bert-base-multilingual-cased"
FEATURE_KEYS = ["crosslingual_embedding_similarity"]

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


def safe_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model_name).strip("_")


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


def top_candidate(rank: dict[str, int]) -> str:
    return min(CANDIDATES, key=lambda letter: int(rank[letter]))


def winner_from_rank(rank: dict[str, int], a: str, b: str) -> str:
    return a if int(rank[a]) < int(rank[b]) else b


def load_cache(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected object")
    return {str(key): value for key, value in data.items()}


def save_cache(path: Path, cache: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False) + "\n", encoding="utf-8")


def embedding_key(text: str) -> str:
    return text.strip()


def collect_texts(rows: list[dict[str, Any]]) -> list[str]:
    seen = set()
    texts = []
    for row in rows:
        for text in [row["source_text"], *(row[f"candidate_{letter}"] for letter in CANDIDATES)]:
            key = embedding_key(str(text))
            if key not in seen:
                seen.add(key)
                texts.append(key)
    return texts


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = torch.sum(last_hidden * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def ensure_embeddings(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    cache_path: Path,
    batch_size: int,
    device: torch.device,
) -> dict[str, list[float]]:
    cache = load_cache(cache_path)
    texts = collect_texts(rows)
    missing = [text for text in texts if text not in cache]
    print(f"embedding cache: {len(cache)} existing, {len(missing)} missing, {len(texts)} total unique texts")
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            output = model(**encoded)
            pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu()
            for text, vector in zip(batch, pooled):
                cache[text] = [round(float(value), 8) for value in vector.tolist()]
            save_cache(cache_path, cache)
            print(f"embedded {min(start + len(batch), len(missing))}/{len(missing)} missing texts")
    return cache


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if not mag_a or not mag_b:
        return 0.0
    return dot / (mag_a * mag_b)


def compute_features(
    rows: list[dict[str, Any]],
    embeddings: dict[str, list[float]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, float]]]]:
    enriched = []
    features_by_id = {}
    for row in rows:
        sample_id = str(row["id"])
        source_vec = embeddings[embedding_key(str(row["source_text"]))]
        row_features = {}
        for letter in CANDIDATES:
            candidate_vec = embeddings[embedding_key(str(row[f"candidate_{letter}"]))]
            row_features[letter] = {
                "crosslingual_embedding_similarity": round(cosine(source_vec, candidate_vec), 6),
            }
        features_by_id[sample_id] = row_features
        enriched.append({**row, "local_embedding_features": row_features})
    return enriched, features_by_id


def top1_average_rows(rankings: dict[str, dict[str, dict[str, int]]], features_by_id: dict[str, Any]) -> list[dict[str, Any]]:
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


def standardize_matrix(matrix: list[list[float]]) -> list[list[float]]:
    cols = len(matrix[0])
    means = [sum(row[col] for row in matrix) / len(matrix) for col in range(cols)]
    stds = []
    for col in range(cols):
        variance = sum((row[col] - means[col]) ** 2 for row in matrix) / len(matrix)
        stds.append(math.sqrt(variance) or 1.0)
    return [[(value - means[col]) / stds[col] for col, value in enumerate(row)] for row in matrix]


def fit_logistic(x: list[list[float]], y: list[int], epochs: int = 2500, lr: float = 0.08) -> tuple[float, list[float], float]:
    x_scaled = standardize_matrix(x)
    weights = [0.0 for _ in x_scaled[0]]
    bias = 0.0
    l2 = 0.01
    n = len(y)
    for _ in range(epochs):
        grad_w = [0.0 for _ in weights]
        grad_b = 0.0
        for row, label in zip(x_scaled, y):
            z = max(-35.0, min(35.0, bias + sum(weight * value for weight, value in zip(weights, row))))
            pred = 1.0 / (1.0 + math.exp(-z))
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
        correct += (1 if z >= 0 else 0) == label
    return bias, weights, correct / n


def pairwise_preference_rows(rankings: dict[str, dict[str, dict[str, int]]], features_by_id: dict[str, Any]) -> list[dict[str, Any]]:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("direction", choices=sorted(CONFIGS))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    cfg = CONFIGS[args.direction]
    out_dir = cfg["out_dir"]
    out_json_dir = out_dir / "json"
    out_csv_dir = out_dir / "csv"
    prefix = cfg["prefix"]
    model_slug = safe_model_name(args.model)
    cache_path = out_json_dir / f"{prefix}.local_embedding.{model_slug}.cache.json"

    rows = read_json(cfg["dataset"])
    rankings = load_rankings(cfg["model_files"])
    expected_ids = {str(row["id"]) for row in rows}
    missing = {model: sorted(expected_ids - set(model_rows)) for model, model_rows in rankings.items()}
    missing = {model: ids for model, ids in missing.items() if ids}
    if missing:
        raise ValueError(f"ranking outputs missing ids: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=not args.allow_download)
    model = AutoModel.from_pretrained(args.model, local_files_only=not args.allow_download).to(device)

    embeddings = ensure_embeddings(rows, tokenizer, model, cache_path, args.batch_size, device)
    enriched, features_by_id = compute_features(rows, embeddings)
    top1_rows = top1_average_rows(rankings, features_by_id)
    preference_rows = pairwise_preference_rows(rankings, features_by_id)

    write_json(out_json_dir / f"{prefix}.local_embedding_features.by_candidate.json", enriched)
    write_csv(
        out_csv_dir / f"{prefix}.local_embedding_features.model_top1_averages.csv",
        top1_rows,
        ["model", "n", "top_A", "top_B", "top_C"] + [f"avg_{key}" for key in FEATURE_KEYS],
    )
    write_csv(
        out_csv_dir / f"{prefix}.local_embedding_features.pairwise_logistic_preferences.csv",
        preference_rows,
        ["model", "feature", "standardized_coefficient", "training_accuracy", "intercept", "n_pairwise_observations"],
    )
    write_json(
        out_json_dir / f"{prefix}.local_embedding_features.method_notes.json",
        [
            {
                "embedding_model": args.model,
                "device": str(device),
                "embedding": "mean-pooled last hidden states with attention mask, L2-normalized",
                "crosslingual_embedding_similarity": "cosine(source_text_embedding, candidate_embedding)",
                "local_files_only": not args.allow_download,
            }
        ],
    )
    print(f"wrote local embedding features for {args.direction} to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
