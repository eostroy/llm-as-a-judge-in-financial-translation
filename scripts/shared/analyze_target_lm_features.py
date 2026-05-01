#!/usr/bin/env python
"""Run target-language LM fluency features for EC or CE ranking analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


CANDIDATES = ("A", "B", "C")
PAIRS = tuple(combinations(CANDIDATES, 2))
FEATURE_KEYS = ["target_lm_log_perplexity", "target_lm_naturalness_score"]

CONFIGS = {
    "ec": {
        "dataset": Path("ec/datasets/ffn_200ec.with_candidates.shuffled.json"),
        "out_dir": Path("ec/results/model_based_metrics/analysis/pilot"),
        "prefix": "ffn_200ec",
        "default_model": "uer/gpt2-chinese-cluecorpussmall",
        "target_language": "Chinese",
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
        "default_model": "distilgpt2",
        "target_language": "English",
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


def safe_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model_name).strip("_")


def load_rankings(model_files: dict[str, Path]) -> dict[str, dict[str, dict[str, int]]]:
    rankings = {}
    for model, path in model_files.items():
        rows = read_json(path)
        rankings[model] = {str(row["id"]): {letter: int(value) for letter, value in row["rank"].items()} for row in rows}
    return rankings


def load_cache(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_cache(path: Path, cache: dict[str, dict[str, float]]) -> None:
    write_json(path, cache)


def text_key(text: str) -> str:
    return text.strip()


def score_text(
    text: str,
    tokenizer: Any,
    model: Any,
    device: torch.device,
    max_length: int,
) -> dict[str, float]:
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    if input_ids.numel() < 2:
        return {
            "target_lm_log_perplexity": 0.0,
            "target_lm_perplexity": 1.0,
            "target_lm_naturalness_score": -0.0,
            "target_lm_token_count": float(input_ids.numel()),
        }
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    log_ppl = float(output.loss.detach().cpu())
    return {
        "target_lm_log_perplexity": round(log_ppl, 6),
        "target_lm_perplexity": round(math.exp(min(log_ppl, 20.0)), 6),
        "target_lm_naturalness_score": round(-log_ppl, 6),
        "target_lm_token_count": float(input_ids.numel()),
    }


def ensure_lm_scores(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    cache_path: Path,
    device: torch.device,
    max_length: int,
) -> dict[str, dict[str, float]]:
    cache = load_cache(cache_path)
    texts = sorted({text_key(str(row[f"candidate_{letter}"])) for row in rows for letter in CANDIDATES})
    missing = [text for text in texts if text not in cache]
    print(f"target LM cache: {len(cache)} existing, {len(missing)} missing, {len(texts)} total unique texts")
    model.eval()
    for idx, text in enumerate(missing, start=1):
        cache[text] = score_text(text, tokenizer, model, device, max_length)
        if idx % 25 == 0 or idx == len(missing):
            save_cache(cache_path, cache)
            print(f"LM scored {idx}/{len(missing)} missing texts")
    return cache


def compute_features(
    rows: list[dict[str, Any]],
    scores: dict[str, dict[str, float]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, float]]]]:
    enriched = []
    features_by_id = {}
    for row in rows:
        sample_id = str(row["id"])
        features = {}
        for letter in CANDIDATES:
            features[letter] = scores[text_key(str(row[f"candidate_{letter}"]))]
        features_by_id[sample_id] = features
        enriched.append({**row, "target_lm_features": features})
    return enriched, features_by_id


def top_candidate(rank: dict[str, int]) -> str:
    return min(CANDIDATES, key=lambda letter: int(rank[letter]))


def winner_from_rank(rank: dict[str, int], a: str, b: str) -> str:
    return a if int(rank[a]) < int(rank[b]) else b


def top1_average_rows(
    rankings: dict[str, dict[str, dict[str, int]]],
    features_by_id: dict[str, dict[str, dict[str, float]]],
) -> list[dict[str, Any]]:
    rows = []
    for model, model_rows in sorted(rankings.items()):
        totals = {key: 0.0 for key in FEATURE_KEYS}
        top_counts = Counter()
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


def standardize_column(values: list[float]) -> list[float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance) or 1.0
    return [(value - mean) / std for value in values]


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def fit_univariate_logistic(x: list[float], y: list[int], lr: float = 0.08, epochs: int = 2500) -> tuple[float, float, float]:
    xs = standardize_column(x)
    bias = 0.0
    weight = 0.0
    n = len(y)
    for _ in range(epochs):
        grad_b = 0.0
        grad_w = 0.0
        for value, target in zip(xs, y):
            pred = sigmoid(bias + weight * value)
            err = pred - target
            grad_b += err
            grad_w += err * value
        bias -= lr * grad_b / n
        weight -= lr * grad_w / n
    correct = 0
    for value, target in zip(xs, y):
        pred = 1 if sigmoid(bias + weight * value) >= 0.5 else 0
        correct += int(pred == target)
    return bias, weight, correct / n


def pairwise_preference_rows(
    rankings: dict[str, dict[str, dict[str, int]]],
    features_by_id: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for model, model_rows in sorted(rankings.items()):
        for key in FEATURE_KEYS:
            x = []
            y = []
            for sample_id, rank in model_rows.items():
                features = features_by_id[sample_id]
                for a, b in PAIRS:
                    x.append(features[a][key] - features[b][key])
                    y.append(1 if winner_from_rank(rank, a, b) == a else 0)
            bias, weight, accuracy = fit_univariate_logistic(x, y)
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
    parser.add_argument("--model")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    cfg = CONFIGS[args.direction]
    model_name = args.model or cfg["default_model"]
    out_dir = cfg["out_dir"]
    out_json_dir = out_dir / "json"
    out_csv_dir = out_dir / "csv"
    prefix = cfg["prefix"]
    model_slug = safe_model_name(model_name)
    cache_path = out_json_dir / f"{prefix}.target_lm.{model_slug}.cache.json"

    rows = read_json(cfg["dataset"])
    rankings = load_rankings(cfg["model_files"])
    expected_ids = {str(row["id"]) for row in rows}
    missing = {model: sorted(expected_ids - set(model_rows)) for model, model_rows in rankings.items()}
    missing = {model: ids for model, ids in missing.items() if ids}
    if missing:
        raise ValueError(f"ranking outputs missing ids: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=not args.allow_download)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=not args.allow_download).to(device)

    scores = ensure_lm_scores(rows, tokenizer, model, cache_path, device, args.max_length)
    enriched, features_by_id = compute_features(rows, scores)
    top1_rows = top1_average_rows(rankings, features_by_id)
    preference_rows = pairwise_preference_rows(rankings, features_by_id)

    write_json(out_json_dir / f"{prefix}.target_lm_features.by_candidate.json", enriched)
    write_csv(
        out_csv_dir / f"{prefix}.target_lm_features.model_top1_averages.csv",
        top1_rows,
        ["model", "n", "top_A", "top_B", "top_C"] + [f"avg_{key}" for key in FEATURE_KEYS],
    )
    write_csv(
        out_csv_dir / f"{prefix}.target_lm_features.pairwise_logistic_preferences.csv",
        preference_rows,
        ["model", "feature", "standardized_coefficient", "training_accuracy", "intercept", "n_pairwise_observations"],
    )
    write_json(
        out_json_dir / f"{prefix}.target_lm_features.method_notes.json",
        [
            {
                "target_language": cfg["target_language"],
                "model": model_name,
                "device": str(device),
                "target_lm_log_perplexity": "causal LM cross-entropy loss on the candidate translation; lower is more fluent under the target-language LM",
                "target_lm_naturalness_score": "-target_lm_log_perplexity; higher is more fluent under the target-language LM",
                "max_length": args.max_length,
                "local_files_only": not args.allow_download,
            }
        ],
    )
    print(f"wrote target LM features for {args.direction} to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
