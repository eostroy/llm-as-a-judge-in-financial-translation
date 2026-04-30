#!/usr/bin/env python
"""Rerun EC deep-feature analysis with OpenRouter Gemini embeddings.

This adds real cross-lingual embedding cosine similarity from
google/gemini-embedding-2-preview while keeping the non-embedding proxy
features from analyze_ec_deep_features.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from analyze_ec_deep_features import (
    CANDIDATES,
    DATASET,
    DEEP_FEATURE_KEYS,
    MODEL_FILES,
    OUT_DIR,
    PAIRS,
    candidate_features,
    fit_logistic,
    read_json,
    top_candidate,
    winner_from_rank,
    write_csv,
    write_json,
)


OPENROUTER_API_KEY = ""
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = "google/gemini-embedding-2-preview"
EMBEDDING_DIMENSIONS = 768
EMBEDDING_CACHE = OUT_DIR / "ffn_200ec.gemini_embedding_2_preview.cache.json"

EMBEDDING_FEATURE_KEYS = [
    "embedding_similarity",
    "extra_number_count",
    "extra_number_ratio",
    "extra_entity_count",
    "specificity_expansion",
    "financial_register_score",
    "statistical_register_score",
    "register_score",
    "syntactic_complexity",
    "translationese_score",
]


def get_openrouter_key_from_rank_script() -> str:
    script = Path("scripts/rank_translation_candidates_openrouter.py")
    if not script.exists():
        return ""
    match = re.search(r'OPENROUTER_API_KEY\s*=\s*"([^"]+)"', script.read_text(encoding="utf-8-sig"))
    return match.group(1) if match else ""


def load_rankings() -> dict[str, dict[str, dict[str, int]]]:
    rankings: dict[str, dict[str, dict[str, int]]] = {}
    for model, path in MODEL_FILES.items():
        rows = read_json(path)
        rankings[model] = {str(row["id"]): {key: int(value) for key, value in row["rank"].items()} for row in rows}
    return rankings


def embedding_key(text: str) -> str:
    return text.strip()


def load_cache(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected object cache")
    return {str(key): value for key, value in data.items()}


def save_cache(path: Path, cache: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False) + "\n", encoding="utf-8")


def call_embeddings(api_key: str, texts: list[str], timeout: int, retries: int) -> list[list[float]]:
    url = OPENROUTER_BASE_URL.rstrip("/") + "/embeddings"
    payload: dict[str, Any] = {
        "model": EMBEDDING_MODEL,
        "input": texts,
        "dimensions": EMBEDDING_DIMENSIONS,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "finance-translation-embedding-analysis",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            data = sorted(result["data"], key=lambda item: item["index"])
            vectors = [item["embedding"] for item in data]
            if len(vectors) != len(texts):
                raise ValueError(f"embedding count mismatch: got {len(vectors)}, expected {len(texts)}")
            return vectors
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {exc.code} {exc.reason}: {detail}")
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
        if attempt < retries:
            wait_s = min(2**attempt, 30)
            print(f"embedding attempt {attempt} failed: {last_error}; retrying in {wait_s}s")
            time.sleep(wait_s)
    raise RuntimeError(f"embedding failed after {retries} attempts: {last_error}")


def collect_texts(dataset_rows: list[dict[str, Any]]) -> list[str]:
    texts = []
    seen = set()
    for row in dataset_rows:
        for text in [row["source_text"], *(row[f"candidate_{letter}"] for letter in CANDIDATES)]:
            key = embedding_key(str(text))
            if key not in seen:
                seen.add(key)
                texts.append(key)
    return texts


def ensure_embeddings(
    dataset_rows: list[dict[str, Any]],
    api_key: str,
    cache_path: Path,
    batch_size: int,
    timeout: int,
    retries: int,
) -> dict[str, list[float]]:
    cache = load_cache(cache_path)
    texts = collect_texts(dataset_rows)
    missing = [text for text in texts if text not in cache]
    print(f"embedding cache: {len(cache)} existing, {len(missing)} missing, {len(texts)} total unique texts")
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        vectors = call_embeddings(api_key, batch, timeout=timeout, retries=retries)
        for text, vector in zip(batch, vectors):
            cache[text] = vector
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


def build_embedding_features(
    dataset_rows: list[dict[str, Any]],
    embeddings: dict[str, list[float]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, float]]]]:
    features_by_id: dict[str, dict[str, dict[str, float]]] = {}
    enriched = []
    for row in dataset_rows:
        sample_id = str(row["id"])
        source_vec = embeddings[embedding_key(str(row["source_text"]))]
        row_features: dict[str, dict[str, float]] = {}
        for letter in CANDIDATES:
            candidate = str(row[f"candidate_{letter}"])
            base = candidate_features(row, letter)
            base.pop("cross_lingual_semantic_similarity_proxy", None)
            base.pop("nli_consistency_proxy", None)
            candidate_vec = embeddings[embedding_key(candidate)]
            row_features[letter] = {
                **base,
                "embedding_similarity": round(cosine(source_vec, candidate_vec), 6),
            }
        features_by_id[sample_id] = row_features
        enriched.append({**row, "gemini_embedding_features": row_features})
    return enriched, features_by_id


def top1_average_rows(
    rankings: dict[str, dict[str, dict[str, int]]],
    features_by_id: dict[str, dict[str, dict[str, float]]],
) -> list[dict[str, Any]]:
    rows = []
    for model, model_rows in sorted(rankings.items()):
        totals = {key: 0.0 for key in EMBEDDING_FEATURE_KEYS}
        top_counts = {letter: 0 for letter in CANDIDATES}
        for sample_id, rank in model_rows.items():
            top = top_candidate(rank)
            top_counts[top] += 1
            for key in EMBEDDING_FEATURE_KEYS:
                totals[key] += features_by_id[sample_id][top][key]
        n = len(model_rows)
        row = {"model": model, "n": n, "top_A": top_counts["A"], "top_B": top_counts["B"], "top_C": top_counts["C"]}
        row.update({f"avg_{key}": round(totals[key] / n, 6) for key in EMBEDDING_FEATURE_KEYS})
        rows.append(row)
    return rows


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
                x.append([features[a][key] - features[b][key] for key in EMBEDDING_FEATURE_KEYS])
                y.append(1 if winner_from_rank(rank, a, b) == a else 0)
        bias, weights, accuracy = fit_logistic(x, y)
        for key, weight in zip(EMBEDDING_FEATURE_KEYS, weights):
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


def feature_rank_rows(top1_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lower_is_more = {"extra_number_count", "extra_number_ratio", "extra_entity_count", "specificity_expansion", "translationese_score"}
    rows = []
    for key in EMBEDDING_FEATURE_KEYS:
        metric = f"avg_{key}"
        ordered = sorted(top1_rows, key=lambda row: row[metric], reverse=key not in lower_is_more)
        for rank, row in enumerate(ordered, start=1):
            rows.append({"feature": key, "model": row["model"], "value": row[metric], "rank_for_feature": rank})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key", default=os.getenv("OPENROUTER_API_KEY") or OPENROUTER_API_KEY or get_openrouter_key_from_rank_script())
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--cache", type=Path, default=EMBEDDING_CACHE)
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("No OpenRouter API key provided.")

    dataset_rows = read_json(DATASET)
    rankings = load_rankings()
    embeddings = ensure_embeddings(
        dataset_rows,
        api_key=args.api_key,
        cache_path=args.cache,
        batch_size=args.batch_size,
        timeout=args.timeout,
        retries=args.retries,
    )
    enriched, features_by_id = build_embedding_features(dataset_rows, embeddings)
    top1_rows = top1_average_rows(rankings, features_by_id)
    preference_rows = pairwise_preference_rows(rankings, features_by_id)
    rank_rows = feature_rank_rows(top1_rows)

    write_json(OUT_DIR / "ffn_200ec.gemini_embedding_features.by_candidate.json", enriched)
    write_csv(
        OUT_DIR / "ffn_200ec.gemini_embedding_features.model_top1_averages.csv",
        top1_rows,
        ["model", "n", "top_A", "top_B", "top_C"] + [f"avg_{key}" for key in EMBEDDING_FEATURE_KEYS],
    )
    write_csv(
        OUT_DIR / "ffn_200ec.gemini_embedding_features.pairwise_logistic_preferences.csv",
        preference_rows,
        ["model", "feature", "standardized_coefficient", "training_accuracy", "intercept", "n_pairwise_observations"],
    )
    write_csv(
        OUT_DIR / "ffn_200ec.gemini_embedding_features.model_feature_ranks.csv",
        rank_rows,
        ["feature", "model", "value", "rank_for_feature"],
    )
    write_json(
        OUT_DIR / "ffn_200ec.gemini_embedding_features.method_notes.json",
        [
            {
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dimensions": EMBEDDING_DIMENSIONS,
                "embedding_similarity": "cosine(source_text_embedding, candidate_embedding)",
                "local_semantic_proxy_features": "removed from this embedding-based run",
            }
        ],
    )
    print(f"wrote Gemini embedding analysis to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



