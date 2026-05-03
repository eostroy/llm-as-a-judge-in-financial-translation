#!/usr/bin/env python
"""Analyze CE syntax and information-structure features with Stanza.

Metrics:
- clause_count
- dependency_depth
- mean_dependency_distance
- max_dependency_distance
- normalized_dependency_distance
- nominalization_ratio
- modifier_density
- coordination_count
- passive_count
- sentence_compression_ratio
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import stanza

from analyze_ce_deep_features import (
    CANDIDATES,
    DATASET,
    MODEL_FILES,
    PAIRS,
    fit_logistic,
    read_json,
    top_candidate,
    winner_from_rank,
    write_csv,
    write_json,
)


OUT_DIR = Path("ce/results/parser_derived_syntactic_metrics/analysis/pilot")
OUT_JSON_DIR = OUT_DIR / "json"
OUT_CSV_DIR = OUT_DIR / "csv"
FEATURE_KEYS = [
    "clause_count",
    "dependency_depth",
    "mean_dependency_distance",
    "max_dependency_distance",
    "normalized_dependency_distance",
    "nominalization_ratio",
    "modifier_density",
    "coordination_count",
    "passive_count",
    "sentence_compression_ratio",
]

CACHE_PATH = OUT_JSON_DIR / "ecpcfe_200ce.syntax_info_features.cache.json"
CLAUSE_DEPRELS = {"root", "ccomp", "xcomp", "acl", "advcl", "parataxis", "conj", "relcl"}
MODIFIER_DEPRELS = {"amod", "advmod", "nmod", "acl", "compound", "det", "case", "obl", "appos"}
NOMINALIZATION_SUFFIXES = (
    "tion",
    "sion",
    "ment",
    "ness",
    "ity",
    "ance",
    "ence",
    "ism",
    "ship",
    "age",
)


def source_char_count(source: str) -> int:
    return max(len(re.findall(r"[\u4e00-\u9fff]", source)), 1)


def target_word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)?", text))


def load_rankings() -> dict[str, dict[str, dict[str, int]]]:
    rankings: dict[str, dict[str, dict[str, int]]] = {}
    for model, path in MODEL_FILES.items():
        rows = read_json(path)
        rankings[model] = {str(row["id"]): {key: int(value) for key, value in row["rank"].items()} for row in rows}
    return rankings


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


def max_dependency_depth(sentence: Any) -> int:
    children: dict[int, list[int]] = {}
    for word in sentence.words:
        children.setdefault(int(word.head), []).append(int(word.id))

    def depth(node_id: int) -> int:
        child_ids = children.get(node_id, [])
        if not child_ids:
            return 0
        return 1 + max(depth(child_id) for child_id in child_ids)

    return max((depth(root_id) for root_id in children.get(0, [])), default=0)


def dependency_distance_stats(sentences: list[Any]) -> tuple[float, float, float]:
    distances = []
    token_count = 0
    for sentence in sentences:
        sentence_words = list(sentence.words)
        token_count += len(sentence_words)
        for word in sentence_words:
            head = int(word.head)
            if head == 0:
                continue
            distances.append(abs(int(word.id) - head))
    if not distances:
        return 0.0, 0.0, 0.0
    mean_distance = sum(distances) / len(distances)
    max_distance = max(distances)
    normalized_distance = mean_distance / max(token_count - len(sentences), 1)
    return mean_distance, float(max_distance), normalized_distance


def is_passive(word: Any) -> bool:
    deprel = str(word.deprel).lower()
    feats = str(getattr(word, "feats", "") or "").lower()
    return "pass" in deprel or "voice=pass" in feats


def stanza_features(nlp: Any, source: str, candidate: str) -> dict[str, float]:
    doc = nlp(candidate)
    words = [word for sentence in doc.sentences for word in sentence.words]
    token_count = max(len(words), 1)
    content_words = [
        word for word in words if word.upos in {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"}
    ]
    content_count = max(len(content_words), 1)
    sentence_count = max(len(doc.sentences), 1)

    clause_dep_count = sum(1 for word in words if word.deprel in CLAUSE_DEPRELS)
    clause_count = max(clause_dep_count, sentence_count)

    depths = [max_dependency_depth(sentence) for sentence in doc.sentences]
    dependency_depth = max(depths) if depths else 0
    mean_dependency_distance, max_dependency_distance, normalized_dependency_distance = dependency_distance_stats(
        doc.sentences
    )

    nouns = sum(1 for word in words if word.upos in {"NOUN", "PROPN"})
    nominal_suffix = sum(
        1
        for word in words
        if word.upos in {"NOUN", "ADJ"} and str(word.text).lower().endswith(NOMINALIZATION_SUFFIXES)
    )
    of_phrases = sum(1 for word in words if str(word.text).lower() == "of")
    nominalization_ratio = (nouns + 0.7 * nominal_suffix + 0.35 * of_phrases) / content_count

    modifier_count = sum(1 for word in words if word.deprel in MODIFIER_DEPRELS)
    modifier_density = modifier_count / token_count

    coordination_count = sum(1 for word in words if word.deprel in {"cc", "conj"}) + sum(
        1 for word in words if str(word.text).lower() in {"and", "or", "but", "nor"}
    )

    passive_count = sum(1 for word in words if is_passive(word))

    sentence_compression_ratio = target_word_count(candidate) / source_char_count(source)

    return {
        "clause_count": float(clause_count),
        "dependency_depth": float(dependency_depth),
        "mean_dependency_distance": round(mean_dependency_distance, 6),
        "max_dependency_distance": round(max_dependency_distance, 6),
        "normalized_dependency_distance": round(normalized_dependency_distance, 6),
        "nominalization_ratio": round(nominalization_ratio, 6),
        "modifier_density": round(modifier_density, 6),
        "coordination_count": float(coordination_count),
        "passive_count": float(passive_count),
        "sentence_compression_ratio": round(sentence_compression_ratio, 6),
    }


def compute_features(
    dataset_rows: list[dict[str, Any]], cache_path: Path
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, float]]]]:
    cache = load_cache(cache_path)
    nlp = stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse", verbose=False, use_gpu=False)
    features_by_id: dict[str, dict[str, dict[str, float]]] = {}
    enriched = []
    total = len(dataset_rows) * len(CANDIDATES)
    done = 0
    for row in dataset_rows:
        sample_id = str(row["id"])
        row_features: dict[str, dict[str, float]] = {}
        for letter in CANDIDATES:
            cache_key = f"{sample_id}:{letter}"
            if cache_key not in cache or any(key not in cache[cache_key] for key in FEATURE_KEYS):
                cache[cache_key] = stanza_features(nlp, str(row["source_text"]), str(row[f"candidate_{letter}"]))
                save_cache(cache_path, cache)
            row_features[letter] = cache[cache_key]
            done += 1
            if done % 50 == 0:
                print(f"parsed {done}/{total} candidate translations")
        features_by_id[sample_id] = row_features
        enriched.append({**row, "syntax_information_features": row_features})
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


def feature_rank_rows(top1_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    lower_is_more = {"passive_count"}
    for key in FEATURE_KEYS:
        metric = f"avg_{key}"
        ordered = sorted(top1_rows, key=lambda row: row[metric], reverse=key not in lower_is_more)
        for rank, row in enumerate(ordered, start=1):
            rows.append({"feature": key, "model": row["model"], "value": row[metric], "rank_for_feature": rank})
    return rows


def main() -> int:
    dataset_rows = read_json(DATASET)
    rankings = load_rankings()
    expected_ids = {str(row["id"]) for row in dataset_rows}
    missing = {model: sorted(expected_ids - set(rows)) for model, rows in rankings.items()}
    missing = {model: ids for model, ids in missing.items() if ids}
    if missing:
        raise ValueError(f"ranking outputs missing ids: {missing}")

    enriched, features_by_id = compute_features(dataset_rows, CACHE_PATH)
    top1_rows = top1_average_rows(rankings, features_by_id)
    preference_rows = pairwise_preference_rows(rankings, features_by_id)
    rank_rows = feature_rank_rows(top1_rows)

    write_json(OUT_JSON_DIR / "ecpcfe_200ce.syntax_info_features.by_candidate.json", enriched)
    write_csv(
        OUT_CSV_DIR / "ecpcfe_200ce.syntax_info_features.model_top1_averages.csv",
        top1_rows,
        ["model", "n", "top_A", "top_B", "top_C"] + [f"avg_{key}" for key in FEATURE_KEYS],
    )
    write_csv(
        OUT_CSV_DIR / "ecpcfe_200ce.syntax_info_features.pairwise_logistic_preferences.csv",
        preference_rows,
        ["model", "feature", "standardized_coefficient", "training_accuracy", "intercept", "n_pairwise_observations"],
    )
    write_csv(
        OUT_CSV_DIR / "ecpcfe_200ce.syntax_info_features.model_feature_ranks.csv",
        rank_rows,
        ["feature", "model", "value", "rank_for_feature"],
    )
    write_json(
        OUT_JSON_DIR / "ecpcfe_200ce.syntax_info_features.method_notes.json",
        [
            {
                "parser": "stanza en tokenize,pos,lemma,depparse",
                "clause_count": "max of dependency clause heads and parser sentence count",
                "dependency_depth": "maximum root-to-token dependency depth across parsed sentences",
                "mean_dependency_distance": "mean absolute token distance between each dependent and its syntactic head, excluding root edges",
                "max_dependency_distance": "maximum absolute token distance between a dependent and its syntactic head",
                "normalized_dependency_distance": "mean_dependency_distance divided by non-root token count, reducing sentence-length effects",
                "nominalization_ratio": "NOUN/PROPN plus nominal suffix and of-phrase proxy per content token",
                "sentence_compression_ratio": "English candidate word count divided by Chinese source character count",
            }
        ],
    )
    print(f"wrote CE syntax/information-structure analysis for {len(dataset_rows)} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
