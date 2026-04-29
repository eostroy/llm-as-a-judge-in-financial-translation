#!/usr/bin/env python
"""Analyze completed translation-ranking outputs.

This script intentionally ignores GPT-5.5 and the live DeepSeek run. It uses
the four completed shuffled-position model outputs and writes derived analysis
files without modifying the source dataset or ranking outputs.

The current analysis run skips pairwise logistic regression by request.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any


DATASET = Path("ffn_200ec.with_candidates.shuffled.jsonl")
OUT_DIR = Path("analysis_outputs")
MODEL_FILES = {
    "openai__gpt-5.2": Path("openrouter_rankings/ffn_200ec.with_candidates.shuffled.ranked.openai__gpt-5.2.jsonl"),
    "google__gemini-3-flash-preview": Path(
        "openrouter_rankings/ffn_200ec.with_candidates.shuffled.ranked.google__gemini-3-flash-preview.jsonl"
    ),
    "anthropic__claude-sonnet-4.6": Path(
        "openrouter_rankings/ffn_200ec.with_candidates.shuffled.ranked.anthropic__claude-sonnet-4.6.jsonl"
    ),
    "moonshotai__kimi-k2.5": Path(
        "openrouter_rankings/ffn_200ec.with_candidates.shuffled.ranked.moonshotai__kimi-k2.5.jsonl"
    ),
}
CANDIDATES = ("A", "B", "C")
PAIRS = tuple(combinations(CANDIDATES, 2))

FEATURE_KEYS = [
    "length_chars",
    "length_ratio_to_source_words",
    "number_preservation",
    "entity_preservation",
    "term_coverage",
    "semantic_similarity_proxy",
    "fluency_proxy",
]

TERM_GLOSSARY: dict[str, list[str]] = {
    "bank": ["银行"],
    "banks": ["银行"],
    "central bank": ["央行", "中央银行"],
    "interest rate": ["利率"],
    "inflation": ["通胀", "通货膨胀"],
    "deflation": ["通缩", "通货紧缩"],
    "market": ["市场"],
    "stock": ["股票", "股市"],
    "bond": ["债券"],
    "debt": ["债务", "债"],
    "loan": ["贷款", "借款"],
    "credit": ["信贷", "信用"],
    "mortgage": ["抵押贷款", "按揭"],
    "fund": ["基金"],
    "money market fund": ["货币市场基金"],
    "asset": ["资产"],
    "assets": ["资产"],
    "investment": ["投资"],
    "investor": ["投资者"],
    "investors": ["投资者"],
    "revenue": ["收入", "营收"],
    "profit": ["利润"],
    "earnings": ["盈利", "收益"],
    "growth": ["增长"],
    "recession": ["衰退"],
    "regulation": ["监管"],
    "financial reform": ["金融改革"],
    "oil": ["石油", "原油"],
    "demand": ["需求"],
    "supply": ["供应", "供给"],
    "currency": ["货币"],
    "yuan": ["人民币", "元"],
    "dollar": ["美元"],
    "trade": ["贸易"],
    "tariff": ["关税"],
    "exports": ["出口"],
    "imports": ["进口"],
}

ENTITY_GLOSSARY: dict[str, list[str]] = {
    "China": ["中国"],
    "Chinese": ["中国", "中方"],
    "Beijing": ["北京"],
    "Hebei": ["河北"],
    "Alibaba": ["阿里巴巴"],
    "Alipay": ["支付宝"],
    "China UnionPay": ["中国银联", "银联"],
    "UnionPay": ["银联"],
    "Yu": ["余额宝"],
    "Yu'e Bao": ["余额宝"],
    "Wall Street": ["华尔街"],
    "Clinton": ["克林顿"],
    "Sanders": ["桑德斯"],
    "Warren": ["沃伦"],
    "Dodd-Frank": ["多德-弗兰克"],
    "United States": ["美国"],
    "US": ["美国", "美方"],
    "U.S.": ["美国", "美方"],
    "Europe": ["欧洲"],
    "European": ["欧洲"],
    "Japan": ["日本"],
    "Japanese": ["日本"],
    "Asia-Pacific": ["亚太"],
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def numeric_tokens(text: str) -> list[str]:
    tokens = re.findall(r"\d+(?:\.\d+)?", text)
    return [token.rstrip("0").rstrip(".") if "." in token else token for token in tokens]


def multiset_overlap_ratio(source_items: list[str], target_items: list[str]) -> float:
    if not source_items:
        return 1.0
    source_counts = Counter(source_items)
    target_counts = Counter(target_items)
    matched = sum(min(count, target_counts[item]) for item, count in source_counts.items())
    return matched / sum(source_counts.values())


def extract_entities(source: str) -> list[str]:
    entities = set()
    for pattern in ENTITY_GLOSSARY:
        if re.search(rf"\b{re.escape(pattern)}\b", source, flags=re.IGNORECASE):
            entities.add(pattern)
    for match in re.finditer(r"\b(?:[A-Z][A-Za-z.'-]+|[A-Z]{2,})(?:\s+(?:[A-Z][A-Za-z.'-]+|[A-Z]{2,}))*", source):
        text = match.group(0).strip()
        if text.lower() not in {"the", "a", "an", "as", "but", "in", "on"} and len(text) > 1:
            entities.add(text)
    return sorted(entities)


def coverage_from_glossary(source: str, candidate: str, glossary: dict[str, list[str]]) -> float:
    present_terms = [
        term for term in glossary if re.search(rf"\b{re.escape(term)}\b", source, flags=re.IGNORECASE)
    ]
    if not present_terms:
        return 1.0
    matched = 0
    lower_candidate = candidate.lower()
    for term in present_terms:
        translations = glossary[term]
        if term.lower() in lower_candidate or any(item in candidate for item in translations):
            matched += 1
    return matched / len(present_terms)


def entity_preservation(source: str, candidate: str) -> float:
    entities = extract_entities(source)
    if not entities:
        return 1.0
    matched = 0
    lower_candidate = candidate.lower()
    for entity in entities:
        translations = ENTITY_GLOSSARY.get(entity, [])
        if entity.lower() in lower_candidate or any(item in candidate for item in translations):
            matched += 1
    return matched / len(entities)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def fluency_proxy(candidate: str) -> float:
    if not candidate:
        return 0.0
    length = len(candidate)
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", candidate))
    ascii_letters = len(re.findall(r"[A-Za-z]", candidate))
    mojibake_chars = len(re.findall(r"[�鈥€俓閴鍥藉]", candidate))
    dangling_quote_penalty = 0.08 if candidate.count('"') % 2 else 0.0
    repeated_punct_penalty = min(len(re.findall(r"([，。,.!?！？])\1+", candidate)) * 0.04, 0.2)
    chinese_ratio = chinese_chars / length
    ascii_penalty = min(ascii_letters / length, 0.25)
    mojibake_penalty = min(mojibake_chars / length, 0.5)
    score = 0.75 * chinese_ratio + 0.25 - ascii_penalty - mojibake_penalty - dangling_quote_penalty - repeated_punct_penalty
    return round(clamp(score), 6)


def candidate_features(row: dict[str, Any], letter: str) -> dict[str, float]:
    source = str(row["source_text"])
    candidate = str(row[f"candidate_{letter}"])
    source_words = max(len(re.findall(r"\b\w+\b", source)), 1)
    length_chars = len(candidate)
    length_ratio = length_chars / source_words
    number_score = multiset_overlap_ratio(numeric_tokens(source), numeric_tokens(candidate))
    entity_score = entity_preservation(source, candidate)
    term_score = coverage_from_glossary(source, candidate, TERM_GLOSSARY)
    length_fit = math.exp(-abs(math.log(max(length_ratio, 0.01) / 2.2)))
    fluency = fluency_proxy(candidate)
    semantic_proxy = (
        0.35 * number_score
        + 0.25 * entity_score
        + 0.25 * term_score
        + 0.10 * length_fit
        + 0.05 * fluency
    )
    return {
        "length_chars": float(length_chars),
        "length_ratio_to_source_words": round(length_ratio, 6),
        "number_preservation": round(number_score, 6),
        "entity_preservation": round(entity_score, 6),
        "term_coverage": round(term_score, 6),
        "semantic_similarity_proxy": round(clamp(semantic_proxy), 6),
        "fluency_proxy": fluency,
    }


def winner_from_rank(rank: dict[str, int], a: str, b: str) -> str:
    return a if int(rank[a]) < int(rank[b]) else b


def top_candidate(rank: dict[str, int]) -> str:
    return min(CANDIDATES, key=lambda letter: int(rank[letter]))


def load_rankings() -> dict[str, dict[str, dict[str, int]]]:
    rankings: dict[str, dict[str, dict[str, int]]] = {}
    for model, path in MODEL_FILES.items():
        model_rows = {}
        for row in read_jsonl(path):
            rank = {letter: int(value) for letter, value in row["rank"].items()}
            if sorted(rank) != list(CANDIDATES) or sorted(rank.values()) != [1, 2, 3]:
                raise ValueError(f"{path}: bad rank for id={row.get('id')}: {rank}")
            model_rows[str(row["id"])] = rank
        rankings[model] = model_rows
    return rankings


def build_pairwise(rankings: dict[str, dict[str, dict[str, int]]]) -> list[dict[str, Any]]:
    rows = []
    for model, model_rows in rankings.items():
        for sample_id in sorted(model_rows):
            rank = model_rows[sample_id]
            for a, b in PAIRS:
                rows.append(
                    {
                        "id": sample_id,
                        "model": model,
                        "pair": f"{a}-{b}",
                        "winner": winner_from_rank(rank, a, b),
                        "rank_A": rank["A"],
                        "rank_B": rank["B"],
                        "rank_C": rank["C"],
                    }
                )
    return rows


def pairwise_agreement(pairwise_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, dict[tuple[str, str], str]] = defaultdict(dict)
    for row in pairwise_rows:
        by_model[row["model"]][(row["id"], row["pair"])] = row["winner"]
    rows = []
    for left, right in combinations(sorted(by_model), 2):
        common = sorted(set(by_model[left]) & set(by_model[right]))
        same = sum(1 for key in common if by_model[left][key] == by_model[right][key])
        total = len(common)
        rows.append(
            {
                "model_1": left,
                "model_2": right,
                "same_pairwise_judgments": same,
                "different_pairwise_judgments": total - same,
                "total_pairwise_judgments": total,
                "agreement": round(same / total, 6) if total else "",
            }
        )
    return rows


def enriched_dataset(dataset_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, float]]]]:
    features_by_id: dict[str, dict[str, dict[str, float]]] = {}
    enriched = []
    for row in dataset_rows:
        sample_id = str(row["id"])
        features = {letter: candidate_features(row, letter) for letter in CANDIDATES}
        features_by_id[sample_id] = features
        enriched.append({**row, "candidate_features": features})
    return enriched, features_by_id


def average_feature_rows(
    rankings: dict[str, dict[str, dict[str, int]]],
    features_by_id: dict[str, dict[str, dict[str, float]]],
) -> list[dict[str, Any]]:
    rows = []
    for model, model_rows in sorted(rankings.items()):
        totals = {key: 0.0 for key in FEATURE_KEYS}
        count = 0
        top_counts = Counter()
        for sample_id, rank in model_rows.items():
            winner = top_candidate(rank)
            top_counts[winner] += 1
            feats = features_by_id[sample_id][winner]
            for key in FEATURE_KEYS:
                totals[key] += feats[key]
            count += 1
        row = {
            "model": model,
            "n": count,
            "top_A": top_counts["A"],
            "top_B": top_counts["B"],
            "top_C": top_counts["C"],
        }
        row.update({f"avg_{key}": round(totals[key] / count, 6) for key in FEATURE_KEYS})
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


def fit_logistic_regression(x: list[list[float]], y: list[int], epochs: int = 2500, lr: float = 0.08) -> tuple[float, list[float], float]:
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


def logistic_preference_rows(
    rankings: dict[str, dict[str, dict[str, int]]],
    features_by_id: dict[str, dict[str, dict[str, float]]],
) -> list[dict[str, Any]]:
    rows = []
    for model, model_rows in sorted(rankings.items()):
        x: list[list[float]] = []
        y: list[int] = []
        for sample_id, rank in model_rows.items():
            features = features_by_id[sample_id]
            for a, b in PAIRS:
                x.append([features[a][key] - features[b][key] for key in FEATURE_KEYS])
                y.append(1 if winner_from_rank(rank, a, b) == a else 0)
        bias, weights, accuracy = fit_logistic_regression(x, y)
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


def disagreement_rows(
    dataset_by_id: dict[str, dict[str, Any]],
    rankings: dict[str, dict[str, dict[str, int]]],
) -> list[dict[str, Any]]:
    rows = []
    model_names = sorted(rankings)
    common_ids = sorted(set.intersection(*(set(rankings[model]) for model in model_names)))
    for sample_id in common_ids:
        pair_disagreements = 0
        pair_total = 0
        pair_votes: dict[str, dict[str, int]] = {}
        for a, b in PAIRS:
            votes = Counter(winner_from_rank(rankings[model][sample_id], a, b) for model in model_names)
            pair_votes[f"{a}-{b}"] = dict(votes)
            for left, right in combinations(model_names, 2):
                pair_total += 1
                if winner_from_rank(rankings[left][sample_id], a, b) != winner_from_rank(rankings[right][sample_id], a, b):
                    pair_disagreements += 1
        rank_patterns = Counter(
            json.dumps(rankings[model][sample_id], sort_keys=True, ensure_ascii=False) for model in model_names
        )
        top1_votes = Counter(top_candidate(rankings[model][sample_id]) for model in model_names)
        row = dataset_by_id[sample_id]
        rows.append(
            {
                "id": sample_id,
                "pairwise_disagreement_rate": round(pair_disagreements / pair_total, 6),
                "pairwise_disagreements": pair_disagreements,
                "pairwise_comparisons": pair_total,
                "distinct_rank_patterns": len(rank_patterns),
                "top1_votes": dict(top1_votes),
                "pair_votes": pair_votes,
                "model_ranks": {model: rankings[model][sample_id] for model in model_names},
                "source_text": row["source_text"],
                "candidate_A": row["candidate_A"],
                "candidate_B": row["candidate_B"],
                "candidate_C": row["candidate_C"],
            }
        )
    rows.sort(key=lambda item: (-item["pairwise_disagreement_rate"], -item["distinct_rank_patterns"], item["id"]))
    return rows


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_rows = read_jsonl(DATASET)
    dataset_by_id = {str(row["id"]): row for row in dataset_rows}
    rankings = load_rankings()

    missing = {
        model: [sample_id for sample_id in dataset_by_id if sample_id not in rows]
        for model, rows in rankings.items()
    }
    bad = {model: ids for model, ids in missing.items() if ids}
    if bad:
        raise ValueError(f"completed model outputs are missing ids: {bad}")

    pairwise_rows = build_pairwise(rankings)
    write_jsonl(OUT_DIR / "ffn_200ec.four_models.pairwise_judgments.jsonl", pairwise_rows)

    agreement_rows = pairwise_agreement(pairwise_rows)
    write_csv(
        OUT_DIR / "ffn_200ec.four_models.pairwise_agreement.csv",
        agreement_rows,
        [
            "model_1",
            "model_2",
            "same_pairwise_judgments",
            "different_pairwise_judgments",
            "total_pairwise_judgments",
            "agreement",
        ],
    )

    enriched, features_by_id = enriched_dataset(dataset_rows)
    write_jsonl(OUT_DIR / "ffn_200ec.with_candidates.features.jsonl", enriched)

    top1_rows = average_feature_rows(rankings, features_by_id)
    write_csv(
        OUT_DIR / "ffn_200ec.four_models.top1_average_features.csv",
        top1_rows,
        ["model", "n", "top_A", "top_B", "top_C"] + [f"avg_{key}" for key in FEATURE_KEYS],
    )

    high_disagreement = disagreement_rows(dataset_by_id, rankings)
    write_jsonl(OUT_DIR / "ffn_200ec.four_models.high_disagreement_samples.jsonl", high_disagreement)
    write_csv(
        OUT_DIR / "ffn_200ec.four_models.high_disagreement_summary.csv",
        [
            {
                "id": row["id"],
                "pairwise_disagreement_rate": row["pairwise_disagreement_rate"],
                "pairwise_disagreements": row["pairwise_disagreements"],
                "distinct_rank_patterns": row["distinct_rank_patterns"],
                "top1_votes": json.dumps(row["top1_votes"], ensure_ascii=False, sort_keys=True),
                "model_ranks": json.dumps(row["model_ranks"], ensure_ascii=False, sort_keys=True),
            }
            for row in high_disagreement
        ],
        [
            "id",
            "pairwise_disagreement_rate",
            "pairwise_disagreements",
            "distinct_rank_patterns",
            "top1_votes",
            "model_ranks",
        ],
    )

    print(f"wrote analysis files to {OUT_DIR}")
    print(f"pairwise judgments: {len(pairwise_rows)}")
    print(f"enriched dataset rows: {len(enriched)}")
    print(f"high disagreement samples: {len(high_disagreement)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
