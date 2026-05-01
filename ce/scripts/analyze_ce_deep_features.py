#!/usr/bin/env python
"""Extract lightweight rule-based control features for CE candidate rankings.

The script compares the four completed shuffled CE ranking outputs. It does
not compute semantic similarity or NLI proxies; those are handled by separate
model-based embedding and cross-lingual NLI analyses.
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


DATASET = Path("ce/datasets/ecpcfe_200ce.with_candidates.shuffled.json")
OUT_DIR = Path("ce/results/rule_based_proxy_features/analysis/pilot")
OUT_JSON_DIR = Path("ce/results/rule_based_proxy_features/analysis/pilot/json")
OUT_CSV_DIR = Path("ce/results/rule_based_proxy_features/analysis/pilot/csv")
MODEL_FILES = {
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
}
CANDIDATES = ("A", "B", "C")
PAIRS = tuple(combinations(CANDIDATES, 2))

DEEP_FEATURE_KEYS = [
    "extra_number_count",
    "extra_number_ratio",
    "extra_entity_count",
    "financial_register_score",
    "statistical_register_score",
    "register_score",
    "translationese_score",
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
    "growth": ["增长", "增速"],
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
    "industrial output": ["工业增加值", "工业产出"],
    "retail sales": ["社会消费品零售总额", "零售销售"],
}

ENTITY_GLOSSARY: dict[str, list[str]] = {
    "China": ["中国"],
    "Chinese": ["中国", "中方"],
    "National Bureau of Statistics": ["国家统计局"],
    "Beijing": ["北京"],
    "Hebei": ["河北"],
    "Alibaba": ["阿里巴巴"],
    "Alipay": ["支付宝"],
    "China UnionPay": ["中国银联", "银联"],
    "UnionPay": ["银联"],
    "Yu'e Bao": ["余额宝"],
    "Wall Street": ["华尔街"],
    "Clinton": ["克林顿"],
    "Sanders": ["桑德斯"],
    "Warren": ["沃伦"],
    "Dodd-Frank": ["多德-弗兰克"],
    "United States": ["美国", "美方"],
    "US": ["美国", "美方"],
    "U.S.": ["美国", "美方"],
    "Europe": ["欧洲"],
    "European": ["欧洲"],
    "Japan": ["日本"],
    "Japanese": ["日本"],
    "Asia-Pacific": ["亚太"],
}

FINANCIAL_TERMS = [
    "金融", "银行", "央行", "利率", "通胀", "通货膨胀", "通缩", "债券", "债务", "贷款", "信贷",
    "基金", "资产", "投资", "投资者", "收益", "盈利", "利润", "收入", "营收", "市场", "股市",
    "股票", "汇率", "货币", "美元", "人民币", "贸易", "关税", "出口", "进口", "监管", "改革",
]
STATISTICAL_TERMS = [
    "同比", "环比", "百分点", "增速", "增长", "下降", "上升", "高于", "低于", "收窄", "扩大",
    "达到", "增至", "降至", "数据显示", "统计", "总额", "规模", "占比", "均值", "指数", "趋势",
]
SPECIFICITY_MARKERS = ["即", "也就是", "例如", "包括", "其中", "具体", "尤其", "进一步", "主要", "相关", "所谓"]
TRANSLATIONESE_MARKERS = [
    "进行", "对于", "关于", "方面", "具有", "有关", "其", "该", "之", "所", "被", "基于", "由于",
    "从", "来看", "而言", "相较于", "与其说",
]
NEGATION_EN = ["not", "no", "never", "without", "neither", "nor", "cannot", "can't", "n't"]
NEGATION_ZH = ["不", "未", "无", "没有", "并非", "不能", "不会", "从未", "并不"]
UP_EN = ["increase", "increased", "rises", "rose", "growth", "grew", "higher", "up", "gain", "gains", "expand"]
DOWN_EN = ["decrease", "decreased", "fall", "fell", "lower", "down", "decline", "declined", "drop", "dropped", "contract"]
UP_ZH = ["增长", "上升", "上涨", "提高", "高于", "增加", "扩大", "走高", "增至"]
DOWN_ZH = ["下降", "下跌", "降低", "低于", "减少", "收窄", "放缓", "降至", "回落"]


def read_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected JSON array")
    return data


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def count_terms(text: str, terms: list[str]) -> int:
    return sum(text.count(term) for term in terms)


def numeric_tokens(text: str) -> list[str]:
    raw = re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", text)
    return [token.replace(",", "").rstrip("0").rstrip(".") if "." in token else token.replace(",", "") for token in raw]


def extra_number_count(source: str, candidate: str) -> int:
    source_counts = Counter(numeric_tokens(source))
    extra = 0
    for number, count in Counter(numeric_tokens(candidate)).items():
        extra += max(0, count - source_counts[number])
    return extra


def overlap_ratio(source_items: list[str], target_items: list[str]) -> float:
    if not source_items:
        return 1.0
    source_counts = Counter(source_items)
    target_counts = Counter(target_items)
    matched = sum(min(count, target_counts[item]) for item, count in source_counts.items())
    return matched / sum(source_counts.values())


def source_entities(source: str) -> list[str]:
    entities = set()
    for entity in ENTITY_GLOSSARY:
        if re.search(rf"\b{re.escape(entity)}\b", source, flags=re.IGNORECASE):
            entities.add(entity)
    for match in re.finditer(r"\b(?:[A-Z][A-Za-z.'-]+|[A-Z]{2,})(?:\s+(?:[A-Z][A-Za-z.'-]+|[A-Z]{2,}))*", source):
        entity = match.group(0).strip()
        if entity.lower() not in {"the", "a", "an", "as", "but", "in", "on", "and", "or"} and len(entity) > 1:
            entities.add(entity)
    return sorted(entities)


def entity_preservation(source: str, candidate: str) -> float:
    entities = source_entities(source)
    if not entities:
        return 1.0
    matched = 0
    lower_candidate = candidate.lower()
    for entity in entities:
        translations = ENTITY_GLOSSARY.get(entity, [])
        if entity.lower() in lower_candidate or any(item in candidate for item in translations):
            matched += 1
    return matched / len(entities)


def candidate_named_chunks(candidate: str) -> list[str]:
    suffixes = "公司|银行|集团|基金|交易所|证券|委员会|统计局|央行|政府|法院|部门|协会|市场|平台|机构"
    chunks = set(re.findall(rf"[\u4e00-\u9fffA-Za-z0-9·'’-]{{2,18}}(?:{suffixes})", candidate))
    return sorted(chunk for chunk in chunks if chunk not in {"银行", "基金", "市场", "政府", "机构"})


def extra_entity_count(source: str, candidate: str) -> int:
    allowed = set()
    lower_source = source.lower()
    for entity, translations in ENTITY_GLOSSARY.items():
        if entity.lower() in lower_source:
            allowed.update(translations)
    extra = 0
    for chunk in candidate_named_chunks(candidate):
        if not any(name in chunk or chunk in name for name in allowed):
            extra += 1
    return extra


def glossary_coverage(source: str, candidate: str, glossary: dict[str, list[str]]) -> float:
    present = [term for term in glossary if re.search(rf"\b{re.escape(term)}\b", source, flags=re.IGNORECASE)]
    if not present:
        return 1.0
    lower_candidate = candidate.lower()
    matched = 0
    for term in present:
        if term.lower() in lower_candidate or any(translation in candidate for translation in glossary[term]):
            matched += 1
    return matched / len(present)


def direction_score(source: str, candidate: str) -> float:
    source_lower = source.lower()
    source_up = any(token in source_lower for token in UP_EN)
    source_down = any(token in source_lower for token in DOWN_EN)
    candidate_up = any(token in candidate for token in UP_ZH)
    candidate_down = any(token in candidate for token in DOWN_ZH)
    if not (source_up or source_down):
        return 1.0
    score = 1.0
    if source_up and not candidate_up:
        score -= 0.35
    if source_down and not candidate_down:
        score -= 0.35
    if source_up and candidate_down and not source_down:
        score -= 0.35
    if source_down and candidate_up and not source_up:
        score -= 0.35
    return clamp(score)


def negation_score(source: str, candidate: str) -> float:
    source_has = any(re.search(rf"\b{re.escape(token)}\b", source, flags=re.IGNORECASE) for token in NEGATION_EN)
    candidate_has = any(token in candidate for token in NEGATION_ZH)
    return 1.0 if source_has == candidate_has else 0.65


def sentence_lengths_zh(text: str) -> list[int]:
    pieces = [piece.strip() for piece in re.split(r"[。！？!?；;]", text) if piece.strip()]
    return [len(piece) for piece in pieces] or [len(text)]


def translationese_score(candidate: str) -> float:
    marker_hits = count_terms(candidate, TRANSLATIONESE_MARKERS)
    de_ratio = candidate.count("的") / max(len(re.findall(r"[\u4e00-\u9fff]", candidate)), 1)
    long_sentence_ratio = sum(1 for length in sentence_lengths_zh(candidate) if length > 85) / len(sentence_lengths_zh(candidate))
    passive_or_nominal = count_terms(candidate, ["被", "进行", "具有", "有关", "方面"])
    raw = 0.35 * min(marker_hits / 12, 1.5) + 0.25 * min(de_ratio / 0.08, 1.5)
    raw += 0.20 * long_sentence_ratio + 0.20 * min(passive_or_nominal / 5, 1.5)
    return round(clamp(raw, 0.0, 1.5), 6)


def register_scores(candidate: str) -> tuple[float, float, float]:
    chinese_chars = max(len(re.findall(r"[\u4e00-\u9fff]", candidate)), 1)
    financial = count_terms(candidate, FINANCIAL_TERMS) / chinese_chars * 80
    statistical = count_terms(candidate, STATISTICAL_TERMS) / chinese_chars * 80
    financial_score = clamp(financial)
    statistical_score = clamp(statistical)
    combined = clamp(0.55 * financial_score + 0.45 * statistical_score)
    return round(financial_score, 6), round(statistical_score, 6), round(combined, 6)


def candidate_features(row: dict[str, Any], letter: str) -> dict[str, float]:
    source = str(row["source_text"])
    candidate = str(row[f"candidate_{letter}"])
    source_numbers = max(len(numeric_tokens(source)), 1)
    extra_num = extra_number_count(source, candidate)
    extra_ent = extra_entity_count(source, candidate)
    financial, statistical, register = register_scores(candidate)
    return {
        "extra_number_count": float(extra_num),
        "extra_number_ratio": round(extra_num / source_numbers, 6),
        "extra_entity_count": float(extra_ent),
        "financial_register_score": financial,
        "statistical_register_score": statistical,
        "register_score": register,
        "translationese_score": translationese_score(candidate),
    }


TERM_GLOSSARY = {
    "银行": ["bank", "banking"],
    "中央银行": ["central bank"],
    "央行": ["central bank"],
    "利率": ["interest rate", "rate"],
    "通胀": ["inflation"],
    "通货膨胀": ["inflation"],
    "市场": ["market"],
    "股票": ["stock", "equity"],
    "债券": ["bond"],
    "债务": ["debt"],
    "贷款": ["loan", "lending"],
    "信贷": ["credit"],
    "基金": ["fund"],
    "资产": ["asset", "assets"],
    "投资": ["investment", "invest"],
    "投资者": ["investor", "investors"],
    "收入": ["revenue", "income"],
    "收益": ["return", "gain", "revenue"],
    "利润": ["profit", "profits"],
    "增长": ["growth", "increase"],
    "下降": ["decline", "decrease", "fall"],
    "监管": ["regulation", "regulatory"],
    "需求": ["demand"],
    "供给": ["supply"],
    "货币": ["money", "currency", "monetary"],
    "工资": ["wage", "wages", "pay"],
    "工会": ["union", "labor union"],
    "企业": ["firm", "company", "enterprise"],
    "公司": ["company", "corporation", "firm"],
    "合同": ["contract"],
    "成本": ["cost", "costs"],
    "价格": ["price"],
    "交易": ["transaction", "trade"],
}

ENTITY_GLOSSARY = {
    "中国": ["China", "Chinese"],
    "美国": ["United States", "U.S.", "US", "America", "American"],
    "德国": ["Germany", "German"],
    "日本": ["Japan", "Japanese"],
    "欧洲": ["Europe", "European"],
    "阿里巴巴": ["Alibaba"],
    "支付宝": ["Alipay"],
    "银联": ["UnionPay"],
    "中国银联": ["China UnionPay", "UnionPay"],
    "华尔街": ["Wall Street"],
    "克林顿": ["Clinton"],
    "桑德斯": ["Sanders"],
    "沃伦": ["Warren"],
    "弗里德曼": ["Friedman"],
    "西蒙": ["Simon"],
    "科斯": ["Coase"],
    "IBM": ["IBM"],
    "微软": ["Microsoft"],
    "甲骨文": ["Oracle"],
}

FINANCIAL_TERMS = [
    "financial", "finance", "bank", "banking", "central bank", "interest rate", "inflation", "bond",
    "debt", "loan", "credit", "fund", "asset", "investment", "investor", "revenue", "profit",
    "income", "market", "stock", "currency", "trade", "tariff", "regulation", "wage", "union",
    "firm", "company", "contract", "cost", "price", "transaction",
]
STATISTICAL_TERMS = [
    "percent", "percentage point", "year on year", "month on month", "increase", "decrease",
    "decline", "growth", "higher", "lower", "data", "statistics", "total", "scale", "ratio",
    "average", "index", "trend", "reached", "rose", "fell",
]
SPECIFICITY_MARKERS = [
    "namely", "that is", "in other words", "for example", "including", "especially", "specifically",
    "in particular", "notably", "respectively", "such as",
]
TRANSLATIONESE_MARKERS = [
    "with regard to", "in terms of", "as far as", "it is worth", "it should be", "there is",
    "there are", "carry out", "conduct", "make a", "do not need to", "the former", "the latter",
]
UP_ZH = ["增长", "上升", "上涨", "提高", "高于", "增加", "扩大", "走高", "增至"]
DOWN_ZH = ["下降", "下跌", "降低", "低于", "减少", "收窄", "放缓", "降至", "回落"]
UP_EN = ["increase", "increased", "rises", "rose", "growth", "grew", "higher", "up", "gain", "gains", "expand"]
DOWN_EN = ["decrease", "decreased", "fall", "fell", "lower", "down", "decline", "declined", "drop", "dropped", "contract", "narrow"]


def _contains_source_term(source: str, term: str) -> bool:
    if re.search(r"[\u4e00-\u9fff]", term):
        return term in source
    return bool(re.search(rf"\b{re.escape(term)}\b", source, flags=re.IGNORECASE))


def source_entities(source: str) -> list[str]:
    entities = {entity for entity in ENTITY_GLOSSARY if _contains_source_term(source, entity)}
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*)*", source):
        entities.add(match.group(0).strip())
    return sorted(entities)


def entity_preservation(source: str, candidate: str) -> float:
    entities = source_entities(source)
    if not entities:
        return 1.0
    lower_candidate = candidate.lower()
    matched = 0
    for entity in entities:
        translations = ENTITY_GLOSSARY.get(entity, [entity])
        if any(item.lower() in lower_candidate for item in translations):
            matched += 1
    return matched / len(entities)


def candidate_named_chunks(candidate: str) -> list[str]:
    chunks = set(re.findall(r"\b[A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*)*", candidate))
    stop = {"The", "This", "That", "It", "There", "However", "For", "In", "And", "But"}
    return sorted(chunk for chunk in chunks if chunk.split()[0] not in stop and len(chunk) > 1)


def extra_entity_count(source: str, candidate: str) -> int:
    allowed = set()
    for entity in source_entities(source):
        allowed.update(item.lower() for item in ENTITY_GLOSSARY.get(entity, [entity]))
    extra = 0
    for chunk in candidate_named_chunks(candidate):
        lower_chunk = chunk.lower()
        if not any(name in lower_chunk or lower_chunk in name for name in allowed):
            extra += 1
    return extra


def glossary_coverage(source: str, candidate: str, glossary: dict[str, list[str]]) -> float:
    present = [term for term in glossary if _contains_source_term(source, term)]
    if not present:
        return 1.0
    lower_candidate = candidate.lower()
    matched = 0
    for term in present:
        if any(translation.lower() in lower_candidate for translation in glossary[term]):
            matched += 1
    return matched / len(present)


def direction_score(source: str, candidate: str) -> float:
    source_up = any(token in source for token in UP_ZH)
    source_down = any(token in source for token in DOWN_ZH)
    candidate_lower = candidate.lower()
    candidate_up = any(token in candidate_lower for token in UP_EN)
    candidate_down = any(token in candidate_lower for token in DOWN_EN)
    if not (source_up or source_down):
        return 1.0
    score = 1.0
    if source_up and not candidate_up:
        score -= 0.35
    if source_down and not candidate_down:
        score -= 0.35
    if source_up and candidate_down and not source_down:
        score -= 0.35
    if source_down and candidate_up and not source_up:
        score -= 0.35
    return clamp(score)


def negation_score(source: str, candidate: str) -> float:
    source_has = any(token in source for token in NEGATION_ZH)
    candidate_has = any(re.search(rf"\b{re.escape(token)}\b", candidate, flags=re.IGNORECASE) for token in NEGATION_EN)
    return 1.0 if source_has == candidate_has else 0.65


def sentence_lengths_en(text: str) -> list[int]:
    pieces = [piece.strip() for piece in re.split(r"[.!?;]+", text) if piece.strip()]
    return [len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)?", piece)) for piece in pieces] or [0]


def translationese_score(candidate: str) -> float:
    lower = candidate.lower()
    marker_hits = count_terms(lower, TRANSLATIONESE_MARKERS)
    of_ratio = len(re.findall(r"\bof\b", lower)) / max(len(re.findall(r"[A-Za-z]+", lower)), 1)
    long_sentence_ratio = sum(1 for length in sentence_lengths_en(candidate) if length > 38) / len(sentence_lengths_en(candidate))
    passive_or_nominal = count_terms(lower, ["be", "been", "being", "tion", "ment", "ness", "ity"])
    raw = 0.35 * min(marker_hits / 8, 1.5) + 0.25 * min(of_ratio / 0.08, 1.5)
    raw += 0.20 * long_sentence_ratio + 0.20 * min(passive_or_nominal / 14, 1.5)
    return round(clamp(raw, 0.0, 1.5), 6)


def register_scores(candidate: str) -> tuple[float, float, float]:
    words = max(len(re.findall(r"[A-Za-z]+", candidate)), 1)
    lower = candidate.lower()
    financial = count_terms(lower, FINANCIAL_TERMS) / words * 35
    statistical = count_terms(lower, STATISTICAL_TERMS) / words * 35
    financial_score = clamp(financial)
    statistical_score = clamp(statistical)
    combined = clamp(0.55 * financial_score + 0.45 * statistical_score)
    return round(financial_score, 6), round(statistical_score, 6), round(combined, 6)


def top_candidate(rank: dict[str, int]) -> str:
    return min(CANDIDATES, key=lambda letter: int(rank[letter]))


def winner_from_rank(rank: dict[str, int], a: str, b: str) -> str:
    return a if int(rank[a]) < int(rank[b]) else b


def load_rankings() -> dict[str, dict[str, dict[str, int]]]:
    rankings: dict[str, dict[str, dict[str, int]]] = {}
    for model, path in MODEL_FILES.items():
        rows = read_json(path)
        rankings[model] = {str(row["id"]): {k: int(v) for k, v in row["rank"].items()} for row in rows}
    return rankings


def enrich_dataset(dataset_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, float]]]]:
    features_by_id = {}
    enriched = []
    for row in dataset_rows:
        sample_id = str(row["id"])
        features = {letter: candidate_features(row, letter) for letter in CANDIDATES}
        features_by_id[sample_id] = features
        enriched.append({**row, "deep_candidate_features": features})
    return enriched, features_by_id


def top1_average_rows(
    rankings: dict[str, dict[str, dict[str, int]]],
    features_by_id: dict[str, dict[str, dict[str, float]]],
) -> list[dict[str, Any]]:
    rows = []
    for model, model_rows in sorted(rankings.items()):
        totals = {key: 0.0 for key in DEEP_FEATURE_KEYS}
        top_counts = Counter()
        for sample_id, rank in model_rows.items():
            top = top_candidate(rank)
            top_counts[top] += 1
            for key in DEEP_FEATURE_KEYS:
                totals[key] += features_by_id[sample_id][top][key]
        n = len(model_rows)
        row = {"model": model, "n": n, "top_A": top_counts["A"], "top_B": top_counts["B"], "top_C": top_counts["C"]}
        row.update({f"avg_{key}": round(totals[key] / n, 6) for key in DEEP_FEATURE_KEYS})
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


def fit_logistic(x: list[list[float]], y: list[int], epochs: int = 2400, lr: float = 0.07) -> tuple[float, list[float], float]:
    x_scaled = standardize_matrix(x)
    weights = [0.0 for _ in x_scaled[0]]
    bias = 0.0
    l2 = 0.015
    n = len(y)
    for _ in range(epochs):
        grad_w = [0.0 for _ in weights]
        grad_b = 0.0
        for row, label in zip(x_scaled, y):
            z = max(-35.0, min(35.0, bias + sum(w * v for w, v in zip(weights, row))))
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
        z = bias + sum(w * v for w, v in zip(weights, row))
        correct += (1 if z >= 0 else 0) == label
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
                x.append([features[a][key] - features[b][key] for key in DEEP_FEATURE_KEYS])
                y.append(1 if winner_from_rank(rank, a, b) == a else 0)
        bias, weights, accuracy = fit_logistic(x, y)
        for key, weight in zip(DEEP_FEATURE_KEYS, weights):
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
    lower_is_more = {"extra_number_count", "extra_number_ratio", "extra_entity_count", "translationese_score"}
    for key in DEEP_FEATURE_KEYS:
        metric = f"avg_{key}"
        ordered = sorted(top1_rows, key=lambda row: row[metric], reverse=key not in lower_is_more)
        for rank, row in enumerate(ordered, start=1):
            rows.append({"feature": key, "model": row["model"], "value": row[metric], "rank_for_feature": rank})
    return rows


def top1_detail_rows(
    dataset_rows: list[dict[str, Any]],
    rankings: dict[str, dict[str, dict[str, int]]],
    features_by_id: dict[str, dict[str, dict[str, float]]],
) -> list[dict[str, Any]]:
    dataset_by_id = {str(row["id"]): row for row in dataset_rows}
    rows = []
    for sample_id in sorted(dataset_by_id):
        item: dict[str, Any] = {"id": sample_id}
        for model, model_rows in sorted(rankings.items()):
            top = top_candidate(model_rows[sample_id])
            item[f"{model}__top1"] = top
            for key in DEEP_FEATURE_KEYS:
                item[f"{model}__{key}"] = features_by_id[sample_id][top][key]
        rows.append(item)
    return rows


def main() -> int:
    OUT_JSON_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CSV_DIR.mkdir(parents=True, exist_ok=True)
    dataset_rows = read_json(DATASET)
    rankings = load_rankings()
    expected_ids = {str(row["id"]) for row in dataset_rows}
    missing = {model: sorted(expected_ids - set(rows)) for model, rows in rankings.items()}
    missing = {model: ids for model, ids in missing.items() if ids}
    if missing:
        raise ValueError(f"ranking outputs missing ids: {missing}")

    enriched, features_by_id = enrich_dataset(dataset_rows)
    top1_rows = top1_average_rows(rankings, features_by_id)
    preference_rows = pairwise_preference_rows(rankings, features_by_id)
    rank_rows = feature_rank_rows(top1_rows)
    detail_rows = top1_detail_rows(dataset_rows, rankings, features_by_id)

    write_json(OUT_JSON_DIR / "ecpcfe_200ce.deep_features.by_candidate.json", enriched)
    write_json(OUT_JSON_DIR / "ecpcfe_200ce.deep_features.model_top1_by_sample.json", detail_rows)
    write_json(
        OUT_JSON_DIR / "ecpcfe_200ce.deep_features.method_notes.json",
        [
            {
                "note": "Rule-based lightweight controls only. Composite semantic_similarity_proxy, nli_consistency_proxy, and specificity_expansion were removed in favor of model-based embedding/NLI analyses.",
                "dataset": str(DATASET),
                "models": sorted(MODEL_FILES),
            }
        ],
    )
    write_csv(
        OUT_CSV_DIR / "ecpcfe_200ce.deep_features.model_top1_averages.csv",
        top1_rows,
        ["model", "n", "top_A", "top_B", "top_C"] + [f"avg_{key}" for key in DEEP_FEATURE_KEYS],
    )
    write_csv(
        OUT_CSV_DIR / "ecpcfe_200ce.deep_features.pairwise_logistic_preferences.csv",
        preference_rows,
        ["model", "feature", "standardized_coefficient", "training_accuracy", "intercept", "n_pairwise_observations"],
    )
    write_csv(
        OUT_CSV_DIR / "ecpcfe_200ce.deep_features.model_feature_ranks.csv",
        rank_rows,
        ["feature", "model", "value", "rank_for_feature"],
    )
    print(f"wrote deep feature analysis for {len(dataset_rows)} samples and {len(rankings)} models to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




