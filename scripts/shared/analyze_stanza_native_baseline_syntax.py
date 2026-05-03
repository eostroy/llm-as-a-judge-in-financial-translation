#!/usr/bin/env python
"""Run the same Stanza syntax features on native baselines and candidates.

This supersedes rule-based syntax proxies for native-baseline comparison. The
feature definitions mirror the existing EC/CE `analyze_*_syntax_info_features.py`
scripts and add parser-derived `token_count`, `sentence_count`, and
`avg_sentence_length` for direct native-vs-candidate comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from itertools import combinations
from pathlib import Path
from typing import Any

import stanza


ROOT = Path(__file__).resolve().parents[2]
BASELINE_ROOT = ROOT / "data" / "external_baselines"
BASELINES = {
    "zh": BASELINE_ROOT / "zh_native_finance_kenpache" / "clean.txt",
    "en": BASELINE_ROOT / "en_native_finance_kenpache" / "clean.txt",
}
VERSIONS = ("kimi-version", "claude-version")
DIRECTIONS = {
    "ec": {"prefix": "ffn_200ec", "language": "zh", "source_language": "en"},
    "ce": {"prefix": "ecpcfe_200ce", "language": "en", "source_language": "zh"},
}
CANDIDATES = ("A", "B", "C")
PAIRS = tuple(combinations(CANDIDATES, 2))

COMMON_EXTRA_KEYS = ["token_count", "sentence_count", "avg_sentence_length"]
EC_FEATURE_KEYS = [
    "clause_count",
    "dependency_depth",
    "mean_dependency_distance",
    "max_dependency_distance",
    "normalized_dependency_distance",
    "nominalization_ratio",
    "modifier_density",
    "coordination_count",
    "passive_or_beishi_count",
    "sentence_compression_ratio",
    *COMMON_EXTRA_KEYS,
]
CE_FEATURE_KEYS = [
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
    *COMMON_EXTRA_KEYS,
]

EC_CLAUSE_DEPRELS = {"root", "ccomp", "xcomp", "acl", "advcl", "parataxis", "conj"}
EC_MODIFIER_DEPRELS = {"amod", "advmod", "nmod", "acl", "det", "clf", "mark", "case"}
EC_COORDINATION_WORDS = {"和", "及", "以及", "并", "并且", "或", "或者", "与", "、"}
EC_PASSIVE_WORDS = {"被", "由", "为", "受到", "遭到", "经", "让", "给"}
EC_NOMINALIZATION_SUFFIXES = ("性", "化", "度", "率", "者", "方", "额", "量", "值", "力")
EC_PUNCT_CLAUSE_SPLIT = re.compile(r"[。！？!?；;]")

CE_CLAUSE_DEPRELS = {"root", "ccomp", "xcomp", "acl", "advcl", "parataxis", "conj", "relcl"}
CE_MODIFIER_DEPRELS = {"amod", "advmod", "nmod", "acl", "compound", "det", "case", "obl", "appos"}
CE_NOMINALIZATION_SUFFIXES = (
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


def read_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected JSON array")
    return data


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_cache(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def save_cache(path: Path, cache: dict[str, dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False) + "\n", encoding="utf-8")


def cjk_count(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))


def repair_mojibake(text: str, expected_language: str) -> str:
    candidates = [text]
    for encoding in ("latin1", "cp1252", "gbk"):
        try:
            candidates.append(text.encode(encoding).decode("utf-8"))
        except UnicodeError:
            pass
    if expected_language == "zh":
        return max(candidates, key=lambda value: (cjk_count(value), -value.count("\ufffd")))
    return max(candidates, key=lambda value: (-value.count("\ufffd"), len(value)))


def normalize_text(text: str, language: str) -> str:
    return re.sub(r"\s+", " ", repair_mojibake(text, language).replace("\u00a0", " ")).strip()


def source_word_count(source: str) -> int:
    return max(len(re.findall(r"\b\w+\b", source)), 1)


def source_char_count(source: str) -> int:
    return max(cjk_count(source), 1)


def target_word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)?", text))


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


def ec_doc_features(doc: Any, source: str, candidate: str) -> dict[str, float]:
    words = [word for sentence in doc.sentences for word in sentence.words]
    token_count = max(len(words), 1)
    content_count = max(sum(1 for word in words if word.upos not in {"PUNCT", "PART", "SYM"}), 1)
    sentence_count = max(len(doc.sentences), 1)

    clause_dep_count = sum(1 for word in words if word.deprel in EC_CLAUSE_DEPRELS)
    punct_clause_count = len([piece for piece in EC_PUNCT_CLAUSE_SPLIT.split(candidate) if piece.strip()])
    clause_count = max(clause_dep_count, punct_clause_count, sentence_count)

    depths = [max_dependency_depth(sentence) for sentence in doc.sentences]
    dependency_depth = max(depths) if depths else 0
    mean_dependency_distance, max_dependency_distance, normalized_dependency_distance = dependency_distance_stats(
        doc.sentences
    )

    nominal_pos = sum(1 for word in words if word.upos in {"NOUN", "PROPN", "PRON"})
    nominal_suffix = sum(1 for word in words if str(word.text).endswith(EC_NOMINALIZATION_SUFFIXES))
    de_nominal = candidate.count("的") + candidate.count("之")
    nominalization_ratio = (nominal_pos + 0.5 * nominal_suffix + 0.35 * de_nominal) / content_count

    modifier_count = sum(1 for word in words if word.deprel in EC_MODIFIER_DEPRELS)
    coordination_count = sum(1 for word in words if word.deprel in {"cc", "conj"}) + sum(
        candidate.count(word) for word in EC_COORDINATION_WORDS
    )
    passive_or_beishi_count = sum(candidate.count(word) for word in EC_PASSIVE_WORDS) + sum(
        1 for word in words if "pass" in str(word.deprel).lower()
    )

    return {
        "clause_count": float(clause_count),
        "dependency_depth": float(dependency_depth),
        "mean_dependency_distance": round(mean_dependency_distance, 6),
        "max_dependency_distance": round(max_dependency_distance, 6),
        "normalized_dependency_distance": round(normalized_dependency_distance, 6),
        "nominalization_ratio": round(nominalization_ratio, 6),
        "modifier_density": round(modifier_count / token_count, 6),
        "coordination_count": float(coordination_count),
        "passive_or_beishi_count": float(passive_or_beishi_count),
        "sentence_compression_ratio": round(cjk_count(candidate) / source_word_count(source), 6),
        "token_count": float(token_count),
        "sentence_count": float(sentence_count),
        "avg_sentence_length": round(token_count / sentence_count, 6),
    }


def ce_doc_features(doc: Any, source: str, candidate: str) -> dict[str, float]:
    words = [word for sentence in doc.sentences for word in sentence.words]
    token_count = max(len(words), 1)
    content_words = [word for word in words if word.upos in {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"}]
    content_count = max(len(content_words), 1)
    sentence_count = max(len(doc.sentences), 1)

    clause_dep_count = sum(1 for word in words if word.deprel in CE_CLAUSE_DEPRELS)
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
        if word.upos in {"NOUN", "ADJ"} and str(word.text).lower().endswith(CE_NOMINALIZATION_SUFFIXES)
    )
    of_phrases = sum(1 for word in words if str(word.text).lower() == "of")
    nominalization_ratio = (nouns + 0.7 * nominal_suffix + 0.35 * of_phrases) / content_count

    modifier_count = sum(1 for word in words if word.deprel in CE_MODIFIER_DEPRELS)
    coordination_count = sum(1 for word in words if word.deprel in {"cc", "conj"}) + sum(
        1 for word in words if str(word.text).lower() in {"and", "or", "but", "nor"}
    )
    passive_count = sum(1 for word in words if is_passive(word))

    return {
        "clause_count": float(clause_count),
        "dependency_depth": float(dependency_depth),
        "mean_dependency_distance": round(mean_dependency_distance, 6),
        "max_dependency_distance": round(max_dependency_distance, 6),
        "normalized_dependency_distance": round(normalized_dependency_distance, 6),
        "nominalization_ratio": round(nominalization_ratio, 6),
        "modifier_density": round(modifier_count / token_count, 6),
        "coordination_count": float(coordination_count),
        "passive_count": float(passive_count),
        "sentence_compression_ratio": round(target_word_count(candidate) / source_char_count(source), 6),
        "token_count": float(token_count),
        "sentence_count": float(sentence_count),
        "avg_sentence_length": round(token_count / sentence_count, 6),
    }


def feature_keys(language: str) -> list[str]:
    return EC_FEATURE_KEYS if language == "zh" else CE_FEATURE_KEYS


def stanza_features(nlp: Any, language: str, source: str, candidate: str) -> dict[str, float]:
    if language == "zh":
        candidate = normalize_text(candidate, "zh")
        source = normalize_text(source, "en")
        return ec_doc_features(nlp(candidate), source, candidate)
    candidate = normalize_text(candidate, "en")
    source = normalize_text(source, "zh")
    return ce_doc_features(nlp(candidate), source, candidate)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if not values:
        return 1.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / len(values)) or 1.0


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def summarize_feature(values: list[float]) -> dict[str, float]:
    return {
        "n": float(len(values)),
        "mean": round(mean(values), 9),
        "std": round(std(values), 9),
        "p05": round(quantile(values, 0.05), 9),
        "median": round(quantile(values, 0.5), 9),
        "p95": round(quantile(values, 0.95), 9),
        "min": round(min(values), 9),
        "max": round(max(values), 9),
    }


def z_score(value: float, stats: dict[str, float]) -> float:
    return round((value - stats["mean"]) / (stats["std"] or 1.0), 9)


def load_baseline_rows(language: str, limit: int | None) -> list[tuple[str, str]]:
    texts = [line.strip() for line in BASELINES[language].read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit:
        texts = texts[:limit]
    source_placeholder = "native baseline source placeholder"
    if language == "en":
        source_placeholder = "原生基准源占位"
    return [(f"{language}_native_{idx + 1:05d}", text) for idx, text in enumerate(texts)]


def parse_baseline_language(
    language: str, limit: int | None, batch_size: int
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    cache_path = BASELINE_ROOT / f"stanza_{language}_native_syntax.cache.json"
    cache = load_cache(cache_path)
    nlp = stanza.Pipeline(language, processors="tokenize,pos,lemma,depparse", verbose=False, use_gpu=False)
    rows: list[dict[str, Any]] = []
    baseline_rows = load_baseline_rows(language, limit)
    source = "native baseline source placeholder"
    if language == "en":
        source = "原生基准源占位"
    missing = [
        (text_id, text)
        for text_id, text in baseline_rows
        if text_id not in cache or any(key not in cache[text_id] for key in feature_keys(language))
    ]
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        if not batch:
            continue
        texts = [normalize_text(text, language) for _, text in batch]
        docs = nlp.bulk_process(texts)
        for (text_id, text), normalized_text, doc in zip(batch, texts, docs):
            if language == "zh":
                cache[text_id] = ec_doc_features(doc, source, normalized_text)
            else:
                cache[text_id] = ce_doc_features(doc, source, normalized_text)
        save_cache(cache_path, cache)
        print(f"parsed {language} native baseline {min(start + len(batch), len(missing))}/{len(missing)} missing")

    for text_id, _ in baseline_rows:
        rows.append({"id": text_id, "language": language, **cache[text_id]})

    stats: dict[str, dict[str, float]] = {}
    for key in feature_keys(language):
        stats[key] = summarize_feature([float(row[key]) for row in rows])
    return rows, stats


def top_candidate(rank: dict[str, int]) -> str:
    return min(CANDIDATES, key=lambda letter: int(rank[letter]))


def load_rankings(version_dir: Path, direction: str, prefix: str) -> dict[str, dict[str, dict[str, int]]]:
    ranking_dir = version_dir / direction / "results" / "model_based_metrics" / "rankings" / "json"
    rankings: dict[str, dict[str, dict[str, int]]] = {}
    for path in sorted(ranking_dir.glob(f"{prefix}.with_candidates.shuffled.ranked.*.json")):
        model = path.name.removeprefix(f"{prefix}.with_candidates.shuffled.ranked.").removesuffix(".json")
        rows = read_json(path)
        rankings[model] = {
            str(row["id"]): {letter: int(value) for letter, value in row["rank"].items()}
            for row in rows
            if "rank" in row
        }
    return rankings


def parse_candidates_if_needed(
    version_dir: Path,
    direction: str,
    prefix: str,
    language: str,
    force_reparse: bool,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, float]]]]:
    result_root = version_dir / direction / "results" / "parser_derived_syntactic_metrics" / "analysis" / "pilot"
    input_path = result_root / "json" / f"{prefix}.syntax_info_features.by_candidate.json"
    rows = read_json(input_path)
    features_by_id: dict[str, dict[str, dict[str, float]]] = {}

    if not force_reparse:
        for row in rows:
            features_by_id[str(row["id"])] = row["syntax_information_features"]
        return rows, features_by_id

    dataset_path = version_dir / direction / "datasets" / f"{prefix}.with_candidates.shuffled.json"
    dataset_rows = read_json(dataset_path)
    cache_path = result_root / "json" / f"{prefix}.stanza_native_baseline_candidate_features.cache.json"
    cache = load_cache(cache_path)
    nlp = stanza.Pipeline(language, processors="tokenize,pos,lemma,depparse", verbose=False, use_gpu=False)
    enriched: list[dict[str, Any]] = []
    total = len(dataset_rows) * len(CANDIDATES)
    tasks: list[tuple[str, str, str, str]] = []
    for row in dataset_rows:
        sample_id = str(row["id"])
        for letter in CANDIDATES:
            key = f"{sample_id}:{letter}"
            if force_reparse or key not in cache or any(name not in cache[key] for name in feature_keys(language)):
                tasks.append((key, str(row["source_text"]), str(row[f"candidate_{letter}"]), letter))

    batch_size = 64
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start : start + batch_size]
        texts = [normalize_text(candidate, language) for _, _, candidate, _ in batch]
        docs = nlp.bulk_process(texts)
        for (key, source, _candidate, _letter), text, doc in zip(batch, texts, docs):
            if language == "zh":
                cache[key] = ec_doc_features(doc, normalize_text(source, "en"), text)
            else:
                cache[key] = ce_doc_features(doc, normalize_text(source, "zh"), text)
        save_cache(cache_path, cache)
        print(f"parsed {version_dir.name}/{direction} candidates {min(start + len(batch), len(tasks))}/{len(tasks)} missing")

    done = 0
    for row in dataset_rows:
        sample_id = str(row["id"])
        row_features: dict[str, dict[str, float]] = {}
        for letter in CANDIDATES:
            key = f"{sample_id}:{letter}"
            row_features[letter] = cache[key]
            done += 1
            if done % 100 == 0:
                print(f"loaded {version_dir.name}/{direction} candidates {done}/{total}")
        features_by_id[sample_id] = row_features
        enriched.append({**row, "syntax_information_features": row_features})
    return enriched, features_by_id


def summarize_candidate_rows(rows: list[dict[str, Any]], group_keys: list[str], keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    out: list[dict[str, Any]] = []
    for group_key, group in sorted(grouped.items()):
        result = dict(zip(group_keys, group_key))
        result["n"] = len(group)
        for key in keys:
            values = [float(row[key]) for row in group]
            result[f"{key}_mean"] = round(mean(values), 9)
            result[f"{key}_std"] = round(std(values), 9)
            z_key = f"{key}_native_z"
            if z_key in group[0]:
                result[f"{key}_native_z_mean"] = round(mean([float(row[z_key]) for row in group]), 9)
        out.append(result)
    return out


def analyze_version_direction(
    version_dir: Path,
    direction: str,
    cfg: dict[str, str],
    baseline_stats: dict[str, dict[str, float]],
    force_reparse: bool,
) -> None:
    prefix = cfg["prefix"]
    language = cfg["language"]
    result_root = version_dir / direction / "results" / "parser_derived_syntactic_metrics" / "analysis" / "pilot"
    enriched_rows, features_by_id = parse_candidates_if_needed(version_dir, direction, prefix, language, force_reparse)
    first_sample = next(iter(features_by_id.values()))
    first_features = next(iter(first_sample.values()))
    keys = [key for key in feature_keys(language) if key in first_features and key in baseline_stats]

    flat_rows: list[dict[str, Any]] = []
    output_json = []
    for row in enriched_rows:
        sample_id = str(row["id"])
        normalized_by_candidate: dict[str, dict[str, float]] = {}
        for letter in CANDIDATES:
            feats = features_by_id[sample_id][letter]
            normalized = dict(feats)
            for key in keys:
                normalized[f"{key}_native_z"] = z_score(float(feats[key]), baseline_stats[key])
            normalized_by_candidate[letter] = normalized
            flat_rows.append(
                {
                    "version": version_dir.name,
                    "direction": direction.upper(),
                    "id": sample_id,
                    "candidate": letter,
                    **normalized,
                }
            )
        output_json.append({**row, "stanza_native_baseline_syntax_features": normalized_by_candidate})

    family = "stanza_native_baseline_syntax"
    write_json(result_root / "json" / f"{prefix}.{family}.by_candidate.json", output_json)
    write_csv(result_root / "csv" / f"{prefix}.{family}.candidate_features.csv", flat_rows)
    write_csv(
        result_root / "csv" / f"{prefix}.{family}.candidate_vs_baseline_summary.csv",
        summarize_candidate_rows(flat_rows, ["version", "direction"], keys),
    )

    rankings = load_rankings(version_dir, direction, prefix)
    top_rows: list[dict[str, Any]] = []
    for model, model_rows in rankings.items():
        for sample_id, rank in model_rows.items():
            top = top_candidate(rank)
            top_rows.append(
                {
                    "version": version_dir.name,
                    "direction": direction.upper(),
                    "model": model,
                    "id": sample_id,
                    "top_candidate": top,
                    **flat_row_features(features_by_id[sample_id][top], baseline_stats, keys),
                }
            )
    if top_rows:
        write_csv(result_root / "csv" / f"{prefix}.{family}.model_top1_features.csv", top_rows)
        write_csv(
            result_root / "csv" / f"{prefix}.{family}.model_top1_vs_baseline_summary.csv",
            summarize_candidate_rows(top_rows, ["version", "direction", "model"], keys),
        )


def flat_row_features(
    feats: dict[str, float], baseline_stats: dict[str, dict[str, float]], keys: list[str]
) -> dict[str, float]:
    output = dict(feats)
    for key in keys:
        output[f"{key}_native_z"] = z_score(float(feats[key]), baseline_stats[key])
    return output


def write_overview(baseline_stats: dict[str, dict[str, dict[str, float]]]) -> None:
    rows: list[dict[str, Any]] = []
    summary_files = []
    for version in VERSIONS:
        for direction, cfg in DIRECTIONS.items():
            prefix = cfg["prefix"]
            path = (
                ROOT
                / version
                / direction
                / "results"
                / "parser_derived_syntactic_metrics"
                / "analysis"
                / "pilot"
                / "csv"
                / f"{prefix}.stanza_native_baseline_syntax.candidate_vs_baseline_summary.csv"
            )
            if path.exists():
                summary_files.append((cfg["language"], path))
    wanted = [
        "clause_count",
        "dependency_depth",
        "mean_dependency_distance",
        "normalized_dependency_distance",
        "nominalization_ratio",
        "modifier_density",
        "coordination_count",
        "passive_or_beishi_count",
        "passive_count",
        "avg_sentence_length",
    ]
    for language, path in summary_files:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            row = next(csv.DictReader(handle))
        out = {
            "version": row["version"],
            "direction": row["direction"],
            "target_language": language,
            "n": row["n"],
        }
        for key in wanted:
            z_key = f"{key}_native_z_mean"
            if z_key in row:
                out[z_key] = row[z_key]
        rows.append(out)
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        leading = ["version", "direction", "target_language", "n"]
        fieldnames = leading + [key for key in fieldnames if key not in leading]
        write_csv(BASELINE_ROOT / "stanza_native_syntax_candidate_comparison_overview.csv", rows, fieldnames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-limit", type=int, default=0, help="0 means all saved native baseline texts.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--force-reparse-candidates", action="store_true")
    args = parser.parse_args()
    limit = args.baseline_limit or None

    all_baseline_stats: dict[str, dict[str, dict[str, float]]] = {}
    baseline_summary_rows: list[dict[str, Any]] = []
    for language in ("zh", "en"):
        baseline_rows, stats = parse_baseline_language(language, limit, args.batch_size)
        all_baseline_stats[language] = stats
        write_json(BASELINE_ROOT / f"stanza_{language}_native_syntax.by_text.json", baseline_rows)
        write_csv(BASELINE_ROOT / f"stanza_{language}_native_syntax.by_text.csv", baseline_rows)
        for key, summary in stats.items():
            baseline_summary_rows.append({"language": language, "feature": key, **summary})

    write_json(BASELINE_ROOT / "stanza_native_syntax_baseline_stats.json", all_baseline_stats)
    write_csv(BASELINE_ROOT / "stanza_native_syntax_baseline_stats.csv", baseline_summary_rows)

    for version in VERSIONS:
        version_dir = ROOT / version
        if not version_dir.exists():
            continue
        for direction, cfg in DIRECTIONS.items():
            analyze_version_direction(
                version_dir,
                direction,
                cfg,
                all_baseline_stats[cfg["language"]],
                force_reparse=args.force_reparse_candidates,
            )
    write_overview(all_baseline_stats)
    print("Wrote Stanza native syntax baselines and candidate comparisons.")


if __name__ == "__main__":
    main()
