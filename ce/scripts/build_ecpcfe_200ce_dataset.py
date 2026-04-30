#!/usr/bin/env python
"""Build a 200-row CE parallel dataset from untagged ECPCFE files.

The output is a JSON array, following the field style of the existing FFN EC
benchmark while using Chinese source text and English reference translation.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EC_REFERENCE = Path("ec/datasets/ffn_200ec.json")
ECPCFE_ROOT = Path("data/raw/ecpcfe_untagged")
OUTPUT = Path("ce/datasets/ecpcfe_200ce.json")
SEED = 20260430
TARGET_SIZE = 200
DOMAIN_ORDER = [
    "financial_markets",
    "macroeconomics",
    "business_investment_commentary",
    "corporate_earnings",
    "international_trade",
    "banking_financial_institutions",
]

BOOK_DOMAIN = {
    "世界是平的": "international_trade",
    "个人主义和经济秩序": "macroeconomics",
    "人类行为的经济分析": "macroeconomics",
    "价格理论": "macroeconomics",
    "企业 合同 财务结构": "business_investment_commentary",
    "企业 市场 法律": "business_investment_commentary",
    "偏好的经济分析": "macroeconomics",
    "内部流动性与外部流动性": "banking_financial_institutions",
    "动物精神": "financial_markets",
    "宏观经济思想七学派": "macroeconomics",
    "投资者与市场：组合选择、资产定价及投资建议": "financial_markets",
    "行为经济学及应用": "financial_markets",
    "论经济学和经济学家": "business_investment_commentary",
    "货币的非国家化": "banking_financial_institutions",
    "资本主义与自由": "international_trade",
    "集团和组织理论": "corporate_earnings",
}


def read_json(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path}: expected a JSON array")
        return data
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def decode_best(path: Path, prefer_chinese: bool) -> str:
    data = path.read_bytes()
    candidates = ["utf-8-sig", "gb18030", "gbk", "big5"]
    scored = []
    for encoding in candidates:
        text = data.decode(encoding, errors="replace")
        replacements = text.count("\ufffd")
        if prefer_chinese:
            signal = sum("\u4e00" <= char <= "\u9fff" for char in text)
        else:
            signal = sum(("A" <= char <= "Z") or ("a" <= char <= "z") for char in text)
        scored.append((signal - replacements * 20, encoding, text))
    return max(scored, key=lambda item: item[0])[2]


def strip_seg(line: str) -> str:
    line = re.sub(r"^\s*<seg\s+id=\"?\d+\"?\s*>\s*", "", line)
    line = re.sub(r"\s*</seg>\s*$", "", line)
    line = re.sub(r"<[^>]+>", "", line)
    return line.strip()


def normalize_zh(text: str) -> str:
    text = strip_seg(text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"\s+([，。！？；：、）》】])", r"\1", text)
    text = re.sub(r"([（《【])\s+", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_en(text: str) -> str:
    text = strip_seg(text)
    text = text.replace("\ufffd\ufffd", '"').replace("\ufffd", '"')
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def zh_chars(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))


def en_words(text: str) -> int:
    return len(re.findall(r"\b[A-Za-z]+(?:[-'][A-Za-z]+)?\b|\d+(?:\.\d+)?", text))


def infer_difficulty(zh_count: int, en_count: int) -> str:
    if zh_count >= 118 or en_count >= 76:
        return "hard"
    if zh_count >= 72 or en_count >= 48:
        return "medium"
    return "easy"


def base_name(path: Path) -> str:
    name = path.name
    for suffix in [".tmx.zh-CN.txt", ".tmx.zh-TW.txt", ".zh-CN.txt", ".zh-TW.txt"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def english_path_for(zh_path: Path, en_dir: Path) -> Path:
    base = base_name(zh_path)
    if ".tmx." in zh_path.name:
        return en_dir / f"{base}.tmx.en-US.txt"
    return en_dir / f"{base}.en-US.txt"


def target_counts(reference_rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(row[key] for row in reference_rows))


def build_candidates(reference_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    zh_dir = ECPCFE_ROOT / "Nobel-Chinese"
    en_dir = ECPCFE_ROOT / "Nobel-English"
    ec_zh_counts = [int(row["zh_char_count"]) for row in reference_rows]
    ec_en_counts = [int(row["en_word_count"]) for row in reference_rows]
    zh_min, zh_max = min(ec_zh_counts), max(ec_zh_counts)
    en_min, en_max = min(ec_en_counts), max(ec_en_counts)
    candidates = []

    for zh_path in sorted(zh_dir.glob("*.txt")):
        en_path = english_path_for(zh_path, en_dir)
        if not en_path.exists():
            continue
        book = base_name(zh_path)
        domain = BOOK_DOMAIN.get(book, "business_investment_commentary")
        zh_lines = [normalize_zh(line) for line in decode_best(zh_path, True).splitlines()]
        en_lines = [normalize_en(line) for line in decode_best(en_path, False).splitlines()]
        pair_count = min(len(zh_lines), len(en_lines))
        zh_lines = zh_lines[:pair_count]
        en_lines = en_lines[:pair_count]

        for start in range(pair_count):
            for window in range(1, 6):
                end = start + window
                if end > pair_count:
                    break
                source = "".join(zh_lines[start:end]).strip()
                translation = " ".join(en_lines[start:end]).strip()
                zhc = zh_chars(source)
                enw = en_words(translation)
                if zhc < zh_min or zhc > zh_max or enw < en_min or enw > en_max:
                    continue
                if not source or not translation:
                    continue
                if source.count("\ufffd") or translation.count("\ufffd"):
                    continue
                candidates.append(
                    {
                        "book": book,
                        "domain_subtype": domain,
                        "source_text": source,
                        "human_translation": translation,
                        "source_segment_start": start + 1,
                        "source_segment_end": end,
                        "zh_char_count": zhc,
                        "en_word_count": enw,
                        "difficulty": infer_difficulty(zhc, enw),
                    }
                )
    return candidates


def sample_dataset(candidates: list[dict[str, Any]], reference_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    domain_targets = target_counts(reference_rows, "domain_subtype")
    difficulty_targets = target_counts(reference_rows, "difficulty")
    selected: list[dict[str, Any]] = []
    used_keys = set()
    used_segments: dict[str, set[int]] = defaultdict(set)

    joint_targets = allocate_joint_targets(domain_targets, difficulty_targets)
    by_cell = defaultdict(list)
    for item in candidates:
        by_cell[(item["domain_subtype"], item["difficulty"])].append(item)
    for items in by_cell.values():
        rng.shuffle(items)

    def overlap_score(item: dict[str, Any]) -> float:
        overlap = sum(
            segment in used_segments[item["book"]]
            for segment in range(item["source_segment_start"], item["source_segment_end"] + 1)
        )
        return overlap + rng.random() * 0.01

    for domain in DOMAIN_ORDER:
        for difficulty in ["easy", "medium", "hard"]:
            need = joint_targets.get((domain, difficulty), 0)
            pool = list(by_cell.get((domain, difficulty), []))
            pool.sort(key=overlap_score)
            picked = 0
            for item in pool:
                if picked >= need:
                    break
                key = (item["book"], item["source_segment_start"], item["source_segment_end"])
                if key in used_keys:
                    continue
                selected.append(item)
                picked += 1
                used_keys.add(key)
                for segment in range(item["source_segment_start"], item["source_segment_end"] + 1):
                    used_segments[item["book"]].add(segment)
            if picked != need:
                raise RuntimeError(f"only picked {picked}/{need} for {(domain, difficulty)}")

    if len(selected) != TARGET_SIZE:
        raise RuntimeError(f"selected {len(selected)} rows, expected {TARGET_SIZE}")

    rng.shuffle(selected)
    output = []
    for index, item in enumerate(selected, start=1):
        output.append(
            {
                "id": f"ECPCFE_CE_{index:04d}",
                "direction": "CE",
                "source_lang": "Chinese",
                "target_lang": "English",
                "source_text": item["source_text"],
                "human_translation": item["human_translation"],
                "domain_subtype": item["domain_subtype"],
                "difficulty": item["difficulty"],
                "corpus_source": "ECPCFE",
                "source_book": item["book"],
                "source_segment_start": item["source_segment_start"],
                "source_segment_end": item["source_segment_end"],
                "zh_char_count": item["zh_char_count"],
                "en_word_count": item["en_word_count"],
            }
        )
    return output


def allocate_joint_targets(domain_targets: dict[str, int], difficulty_targets: dict[str, int]) -> dict[tuple[str, str], int]:
    total = sum(domain_targets.values())
    difficulties = ["easy", "medium", "hard"]
    targets: dict[tuple[str, str], int] = {}
    fractions = []
    row_remaining = dict(domain_targets)
    col_remaining = dict(difficulty_targets)

    for domain in DOMAIN_ORDER:
        for difficulty in difficulties:
            expected = domain_targets.get(domain, 0) * difficulty_targets.get(difficulty, 0) / total
            base = int(expected)
            targets[(domain, difficulty)] = base
            row_remaining[domain] -= base
            col_remaining[difficulty] -= base
            fractions.append((expected - base, domain, difficulty))

    for _, domain, difficulty in sorted(fractions, reverse=True):
        if row_remaining.get(domain, 0) > 0 and col_remaining.get(difficulty, 0) > 0:
            targets[(domain, difficulty)] += 1
            row_remaining[domain] -= 1
            col_remaining[difficulty] -= 1

    if any(row_remaining.values()) or any(col_remaining.values()):
        for domain in DOMAIN_ORDER:
            while row_remaining.get(domain, 0) > 0:
                difficulty = max(difficulties, key=lambda item: col_remaining.get(item, 0))
                if col_remaining.get(difficulty, 0) <= 0:
                    break
                targets[(domain, difficulty)] += 1
                row_remaining[domain] -= 1
                col_remaining[difficulty] -= 1

    if any(row_remaining.values()) or any(col_remaining.values()):
        raise RuntimeError(f"could not allocate joint targets: rows={row_remaining}, cols={col_remaining}")
    return targets


def main() -> int:
    reference_rows = read_json(EC_REFERENCE)
    candidates = build_candidates(reference_rows)
    output = sample_dataset(candidates, reference_rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"candidate pool: {len(candidates)}")
    print(f"wrote {len(output)} rows to {OUTPUT}")
    print("domain:", dict(Counter(row["domain_subtype"] for row in output)))
    print("difficulty:", dict(Counter(row["difficulty"] for row in output)))
    print("zh range:", min(row["zh_char_count"] for row in output), max(row["zh_char_count"] for row in output))
    print("en range:", min(row["en_word_count"] for row in output), max(row["en_word_count"] for row in output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


