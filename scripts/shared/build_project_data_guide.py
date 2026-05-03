#!/usr/bin/env python
"""Build a readable file index and data guide for the project."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VERSIONS = {
    "kimi-version": "domestic-generator version: candidates were generated in the Kimi-based run",
    "claude-version": "overseas-generator version: candidates were generated in the Claude-based run",
}


def classify(path: Path) -> dict[str, str]:
    parts = path.parts
    name = path.name
    text = str(path).replace("\\", "/")

    version = parts[0] if parts and parts[0] in VERSIONS else ""
    direction = "EC English-to-Chinese" if "/ec/" in text else "CE Chinese-to-English" if "/ce/" in text else ""
    file_type = path.suffix.lstrip(".").upper()
    category = "other"
    meaning = "Project data or auxiliary file."
    use = "Reference as needed."
    caution = ""

    if "/datasets/" in text:
        category = "dataset"
        if ".with_candidates.shuffled." in name:
            meaning = "Blind-judge dataset with candidate labels shuffled to A/B/C."
            use = "Use as the direct input for judge-model ranking."
        elif ".with_candidates." in name:
            meaning = "Dataset with three candidate translations before blind shuffling."
            use = "Use to inspect candidate texts and candidate source labels."
        elif ".with_variants." in name:
            meaning = "Dataset with generated translation variants."
            use = "Use to audit candidate generation before final candidate packing."
        else:
            meaning = "Base 200-sample dataset with source and reference/human translation."
            use = "Use as the source corpus subset for this direction."
    elif "/rankings/" in text:
        category = "judge ranking output"
        meaning = "Blind ranking result produced by one judge model."
        use = "Use to compute agreement, consensus, and pairwise preference data."
    elif "five_models.pairwise_agreement" in name:
        category = "model agreement"
        meaning = "Pairwise agreement rates among the five judge models."
        use = "Use to support claims about overall judging stability."
    elif "model_vs_consensus_agreement" in name:
        category = "model agreement"
        meaning = "Each judge model's agreement with the consensus ranking."
        use = "Use to compare model-level closeness to group consensus."
    elif "consensus_borda" in name:
        category = "consensus ranking"
        meaning = "Consensus ranking built from Borda aggregation across judge models."
        use = "Use to inspect group-level preferred candidates."
    elif "condorcet" in name:
        category = "consensus ranking"
        meaning = "Condorcet-style pairwise majority summary."
        use = "Use to inspect majority preference cycles or stable winners."
    elif "high_disagreement" in name:
        category = "model disagreement"
        meaning = "Samples with high disagreement among judge models."
        use = "Use for qualitative error analysis."
    elif "minority_model" in name:
        category = "model disagreement"
        meaning = "Cases where one model diverges from the other judge models."
        use = "Use for model-specific difference analysis."
    elif "local_embedding" in name:
        category = "LaBSE semantic similarity"
        meaning = "Cross-lingual semantic similarity computed with local LaBSE embeddings."
        use = "Use as the main semantic similarity evidence, especially for CE."
    elif "crosslingual_nli" in name:
        category = "cross-lingual NLI"
        meaning = "XNLI-style entailment, contradiction, omission, and boundary-risk features."
        use = "Use to analyze semantic completeness and omission/contradiction risk."
    elif "target_lm" in name:
        category = "target-language LM"
        meaning = "Target-language naturalness score from a local language model."
        use = "Use cautiously as an auxiliary fluency/naturalness proxy."
    elif "syntax_info" in name:
        category = "dependency syntax"
        meaning = "Parser-derived syntactic metrics including depth and dependency distance."
        use = "Use to analyze structural load and dependency-span patterns."
    elif "deep_features" in name:
        category = "rule/proxy features"
        meaning = "Rule-based or heuristic control features such as number/entity/register proxies."
        use = "Use as auxiliary controls, not as high-confidence linguistic measurement."
    elif "joint_features.cluster_robust" in name:
        category = "joint logistic with cluster-robust SE"
        meaning = "Joint pairwise logistic regression with standard errors clustered by sample ID."
        use = "Use as the preferred inferential version of the joint regression."
    elif "joint_features.pairwise_logistic" in name:
        category = "joint logistic"
        meaning = "Joint pairwise logistic regression coefficients without cluster-robust SE."
        use = "Use for coefficient direction; prefer cluster-robust file for significance."
    elif "joint_features.pairwise_observations" in name:
        category = "pairwise regression design"
        meaning = "Expanded pairwise observations: three candidate pairs per sample per judge model."
        use = "Use as the source table behind joint logistic regression."
    elif "model_top1_averages" in name:
        category = "top-1 feature average"
        meaning = "Average feature values of the candidates ranked first by each judge model."
        use = "Use to describe what the models tend to select."
    elif "pairwise_logistic_preferences" in name:
        category = "single-feature logistic"
        meaning = "Pairwise logistic preference coefficients for one feature family."
        use = "Use as exploratory feature-family evidence."
    elif "method_notes" in name:
        category = "method notes"
        meaning = "Machine-readable notes describing how the corresponding data were computed."
        use = "Read before interpreting the matching output file."
    elif ".cache." in name or name.endswith(".cache.json"):
        category = "cache"
        meaning = "Intermediate model/parser cache."
        use = "Usually not needed for interpretation; useful for reproducibility/debugging."

    if "sentence_compression_ratio" in name:
        caution = "Despite the old name, interpret as target/source length ratio, not universal compression quality."

    return {
        "version": version,
        "version_meaning": VERSIONS.get(version, ""),
        "direction": direction,
        "format": file_type,
        "category": category,
        "path": str(path).replace("\\", "/"),
        "meaning": meaning,
        "main_use": use,
        "caution": caution,
    }


def build_index() -> list[dict[str, str]]:
    rows = []
    for version in VERSIONS:
        for path in sorted((ROOT / version).rglob("*")):
            if path.suffix.lower() not in {".csv", ".json"}:
                continue
            rows.append(classify(path.relative_to(ROOT)))
    return rows


def write_index(rows: list[dict[str, str]]) -> None:
    out = ROOT / "FILE_INDEX.csv"
    fields = ["version", "version_meaning", "direction", "format", "category", "path", "meaning", "main_use", "caution"]
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_guide(rows: list[dict[str, str]]) -> None:
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["version"], row["category"])
        counts[key] = counts.get(key, 0) + 1

    category_lines = []
    for version in VERSIONS:
        category_lines.append(f"### {version}")
        for (row_version, category), count in sorted(counts.items()):
            if row_version == version:
                category_lines.append(f"- {category}: {count} files")
        category_lines.append("")

    text = f"""# Project Structure and Data Guide

This guide is the readable navigation layer for the project. It does not replace
the original data files. It explains what the directories and repeated file-name
patterns mean, and flags names that can be misleading if read too literally.

## Version Folders

- `kimi-version/`: candidate translations were generated in the domestic-model
  version. Use this as the original candidate-source control condition.
- `claude-version/`: candidate translations were generated in the overseas-model
  version. Use this as the counter-check against candidate-generator self-bias.

Both version folders keep the same structure, so EC/CE and model-level results
can be compared directly.

## Direction Folders

- `ec/`: English-to-Chinese task, based on the FFN finance news subset.
- `ce/`: Chinese-to-English task, based on the ECPCFE finance/economics subset.

## Main Result Families

- `datasets/`: 200-sample task files. `with_candidates.shuffled.json` is the
  direct blind-ranking input because candidate labels have been shuffled.
- `results/model_based_metrics/rankings/json/`: raw ranking outputs from judge
  models. These are the direct observations of LLM-as-a-judge behavior.
- `results/model_based_metrics/analysis/pilot/`: agreement, consensus,
  semantic similarity, NLI, target-language LM, and joint regression outputs.
- `results/parser_derived_syntactic_metrics/analysis/pilot/`: dependency parser
  features, including dependency depth and dependency distance.
- `results/rule_based_proxy_features/analysis/pilot/`: rule/proxy controls such
  as number/entity/register/translationese-style indicators.
- `results/*/analysis/human_reference/`: comparisons between model-selected
  candidates and the available human/reference translation when present.

## Preferred Files for Current Interpretation

- Model agreement:
  `*.five_models.pairwise_agreement.csv`
- Model-vs-consensus comparison:
  `*.five_models.model_vs_consensus_agreement.csv`
- Consensus ranking:
  `*.five_models.consensus_borda_summary.csv`
- LaBSE semantic similarity:
  `*.local_embedding_features.*`
- Cross-lingual NLI:
  `*.crosslingual_nli_features.*`
- Dependency syntax:
  `*.syntax_info_features.*`
- Joint regression, preferred inferential version:
  `*.joint_features.cluster_robust_pairwise_logistic.csv`

The older `*.joint_features.pairwise_logistic_preferences.csv` files still show
the same coefficient directions, but they do not correct the standard errors for
the fact that each sample contributes three pairwise comparisons. Prefer the
cluster-robust version when discussing statistical support.

## Important Naming Clarifications

### `sentence_compression_ratio`

This name is potentially misleading. In the current files it is a target/source
length ratio, not a universal measure of good compression.

- EC: Chinese target character count divided by English source word count.
- CE: English target word count divided by Chinese source character count.

Interpret it as `target_source_length_ratio` or `translation_expansion_ratio`.
In EC, a higher value can mean that the Chinese translation is more explicit or
information-bearing, not necessarily structurally heavier. Do not use this
single variable alone to claim that models prefer more or less compression.

### `deep_features`

This folder name means rule/proxy controls. These features are useful as
controls, but they are not as direct as LaBSE, XNLI, or dependency-parser
features.

### `local_embedding`

These are local LaBSE semantic-similarity features. The name is historical; it
does not mean all local embedding models. In the current project, LaBSE is the
relevant model for interpretation.

## File Counts by Category

{chr(10).join(category_lines)}
## Full Index

See `FILE_INDEX.csv` in the project root. It lists every CSV/JSON file with:

- version
- direction
- format
- category
- path
- plain-language meaning
- suggested use
- caution notes
"""
    (ROOT / "PROJECT_STRUCTURE_AND_DATA_GUIDE.md").write_text(text, encoding="utf-8")


def write_version_guides() -> None:
    for version, meaning in VERSIONS.items():
        text = f"""# Data Guide for {version}

{meaning}

This folder mirrors the other version folder. Use the same relative paths across
`kimi-version/` and `claude-version/` when comparing candidate-source effects.

## Quick Navigation

- EC dataset: `ec/datasets/ffn_200ec.with_candidates.shuffled.json`
- CE dataset: `ce/datasets/ecpcfe_200ce.with_candidates.shuffled.json`
- EC judge rankings: `ec/results/model_based_metrics/rankings/json/`
- CE judge rankings: `ce/results/model_based_metrics/rankings/json/`
- EC analysis CSVs: `ec/results/model_based_metrics/analysis/pilot/csv/`
- CE analysis CSVs: `ce/results/model_based_metrics/analysis/pilot/csv/`
- EC syntax CSVs: `ec/results/parser_derived_syntactic_metrics/analysis/pilot/csv/`
- CE syntax CSVs: `ce/results/parser_derived_syntactic_metrics/analysis/pilot/csv/`
- Rule/proxy controls: `*/results/rule_based_proxy_features/analysis/pilot/`

For detailed file-by-file explanations, open the root-level
`PROJECT_STRUCTURE_AND_DATA_GUIDE.md` and `FILE_INDEX.csv`.
"""
        (ROOT / version / "DATA_GUIDE.md").write_text(text, encoding="utf-8")


def main() -> int:
    rows = build_index()
    write_index(rows)
    write_guide(rows)
    write_version_guides()
    print(f"wrote FILE_INDEX.csv and PROJECT_STRUCTURE_AND_DATA_GUIDE.md for {len(rows)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
