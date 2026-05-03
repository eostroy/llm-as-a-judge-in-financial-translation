#!/usr/bin/env python
"""Build review-ready summaries and an organized analysis-output package.

The package is a curated copy under data/organized_analysis_outputs. It does not
move or delete the original result files used by existing scripts.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(".")
PACKAGE_DIR = Path("data/organized_analysis_outputs")
SUMMARY_DIR = Path("data/analysis_summaries")
CANDIDATES = ("A", "B", "C")

VERSIONS = {
    "kimi-version": {
        "generator_model": "moonshotai__kimi-k2.5",
        "generator_slug": "kimi_k2_5",
    },
    "claude-version": {
        "generator_model": "anthropic__claude-sonnet-4.6",
        "generator_slug": "claude_sonnet_4_6",
    },
}

DIRECTIONS = {
    "ec": "ffn_200ec",
    "ce": "ecpcfe_200ce",
}

MODEL_ORDER = [
    "openai__gpt-5.2",
    "google__gemini-3-flash-preview",
    "anthropic__claude-sonnet-4.6",
    "moonshotai__kimi-k2.5",
    "deepseek__deepseek-v4-flash",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def top_candidate(rank: dict[str, int]) -> str:
    return min(rank, key=lambda letter: (int(rank[letter]), letter))


def human_letter_from_maps(
    version: str,
    direction: str,
    sample_id: str,
    row: dict[str, Any],
    map_by_id: dict[str, dict[str, str]],
    variants_by_index: dict[int, dict[str, Any]],
    row_index: int,
) -> str:
    source_map = row.get("candidate_source_map")
    if isinstance(source_map, dict):
        for letter, source in source_map.items():
            if source == "human_translation":
                return letter

    position_map = row.get("candidate_position_map")
    if isinstance(position_map, dict):
        for display_letter, original_letter in position_map.items():
            if original_letter == "A":
                return display_letter

    external_map = map_by_id.get(sample_id)
    if external_map:
        for display_letter, original_letter in external_map.items():
            if original_letter == "A":
                return display_letter

    variants = variants_by_index.get(row_index)
    if variants and variants.get("human_translation"):
        human = variants["human_translation"]
        for letter in CANDIDATES:
            if row.get(f"candidate_{letter}") == human:
                return letter

    raise ValueError(f"Cannot infer human candidate letter for {version}/{direction}/{sample_id}")


def load_external_position_map(version_dir: Path, direction: str, prefix: str) -> dict[str, dict[str, str]]:
    map_by_id: dict[str, dict[str, str]] = {}

    map_file = version_dir / direction / "datasets" / f"{prefix}.candidate_position_map.json"
    if map_file.exists():
        for row in read_json(map_file):
            mapping = row.get("display_to_original_candidate") or row.get("candidate_position_map")
            if mapping:
                map_by_id[str(row["id"])] = mapping

    with_candidates = version_dir / direction / "datasets" / f"{prefix}.with_candidates.json"
    if with_candidates.exists():
        for row in read_json(with_candidates):
            mapping = row.get("candidate_position_map")
            if mapping:
                map_by_id[str(row["id"])] = mapping
                original_id = str(row.get("original_id", ""))
                if original_id:
                    map_by_id[original_id] = mapping
    return map_by_id


def compute_human_reference_top1() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    judge_set_rows: list[dict[str, Any]] = []

    for version, version_cfg in VERSIONS.items():
        version_dir = Path(version)
        excluded_model = version_cfg["generator_model"]
        for direction, prefix in DIRECTIONS.items():
            dataset_path = version_dir / direction / "datasets" / f"{prefix}.with_candidates.shuffled.json"
            variants_path = version_dir / direction / "datasets" / f"{prefix}.with_variants.json"
            rows = read_json(dataset_path)
            variants = read_json(variants_path) if variants_path.exists() else []
            variants_by_index = {idx: row for idx, row in enumerate(variants)}
            map_by_id = load_external_position_map(version_dir, direction, prefix)

            human_by_id: dict[str, str] = {}
            for idx, row in enumerate(rows):
                sample_id = str(row["id"])
                human_by_id[sample_id] = human_letter_from_maps(
                    version,
                    direction,
                    sample_id,
                    row,
                    map_by_id,
                    variants_by_index,
                    idx,
                )

            for model in MODEL_ORDER:
                rank_path = (
                    version_dir
                    / direction
                    / "results/model_based_metrics/rankings/json"
                    / f"{prefix}.with_candidates.shuffled.ranked.{model}.json"
                )
                if not rank_path.exists():
                    continue
                rank_rows = read_json(rank_path)
                n = 0
                top1_human = 0
                top2_or_better = 0
                rank_sum = 0
                for rank_row in rank_rows:
                    sample_id = str(rank_row["id"])
                    rank = {letter: int(value) for letter, value in rank_row["rank"].items()}
                    human_letter = human_by_id[sample_id]
                    human_rank = int(rank[human_letter])
                    model_top = top_candidate(rank)
                    n += 1
                    rank_sum += human_rank
                    top1_human += int(human_rank == 1)
                    top2_or_better += int(human_rank <= 2)
                    detail_rows.append(
                        {
                            "version": version,
                            "direction": direction.upper(),
                            "dataset": prefix,
                            "judge_model": model,
                            "sample_id": sample_id,
                            "human_candidate": human_letter,
                            "top1_candidate": model_top,
                            "human_rank": human_rank,
                            "human_is_top1": human_rank == 1,
                            "human_is_top2_or_better": human_rank <= 2,
                        }
                    )
                summary_rows.append(
                    {
                        "version": version,
                        "direction": direction.upper(),
                        "dataset": prefix,
                        "judge_model": model,
                        "judge_set": "all_judges_5model",
                        "is_candidate_generator_judge": model == excluded_model,
                        "n_samples": n,
                        "human_top1_count": top1_human,
                        "human_top1_rate": round(top1_human / n, 6),
                        "human_top2_or_better_rate": round(top2_or_better / n, 6),
                        "mean_human_rank": round(rank_sum / n, 6),
                        "random_top1_baseline": round(1 / 3, 6),
                    }
                )

            included_models = [model for model in MODEL_ORDER if model != excluded_model]
            sub = [
                row
                for row in summary_rows
                if row["version"] == version
                and row["direction"] == direction.upper()
                and row["judge_model"] in included_models
            ]
            judge_set_rows.append(
                {
                    "version": version,
                    "direction": direction.upper(),
                    "dataset": prefix,
                    "judge_set": f"judges_excluding_candidate_generator_{version_cfg['generator_slug']}",
                    "excluded_judge_model": excluded_model,
                    "included_judge_models": ";".join(included_models),
                    "n_judges": len(included_models),
                    "mean_human_top1_rate": round(sum(float(row["human_top1_rate"]) for row in sub) / len(sub), 6),
                    "mean_human_top2_or_better_rate": round(
                        sum(float(row["human_top2_or_better_rate"]) for row in sub) / len(sub),
                        6,
                    ),
                    "mean_human_rank": round(sum(float(row["mean_human_rank"]) for row in sub) / len(sub), 6),
                    "random_top1_baseline": round(1 / 3, 6),
                }
            )

    return detail_rows, summary_rows, judge_set_rows


def normalize_old_name(path: Path) -> str:
    name = path.name
    name = name.replace(".five_models.", ".all_judges_5model.")
    name = name.replace(".four_models.", ".legacy_four_judges_not_self_excluded.")
    name = name.replace(".joint_features.cluster_robust_pairwise_logistic.", ".all_judges_5model.joint_features.cluster_robust_pairwise_logistic.")
    return name


def copy_artifact(src: Path, dest_dir: Path, version: str | None, direction: str | None, note: str, manifest: list[dict[str, Any]]) -> None:
    if not src.exists() or src.name.endswith(".cache.json") or ".cache." in src.name:
        return
    prefix = ""
    if version and direction:
        prefix = f"{version}__{direction.upper()}__"
    dest_name = prefix + normalize_old_name(src)
    dest = dest_dir / dest_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    manifest.append(
        {
            "organized_path": str(dest).replace("\\", "/"),
            "source_path": str(src).replace("\\", "/"),
            "version": version or "",
            "direction": direction.upper() if direction else "",
            "category": str(dest_dir.relative_to(PACKAGE_DIR)).replace("\\", "/"),
            "note": note,
        }
    )


def write_readme(path: Path, title: str, body: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "README.md").write_text(f"# {title}\n\n{body.strip()}\n", encoding="utf-8")


def build_package() -> list[dict[str, Any]]:
    if PACKAGE_DIR.exists():
        shutil.rmtree(PACKAGE_DIR)
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    detail_rows, summary_rows, judge_set_rows = compute_human_reference_top1()
    alignment_dir = PACKAGE_DIR / "01_human_reference_alignment"
    write_csv(
        alignment_dir / "human_reference_top1_by_sample.csv",
        detail_rows,
        [
            "version",
            "direction",
            "dataset",
            "judge_model",
            "sample_id",
            "human_candidate",
            "top1_candidate",
            "human_rank",
            "human_is_top1",
            "human_is_top2_or_better",
        ],
    )
    write_csv(
        alignment_dir / "human_reference_top1_by_judge.csv",
        summary_rows,
        [
            "version",
            "direction",
            "dataset",
            "judge_model",
            "judge_set",
            "is_candidate_generator_judge",
            "n_samples",
            "human_top1_count",
            "human_top1_rate",
            "human_top2_or_better_rate",
            "mean_human_rank",
            "random_top1_baseline",
        ],
    )
    write_csv(
        alignment_dir / "human_reference_top1_by_self_excluded_judge_set.csv",
        judge_set_rows,
        [
            "version",
            "direction",
            "dataset",
            "judge_set",
            "excluded_judge_model",
            "included_judge_models",
            "n_judges",
            "mean_human_top1_rate",
            "mean_human_top2_or_better_rate",
            "mean_human_rank",
            "random_top1_baseline",
        ],
    )
    write_json(alignment_dir / "human_reference_top1_by_sample.json", detail_rows)

    for file in alignment_dir.glob("*"):
        if file.is_file():
            manifest.append(
                {
                    "organized_path": str(file).replace("\\", "/"),
                    "source_path": "generated",
                    "version": "",
                    "direction": "",
                    "category": "01_human_reference_alignment",
                    "note": "Human reference top-1 alignment statistics.",
                }
            )

    for version in VERSIONS:
        for direction, prefix in DIRECTIONS.items():
            analysis_root = Path(version) / direction / "results"
            model_csv = analysis_root / "model_based_metrics/analysis/pilot/csv"
            model_json = analysis_root / "model_based_metrics/analysis/pilot/json"
            parser_csv = analysis_root / "parser_derived_syntactic_metrics/analysis/pilot/csv"
            rule_csv = analysis_root / "rule_based_proxy_features/analysis/pilot/csv"

            for src in sorted(model_csv.glob(f"{prefix}.five_models.*.csv")) + sorted(model_json.glob(f"{prefix}.five_models.*.json")):
                copy_artifact(src, PACKAGE_DIR / "02_judge_agreement_and_consensus" / "all_judges_5model", version, direction, "Legacy five_models renamed to all_judges_5model.", manifest)
            for src in sorted(model_csv.glob(f"{prefix}.four_models.*.csv")) + sorted(model_json.glob(f"{prefix}.four_models.*.json")):
                copy_artifact(src, PACKAGE_DIR / "02_judge_agreement_and_consensus" / "legacy_four_judges_not_self_excluded", version, direction, "Historical four_models outputs; not a valid self-exclusion set for Kimi.", manifest)
            for src in sorted(model_csv.glob(f"{prefix}.judges_excluding_candidate_generator_*")) + sorted(model_json.glob(f"{prefix}.judges_excluding_candidate_generator_*")):
                copy_artifact(src, PACKAGE_DIR / "03_self_excluded_judge_sets", version, direction, "Judge set excluding the candidate generator itself.", manifest)
            for src in sorted(model_csv.glob(f"{prefix}.*cluster_robust_pairwise_logistic*")):
                if "judges_excluding_candidate_generator" not in src.name:
                    copy_artifact(src, PACKAGE_DIR / "04_cluster_robust_regressions" / "model_based_all_judges", version, direction, "Cluster-robust model-based regression output.", manifest)
            for src in sorted(parser_csv.glob(f"{prefix}.*cluster_robust_pairwise_logistic*")):
                copy_artifact(src, PACKAGE_DIR / "04_cluster_robust_regressions" / "parser_syntax", version, direction, "Cluster-robust parser-derived syntax regression output.", manifest)
            for src in sorted(rule_csv.glob(f"{prefix}.*cluster_robust_pairwise_logistic*")):
                copy_artifact(src, PACKAGE_DIR / "04_cluster_robust_regressions" / "rule_based_proxy", version, direction, "Cluster-robust rule-based proxy regression output.", manifest)
            for src in sorted(model_json.glob(f"{prefix}.*features.by_candidate.json")):
                copy_artifact(src, PACKAGE_DIR / "05_feature_matrices" / "model_based_features", version, direction, "Per-candidate model-based feature matrix.", manifest)
            for src in sorted((analysis_root / "parser_derived_syntactic_metrics/analysis/pilot/json").glob(f"{prefix}.*features.by_candidate.json")):
                copy_artifact(src, PACKAGE_DIR / "05_feature_matrices" / "parser_syntax_features", version, direction, "Per-candidate parser-derived syntax feature matrix.", manifest)
            for src in sorted((analysis_root / "rule_based_proxy_features/analysis/pilot/json").glob(f"{prefix}.*features.by_candidate.json")):
                copy_artifact(src, PACKAGE_DIR / "05_feature_matrices" / "rule_based_proxy_features", version, direction, "Per-candidate rule-based proxy feature matrix.", manifest)

    for src in sorted(SUMMARY_DIR.glob("*")):
        if src.is_file():
            copy_artifact(src, PACKAGE_DIR / "06_stability_and_review_summaries", None, None, "Cross-version/direction stability summary or review note.", manifest)

    native_dir = Path("data/external_baselines")
    if native_dir.exists():
        for pattern in [
            "stanza_native_syntax_baseline_stats.*",
            "stanza_native_syntax_candidate_comparison_overview.csv",
            "STANZA_NATIVE_SYNTAX_METHOD_NOTES.md",
        ]:
            for src in sorted(native_dir.glob(pattern)):
                copy_artifact(src, PACKAGE_DIR / "07_native_baseline_syntax", None, None, "Native finance baseline syntax comparison.", manifest)

    write_readme(
        alignment_dir,
        "Human Reference Alignment",
        """
Reports how often each judge ranked the human reference translation first.
`human_reference_top1_by_judge.csv` is the main table for the paper. The
random top-1 baseline is 1/3 because each sample has three candidates.
""",
    )
    write_readme(
        PACKAGE_DIR / "02_judge_agreement_and_consensus",
        "Judge Agreement And Consensus",
        """
Contains pairwise agreement, model-vs-consensus agreement, Borda summaries, and
related consensus artifacts. Historical `five_models` outputs are copied with
the clearer name `all_judges_5model`; historical `four_models` outputs are
marked as `legacy_four_judges_not_self_excluded`.
""",
    )
    write_readme(
        PACKAGE_DIR / "03_self_excluded_judge_sets",
        "Self-Excluded Judge Sets",
        """
Contains the sensitivity analyses that exclude the candidate generator itself:
Kimi-version excludes Kimi K2.5; Claude-version excludes Claude Sonnet 4.6.
These are the preferred files for addressing self-preference concerns.
""",
    )
    write_readme(
        PACKAGE_DIR / "04_cluster_robust_regressions",
        "Cluster-Robust Regressions",
        """
Contains cluster-robust pairwise logistic regression outputs. Use these for
inference because each source sample contributes three pairwise observations.
""",
    )
    write_readme(
        PACKAGE_DIR / "05_feature_matrices",
        "Feature Matrices",
        """
Contains per-candidate feature matrices used by regression analyses. Cache files
are intentionally excluded from this review-ready package.
""",
    )
    write_readme(
        PACKAGE_DIR / "06_stability_and_review_summaries",
        "Stability And Review Summaries",
        """
Contains cross-version stability tiers and concise finding summaries, including
the translationese-subfeature replacement results.
""",
    )
    write_readme(
        PACKAGE_DIR / "07_native_baseline_syntax",
        "Native Baseline Syntax",
        """
Contains external native finance baseline syntax summaries and candidate-vs-
native z-score overviews. These support interpretation of target-language
structural deviation.
""",
    )

    write_csv(
        PACKAGE_DIR / "MANIFEST.csv",
        manifest,
        ["organized_path", "source_path", "version", "direction", "category", "note"],
    )
    write_json(PACKAGE_DIR / "MANIFEST.json", manifest)
    write_package_guide()
    return manifest


def write_package_guide() -> None:
    guide = """# Organized Analysis Outputs Guide

This folder is a review-ready, functionally organized copy of the key computed
CSV/JSON/MD outputs. It does not replace the original project result folders.

## Naming Policy

- Historical `five_models` outputs are renamed to `all_judges_5model`.
- Historical `four_models` outputs are renamed to
  `legacy_four_judges_not_self_excluded`.
- Preferred self-preference sensitivity files use
  `judges_excluding_candidate_generator_<model_slug>`.
- Files are prefixed with `<version>__<DIRECTION>__` so outputs from the two
  candidate-generator versions can be compared without ambiguity.

## Folder Map

1. `01_human_reference_alignment/`  
   Human-reference top-1 rates by judge and by self-excluded judge set.

2. `02_judge_agreement_and_consensus/`  
   Pairwise agreement, model-vs-consensus agreement, Borda summaries, and
   historical all-judge/four-judge outputs under clearer names.

3. `03_self_excluded_judge_sets/`  
   Preferred sensitivity analyses excluding the candidate generator itself.

4. `04_cluster_robust_regressions/`  
   Cluster-robust pairwise logistic regressions, grouped by feature source.

5. `05_feature_matrices/`  
   Per-candidate feature matrices used as regression inputs.

6. `06_stability_and_review_summaries/`  
   Cross-version stability tier summaries and concise finding notes.

7. `07_native_baseline_syntax/`  
   External native finance baseline syntax distributions and candidate
   comparison overviews.

## Recommended Reading Order

Start with:

- `01_human_reference_alignment/human_reference_top1_by_judge.csv`
- `03_self_excluded_judge_sets/*joint_features_with_translationese_subfeatures.cluster_robust_pairwise_logistic.csv`
- `06_stability_and_review_summaries/self_excluded_judge_set_translationese_subfeature_stability_tiers.csv`
- `07_native_baseline_syntax/stanza_native_syntax_candidate_comparison_overview.csv`

Then use `MANIFEST.csv` to trace any organized file back to its source.

## Important Interpretation Notes

- Regression accuracy fields are in-sample explanatory fit, not held-out
  prediction accuracy.
- Cluster-robust files should be preferred for inference.
- The self-excluded judge-set outputs are the cleanest files for addressing
  candidate-generator self-preference.
- Translationese interpretation should rely primarily on the subfeature
  replacement outputs rather than the old weighted composite score.
"""
    (PACKAGE_DIR / "README.md").write_text(guide, encoding="utf-8")


def main() -> int:
    manifest = build_package()
    print(f"wrote {PACKAGE_DIR} with {len(manifest)} organized artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
