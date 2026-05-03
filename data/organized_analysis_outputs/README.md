# Organized Analysis Outputs Guide

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
