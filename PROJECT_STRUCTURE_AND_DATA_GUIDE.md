# Project Structure and Data Guide

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

### kimi-version
- LaBSE semantic similarity: 10 files
- cache: 1 files
- consensus ranking: 16 files
- cross-lingual NLI: 12 files
- dataset: 10 files
- dependency syntax: 20 files
- joint logistic: 2 files
- joint logistic with cluster-robust SE: 4 files
- judge ranking output: 19 files
- method notes: 3 files
- model agreement: 6 files
- model disagreement: 16 files
- other: 23 files
- pairwise regression design: 2 files
- rule/proxy features: 12 files
- single-feature logistic: 1 files
- target-language LM: 10 files
- top-1 feature average: 1 files

### claude-version
- LaBSE semantic similarity: 10 files
- consensus ranking: 8 files
- cross-lingual NLI: 12 files
- dataset: 8 files
- dependency syntax: 20 files
- joint logistic: 2 files
- joint logistic with cluster-robust SE: 4 files
- judge ranking output: 10 files
- method notes: 2 files
- model agreement: 4 files
- model disagreement: 8 files
- other: 12 files
- pairwise regression design: 2 files
- rule/proxy features: 12 files
- target-language LM: 10 files

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
