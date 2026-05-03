# Data Guide for kimi-version

domestic-generator version: candidates were generated in the Kimi-based run

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
