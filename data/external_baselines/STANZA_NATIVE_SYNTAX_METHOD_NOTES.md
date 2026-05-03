# Stanza Native Syntax Baseline Notes

This is the parser-based native-baseline comparison for candidate translations.
It supersedes the earlier rule-based syntax proxy experiment.

## Parser

- Parser: Stanza `tokenize,pos,lemma,depparse`
- Chinese baseline parser: `zh` / `zh-hans`
- English baseline parser: `en`
- Candidate translation comparison reuses the existing project Stanza outputs in
  `parser_derived_syntactic_metrics`, so candidate and baseline z-scores are
  computed from the same feature definitions.

## Baseline Files

- Chinese native baseline: `zh_native_finance_kenpache/clean.txt`
- English native baseline: `en_native_finance_kenpache/clean.txt`
- Parsed baseline rows:
  - `stanza_zh_native_syntax.by_text.csv/json`
  - `stanza_en_native_syntax.by_text.csv/json`
- Baseline distributions:
  - `stanza_native_syntax_baseline_stats.csv/json`
- Candidate comparison overview:
  - `stanza_native_syntax_candidate_comparison_overview.csv`

## Compared Parser Features

The candidate comparisons use parser-derived fields already present in the
project:

- `clause_count`
- `dependency_depth`
- `mean_dependency_distance`
- `max_dependency_distance`
- `normalized_dependency_distance`
- `nominalization_ratio`
- `modifier_density`
- `coordination_count`
- `passive_or_beishi_count` for Chinese targets
- `passive_count` for English targets
- `sentence_compression_ratio`

The baseline parser also computes `token_count`, `sentence_count`, and
`avg_sentence_length`, but these are not included in the default candidate
z-score overview unless candidate translations are explicitly reparsed with
`--force-reparse-candidates`.

## Reproduce

```powershell
$env:PYTHONPATH='C:\tmp\stanza_site'
python scripts/shared/analyze_stanza_native_baseline_syntax.py --batch-size 128
```
