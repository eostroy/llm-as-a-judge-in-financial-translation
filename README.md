# LLM-as-a-Judge Translation Benchmark

This workspace contains EC/CE translation benchmark data, model ranking outputs,
and analysis scripts for comparing LLM judge preferences.

## Directory Layout

- `benchmark/`  
  Final benchmark JSON files used for generation, ranking, and analysis.

- `data/raw/`  
  Original source materials, including FFN raw data and untagged ECPCFE text files.

- `data/interim/`  
  Intermediate converted datasets kept for reproducibility.

- `frontend/`  
  Browser-based JSON translation reader/editor.

- `prompts/`  
  Prompt templates for candidate ranking.

- `results/rankings/ec/`  
  EC model ranking outputs.

- `results/analysis/ec_pilot/`  
  EC pilot analysis outputs, including consensus, agreement, embedding features,
  and syntax/information-structure features.

- `results/logs/`  
  Runtime logs split by task type.

- `scripts/`  
  Dataset construction, candidate generation, ranking, and analysis scripts.

## Main Files

- EC test set: `benchmark/ffn_200ec.with_candidates.shuffled.json`
- CE test set: `benchmark/ecpcfe_200ce.with_variants.json`
- JSON editor: `frontend/json_translation_editor.html`
- Ranking script template: `scripts/rank_translation_candidates_openrouter.py`
- CE candidate generation: `scripts/generate_translation_candidates.py`
- EC embedding analysis: `scripts/analyze_ec_embedding_features.py`
- EC syntax analysis: `scripts/analyze_ec_syntax_info_features.py`
