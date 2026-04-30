# LLM-as-a-Judge Translation Benchmark

This workspace contains parallel EC/CE translation benchmark data, model
ranking outputs, and analysis scripts for comparing LLM judge preferences.

## Directory Layout

- `ec/`  
  English-to-Chinese benchmark area. Contains EC datasets, EC ranking outputs,
  EC logs, and EC analysis scripts/results.

- `ce/`  
  Chinese-to-English benchmark area. Contains CE datasets, CE ranking outputs,
  CE logs, and CE data-generation/ranking scripts.

- `data/raw/`  
  Original source materials, including FFN raw data and untagged ECPCFE text
  files used to construct benchmark samples.

- `frontend/`  
  Browser-based JSON translation reader/editor.

- `prompts/`  
  Prompt templates shared by ranking scripts.

- `scripts/shared/`  
  Legacy or shared utility scripts that are not tied to one direction.

## Main Files

- EC shuffled test set: `ec/datasets/ffn_200ec.with_candidates.shuffled.json`
- EC rankings: `ec/results/rankings/`
- EC analysis: `ec/results/analysis/pilot/`

- CE shuffled test set: `ce/datasets/ecpcfe_200ce.with_candidates.shuffled.json`
- CE rankings: `ce/results/rankings/`
- CE original generated variants: `ce/datasets/ecpcfe_200ce.with_variants.json`

- JSON editor: `frontend/json_translation_editor.html`
- EC ranking script: `ec/scripts/rank_translation_candidates_openrouter.py`
- CE ranking script: `ce/scripts/rank_translation_candidates_openrouter.py`
- CE candidate generation: `ce/scripts/generate_translation_candidates.py`
