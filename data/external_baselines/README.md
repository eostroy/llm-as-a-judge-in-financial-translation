# External Native Finance Baselines

These corpora are saved for target-language baseline normalization of translationese-style features.

## zh_native_finance_kenpache

- Source: `Kenpache/multilingual-financial-sentiment`
- URL: https://huggingface.co/datasets/Kenpache/multilingual-financial-sentiment
- Selection: rows with `language == "zh"`
- Clean text: one financial-news sentence per line

## en_native_finance_kenpache

- Source: `Kenpache/multilingual-financial-sentiment`
- URL: https://huggingface.co/datasets/Kenpache/multilingual-financial-sentiment
- Selection: rows with `language == "en"`
- Clean text: one financial-news sentence per line

Each subdirectory contains:

- `clean.txt`: normalized one-text-per-line baseline input
- `raw/records.jsonl`: row-level provenance and retained source fields
- `metadata.json`: retrieval date, source, split, field choices, and counts
