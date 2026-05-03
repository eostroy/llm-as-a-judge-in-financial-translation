# Self-Excluded Judge Set Findings

This summary uses only judge sets that exclude the candidate generator itself.

## File Naming Rule

All newly generated self-excluded outputs use:

`<dataset>.judges_excluding_candidate_generator_<excluded_model_slug>.<artifact>`

Examples:

- `ffn_200ec.judges_excluding_candidate_generator_kimi_k2_5.joint_features.cluster_robust_pairwise_logistic.csv`
- `ecpcfe_200ce.judges_excluding_candidate_generator_claude_sonnet_4_6.consensus_borda_summary.csv`

## Exclusion Rule

- `kimi-version`: excludes `moonshotai__kimi-k2.5`.
- `claude-version`: excludes `anthropic__claude-sonnet-4.6`.

## Stability Tiers

The tier file is `self_excluded_judge_set_feature_stability_tiers.csv`.
Each direction-feature row pools 8 estimates: 2 candidate-generator versions x
4 included judge models.

- `no_exception`: all 8 estimates are significant at p < .05 and share the same sign.
- `mostly_supported_with_exceptions`: at least 5/8 estimates are significant and at least 6/8 coefficients share the same sign.
- `no_stable_evidence`: all other cases.

## Main Tiered Findings

### No Exception

- CE: `crosslingual_embedding_similarity` is consistently positive.
- EC: `translationese_score` is consistently negative.

### Mostly Supported With Exceptions

- CE: `sentence_compression_ratio` is consistently negative in sign and significant in 7/8 estimates.
- EC: `extra_number_ratio` is consistently negative in sign and significant in 5/8 estimates.
- EC: `nli_omission_risk` is consistently negative in sign and significant in 5/8 estimates.

### No Stable Evidence

All remaining direction-feature combinations lack stable evidence under the
self-excluded judge-set analysis. This includes `bidirectional_entailment`,
target LM naturalness, and most individual dependency-syntax features.

## Translationese Subfeature Replacement

The weighted composite `translationese_score` was also replaced with independent
translationese diagnostic subfeatures. Outputs use:

`<dataset>.judges_excluding_candidate_generator_<excluded_model_slug>.joint_features_with_translationese_subfeatures.<artifact>`

The tier file is:

`self_excluded_judge_set_translationese_subfeature_stability_tiers.csv`

The replacement model keeps the same self-excluded judge sets and the same joint
feature framework, but removes `translationese_score` and adds:

- `target_function_word_density`
- `target_passive_density`
- `target_pronoun_subject_ratio`
- `target_explicit_connective_density`
- `target_nominalization_suffix_density`
- `target_avg_sentence_length`

### No Exception Under Subfeature Replacement

- CE: `crosslingual_embedding_similarity` remains consistently positive.
- CE: `target_passive_density` is consistently negative.
- EC: `target_function_word_density` is consistently negative.
- EC: `target_explicit_connective_density` is consistently negative.

### Mostly Supported With Exceptions Under Subfeature Replacement

- CE: `sentence_compression_ratio` remains mostly negative.
- EC: `extra_number_ratio` remains mostly negative.
- EC: `nli_omission_risk` is mostly negative.

### Interpretation

The original EC translationese finding survives the replacement, but its more
precise mechanism is no longer a hand-weighted composite score. In the
self-excluded analysis, EC preferences are most consistently associated with
lower target-side function-word density, especially the Chinese DE particle, and
lower explicit connective density. This supports a more interpretable claim
about translationese diagnostics without relying on arbitrary composite weights.
