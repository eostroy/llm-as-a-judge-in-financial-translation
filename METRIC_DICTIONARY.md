# Metric Dictionary

This dictionary explains the main metric names used in the CSV/JSON outputs.
It is meant for interpretation, not for changing script-facing column names.

## Judge Ranking and Agreement

| Name | Plain meaning | Use | Caution |
| --- | --- | --- | --- |
| `rank` | A/B/C candidate ranking assigned by one judge model | Raw LLM-as-a-judge observation | Lower number means better rank |
| `pairwise_agreement` | Agreement rate between two judge models over candidate pairs | Overall stability of model judgments | Agreement does not reveal why models agree |
| `model_vs_consensus_agreement` | Agreement between one model and the group consensus | Model-level closeness to group preference | Consensus is descriptive, not a human gold standard |
| `consensus_borda` | Borda-score aggregate ranking across models | Group-level preferred candidate | Sensitive to the set of judge models included |
| `condorcet` | Pairwise majority winner summary | Majority preference stability | Cycles can occur |

## Semantic and NLI Metrics

| Name | Plain meaning | Use | Caution |
| --- | --- | --- | --- |
| `crosslingual_embedding_similarity` | LaBSE cosine similarity between source and candidate translation | Main cross-lingual semantic similarity indicator | Stronger and more stable in CE than EC in current results |
| `bidirectional_entailment` | Combined source-to-target and target-to-source entailment strength | Semantic equivalence / mutual support | Model-based NLI score, not a human entailment label |
| `nli_omission_risk` | Risk that candidate omits source information | Source information completeness | Negative regression coefficient means lower omission risk is preferred |
| `nli_contradiction_risk` | Risk of contradiction between source and candidate | Semantic inconsistency risk | Rare or noisy contradictions may weaken inference |
| `nli_boundary_risk` | Risk of unstable information boundary between source and candidate | Information-boundary control | Interpret cautiously; direction is not fully stable across versions |

## Dependency Syntax Metrics

| Name | Plain meaning | Use | Caution |
| --- | --- | --- | --- |
| `dependency_depth` | Maximum root-to-token dependency depth | Hierarchical syntactic depth | Parser-derived, so parsing errors matter |
| `mean_dependency_distance` | Mean token distance between dependent and syntactic head | Average local dependency span | Length-sensitive unless normalized |
| `max_dependency_distance` | Longest dependency span in the sentence/sample | Extreme structural span | Can be driven by one long construction |
| `normalized_dependency_distance` | Mean dependency distance divided by non-root token count | Length-adjusted structural span | Better for comparing candidates of different lengths |
| `clause_count` | Approximate number of clauses/sentences | Clause-level segmentation/complexity | Approximation based on parser and punctuation |
| `modifier_density` | Modifier-heavy expression density | Local descriptive density | Language-specific parser behavior can affect it |
| `nominalization_ratio` | Nominalized or noun-heavy expression proxy | Written/nominal style proxy | Partly heuristic |
| `passive_or_beishi_count` | Passive or bei-style construction count | Passive construction indicator | Language-specific and sparse |

## Length and Expression Metrics

| Name | Better readable label | Plain meaning | Caution |
| --- | --- | --- | --- |
| `sentence_compression_ratio` | `target_source_length_ratio` / `translation_expansion_ratio` | EC: Chinese target characters divided by English source words. CE: English target words divided by Chinese source characters. | The historical name is misleading. It is not a universal measure of good compression. Higher EC values can mean fuller Chinese information expression. |
| `target_lm_naturalness_score` | target-language LM naturalness | Negative log-perplexity style target-language score; higher usually means more natural under the local LM | Auxiliary model-based proxy only |
| `translationese_score` | translationese-style proxy | Rule/proxy signal for translation-like wording | Heuristic; use as a control, not a core proof |
| `register_score` | register/style control proxy | Rule-based style/register signal | Heuristic |

## Rule/Control Metrics

| Name | Plain meaning | Use | Caution |
| --- | --- | --- | --- |
| `extra_number_ratio` | Extra number mismatch ratio | Numeric fidelity/control | Rule-based proxy |
| `extra_entity_count` | Extra entity count | Entity addition/control | Rule-based proxy |
| `deep_features` | Rule/proxy control feature family | Auxiliary controls in regression | The name is historical; these are not deep neural features |

## Regression Columns

| Name | Plain meaning | Use | Caution |
| --- | --- | --- | --- |
| `standardized_coefficient` | Logistic coefficient after feature standardization | Direction and relative strength of preference association | Not causal; sign matters more than raw scale |
| `cluster_robust_se` | Standard error clustered by `sample_id` | Corrects for three pairwise rows from the same sample | Prefer this for significance |
| `p_cluster` | p-value using clustered SE | Statistical support after correcting within-sample dependence | Still exploratory because features are many and sample size is modest |
| `ci95_low`, `ci95_high` | 95% confidence interval using clustered SE | Whether the coefficient plausibly crosses zero | Interpret alongside cross-version/model consistency |
| `training_accuracy` | Pairwise logistic in-sample accuracy | Descriptive fit of the feature model | Not a held-out prediction score |

