# LLM-as-a-Judge 财经译文质量排序项目

本项目研究大语言模型作为译文质量评审者时的排序稳定性与偏好机制。当前项目包含两个平行版本：

- `kimi-version/`：国产模型生成候选译文版本。
- `claude-version/`：海外模型生成候选译文版本。

两个版本的目录结构保持一致，便于直接比较候选来源是否影响 LLM-as-a-judge 的排序结果。

## 先看这些文件

- 中文项目结构说明：`项目结构与数据指南.md`
- 中文指标词典：`指标词典.md`
- 完整 CSV/JSON 文件索引：`FILE_INDEX.csv`
- 英文结构说明备份：`PROJECT_STRUCTURE_AND_DATA_GUIDE.md`
- 英文指标词典备份：`METRIC_DICTIONARY.md`

## 方向说明

- `ec/`：英译中任务，数据来自 FFN 财经新闻语料。
- `ce/`：中译英任务，数据来自 ECPCFE 财经/经济语料。

## 当前最重要的分析文件类型

- 模型间一致性：`*.five_models.pairwise_agreement.csv`
- 模型与共识一致性：`*.five_models.model_vs_consensus_agreement.csv`
- 共识排序：`*.five_models.consensus_borda_summary.csv`
- LaBSE 跨语言语义相似度：`*.local_embedding_features.*`
- 跨语言 NLI：`*.crosslingual_nli_features.*`
- 依存句法指标：`*.syntax_info_features.*`
- 推荐使用的联合回归结果：`*.joint_features.cluster_robust_pairwise_logistic.csv`

旧的 `*.joint_features.pairwise_logistic_preferences.csv` 仍可用于查看系数方向，但没有处理同一样本内三个 pairwise 判断的相关性。涉及统计显著性时，应优先使用带 `cluster_robust` 的文件。

