# Session 主要内容整理

## 一、项目理解与审稿

对项目做了全面审查：两个版本（kimi-version / claude-version）× 两个方向（EC 英译中 / CE 中译英）× 200 条样本 × 5 评审模型盲评排序 × 多层特征分析。

撰写了外部审稿意见（`外部审稿意见.md`），指出了 API 密钥泄露、自偏好混淆、手写梯度下降、多重比较、NLI 共线性等问题。审稿意见随项目迭代不断更新。

## 二、EC 数据从 200 翻到 400

从 FFN Excel（body sheet，~1013 行可用）抽取了新的 200 条源文（id 201–400），为两版本分别生成候选译文（Kimi K2.5 和 Claude Sonnet 4.6），调用 4 个评审模型盲评排序（排除自身候选生成器），合并新旧数据到 400 条。

**关键数据：**
- kimi-version EC 400 条 pairwise 一致性：0.8293（200 条时 0.8360）
- claude-version EC 400 条 pairwise 一致性：0.8433（200 条时 0.8347）

## 三、特征体系

最终使用的 **22 个特征**，分五族：

| 族 | 特征数 | 来源 |
|------|------|------|
| LaBSE 语义 | 1 | 本地 LaBSE 模型（8.8 GB） |
| NLI | 4 | mDeBERTa-v3-base-mnli-xnli（552 MB） |
| 依存句法 | 10 | Stanza（zh） |
| 翻译腔 | 6 | 正则 + 词表（独立子特征，不加权） |
| 句法复杂度 + 词汇专业化 | 2 | clause_density + lexical_frequency_deviation |

**已删除的弱特征：** 规则式 proxy（extra_number_ratio、extra_entity_count、register_score）、篇章级特征（mtld、cohesion_mean_overlap、content_word_ratio）、explicit_subject_ratio。

## 四、特征相关性分析

详见 `特征相关性分析.md`，包含 Pearson r、Spearman r_s、VIF 三项分析。

**关键发现：**
- NLI 族内部严重共线性：`bidirectional_entailment` 与 `nli_omission_risk` 的 Pearson r ≈ −0.91 ~ −0.96，Spearman r_s ≈ −0.98，VIF ≈ 20–50
- 翻译腔子特征之间无高相关，各自独立，VIF 全 < 1.5
- 句法族和 LaBSE 族与其他特征族无跨族高相关

## 五、联合回归与稳定性分层（22 特征，400 条 EC）

使用 cluster-robust SE（按 sample_id 聚类）、L2 正则化 logistic 回归、自排除评审集（排除候选生成器自身）。

### 共识（no_exception，8/8 显著且同号）

| 特征 | 方向 | 含义 |
|------|------|------|
| `target_function_word_density` | − | 所有模型排斥"的"过多 |
| `target_explicit_connective_density` | − | 所有模型排斥显式连接词堆叠 |
| `target_passive_density` | − | 所有模型排斥被动结构 |
| `lexical_frequency_deviation` | − | 所有模型偏好更专业化的词汇 |

### 模型差异（核心发现）

**GPT-5.2 是五个模型中最"异类"的一个。** 它有四个独有显著特征：

| 独有特征 | 方向 | 其他模型 |
|------|------|------|
| `crosslingual_embedding_similarity` | + | 全不显著 |
| `dependency_depth` | − | 全不显著 |
| `passive_or_beishi_count` | − | 全不显著 |
| `target_avg_sentence_length` | + | 全不显著 |

GPT-5.2 在 EC 方向额外依赖**语义相似度**和**句法简单度**——它是一个"语义稽核员"。其他四个模型几乎纯粹依赖翻译腔信号做判断。

**其他模型差异：**

- **DeepSeek**：唯一显著排斥 `target_pronoun_subject_ratio`（句首代词）
- **Kimi**：唯一显著偏好 `nominalization_ratio`（名物化比例）
- **Gemini + GPT-5.2**：唯二对 `nli_boundary_risk` 敏感
- **Claude**：最纯粹依赖翻译腔——仅有的模型差异特征为零（所有独有特征都不显著）

## 六、样本定性分析

在全票通过样本中，共识最佳的译文"的"密度显著低于最差译文。"的"密度 = 0.000 的译文（完全不用"的"）在多个样本中被全票选为最佳。72% 的样本中，最佳候选的"的"密度低于最差候选。

## 七、叙事线转向：从"特征解释"到"模型对比"

研究的原始目标是**五个评审模型的排序偏好对比**，但在分析过程中滑向了"什么特征驱动 LLM 偏好"的特征解释框架。重新定位后：

- **共识部分**（no_exception）：所有模型都同意的基础标准——排斥翻译腔和偏好专业化词汇
- **分歧部分**（模型独有特征）：GPT-5.2 额外依赖语义相似度和句法简单度——这在 EC 方向被其他四个模型忽略

核心叙事线：**GPT-5.2 是个"语义稽核员"，其他四个是"语感裁判"。LLM-as-a-judge 的"共识"来源于共享的低维信号（翻译腔），而模型特有的训练偏好仍然在语义维度上产生可观测的分化。**

## 八、待定事项

- CE 方向尚未扩展（ECPCFE 数据约 20,000 段可用）
- 翻译腔分类器尚未训练（是否还有必要？如果叙事线转为模型对比，分类器的价值下降）
- LOO 特征重要性分析（已计划，未执行）
- API 密钥清理

## 九、关键文件

| 文件 | 内容 |
|------|------|
| `外部审稿意见.md` | 完整审稿意见 |
| `特征相关性分析.md` | Pearson + Spearman + VIF |
| `CLOUD_HANDOFF.md` | 云端工作交接文档 |
| `claude-version/ec/datasets/ffn_400ec.*.json` | 400 条数据集 |
| `kimi-version/ec/datasets/ffn_400ec.*.json` | 400 条数据集 |
| `data/analysis_summaries/` | 稳定性分层与发现摘要 |
