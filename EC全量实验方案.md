# EC 全量实验方案

## 模型配置

| 角色 | 模型 | API | 参数 |
|------|------|------|------|
| 候选生成 | Claude Sonnet 4.6 | OpenRouter | `max_tokens: 8192, temperature: 0.3` |
| 质检 | Kimi K2.5 | MoonShot | `thinking: disabled, max_tokens: 2048` |
| 评审 1 | GPT-5.5 | OpenRouter | `max_tokens: 2048, temperature: 0.3` |
| 评审 2 | Gemini 3.1 Flash Lite Preview | OpenRouter | `max_tokens: 2048, temperature: 0.3` |
| 评审 3 | DeepSeek V4 Flash | DeepSeek 官方 | `max_tokens: 2048, temperature: 0.3` |
| 评审 4 | Kimi K2.6 | MoonShot | `thinking: disabled, max_tokens: 2048` |

全部关闭思考模式。评审 temperature=0.3，用于保留低幅度判断波动（候选质量接近时确定性排序容易产生虚假一致）；每样本重复 3 次并重新打乱，用于估计模型内部一致性。生成模型与质检模型分离，生成模型与评审模型分离。

---

## 一、候选生成

**API**：OpenRouter `anthropic/claude-sonnet-4.6`

**方式**：一次 API 调用生成全部 6 个候选（A–F 同框），消除单独生成 vs 打包生成的 confound。

### 系统提示词

```
你是一名专业财经翻译专家。只返回JSON，不返回其他内容。
```

### 用户提示词

```text
你是一名专业财经翻译专家。请将以下英文源文翻译为 6 条中文译文。6 条译文必须：
- 语义准确，完整保留数字、时间、主体、涨跌方向、因果关系和专业术语。
- 整体质量相同，均可作为可交付的财经译文。差异仅来自语言组织方式。
- 每条译文独立完整，不得在译文中添加解释性括号或说明。

源文（英文）：
{source}

请生成以下 6 个版本：

A_baseline_balanced：
准确、自然、平衡的译文，符合中文财经报道常见写法。不过度意译、不刻意书面化、不刻意贴近源文结构。

B_source_syntax_preserving：
尽量保留源文的句法主干、信息顺序和修饰关系，但仍需符合中文基本表达习惯。

C_target_language_restructured：
按照中文财经文本习惯调整信息顺序。可以将英语中后置的原因、条件、时间状语提前，使表达更符合中文行文方式。调整应限于局部语序优化，不应大幅改变源文整体段落结构和信息推进节奏。

D_cohesion_explicit：
适度显化句内或句间逻辑关系，如因果、转折、递进、指代、概括等。不得添加源文没有的事实信息。

E_formal_register：
提高书面化程度和财经报道语域特征，使用更正式、更规范的现代中文词汇和表达。不得改变事实含义。必须全部使用现代中文，严格禁止出现任何英文词汇。严格禁止使用文言或近文言的表达（包括但不限于"之/彼/然/业已/遂/逮/哉/乎/矣/焉/耳/且夫/盖/伏惟/窃以为/岂/曷/讵/胡/奚"等），也不得使用"函/牍/兹/兹因/据此/查/呈/奉/尚"等旧式公文套话。可以用的正式表达包括：较正式的双音节词、财经领域术语、规范的书面句式，但整体必须读起来像现代财经新闻报道。

F_information_unpacking：
有针对性地拆分原文中特别密集的结构（如多重定语从句嵌套、长修饰链、复杂名词短语），使信息更易理解。适度分句即可，不需要将所有信息拆成细碎短句。不得添加源文没有的事实、背景、解释或过渡词。整体节奏应接近现代财经报道的正常行文，不要写成新闻简讯或儿童读物。

硬性要求：
1. 6 条译文质量必须非常接近，不能让某一条明显优于或劣于其他。
2. 差异必须主要来自指定语言特征，而非翻译质量本身。
3. 输出必须是合法 JSON。

输出格式：
{
  "variants": [
    {"id": "A", "feature_target": "baseline_balanced", "translation": "..."},
    {"id": "B", "feature_target": "source_syntax_preserving", "translation": "..."},
    {"id": "C", "feature_target": "target_language_restructured", "translation": "..."},
    {"id": "D", "feature_target": "cohesion_explicit", "translation": "..."},
    {"id": "E", "feature_target": "formal_register", "translation": "..."},
    {"id": "F", "feature_target": "information_unpacking", "translation": "..."}
  ]
}
```

---

## 二、质检

**API**：MoonShot `kimi-k2.5`

### 系统提示词

```
你是一名财经翻译质检员。只返回JSON，不返回其他内容。
```

### 用户提示词

```text
你是一名财经翻译质检员。请检查以下候选译文是否满足实验要求。

实验要求：
这些候选译文应整体质量接近，不能存在明显误译、漏译、数字错误、主体错误、时间错误、涨跌方向错误、因果关系错误或明显不自然表达。候选之间允许存在语言特征差异，但不能存在明显质量梯度。

源文：
{source}

候选译文：
{candidates_json}

请逐条检查：
1. 是否保留源文核心信息。
2. 是否存在数字、时间、主体、方向、术语错误。
3. 是否存在明显漏译或增译。
4. 是否明显低于其他候选质量。
5. 是否符合其目标语言特征。

输出 JSON：
{
  "overall_pass": true/false,
  "candidate_checks": [
    {
      "id": "A",
      "pass": true/false,
      "problems": [],
      "quality_gap": "none/minor/major",
      "feature_match": "high/medium/low"
    }
  ],
  "need_regeneration": ["..."],
  "reason": "..."
}
```

**质检规则**：只要出现以下情况，标记为不合格并整组重新生成：

- 数字、百分比、时间、主体错误
- 涨跌、增减、因果方向错误
- 明显漏译或增译
- 为特征差异牺牲自然度
- 明显比其他候选差
- 未体现目标特征

---

## 三、评审排序

**API**：
- GPT-5.5：OpenRouter `openai/gpt-5.5`
- Gemini 3.1 Flash Lite：OpenRouter `google/gemini-3.1-flash-lite-preview`
- DeepSeek V4 Flash：DeepSeek 官方 `deepseek-v4-flash`
- Kimi K2.6：MoonShot `kimi-k2.6`

### 系统提示词

```
你是一名专业财经翻译评审。只返回JSON，不要其他内容。
```

### 用户提示词

```text
你是一名专业财经翻译评审。请根据源文，对 6 条候选译文进行质量排序。

候选译文可能在表达方式上存在差异，但这些差异不必然代表质量高低，也不代表应该偏向有差异的表达。请优先判断译文是否准确、完整、自然、符合财经语境。若多条译文均准确自然，再根据整体语言组织选择更适合作为正式财经译文的一条。

源文：
{source}

候选译文：
T1: {translation_1}
T2: {translation_2}
T3: {translation_3}
T4: {translation_4}
T5: {translation_5}
T6: {translation_6}

请返回JSON对象，ranking数组包含T1到T6的完整排序，从最佳到最差。

例如：{"ranking": ["T3", "T1", "T5", "T2", "T6", "T4"]}
```

- A–F → T1–T6 匿名化，每样本每模型独立打乱
- **排序重复 3 次**，每次重新打乱 T1–T6
- 提示词中无任何特征标签或类型暗示
- **聚合方式**：Pairwise 回归用 3 次排序的多数 winner 确定每一对的方向（15 对/样本 → 24,000 对聚合后）；Plackett-Luce 用原始完整排序（3 次全保留，不聚合）
- 同时保留 3 次原始排序，用于计算模型内部 Kendall τ 一致性

---

## 四、特征提取

**核心原则：每个生成维度（B–F）至少 1–2 个可测文本指标支撑，标签不是唯一证据。**

### 设计标签（5 个）

| 变量 | 取值 |
|------|------|
| is_source_syntax | B=1 |
| is_restructured | C=1 |
| is_cohesion_explicit | D=1 |
| is_formal_register | E=1 |
| is_unpacking | F=1 |

A 全部取 0，作为参照组。

### 语义控制特征（5 个）

| 特征 | 来源 | 用途 |
|------|------|------|
| crosslingual_embedding_similarity | LaBSE (8.8 GB) | 排除语义偏离 |
| bidirectional_entailment | mDeBERTa (552 MB) | 排除遗漏风险 |
| nli_omission_risk | mDeBERTa | 排除遗漏风险 |
| nli_boundary_risk | mDeBERTa | 排除信息边界问题 |
| nli_contradiction_risk | mDeBERTa | 排除矛盾风险 |

### 句法特征（5 个）—— 对应 B（源文结构保留）和 C（目标语重组）

| 特征 | 来源 | 预期 |
|------|------|------|
| mean_dependency_distance | Stanza zh | B 的依存距离可能更接近英文源语结构 |
| dependency_tree_depth | Stanza zh | B/C 可能有不同的树深度 |
| longest_modifier_chain | Stanza zh | B 保留源文修饰链，可能更长 |
| clause_dependency_count | Stanza zh | C 重组可能改变从句结构 |
| coordination_count | Stanza zh | — |

### 衔接特征（3 个）—— 对应 D（衔接显化）

| 特征 | 计算方式 | 预期 |
|------|------|------|
| explicit_connective_density | 连接词次数 ÷ 总字数 × 100 | D 应最高 |
| reference_expression_density | 指代表达（这/那/该/此/其/上述/前者/后者）次数 ÷ 总字数 × 100 | D 应最高 |
| formal_reference_density | 正式指代（该/此/其/上述/前者/后者）次数 ÷ 总字数 × 100 | D 和 E 均可能偏高 |

### 语域特征（0 个）—— E（正式语域）暂不设可测指标

E 的 β4 识别依赖标签。正式语域的可操作化难度较高（双音节比受分词影响、财经书面词表覆盖面有限），全量先行验证标签效应，后续视需要补充。

### 信息解包特征（4 个）—— 对应 F（信息解包）

| 特征 | 计算方式 | 预期 |
|------|------|------|
| sentence_count | Stanza 解析出的句子数 | F 应最多 |
| avg_sentence_length | 总字数 ÷ 句子数 | F 应最短 |
| segment_count | 按 ，；：。！？ 切分后的有效片段数 | F 应最多 |
| avg_segment_length | 总字数 ÷ segment_count | F 应最短 |

### 总计

**5 标签 + 5 语义 + 5 句法 + 3 衔接 + 4 解包 = 22 特征**（E 暂无文本指标）

每个生成维度都有对应的文本特征作为独立验证，不再依赖纯标签效应。

---

## 五、回归分析

### Pairwise 设计

每条样本 6 候选 → 15 对。所有自变量取**候选之间的差值**：

```
Δfeature_k = feature_k(candidate_i) − feature_k(candidate_j)
```

### 分层模型

**模型 1（仅标签）**：
```
logit(P(i beats j)) = β0 + Σ β_k × Δlabel_k
```

**模型 2（加语义控制）**：
```
logit(P(i beats j)) = 模型1 + Σ β_k × Δsemantic_k
```

**模型 3（加句法 + 衔接 + 解包）**：
```
logit(P(i beats j)) = 模型2 + Σ β_k × Δsyntax_k + Σ β_k × Δcohesion_k + Σ β_k × Δunpacking_k
```

**模型 4（加 judge 交互项）**：
```
logit(P(i beats j)) = 模型3 + Σ Σ β_km × Δfeature_k × judge_model_m
```

**模型 5（Plackett-Luce）**：对完整排序做稳健性验证。

### 统计方法

- 主回归：`statsmodels.GLM` (Binomial family) 或 `statsmodels.Logit` + cluster-robust SE（按 sample_id 聚类）
- 稳健性复核：`sklearn.LogisticRegression` (L2) —— 用于验证系数方向一致性
- 手写梯度下降：仅用于代码复核，不用于主推断
- 跨模型汇总：统计每个特征在 4 模型中的显著性和系数方向
- 稳定性分层：no_exception / mostly_supported / no_stable_evidence

### 回归公式（完整写法）

```
logit(P(candidate_i wins over j)) =
    β0
  + β1·Δis_source_syntax
  + β2·Δis_restructured
  + β3·Δis_cohesion_explicit
  + β4·Δis_formal_register
  + β5·Δis_unpacking
  + β6·ΔLaBSE_similarity
  + β7·ΔNLI_features
  + β8·Δsyntax_features
  + β9·Δcohesion_features
  + β10·Δunpacking_features
  + Σ β_k·Δfeature_k × judge_model
  + ε,  clustered by sample_id
```

---

## 六、样本量与统计力

- 样本量 N = 400（EC 方向）
- 每条样本 15 对，3 次排序全部保留：15 × 3 = 45 对/样本
- 原始 pairwise 观察值：400 × 45 × 4 评审模型 = 72,000 对
- 聚合后（3 次排序取多数 winner）：400 × 15 × 4 = 24,000 对（主分析用聚合版）
- 聚类数：400（sample_id）

| 模型 | 自由参数（约） | 聚类数/参数比 | 可靠性 |
|------|------|------|------|
| M1（5 标签） | ~6 | 67:1 | 充分 |
| M2（+5 语义） | ~11 | 36:1 | 充分 |
| M3（+句法+衔接+语域+解包） | ~20 | 20:1 | 充分 |
| M4（+judge 交互项） | ~40 | 10:1 | **探索性**，交互项系数不用于主结论推断 |

M1–M3 为主结论模型，M4 为辅助探索。

---

## 七、执行规范

### 7.1 质检重生成

质检不合格时**整组 6 条重新生成**，不单独替换。该样本标记为二次生成版本，回归中做敏感性分析（排除后重跑）。

### 7.2 模型版本验证

启动前对每个评审模型调用 API 一次，确认返回 model 字段与预期一致。验证结果记录到运行日志。

### 7.3 特征共线性检查

特征提取完成后，对所有自变量跑 Pearson + Spearman + VIF。若有 VIF > 10 的特征对，主模型做拆分或排除处理。

### 7.4 回归实现

- 主回归：`statsmodels.GLM` (Binomial family) 或 `statsmodels.Logit` + cluster-robust SE（按 sample_id）
- 稳健性复核：`sklearn.LogisticRegression` (L2) —— 验证系数方向一致性
- 手写梯度下降：仅用于自查，不用于主推断
- Plackett-Luce：`choix` 库 `plackett_luce` 拟合原始完整排序，bootstrap 100 次估计 SE，与 pairwise 系数方向对比

### 7.5 质检锚定

全量完成后随机抽 40 条样本（10%），人工判断 6 候选质量是否接近。计算 Kimi QC 通过率与人工判断的一致率。

---

## 八、当前范围

本方案仅覆盖 **EC（英译中）**。中译英需要单独写候选生成提示词。
