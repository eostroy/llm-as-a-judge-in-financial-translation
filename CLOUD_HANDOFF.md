# 云端工作交接

## 当前状态

项目已完成首轮 200 条/方向的分析，现在要把数据规模从每方向 200 条翻倍到 400 条，并在 EC 方向引入基于分类器的翻译腔度量方式。

## 做了什么（本轮对话中）

1. **审稿意见**：完成了外部审稿意见（见 `外部审稿意见.md`），审稿人对自排除评审集分析、翻译腔子特征替换、人工 Top-1 对齐率等新增内容给予了肯定。

2. **翻译腔子特征**：当前使用 6 个手写语言学规则的子特征（功能词密度、被动密度、句首代词比、连接词密度、名物化后缀密度、平均句长），在 EC 方向全部 8/8 跨版本跨模型显著（no_exception 层级）。

3. **删除了两个 GPT-2 模型**：
   - `gpt2-chinese-cluecorpussmall`（1.2 GB）
   - `distilgpt2`（3.0 GB）
   - 准备用更大的模型替代（如 Qwen2.5-0.5B）做目标语自然度打分

4. **确定了扩展计划**：
   - EC：200 → 400 条（从 FFN Excel body+title sheets，约 1822 条原始数据可用）
   - CE：200 → 400 条（从 ECPCFE TMX 文件，约 20000 段可用，需要质量筛选）
   - 优先做 EC，其次是 CE

## 待执行：数据规模翻倍

### 核心原则
- 已生成的 200 条数据全部保留，只新增 200 条
- 全量分析结果在 400 条全集上重跑

### 步骤概览
1. **构建新增源文数据集**（id 201–400）
   - EC：从 `claude-version/data/raw/FFN_corpus_both_news_title_and_body.xlsx` 的 body sheet 取未使用的行
   - CE：从 `claude-version/data/raw/ecpcfe_untagged/` 过滤新的段落
2. **生成候选译文**（新增 200 条 × 2 方向 × 2 版本 = 800 次 API 调用）
   - Kimi-version 用 Kimi K2.5 API
   - Claude-version 用 Claude Sonnet 4.6 API（OpenRouter）
3. **五个评审模型盲评排序**（新增 200 条 × 5 模型 = 1000 次 API 调用）
4. **合并新旧数据到 400 条版本**
5. **全量重跑特征提取**（本地 6 个模型）
6. **全量重跑分析管线**（一致性、联合回归、自排除、稳定性分层）

## 待执行：翻译腔分类器（方式三）

### 逻辑
训练一个二分类器区分"翻译中文"和"原生中文"，用 P(translated) 作为翻译腔分数，替代当前的手写规则子特征。

### 训练数据
- **正例（翻译中文）**：EC 方向所有候选译文（200 条 × 3 候选 × 2 版本 = 1200 条，翻倍后 2400 条）
- **负例（原生中文）**：THUCNews 财经子集 + Kenpache 中文财经语料（需过滤，只保留报道中国国内经济的文章）
- 正负例各 1200–2400 条即可

### 模型
`hfl/chinese-roberta-wwm-ext`，HuggingFace 直接加载，fine-tune 3 epochs

### 训练完怎么用
分类器输出的 P(translated) 作为独立连续特征进入联合回归，替换六个手写子特征。保留两套方案做对比（旧的手写规则 vs 新的分类器分数），作为方法学贡献。

### 注意事项
- 负例不能有翻译腔：过滤 THUCNews 中可能从英文翻译的国际财经新闻
- 长度对齐：候选译文平均约 120 字，负例也要 80–200 字
- 领域对齐：财经新闻风格
- 防止分类器偷懒：翻译端用两个不同生成器（Kimi + Claude），原生端用多个来源（THUCNews + Kenpache）

## 未执行的步骤

以下步骤尚未开始：
1. 构建新增源文数据集
2. 生成候选译文
3. 评审模型排序
4. 翻译腔分类器训练数据准备
5. 翻译腔分类器训练
6. 下载更大的目标语 LM（替代已删除的 GPT-2）
7. 全量重跑特征提取
8. 全量重跑分析管线

## API 密钥安全

项目中多个脚本的 API 密钥以明文写死在代码中（OpenRouter 和 DeepSeek），尚未清理。在云端运行时，建议通过环境变量读取密钥，不要复用代码中的硬编码密钥。

## 关键文件位置

- 项目根目录：`~/llm-as-a-judge/`（或 AutoDL 上的对应路径）
- EC 原始数据：`claude-version/data/raw/FFN_corpus_both_news_title_and_body.xlsx`
- CE 原始数据：`claude-version/data/raw/ecpcfe_untagged/Nobel-Chinese/*.txt` 和 `Nobel-English/*.txt`
- 分析脚本（自排除等）：`scripts/shared/`
- 版本分析脚本：`claude-version/scripts/` 和 `kimi-version/scripts/`
- 组织化输出：`data/organized_analysis_outputs/`
- 外部基线语料：`data/external_baselines/`
- Kenpache 中文：`data/external_baselines/zh_native_finance_kenpache/clean.txt`
- Kenpache 英文：`data/external_baselines/en_native_finance_kenpache/clean.txt`

## 本地模型清单（云端可能需要重新下载）

- LaBSE（8.8 GB）— 跨语言语义相似度
- mDeBERTa-v3-base-mnli-xnli（552 MB）— 跨语言 NLI
- Stanza 中文 + 英文 — 依存句法
- 目标语 LM（已删除，需重新下载更大的替代模型）
