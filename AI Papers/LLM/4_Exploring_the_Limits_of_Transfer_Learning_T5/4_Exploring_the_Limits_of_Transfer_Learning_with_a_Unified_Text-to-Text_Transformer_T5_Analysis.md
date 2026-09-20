# Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer（T5）— 论文分析

> Raffel et al., Google, 2019。原文：[arXiv:1910.10683](https://arxiv.org/abs/1910.10683)。

## 先给结论

T5 的主要贡献是把几乎所有 NLP 任务统一表示为 text-to-text：输入和输出都被写成文本，模型统一使用 encoder-decoder Transformer 生成目标字符串。论文同时系统比较预训练目标、架构、数据、迁移方式和规模，最终用 C4 语料和 T5 模型在摘要、问答、分类等任务上达到很强结果。

它解决的核心不是一个单点结构创新，而是回答“预训练系统中的哪些设计选择真正重要”，并提供一套统一接口和可复用实现。

## 1. 研究问题与价值

预训练 NLP 当时存在许多不一致：有的模型用 encoder、有的用 decoder，有的用语言模型目标、有的用 denoising；下游任务有分类、抽取、排序和生成不同头部。研究者很难判断差异来自架构、数据还是目标。

T5 问：能否把任务统一成文本到文本，并系统研究预训练配方，从而形成一个通用、简单、强大的迁移学习框架？

价值是降低任务专用代码和架构差异；任何任务都可写成 `输入文本 -> 目标文本`，模型、损失和推理接口保持一致。

## 2. 之前方法与不足

- BERT encoder 需要为分类、序列标注和 QA 编写不同输出头。
- GPT decoder 适合生成，但对双向理解和 span 抽取不够自然。
- 早期 seq2seq 预训练缺少系统的目标、数据和规模比较。
- 不同论文使用不同数据和预算，单看 benchmark 不能判断哪个组件有效。

## 3. 作者可能的思考路径

1. 分类标签也可以写成字符串，例如输入 `sentiment: I loved it`，输出 `positive`。
2. 抽取任务可以把答案 span 原样生成；翻译和摘要本来就是文本生成。
3. 如果所有任务都采用同一格式，就能消除任务头和训练接口差异。
4. 还需要一个能从大规模未标注文本学习的 denoising 目标；将连续 span 替换成 sentinel token，可以训练模型恢复缺失内容。
5. 最后系统扫过预训练目标、模型、数据、微调和多任务混合，寻找可推广的配方。

## 4. 核心 intuition

T5 把任务差异从“神经网络结构”移动到“输入前缀和输出字符串”。模型只需要学会条件生成。这样，分类、问答、摘要和翻译都变成同一个最大似然问题，工程简单，预训练表示又能直接服务生成任务。

## 5. 方法与完整例子

情感分类例子：

```text
输入: sentiment: The movie was unexpectedly moving.
目标: positive
```

1. 用 SentencePiece 对输入和目标分词。
2. 预训练阶段从 C4 文本中随机遮挡连续 span，例如 `The movie <extra_id_0> moving`。
3. encoder 读取带 sentinel 的损坏文本；decoder 输出 `<extra_id_0> was unexpectedly` 等被删掉的片段。
4. 下游微调时把任务名写进输入，目标就是标签或完整答案字符串。
5. 推理时自回归生成目标；分类不需要额外 softmax 头，只需比较生成标签。

T5 使用 encoder-decoder Transformer，base 约 220M 参数，large 770M，3B、11B 等更大配置；训练数据是约 750GB 的 Colossal Clean Crawled Corpus（C4）。

## 6. 数学推导

### Span corruption

给原文本 \(x\)，随机选择 span 集合 \(S\)，得到损坏输入 \(\tilde x\)。目标是最大化被删除 span 的条件概率：

\[
\mathcal L=-\sum_{t=1}^{|y|}\log p_\theta(y_t\mid \tilde x,y_{<t}),
\]

其中 \(y\) 由 sentinel token 和原 span 组成。相比 token-level MLM，span corruption 要求模型重建连续片段，更接近生成式下游任务。

### Text-to-text 统一目标

每个任务被写成输入 \(x^{task}\) 和目标文本 \(y^{task}\)：

\[
\theta^*=\arg\min_\theta -\log p_\theta(y^{task}\mid x^{task}).
\]

分类只是目标词很短，问答和摘要是较长序列；损失形式不变。

### 注意力复杂度

encoder self-attention、decoder masked self-attention 和 encoder-decoder attention 共同完成理解与生成。训练时可并行目标序列，推理时 decoder 仍自回归，故生成延迟随输出长度增加。

## 7. 实验：问题 -> 设计 -> 答案

### 哪个预训练目标最好？

**问题**：language modeling、MLM、span corruption 哪个更适合迁移？  
**设计**：在相近计算和模型规模下比较多种目标。  
**答案**：span corruption 配合 encoder-decoder 在整体迁移上表现强，且目标与生成任务更匹配。

### text-to-text 是否损失分类性能？

**问题**：统一接口会不会比专门分类头差？  
**设计**：将 GLUE、SuperGLUE、问答、摘要、翻译全部转成文本输入输出。  
**答案**：T5 在多类任务上达到或接近当时 SOTA，说明统一化没有牺牲通用性。

### 数据与规模的重要性

**问题**：性能主要来自模型结构还是数据、训练步数？  
**设计**：比较 C4 过滤、数据大小、模型规模、训练步数、任务混合和预训练/微调方式。  
**答案**：数据质量、训练充分度和规模都重要；单纯更换架构往往不如改善数据和训练预算稳定。

### 多任务训练

**问题**：是否应把多个监督任务一起训练？  
**设计**：不同任务采样比例，比较单任务微调和多任务混合。  
**答案**：合适的多任务混合可以改善泛化，但采样比例和任务冲突需要调节。

## 8. Takeaways

1. 统一 text-to-text 接口是 T5 最可迁移的工程思想。
2. 预训练目标、数据质量和训练充分度需要一起调优。
3. 生成式统一带来简单性，也让分类推理变成字符串生成，可能有格式和校准问题。
4. C4 的清洗与数据授权、重复和污染仍影响可复现性。

## 9. 最脆弱的假设

假设任何任务都能用自然语言字符串表达，并且 token-level 生成损失与任务指标一致。对于精确数值、结构化约束、长文档和需要可靠置信度的任务，这一假设可能不成立。

## 10. 一周最小复现

用一个小型 T5（例如 6 层）在 WikiText 上做 span corruption，然后在 SST-2、简单 QA 和摘要子集上统一 text-to-text 微调。对比专门分类头和生成标签的性能、训练代码复杂度、推理延迟，验证接口统一的收益与代价。

## 11. 反例设计

设计严格的结构化输出任务：输入是表格，输出必须满足 JSON schema 或精确算术约束。若 T5 的字符串生成常产生格式错误，而专用判别/约束解码器稳定，则说明“所有任务都适合 text-to-text”需要边界。

## 12. Follow-up idea

构建**可验证的 text-to-text**：模型仍使用统一输入输出，但在训练和解码时增加任务类型对应的约束验证器。验证器对数字、JSON、集合、代码和证明步骤进行结构检查，并把错误反馈转成下一轮条件。研究重点是保留统一接口，同时解决生成字符串与离散任务指标不一致的问题。

## 参考资料

- [T5 原论文](https://arxiv.org/abs/1910.10683)
- [C4 数据与 T5 代码](https://github.com/google-research/text-to-text-transfer-transformer)
- [BERT](https://arxiv.org/abs/1810.04805)
