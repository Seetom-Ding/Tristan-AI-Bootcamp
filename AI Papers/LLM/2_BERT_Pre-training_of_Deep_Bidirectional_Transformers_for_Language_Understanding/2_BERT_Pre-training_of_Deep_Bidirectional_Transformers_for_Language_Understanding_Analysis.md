# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding — 论文分析

> Devlin et al., Google, 2018。原文：[arXiv:1810.04805](https://arxiv.org/abs/1810.04805)。

## 先给结论

BERT 把“先在无标注文本上预训练，再用很小的任务头微调”变成 NLP 的标准范式。它的关键选择是使用 Transformer encoder 的**双向上下文**，并通过 Masked Language Model（MLM）避免普通左到右语言模型无法同时读取左右文的问题。另一个辅助目标 Next Sentence Prediction（NSP）试图让表示学习句间关系。

## 1. 研究问题与价值

2018 年，NLP 仍然经常为每个任务设计独立架构；ELMo 用双向语言模型生成上下文表示，但不同层的表示组合复杂；GPT 使用单向 Transformer，适合生成但不能在每一层同时看左右上下文。

问题是：能否从海量无标注文本预训练一个真正深度双向、可迁移的语言表示，并在问答、自然语言推理、分类等任务上只添加一个简单输出层？

价值是把昂贵的语言知识学习从每个下游任务中抽离出来。一个预训练模型可以复用于 GLUE、SQuAD、SWAG 等任务，显著降低标注需求和任务特定工程。

## 2. 之前方法的不足

- **任务专用监督学习**：每个任务都需要大量标签和专门架构。
- **ELMo**：双向信息来自两个独立单向 LM，使用方式和层间组合复杂。
- **GPT**：左到右目标使每个 token 看不到右侧上下文，不适合某些理解任务。
- **浅层特征迁移**：固定 embedding 或少量层不能充分利用深层上下文。

BERT 的统一 encoder 和端到端微调解决了工程碎片化，但代价是它不适合直接自回归生成。

## 3. 作者可能的思考路径

1. 下游理解任务需要某个词左右两边的证据，例如判断“bank”是银行还是河岸。
2. 普通语言模型若预测当前位置，不能看到答案 token 自己，否则目标泄漏；随机遮挡一部分 token，就可以让模型根据左右上下文恢复它。
3. 句子关系也影响问答和推理，于是把两个句子拼接起来，用 NSP 训练模型判断第二句是否真实跟在第一句后面。
4. 预训练得到通用参数后，下游任务只需一个线性层或 span 头；如果这成立，就能用同一模型覆盖很多 NLP 任务。

## 4. 核心 intuition

BERT 不要求模型生成完整文本，而要求模型在被遮住的地方“填空”。为了填对一个词，模型必须把左右上下文、句内语法和跨句关系压缩到每个位置的表示中。微调时，任务头读取这些上下文表示即可。

## 5. 方法与完整例子

输入句子对：

```text
[CLS] The animal did not cross the road [SEP]
because it was tired [SEP]
```

1. WordPiece 将文本切分成 token，加入 `[CLS]`、`[SEP]`、segment embedding 和 position embedding。
2. 预训练时随机选择 15% token：80% 换成 `[MASK]`，10% 换成随机词，10% 保持不变。
3. Transformer encoder 的每一层允许任意位置互相注意，因此被遮住的 `animal` 同时读取左右上下文。
4. MLM head 预测被选 token 的原词；NSP head 读取 `[CLS]`，判断第二句是真后续还是随机句。
5. 下游问答时，给定文章和问题，把答案起止位置作为两个分类问题；分类时只需在 BERT 顶部加线性层，并微调整个模型。

## 6. 核心数学

### MLM

给定被遮挡位置集合 \(M\)，目标为：

\[
\mathcal L_{MLM}=-\sum_{i\in M}\log p_\theta(x_i\mid x_{\backslash M}).
\]

模型根据未遮挡 token 预测原 token。由于只在 15% 位置计算损失，训练信号比标准 LM 稀疏；这是 BERT 的一个效率代价。

### NSP

令标签 \(y\in\{0,1\}\) 表示句子 B 是否是 A 的真实下一句：

\[
\mathcal L_{NSP}=-y\log p-(1-y)\log(1-p).
\]

总损失：

\[
\mathcal L=\mathcal L_{MLM}+\mathcal L_{NSP}.
\]

### 下游微调

分类使用 `[CLS]` 表示 \(h_{CLS}\)：\(p=\operatorname{softmax}(Wh_{CLS})\)。抽取式问答对每个上下文位置预测 start/end 概率。核心是共享同一个预训练 encoder。

## 7. 实验：问题 -> 设计 -> 答案

### 能否成为通用表示？

**问题**：一个预训练 encoder 是否能覆盖多种 NLP 任务？  
**设计**：在 GLUE、MultiNLI、SQuAD v1.1/v2.0、SWAG 等 11 个任务上端到端微调。  
**答案**：论文报告 GLUE 80.5、MultiNLI 86.7、SQuAD v1.1 F1 93.2、SQuAD v2.0 F1 83.1，刷新当时 SOTA。

### 双向 MLM 是否优于单向 LM？

**问题**：性能来自更大模型还是目标本身？  
**设计**：比较 MLM、单向 LM、不同遮挡比例和是否使用 NSP。  
**答案**：双向预训练明显更适合理解任务；随机遮挡和深层 encoder 贡献较大。

### 预训练规模是否重要？

**问题**：BERT Large 为什么更强？  
**设计**：比较 BERT Base（约 110M）和 Large（约 340M），并改变预训练步数。  
**答案**：更大模型和更长预训练通常提升下游性能，但收益取决于任务和数据。

### 局限是否暴露？

**问题**：它是否适合生成和长文本？  
**设计**：考察生成式任务、序列长度和 token masking。  
**答案**：BERT 是理解型 encoder，不是自然的开放式生成器；每次只对部分 token 产生训练信号，且预训练 `[MASK]` 与下游输入存在分布差异。

## 8. Takeaways

1. MLM 让深度 Transformer 可以利用左右上下文。
2. 预训练—微调范式比每个任务单独训练更高效。
3. `[CLS]`、segment embedding 和简单任务头形成了可复用接口。
4. BERT 的目标针对理解，不应直接等同于 GPT 的生成目标。

## 9. 最脆弱的假设

最脆弱的是：随机遮挡词能代表下游任务需要的语言理解，且预训练表示能自然迁移到所有任务。若任务需要生成、极长上下文或精确检索，MLM 的局部填空目标并不一定匹配。

## 10. 一周最小复现

在 WikiText 或小型中文语料上训练一个 4 层 Transformer encoder，随机 mask 15% token；再在 SST-2 或一个二分类数据集上微调。对比：随机初始化、单向 LM 预训练、MLM 预训练；记录验证准确率、收敛速度和 mask 比例的影响。这能验证迁移学习机制，不需要复现 BERT Large。

## 11. 反例设计

用一个需要逐字生成长答案的任务，比较 BERT encoder+额外生成器与 decoder-only LM；或者构造一个答案依赖超过预训练最大长度的跨段任务。如果 BERT 表示需要复杂桥接层才能工作，就说明“一个简单任务头即可迁移”的边界很明显。

## 12. Follow-up idea

设计**任务需求驱动的遮挡策略**：预训练前先估计任务需要的是局部语义、句间关系、长距离检索还是生成规划，动态选择 span mask、句级 mask、实体 mask 和左到右 mask 的混合比例。目标是在相同预训练计算下，让一个模型同时获得理解和生成所需的表示，而不是把所有任务都交给固定 15% token mask。实验应比较不同任务族的迁移、长上下文能力和生成质量。

## 参考资料

- [BERT 原论文](https://arxiv.org/abs/1810.04805)
- [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [T5](https://arxiv.org/abs/1910.10683)
