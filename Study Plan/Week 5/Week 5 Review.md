## Knowledge

### Q1

字符级、词级和子词级 tokenizer 如何在 vocabulary 大小、序列长度、未知词与跨语言适应性之间取舍？不要只列优缺点，请解释这些因素为何互相制约。

**我的回答：**

### Q2

token embedding 与 position embedding 分别提供什么信息？如果移除 position embedding，无位置的 self-attention 对输入顺序会表现出什么性质，为什么？

**我的回答：**

### Q3

写出 scaled dot-product attention 的计算流程，并解释除以 `sqrt(d_k)` 与 causal mask 各自解决什么问题；二者是否可以互相替代？

**我的回答：**

## Coding

### Task

不参考本周代码，写出单头 causal self-attention 的核心 PyTorch 实现，标注输入、scores、weights 与输出 shape；再写一个最小测试，验证修改未来 token 不会改变过去位置的输出。

**我的回答 / 代码位置：**

## Debugging

### Scenario

你的 next-token 模型训练 loss 很快接近 0，但从一个短提示开始逐 token 生成时输出混乱。请提出至少四个假设，给出排查顺序、每一步要观察的证据，并设计能直接检测未来信息泄漏的测试。

**我的分析：**

## Thinking

### Question

提交一份简短架构实验笔记：比较本周 RNN baseline 与 causal self-attention 模型。说明要研究的限制或权衡、公平 baseline、必须控制的变量、主要指标、一个消融实验，以及什么结果会反驳你的原判断。

**我的回答：**

## Reflection

- 本周最重要的概念变化：
- 最有价值的一次失败或调试：
- 目前仍无法独立实现的环节：
- 实际投入时间与节奏：

## Mentor Assessment

> 由 Mentor 在收到回答后填写。

- **Knowledge**：暂无证据
- **Coding**：暂无证据
- **Debugging**：暂无证据
- **Thinking**：暂无证据
- **已掌握**：
- **薄弱点**：
- **常见错误**：
- **科研产物**：待提交 RNN 与 causal self-attention 架构实验笔记
- **复习建议**：
- **下一周调整**：依据本次评估决定 Week 6 的 Transformer Block 深度与补强内容。
