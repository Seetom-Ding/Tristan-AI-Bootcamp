# From Tokens to Causal Self-Attention

> **Week Mission**
>
> 从文本表示出发，借助 Codex 阅读、审查和修改模型代码，真正理解 causal self-attention 的架构、数据流与正确性边界。

# 本周目标

- 能画出 Text → Token → Embedding → Sequence Model → Logits → Loss 的完整架构与数据流

- 能阅读陌生实现，解释 token embedding、position embedding、RNN 与 attention 的职责和 shape 变化

- 能预测代码行为，用测试和中间结果验证 RNN 的顺序瓶颈与 attention 的信息聚合方式

- 只独立补全 self-attention 的核心矩阵表达式，并能审查、修改和调试其余由 Codex 生成的实现

- 完成《Attention Is All You Need》的分段初读，能区分论文问题、主张、证据、局限及其与代码的对应关系
