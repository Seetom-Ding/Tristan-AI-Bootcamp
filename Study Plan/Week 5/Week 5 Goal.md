# From Tokens to Causal Self-Attention

> **Week Mission**
>
> 从文本表示出发，理解序列建模的历史瓶颈，并独立实现不会泄漏未来信息的 causal self-attention。

# 本周目标

- 能把原始文本转换为可逆 token ids，并构造 next-token 训练 batch

- 理解 token embedding 与 position embedding 的不同职责和 shape 变化

- 能解释 RNN 的顺序计算与长依赖瓶颈，以及 attention 试图解决的问题

- 从零实现单头 scaled dot-product self-attention 与 causal mask

- 用因果性测试、数值核对和公平实验验证实现，而不只确认代码能运行
