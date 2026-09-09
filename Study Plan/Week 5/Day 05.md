# Causal Masking without Future Leakage

⭐ **Difficulty**：★★★★★

💎 **Reward**：+100 XP

🏅 **Current Rank**：AI Novice   **XP**：300 / 500

## 上午（09:00--11:30）

### 网课

- 回看 [CS224N 2026 Assignment 3](https://web.stanford.edu/class/cs224n/assignments_w26/a3.pdf) 中 decoder-only Transformer 与位置表示相关内容，明确 next-token 训练为何必须阻止访问未来 token。

- 画出长度为 6 的 causal mask，逐行说明位置 `t` 可以关注哪些 key，以及 padding mask 与 causal mask 解决的问题为何不同。

### 官方教程

- 查阅 [PyTorch `torch.tril`](https://docs.pytorch.org/docs/stable/generated/torch.tril.html)，用下三角矩阵构造 causal mask。

- 复查 [PyTorch Scaled Dot Product Attention Tutorial](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html) 中 `is_causal` 的行为，只用作验证自己的 mask。

## 下午（15:00--17:30）

### Coding Lab

- 为 Day 04 的单头 attention 加入 causal mask，在 softmax 前屏蔽未来位置，并检查输出不存在 NaN。

- 编写因果性测试：只修改某位置之后的 token，验证该位置及之前的输出保持不变；再故意移除 mask，确认测试能够发现泄漏。

- 把 token embedding、position embedding、causal attention 和输出投影连接为最小 next-token 模型，完成一次前向、loss 与 backward。

## Challenge

一个语言模型训练 loss 异常低，但逐 token 生成完全不可用。为什么“训练时看到了未来 token”是优先排查项？你会用什么最小测试证明或排除它？

## Today's Checklist

- [ ] 正确构造并应用 causal mask

- [ ] 因果性测试能通过正确实现并捕获无 mask 实现

- [ ] 最小 causal attention 模型完成 forward、loss 与 backward
