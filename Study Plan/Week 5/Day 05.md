# Test Causality and Future Leakage

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

- 先写因果性测试规格：允许变化与必须保持不变的输出位置、容差、至少两个边界情况；再让 Codex 为 Day 04 模块加入 causal mask 和测试骨架。

- 审查 mask 的方向、应用时机、填充值和 broadcast shape；运行“修改未来 token 不影响过去输出”的测试，并解释测试为何能检测泄漏。

- 让 Codex 分别注入“mask 方向反了”和“softmax 后才 mask”两个错误，不提前查看答案；依靠 attention matrix、因果性测试和 NaN 检查定位并修复，再连接最小 next-token 模型。

## Challenge

如果因果性测试通过，是否就能证明整个语言模型没有未来信息泄漏？列出数据构造、target shift 和 generation 中仍需检查的路径。

## Today's Checklist

- [ ] 能从代码解释 causal mask 的方向、时机、数值和 shape

- [ ] 因果性测试能通过正确实现并捕获两类错误实现

- [ ] 完成两次有证据的 bug 定位，并追踪完整 forward、loss 与 generation 路径
