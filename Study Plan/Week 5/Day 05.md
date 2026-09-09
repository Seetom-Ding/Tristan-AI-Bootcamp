# Test Causality and Future Leakage

⭐ **Difficulty**：★★★★★

💎 **Reward**：+100 XP

🏅 **Current Rank**：AI Novice   **XP**：300 / 500

## 上午（09:00--11:30）

### 基础论文精读

- 阅读本地 [Attention Is All You Need 英文原文](<D:/学习/信息研一学习资料/AI Bootcamp/AI Papers/LLM/1_Attention_Is_All_You_Need/1_Attention_Is_All_You_Need.pdf>) 的 Section 3.2.3、3.5 和 Section 4 / Table 1（PDF 第 5–7 页），找出 causal mask、位置编码、复杂度与顺序操作数的原始论述。

- 先独立写一页基础论文笔记：Problem、Motivation、Main Mechanism、Claim & Evidence、Limitation、Code Mapping；完成后再看同目录的 [学术型解读](<D:/学习/信息研一学习资料/AI Bootcamp/AI Papers/LLM/1_Attention_Is_All_You_Need/Transformer_深度解析_academic.html>) 修正遗漏，不照抄解读。

### 官方教程

- 查阅 [PyTorch `torch.tril`](https://docs.pytorch.org/docs/stable/generated/torch.tril.html)，用下三角矩阵构造 causal mask。

- 画出长度为 6 的 causal mask，逐行说明位置 `t` 可以关注哪些 key，以及 padding mask 与 causal mask 解决的问题为何不同。

## 下午（15:00--17:30）

### Coding Lab

- 先写因果性测试规格：允许变化与必须保持不变的输出位置、容差、至少两个边界情况；再让 Codex 为 Day 04 模块加入 causal mask 和测试骨架。

- 审查 mask 的方向、应用时机、填充值和 broadcast shape；运行“修改未来 token 不影响过去输出”的测试，并解释测试为何能检测泄漏。

- 让 Codex 分别注入“mask 方向反了”和“softmax 后才 mask”两个错误，不提前查看答案；依靠 attention matrix、因果性测试和 NaN 检查定位并修复，再连接最小 next-token 模型。

## Challenge

论文 Table 1 中 self-attention 的 sequential operations 是 `O(1)`，是否意味着计算量和运行时间也是 `O(1)`？结合 `O(n²·d)`、并行性与硬件条件解释。

## Today's Checklist

- [ ] 能从代码解释 causal mask 的方向、时机、数值和 shape

- [ ] 因果性测试能通过正确实现并捕获两类错误实现

- [ ] 完成两次有证据的 bug 定位和一页基础论文笔记
