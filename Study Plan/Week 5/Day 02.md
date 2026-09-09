# Token Embeddings and Position

⭐ **Difficulty**：★★★☆☆

💎 **Reward**：+70 XP

🏅 **Current Rank**：AI Novice   **XP**：60 / 500

## 上午（09:00--11:30）

### 网课

- 阅读 [CS224N 2026 Lecture 2：Word Vectors](https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture02-wordvecs.pdf) 中离散符号、分布式表示与 embedding 的部分，理解 one-hot 与稠密向量的差别。

- 区分 token id、token embedding、position embedding：id 是索引，embedding 是可学习表示，position 提供顺序信息。

### 官方教程

- 阅读 [PyTorch `nn.Embedding`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)，确认输入 dtype、输入输出 shape 和 embedding 权重 shape。

- 阅读 [CS224N 2026 Assignment 3](https://web.stanford.edu/class/cs224n/assignments_w26/a3.pdf) 的 Position Embeddings Exploration，只研究为什么无位置的 self-attention 无法区分序列顺序，不提交课程答案。

## 下午（15:00--17:30）

### Coding Lab

- 创建 token embedding 与 learned position embedding，将 `[B, T]` token ids 映射为 `[B, T, C]`。

- 用断言检查 `T` 不超过最大上下文长度，并在 forward 中标注每一步 Tensor shape。

- 对同一个 token 放在两个不同位置，分别检查 token 部分、position 部分和相加后表示；完成一次反向传播，确认两张 embedding 表都有预期梯度。

## Challenge

如果两个相同 token 出现在不同位置，它们进入模型的向量应该完全相同吗？分别讨论“没有位置表示”和“加入位置表示”时模型能与不能表达什么。

## Today's Checklist

- [ ] 完成 token embedding 与 learned position embedding 模块

- [ ] 从 `[B, T]` 正确得到 `[B, T, C]` 并通过 shape 断言

- [ ] 用梯度与样例验证 token 信息和位置信息都进入模型
