# Trace Token Embeddings and Position

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

- 让 Codex 生成一个刻意保持简洁的 `TokenPositionEmbedding` 模块和最小测试；在运行前标注每行输入输出 shape、参数表大小和广播发生的位置。

- 阅读实现后，用自己的话解释 token id、查表、position ids、向量相加以及 `[B, T, C]` 输出之间的关系；删除任何无法解释的封装。

- 修改代码加入 `T <= max_context` 断言和“关闭 position embedding”开关；检查同一 token 在不同位置的表示，并通过一次 backward 审查两张 embedding 表的梯度。

## Challenge

不运行代码，先预测交换两个 token 后，开启与关闭 position embedding 时各个中间 Tensor 如何变化；再用程序验证，解释预测不一致之处。

## Today's Checklist

- [ ] 为 embedding 模块完成逐行 shape 与参数量标注

- [ ] 能不看说明解释 `[B, T] → [B, T, C]` 的完整数据流

- [ ] 完成位置开关修改，并用样例与梯度验证其行为
