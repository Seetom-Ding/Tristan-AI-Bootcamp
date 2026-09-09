# Scaled Dot-Product Self-Attention

⭐ **Difficulty**：★★★★☆

💎 **Reward**：+90 XP

🏅 **Current Rank**：AI Novice   **XP**：210 / 500

## 上午（09:00--11:30）

### 网课

- 阅读 [CS224N Self-Attention and Transformers Notes](https://web.stanford.edu/class/cs224n/readings/cs224n-self-attention-transformers-2023_draft.pdf) 中 self-attention、Query、Key、Value 与 scaled dot-product attention 的部分。

- 阅读 [CS224N 2026 Assignment 3](https://web.stanford.edu/class/cs224n/assignments_w26/a3.pdf) 的 Attention Exploration，重点建立“按相似度从 Value 聚合信息”的直觉，不查或复制答案。

### 官方教程

- 查阅 [PyTorch `softmax`](https://docs.pytorch.org/docs/stable/generated/torch.softmax.html)，明确 attention 权重沿哪个维度归一化。

- 浏览 [PyTorch Scaled Dot Product Attention Tutorial](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html) 的公式与接口部分；今天先实现朴素版本，官方算子只用于结果核对。

## 下午（15:00--17:30）

### Coding Lab

- 从 `[B, T, C]` 输入出发，独立实现单头 `Q=XWq`、`K=XWk`、`V=XWv`、scaled scores、softmax weights 与加权求和，不调用 `nn.MultiheadAttention`。

- 在代码中写出并断言 `Q/K/V`、scores、weights、output 的 shape；检查每个 query 对应的 attention weights 之和接近 1。

- 在关闭 dropout 的条件下，与 `torch.nn.functional.scaled_dot_product_attention` 对照数值结果，并可视化一个样例的 attention matrix。

## Challenge

为什么 scores 要除以 `sqrt(d_k)`？从点积方差、softmax 饱和和梯度三个角度给出一条连贯解释，并提出一个能观察该现象的小实验。

## Today's Checklist

- [ ] 从零实现单头 scaled dot-product self-attention

- [ ] 所有 shape 与权重归一化断言通过

- [ ] 与官方算子完成数值核对并保存 attention matrix
