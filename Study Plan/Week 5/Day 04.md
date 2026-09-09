# Audit Scaled Dot-Product Self-Attention

⭐ **Difficulty**：★★★★☆

💎 **Reward**：+90 XP

🏅 **Current Rank**：AI Novice   **XP**：210 / 500

## 上午（09:00--11:30）

### 网课

- 阅读 [CS224N Self-Attention and Transformers Notes](https://web.stanford.edu/class/cs224n/readings/cs224n-self-attention-transformers-2023_draft.pdf) 中 self-attention、Query、Key、Value 与 scaled dot-product attention 的部分。

- 阅读 [CS224N 2026 Assignment 3](https://web.stanford.edu/class/cs224n/assignments_w26/a3.pdf) 的 Attention Exploration，重点建立“按相似度从 Value 聚合信息”的直觉，不查或复制答案。

### 官方教程

- 查阅 [PyTorch `softmax`](https://docs.pytorch.org/docs/stable/generated/torch.softmax.html)，明确 attention 权重沿哪个维度归一化。

- 浏览 [PyTorch Scaled Dot Product Attention Tutorial](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html) 的公式与接口部分；重点建立“数学公式—代码行—Tensor shape”的映射。

## 下午（15:00--17:30）

### Coding Lab

- 让 Codex 生成一个可读的单头 self-attention 骨架、shape 断言和测试，但将 scaled scores、缩放、softmax 与 Value 聚合四个核心表达式留空。

- 先根据公式独立补全这四个表达式，再逐行审查 `Q/K/V`、scores、weights、output 的 shape，并说明 softmax 为什么必须沿 key 维度计算。

- 在关闭 dropout 的条件下，与 `torch.nn.functional.scaled_dot_product_attention` 对照；若结果不一致，先用 shape、缩放和归一化维度定位，再向 Codex 请求提示而不是完整答案。

## Challenge

给你一段能运行但把 softmax 写在错误维度的 attention 代码，你怎样只看 attention matrix 的统计性质发现问题？写出判断后再构造测试。

## Today's Checklist

- [ ] 完成公式、代码行与 Tensor shape 的三向映射

- [ ] 独立补全四个核心矩阵表达式并解释每一步

- [ ] 通过断言、官方算子和 attention matrix 三种证据验证实现
