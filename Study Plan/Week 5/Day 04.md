# Audit Scaled Dot-Product Self-Attention

⭐ **Difficulty**：★★★★☆

💎 **Reward**：+90 XP

🏅 **Current Rank**：AI Novice   **XP**：210 / 500

## 上午（09:00--11:30）

### 网课

- 阅读 [CS224N Self-Attention and Transformers Notes](https://web.stanford.edu/class/cs224n/readings/cs224n-self-attention-transformers-2023_draft.pdf) 中 self-attention、Query、Key、Value 与 scaled dot-product attention 的部分。

- 阅读本地 [Attention Is All You Need 英文原文](<D:/学习/信息研一学习资料/AI Bootcamp/AI Papers/LLM/1_Attention_Is_All_You_Need/1_Attention_Is_All_You_Need.pdf>) 的 Figure 1、Figure 2 与 Section 3.2.1（PDF 第 3–4 页），重点建立“整体架构—attention 公式—代码实现”的对应关系。

### 官方教程

- 查阅 [PyTorch `softmax`](https://docs.pytorch.org/docs/stable/generated/torch.softmax.html)，明确 attention 权重沿哪个维度归一化。

- 浏览 [PyTorch Scaled Dot Product Attention Tutorial](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html) 的公式与接口部分；重点建立“数学公式—代码行—Tensor shape”的映射。

## 下午（15:00--17:30）

### Coding Lab

- 让 Codex 生成一个可读的单头 self-attention 骨架、shape 断言和测试，但将 scaled scores、缩放、softmax 与 Value 聚合四个核心表达式留空。

- 先根据论文公式独立补全这四个表达式，再逐行审查 `Q/K/V`、scores、weights、output 的 shape，并说明 softmax 为什么必须沿 key 维度计算。

- 在关闭 dropout 的条件下，与 `torch.nn.functional.scaled_dot_product_attention` 对照；将论文公式逐项映射到代码行。若结果不一致，先用 shape、缩放和归一化维度定位，再向 Codex 请求提示而不是完整答案。

## Challenge

给你一段能运行但把 softmax 写在错误维度的 attention 代码，你怎样只看 attention matrix 的统计性质发现问题？写出判断后再构造测试。

## Today's Checklist

- [ ] 完成 Figure 1、论文公式、代码行与 Tensor shape 的对应图

- [ ] 独立补全四个核心矩阵表达式并解释每一步

- [ ] 通过断言、官方算子和 attention matrix 三种证据验证实现
