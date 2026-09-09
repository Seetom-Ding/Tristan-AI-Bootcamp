# Language Models and the RNN Bottleneck

⭐ **Difficulty**：★★★☆☆

💎 **Reward**：+80 XP

🏅 **Current Rank**：AI Novice   **XP**：130 / 500

## 上午（09:00--11:30）

### 网课

- 学习 [CS224N 2026：Language Models and RNNs](https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture04-rnnlm.pdf)，重点理解自回归分解、hidden state 与时间步依赖。

- 选读 [CS224N Language Models, RNN, GRU and LSTM Notes](https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes05-LM_RNN.pdf) 中 RNN 的梯度与长依赖部分；LSTM、GRU 只理解门控为何缓解问题，不展开成独立主线。

### 官方教程

- 查阅 [PyTorch `nn.RNN`](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html)，画出输入、输出与 hidden state 的 shape 关系。

- 比较 RNN 与 CNN/MLP：参数共享发生在哪个维度，为什么时间步之间的依赖限制并行计算。

## 下午（15:00--17:30）

### Coding Lab

- 使用 Day 01 的字符级 batch 与 Day 02 的 embedding，搭建一个单层 `nn.RNN` next-token baseline。

- 训练到 loss 明显低于随机猜测基线，记录 seed、上下文长度、参数量、train loss、validation loss 和一段生成文本。

- 分别用 `T=16、64、128` 完成前向计时，每种设置先 warm-up 再重复测量；只记录现象，不据此作硬件性能结论。

## Challenge

为什么 RNN 即使参数量很小，也难以在序列长度维度充分并行？区分“能批量处理多个样本”和“能并行处理同一样本的多个时间步”。

## Today's Checklist

- [ ] 完成可训练的字符级 RNN next-token baseline

- [ ] 记录训练、验证与生成结果以及关键 Tensor shape

- [ ] 完成三种上下文长度的重复计时并写出谨慎观察
