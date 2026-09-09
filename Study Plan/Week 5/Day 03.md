# Read the RNN Language-Model Bottleneck

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

- 先画出 `token ids → embedding → RNN → logits → cross-entropy` 架构图，再让 Codex 生成与图一致、没有多余抽象的单层 `nn.RNN` next-token baseline。

- 逐段阅读 forward 与 generation loop：标注 input、output、hidden、logits 的 shape，指出 hidden state 如何把历史信息传到下一时间步；用断点或打印验证一次完整路径。

- 修改一个可观察行为，例如 hidden size 或上下文长度；训练到 loss 低于随机猜测基线，并对 `T=16、64、128` 重复计时，记录参数量、验证 loss、生成样例与谨慎结论。

## Challenge

在不看实现的情况下，预测 generation loop 为什么不能像训练 forward 那样一次得到所有未来 token；结合代码指出自回归依赖发生在哪一行。

## Today's Checklist

- [ ] 完成 RNN 语言模型架构图并与代码逐模块对应

- [ ] 能追踪 forward、loss 和 generation 的关键 Tensor shape

- [ ] 完成一次代码修改和三种上下文长度实验，并解释结果边界
