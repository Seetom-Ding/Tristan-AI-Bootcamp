🏆 **Reward：+100 XP**

完成一个全新的项目（不要参考之前的代码）：

- 自选小型文本语料，实现字符级 vocabulary、encode、decode 与 next-token batch

- 实现 token embedding 和 learned position embedding，并标注全部关键 Tensor shape

- 不使用 `nn.MultiheadAttention`，从零实现单头 scaled dot-product self-attention

- 正确加入 causal mask，并用“改变未来 token 不影响过去输出”的单元测试排除信息泄漏

- 接入输出投影与训练循环，使 validation loss 低于随机猜测基线，并生成一段样本文本

- 固定 seed，记录参数量、上下文长度、train / validation loss 与至少一次失败实验

- 写一段结论：当前模型相对 RNN baseline 的优势、局限，以及 Week 6 加入多头、FFN、残差和 LayerNorm 后最希望验证的假设

成功完成 Boss、提交 Week 5 Review，且 Knowledge、Coding、Debugging、Thinking 四个维度均获得可用证据后：

## 🎉 Rank Up

**AI Novice** ↓ **Deep Learning Apprentice**

**Final XP：500 / 500**
