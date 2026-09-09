# From Text to Tokens

⭐ **Difficulty**：★★☆☆☆

💎 **Reward**：+60 XP

🏅 **Current Rank**：AI Novice   **XP**：0 / 500

## 上午（09:00--11:30）

### 网课

- 浏览 [Stanford CS224N 2026 课程主页与课程表](https://web.stanford.edu/class/cs224n/)，只关注 History of NLP、Word Vectors、Language Models and RNNs、Transformers 四个节点，画出本阶段概念路线。

- 理解语言模型的输入为什么不能直接是字符串：文本需要经过 normalization、tokenization、token-to-id，才能成为形状为 `[B, T]` 的整数 Tensor。

### 官方教程

- 阅读 [Hugging Face Tokenizers Quicktour](https://huggingface.co/docs/tokenizers/quicktour)，重点理解 vocabulary、special token、encode、decode、padding 与 attention mask；今天不依赖库完成核心实现。

- 在同一组中英文与代码样例上比较字符级、词级和子词级切分，记录 vocabulary 大小、序列长度、未知词与可逆性差异。

## 下午（15:00--17:30）

### Coding Lab

- 选择一个小型纯文本语料，清洗后划分 train / validation，并记录随机种子与划分规则。

- 不使用 tokenizer 库，实现字符级 `encode(text) -> ids` 与 `decode(ids) -> text`，加入至少一个未知字符处理策略。

- 编写 batch 构造函数：从 token ids 采样长度为 `T` 的输入 `x`，并令目标 `y` 为向后移动一位的 token；用断言检查二者均为 `[B, T]`、dtype 为 `torch.long`。

## Challenge

面对同时包含中文、英文、emoji 和 Python 代码的语料，字符级与子词级 tokenizer 会怎样影响 vocabulary、上下文长度和未知字符？先写出判断，再说明需要什么数据才能验证。

## Today's Checklist

- [ ] 完成字符级 vocabulary、encode 与 decode，且往返测试通过

- [ ] 生成形状正确的 next-token 输入与目标 batch

- [ ] 写出三种 token 粒度的对比表与一条待验证判断
