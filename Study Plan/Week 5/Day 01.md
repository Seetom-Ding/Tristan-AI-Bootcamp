# Map the Text-to-Token Pipeline

⭐ **Difficulty**：★★☆☆☆

💎 **Reward**：+60 XP

🏅 **Current Rank**：AI Novice   **XP**：0 / 500

## 上午（09:00--11:30）

### 网课

- 浏览 [Stanford CS224N 2026 课程主页与课程表](https://web.stanford.edu/class/cs224n/)，只关注 History of NLP、Word Vectors、Language Models and RNNs、Transformers 四个节点，画出本阶段概念路线。

- 理解语言模型的输入为什么不能直接是字符串：文本需要经过 normalization、tokenization、token-to-id，才能成为形状为 `[B, T]` 的整数 Tensor。

### 官方教程

- 阅读 [Hugging Face Tokenizers Quicktour](https://huggingface.co/docs/tokenizers/quicktour)，重点理解 vocabulary、special token、encode、decode、padding 与 attention mask；关注各组件在完整数据流中的职责。

- 在同一组中英文与代码样例上比较字符级、词级和子词级切分，记录 vocabulary 大小、序列长度、未知词与可逆性差异。

## 下午（15:00--17:30）

### Coding Lab

- 先写一份交给 Codex 的实现规格：输入输出、`[B, T]` shape、未知字符策略、train / validation 划分、随机种子和必须通过的测试。

- 让 Codex 按规格生成字符级 vocabulary、`encode`、`decode` 与 next-token batch 骨架；逐函数说明它在流水线中的位置，不接受无法解释的代码。

- 在运行前预测普通文本、未知字符和空文本三类输入的输出；运行后补充或修改至少两个测试，检查可逆性、`x/y` 一位错位、shape 与 dtype。

## Challenge

Codex 生成的 tokenizer 在训练语料上往返测试通过，是否足以证明它能处理中文、英文、emoji 和 Python 代码？指出至少三个仍可能隐藏的失败边界及其测试方法。

## Today's Checklist

- [ ] 完成并能口头解释 Text → Token → ID → Batch 数据流图

- [ ] 审查 Codex 生成的 tokenizer 与 batch 代码，所有函数均能说明职责

- [ ] 补充至少两个边界测试，并写出三种 token 粒度的对比判断
