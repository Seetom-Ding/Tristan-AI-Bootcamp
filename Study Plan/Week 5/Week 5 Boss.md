🏆 **Reward：+100 XP**

完成一个全新的项目（不要参考之前的代码）：

- 先写最小字符级 causal attention 语言模型的架构规格，再让 Codex 生成新的工程骨架；在报告中标明哪些部分由 Codex 生成、哪些由你修改和验证

- 画出 Text → Token → Embedding → Causal Attention → Logits → Loss / Generation 数据流，并把每个模块映射到具体代码位置

- 逐模块审查 tokenizer、batch、embedding、attention、loss 与 generation loop；标注关键 Tensor shape、参数和不变量，删除无法解释的抽象

- attention 工程代码允许由 Codex 生成，但 scaled scores、缩放、softmax 与 Value 聚合四个核心表达式由你独立补全并说明

- 设计并运行 encode/decode、target shift、attention 权重归一化与因果性测试；让 Codex 注入一个 mask 或 shape 错误，依靠证据独立定位后修复

- 修改一个架构选项，例如关闭 position embedding，并在固定 seed、数据和训练预算下比较 validation loss、生成样例与行为差异

- 提交架构审查与实验结论：模型在做什么、为什么没有未来泄漏、Codex 曾生成什么问题、证据是否支持你的判断，以及 Week 6 最值得验证的假设

成功完成 Boss、提交 Week 5 Review，且 Knowledge、Coding、Debugging、Thinking 四个维度均获得可用证据后：

## 🎉 Rank Up

**AI Novice** ↓ **Deep Learning Apprentice**

**Final XP：500 / 500**
