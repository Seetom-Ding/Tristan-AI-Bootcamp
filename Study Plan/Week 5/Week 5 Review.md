## Knowledge

### Q1

字符级、词级和子词级 tokenizer 如何在 vocabulary 大小、序列长度、未知词与跨语言适应性之间取舍？不要只列优缺点，请解释这些因素为何互相制约。

**我的回答：**

### Q2

token embedding 与 position embedding 分别提供什么信息？如果移除 position embedding，无位置的 self-attention 对输入顺序会表现出什么性质，为什么？

**我的回答：**

### Q3

写出 scaled dot-product attention 的计算流程，并解释除以 `sqrt(d_k)` 与 causal mask 各自解决什么问题；二者是否可以互相替代？

**我的回答：**

### Q4

《Attention Is All You Need》的 Table 1 同时写出 self-attention 每层复杂度 `O(n²·d)` 和 sequential operations `O(1)`。这两个结论为什么不矛盾？它们分别描述什么？

**我的回答：**

## Coding

### Task

由 Mentor 提供一段本周未见过的 causal self-attention 实现。请逐段解释输入、Q/K/V、scores、weights 与输出的数据流和 shape，指出至少一个潜在正确性风险，并通过修改代码或增加测试验证判断；不要求从空文件重写完整模块。

**我的回答 / 代码位置：**

## Debugging

### Scenario

你的 next-token 模型训练 loss 很快接近 0，但从一个短提示开始逐 token 生成时输出混乱。请提出至少四个假设，给出排查顺序、每一步要观察的证据，并设计能直接检测未来信息泄漏的测试。

**我的分析：**

## Thinking

### Question

结合《Attention Is All You Need》提交一份简短论文—架构实验笔记：概括 Problem、Motivation、Main Mechanism、Claim & Evidence、Limitation、Code Mapping；再选择论文关于 RNN 与 self-attention 的一项主张，设计公平 baseline、控制变量、指标和消融实验，并说明什么结果会反驳你的判断。

**我的回答：**

## Reflection

- 本周最重要的概念变化：
- 最有价值的一次失败或调试：
- Codex 负责了什么、我亲自验证了什么：
- 论文中目前仍不理解或不确信的主张：
- 目前仍无法独立实现的环节：
- 实际投入时间与节奏：

## Mentor Assessment

> 由 Mentor 在收到回答后填写。

- **Knowledge**：暂无证据
- **Coding**：暂无证据
- **Debugging**：暂无证据
- **Thinking**：暂无证据
- **已掌握**：
- **薄弱点**：
- **常见错误**：
- **科研产物**：待提交《Attention Is All You Need》基础论文—架构实验笔记
- **复习建议**：
- **下一周调整**：依据本次评估决定 Week 6 的 Transformer Block 深度与补强内容。
