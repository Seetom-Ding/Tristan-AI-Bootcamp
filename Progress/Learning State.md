# AI Bootcamp Learning State

- **Last Updated**：2026-09-09（Week 5 已按学生确认的 Codex 辅助学习方式重构；Week 1–4 周评估仍为空）
- **Current Week**：Week 5
- **Week Status**：计划已创建并开启；属于未获得前四周掌握证据时的基线计划
- **Current Rank**：AI Novice
- **XP**：0 / 500
- **Roadmap Stage**：Stage 2 — Modern Deep Learning and LLM Foundations
- **Next-stage Readiness**：待验证

## Current Mastery Snapshot

| Dimension | Level | Confidence | Evidence |
|---|---:|---|---|
| Knowledge | 暂无证据 | — | 尚未完成周评估 |
| Coding | 暂无证据 | — | 尚未完成周评估 |
| Debugging | 暂无证据 | — | 尚未完成周评估 |
| Thinking | 暂无证据 | — | 尚未完成周评估 |

## Topics Mastered

- 暂无已验证条目

## Weak Concepts

- 待验证

## Common Mistakes

- 待验证

## Coding and Debugging Evidence

- 暂无证据

## Confirmed Learning Method

- Codex 可以生成工程骨架、重复代码与初始测试，不再把长篇手敲或 API 记忆作为主要学习成果。
- 学生主要负责理解模型架构与目标、追踪数据流和 Tensor shape、阅读陌生代码、预测行为、修改模块、定位错误并验证结论。
- 对 attention 核心矩阵运算、causal mask、KV cache 等高价值机制，保留一次最小独立补全或推导，用来检验是否真正理解。
- 评估重点是能否解释、审查、修改、调试和设计可信实验，而不是能否从空文件背写完整项目。

## Research-thinking Progress

- 当前阶段目标：从文本、token 与序列表示过渡到 causal self-attention，并开始进行架构权衡与公平实验。
- 当前证据：暂无

## Pace and Workload

- 待通过 Week 5 Reflection 确认

## Active Review Priorities

- 执行 `Study Plan\Week 5`，完成 Boss 后填写 `Week 5 Review.md`，用新的直接证据进行首次正式评估
- Coding 证据优先采集陌生代码理解、架构映射、关键修改、测试设计和有证据的调试过程
- Week 1–4 Review 均未填写；如后续发现先修缺口，在 Week 6 只加入针对性补强，不倒推低分

## Stage Transition Note

- 2026-09-09：学生明确要求直接开启 Week 5，因此当前周与路线阶段已切换。
- 该切换只表示开始新的学习阶段，不表示 Week 1–4 已通过掌握度评估。
- Week 5 采用 Codex 辅助、理解优先的基线：tokenization、embedding、RNN 历史瓶颈、单头 self-attention 与 causal mask；多头和完整 Transformer Block 留到 Week 6。

## Fixed Foundation Baseline

- Week 1：Tensor、Autograd、MLP、MNIST 与训练实验
- Week 2：CNN、CIFAR-10、数据增强与受控实验
- Week 3：ResNet、BatchNorm、正则化与训练稳定性
- Week 4：Transfer Learning、Fine-tuning 与阶段综合项目
- 状态：Week 1–4 主体内容已冻结；评估只能添加复习建议，不重排计划。

## Week 9+ Research Directions

| Priority | Direction | Difficulty | Compute | CCF-A Potential | AI Systems Fit | Recommendation |
|---:|---|---:|---:|---:|---:|---:|
| 1 | LLM Inference Optimization | 3/5 | 1/5 | 5/5 | 5/5 | 5/5 |
| 1 | LLM Serving / Runtime | 4/5 | 1/5 | 5/5 | 5/5 | 5/5 |
| 2 | FlashAttention / CUDA Kernel | 5/5 | 1/5 | 5/5 | 5/5 | 4/5 |
| 3 | Distributed Training | 4/5 | 3/5 | 5/5 | 5/5 | 3/5 |
| 3 | LLM Compression / Quantization | 3/5 | 2/5 | 4/5 | 4/5 | 3/5 |

- 主线：LLM Inference Optimization + LLM Serving / Runtime
- 第二主线：FlashAttention / CUDA Kernel
- 后续分支：Distributed Training、LLM Compression / Quantization
- 原则：Week 9+ 根据能力证据进入，不因方向已确认而跳过 Transformer、性能度量和实验设计基础。
- 课程安排：Week 5–8 选学 CS224N 的 Transformer / LLM 基础；Week 9+ 以 CS336: Language Modeling from Scratch 为主干，并按评估结果选择作业。

## Long-term Direction

- AI Systems Research
- AI Systems PhD
- AI Infrastructure Engineer / Researcher
- Interests: LLM inference and serving, runtime systems, GPU and memory optimization, distributed training, AI compilers

# Weekly Assessment Log

尚无已完成评估。
