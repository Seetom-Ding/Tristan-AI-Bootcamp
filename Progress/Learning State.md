# AI Bootcamp Learning State

- **Last Updated**：2026-09-20（加入“前沿扫描—路线决策—学习状态”动态更新机制；Week 1–4 周评估仍为空）
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

## Confirmed Paper-reading Method

- 从 Week 5 开始逐步加入基础论文读写，优先使用 `AI Papers\LLM` 中与当周架构直接相关的本地材料。
- 先读英文原文的少量关键章节和图表，遇到语言阻塞再看中文翻译；自己的判断完成后再看解读材料。
- Week 5–8 以一篇基础论文的分段阅读、架构映射和短笔记为主，不要求完整复现，也不以读完页数作为掌握证据。
- 阅读产出至少区分 Problem、Motivation、Main Mechanism、Claim & Evidence、Limitation 和 Code Mapping；论文主张需要通过代码、实验或反例思考进行检验。

## Research-thinking Progress

- 当前阶段目标：从文本、token 与序列表示过渡到 causal self-attention，并通过基础论文读写建立架构、代码、主张与证据之间的联系。
- 当前证据：暂无

## Pace and Workload

- 待通过 Week 5 Reflection 确认

## Active Review Priorities

- 执行 `Study Plan\Week 5`，完成 Boss 后填写 `Week 5 Review.md`，用新的直接证据进行首次正式评估
- Coding 证据优先采集陌生代码理解、架构映射、关键修改、测试设计和有证据的调试过程
- Paper 证据从《Attention Is All You Need》的分段阅读与论文—架构实验笔记开始；不要求 Week 5 完整通读或复现
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

## Research Watchlist and Route Update

- **最近扫描**：2026-09-20；范围为 LLM inference、serving/runtime、KV cache、attention kernel 与系统—硬件协同；以 MLSys 2025/2026 proceedings 和 arXiv 一手论文为主。
- **扫描结论**：当前主线无需改变。近期工作仍集中在 KV-cache 与内存/IO、prefill/decode 与 TTFT/TPOT、请求调度，以及面向新硬件的 attention kernel 共设计，因此先修顺序“Transformer → 解码与 KV cache → 性能度量 → Serving/Runtime → Kernel”保持有效。
- **近期可追踪样本**：
  - [FlashInfer（MLSys 2025）](https://proceedings.mlsys.org/paper_files/paper/2025/hash/dbf02b21d77409a2db30e56866a8ab3a-Abstract-Conference.html)：把 KV-cache 布局、可组合 attention kernel 和动态调度放在同一推理引擎中；适合 Week 9+ 在理解 KV cache、batching 和 GPU memory 后阅读。
  - [SOLA（MLSys 2025）](https://proceedings.mlsys.org/paper_files/paper/2025/hash/bc82dbfbfa43232be85b8d9838f49c3e-Abstract-Conference.html)：以 TTFT/TPOT 和 SLO attainment 为核心的状态感知调度；提示 Serving 学习不能只看平均吞吐，还要看请求级指标与公平性。
  - [FlashAttention-4（arXiv, 2026）](https://arxiv.org/abs/2603.05451)：针对 Blackwell 的异构硬件瓶颈进行算法—kernel 共设计；暂列后续 kernel 主线，先补 GPU memory hierarchy、roofline、Triton/CUDA 与 profiling。
  - [CacheFlow（arXiv, 2026）](https://arxiv.org/abs/2604.25080)：将长上下文 KV-cache 恢复视为跨 token/layer/GPU 的计算—IO 调度问题；暂列长上下文与缓存恢复专题，不提前挤占 Week 5–8 基础。
  - [SHIP（MLSys 2026）](https://proceedings.mlsys.org/paper_files/paper/2026/hash/9c20f16b05f5e5e70fa07e2a4364b80e-Abstract-Conference.html)：展示 SRAM-based 大规模推理管线与内存带宽瓶颈；作为硬件—系统协同的远期案例，不作为当前必读。
- **本轮路线决策**：Week 5–8 不改核心内容；从 Week 8 系统桥开始强化 TTFT/TPOT、prefill/decode、KV-cache memory accounting 和可复现实验。Week 9+ 优先安排 FlashInfer/SOLA 类型的 Serving 论文卡，再根据掌握度进入 kernel 与长上下文缓存专题。
- **暂缓项**：需要大规模硬件、复杂分布式部署或尚未有可复现实物证据的热点；保留在 watchlist，不因标题新颖而改变下一周负荷。
- **下次扫描规则**：创建下一份 Week 5+ 周计划前重新检索；每次只选一个当前可行动的新增阅读，并记录“为什么现在、先修缺口、验证方式”。

## Long-term Direction

- AI Systems Research
- AI Systems PhD
- AI Infrastructure Engineer / Researcher
- Interests: LLM inference and serving, runtime systems, GPU and memory optimization, distributed training, AI compilers

# Weekly Assessment Log

尚无已完成评估。
