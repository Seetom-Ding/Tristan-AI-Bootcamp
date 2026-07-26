# LoRA: Low-Rank Adaptation of Large Language Models

论文：Edward Hu 等，ICLR 2022。

分析依据：

- 本地论文 PDF：[[LORA.pdf]]
- [arXiv v2](https://arxiv.org/abs/2106.09685)
- [Microsoft 官方代码](https://github.com/microsoft/LoRA)

本文使用四类标签区分信息性质：

- **【论文】**：论文明确声称或实验直接支持。
- **【前置文献】**：LoRA 之前已经发表的研究结论。
- **【推断】**：依据论文证据作出的合理解释。
- **【猜想】**：尚未充分验证的研究设想。

## 核心结论

LoRA 最关键的选择，是对微调产生的权重增量 $\Delta W$ 做低秩参数化，同时冻结原始权重 $W_0$。它压缩的是“模型需要怎样改变”，并没有压缩预训练模型本身。

---

## 1. 论文解决了什么问题？为什么重要？

### 1.1 研究问题

预训练模型有参数 $\theta_0$。面对摘要、分类、NL2SQL 等下游任务，传统全参数微调会把所有参数更新成：

$$
\theta_{\text{task}}=\theta_0+\Delta\theta_{\text{task}}
$$

每个任务都会产生一个与原模型同样大的新模型。

GPT-3 有 1750 亿参数，以 FP16 保存，仅一份权重就约 350GB。100 个任务各保存一份完整模型，需要约 35TB。

**【论文】** LoRA 想解决的问题是：

> 能否只学习一个极小的任务增量，使模型获得接近全参数微调的效果，同时减少训练显存、任务 checkpoint 大小和线上推理开销？

它同时关心三个资源：

1. **训练资源**：全参数微调需要保存全部梯度和 Adam 优化器状态。
2. **存储资源**：每个任务都保存完整 checkpoint，成本随任务数线性增长。
3. **服务资源**：任务切换需要加载不同的大模型；额外模块还可能增加推理延迟。

**【论文】** 在 GPT-3 175B 上，作者报告 LoRA 将可训练参数减少约 10,000 倍，将训练显存从约 1.2TB 降到 350GB，并在合并权重后不引入额外推理延迟。[论文摘要](https://arxiv.org/abs/2106.09685)

这里的“10,000 倍”是**可训练参数或任务 checkpoint** 的减少，不代表整个模型显存减少 10,000 倍。冻结的 175B 底座仍然需要存在。

### 1.2 解决问题的价值

如果 LoRA 成立，模型部署可以变成：

$$
\text{一个共享底座}+\text{许多几十 MB 的任务插件}
$$

它带来三类价值：
```

```
- 一个团队可以为几十个业务共享同一底座，只保存小型任务增量。
- 没有能力全量微调大模型的研究者也能进行下游适配。
- 任务插件可以快速加载、交换、组合，形成后来广泛使用的 PEFT 生态。

---

## 2. 之前有人解决过吗？不足是什么？

LoRA 没有发明“参数高效微调”。它解决的是已有方法在**质量、存储、训练成本、推理延迟**之间的组合矛盾。

| 方法 | 学习什么 | 优点 | LoRA 作者强调的不足 |
|---|---|---|---|
| 全参数微调 | 全部模型参数 | 表达能力最强 | 每个任务需要完整 checkpoint，梯度和优化器状态昂贵 |
| Partial FT / BitFit | 少数层或 bias | 实现简单、参数少 | 选择哪些参数依赖经验，表达能力可能不足 |
| Adapter | Transformer 层之间的小型 MLP | 冻结底座，每个任务只存 adapter | 增加网络深度和串行算子，可能增加线上延迟 |
| Prompt / Prefix Tuning | 连续虚拟 token 或每层 prefix activation | 参数极少，底座冻结 | 占用上下文，优化敏感，参数增加不一定改善性能 |
| 预训练权重低秩分解 | 用低秩矩阵表示原始权重 | 压缩模型 | 通常用于压缩或从头训练，没有直接解决冻结模型的任务适配 |

**【前置文献】** Houlsby Adapter 在 GLUE 上用每任务约 3.6% 的新增参数，达到距离全参数微调约 0.4% 的性能，说明“共享底座 + 小模块”已经可行。[Adapter 论文](https://arxiv.org/abs/1902.00751)

**【前置文献】** Prefix-Tuning 冻结语言模型，通过连续虚拟 token 控制生成；原论文报告只训练约 0.1% 参数即可取得有竞争力的表现，在其低数据生成实验中甚至优于全量微调。[Prefix-Tuning](https://aclanthology.org/2021.acl-long.353/)

**【论文】** LoRA 作者测得 Adapter 在 GPT-2 medium 上会增加延迟；在 batch size 为 1、序列长度为 128 时，两种 Adapter 分别增加约 20.7% 和 30.3%。Prefix 方法在论文的 GPT-3 实验中还表现出非单调性：增加虚拟 token 后性能可能下降。

**【推断】** LoRA 的实验不能证明 Prefix-Tuning 普遍不适合低数据。Prefix 原论文和 LoRA 论文使用了不同模型、任务及训练配置。LoRA 只能证明在它测试的 GPT-3 MNLI 设置中，LoRA 的低数据表现更好。

---

## 3. 重建作者可能的思考路径

以下是**【基于证据的重建】**，不是作者逐字描述的心理过程。

### 第一步：把微调看成“寻找增量”

预训练已经完成了绝大部分工作。下游微调真正学习的是：

$$
\Delta\theta=\theta_{\text{fine-tuned}}-\theta_0
$$

因此，每个任务完整保存 $\theta_0+\Delta\theta$ 很浪费，因为巨大的 $\theta_0$ 在所有任务间完全相同。

自然的问题变成：

> $\Delta\theta$ 是否比 $\theta_0$ 简单得多？

### 第二步：排除不符合系统需求的表示方式

若增加 Adapter MLP，任务参数确实变少，但网络更深了。若增加 prefix token，网络没有变深，却占用了输入位置。

于是可以提出一个工程约束：

> 任务增量最好能在训练时独立学习，在推理前直接写回原矩阵，使推理图保持不变。

线性权重增量满足这个条件：

$$
W_{\text{deployed}}=W_0+\Delta W
$$

### 第三步：从 intrinsic dimension 得到启发

**【前置文献】** Li 等发现，许多神经网络任务虽然生活在巨大的参数空间中，但在随机低维子空间中也能找到有效解。Aghajanyan 等进一步发现，语言模型微调具有很低的 intrinsic dimension；例如只优化约 200 个低维参数，经随机投影回 RoBERTa 的完整参数空间，也能在 MRPC 上达到全量微调约 90% 的表现。[Intrinsic Dimension 研究](https://arxiv.org/abs/2012.13255)

这意味着成功微调可能只需要少量独立方向。

### 第四步：寻找符合矩阵结构的低维表示

Transformer 的主要参数是二维矩阵。如果直接把整个 $\Delta\theta$ 投影到一个随机低维向量，虽然参数少，却难以利用硬件友好的矩阵乘法，也不方便分别控制不同层。

矩阵理论给出一个更自然的候选：

$$
\Delta W=BA
$$

其中 $A$ 和 $B$ 都很窄。只要中间维度 $r$ 很小，$\Delta W$ 的秩最多为 $r$。

### 第五步：检查候选是否同时满足工程约束

这个候选具有三个直接性质：

- 参数量从 $dk$ 降为 $r(d+k)$。
- 前向传播仍然是矩阵乘法。
- 训练完成后可以把 $BA$ 加进 $W_0$，恢复原始推理图。

因此，即使还不知道它是否有效，它已经是一个非常值得实验的假设。

需要看清这里的逻辑链：

> 已有研究证明的是“有效解可能存在于低维参数子空间”；LoRA 进一步猜测“每个权重矩阵的任务增量可能具有低矩阵秩”。

这两件事有关，但没有数学上的必然关系。这也是 LoRA 最关键、最脆弱的一步。

---

## 4. LoRA 的核心 intuition

一个 $d\times k$ 的完整更新矩阵可以改变大量彼此独立的方向。但一个具体下游任务可能只需要几种协调一致的改变。

LoRA 把更新写成：

$$
\Delta W=BA=\sum_{i=1}^{r}b_i a_i^\top
$$

每个 $a_i$ 从输入表示中检测一个方向，每个 $b_i$ 决定如何把这个方向写回输出。秩 $r$ 表示允许任务使用多少个独立的“改变通道”。

最简洁的理解是：

> 预训练模型已经拥有绝大多数能力。下游训练主要负责找出少数需要放大、抑制或重新组合的方向。

---

## 5. 具体方法与完整 pipeline

以论文中的 WikiSQL 任务为例。

### 5.1 输入和目标

输入：

```text
Table: Employee(name, join_year, department)
Question: How many employees joined after 2020?
```

目标：

```sql
SELECT COUNT(*) FROM Employee WHERE join_year > 2020
```

训练时把表结构和自然语言问题编码为 $x$，把 SQL token 序列作为 $y$。

### 5.2 修改模型

对 Transformer 每层的注意力投影矩阵，论文主要选择 $W_q$ 和 $W_v$。

原始计算为：

$$
q=W_q^0h,\qquad v=W_v^0h
$$

加入 LoRA 后：

$$
q=W_q^0h+\frac{\alpha}{r}B_qA_qh
$$

$$
v=W_v^0h+\frac{\alpha}{r}B_vA_vh
$$

其中：

- $W_q^0,W_v^0$ 来自预训练模型，全程冻结。
- $A\in\mathbb{R}^{r\times d}$ 负责降到 $r$ 维。
- $B\in\mathbb{R}^{d\times r}$ 负责映射回模型维度。
- $r\ll d$。

### 5.3 初始化

论文使用：

- $A$：随机初始化。
- $B$：初始化为零。

因此训练开始时：

$$
BA=0
$$

模型第一步的行为和原始预训练模型完全一致，不会因为随机 LoRA 参数突然破坏原模型输出。

### 5.4 训练

模型使用 teacher forcing，最大化目标 SQL 的条件概率：

$$
\max_{A,B}\sum_t
\log p_{\theta_0+\Delta\theta(A,B)}
(y_t\mid x,y_{<t})
$$

反向传播会经过冻结的底座，但优化器只更新 LoRA 的 $A,B$。

训练结束后，只保存这些小矩阵。

### 5.5 推理

推理前计算：

$$
W_q'=W_q^0+\frac{\alpha}{r}B_qA_q
$$

$$
W_v'=W_v^0+\frac{\alpha}{r}B_vA_v
$$

然后使用 $W_q',W_v'$ 正常推理。计算图中不再需要额外 LoRA 分支，因此不会引入额外推理层。

官方实现会在 `eval()` 时合并权重，在重新进入 `train()` 时撤销合并。[官方实现说明](https://github.com/microsoft/LoRA)

### 5.6 GPT-3 上的参数计算

GPT-3 175B 的模型维度 $d=12{,}288$，层数 $L=96$。若对每层的 $W_q,W_v$ 使用 $r=4$：

$$
N_{\text{LoRA}}=2L\cdot r(d+d)
$$

$$
=2\times96\times4\times24{,}576
\approx18.9\text{M}
$$

FP16 保存大约 36MB。完整 GPT-3 checkpoint 约 350GB，所以一个任务的适配参数确实缩小了约四个数量级。

---

## 6. 核心数学推导

### 6.1 什么是矩阵的秩？

矩阵的秩可以理解为它包含多少个独立变化方向。

若：

$$
W=BA,\quad
B\in\mathbb{R}^{d\times r},\quad
A\in\mathbb{R}^{r\times k}
$$

那么：

$$
\operatorname{rank}(BA)\le r
$$

因为信息必须经过只有 $r$ 维的中间空间。

### 6.2 参数为什么减少？

完整更新矩阵：

$$
\Delta W\in\mathbb{R}^{d\times k}
$$

参数量：

$$
dk
$$

LoRA：

$$
B\in\mathbb{R}^{d\times r},\quad
A\in\mathbb{R}^{r\times k}
$$

参数量：

$$
dr+rk=r(d+k)
$$

压缩比为：

$$
\frac{dk}{r(d+k)}
$$

当 $d=k=4096,r=8$ 时：

$$
\frac{4096^2}{8(4096+4096)}=256
$$

也就是单个矩阵的可训练参数减少 256 倍。

### 6.3 为什么低秩等于少数方向？

展开乘积：

$$
BA=
\begin{bmatrix}
b_1&b_2&\cdots&b_r
\end{bmatrix}
\begin{bmatrix}
a_1^\top\\
a_2^\top\\
\vdots\\
a_r^\top
\end{bmatrix}
$$

所以：

$$
BA=\sum_{i=1}^{r}b_i a_i^\top
$$

每一项 $b_i a_i^\top$ 都是 rank-1 更新。LoRA 把任务适配限制为最多 $r$ 个 rank-1 改变的叠加。

### 6.4 梯度如何流动？

记 LoRA 分支输出为：

$$
z=\frac{\alpha}{r}BAx
$$

损失对输出的梯度为：

$$
g=\frac{\partial\mathcal L}{\partial z}
$$

根据链式法则：

$$
\frac{\partial\mathcal L}{\partial B}
=
\frac{\alpha}{r}g(Ax)^\top
$$

$$
\frac{\partial\mathcal L}{\partial A}
=
\frac{\alpha}{r}B^\top g x^\top
$$

因为初始化时 $B=0$：

$$
\frac{\partial\mathcal L}{\partial A}=0
$$

但只要随机初始化的 $A$ 不为零，通常：

$$
\frac{\partial\mathcal L}{\partial B}\ne0
$$

因此第一步主要更新 $B$。当 $B$ 离开零点后，$A$ 也开始获得梯度。这解释了“随机 $A$、零 $B$”为什么既保持初始模型不变，又能开始学习。

这段梯度推导是**【基于论文方法的数学推导】**，论文没有把它完整展开。

### 6.5 $\alpha/r$ 的作用

论文使用：

$$
\Delta W=\frac{\alpha}{r}BA
$$

当改变 $r$ 时，未缩放的 $BA$ 的幅度和优化行为也会变化。除以 $r$ 可以降低不同 rank 之间重新调整超参数的需求。

**【论文】** 作者通常固定 $\alpha$，没有为每个 $r$ 单独搜索它。

**【后续文献】** 后来的 LoRA+ 指出，原始 LoRA 给 $A,B$ 使用相同学习率，在模型宽度很大时可能并非最优；它提出让 $B$ 的学习率显著高于 $A$。这表明原论文的优化参数化仍有改进空间。[LoRA+](https://arxiv.org/abs/2402.12354)

### 6.6 LoRA 和 SVD 的关系

SVD 告诉我们，一个矩阵最重要的变化可以按奇异值从大到小排列。保留前 $r$ 个方向会得到最佳 Frobenius 范数意义下的 rank-$r$ 近似。

但 LoRA 没有执行：

1. 先完成全参数微调；
2. 得到完整 $\Delta W$；
3. 再对 $\Delta W$ 做 SVD。

LoRA 直接训练 $A,B$，从一开始就把搜索空间限制在 rank-$r$ 矩阵集合中。因此它节省训练资源，而事后 SVD 只能节省存储。

### 6.7 论文有理论保证吗？

**【论文】** 原始 LoRA 没有证明真实语言模型的最优更新一定低秩。它提供的是经验支持：

- $r=1,2,4,8,64$ 的性能常常很接近。
- 不同 rank 学到的首要奇异方向有明显重合。
- 不同随机种子也会学到部分共同方向。
- $\Delta W$ 似乎主要放大 $W_0$ 中已有但没有被强调的方向。

因此，“下游更新具有低 intrinsic rank”在这篇论文中属于**经验假设**，不是定理。

---

## 7. 实验如何验证 claim？

### 问题一：LoRA 能否接近全参数微调？

**问题 →** 训练参数减少后，性能是否明显下降？

**实验 →** 在 RoBERTa、DeBERTa、GPT-2、GPT-3 上测试 GLUE、E2E、DART、WebNLG、WikiSQL、MNLI 和 SAMSum，比较全参数微调、BitFit、Adapter、Prefix 和 LoRA。

**答案 →** 在论文测试范围内，LoRA 通常与全参数微调持平或更好，同时使用少得多的任务参数。官方仓库给出了 RoBERTa、DeBERTa 和 GPT-2 的复现代码与 checkpoint。[官方结果](https://github.com/microsoft/LoRA)

这里的“更好”不说明 LoRA 的表达能力超过全参数微调。**【推断】** 低秩约束可能起到了正则化作用；全参数基线的超参数、数据规模和随机波动也会影响比较。

### 问题二：Adapter 的额外层真的有推理成本吗？

**问题 →** Adapter 参数虽少，额外延迟是否可以忽略？

**实验 →** 在 GPT-2 medium 上改变 batch size 和序列长度，测量单次 forward latency。

**答案 →** 长序列和大 batch 时额外延迟较小；batch size 为 1、短序列的线上场景中，作者测得约 20%–30% 的增加。

### 问题三：应该把 LoRA 放在哪些矩阵？

**问题 →** 固定参数预算时，集中提高一个矩阵的 rank，还是覆盖更多矩阵？

**实验 →** 在 GPT-3 上固定约 18M 参数，比较 $W_q,W_k,W_v,W_o$ 及其组合。

**答案 →** $W_q+W_v$ 整体表现最好；覆盖更多类型、使用较小 rank，常常优于只适配一个矩阵并提高 rank。

这支持“更新方向的覆盖范围可能比单点 rank 更重要”，但结果只来自 WikiSQL 和 MultiNLI。

### 问题四：多小的 rank 足够？

**问题 →** 性能是否会随 $r$ 明显增加？

**实验 →** 比较 $r=1,2,4,8,64$，同时改变被适配的矩阵。

**答案 →** 对论文中的 GPT-3 任务，$W_q+W_v$ 甚至在 $r=1$ 时已有竞争力，进一步增大 rank 收益很小。GPT-2 的最优值大致落在 4–16。

### 问题五：低 rank 真学到了稳定方向吗？

**问题 →** 小 rank 的成功是否只是偶然？

**实验 →** 对不同 rank、不同随机种子学到的矩阵做 SVD，测量顶层奇异子空间的重合程度，并与随机高斯矩阵比较。

**答案 →** 首要方向在不同 rank 和部分随机种子之间有明显重合，其余方向重合较弱。作者据此推测额外方向可能包含更多训练噪声。

“额外方向是噪声”属于作者的解释，不是实验直接观测到的事实。

### 问题六：LoRA 在低数据下表现如何？

**问题 →** 参数约束是否有利于样本效率？

**实验 →** 从 MNLI 采样 100、1K、10K 和完整训练集，在 GPT-3 上比较全量微调、Prefix 和 LoRA。

**答案 →** LoRA 在 100 样本和完整数据设置中优于论文的全量微调基线，在 1K 和 10K 设置中大致相当；Prefix 在 100 样本时明显较弱。

### 问题七：LoRA 能否与其他方法组合？

**问题 →** LoRA 和 Prefix 是否学习了不同类型的适配？

**实验 →** 组合 LoRA 与 Prefix-Embedding / Prefix-Layer。

**答案 →** LoRA + Prefix-Embedding 在 WikiSQL 上进一步提高，但在 MNLI 上没有提高；LoRA + Prefix-Layer 反而略差。论文只能支持“某些任务上具有互补性”。

---

## 8. Takeaways

1. LoRA 的研究对象是任务更新 $\Delta W$，原始权重 $W_0$ 始终保留。
2. 低秩分解把 $dk$ 个参数降成 $r(d+k)$，前提是 $r\ll d,k$。
3. $A$ 可以理解为识别少量任务方向，$B$ 把这些方向写回模型表示。
4. 线性结构允许部署前执行 $W_0+BA$，因此保持原始推理图。
5. 论文在多个模型和 NLP 任务上证明“小 rank 可以很好用”，没有证明所有任务更新都低秩。
6. “低 intrinsic dimension”与“每层更新矩阵低 rank”是两个不同概念，前者只是后者的灵感来源。
7. LoRA 大幅减少梯度、优化器状态和任务 checkpoint，仍然需要底座权重和反向传播 activation。
8. LoRA 的影响力很大程度来自它同时满足性能、易实现、可交换和可合并，而不只是参数量少。

---

## 9. 最脆弱的假设是什么？

最脆弱的假设是：

> 对目标任务有用的模型变化，可以在每个被选择的权重矩阵中，用一个固定且很小的 rank 表达。

这里包含两次未经证明的跳跃：

1. 整个微调问题的有效解位于低维参数子空间；
2. 这个低维性恰好表现为每个 $d\times k$ 更新矩阵的低矩阵秩。

一个向量化后的参数更新可能只有少数自由度，但这些自由度映射到某个矩阵后仍可能是满秩的。因此 intrinsic dimension 小，并不自动推出 matrix rank 小。

**【论文】** 作者自己在脚注中承认，小 rank 不会对所有任务有效。如果下游任务使用预训练没有覆盖的语言，完整重训可能明显优于小 rank LoRA。

其他脆弱点包括：

- $r$ 和目标矩阵主要靠经验选择。
- GPT-3 175B 实验无法由外部研究者完整复现。
- 部分基线数据来自前置论文，训练配置并非全部统一。
- 原论文集中于分类、结构化生成和摘要，没有覆盖现代 instruction tuning、复杂推理和跨模态适配。
- LoRA 减少了 optimizer memory，长序列训练中 activation 仍可能是主要瓶颈；后续 LoRA-FA 正是针对这一问题提出改进。[LoRA-FA](https://arxiv.org/abs/2308.03303)
- 后续 DoRA 发现 LoRA 与全参数微调之间仍可能有性能差距，并把权重的幅度和方向分开处理。[DoRA](https://arxiv.org/abs/2402.09353)

---

## 10. 一周内可以做什么最小复现实验？

建议验证最核心、同时最容易复现的一点：

> 在一个标准任务上，LoRA 的性能很快随 rank 饱和；合并权重后输出保持一致且没有稳定的推理延迟差异。

### 实验配置

- 模型：`roberta-base`
- 数据：SST-2
- 方法：
  - Full fine-tuning
  - LoRA $r\in\{1,2,4,8,16\}$
- LoRA 位置：每层 attention 的 query 和 value
- 指标：
  - 验证集准确率
  - 可训练参数量
  - 峰值显存
  - 每秒训练样本数
  - 合并前后 logits 最大误差
  - 合并后推理延迟
- 第一轮所有配置跑一个 seed；Full、$r=1$、$r=8$ 再补三个 seed。

### 一周安排

- 第 1 天：搭好数据、评估和显存测量脚本。
- 第 2 天：完成全参数微调基线。
- 第 3–4 天：完成 rank sweep。
- 第 5 天：补三个随机种子。
- 第 6 天：检查 merge 前后 logits，测量 latency。
- 第 7 天：绘制 accuracy–rank、accuracy–parameter、memory–method 曲线。

### 判定标准

若观察到：

- $r=4$ 或 $r=8$ 与全量微调差距小于约 1 个百分点；
- $r=8$ 到 $r=16$ 的收益已经很小；
- 可训练参数低于模型总参数的 1%；
- FP32 下 merge 前后 logits 误差接近数值精度；
- merge 后延迟与原始线性层落在测量噪声内；

那么就复现了 LoRA 最重要的经验现象。

这属于**概念复现**。若要尽量贴近论文配置，应从 Microsoft 官方仓库开始；若优先考虑一周内完成，可以使用现代 PEFT 实现，但要记录版本和默认配置差异。

---

## 11. 如果反对它，如何设计反例？

### 11.1 数学上最干净的反例

设某一层：

$$
W_0=0,\qquad W_{\text{target}}=I_d
$$

那么目标更新是：

$$
\Delta W=I_d
$$

它的秩为 $d$，所有奇异值都等于 1。任何 rank-$r$ LoRA 更新 $BA$ 都满足：

$$
\operatorname{rank}(BA)\le r
$$

根据最佳低秩逼近理论：

$$
\min_{\operatorname{rank}(M)\le r}
\|I_d-M\|_F^2=d-r
$$

只要 $r\ll d$，误差就无法消失。如果训练数据覆盖全部输入方向，LoRA 无法通过数据分布的空缺掩盖这个误差。

这个反例说明：低秩更新不是通用表达形式。

### 11.2 更现实的 NLP 反例

构造一条“任务距离”阶梯：

1. 与预训练非常接近的英文分类；
2. 新领域英文；
3. 预训练覆盖很少但 tokenizer 支持的语言；
4. 需要学习大量新词义映射或新形式系统的任务。

同时比较：

- Full FT；
- 只改 $W_q,W_v$ 的 LoRA；
- 所有线性层 LoRA；
- rank 从 1 一直增加到接近 $d$。

如果任务越远，LoRA 所需 rank 越高，并且小 rank 与 Full FT 的差距系统性扩大，就能反驳“小 rank 普遍足够”的强版本。

这不会推翻论文原始实验，因为作者没有声称 LoRA 对所有分布迁移都有效。它会明确 LoRA 的适用边界。

---

## 12. 一个非增量式 follow-up idea

### Task-Geometry Compiler：任务几何编译器

这是**【研究猜想】**，新颖性没有经过穷尽式文献检索保证。

LoRA 预先规定所有任务都使用“低秩矩阵”这种编码。后续研究大多仍留在这个框架内：

- AdaLoRA 动态分配各层的 rank 预算。[AdaLoRA](https://arxiv.org/abs/2303.10512)
- QLoRA 量化冻结的底座权重。[QLoRA](https://arxiv.org/abs/2305.14314)
- DoRA 分离权重幅度和方向。
- LoRA+ 改变 $A,B$ 的相对学习率。

这个 follow-up 会改变问题定义：

> 不预设任务更新一定低秩。先测量目标任务需要的“功能变化”，再自动选择最便宜的参数表示。

### 12.1 核心假设

不同任务可能拥有不同的更新几何：

- 接近预训练分布的任务可能确实低秩；
- 新词汇任务可能需要局部稀疏更新；
- 风格或标定任务可能主要需要对角缩放；
- 远距离迁移可能需要少数层的高秩更新；
- 某些任务需要无法合并为线性矩阵的小型非线性模块。

### 12.2 方法

#### 第一步：校准

在少量下游数据上计算梯度、激活和 Jacobian 的随机 sketch，不保存完整全参数梯度历史。

#### 第二步：建立候选更新语言

每层允许从以下表示中选择：

- low-rank；
- sparse；
- diagonal；
- block-structured；
- 少数层高秩更新；
- 必要时小型非线性 residual。

#### 第三步：优化功能误差

使用少量全参数 probe step 作为临时 teacher，优化：

$$
\min_{\delta}
D_{\mathrm{KL}}
\left(
p_{\text{probe-FT}}
\,\|\,p_{\theta_0+\delta}
\right)
+\lambda\,\text{Storage}(\delta)
+\mu\,\text{Latency}(\delta)
$$

它寻找的是：在真实存储和延迟预算下，哪种更新编码最接近全参数微调产生的功能变化。

#### 第四步：编译部署

能合并的线性更新写回原权重；只有确实需要非线性变化的少数层保留额外模块。

### 12.3 研究价值

AdaLoRA 等方法仍然假设正确答案属于低秩矩阵族。Task-Geometry Compiler 将“更新应该是什么结构”本身变成可学习问题。

它可能回答一个更有科学价值的问题：

> 低秩究竟是任务适配的普遍规律，还是接近预训练分布时最常见的一种局部现象？

### 12.4 关键验证实验

建立任务距离阶梯，在相同 checkpoint 大小和实测延迟下比较：

- LoRA；
- AdaLoRA；
- DoRA；
- Full FT；
- Task-Geometry Compiler。

真正有说服力的结果应当是：

- 近分布任务自动选择低秩；
- 新词汇任务选择 sparse 或 embedding 更新；
- 远距离任务把预算集中到少数高秩层；
- 在统一资源预算下优于固定 LoRA；
- 结构选择在不同随机种子间具有稳定性。

如果编译器最后几乎总选择低秩，它也会反过来给 LoRA 的核心假设提供比原论文更强的证据。

---

## 参考资料

1. Hu, E. J. et al. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). ICLR 2022.
2. Microsoft. [Official LoRA implementation](https://github.com/microsoft/LoRA).
3. Houlsby, N. et al. [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751). ICML 2019.
4. Li, X. L. and Liang, P. [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://aclanthology.org/2021.acl-long.353/). ACL 2021.
5. Aghajanyan, A. et al. [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255).
6. Zhang, Q. et al. [AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2303.10512).
7. Dettmers, T. et al. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314).
8. Hayou, S. et al. [LoRA+: Efficient Low Rank Adaptation of Large Models](https://arxiv.org/abs/2402.12354).
9. Liu, S.-Y. et al. [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353).
10. Zhang, L. et al. [LoRA-FA: Efficient and Effective Low Rank Representation Fine-tuning](https://arxiv.org/abs/2308.03303).
