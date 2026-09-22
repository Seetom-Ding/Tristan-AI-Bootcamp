# 大模型与 Agent 实习学习地图

> 目标：面向中国大厂的大模型、Agent、AI 应用、推理部署和 AI 系统实习岗位，建立一套有依赖关系、可执行、可评估的知识体系。
>
> 当前起点：Bootcamp Week 5，正在学习 Tokenizer、Embedding、Self-Attention 和 Causal Mask。
>
> 使用方法：按照主干顺序学习。每个模块完成“掌握标准”和“实践产出”后再进入下一层。P0 是两个月内必须掌握，P1 是应该掌握，P2 是后续深入或岗位定向。

---

## 总体知识结构

    大模型与 Agent 实习
    ├── 第一层：计算机与工程基础
    │   ├── Python
    │   ├── 数据结构与算法
    │   ├── Linux、网络、数据库
    │   └── C/C++、并发、服务开发
    ├── 第二层：深度学习与 PyTorch
    │   ├── Tensor、Autograd
    │   ├── 训练和优化
    │   └── 实验复现
    ├── 第三层：Transformer 与大模型
    │   ├── Tokenizer、Embedding
    │   ├── Attention、Transformer Block
    │   ├── 语言模型生成
    │   └── KV Cache
    ├── 第四层：大模型应用
    │   ├── RAG
    │   ├── Tool Calling
    │   ├── Agent Workflow
    │   ├── Memory
    │   └── MCP
    ├── 第五层：评测与工程化
    │   ├── 评测
    │   ├── 可观测性
    │   ├── 可靠性
    │   └── 安全
    ├── 第六层：推理与 Serving
    │   ├── Prefill、Decode
    │   ├── KV Cache、Batching
    │   ├── vLLM、SGLang
    │   └── GPU、量化、算子优化
    └── 第七层：岗位专属分支
        ├── 多模态
        ├── 微调与对齐
        ├── 端侧和 NPU
        ├── RTL 与代码模型
        └── 分布式训练

---

## 学习优先级

### P0：两个月内必须深入

- Python 工程
- 数据结构与算法
- Tensor、Autograd、训练基础
- Tokenizer、Embedding、Attention
- Transformer Block
- Causal Language Modeling
- 文本生成
- KV Cache 基础
- RAG 全流程
- Function Calling
- Agent Workflow
- Memory
- MCP 基础
- Agent 评测
- 日志、Tracing、重试、熔断
- vLLM 基础
- TTFT、TPOT、吞吐、显存

### P1：两个月内达到能做项目、能回答面试题

- C++、Linux、网络、数据库
- LoRA、SFT、DPO
- 量化
- Rerank、HyDE、GraphRAG
- SGLang
- Prefix Cache、Continuous Batching
- GPU 内存和带宽
- 多模态基础
- 服务限流、缓存、队列、降级
- Java 服务基础

### P2：先理解概念，后续再深入

- CUDA/Triton Kernel
- FlashAttention 实现
- Tensor Parallelism、Pipeline Parallelism
- 分布式训练
- GRPO 实战
- RTL 后训练
- NPU 算子开发
- 复杂 Multi-Agent OS
- 高级图像算法

---

# 第一层：计算机与工程基础

## 1. Python 工程

### 1.1 语言基础

- 变量、函数、作用域
- 列表、字典、集合、元组
- 字符串和文件处理
- 类、继承、组合
- 模块和包
- 生成器和迭代器
- 装饰器
- 上下文管理器
- 异常处理
- 类型标注

### 1.2 Agent 开发需要的 Python

- asyncio
- 并发请求
- 超时和取消
- HTTP 客户端
- JSON/YAML
- Pydantic
- FastAPI
- 流式响应
- 环境变量和配置管理
- 依赖管理
- pytest
- logging

### 1.3 掌握标准

能够：

- 写一个异步调用 LLM 的 API
- 对模型输出进行结构化校验
- 为工具调用写单元测试
- 正确处理超时、异常和重试
- 读懂陌生的 Agent 项目代码

### 1.4 实践产出

- 一个 FastAPI LLM 接口
- 一个异步工具调用器
- 一组 pytest 测试
- 一套结构化日志

---

## 2. 数据结构与算法

### 2.1 必学数据结构

- 数组、字符串、哈希表
- 链表、栈、队列、双端队列
- 堆、二叉树、图、并查集

### 2.2 必学算法模式

- 双指针、滑动窗口、前缀和
- 二分查找、排序
- DFS、BFS、回溯
- 动态规划基础
- 贪心、拓扑排序
- 最短路径基础
- Top-K
- LRU Cache

### 2.3 掌握标准

- 能写出常见中等题
- 能解释时间复杂度和空间复杂度
- 能根据约束选择数据结构
- 能独立实现 LRU、滑动窗口和 BFS/DFS

---

## 3. Linux、网络、数据库和服务

### 3.1 Linux 与操作系统

- 进程、线程、协程
- CPU 调度
- 虚拟内存
- 文件系统
- I/O
- 锁和并发
- 内存泄漏
- Shell
- 日志查看
- 进程监控

### 3.2 网络

- TCP/IP
- HTTP/HTTPS
- REST API
- WebSocket
- SSE 流式响应
- RPC
- 超时、重试、连接池
- 负载均衡

### 3.3 数据库和中间件

- SQL
- 索引
- 事务
- Redis
- 缓存过期
- 消息队列
- 任务队列
- 向量数据库
- 数据库与缓存一致性

### 3.4 容器与部署

- Docker
- Docker Compose
- 环境变量
- 服务健康检查
- 基础 CI
- Git 分支、提交和回滚

---

## 4. C/C++ 与 Java

### 4.1 C/C++

面向推理和 AI 系统岗位掌握：

- 指针和引用
- 内存布局
- RAII
- 智能指针
- STL
- 模板
- 移动语义
- 编译和链接
- 多线程
- Mutex、Atomic
- 线程池
- 基础 SIMD

### 4.2 Java

只需先掌握：

- 面向对象
- 集合
- 异常
- 线程
- HTTP 服务
- Spring Boot 基本概念
- Redis/MySQL 调用

---

# 第二层：深度学习与 PyTorch

## 1. 数学基础

### 1.1 线性代数

- 向量、矩阵、张量
- 矩阵乘法
- 转置
- 广播
- 内积
- 范数
- 相似度
- Batch、Sequence、Channel

### 1.2 概率

- 概率分布
- 条件概率
- 期望
- 采样
- Softmax
- 交叉熵
- KL 散度

### 1.3 微积分和优化

- 导数和梯度
- 链式法则
- 反向传播
- SGD
- Adam/AdamW
- 学习率
- Warm-up
- 梯度裁剪

---

## 2. PyTorch

- Tensor 创建和索引
- Shape 变换
- 矩阵乘法和 einsum
- Dataset/DataLoader
- Module、Parameter、Forward
- Loss、Backward、Optimizer
- CUDA
- 混合精度
- 模型保存和加载
- 训练/验证循环
- 随机种子
- Profiling

### 2.1 实践产出

- MLP
- CNN
- 简单 RNN
- 单头 Attention
- 最小字符级语言模型
- 训练和验证脚本
- 可复现实验配置

---

# 第三层：Transformer 与大模型

## 1. Tokenizer 与 Embedding

### 1.1 Tokenizer

- 字符级、词级、子词级
- Vocabulary
- Token ID
- Special Token
- Padding
- Unknown Token
- Encode/Decode
- Attention Mask
- Token 长度和成本

### 1.2 Embedding

- Token Embedding
- Position Embedding
- 参数量计算
- RoPE
- 绝对位置和相对位置
- 长上下文限制

### 1.3 掌握标准

能够解释：

    文本
    → Token ID
    → [B,T]
    → Embedding
    → [B,T,C]

---

## 2. Attention

- Query、Key、Value
- Q/K/V 投影
- Scaled Dot-Product Attention
- Softmax 维度
- Causal Mask
- Padding Mask
- Multi-Head Attention
- MHA、MQA、GQA
- Attention 复杂度
- 并行计算
- Future Leakage

核心公式：

    Q = XWq
    K = XWk
    V = XWv

    Attention(Q,K,V)
    = softmax(QKᵀ / √dk + mask)V

### 2.1 实践产出

- 独立实现一次单头 Attention
- 编写 causal mask 测试
- 检测错误 softmax 维度
- 对比 PyTorch 官方实现

---

## 3. Transformer Block

- Residual Connection
- LayerNorm
- RMSNorm
- Feed Forward Network
- GELU
- SwiGLU
- Pre-Norm
- Post-Norm
- Decoder-only
- Encoder-only
- Encoder-Decoder
- 参数量估算
- FLOPs 估算

典型结构：

    输入
    → Norm
    → Attention
    → Residual
    → Norm
    → MLP
    → Residual

---

## 4. 语言模型训练与生成

### 4.1 训练

- Causal Language Modeling
- Next-token Prediction
- Teacher Forcing
- Label Shift
- Cross-Entropy
- Perplexity
- 训练和验证 loss
- 过拟合

### 4.2 生成

- Greedy Search
- Temperature
- Top-k
- Top-p
- Repetition Penalty
- Stop Token
- Max New Tokens
- Beam Search
- Streaming

### 4.3 KV Cache

- Prefill
- Decode
- KV Cache 结构
- KV Cache 显存占用
- 长上下文问题
- Prefix Cache
- KV Cache 量化
- KV Cache Offloading

---

# 第四层：训练、微调和对齐

## 1. 训练数据

- 数据清洗
- 去重
- 质量控制
- 指令数据
- 对话数据
- 偏好数据
- 训练/验证/测试划分
- 数据泄漏

## 2. SFT

- Supervised Fine-Tuning
- Instruction Tuning
- Assistant Loss Mask
- Padding Mask
- 学习率
- Batch Size
- Epoch
- 灾难性遗忘

## 3. LoRA 与 QLoRA

- 低秩矩阵
- Adapter
- Rank
- Alpha
- Dropout
- 参数高效微调
- 量化模型微调
- 全参数微调和 LoRA 的区别

## 4. DPO、RLHF、PPO、GRPO

    SFT
    → 学会指令格式

    DPO
    → 利用偏好数据直接优化

    RLHF/PPO
    → 奖励模型加强化学习

    GRPO
    → 用相对奖励优化推理任务

至少能回答：

- Prompt 什么时候足够？
- RAG 什么时候更合适？
- LoRA 什么时候更合适？
- DPO 解决什么问题？
- 强化学习为什么不稳定？

---

# 第五层：RAG 知识库

## 1. 文档处理

- PDF 解析
- Markdown 解析
- Word/网页解析
- OCR
- 文本清洗
- 标题识别
- 表格处理
- 代码处理
- Metadata
- 文档版本
- 热更新

## 2. Chunk 切分

- 固定长度
- 按标题
- 按语义
- Overlap
- Parent-Child Chunk
- 表格切分
- 代码切分
- Chunk 太大或太小的影响

## 3. 检索

### 3.1 Dense Retrieval

- Embedding
- Cosine Similarity
- 向量数据库
- Top-k
- 相似度阈值

### 3.2 Sparse Retrieval

- BM25
- 倒排索引
- 关键词匹配
- 中文分词

### 3.3 Hybrid Retrieval

    BM25
    +
    向量检索
    → 候选合并
    → Rerank
    → 上下文构造

## 4. 高级 RAG

- Rerank
- Query Rewrite
- Multi-Query
- HyDE
- Parent-Document Retrieval
- GraphRAG
- Agentic RAG
- Corrective RAG
- Self-RAG

## 5. RAG 评测

- Recall@k
- Precision@k
- MRR
- NDCG
- Hit Rate
- Answer Accuracy
- Citation Accuracy
- Context Relevance
- Faithfulness
- 幻觉率
- 延迟
- Token 成本

---

# 第六层：Agent 系统

## 1. Agent 基本结构

    用户目标
    → 任务理解
    → 规划
    → 工具选择
    → 工具调用
    → 读取结果
    → 判断是否继续
    → 输出答案

需要掌握：

- Agent State
- Planner
- Executor
- Tool
- Observation
- Final Answer
- Stop Condition
- Agent Loop

## 2. Agent 类型

- Workflow
- ReAct
- Plan-and-Execute
- Router
- Reflection
- Multi-Agent
- Human-in-the-Loop

重点比较：

- 固定 Workflow 什么时候更好？
- Agent 什么时候更灵活？
- Multi-Agent 是否真的有必要？
- 如何避免无限循环？

## 3. Context Engineering

- Prompt Engineering：优化指令
- Context Engineering：组织模型需要的信息
- Harness Engineering：通过工具、测试和环境约束模型
- Loop Engineering：设计 Agent 循环、停止和恢复

## 4. Function Calling

    模型读取 Tool Schema
    → 选择工具
    → 生成参数
    → 后端校验
    → 执行工具
    → 返回结果
    → 模型继续推理

需要掌握：

- JSON Schema
- 结构化输出
- 参数校验
- 工具选择
- 工具结果解析
- 工具失败处理
- 幂等性
- 权限控制

## 5. Memory

- 当前上下文
- 短期会话记忆
- 长期用户记忆
- 结构化记忆
- 向量记忆
- 摘要记忆
- 上下文压缩
- 记忆检索
- 记忆更新
- 记忆删除
- 记忆过期

## 6. MCP

- MCP Client
- MCP Server
- Tools
- Resources
- Prompts
- tools/list
- tools/call
- 输入输出 Schema
- 工具版本
- 权限
- 人工确认

## 7. Agent 可靠性

- Timeout
- Retry
- Exponential Backoff
- Circuit Breaker
- Fallback Model
- 最大步骤数
- 任务取消
- 输出校验
- 错误恢复
- 人工介入
- 幂等调用

## 8. Agent 安全

- Prompt Injection
- Tool Poisoning
- 越权调用
- 数据泄露
- 敏感信息过滤
- 最小权限
- 沙箱
- 审计日志
- 外部工具信任边界

---

# 第七层：评测、可观测性与实验

## 1. 评测层级

    模型
    → 检索
    → 工具
    → Agent 节点
    → 端到端任务
    → 线上系统

## 2. 指标

- Answer Accuracy
- Retrieval Recall@k
- Citation Accuracy
- Tool Call Success Rate
- Task Success Rate
- 平均 Agent 步数
- Token 数
- 成本
- TTFT
- TPOT
- p50/p95/p99
- 峰值显存
- 错误率

## 3. 可观测性

- Request ID
- Trace ID
- Span
- Prompt 版本
- 模型版本
- Tool 输入输出
- 每个节点耗时
- 错误类型
- 失败重放
- 运行轨迹
- 线上告警

## 4. 实验设计

- 控制变量
- Baseline
- Ablation
- 固定数据
- 固定随机种子
- Warm-up
- 重复测量
- 均值与分位数
- 硬件和软件环境记录
- 结果可复现

---

# 第八层：推理与 Serving

## 1. 推理基础

- Transformers 推理
- Prefill
- Decode
- KV Cache
- Sampling
- Streaming
- Batch
- Dynamic Batching
- Continuous Batching

## 2. 推理框架

学习顺序：

    Transformers
    → vLLM
    → SGLang
    → TensorRT-LLM
    → llama.cpp/Ollama

需要比较：

- 部署难度
- 速度
- 显存
- Batching
- 量化
- 多模型支持
- 工具调用
- OpenAI 兼容接口

## 3. 性能指标

- TTFT
- TPOT
- ITL
- Throughput
- Requests per Second
- p50/p95/p99
- 峰值显存
- 单 token 成本
- 启动时间
- 错误率

## 4. GPU 与算子

- GPU 显存层级
- Global Memory
- Shared Memory
- Register
- Memory Bandwidth
- Compute Bound
- Memory Bound
- Arithmetic Intensity
- Kernel Fusion
- CUDA Graph
- FlashAttention
- Triton

## 5. 量化和优化

- FP32
- FP16
- BF16
- INT8
- INT4
- PTQ
- QAT
- GPTQ
- AWQ
- SmoothQuant
- 权重量化
- 激活量化
- KV Cache 量化
- 剪枝
- 算子融合
- 图优化

## 6. Serving 系统

    客户端
    → 网关
    → 限流
    → 请求队列
    → Scheduler
    → Model Worker
    → KV Cache
    → 流式响应
    → 日志与监控

需要掌握：

- 多模型路由
- 负载均衡
- 优先级调度
- 失败转移
- 自动扩缩容
- 资源隔离
- 请求取消
- 缓存
- 降级
- 成本控制

---

# 第九层：岗位专属分支

## 1. 多模态方向

- 图像 Tensor
- CNN
- ViT
- Image Encoder
- Text Encoder
- Projector
- Cross-Attention
- OCR
- 图文问答
- 多模态 RAG
- 图像质量评测
- OpenCV
- 图像异常检测
- 风格、虚化、曝光、界面元素识别

## 2. 端侧模型优化

- C/C++
- SIMD
- OpenCV
- CUDA
- MNN
- QNN
- NPU
- 算子融合
- 图优化
- 剪枝
- 量化
- 模型转换
- 端侧内存和功耗
- 设备适配

## 3. 模型后训练与 RTL

- Verilog
- RTL
- SVA
- Yosys
- Verilator
- RTL 数据集
- Spec 到 RTL 对齐
- SFT
- DPO
- GRPO
- GenBench
- VerilogEval
- RTL Review
- Bug Detection
- 代码模型评测

## 4. 分布式训练

- Data Parallelism
- Tensor Parallelism
- Pipeline Parallelism
- ZeRO
- FSDP
- NCCL
- 通信开销
- 梯度同步
- Checkpoint
- 多 GPU 训练

---

# 第十层：项目与面试能力

## 1. 项目 A：可评测 Agent

必须包含：

- 垂直领域文档
- Chunk
- BM25 + 向量检索
- Rerank
- 引用回答
- 3 个工具
- LangGraph
- Memory
- MCP
- 重试和熔断
- 评测集
- Trace
- README
- 失败案例

## 2. 项目 B：推理性能实验

必须包含：

- Transformers 与 vLLM 对比
- 不同输入长度
- 不同并发数
- TTFT、TPOT、吞吐
- p50/p95
- 峰值显存
- Prefill/Decode 分析
- KV Cache 解释
- 原始实验日志
- 性能图表

## 3. 项目讲解框架

每个项目都要能回答：

- 为什么做？
- 系统怎么设计？
- 自己负责什么？
- 为什么选择这个方案？
- 指标如何定义？
- 遇到什么失败？
- 如何定位？
- 如何验证修复？
- 还有什么局限？
- 下一步怎么优化？

---

# 两个月执行顺序

## 第 1 周

- 完成 Causal Attention
- 完成 Transformer 基础
- 复习 Python 和算法
- 选定项目 A 的业务领域

## 第 2 周

- 学 Transformer Block
- 学 RoPE、RMSNorm、MLP
- 完成最小语言模型
- 完成文档解析和第一版 Chunk

## 第 3 周

- 学 RAG
- 完成 BM25 + 向量检索
- 学 Function Calling
- 接入三个工具

## 第 4 周

- 学 LangGraph
- 加入 Memory
- 加入 MCP
- 加入重试、超时、熔断

## 第 5 周

- 建立评测集
- 测 Recall@k、MRR、工具成功率
- 增加 Rerank 或 Query Rewrite
- 记录失败案例

## 第 6 周

- 学 vLLM
- 学 Prefill/Decode
- 学 KV Cache 和 PagedAttention
- 完成性能基准

## 第 7 周

- 补 C++、Linux、网络
- 完成项目 README
- 做系统设计题
- 优化项目稳定性

## 第 8 周

- 两轮完整模拟面
- 准备项目讲解
- 准备 20 道 LLM/Agent 高频题
- 准备 20 道算法题
- 开始集中投递

---

# 最终能力闭环

    Python / 数据结构 / Linux
        ↓
    PyTorch / Tensor / Autograd
        ↓
    Tokenizer / Embedding / Attention
        ↓
    Transformer / 语言模型生成
        ↓
    KV Cache / 推理性能
        ↓
    RAG / 检索增强
        ↓
    Tool Calling / MCP
        ↓
    Agent Workflow / Memory
        ↓
    评测 / Tracing / 可靠性
        ↓
    vLLM / Serving / 部署
        ↓
    C++ / GPU / 系统优化

当前最重要的路线是：

    Transformer
    → 语言模型生成
    → RAG
    → Tool Calling
    → Agent
    → 评测
    → Serving

多模态、RTL、复杂后训练、CUDA Kernel 和分布式训练属于后续分支，不应打断这条主线。

