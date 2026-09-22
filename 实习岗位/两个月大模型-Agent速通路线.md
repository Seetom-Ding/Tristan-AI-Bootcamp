# 两个月大模型 / Agent 实习面试速通路线

> **2026-09-22 更新：本文件保留为初版规划参考。当前执行以[项目驱动学习路线：先 Agent，再推理](项目驱动学习路线-先Agent后推理.md)为准。** 新路线采用最小版本驱动、先 A 后 B、早期评测与日志；下方旧的周次安排和“必须完成”扩展项不再作为当前执行要求。

> 起点：Bootcamp Week 5，正在学习 tokenizer、embedding、self-attention、causal mask；Week 1–4 尚无正式评估证据。
> 目标：八周后能投递并面试 Agent 应用开发、LLM 应用后端、AI 全栈和初级推理/服务方向实习。

## 总策略

主线是“可评测的 Agent 应用 + 推理性能实验”。两个月内不把 RTL 后训练、复杂多模态算法、CUDA kernel 从零实现作为主线；它们适合作为后续分支。

## 两个必须完成的项目

### 项目 A：可观测、可评测的企业知识库 Agent

- 场景：选择一个垂直领域（招聘/课程/实验室文档/产品手册），准备 20–50 份公开文档。
- 流程：结构化解析 → 分块 → BM25 + 向量混合召回 → 可选 rerank → 带引用回答。
- Agent：LangGraph 状态图；至少 3 个工具（检索、计算/结构化查询、一个外部 API 或本地 MCP 工具）。
- 可靠性：Pydantic/JSON Schema、超时、重试上限、熔断、工具结果校验、写操作人工确认。
- 记忆：短期 thread state；长期用户偏好/事实单独存储，不能把全部聊天记录无界塞回 prompt。
- 评测：50–100 条问题集；记录 Recall@k/MRR、引用正确率、答案正确率、tool-call success、端到端成功率、p50/p95 延迟、token 成本。
- 可观测性：每次运行记录 request_id、模型/提示词版本、节点耗时、工具参数摘要、错误类型和最终结果。
- 交付：README 架构图、安装命令、10 条失败案例、评测脚本、演示视频/GIF、实验表。

### 项目 B：小模型推理服务性能基准

- 用同一个小型 Qwen/开源指令模型比较 Transformers eager 与 vLLM；有 GPU 再加入 4-bit/8-bit 量化。
- 固定模型、数据、随机种子和 warm-up；改变输入长度、输出长度、并发数。
- 记录 TTFT、TPOT/ITL、tokens/s、req/s、p50/p95、峰值显存和失败率。
- 画出 prefill/decode、KV cache、batch size 对指标的影响。
- 加分项：写一个 KV-cache block allocator 模拟器（Python 可用，C++ 更好），测试分配、释放、复用和碎片率。
- 不写“提升 30%”之类的数字，除非你真的跑过并保留环境、命令和原始结果。

## 八周安排

| 周次 | 学习与面试 | 项目交付 |
|---|---|---|
| 第 1 周（当前 Week 5） | 完成 causal attention Boss；复习 tokenization、embedding、mask、next-token loss；Python/数据结构每天 45 分钟 | 项目 A 选题、数据目录、最小 tokenizer/RAG baseline |
| 第 2 周（Week 6） | Transformer block、RoPE、RMSNorm、MLP、decoder-only；能解释 KV cache 为什么存在 | 小 GPT/解码循环；A 的文档解析和 chunk 版本一 |
| 第 3 周（Week 7） | Hugging Face 推理、sampling、SFT/LoRA/DPO 高层区别；FastAPI、Pydantic、asyncio | A 的混合检索、引用回答、3 个工具 |
| 第 4 周（Week 8） | ReAct vs workflow、function calling、context engineering、memory、MCP、安全与人工确认 | LangGraph 状态图、短长期记忆、重试/熔断、MCP 工具；跑通端到端 Demo |
| 第 5 周（Week 9） | RAG 评测、rerank、query rewrite、幻觉与数据泄漏；开始读 vLLM serving 文档 | A 的 50–100 条评测集、Recall@k/引用/工具成功率/错误分类 |
| 第 6 周（Week 10） | Prefill/Decode、KV cache、PagedAttention、continuous batching、TTFT/TPOT、量化取舍 | 项目 B 跑完两种后端和一组并发/长度矩阵；保存原始日志 |
| 第 7 周（Week 11） | 系统设计：限流、缓存、队列、观测、降级；C++/OS/网络各补最常问部分 | A/B README、架构图、失败案例、性能图表；可选 C++ allocator |
| 第 8 周（Week 12） | 项目深挖、LLM 八股、算法题、两轮模拟面；按弱项回补 | 简历两条项目 bullet、3 分钟项目讲解、20 个高频题口述稿、完整投递包 |

## 每周节奏

周一至周四：上午 2 小时基础/论文，下午 2 小时项目，晚间 45–60 分钟算法与八股。周五做集成、写实验记录和一次 60–90 分钟模拟面。若每天只有 3 小时，保留“项目 + 面试题”，论文阅读压缩为关键图表和结论。

## 面试必备清单

- LLM：tokenizer、embedding、attention、RoPE、LayerNorm/RMSNorm、causal mask、KV cache、prefill/decode、sampling、SFT/LoRA/DPO。
- Agent：function/tool calling 全链路、ReAct 与 workflow、多 Agent 何时值得用、memory 分层、MCP、结构化输出、重试/超时/熔断、人审和权限。
- RAG：chunk、BM25 vs dense、hybrid、rerank、query rewrite、Recall@k/MRR、引用与 groundedness、热更新。
- Serving：vLLM/SGLang 的定位、PagedAttention、continuous batching、TTFT/TPOT/吞吐/显存、量化的质量—速度—显存权衡。
- 工程：Python async/FastAPI/Pydantic/Git/Linux、Redis/SQL 基础、日志与测试；若投推理/系统岗，再补 C++ STL、RAII、并发、内存和网络。
- 算法：数组/哈希、双指针、滑窗、栈队列、二叉树、DFS/BFS、堆、二分、并查集、LRU；以能写对和讲复杂度为目标。

## 与截图岗位的匹配

- 最匹配：阿里 AI 全栈、快手 Agent、阿里大模型算法中的 Agent/LLM 应用部分。
- 次匹配：OPPO AI 系统工程、通用推理/部署岗；项目 B 和 C++ allocator 是关键补强。
- 暂不作为两个月主投：小米 RTL/后训练、vivo 影像算法、需要深 CUDA/MNN/QNN 的模型优化岗。可以投，但不要把准备时间从主线项目抽走。

## 简历与面试交付标准

每个项目都要能在 3 分钟内讲清“问题—架构—自己写的部分—指标—失败—下一步”。简历只写实际测得的数字，例如：

- 构建 LangGraph + MCP 的垂直知识库 Agent，完成混合检索、工具调用、记忆和失败熔断；在 N 条评测问题上达到 Recall@k=__、tool-call success=__，p95 延迟=__ms。
- 对同一模型比较 Transformers 与 vLLM，在并发/输入长度矩阵下测量 TTFT、TPOT、吞吐和峰值显存；定位瓶颈为 __，验证 __ 优化使 __ 指标变化 __%。

## 每周自测门槛

周末随机抽 10 题，项目追问 5 题，1 道中等算法题；至少 80% 能在 2 分钟内回答，项目问题不能靠翻代码。达不到时优先修项目证据和表达，不再盲目扩展新框架。

