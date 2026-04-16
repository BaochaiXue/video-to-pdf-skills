# Berkeley Agentic AI Fall 2025：对 Spring 2025 主书的课程级增补

## 1. 本章范围与证据边界

这一章不是重新把 `CS294/194-196: Agentic AI (Fall 2025)` 逐讲完整转写一遍，而是基于官方公开来源，给 `CS294/194-280: Advanced Large Language Model Agents (Spring 2025)` 主书补上一层**课程级增量解释**。证据边界只有四类：

- 官方课程页：<https://rdi.berkeley.edu/agentic-ai/f25>
- 官方 MOOC 页：<https://agenticai-learning.org/f25>
- 官方 Berkeley RDI playlist：<https://www.youtube.com/playlist?list=PLS01nW3RtgoqGkm4UeqNeZLccW-OGc1fJ>
- 官方课程页显式列出的 slides、readings，以及 `Oct 6` 的官方但未列入 playlist 的 unlisted recording：<https://www.youtube.com/watch?v=VfOA2a0dj4w>

因此，本章的立场非常明确：

1. **能从 official page、slides、playlist、official readings 直接支撑的内容，就按事实写。**
2. **只能从 lecture 标题和 reading 标题推断的内容，要明确说这是保守解释，而不是已验证的 slide-level 细节。**
3. **本章服务于 Spring 2025 主书的升级，而不是替代主书。**

这门 Fall 2025 课的一个关键信号来自 MOOC 页本身。官方直接写出：它是建立在 **Fall 2024 LLM Agents MOOC** 和 **Spring 2025 Advanced LLM Agents MOOC** 基础之上的。这意味着 Fall 2025 不是简单重复，而是在原来“reasoning / planning / tools / multimodal / theorem proving / safety”框架上，再往**评测、可验证后训练、multi-agent、scientific discovery、deployment、embodiment**这些更“系统工程”和“真实部署”方向推进。

## 2. 为什么 Fall 2025 是主书的真实后续，而不是同义改名

Spring 2025 主书已经覆盖了高级 LLM agent 的核心研究主线：

- 推理时计算（inference-time computation）
- 学习推理（learning to reason）
- 记忆与规划（memory and planning）
- 开放 post-training recipe
- coding agents 与漏洞检测
- multimodal / web / GUI / OS agents
- formal mathematics / theorem proving
- abstraction / discovery
- safe and secure agentic AI

如果只看这个列表，会以为 Berkeley 这条线已经把“agent”讲完了。但 Fall 2025 的官方课程描述和具体 syllabus 显示，它把重点向另外几个方向明显移动：

1. **evaluation 从附属 benchmark 变成课程主轴。**
2. **post-training 不再只是 preference optimization，而是 environment-feedback-aligned、verifier-centered 的 agent training。**
3. **agent 的真正难点被重新定位到 system design、infra、grader、orchestration、observability、access control、simulation。**
4. **multi-agent 和 recursive self-improvement 被提升为独立主题，而不是 planning 的旁支。**
5. **scientific discovery、paper agents、embodiment 把 agent 从“会用工具的语言模型”推向“可在复杂环境中持续运行的研究或控制系统”。**

换句话说，Spring 2025 更像是在回答：**高级 agent 需要哪些能力模块？**  
而 Fall 2025 更像是在追问：**当这些能力真的要落地到 benchmark、训练 recipe、产品部署、群体交互和现实环境时，系统应该如何组织？**

## 3. 第一条增量主线：从“LLM 能做什么”转向“agent system 如何组织”

`Sep 15` 的 `LLM Agents Overview` slides 给了一个非常关键的框架。公开 slide 文本里明确出现：

- `Pretraining`
- `Reasoning RL`
- `Classic post-training / RLHF`
- `Evaluation`
- `Systems and infra to scale`

这件事的重要性不在于它重新讲一遍 LLM pipeline，而在于它把 **evaluation** 和 **systems & infra** 提升到了与模型算法并列的位置。Spring 2025 主书当然已经涉及环境反馈、tool use 和系统闭环，但 Fall 2025 在课程一开始就告诉你：**agent 时代的关键瓶颈不再只是模型会不会推理，而是训练、评测、部署、扩展这整条链路能不能稳定运行。**

`Sep 22` 的 `Evolution of system designs from an AI engineer perspective` 又把这个方向推进了一步。Yangqing Jia 的公开 slides 里最值得注意的，不是某个具体模型，而是三个判断：

- 新算法会持续推动模型能力；
- 应用空间会持续膨胀；
- `AI infra has become the 3rd pillar of enterprise IT strategy`

这说明 Fall 2025 对 agent 的看法已经明显偏向**系统工程对象**。主书在 Spring 2025 里更多讨论模型、verifier、tool loop、workflow design；而 Fall 2025 这里开始强调：

- developer efficiency
- infra efficiency
- multi-cloud supply chain management
- AI-native platform to unify development, training, and inference

这部分对主书的增量意义在于：它把 agent 研究从“方法论层”继续推进到“组织与基础设施层”。一个 agent 不是一个 prompt，不是一个模型，也不是一个单独 benchmark 结果，而是一个需要被**供应链化、平台化、审计化**的系统。

## 4. 第二条增量主线：从 preference-aligned post-training 走向 verifier-centered agent training

Spring 2025 主书中，`Learning to reason with LLMs` 和 `Open Training Recipes for Reasoning in Language Models` 已经讨论了 DPO、PPO、GRPO、开放 post-training recipe，以及 reasoning data 和 verifier 的关系。但 Fall 2025 把这个主题显著具体化了。

### 4.1 Jiantao Jiao：agentic model 不再只是 human preference model

`Sep 29` 的 `Post-Training Verifiable Agents` slides 里，最关键的定义不是某个 loss，而是对模型目标的重新改写：

- 早期 chat model：`Human Aligned Models`
- agentic model：`Environment Feedback Aligned Models`

这个区分非常重要。它意味着系统的优化目标，从“让用户偏好更满意”转向“让 agent 在环境中取得可验证的正确结果”。Jiantao 的 slides 明确把 agent training 拆成三个要素：

1. `environment`
2. `tools`
3. `verifier`

也就是说，一个 agentic model 不是只在上下文窗口里生成看起来合理的文本；它要读取环境状态、调用工具、再根据 verifier 返回的信号被训练。这里的 verifier 可以是：

- code unit tests
- proof checker
- DOM script
- ground-truth state verifier

这和主书的 Spring 2025 形成了很强的呼应。主书已经说明，强 agent 往往依赖外部反馈；Fall 2025 则进一步把这个观点上升成训练范式：**环境反馈不是推理阶段的附加组件，而是训练目标本身的一部分。**

### 4.2 Weizhu Chen：真正困难的是 grader、rubric 和 product constraints

如果 Jiantao 讲的是“目标函数变了”，那 `Oct 13` 的 `Some Challenges and Lessons from Training Agentic Models` 讲的是：**真正难的是怎么把这个目标做成能训练、能上线、能稳定优化的工程对象。**

公开 slides 里有一个很典型的 coding-agent 结构：

- problem prompt
- code repository
- tool calls / tool responses
- Linux system
- unit tests
- model patch + PR description

这说明 agent training 已经不是“对着答案做监督学习”，而是要在一个类似软件工程 sandbox 的环境里学习一整套交互行为。Weizhu 的 slides 还反复强调三件事：

1. **rubric 必须可评分、可解释、可组合。**
2. **verifiable data 与 non-verifiable data 需要混合治理。**
3. **product grader 远比单一 unit test 复杂。**

对于软件 agent，公开 slides 明确列出 grader 的多个层次：

- unit-test grader
- patch grader
- rollout grader
- behavior grader
- user-experience grader
- task-specific grader

这一步是对 Spring 2025 主书的直接增强。主书里我们已经能看到 coding agent、漏洞检测、formal verification、OS 环境等内容，但 Fall 2025 让一个更“工业级”的事实变得清楚：**agent 训练的难点，经常不是模型结构，而是你能否定义出一套足够细、足够可靠、又不会把系统带偏的 grader stack。**

### 4.3 为什么这会改写主书对后训练的理解

如果把 Spring 2025 和 Fall 2025 放在一起看，一个更完整的 post-training 图景是：

- **preference alignment** 解决“用户觉得像不像一个好助手”
- **reasoning RL** 解决“模型会不会在有明确答案的问题上形成更强推理轨迹”
- **verifiable agent training** 解决“模型能不能在环境里持续地产生可被 checker / tests / rubric 验证的动作和结果”

因此，主书里有关 post-training 的章节，在读完这个 supplement 后，最好升级成一句更严格的话：  
**post-training 的最终对象不是静态答案，而是受环境、工具、verifier、grader 和 product constraints 共同约束的 agent policy。**

## 5. 第三条增量主线：evaluation 不再是结果汇报，而是 harness 本身

Spring 2025 主书已经大量涉及 benchmark 和 verifier，但 Fall 2025 的最大新意之一，是把 evaluation 提升为课程级基础设施。

### 5.1 Oct 6：先建立 taxonomy，再谈 benchmark

`Oct 6` 的 `Agent Evaluation & Project Overview` 虽然只是课程中的一讲，但它在整体结构里非常关键。公开 slides 明确给出四组区分：

- `close-ended` vs `open-ended`
- `verifiable` vs `non-verifiable`
- `static` vs `dynamic`
- `taxonomy of agent eval tasks`

同时 slides 明确提出了 `Outcome Validity`。这背后的含义是：一个 agent eval 不是“只要能自动评分就行”，而是要问：

- 这个任务真的对应我们想测的 agent capability 吗？
- 这个评分信号能区分真正完成任务与投机性通过吗？
- 这个环境能否逼迫 agent 暴露 memory、planning、tool use、adaptation 等真实能力？

对主书而言，这意味着 evaluation 不应再被当作 chapter 结尾的 benchmark list，而应当被当作**agent harness 的一部分**。

### 5.2 Survey：agent eval 已经形成独立知识结构

官方 reading `Survey on Evaluation of LLM-based Agents` 把 agent eval 拆成四层：

1. capability evaluation  
   planning, tool use, self-reflection, memory
2. application-specific evaluation  
   web agents, software engineering agents, scientific agents, conversational agents
3. generalist agent evaluation
4. evaluation frameworks

这说明 Fall 2025 相比 Spring 2025 的一个重要变化是：**课程不再把 evaluation 看成某个 domain 的局部细节，而是把它视为组织整个 agent field 的骨架。**

### 5.3 Oct 27：没有误差条的 agent leaderboard 很可能会误导你

`Oct 27` 的 `Predictable Noise in LLM` 与官方 reading `Adding Error Bars to Evals` 一起，把另一个经常被忽略的问题推到了台前：**小 benchmark、hard benchmark、generative benchmark 的统计噪声很大。**

公开 slides 与 reading 共同支持几个关键判断：

- 很多 agent eval 的 test set 比 ImageNet 小得多
- 但每道题的生成和执行成本更高
- 只看 pass rate 的百分点差异，经常无法判断提升是否稳健
- 对模型比较应该更多使用 paired comparison，而不是只比较 summary statistic
- evaluation 设计需要 power analysis，而不是做完再看数字

这对 Spring 2025 主书是一个非常实质的升级。主书里我们已经见过很多 execution-based benchmark，但 Fall 2025 逼着我们再加一层问题：  
**这个 benchmark 的方差结构是什么？它的 improvement 是真实进步，还是样本噪声？**

所以，在新的 agent textbook 里，evaluation 不能只写成“列举 benchmark”。它必须被写成：

- task design
- verifier design
- statistical design
- reporting discipline

这四者共同组成的 harness。

## 6. 第四条增量主线：multi-agent 不只是分工协作，而是 population-level dynamics

Spring 2025 主书讨论过 planning、memory、workflow orchestration，也涉及过 abstraction 和 discovery。但 Fall 2025 的 `Multi-Agent AI` 与 `Multi-Agent Systems in the Era of LLMs` 提醒我们：**multi-agent 不只是“多开几个 worker”。**

### 6.1 Noam Brown：self-play、minimax 与 exploitability

`Oct 20` 的公开 slides 直接从 `self-play`、`minimax equilibrium`、`population best response`、`exploitability` 出发。这里真正值得带回主书的不是扑克例子本身，而是下面这条思想：

> 当系统变成多主体交互时，能力的定义不再只是单个 agent 的平均任务成功率，而是它在一个 population 里的稳定性、可被利用程度，以及能否通过 self-play 或 recursive improvement 继续优化。

这和 Spring 2025 主书的关系是：

- 主书更强调单 agent 在外部环境中的 reasoning / planning / tooling
- Fall 2025 开始强调多个 agent、多个策略、以及 population interaction 下的稳定解

这会改变我们对“agent scaling”的理解。很多人把 scaling 想成更多参数、更多 inference-time compute、更多工具。但 Fall 2025 的 multi-agent 视角说明，另一个可扩展维度是：**让 agent 在相互博弈、协作或竞争的环境中，学习更强的策略稳定性。**

### 6.2 Nov 17：LLM 时代的 multi-agent system 更接近系统课题

`Nov 17` 官方只公开了题目 `Multi-Agent Systems in the Era of LLMs` 和录播，没有公开 slides/readings。因此这里必须保守。最稳妥的解释是：课程在 `Oct 20` 的 self-play / game-theoretic view 之后，又安排了一讲更系统导向的 multi-agent session，说明 multi-agent 已经不只是理论点缀，而是 Berkeley 认为值得单独展开的主轴。

因此，把这两讲连起来看，Fall 2025 对主书的增量不是“multi-agent 很重要”这种空话，而是：

- multi-agent 是 agent scaling 的独立方向
- multi-agent 需要新的稳定性与 exploitability 语言
- multi-agent system 需要单独的 coordination / protocol / role design 视角

## 7. 第五条增量主线：agent 开始进入 scientific discovery，并把论文本身变成 agent

Spring 2025 主书第 11 讲已经谈到 `Abstraction and Discovery with Large Language Model Agents`。但 Fall 2025 把“discovery”进一步推向了更具体、更工程化的对象。

### 7.1 Nov 3：从 discovery 口号到 scientific workflow

`Nov 3` 的题目是 `AI Agents to Automate Scientific Discoveries`。官方 reading 一篇是 Nature 文章 `The Virtual Lab of AI agents designs new SARS-CoV-2 nanobodies`，另一篇是 `Paper2Agent: Reimagining Research Papers As Interactive and Reliable AI Agents`。

即使不借助未公开 slides，这组 reading 组合也足够传达一个清晰方向：

- 一条线走向 **AI agents 参与科学实验或设计闭环**
- 另一条线走向 **把论文、代码和方法包装成可交互、可测试、可复用的 paper agent**

`Paper2Agent` 的公开摘要尤其关键。它明确提出：系统可以用多个 agents 去分析论文和代码库，构造 `Model Context Protocol (MCP) server`，再通过自动生成和运行测试来提高可靠性。这不是简单的“论文问答机器人”，而是在把 research artifact 变成**带工具接口、能执行 workflow、能被验证的 agent interface**。

这对主书是一个很强的延伸。Spring 2025 已经让我们看到 agent 可以做 theorem proving、coding、web/GUI tasks；Fall 2025 则把 agent 往“研究基础设施”方向推进：  
**agent 不只是调用工具完成任务，它还可以成为论文、代码和科学方法的可操作封装层。**

### 7.2 这会怎样改写我们对 research assistant 的想象

读完这组内容之后，主书里的“research assistant”不应再被想成一个会 summarize paper 的模型，而应被想成一个更强的对象：

- 它理解论文
- 它能调用论文配套代码
- 它能重放或校验原文结果
- 它能在新 query 下执行可追溯 workflow

一旦做到这里，scientific agent 的核心能力就不只是检索和写作，而是**可靠地把 scientific artifact 转换成可操作的 agentic interface**。

## 8. 第六条增量主线：deployment 才是 agent 真正进入现实世界时最厚的一层

`Nov 10` 的 `Practical Lessons from Deploying Real-World AI Agents` 是 Fall 2025 最像“现实世界 agent 课”的一讲。Clay Bavor 的公开 slides 里最值得带回主书的是 `Agent Iceberg`。

表层当然还是熟悉的东西：

- LLM
- RAG
- tool use

但真正厚重的底层是：

- complex workflows & orchestration
- observability and monitoring
- regression testing
- user simulation
- prompt injection protection
- knowledge partitioning
- role-based access controls
- reporting & audit
- staging and release management
- model migration and failover

这张 `iceberg` 图的意义很大。它把 Spring 2025 主书中许多分散出现的安全、tooling、workflow、benchmark、memory、access control 线索，压缩成一个非常工程化的结论：  
**真正可部署的 agent，主要成本不在“会不会答题”，而在底层平台能不能持续地测试、观察、隔离、上线、回滚、审计和模拟。**

Clay Bavor 的 slides 还有一句很值得保留的话：`Easy to create demos, harder to make everything work consistently, with high quality and at scale.`  
这恰好是 Fall 2025 相比 Spring 2025 的重要差别。Spring 2025 已经让人相信高级 agent 在方法上是可能的；Fall 2025 则开始认真讨论：**为什么 demo 远远不是 product。**

## 9. 第七条增量主线：从多模态和 GUI 走向 embodied / autonomous agents

Spring 2025 主书已经有两讲很强的 multimodal material：

- `Multimodal Autonomous AI Agents`
- `Multimodal Agents – From Perception to Action`

这些章节已经把 perception-action loop、GUI grounding、OS/Web benchmarks 讲得很细。Fall 2025 的 `Dec 1 Autonomous Agents: Embodiment, Interaction, and Learning` 再往前走了一步：它不只是讨论屏幕上的交互，而是把 agent 放回 embodied / autonomous settings。

虽然官方没有公开 slides，但课程提供的两个 readings 已经把方向讲清楚了：

- `Outracing Champion Gran Turismo Drivers with Deep Reinforcement Learning`
- `SLAC: Simulation-Pretrained Latent Action Space for Whole-Body Real-World RL`

这说明课程在这里想要强调的，不是简单“机器人也能用 LLM”，而是：

1. embodied agent 的动作空间更连续、更受物理约束
2. 训练往往更依赖 simulation、latent action abstraction、control pipeline
3. interaction loop 从浏览器/GUI 的离散操作，转向与真实世界状态耦合的连续决策

对主书的意义是：  
**如果 Spring 2025 的 multimodal chapters 主要告诉我们 agent 怎样看懂屏幕并执行 GUI / OS 任务，那么 Fall 2025 开始把这个问题推进到真正的 embodied autonomy。**

## 10. 第八条增量主线：安全与安全性不再只是最后一讲的“风险提醒”

`Dec 8` 的 closing session 叫 `Agentic AI Safety & Security`。公开材料只有课程页和官方录播，没有公开 slides/readings，因此这里只能保守解释。

但即使如此，这一讲在课程结构里的位置本身就很有信息量：

- 它是整门课的收官讲次
- 课程页还专门给出时间调整提醒
- 它直接把 `Safety` 和 `Security` 放进标题

这说明 Fall 2025 对安全问题的态度，比“顺便提醒一下风险”更强。它更像在说：  
**当 agent 真的进入 training, evaluation, deployment, scientific workflow, multi-agent interaction 和 embodied setting 之后，安全与安全性必须作为 system-level closure 被重新处理。**

这与 Spring 2025 主书的最后一讲形成连续性。主书已经非常强调 privilege separation、prompt injection、memory poisoning、information flow tracking 和 formal verification；Fall 2025 则把安全再次放到整门课的收尾位置，等于在课程结构上重申：**没有安全闭环，前面所有 capability 讨论都不构成可部署的 agentic system。**

## 11. 读完这个 supplement 以后，主书最应该如何重构

如果要把 Spring 2025 主书升级成更贴近 2025 年底 agent 版图的 textbook，我建议至少做五个结构性调整。

### 11.1 把 evaluation 提前成全书前置框架

不要只在各章末尾列 benchmark。应该在书前面单独讲：

- capability eval
- environment eval
- verifier design
- statistical reliability
- deployment-oriented eval

因为 Fall 2025 已经表明，evaluation 不是附属工具，而是整个 agent harness 的骨架。

### 11.2 把 post-training 重写成“verifier + grader + environment + rubric”的系统问题

主书原本对 DPO/PPO/GRPO 的讨论应该保留，但还需要明确：

- preference alignment 不是 agent training 的终点
- verifiable reward 和 environment interaction 才是 agentic post-training 的下一层
- 真正困难的是 grader composition 和 product constraints

### 11.3 给 deployment 单独留出一个厚章节

`Agent Iceberg` 说明 deployment 不是“工程实现细节”，而是 agent 能否真实工作的主要决定因素之一。未来版本的主书应该把以下内容集中讲清：

- orchestration
- observability
- regression testing
- user simulation
- access control
- audit
- release management

### 11.4 把 multi-agent 从“拓展话题”升成独立 part

Fall 2025 显示 multi-agent 至少包含两层：

- strategy / equilibrium / self-play
- protocol / system / coordination

这已经不是 planning 章节里顺带一提就够的内容。

### 11.5 给 scientific discovery 和 embodied agents 各自留出更明确的落点

Spring 2025 的 discovery 和 multimodal material 已经很强，但现在还需要更明确地区分：

- scientific / research agents
- paper / code / method agents
- GUI / OS agents
- embodied / autonomous agents

它们共享“agent”这个词，但环境、反馈、动作空间、风险和验证方式都不相同。

## 12. 本章小结

如果只用一句话概括 Berkeley `Agentic AI (Fall 2025)` 对 Spring 2025 主书的增量，那就是：

> **它把 advanced LLM agents 从“能力模块的集合”进一步推进成“需要训练、评测、部署、治理和群体交互的系统工程对象”。**

更具体地说，Fall 2025 至少完成了八个推进：

1. 把 `evaluation` 从结果汇报提升为 harness 主轴  
2. 把 `post-training` 从 preference optimization 推向 verifier-centered agent training  
3. 把 `system design / infra` 推到 agent 讨论中心  
4. 把 `multi-agent` 提升为 population-level dynamics 问题  
5. 把 `scientific discovery` 推向可执行的 paper agents / virtual labs  
6. 把 `deployment` 写成可观测、可测试、可审计的产品栈  
7. 把 `multimodal` 继续推进到 embodied / autonomous agents  
8. 把 `safety & security` 作为整门课的收束，而不是尾声装饰

## 13. 局限与后续工作

本 supplement 是扎实的课程级补章，但仍然有三个残余风险：

1. `Sep 8`, `Nov 3`, `Nov 17`, `Dec 1`, `Dec 8` 没有完整的公开 slides 或 transcript 层，因此这些部分必须比 Spring 2025 主书更保守。
2. `Sep 29` 的两篇官方 OpenAI readings 在当前环境下只能稳定拿到标题和 URL，不能稳定抽取正文。
3. 本 workspace 只做 course-level extension，没有为 Fall 2025 每一讲重建新的 `transcript.jsonl / coverage_units.jsonl / lecture.tex` 流水线。

因此，本章最适合的角色是：**作为 Spring 2025 主书的高质量“后续课程增量章”**。如果后续要把 Fall 2025 完整升格成第二本教材，就需要再为每一讲单独跑 lecture-level harness。
