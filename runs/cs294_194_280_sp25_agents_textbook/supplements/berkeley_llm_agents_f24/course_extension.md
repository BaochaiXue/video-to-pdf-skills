# 补章：伯克利 Fall 2024《Large Language Model Agents》如何扩展 Spring 2025 主教材

## 1. 补章定位与来源边界

这一补章不是要替代 `/runs/cs294_194_280_sp25_agents_textbook` 已经完成的 Spring 2025 主教材，而是要说明：如果把 `CS294/194-196: Large Language Model Agents (Fall 2024)` 作为前传和侧翼材料纳入阅读，现有教材会在哪些维度变得更完整。这个问题很重要，因为 Spring 2025 的公开课程明显更偏向“advanced”一侧：它把大量篇幅投向推理时计算（inference-time computation）、后训练推理（post-training reasoning）、formal mathematics、theorem proving 与 safe/secure agentic AI。Fall 2024 则把视角更多放在 agent 的工程系统面：framework、compound systems、enterprise workflows、software agents、robotics、evaluation policy 与 open-source 生态。

本补章只使用官方且公开可访问的来源：

- 伯克利 Berkeley RDI 官方课程页：`https://rdi.berkeley.edu/llm-agents/f24`
- 公开 MOOC 页：`https://llmagents-learning.org/f24`
- 2026 年 4 月 16 日观察到的公开重定向目标：`https://agenticai-learning.org/f24`
- Berkeley RDI 官方 YouTube playlist：`https://www.youtube.com/playlist?list=PLS01nW3RtgopsNLeM936V4TNSsvvVglLc`
- 课程页中列出的每讲公开 slides、public edited videos 和 public readings

需要明确两类缺口。第一，官方课程页里的 `Original Recording` 指向 Berkeley `bCourses`，这属于官方来源，但不是公开来源，因此不能用作这个 supplement 的证据层。第二，官方 playlist 有 `12` 个公开 lecture 视频，以及 `2` 个隐藏的 private videos；这两个 private items 公开元数据里只有 opaque video id，没有标题、字幕或内容，因此只能如实记录为 access gap，不能假装覆盖。

因此，这一补章的目标不是“逐分钟复刻 Fall 2024 全课程”，而是“基于可公开审计的官方 source-of-truth，把 Fall 2024 对 Spring 2025 的结构性增量重写成教材章节”。

## 2. 为什么 Spring 2025 还需要 Fall 2024 这一层

如果只读 Spring 2025 主教材，读者会得到一条非常强的主线：高级 LLM agent 的核心不只是会不会调用工具，而是如何在推理时组织额外计算、如何借由 verifier 或环境反馈提升 reasoning、如何在 formal environment 中把语言模型变成可检验的 proof-search system、以及如何在 deployment boundary 上重新定义安全与权限控制。这条主线非常适合研究“advanced agents”。

但在工程上，很多团队真正先遇到的问题不是 theorem prover，而是这些更“朴素”却更常见的系统设计问题：

- agent 到底应该是单模型加 prompt，还是多个角色协作的 workflow？
- retrieval、grounding、memory、tool state 和 execution trace 应该如何进入系统边界？
- 如何把 prompt 组织成可优化、可替换、可评估的 program，而不是一段越来越长的自然语言？
- 为什么 enterprise workflows、software agents 和 robotics 会迫使我们重新思考 planning、evaluation 和 failure recovery？

Fall 2024 的价值正在这里。它不是把 Spring 2025 再讲一遍，而是把许多在 Spring 2025 中被默认掉的“agent infrastructure 与 workflow engineering 前提”补齐。换句话说，Spring 2025 更像是“高级专题书”，而 Fall 2024 像是“agent systems builder 的系统化前导层”。两者拼起来，教材才真正有 textbook 味道，而不是只剩若干高阶 lecture 的并集。

## 3. 从推理到 agent 定义

### 3.1 `LLM Reasoning` 不是旁支，而是 agent 的最小认知前提

Fall 2024 第一讲 `LLM Reasoning` 对应的 readings 非常能说明课程的切入方式：`Chain-of-Thought Reasoning Without Prompting`、`Large Language Models Cannot Self-Correct Reasoning Yet`、`Premise Order Matters in Reasoning with Large Language Models`、以及 `Chain-of-Thought Empowers Transformers to Solve Inherently Serial Problems`。它们共同强调一个事实：智能体系统的外部动作能力，最后仍要落回模型内部是否能维持多步推理、在错误中恢复、以及在顺序结构中保持稳定性。

这和 Spring 2025 的第一部分并不冲突，反而是前置铺垫。Spring 2025 会更直接讨论 self-consistency、Tree of Thoughts、verifier-guided reasoning、judgment learning、IRPO 等更先进的做法；Fall 2024 则先把问题设置得更基本也更工程化：如果一个模型在前提顺序变化时就可能失稳，在“自我纠错”这件事上又没有想象中可靠，那么任何上层 planner、tool loop 或 workflow 都不能被浪漫化。换句话说，agent 不是在一个完美 reasoner 之上再加工具，而是在一个能力有边界、对输入组织敏感、且需要外部 scaffolding 的 reasoner 之上做系统工程。

### 3.2 `LLM agents: brief history and overview` 把“会想”改写成“会与环境交互”

第二讲用 `WebShop` 与 `ReAct` 把课堂从 reasoning 带进 interaction。这里最重要的不是“又多了一种 benchmark”，而是 agent 的定义被改写了。一个仅仅输出答案的模型，与一个需要观察环境、生成动作、读取反馈、再修正下一步的系统，在 evaluation 和 architecture 上都是不同对象。

`WebShop` 让人看到语言模型可以被放进近似真实的任务环境中，目标不再是句子级正确，而是任务级完成；`ReAct` 则让 reasoning trace 和 action trace 合流，把“想”与“做”写成同一条轨迹。Spring 2025 的主教材当然会继续谈 planning、memory、environment feedback，但 Fall 2024 的这一步更像定义了 agent 问题本身：agent 不是 prompt 写得更长的 chatbot，而是带状态、带动作、带反馈闭环的部署对象。

## 4. Frameworks、tool use 与 compound systems

### 4.1 AutoGen 与 StateFlow：多智能体协作不是魔法，而是接口协议

Fall 2024 第三讲的一个核心贡献，是把“多智能体协作（multi-agent collaboration）”从口号拉回工程语义。`AutoGen` 不是因为“多模型聊天”这件事本身就高级，而是因为它把角色分工、消息路由、终止条件、人工介入点和工具调用接口显式化。`StateFlow` 更进一步：它提醒我们，真正可控的 workflow 往往需要 state-driven execution，而不只是自由对话。

这对 Spring 2025 主教材是实打实的补强。Spring 2025 更关注 reasoning 质量、formal verification 和安全边界，而 Fall 2024 在这里补上的，是 agent framework 为什么必须存在。一个复杂 agent 系统如果没有显式状态机、角色契约、消息边界和可记录 trace，那么所谓的“planning”最后往往只是 prompt 里的一段愿望。

### 4.2 `Building a Multimodal Knowledge Assistant` 把知识接入问题从检索扩展到系统编排

这节的价值在于，它没有把多模态助手写成单个模型能力的展示，而是把文档接入、知识索引、跨模态检索与回答组织看成一整套 pipeline。Spring 2025 的多模态章节会更强地讨论 perception-action loop、GUI grounding、web/OS environment，但 Fall 2024 在这里提前强调了一件常被忽视的事：一旦系统必须读文档、看图、定位证据、再组织回答，agent 设计的重心就会从“模型是否足够强”转向“信息流是否足够干净”。

这也是为什么 Fall 2024 对现有教材的补充不只是“再加一个多模态案例”，而是补上了知识接入型 agent 的基础设施视角。

### 4.3 Enterprise grounding 与 long-context stress test：工具链的价值在于减少幻觉，而不只是增加外部调用

`Enterprise trends for generative AI...` 这一讲把 grounding、RAG 和 long-context evaluation 放到企业语境里。这里最值得吸收的不是某个厂商实现细节，而是问题的重心：在生产环境中，agent 失败往往不是因为模型不会说，而是因为系统没有把可核验信息、上下文窗口和任务状态管理好。

课程 readings 中关于 Vertex grounding 和 `Needle in a Haystack` 的材料，实际上在提醒读者两件事。第一，检索增强（retrieval augmentation）不是简单把更多文档塞进上下文，而是要解决“哪些证据进入推理过程、以什么粒度进入、如何证明模型真的利用了它们”。第二，长上下文本身不是 agent 能力的终点，因为上下文再长，也不意味着系统拥有稳定的检索、选择和更新策略。

### 4.4 DSPy：把 prompt 从文本工艺改成可优化程序

`Compound AI Systems & the DSPy Framework` 对 Spring 2025 主教材的增益尤其明显。Spring 2025 很强，但它更多围绕 reasoning、formal search 与 safety 展开；Fall 2024 在这里补足的，是“programmatic agent engineering”的一整套语言。DSPy 的重要性不只是某个框架本身，而是它把 prompt、demonstration、module composition 和 evaluator 变成可搜索、可调优、可替换的程序对象。

这件事为什么重要？因为一旦系统不再是“一个 prompt 对一个模型”，而是“多模块、多阶段、有中间表示、有评测回路”的 compound system，那么 agent 优化也不该继续停留在 prompt craftsmanship。课程 readings 里“instruction and demonstration optimization”以及“prompt optimization + fine-tuning 互补”恰好说明：真正可扩展的 agent，往往既需要前端 workflow 设计，也需要后端 parameter update 或 offline optimization。Fall 2024 因而把 Spring 2025 中隐含的 harness intuition 明确化了。

## 5. 企业工作流、coding agents 与评测基础设施

### 5.1 `Agents for Software Development`：把 coding agent 看成可执行环境中的闭环控制系统

Spring 2025 主教材已经有非常强的 coding agents 与 vulnerability detection 章节，但 Fall 2024 的 `Agents for Software Development` 仍然提供了另一层必要视角。`SWE-agent` 与 `OpenHands` 的关键不在于“让模型会写代码”，而在于它们把代码仓库、shell、测试、报错日志和 patch history 组成一个真实的 execution environment。在这里，agent 的每一步都要接受环境反馈，而不是停留在自然语言评分。

这类系统告诉我们：coding agent 的难点不只是 program synthesis，而是如何管理 action space、如何决定何时读取更多上下文、何时运行测试、何时回滚假设、何时承认当前策略失败。Spring 2025 更强调安全、漏洞分析和 benchmark 压力测试；Fall 2024 则把“软件开发 agent 的基本作业系统”先摆在台面上。两者结合后，读者才会明白为什么 coding agent 是 agent research 的高价值试验场。

### 5.2 `AI Agents for Enterprise Workflows`：工具使用不该只是 API 调用，而要上升为任务制度

`WorkArena`、`WorkArena++` 与 `TapeAgents` 这一组材料，实际上把“agentic workflow”从 demo 级别推向组织级别。这里的重点不再是一个 agent 会不会调用 calendar、email 或 CRM API，而是：当任务跨越多个页面、多个表单、多个审批状态和多轮上下文时，系统如何保存状态、如何决定 subgoal、如何记录 trace、以及如何将失败样例反馈回优化回路。

这与 Spring 2025 主教材的关系非常直接。Spring 2025 强调 reasoning、memory、planning，但 Fall 2024 在企业工作流里告诉我们，planning 不是抽象词，而是会体现在真实任务分解、权限边界、异常恢复和 benchmark design 中。`WorkArena++` 之所以重要，就是因为它把组合式任务（compositional tasks）带入 agent evaluation；`TapeAgents` 则进一步表明，agent 的研发流程本身也需要一套“记录、调试、比较、优化”的系统化接口。

### 5.3 这两讲如何重写主教材的 Part II

如果把 Fall 2024 融进现有 Spring 2025 textbook，那么 Part II 不应该只停留在“训练 recipe + coding/security”两章，而应当多出一层 agent systems builder 的 framing：同样是 tool use，enterprise workflow、software engineering 和 vulnerability analysis 关注的并不是同一类工具，也不是同一类 failure。前者更关注状态编排和组织流程，后者更关注执行反馈和 correctness，安全章节则更关注 capability escalation 与 misuse surface。Fall 2024 的作用，就是把这些不同任务族的共性与差异提前交代清楚。

## 6. Robotics、具身环境与多模态扩展

### 6.1 `Project GR00T` 让 agent 从屏幕走向具身环境

Spring 2025 的多模态章节已经覆盖 web、GUI、OS 和 perception-to-action，但 Fall 2024 的 `Project GR00T` 进一步把讨论推进到 robotics。这里的关键变化是：环境不再只是网页 DOM 或屏幕截图，而是物理动作、感知噪声、奖励设计、sim-to-real 差距与具身学习。

这使得 `Voyager`、`Eureka` 与 `DrEureka` 这些 readings 非常关键。它们共同提示：在 robotics 中，LLM agent 的价值并不只是“直接输出动作”，更可能体现在奖励函数编写、任务分解、探索策略组织、经验回放语言化，以及 sim-to-real policy adaptation。也就是说，语言模型在这里既可能是 planner，也可能是 reward designer、 curriculum designer，甚至是实验自动化接口的一部分。

### 6.2 这如何扩展 Spring 2025 的 multimodal chapters

Spring 2025 的多模态部分更强调 perception error、grounding error 和 environment feedback 在 web/OS benchmark 中如何相互作用。Fall 2024 的 robotics 讲次则把这些问题提升到更极端的环境：动作成本更高、反馈更延迟、状态不可完全观测、误差更难回滚。因而，Fall 2024 不是重复 Spring 2025 的多模态内容，而是把“agent 与 environment 的耦合”推进到了具身控制层。

## 7. 神经符号决策、robotics 与多智能体协同

### 7.1 `Towards a unified framework of Neural and Symbolic Decision Making`：把 planning 从 prompting 拉回 search 与 optimization

Yuandong Tian 这讲的重要性在于，它让课程从“agent 会不会调用工具”回到“agent 如何组织搜索与决策”这一更底层的问题。`Beyond A*`、`Dualformer`、`Composing Global Optimizers...`、`SurCo` 这些 readings 共同指向一件事：高质量 agent 不一定只靠更好的自然语言 reasoning 产生，它还可能来自更好的搜索动态、更好的优化器组合、更强的 surrogate 近似以及神经方法与符号结构的混合。

对 Spring 2025 主教材来说，这一讲提供了一条很有价值的横向连接。Spring 2025 会把 theorem proving、proof search、verification environment 讲得更深；Fall 2024 则在 formal math 之前就先告诉读者，planning 和 decision making 从来不是只有“让模型想更久”这一条路。agent 的 intelligence 也可以来自对 search space、state abstraction 和 optimization object 的重写。

### 7.2 多智能体协同在 Fall 2024 里如何出现

用户特别要求这一补章解释 `multi-agent collaboration`。需要诚实地说，Fall 2024 公开课程里并没有一讲标题就叫 “Multi-Agent Collaboration”。但这并不意味着这一主题缺席。它主要以三种形式出现：

第一，`AutoGen` 直接把 multi-agent conversation 作为系统抽象；第二，`StateFlow` 和 enterprise workflow 把多角色、多阶段执行写成显式状态机；第三，coding agents 与 robotics systems 都隐含了 planner、executor、verifier、tool interface、memory store 等角色分离。也就是说，Fall 2024 更像是在教“如何把协作写进系统接口”，而不是把 multi-agent 当作单独的理论标签。

这恰好补足 Spring 2025 的一个空白：Spring 2025 更关注 advanced reasoning 和 formal environments，对 workflow-level role decomposition 没有花同等篇幅。Fall 2024 因而提供了一个更适合工程读者的过渡层。

## 8. 开放生态、评测、RSP 与安全可信 agent

### 8.1 `Open-Source and Science in the Era of Foundation Models`

Percy Liang 这讲把课程从“怎么做 agent”拓展到“agent 生态如何扩散能力与风险”。这一步很重要，因为 agent 研究不只发生在单个 lab 或 benchmark 内部。open-source release、科学研究 workflow、以及像 `CyBench` 这样的风险评测，都在改变我们对“能力扩散”与“可审计性”的理解。

如果说 Spring 2025 的主体是“advanced agents can do more with reasoning, verification, and environment feedback”，那么 Fall 2024 在这里补上的问题是：“这些能力一旦进入开放生态、进入科研自动化或高风险 domain，谁来定义 acceptable deployment boundary？” 这使得 open-source 与 science 不再只是政策附录，而是 agent systems 的真实工程条件。

### 8.2 `Measuring Agent capabilities and Anthropic’s RSP`

Ben Mann 这讲的重要性在于，它明确把 capability measurement 与 deployment policy 并置。很多课程会把 benchmark 当成技术问题、把 policy 当成外部治理问题；但对 agent 来说，两者在实践中是连着的。因为一旦 agent 能在 computer-use、tool-use 或 environment navigation 上完成更复杂任务，评测结果就会直接影响允许部署的范围与安全门槛。

这对 Spring 2025 final safety chapter 是实质性补充。Spring 2025 更强调 prompt injection、memory poisoning、privilege separation 和 system boundary；Fall 2024 在这里则补上“为什么这些系统边界最终要进入 scaling policy、risk tier 和 release governance”。

### 8.3 `Towards Building Safe & Trustworthy AI Agents`

Dawn Song 这讲则把安全问题拉回更具体的研究对象：`DecodingTrust` 说明 trustworthiness 不能只看单一 benchmark；`Representation Engineering` 暗示我们可以尝试从内部表征层面理解和干预模型行为；Carlini 系列关于训练数据抽取与 unintended memorization 的工作，则把 agent safety 和 privacy leakage 联系起来。换句话说，agent 的风险不只来自它能不能调用危险工具，也来自模型本身的记忆、泛化与可操控性。

因此，Fall 2024 的安全部分与 Spring 2025 的安全部分不是重复关系。前者更像“评价、政策、隐私、trustworthiness”的外围扩展，后者更像“agent architecture 的内核防线”。两者合并后，教材的安全叙事才完整。

## 9. 如何把 Fall 2024 融入当前 Spring 2025 textbook

如果要把这套 supplement 真正合进现有教材，我建议不是新增一个完全平行的第二本书，而是做三层并入。

第一层是“前传补章”。把 Fall 2024 的 reasoning、history/overview、AutoGen/StateFlow、DSPy 四块内容放在 Spring 2025 的 Part I 和 Part II 之间，作为 agent infrastructure primer。这样读者在进入更 advanced 的 inference-time compute、open training recipes、coding security 之前，已经先知道 workflow、framework 和 compound systems 为什么重要。

第二层是“横向插页”。在 Spring 2025 的 coding agents、multimodal agents、安全与 security 章节之后，各插入一节 “Fall 2024 extension”。前者接 `SWE-agent/OpenHands/WorkArena/TapeAgents`，中间接 `Project GR00T/Voyager/Eureka/DrEureka`，后者接 `CyBench/RSP/DecodingTrust/Representation Engineering`。这样做的好处是，主教材仍然保持 Spring 2025 为主线，但读者在每个核心主题后都能看到 Fall 2024 的系统级展开。

第三层是“阅读路径重排”。对工程读者，我会建议阅读顺序改成：Fall 2024 的 `history/overview -> AutoGen/StateFlow -> enterprise grounding -> DSPy -> software agents -> enterprise workflows`，然后再回到 Spring 2025 的 reasoning、memory/planning、coding security、multimodal、formal math 和 safe agentic AI。这样更符合 builder 的学习曲线。

## 10. 公开来源缺口与诚实边界

这一补章还有三个需要明确保留的边界。

第一，它没有使用 Berkeley `Original Recording`。这意味着一些课堂 Q\&A、现场延伸和未经编辑的解释细节不会进入本章。

第二，它没有覆盖 playlist 里两个 hidden private videos 的内容，因为公共元数据无法告诉我们它们到底是什么。既然不知道，就不应编造。

第三，它虽然覆盖了 `planning、tool use、compound systems、coding agents、robotics、evaluation、multi-agent collaboration、safety` 这些主题，但这种覆盖是“course-level synthesis”，不是 `12` 个 lecture 的逐分钟复刻。对于已经有 Spring 2025 主教材的当前项目，这种做法是合理的：我们需要的是一个能把 Fall 2024 纳入书中结构的补章，而不是再复制一遍整个项目的 lecture harness。

在这个意义上，Fall 2024 对 Spring 2025 的最好定义不是“旧版课程”，而是“让 advanced agent textbook 更像一本完整教材的系统工程层”。Spring 2025 负责把智能体研究推向 reasoning、formal verification 和 secure deployment 的前沿；Fall 2024 则把这些高级主题落回 framework、workflow、evaluation、enterprise tooling、robotics 与 policy 的现实地面。两者合起来，读者既能看到 agent 的 frontier，也能看到 agent 何以成为可部署系统。
