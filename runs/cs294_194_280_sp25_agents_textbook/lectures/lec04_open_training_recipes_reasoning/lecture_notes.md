# Open Training Recipes for Reasoning in Language Models

## 本讲学习目标

- 理解为什么 Hanna Hajishirzi 把“开放、可复现、可审计”视为训练 recipe 的组成部分，而不是附属价值观。
- 理解现代 open recipe 如何把预训练、监督微调、偏好优化、RLVR 和 test-time scaling 接成一条连续管线。
- 分清楚 SFT、DPO、PPO、RLVR 各自解决的问题、需要的监督信号、工程代价与失败模式。
- 看懂 `s1 / s1K / budget forcing` 为什么是本讲中关于 reasoning 的最小 recipe。

## 1. 背景与问题设置

Hanna Hajishirzi 在开头并没有直接进入“如何做 reasoning model”，而是先论证为什么开放生态仍然是语言模型科学的前提。她的论点很硬：如果研究者只能看 closed API 的最终表现，却无法访问训练数据、配方、模型权重和中间决策，那么很多“最好的工程经验”就会变成不可验证的 folklore。对课程而言，这一点尤其重要，因为 agent 与 reasoning 不是单一 trick，而是一串相互作用的 recipe 选择：训练数据怎么来、偏好信号如何构造、什么时候应该用 RL、什么时候应该把预算放到 test-time。

这也是本讲和第一讲的直接衔接处。第一讲讲的是 inference-time techniques；这一讲则把视角向前推一步：如果你想让模型在测试时更会 reasoning，训练阶段应该先准备什么。

## 2. 现代 open recipe 的总流程

Slides 把整个路线拆成三个大阶段：pre-training、post-training、test-time inference。Hajishirzi 的关键观点不是“某个阶段最重要”，而是这三个阶段相互补位：

- 预训练负责通用语言建模与知识底座。
- 后训练负责把模型塑造成会聊天、会遵循指令、会做 reasoning、会用工具的助手。
- 测试时推理负责在固定模型权重下，继续用额外计算换质量。

Tulu 3 的 staged recipe 很适合作为教材化抽象：先做 instruction tuning，再做 preference tuning，然后在适合的任务上做 reinforcement learning with verifiable rewards。这个结构的优点在于，监督信号越来越昂贵但也越来越针对性。越往后走，目标越像“打磨特定能力”，而不是“全面重塑模型”。

### 2.1 监督微调与数据配方

监督微调（supervised finetuning, SFT）在 slides 里被讲得很朴素：给模型 prompt-completion 对，最大化正确输出的条件似然。但 lecture 很强调，真正难的不是优化目标，而是数据 recipe。原因有三层：

第一，能力目标并不单一。聊天、知识问答、reasoning、coding、安全、multilinguality 彼此并不等价，混在一起训练时会互相竞争预算。

第二，reasoning 数据不是“更多最终答案”就够了。Slides 多次强调 chain-of-thought 数据的价值：它不仅让模型更容易处理多步问题，还把错误暴露在轨迹层面，使后续筛选和验证成为可能。

第三，高质量 reasoning data 非常贵，所以需要混合 recipe。lecture 讨论了 Self-Instruct、已有公开 instruction datasets、hybrid preferences、persona-driven synthesis 以及数据过滤。这里最重要的设计原则不是“全都加”，而是：

- 先按目标能力设 evaluator。
- 再按 evaluator 反推需要的 query 分布和数据来源。
- 用筛选、去重、许可证检查、decontamination 约束数据质量。

Persona-driven data generation 很关键，因为它代表了一种更系统的扩展方式：不是盲目采样更多 synthetic data，而是先定义 persona / capability，再控制题目难度、风格与领域覆盖。slides 里的 math、grade-school math、coding、precise instruction following 例子都在说明这点。

此外，lecture 明确肯定了用 voting / self-consistency 去过滤 reasoning traces。这里的核心并不是“多数票一定正确”，而是它给了一个便宜、可并行、和最终答案强相关的 quality heuristic。对开放 recipe 来说，这很重要，因为很多团队没有预算做大规模人工标注。

### 2.2 Preference tuning：DPO 与 PPO 的关系

Lecture 的另一条主线，是把 RLHF 彻底拆开讲清楚。Hajishirzi 没有把 DPO 和 PPO 当成宗教之争，而是放到同一个 recipe 框架里：

- 你先要有 prompts 和 candidate responses。
- 然后你要有 preference data，说明哪个回答更好。
- 你可以再训练 reward model。
- 最后用某种 policy optimization 方法让模型偏向被偏好的回答。

在这个框架下，PPO 和 DPO 的区别就清楚了。PPO 走的是“reward model + RL”这条较重路径。它的优点是表达力强、通常结果更好；代价是实现复杂、吞吐更差、显存压力更大。DPO 走的是“直接在 preference pairs 上更新策略”的较轻路径。它的优点是简单、便宜、适合快速迭代；缺点是常常会输给更认真调过的 PPO。

Lecture 的态度和 reading《Unpacking DPO and PPO》是一致的：算法当然重要，但并不是唯一决定因素。高质量偏好数据通常比“DPO 还是 PPO”更重要；更大的 reward model 也不是稳定有效的银弹；policy training prompt 的构造同样会影响结果。

这段内容对 agent 课程很重要，因为很多工程团队把“用了 DPO”误当成 recipe 已经完整。实际上 preference learning 只是在 pipeline 中解决“输出更符合偏好”的一层问题，它不能自动解决 reasoning correctness、tool grounding 或 verification。

## 3. RLVR：当 final answer 可验证时，为什么要上 RL

Lecture 进入 RL with verifiable rewards（RLVR）时，逻辑非常清楚：如果任务拥有可验证的最终答案或明确约束，例如 GSM8K、MATH 或 instruction following checklists，那么训练时未必要先学一个 reward model。你可以直接写 verification function，按最终答案是否满足约束给 reward。

这一步解决的是 preference tuning 的一个现实痛点：很多 reasoning 任务的中间链路很难标，偏好数据也未必稳定，但最终答案经常是可判断的。于是训练就从“学会像人类喜欢的回答那样说话”，转成“在环境奖励下学会产生更高成功率的最终答案”。

不过 lecture 也明确讲了 RLVR 的边界：

- 奖励往往是二值或稀疏的，因此优化不稳定。
- 它更适合有 gold answer 或 rule-based verifier 的任务，不适合开放式闲聊。
- 它会把优化压力集中到最终可验证结果上，不保证中间 chain-of-thought 一定人类可读或语义优雅。

这也是为什么 slides 一直把 verifier 和 dataset 配对着讲。不是“有了 RL 就自然会 reasoning”，而是“当任务可以被验证时，RL 能放大已有 reasoning base model 的潜力”。

## 4. s1 与 budget forcing：最小 reasoning recipe

本讲后半段最有意思的地方，是 Hajishirzi 把 test-time scaling 接回训练 recipe，讨论 `s1`。核心思想异常朴素：

- 用少量但高质量、困难且多样的 reasoning 样本（s1K）训练一个 sample-efficient reasoning model。
- 在推理时，不只是 sample 更多答案，而是通过 budget forcing 强制模型延长思考轨迹。

这说明 reasoning 并不一定先靠巨量新数据才能起飞。只要基础模型足够强，少量高质量 reasoning data 加合适的 test-time scaling 机制，就能带来可观增益。Slides 甚至把它称为 minimal recipe。

这里最重要的教材结论有两个。

第一，test-time scaling 不是纯部署技巧，它会反向影响训练 recipe 的设计。因为你需要让模型习惯产生可延展、可继续、不会一延长就崩掉的 reasoning trace。

第二，budget forcing 和第一讲的 inference-time compute 讨论完全接上：你真正分配的是计算预算，而不是某个固定 prompt 模板。顺序扩展、并行采样、rejection sampling、conditional control，本质上都在讨论怎么把有限预算投到更高价值的推理步骤上。

## 5. 关键 readings 与课程主线的连接

### Tulu 3

Tulu 3 是 lecture 的主 reading，因为它把开放 post-training 从口号变成 recipe。它最重要的贡献不是某一个 trick，而是把多个阶段的经验写成可复现工艺：数据整理、SFT、preference tuning、reasoning-oriented RL，再加上评估与开源基础设施。

### Unpacking DPO and PPO

这篇 paper 为 lecture 里关于 DPO/PPO 的讨论提供了实证地基。它反复表明：真正的 recipe engineering 不是在两种算法名词之间站队，而是识别哪个环节是当前瓶颈。很多情况下，优先级顺序是数据质量 > learning algorithm > reward model choice > prompt details，而不是反过来。

### OpenScholar

OpenScholar 并不是这场 lecture 的核心算法对象，但它提供了一个很好的“开放 recipe 的 downstream 价值”案例：开放模型、开放检索基础设施和 citation-grounded synthesis 能让科学文献整理变得更可靠。这说明 open ecosystem 的意义不仅在于训练本身，还在于让高置信度 scientific agents 成为可能。

## 6. 失败模式、边界条件与前后讲关系

本讲最容易被误读的地方有三个：

- 误读一：以为 post-training recipe 只是在“把 base model 调得更会聊天”。实际上 lecture 明确把 reasoning、tool use、safety 和 test-time scaling 都纳入 recipe。
- 误读二：以为 DPO / PPO 的选择决定一切。lecture 和 reading 都指出，高质量数据和评估往往更关键。
- 误读三：以为 RLVR 会自动学会“正确思考”。lecture 实际只主张：当 final answer 可验证时，RLVR 是一个非常有力的放大器。

与前一讲的关系很直接：L01 讲 inference-time techniques；L04 解释为什么某些 test-time technique 能在训练之后继续扩展。与后一讲的关系也很清楚：L05 会把这种 recipe 思维带进 coding agents 与 vulnerability detection，讨论 evaluator、environment feedback 和 agent workflow。

