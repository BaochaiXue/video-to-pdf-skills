# Canonical Notation And Glossary

This file resolves terminology and symbol conflicts across the 12 lecture workspaces for `CS294/194-280: Advanced Large Language Model Agents, Spring 2025`.

## Governance

- `reasoning`, `planning`, `search`, and `inference-time computation` are not synonyms.
- `post-training` is the parent category; `RLHF`, `DPO`, `PPO`, `GRPO`, and `RLVR` are different training mechanisms inside or adjacent to that category.
- `tool use`, `function calling`, and `workflow` must stay distinct.
- `formalization`, `formal specification`, `autoformalization`, `verification`, `theorem proving`, and `proof search` must stay distinct.
- `multimodal agent`, `web agent`, `GUI agent`, `OS agent`, and `computer-use agent` form a hierarchy rather than interchangeable labels.
- `least privilege` and `privilege separation` must not be collapsed into one security term.

## Canonical Glossary

| English Term | Preferred Chinese | Canonical Definition | Scope |
| --- | --- | --- | --- |
| inference-time computation | 推理时计算 | Test-time budget spent on longer reasoning, broader sampling, deeper repair loops, verifier calls, or search. | L01, L04, L06, L10 |
| reasoning | 推理 | Producing, evaluating, or revising intermediate steps toward an answer or action. | Course-wide |
| planning | 规划 | Explicit lookahead, decomposition, or model-based choice over future actions. | L03, L06, L07 |
| search | 搜索 | Enumerating, expanding, or ranking candidate reasoning paths, action trajectories, or proof branches. | L01, L06, L08, L10 |
| memory | 记忆 | Mechanisms that retain and retrieve useful state beyond the immediate context window. | L03, L07, L12 |
| post-training | 后训练 | The stage after pretraining that shapes task behavior through instruction tuning, preference optimization, RL, or related alignment methods. | L02, L04 |
| RLHF | 基于人类反馈的强化学习 | A family of post-training pipelines that use human preference or judgment signals, often via reward modeling plus policy optimization. | L02, L04 |
| DPO | 直接偏好优化 | Directly optimizes preference pairs against a reference policy without a separate online RL loop. | L02, L04 |
| PPO | 近端策略优化 | A concrete RL algorithm often used in RLHF pipelines; not a synonym for all preference learning. | L04 |
| GRPO | 组相对策略优化 | Group-relative policy optimization over multiple sampled candidates; keep the acronym explicit when used. | Course-wide |
| RLVR | 可验证奖励强化学习 | RL with rewards derived from an external verification function rather than a learned preference judge. | L04 |
| budget forcing | 预算强制 | A test-time control strategy that explicitly forces additional reasoning or continuation under a larger budget. | L04 |
| self-rewarding language model | 自奖励语言模型 | A model trained to both act and judge, so preference data can be expanded with model-generated evaluations. | L02 |
| IRPO | 迭代推理偏好优化 | Preference optimization specialized for reasoning tasks, typically using verifiable answers and reasoning-aware comparison data. | L02 |
| Meta-Rewarding | 元奖励学习 | Training the judge itself so that evaluation quality improves across iterations. | L02 |
| EvalPlanner | 评估规划器 | A judge that plans its evaluation process instead of assigning a score in one shot. | L02 |
| tool use | 工具使用 | An agent invoking an external executable capability such as search, execution, retrieval, browsing, or verification. | L01, L05, L12 |
| function calling | 函数调用 | A structured API invocation interface; narrower than general tool use. | Course-wide |
| workflow | 工作流 | A coordinated multi-step pipeline combining prompts, tools, control flow, validation, and repair. | L01, L05, L10 |
| coding agent | 编码智能体 | An agent that localizes, edits, executes, debugs, or validates code inside an environment. | L05 |
| agent-computer interface | 智能体-计算机接口 | The observation/action interface exposed to an agent, including tools, feedback shape, and guardrails. | L05 |
| procedural control | 程序式控制 | A design where the outer control flow is hand-authored and the model is called at specific substeps. | L05 |
| dynamic control | 动态控制 | A design where the agent chooses its next tool/action online based on intermediate feedback. | L05 |
| variant analysis | 变体分析 | Using known bugs, patches, or exploit patterns to search for related vulnerabilities in nearby code regions. | L05 |
| pass@k | 前 k 命中率 | The probability that at least one of the first `k` sampled programs is correct. | L05 |
| multimodal agent | 多模态智能体 | An agent that consumes or acts on multiple modalities such as text, images, GUI state, or video. | L06, L07 |
| web agent | Web 智能体 | An agent whose environment is primarily browser- or website-based. | L06 |
| GUI agent | 图形界面智能体 | An agent that grounds and acts over visible interface elements. | L07 |
| OS agent | 操作系统智能体 | An agent operating across desktop or system-level applications and files. | L07 |
| computer-use agent | 计算机使用智能体 | A multimodal agent that executes tasks in realistic OS or multi-application environments. | L07 |
| visual grounding | 视觉定位 | Aligning a language goal to concrete visual entities, UI elements, or layout cues. | L06, L07 |
| execution-based evaluation | 基于执行的评测 | Evaluation by checking environment state or executable outcomes rather than text similarity alone. | L06, L07 |
| guided replay | 引导式回放 | Recovering or synthesizing trajectories from tutorials or demonstrations to create training data. | L07 |
| CoTA | 思维-动作链 | Training data or modeling that jointly represents reasoning traces and action traces. | L07 |
| temporal encoder | 时序编码器 | A module that compresses long frame sequences into a smaller number of latent tokens. | L07 |
| synthetic agentic tasks | 合成智能体任务 | Automatically generated, environment-grounded tasks used to expand agent training distributions. | L06 |
| Plan-Seq-Learn | 规划-编排-学习 | A hierarchical recipe that separates language planning, scene sequencing, and low-level control. | L06 |
| formal mathematics | 形式化数学 | Mathematics written in a machine-checkable proof language accepted by a proof assistant. | L08, L10 |
| formal reasoning | 形式推理 | Reasoning performed inside an explicit formal system with machine-checkable states or constraints. | L09 |
| formalization | 形式化 | The general act of rewriting an informal object into a formal object. | L08-L11 |
| formal representation | 形式表示 | A machine-checkable representation of a problem, theorem, concept, or search state. | L11 |
| formal specification | 形式规范 | A precise formal statement of a theorem, task, or program requirement. | L08, L09 |
| autoformalization | 自动形式化 | Automatic conversion from informal mathematical statements or proofs into formal ones. | L09 |
| verification | 验证 | Checking whether a candidate output satisfies a formal rule, specification, or proof checker. | L02, L04, L08-L12 |
| theorem proving | 定理证明 | Constructing a valid proof inside a formal system. | L08-L10 |
| proof search | 证明搜索 | Algorithmic exploration over proof states, tactics, premises, or proof trees. | L08-L10 |
| theorem equivalence checking | 定理等价性检查 | Determining whether two formal theorem statements are semantically equivalent. | L09 |
| hammer | 锤式自动证明系统 | A system that combines premise selection, ATP invocation, and proof reconstruction for a proof assistant. | L10 |
| premise selection | 前提选择 | Choosing the relevant lemmas or definitions to expose to an automated prover. | L10 |
| thought-augmented proving | 思维增强证明 | Generating explicit informal thoughts before low-level tactics to bias proof search. | L10 |
| sketch-guided proving | 草图引导证明 | Using a high-level proof sketch to guide lower-level formal proving. | L10 |
| discovery agent | 发现型智能体 | An agent that proposes hypotheses, organizes search, uses feedback, and accumulates abstractions. | L11 |
| symbolic regression | 符号回归 | Searching for compact formulas or symbolic programs that explain data. | L11 |
| concept library | 概念库 | A reusable collection of abstractions discovered from successful hypotheses or proofs. | L11 |
| visual concept library | 视觉概念库 | A concept library specialized for reusable visual abstractions. | L11 |
| agentic AI safety/security | 智能体式 AI 安全与安全防护 | Safety and security problems arising once a system can use tools, persist state, and act over environments. | L12 |
| direct prompt injection | 直接提示注入 | Malicious instructions inserted directly into the prompt channel. | L12 |
| indirect prompt injection | 间接提示注入 | Malicious instructions carried by external data that later enters the prompt context. | L12 |
| memory poisoning | 记忆投毒 | Corrupting long-term memory or retrieval stores so malicious content is retrieved across tasks. | L12 |
| least privilege | 最小权限 | Granting only the minimum capability needed for the current step. | L12 |
| privilege separation | 权限分离 | Structuring the system so high-risk logic and high-privilege capability are isolated. | L12 |
| information flow tracking | 信息流跟踪 | Tracking how sensitive data propagates across components, tools, and outputs. | L12 |

## Conflict Resolutions

### Reasoning / planning / search / inference-time computation

- `inference-time computation` is the budget.
- `reasoning` is the internal deliberation or structured intermediate explanation.
- `search` is explicit exploration over multiple candidates.
- `planning` is future-oriented decomposition or action selection, often with a world or environment model.

### Post-training family

- `post-training` is the umbrella stage.
- `RLHF` names the broader human-feedback family.
- `DPO` and `PPO` are optimization choices, not task definitions.
- `RLVR` is distinct because its reward comes from external verification rather than preference judgments.

### Tooling stack

- `tool use` is the general behavior.
- `function calling` is one structured interface for tool use.
- `workflow` includes control flow, evaluation, retries, and repair around tool use.

### Formal stack

- `formalization` is the generic conversion act.
- `formal specification` is the target statement or requirement.
- `autoformalization` is automated formalization.
- `verification` checks an output.
- `theorem proving` produces a proof.
- `proof search` is the algorithmic process inside theorem proving.

### Multimodal stack

- `multimodal agent` is the parent class.
- `web agent`, `GUI agent`, and `OS agent` are environment-specific subclasses.
- `computer-use agent` is the broadest label for agents that operate realistic digital interfaces and apps.

### Security stack

- `least privilege` is a permission assignment rule.
- `privilege separation` is an architectural isolation rule.
- `prompt injection` and `memory poisoning` are attack mechanisms.
- `information flow tracking` is a monitoring and enforcement mechanism.

## Canonical Notation

| Symbol | Canonical Meaning | Local Overloads |
| --- | --- | --- |
| `x` | input query, prompt, or task instance | In L12 it may denote an attack seed; mark the security scope explicitly. |
| `y` | output, answer, response, or completion | In preference learning use `y_w`, `y_l` for winner/loser. |
| `\hat{y}` | aggregated or predicted final answer | Used in self-consistency or verifier-backed decoding. |
| `\pi_\theta` | current policy model | Stable across post-training lectures. |
| `\pi_{\mathrm{ref}}` | reference policy | Stable across DPO/RLHF discussions. |
| `r` / `r_\phi` | reward or verifier-derived score | Use subscripts when the scoring mechanism matters. |
| `V(x,y)` | verification function over an input-output pair | Do not confuse with value functions such as `V(s)` or `v_\phi(s)`. |
| `v_\phi(s)` | value estimate for a state | Preferred in search or planning settings. |
| `Q(s,a)` | action-value estimate | Used in search-heavy or RL-heavy lectures. |
| `\tau` | trajectory | In theorem proving this means a proof trajectory; state that specialization locally. |
| `\mathcal{T}` | set of trajectories or candidate traces | Use only when multiple trajectories are being compared. |
| `s` / `s_t` | state | In theorem proving this is a proof state; in security it can mean agent context state. |
| `a` / `a_t` | action | In theorem proving it can specialize to a tactic. |
| `o_t` | observation or tool output | Preferred in interactive agent environments. |
| `B` | budget | Use for inference-time or interaction budget only. |
| `\mathcal{S}, \mathcal{A}, \mathcal{O}` | state, action, and observation spaces | Preferred in environment-formalized lectures. |
| `c_{\mathrm{file}}, c_{\mathrm{repo}}` | file-level and repo-level context | Reserved for long-context coding or theorem-proving settings. |
| `x_I`, `x_F` | informal and formal versions of a problem statement | Reserved for abstraction / formalization lectures. |
| `P_h`, `P_d` | human-written and dynamic policy | Reserved for the security chapter. |
| `\operatorname{ASR}` | attack success rate | Reserved for security evaluation. |

## Symbol Overload Policy

- Reuse `x`, `y`, `\pi_\theta`, `\pi_{\mathrm{ref}}`, and `B` across the book whenever the generic meaning matches.
- If `s` changes from environment state to proof state or security context state, add a local qualifier in the chapter text.
- If `a` changes from tool action to proof tactic, state the specialization explicitly.
- Use `V` for verifier functions only when arguments are input/output objects; use lowercase `v` or explicit `Q` for value estimation.
- Use subscripts rather than entirely new symbols when the object class is unchanged.
