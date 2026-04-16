# Berkeley Agentic AI F25 Source Notes

本目录只服务于 `berkeley_agentic_ai_f25` supplement，不改主书其他部分。

## 官方来源确认

- 官方课程页：<https://rdi.berkeley.edu/agentic-ai/f25>
- 官方 MOOC 镜像：<https://agenticai-learning.org/f25>
- 官方 Berkeley RDI playlist：<https://www.youtube.com/playlist?list=PLS01nW3RtgoqGkm4UeqNeZLccW-OGc1fJ>
- 官方但未列入 playlist 的 Oct 6 录播：<https://www.youtube.com/watch?v=VfOA2a0dj4w>

关键事实：

- 课程页把 `Agentic AI` 定义为覆盖 `LLM foundations`, `reasoning`, `planning`, `agentic frameworks and infrastructure`, 以及 `code generation`, `robotics`, `web automation`, `scientific discovery` 和风险讨论。
- MOOC 页明确写出：这门课 `is built upon the fundamentals from the Fall 2024 LLM Agents MOOC and Spring 2025 Advanced LLM Agents MOOC`。
- 官方 playlist 只有 `11` 个公开视频。
- `Sep 8 Introduction` 只有 slides。
- `Oct 6 Agent Evaluation & Project Overview` 是官方 syllabus 直接给出的 YouTube 链接，但属于 `unlisted`，因此不在公开 playlist 里。

## 已抽取的公开证据

本 workspace 本地保存了若干公开 slides / reading 文本抽取结果，位于 `sources/`：

- `lecture1.txt`
  - 明确给出 `General LLM training pipeline`：`Pretraining -> Reasoning RL -> Classic post-training / RLHF`
  - 明确强调在 agent 时代，`evaluation` 和 `systems and infra to scale` 已经和算法、数据同等重要
- `yangqing.txt`
  - 把 `AI infra` 提升为企业 IT 的 `third pillar`
  - 强调 `developer efficiency`, `infra efficiency`, `multi-cloud supply chain management`, `AI native platform`
- `jiantao.txt`
  - 区分 `Earlier Chat Models` 与 `Agentic Models`
  - 把 agentic model 概括为 `Environment Feedback Aligned Models`
  - 明确提出 agent 训练的三元组：`environment`, `tools`, `verifier`
- `weizhu.txt`
  - 给出 coding-agent environment simulator 的草图
  - 强调 `rubrics`, `graders`, `data synthesis`, `verifiable` 与 `non-verifiable` 数据混合
  - 讨论 product grader 的复杂性，例如 `unit-test`, `rollout`, `behavior`, `user experience`, `ethics`
- `eval_overview.txt`
  - 把 evaluation 切成 `close-ended / open-ended`, `verifiable / non-verifiable`, `static / dynamic`
  - 强调 `Outcome Validity`
- `eval_survey.txt`
  - survey 把 agent eval 拆成四层：capabilities、application-specific、generalist、frameworks
  - application-specific 中明确列出 `web`, `software engineering`, `scientific`, `conversational` agents
- `predeval.txt` 与 `error_bars.txt`
  - 共同支持一个结论：agent eval 不能只看单个 leaderboard 数字
  - 小 benchmark、hard benchmark 和 generative benchmark 都需要统计噪声分析、paired comparison、power analysis
- `multiagent.txt`
  - 用 `self-play`, `minimax equilibrium`, `population best response`, `exploitability` 来解释 multi-agent improvement
- `deployment.txt`
  - 用 `Agent Iceberg` 表明 LLM/RAG/tool use 只是表层
  - 底层还包括 `complex workflows & orchestration`, `observability`, `regression testing`, `user simulation`, `role-based access controls`, `staging and release management`
- `paper2agent.html`
  - 论文摘要明确提出把 research paper 转换成 `AI agent`
  - 通过多 agent 分析 paper 和 codebase，构造 `Model Context Protocol (MCP) server`，并通过自动生成与运行测试来增强可靠性

## 写作边界

- 对 `Sep 8`, `Nov 3`, `Nov 17`, `Dec 1`, `Dec 8` 这些缺少公开 slides 或完整 reading 文本的讲次，本 supplement 只做保守延伸，不伪造 slide-level 细节。
- 对 OpenAI 官方 reading 页面，若自动提取被反爬拦截，则仅把标题和课程页上下文作为证据，不冒充已经读到页面全文。
