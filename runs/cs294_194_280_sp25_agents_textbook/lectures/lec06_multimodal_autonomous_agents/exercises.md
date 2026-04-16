# Exercises

## Review Questions

1. 为什么 Mind2Web、WebArena、VisualWebArena 不能互相替代？
2. 为什么 HTML-only representation 会在真实网页任务中失真？
3. VisualWebArena 中 visually grounded evaluation 的核心贡献是什么？
4. 为什么长程 agent 任务会出现 exponential error compounding？
5. Tree search 相比 repeated sampling 的改进点是什么？
6. 为什么 value function 质量会直接限制 search gain？
7. InSTA 为什么强调 task generation 与 task verification 两阶段？
8. 为什么 synthetic tasks 能缓解 agent data bottleneck，但不能自动解决 safety 和 realism 问题？

## Deep Questions

1. 如果一个 agent 在 WebArena 上分数高、在 VisualWebArena 上分数低，你会首先怀疑它缺什么能力？
2. 在给定固定预算时，你会优先提高 baseline policy，还是提高 value function，还是增加 search budget？为什么？
3. Plan-Seq-Learn 这种分层架构是否一定优于端到端大模型策略？在什么条件下未必？

## Practice / Reading Extensions

1. 对比阅读 `Mind2Web` 与 `WebArena`，写一段分析：一个数据集为什么不等于一个 environment？
2. 阅读 `Tree Search for Language Model Agents` 项目页，推导 search budget、branching factor 和 depth 的 tradeoff。
3. 为一个你熟悉的网站设计 3 个 visually grounded tasks，并说明如何用 execution-based checks 验证它们。
4. 画出一个简化版 Plan-Seq-Learn pipeline，并说明每个模块失败时会把错误如何传到下游。
