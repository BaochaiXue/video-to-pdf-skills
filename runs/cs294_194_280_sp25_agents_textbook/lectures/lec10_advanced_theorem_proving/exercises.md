# Exercises

## Concept Review

1. 解释 informal mathematics、formal mathematics、autoformalization、theorem proving、verification 的区别。
2. 说明 Lean-STaR 中 thought 与 tactic 的角色分工。
3. 解释 DSP 中 sketch 的作用。
4. premise selection 与 ATP reconstruction 为什么都不可缺？
5. miniCTX 为什么能暴露长上下文依赖问题？

## Deeper Questions

1. 设想一个 theorem proving agent，分析它在 verifier 很强但 retrieval 很弱时会如何失败。
2. 思考 thought generation 与 tree search 之间的关系：是否可以完全用更强搜索替代 thought？
3. 说明 proof optimization 与 proof synthesis 的不同评价标准。

## Formal / Proof Tasks

1. 为一个简单 Lean theorem 设计 thought+tactic 的伪轨迹，并解释每步 thought 的作用。
2. 将某个自然语言 proof sketch 重写成可供 formal prover 使用的子目标列表。
