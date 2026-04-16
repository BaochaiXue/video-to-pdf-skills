# Exercises

## Concept Review

1. 解释 discovery agent 的四个关键能力。
2. formal representation 为什么能缓解 natural-language reasoning 的不可验证性？
3. COPRA 的 prompt synthesis、action parsing 和 backtracking 分别做什么？
4. 为什么 compiler verification 是 theorem-proving agent 的典型应用？
5. 说明 LaSR 中 concept library 与普通 evolutionary search 的差别。

## Deeper Questions

1. 比较 theorem proving 中的 verifier 与 scientific discovery 中的 empirical feedback，有哪些结构性差异？
2. 讨论 abstraction 何时会帮助 search，何时会造成 search bias。
3. 若要把本讲方法扩展到 experiment design，需要新增哪些反馈环路？

## Practice / Formal Tasks

1. 为一个简单 theorem-proving task 设计 COPRA 风格的 search loop，并标出环境反馈节点。
2. 对一组观测数据写出 symbolic regression objective，并说明 concept library 能从哪些成功 hypotheses 中抽象出来。
