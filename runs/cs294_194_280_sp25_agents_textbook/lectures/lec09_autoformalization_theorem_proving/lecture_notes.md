# L09 Lecture Notes

## 讲次信息

- 课程：CS294/194-280: Advanced Large Language Model Agents
- 讲次：L09
- 主题：Language models for autoformalization and theorem proving
- 讲者：Kaiyu Yang

## 本讲主线

本讲要解决的不是“LLM 会不会做数学题”，而是“LLM 如何进入 formal reasoning 环境，并把 informal mathematics 变成可验证对象”。Kaiyu Yang 的核心贡献在于把几个常被混淆的任务拆开：formal specification、verification、theorem proving、proof search、autoformalization。

## 关键结构

1. math LLM 当前 recipe 与能力边界。
2. formal reasoning 的定义与 proof assistant 作为执行环境。
3. LeanDojo / ReProver：theorem proving 的开放基础设施和 retrieval 机制。
4. infinite action space 与 domain-specific proving，例如 LIPS。
5. autoformalization：从 theorem statements 到 proofs，再到 geometry 中的 diagrammatic reasoning。

## 本讲最重要的判断

- formal reasoning 不是单一 benchmark，而是一组互相关联但不同的任务。
- verification 只能“验”，不能替代 formalization 和 search。
- retrieval、symbolic reasoning 和 domain structure 对 theorem proving 仍然非常重要。
- autoformalization 最大的难点不是翻译语法，而是恢复 informal proofs 中被省略的语义与图形信息。
