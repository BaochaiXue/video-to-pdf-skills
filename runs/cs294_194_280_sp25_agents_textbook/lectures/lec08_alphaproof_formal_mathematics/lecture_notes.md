# L08 Lecture Notes

## 讲次信息

- 课程：CS294/194-280: Advanced Large Language Model Agents
- 讲次：L08
- 主题：AlphaProof: when reinforcement learning meets formal mathematics
- 讲者：Thomas Hubert

## 本讲主线

本讲不是简单介绍一个数学 benchmark，而是回答一个更根本的问题：为什么 formal mathematics 可能成为 advanced agents 最理想的长期环境之一。Thomas Hubert 的论点是，只要我们拥有 machine-checkable 的状态转移和完美验证信号，就可以把 AlphaZero 风格的 search + RL recipe 搬到 theorem proving 上。

## 关键结构

1. 形式化数学的动机：严格性、验证性、库化复用和软件栈协同。
2. RL framing：把 Lean proof state 看成状态，把 tactic 看成动作，把 proof success/failure 看成 grounded feedback。
3. Benchmark framing：IMO 2024 作为 Apollo-style milestone，而不是终点。
4. AlphaProof pipeline：formalizer、prover、search、AlphaZero-style RL、test-time RL specialization。
5. 局限性：Mathlib 覆盖、formalization 成本、几何与组合领域的困难、research mathematics 的开放世界复杂性。

## 本讲最重要的判断

- informal reasoning 与 formal verification 之间的差距，是 agent 系统能否真正“被检查”的关键。
- AlphaProof 的核心价值不只是分数，而是展示 theorem proving 可以成为一个 search-and-feedback 闭环。
- perfect verification 不等于“问题已经解决”，因为 formalization bottleneck 和 action-space explosion 仍然存在。
