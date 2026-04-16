# Exercises

## 概念复习题

1. 为什么 CoT 的价值不只是“让回答变长”？
2. Analogical Prompting 与 Few-shot CoT 的主要差别是什么？
3. OPRO 如何把 prompt engineering 变成 optimization loop？
4. Self-Consistency 为什么要求采样多样性？
5. ORM 与 PRM 的区别是什么？

## 深入思考题

1. 如果 verifier 不可靠，Tree of Thoughts 会出现什么系统性偏差？
2. 为什么没有外部反馈时，self-correction 可能比一次性回答更差？
3. 你会如何为开放式长文写作任务构造“足够可靠”的反馈器？

## 实践题

1. 在数学题集上实现 Self-Consistency，并比较不同 sample budget 的效果。
2. 在代码生成任务上实现一个最小版 Self-Debugging loop，比较仅文本反思与执行反馈反思的差别。

## 形式化或证明相关题

1. 试把 “verifier ranking” 写成一个有限候选集上的最优化问题，并解释为什么步骤级评分会改变搜索策略。
2. 讨论在 theorem proving 场景中，为什么 partial-state evaluation 比 final-answer reranking 更重要。

## 安全与 failure analysis 题

1. 如果 self-correction 的反馈来自另一个同分布 LLM，而不是真实环境，这个系统会出现什么失败模式？
2. 在代码 agent 中，execution consistency 与真实安全性之间有哪些错位风险？
