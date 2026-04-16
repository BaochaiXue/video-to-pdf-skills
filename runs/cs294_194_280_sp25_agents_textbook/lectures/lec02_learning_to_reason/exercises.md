# Exercises

## 概念复习题
1. 为什么说 DPO 是 Weston 讲法中的“optimizer primitive”？
2. Self-Rewarding 与标准 RLHF 的最大差异是什么？
3. CoVe 为什么能减轻 hallucination？
4. IRPO 为什么要依赖 verifiable final answers？
5. Meta-Rewarding 为什么要对 judgment 本身做比较？

## 深入思考题
1. 若 judge 与 actor 共享同一基础模型，会有哪些共偏差风险？
2. 在开放式 agent 任务中，什么样的环境反馈可以替代人工偏好？
3. 能否把 EvalPlanner 的思想迁移到安全审计或代码审查？

## 实践题
1. 用一个小型 QA 数据集实现 CoVe 四步流程，比较 hallucination rate。
2. 构造一个有可验证 final answer 的推理任务，模拟 IRPO 的 preference-pair 生成过程。
