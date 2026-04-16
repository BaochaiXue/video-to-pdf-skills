# Exercises

## 概念复习题

1. 为什么 Hajishirzi 把“开放”视为训练 recipe 的组成部分，而不是单独的伦理或社区议题？
2. SFT、preference tuning、RLVR 各自解决什么问题？
3. 为什么 reasoning data 往往需要 chain-of-thought，而不是只有 final answer？
4. DPO 相比 PPO 省掉了什么组件？代价是什么？
5. RLVR 为什么只在一类任务上特别有效？

## 深入思考题

1. 如果你只有极少预算做 post-training，应该先花在更好的 preference data 还是更复杂的优化算法上？给出依据。
2. 为什么 budget forcing 能在某些 reasoning 任务上生效，但在另一些任务上可能只是延长错误推理？
3. OpenScholar 为什么能被视为 open recipe 的 downstream 胜利，而不是与本讲无关的单独系统？

## 实践题

1. 设计一个针对数学 reasoning 模型的 data-mixing plan，明确写出 evaluator、数据来源和过滤策略。
2. 为一个有 gold answer 的任务写出 verifier 函数接口，并讨论它可能被 reward hacking 的方式。

