# Exercises

## Review Questions

1. 为什么 OSWorld 比只含 demonstrations 的 benchmark 更适合评估 computer-use agents？
2. OSWorld 的 task config、observation、action 和 execution-based evaluation 分别扮演什么角色？
3. 为什么高截图分辨率和更长的 history 会改善 baseline performance，但不会从根本上解决问题？
4. AgentTrek 为什么从教程网页而不是人工全程标注出发？
5. CoTA 与普通 CoT 有什么本质不同？
6. AGUVIS 为什么要坚持 pure-vision observation？
7. inner monologue 为什么同时影响高层 reasoning 和低层 grounding？
8. xGen-MM-Vid 和 GenS 分别想解决长视频中的什么瓶颈？

## Deep Questions

1. 如果你设计一个新 benchmark 来连接 OSWorld 与 physical agents，你会保留哪些 execution constraints？
2. 对 GUI agents 来说，统一 action space 与统一 observation space 哪个更难做？为什么？
3. 长视频理解中的 frame selection 能否视为另一种 inference-time search？请结合 GenS 讨论。

## Practice / Reading Extensions

1. 阅读 OSWorld 论文摘要和 slides，设计一个新的 execution-based evaluator，并说明它如何避免误判 alternative correct solutions。
2. 画出 AgentTrek 的 guided replay 数据管线，并指出最容易引入 distribution shift 的步骤。
3. 为一个 GUI task 写一个 CoTA example，显式给出 thought 和 action 序列。
4. 比较 AGUVIS 与任一 text-based GUI agent，在相同任务上列出它们各自更容易失败的点。
