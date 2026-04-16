# L01 Summary

这一讲建立了整门课的基础视角：高级 LLM agents 的能力不仅来自训练阶段，也来自部署阶段如何组织 reasoning。CoT、analogical prompting、OPRO、self-consistency、verifier、Tree of Thoughts 与 self-debugging 虽然形式不同，但都在重新设计 inference-time compute 的分配方式。

本讲最重要的正反两条结论是：

- 正面结论：当任务拥有外部反馈或可评价中间状态时，更多推理预算可以显著改善性能。
- 反面结论：没有可靠外部反馈时，自我修订并不会自动变成更好的 reasoning。

从教材角度看，本讲给出的不是一套固定算法，而是一套分析框架：问题是否需要更长单轨推理、更宽并行搜索、更强 verifier、更深修订，取决于任务结构与反馈质量。
