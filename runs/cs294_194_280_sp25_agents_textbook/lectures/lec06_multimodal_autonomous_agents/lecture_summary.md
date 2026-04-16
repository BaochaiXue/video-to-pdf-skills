# Lecture Summary

本讲把 multimodal autonomous agents 拆成四条主线：一是用 VisualWebArena 这样的 benchmark 去严肃衡量 visually grounded web tasks；二是用 value-guided tree search 把 inference-time compute 转化为更强的长程规划能力；三是承认 agent training 的核心瓶颈是缺数据，并用 InSTA 这类 synthetic task pipeline 去扩展训练分布；四是把同样的 perception-action logic 推向 physical agents，并用 Plan-Seq-Learn 说明语言规划、scene grounding 与低层控制的分工。

最重要的结论不是“多模态模型已经会做网页和机器人任务”，而是：一旦任务跨越长程交互、真实视觉界面和环境反馈，benchmark realism、value-based search、data pipeline 和 grounding quality 缺一不可。
