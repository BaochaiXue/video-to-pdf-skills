# L06 Multimodal Autonomous AI Agents

本讲聚焦 multimodal autonomous agents 在真实网页与物理环境中的系统问题。课程的主线不是“模型看得见图像”这么简单，而是：当任务需要多步交互、真实界面、环境反馈与可执行动作时，agent 必须同时解决 benchmark realism、视觉 grounding、长程 search 和训练数据稀缺四个问题。

- WebArena/Mind2Web/VisualWebArena 构成了 web-agent benchmark 的演化链。
- Tree Search for Language Model Agents 展示了 inference-time search 如何提高 long-horizon task success rate。
- InSTA 把 synthetic task generation 与 automatic verification 结合起来，试图缓解 agent training 的数据瓶颈。
- Plan-Seq-Learn 则把类似问题推广到 physical agents，说明 perception-action loop 在 embodied setting 中更严苛。
