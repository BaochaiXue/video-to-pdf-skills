# Glossary Delta

- 推理时搜索（inference-time search）: 在部署时对候选 action trajectories 做显式探索，而不是只依赖单条 autoregressive rollout。
- 视觉 grounding（visual grounding）: 把自然语言目标对齐到截图中的可交互元素与布局线索。
- execution-based evaluation: 用环境状态和可执行检查脚本判断任务是否真正完成，而非只比对文本答案。
- synthetic agentic tasks: 由模型生成并自动验证、用于 agent training 的任务与轨迹。
- Plan-Seq-Learn: 将长程 manipulation 拆成语言规划、场景 sequencing 与局部 RL control 的分层方法。
