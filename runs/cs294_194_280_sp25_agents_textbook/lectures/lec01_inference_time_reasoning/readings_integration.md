# Readings Integration

## Large Language Models as Optimizers

这篇论文对应讲义中的 OPRO 段落。它把 prompt engineering 改写成显式的优化循环，最重要的思想不是某个具体 prompt，而是“让 LLM 观察候选历史和分数，再继续提出新候选”。这使 prompt 设计从一次性 artisan work 变成可迭代的 inference-time search。

## Large Language Models Cannot Self-Correct Reasoning Yet

这篇论文是本讲中所有“self-improvement”方法的负例基准。它说明没有可靠外部反馈时，第二轮回答不会自然优于第一轮回答，因此任何 self-refine / self-correction 机制都必须首先检查 feedback source 是否可信。

## Teaching Large Language Models to Self-Debug

这篇论文展示了为什么代码任务是 reasoning-time repair loop 的理想场景。程序拥有 execution trace、test output 和 error message，因此修订不是空谈。讲义在 Self-Debugging 一节中把这篇论文作为“有外部反馈时自改进确实成立”的正例。
