# Readings Integration

- `OSWorld` grounds the lecture's benchmark section: it provides the canonical environment, evaluation philosophy, and headline human-vs-agent gap.
- `AGUVIS` grounds the pure-vision GUI-agent section, especially the move away from HTML/AXTree/XML observations, the unified action space, and the two-stage training recipe.
- Additional lecture-cited works such as AgentTrek, TACO, xGen-MM-Vid, and GenS are integrated as slide-grounded content rather than as official supplemental readings.

教材化地看，这两篇 reading 的分工非常清楚。`OSWorld` 解决的是“什么算真实 computer-use benchmark”，因此它定义的是 environment 和 evaluation philosophy；`AGUVIS` 解决的是“在这样的 environment 里，agent 应该怎样统一地看屏幕、表达动作并组织 reasoning”，因此它定义的是 representation 与 training problem。

这也解释了为什么 lecture 不会把 GUI grounding、OS interaction 和 long-video memory 混成一个问题。OSWorld 暴露的是能力缺口，AGUVIS 提供的是其中一类解决路线；两者合起来，才构成从 benchmark diagnosis 到 model design 的完整链条。
