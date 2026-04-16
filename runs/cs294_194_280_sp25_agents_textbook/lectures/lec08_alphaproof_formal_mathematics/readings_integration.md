# Readings Integration

## Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm

这篇 AlphaZero 论文是本讲最直接的 methodological ancestor。Lecture 并不是说“数学像棋”，而是说两者都可以在明确规则下形成 search + grounded feedback 的闭环。AlphaProof 继承的不是棋类特定技巧，而是把策略先验、搜索和值估计绑定在一起的通用 recipe。

更教材化地说，这篇 reading 帮助我们理解 AlphaProof 为什么更像“formal environment 上的 AlphaZero-style agent”，而不是普通 LLM 加 verifier。它强调的是 recipe 的可迁移性，但也让我们看到边界：棋类自带 simulator 与明确终局奖励，而 formal mathematics 仍然要面对 formalization 和 library bottleneck。

## The Future of Mathematics?

这段 Microsoft Research 演讲为本讲补上了 ecosystem 视角：formal mathematics 不只是 benchmark，也是一套会逐渐成熟的 research infrastructure。L08 的系统工程判断与这段 reading 一致，即 Lean/Mathlib 社区建设本身就是 agent 能力边界的一部分。

它的重要作用是防止读者误解 AlphaProof 的成绩来源。Lecture 想表达的不是“模型突然会数学了”，而是“当社区把数学逐步转成 machine-actionable environment 之后，agent recipe 才有可能真正发挥作用”。这一点与 AlphaZero reading 正好互补：前者讲环境成熟度，后者讲学习 recipe。
