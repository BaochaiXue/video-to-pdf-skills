# Readings Integration

## Draft, Sketch, and Prove: Guiding Formal Theorem Provers with Informal Proofs
- URL: https://arxiv.org/abs/2210.12283
- Main question: 如何把 informal proof 变成 formal prover 的有效搜索偏置，而不是把自然语言解释直接当成最终证明。
- Core method: 先从 informal theorem 和 informal proof 构造 draft 与 sketch，再用 sketch 把低层自动证明器引导到更小、更容易的子问题上。
- Key result: 在竞赛数学问题上，sketch-guided proving 将成功率从 20.9% 提升到 39.3%。
- Limitations: 当 informal proof 本身错误、过于模糊，或者 sketch 与 formal statement 对不齐时，搜索仍然可能发散。
- Connection to lecture: 这是 lecture 中“combining informal and formal provers”的核心 reading，直接支撑第 3 节。
- Should appear in sections: 3.2, 3.3, 3.4

## miniCTX: Neural Theorem Proving with (Long-)Contexts
- URL: https://arxiv.org/abs/2408.03350
- Main question: 如何评估 theorem prover 是否能利用真实项目中的长上下文，而不是只会解短小自包含题。
- Core method: 从真实 Lean 项目中抽取带上下文的 theorem，要求模型访问 preceding code、文件结构和跨文件依赖来完成证明。
- Key result: 依赖上下文的方法显著优于只看局部 state 的传统 prover，且该能力并不会被 miniF2F 之类 benchmark 捕获。
- Limitations: 长上下文建模会带来显著的检索、截断和 benchmark 构造成本。
- Connection to lecture: 它对应本讲最后一部分对 research-level formalization 的转向，强调真实项目环境与竞赛题环境的差异。
- Should appear in sections: 5.2, 5.3, 5.4

## Lean-STaR: Learning to Interleave Thinking and Proving
- URL: https://arxiv.org/abs/2407.10040
- Main question: formal proof data 并不显示人类的思考过程，能否在 tactic 前显式学习 thought，从而改善 theorem proving。
- Core method: 为每一步 tactic 生成 synthetic thoughts，在训练和推理时都采用 thought+tactic 交替的策略，再用 expert iteration 强化成功轨迹。
- Key result: Lean-STaR 在 miniF2F test 上超过此前系统，thought augmentation 与 expert iteration 都贡献了增益。
- Limitations: thought 的价值依赖于搜索器与 verifier；错误 thought 也可能污染后续 tactic 选择。
- Connection to lecture: 这篇 paper 构成本讲第 2 节的主轴，并且和课程早先关于 inference-time reasoning 的讨论直接相连。
- Should appear in sections: 2.1, 2.2, 2.4

## ImProver: Agent-Based Automated Proof Optimization
- URL: https://arxiv.org/abs/2410.04753
- Main question: 在证明已经正确的前提下，能否用 agent 式流程优化 proof 的长度、可读性和模块化程度。
- Core method: 用 LLM agent 读取符号化 Lean context，通过 Chain-of-States、error correction 和 retrieval 重写证明。
- Key result: 在本科、竞赛和 research-level theorem 上，ImProver 能在保持正确性的同时让 proof 更短、更易读或更模块化。
- Limitations: 优化目标之间可能冲突，且 proof optimization 仍然依赖 proof assistant 的即时反馈才能稳定进行。
- Connection to lecture: 虽然 slides 主要讲 proving，而非 optimization，但它补上了“agent 不仅能找 proof，也能迭代改写 proof”的一环。
- Should appear in sections: 6.1, 6.2
