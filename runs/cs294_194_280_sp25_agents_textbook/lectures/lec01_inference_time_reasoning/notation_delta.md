# Notation Delta

- $x_i$：第 $i$ 轮已有的 instruction 或候选解。
- $s_i$：候选 $x_i$ 对应的评估分数。
- $\mathcal{E}$：任务 exemplars 或任务说明。
- $\hat{y}$：聚合后的最终答案。
- $a_i$：第 $i$ 条 reasoning path 的完整回答。
- $\tau$：完整或部分 reasoning trajectory。
- $\mathcal{T}$：候选 trajectories 集合。
- $r_{\phi}$：verifier 或 reward model 的评分函数。
- $M, N, D$：模型规模、并行宽度、串行深度。
- $B$：总 inference-time budget。
