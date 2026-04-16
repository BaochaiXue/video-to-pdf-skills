# Notation Delta

- $s$：当前 Lean proof state
- $a$：候选 tactic / proof action
- $Q(s,a)$：在搜索树中对动作价值的估计
- $P_\theta(a\mid s)$：prover model 给出的动作先验
- $f_\phi$：formalizer model
- $\mathcal{N}(p)$：围绕目标难题构造的变体分布
