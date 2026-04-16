# Readings Integration

## An In-Context Learning Agent for Formal Theorem-Proving
- URL: https://arxiv.org/abs/2310.04353
- Main question: 如果没有大量 environment-specific finetuning data，能否只靠强 LLM、proof environment 反馈和 search history 做 formal theorem proving。
- Core method: COPRA 在 stateful backtracking search 中反复调用 GPT-4，执行 tactic，读取错误和新 goals，并把历史和 lemma database 注入下一轮 prompt。
- Key result: 在 miniF2F 与 CompCert Coq 任务上，COPRA 显著优于少样本 GPT-4，并在 pass@1 上超过一些 finetuned baseline。
- Limitations: 系统成本高，依赖高质量 frontier model；若 proof environment 反馈过于稀疏或 prompt 管理不稳定，搜索会快速退化。
- Connection to lecture: 这是数学 discovery 部分的主 reading，直接对应本讲对 theorem-proving agent 的讲解。
- Should appear in sections: 3.1, 3.2, 3.3, 4.1

## Symbolic Regression with a Learned Concept Library
- URL: https://arxiv.org/abs/2409.09359
- Main question: LLM 能否通过诱导和演化抽象 textual concepts，系统性改善 symbolic regression 的搜索效率与发现质量。
- Core method: LaSR 在高质量 hypotheses 中抽取概念，构建 concept library，再用 concept-guided mutations 和 standard evolutionary operators 共同生成新 hypotheses。
- Key result: LaSR 在 Feynman equations 与 synthetic tasks 上优于多种 deep learning 和 evolutionary baseline，并能发现新的 LLM scaling law。
- Limitations: concept quality 的验证仍然困难；概念表示主要依赖自然语言，扩展到更大搜索空间和更复杂感知输入仍是挑战。
- Connection to lecture: 它支撑 lecture 下半场的 abstraction/discovery 主题，说明 agent 不只会证明，还会发明可复用的概念性搜索偏置。
- Should appear in sections: 5.2, 6.1, 6.2, 7.1
