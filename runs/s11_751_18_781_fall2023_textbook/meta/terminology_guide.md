# Terminology Guide

Use Chinese as the main narrative language. Keep standard English terms, paper titles, model names, algorithm names, and benchmark names in English.

At first mention in a lecture note, prefer `中文（English）` style for important concepts.

## Core ASR terms

- 自动语音识别（automatic speech recognition, ASR）
- 语音识别（speech recognition）
- 语音理解（speech understanding）
- 语音合成（text-to-speech, TTS） when the lecture is discussing synthesis
- 口语对话系统（spoken dialog system）
- 词错误率（word error rate, WER）
- 音素（phoneme）
- 音节（syllable）
- 发音词典（pronunciation lexicon）
- 声学模型（acoustic model）
- 语言模型（language model, LM）
- 解码器（decoder）
- 重打分（rescoring）
- 格（lattice）
- 波束搜索（beam search）

## Classical statistical modeling

- 隐马尔可夫模型（hidden Markov model, HMM）
- 高斯混合模型（Gaussian mixture model, GMM）
- 期望最大化算法（expectation-maximization, EM）
- 最大后验（maximum a posteriori, MAP） if needed
- 维特比算法（Viterbi algorithm）
- 前向后向算法（forward-backward algorithm）
- 贝叶斯决策理论（Bayes decision theory）
- 噪声信道模型（noisy channel model）

## End-to-end ASR

- 连接时序分类（connectionist temporal classification, CTC）
- 循环神经网络转导器（recurrent neural network transducer, RNN-T / RNNT）
- 注意力编码器-解码器（attention encoder-decoder）
- 端到端语音识别（end-to-end ASR）
- 对齐（alignment）
- 标签同步搜索（label-synchronous search）
- 时间同步搜索（time-synchronous search）

## Neural and modern speech modeling

- 深度神经网络（deep neural network, DNN）
- 神经网络语言模型（neural network language model, NNLM）
- 自监督学习（self-supervised learning, SSL）
- 语音基础模型（speech foundation model）
- 微调（fine-tuning）
- 多语言语音识别（multilingual ASR）
- 低资源语言（low-resource language）
- 说话人自适应（speaker adaptation） if needed

## Dialog and LLM-related supplementation

- 任务型对话（task-oriented dialog）
- 对话状态跟踪（dialog state tracking, DST）
- 检索增强生成（retrieval-augmented generation, RAG）
- 链式思维（chain-of-thought, CoT）
- 全双工语音对话（full-duplex speech dialogue）

## Style rules

- Do not translate paper titles.
- Do not translate benchmark names.
- Use `RNN-T` consistently unless the source explicitly prefers `RNNT`.
- Use `CS224S 补充` or `延伸解释` when importing non-CMU official material.
- When a source discrepancy exists, say so explicitly instead of normalizing it away.
