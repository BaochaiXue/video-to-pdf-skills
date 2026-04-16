# Exercises

## 概念复习题
1. 什么是 language agent，与普通 LLM application 有何不同？
2. HippoRAG 试图解决当前 RAG 的哪个根本缺陷？
3. 为什么说 implicit reasoning 并不等于“没有推理”？
4. reactive、tree search、model-based planning 三者如何权衡？
5. WebDreamer 中的 world model 起什么作用？

## 深入思考题
1. 若一个 agent 长期在线运行，记忆更新应如何避免 catastrophic forgetting 与 privacy leakage？
2. 你会如何把 grokking 的 insight 用到更大的 agent foundation model 上？
3. world-model planning 是否会放大模型幻觉？应如何防范？

## 实践题
1. 为一个简单网页任务实现 reactive policy 与 model-based policy，对比交互次数和错误率。
2. 设计一个多跳记忆问答案例，比较 dense retrieval 与图扩散检索的差别。
