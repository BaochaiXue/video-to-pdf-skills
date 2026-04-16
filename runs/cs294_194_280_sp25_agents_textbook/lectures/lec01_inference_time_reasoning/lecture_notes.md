# L01 Lecture Notes

## 讲次信息

- 课程：CS294/194-280: Advanced Large Language Model Agents
- 讲次：L01
- 主题：Inference-Time Techniques for LLM Reasoning
- 讲者：Xinyun Chen

## 本讲主线

本讲把 advanced LLM agents 的第一性问题表述为：同一个模型在部署时，如何通过更聪明地分配推理预算，获得更强的 reasoning quality。核心不是继续预训练，而是重新设计推理轨迹的生成、筛选、评价和修订机制。

## 关键方法脉络

1. 单轨推理：
   - standard prompting
   - few-shot CoT
   - zero-shot CoT
   - analogical prompting
   - least-to-most
   - self-discover
2. 宽搜索：
   - self-consistency
   - execution consistency
   - universal self-consistency
3. 评价与搜索：
   - ORM / PRM verifiers
   - Tree of Thoughts
4. 深度自改进：
   - Reflexion
   - Self-Refine
   - Self-Debugging
5. 预算原则：
   - model size vs breadth vs depth
   - Bitter Lesson for reasoning technique design

## 本讲最重要的判断

- 额外推理 token 不是越多越好，关键是花在什么机制上。
- 没有可靠外部反馈时，self-correction 往往不成立。
- 代码任务之所以适合自改进，是因为 execution feedback 提供了真实外部信号。
- 好的 inference-time method 应当随着计算增长持续扩展，而不是依赖脆弱的手工模板。
