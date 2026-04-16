# Exercises

## 概念复习题
1. 解释 direct prompt injection 与 indirect prompt injection 的差别。
2. 为什么 AgentPoison 说明 memory layer 也是 security boundary？
3. 说明 stand-alone LLM evaluation 与 end-to-end agent evaluation 的区别。
4. 什么是 least privilege on tool calls？
5. 为什么 formal verification 在 agentic system 中比在普通聊天模型里更有意义？

## 深入思考题
1. 设计一个支持 email、calendar、payments 的 personal assistant agent，给出你的 privilege decomposition。
2. 若 detection model 误报率较高，如何与 deterministic policy enforcement 结合，避免 utility 崩溃？
3. 讨论 multi-agent system 中 shared memory 的 poisoning 风险与缓解机制。

## 实践题
1. 为一个 browser agent 设计 tool-level policy schema，区分 read-only、navigation、form-submit、payment 四类能力。
2. 阅读 Progent 与 AgentPoison，比较“runtime privilege gate”与“memory integrity”两类防御的覆盖面。
