# Glossary Delta

- agentic AI safety/security：面向可调用工具、可执行动作、可持续运行 agent 的安全问题。
- hybrid system：由神经组件、符号组件、工具、服务和环境反馈组成的复合系统。
- direct prompt injection：恶意指令直接进入 prompt 空间。
- indirect prompt injection：恶意内容埋在外部数据中，被系统读取后进入 prompt。
- memory poisoning：攻击者污染长期记忆或知识库，使恶意内容跨任务持续被检索。
- least privilege：组件只拥有完成当前任务所需的最小权限。
- privilege separation：将高权限能力从高风险逻辑中拆分出来。
- information flow tracking (IFT)：跟踪敏感信息跨组件、跨工具的传播路径。
