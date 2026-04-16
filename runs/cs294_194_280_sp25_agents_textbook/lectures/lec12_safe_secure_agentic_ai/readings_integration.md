# Readings Integration

## Privtrans: Automatically Partitioning Programs for Privilege Separation

This reading gives the lecture historical grounding for privilege separation: long before LLM agents, system security already relied on shrinking privilege boundaries. The lecture reuses the same principle for agent decomposition and tool sandboxing.

## DataSentinel: A Game-Theoretic Detection of Prompt Injection Attacks

This reading corresponds to the lecture's monitoring/detection layer. It is not positioned as a universal patch, but as one component in defense-in-depth for agent pipelines exposed to untrusted content.

## AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases

This reading grounds the lecture's warning that prompt injection is only one part of the attack surface. Memory poisoning and knowledge-base poisoning make the boundary problem persistent across sessions and tasks.

## Progent: Programmable Privilege Control for LLM Agents

This is the lecture's central positive defense result. It turns least privilege from a principle into an enforceable runtime gate over tool calls, which is exactly where many agentic attacks become consequential.
