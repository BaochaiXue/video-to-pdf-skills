# Book Outline

This outline turns the 12 validated lecture workspaces into a textbook architecture.
The unit of authorship remains the lecture workspace, but the unit of reading becomes the thematic part.
The book should read as a course textbook on advanced LLM agents, not as a date-ordered transcript digest.
The Spring 2025 Berkeley course remains the main line of argument. A final extension part folds in cross-course material from Berkeley Fall 2024, Berkeley Fall 2025, and Stanford CS329A Autumn 2025 to keep the book current without collapsing the main course into a miscellaneous survey.

## Architectural Principles

- Keep each lecture as a chapter-level source of truth, then add part introductions and cross-chapter transitions during merge.
- Group chapters by problem family: reasoning, agentic workflows, multimodal interaction, formal mathematics, and safety/security.
- Preserve provenance and evaluator-gated chapter boundaries. A chapter enters the book because its lecture harness passed, not because it is convenient for the outline.
- Use appendices to centralize glossary, notation, paper map, benchmark map, algorithm index, figure provenance, and omission accounting.

## Reader-Facing Table Of Contents

### Frontmatter

1. Title page
2. Preface
3. How to Use This Book
4. Table of Contents

### Part I. Foundations Of Advanced LLM Agents

Purpose:
Establish the core claim of the course: advanced LLM agents are defined not by a single model family but by how compute, feedback, memory, and search are organized at deployment time.

1. Chapter 1. Course Overview and a Framework for Advanced LLM Agents
   Source anchor: synthesized from course frontmatter plus cross-lecture introductions
   Role in the book: define the course-wide vocabulary for reasoning, planning, search, memory, tools, environments, verification, and safety
2. Chapter 2. Inference-Time Techniques for LLM Reasoning
   Lecture source: `lec01_inference_time_reasoning`
   Core thread: test-time compute, CoT, self-consistency, verifier-guided reasoning, Tree of Thoughts, self-debugging
3. Chapter 3. Learning to Reason with LLMs
   Lecture source: `lec02_learning_to_reason`
   Core thread: judgment learning, System 2 style verification, self-rewarding, IRPO, meta-rewarding
4. Chapter 4. Reasoning, Memory, and Planning in Language Agents
   Lecture source: `lec03_reasoning_memory_planning`
   Core thread: memory systems, world models, long-horizon planning, environment-grounded reasoning

Part introduction should explain:
- why inference-time compute and post-training should be treated jointly rather than as competing stories
- why memory and planning are not just extensions of CoT
- how this part sets up the later chapters on tools, multimodality, and formal verification

### Part II. Agentic Workflows, Tools, and Code

Purpose:
Move from abstract reasoning mechanisms to agents that must call tools, interact with code, and survive execution-based evaluation.

5. Chapter 5. Open Training Recipes for Reasoning in Language Models
   Lecture source: `lec04_open_training_recipes_reasoning`
   Core thread: SFT data recipe, preference tuning, RLVR, budget forcing, open post-training pipelines
6. Chapter 6. Coding Agents and AI for Vulnerability Detection
   Lecture source: `lec05_coding_agents_vulnerability_detection`
   Core thread: coding workflows, environment feedback, SWE-Bench style evaluation, security-oriented agent design

Part introduction should explain:
- how training recipes feed directly into agent workflow quality
- why coding agents are the cleanest place to study tool use, verifiers, and dynamic control
- why vulnerability detection is not just another coding benchmark but a capability-and-risk stress test

### Part III. Multimodal And Interactive Agents

Purpose:
Extend agent design from text-only reasoning to perception-action loops in web, GUI, OS, and robotic environments.

7. Chapter 7. Multimodal Autonomous AI Agents
   Lecture source: `lec06_multimodal_autonomous_agents`
   Core thread: VisualWebArena, value-guided search, synthetic task generation, embodied grounding
8. Chapter 8. Multimodal Agents: From Perception to Action
   Lecture source: `lec07_multimodal_agents_perception_to_action`
   Core thread: OSWorld, GUI grounding, action spaces, thought-action traces, video memory

Part introduction should explain:
- why benchmark realism matters more in multimodal settings
- how perception error, grounding error, and planning error interact
- how multimodal environments sharpen the same deployment-time questions introduced in Parts I and II

### Part IV. Formal Mathematics, Verification, And Theorem Proving

Purpose:
Show the strongest current form of environment-grounded reasoning: formal mathematics, where actions are constrained, states are explicit, and success can be mechanically checked.

9. Chapter 9. AlphaProof and Reinforcement Learning for Formal Mathematics
   Lecture source: `lec08_alphaproof_formal_mathematics`
   Core thread: theorem proving as an RL environment, proof-state search, verifier-grounded optimization
10. Chapter 10. Autoformalization and Theorem Proving
    Lecture source: `lec09_autoformalization_theorem_proving`
    Core thread: converting informal mathematics into formal specifications, separating verification from proof construction
11. Chapter 11. Advanced Topics in Theorem Proving
    Lecture source: `lec10_advanced_theorem_proving`
    Core thread: long-context proving, sketch-guided search, ATP integration, research-scale workflows

Part introduction should explain:
- the boundary between informal reasoning, formal specification, verification, theorem proving, and proof search
- why formal mathematics is a privileged environment for agent research rather than a niche application
- how proof assistants function as both tools and evaluators

### Part V. Abstraction, Discovery, Safety, And Security

Purpose:
End the book by widening the scope from solving tasks to discovering abstractions and then constraining powerful agents with explicit security boundaries.

12. Chapter 12. Abstraction and Discovery with Large Language Model Agents
    Lecture source: `lec11_abstraction_discovery_llm_agents`
    Core thread: abstraction as search-space restructuring, theorem proving vs discovery workflows, scientific and formal discovery
13. Chapter 13. Towards Building Safe and Secure Agentic AI
    Lecture source: `lec12_safe_secure_agentic_ai`
    Core thread: prompt injection, memory poisoning, privilege separation, identity, information flow, system boundaries

Part introduction should explain:
- why abstraction and discovery increase both capability and risk
- why safety for agentic systems is an architectural property, not a post-hoc prompt patch
- how the final chapter reframes the entire book in terms of controllable deployment

### Appendices

1. Glossary
2. Notation Table
3. Paper Map
4. Benchmark Map
5. Algorithm Index
6. Figure Provenance
7. Course Schedule Notes and No-Class Dates
8. Course Omission Log
9. Suggested Reading Paths

### Part VI. Cross-Course Extensions And 2025 Updates

Purpose:
Anchor the Spring 2025 main line inside the broader Berkeley/Stanford agent curriculum that unfolded across Fall 2024, Fall 2025, and Stanford CS329A. This part should not replace the main course; it should show where the surrounding course ecosystem adds systems, evaluation, multi-agent, self-improvement, and deployment lessons that the core textbook should explicitly inherit.

14. Chapter 14. Berkeley Fall 2024: Large Language Model Agents as the Systems Baseline
    Source anchor: official Berkeley RDI Fall 2024 course page, MOOC page, playlist, slides, and readings
    Role in the book: recover the earlier Berkeley systems curriculum around RAG, DSPy, AutoGen, software agents, robotics, evaluation, and trustworthy deployment
15. Chapter 15. Berkeley Fall 2025: Agentic AI as the Latest Public Continuation
    Source anchor: official Berkeley RDI Fall 2025 course page, MOOC page, playlist, slides, and readings
    Role in the book: capture the latest public continuation of the Berkeley line, especially agent evaluation, verifiable agents, multi-agent systems, scientific discovery, deployment lessons, and modern safety framing
16. Chapter 16. Stanford CS329A: Self-Improving Agents Through Schedule and Readings
    Source anchor: official Stanford CS329A course site and schedule, with public but non-canonical instructor-affiliated playlist noted separately
    Role in the book: connect the Spring 2025 textbook to a self-improvement-centric curriculum covering coding agents, research automation, evaluation, and agentic search

Part introduction should explain:
- why a textbook on advanced LLM agents should end with adjacent courses rather than pretending one semester is the whole field
- why Fall 2024 provides the systems baseline, Fall 2025 provides the freshest Berkeley public continuation, and Stanford CS329A sharpens the self-improvement lens
- how to treat official sources, public mirrors, unlisted recordings, and instructor-affiliated but non-official playlists with different trust levels

## Recommended Reading Paths

1. Core course path
   Chapters 1 through 13 in order.
2. Reasoning-first path
   Chapters 1 to 5, then 9 to 11, then 13.
3. Agent systems builder path
   Chapters 1, 4, 5, 6, 7, 8, 13.
4. Formal methods path
   Chapters 1, 2, 3, 9, 10, 11, 12, 13.
5. Safety and security path
   Chapters 1, 4, 6, 8, 12, 13.
6. Course-extension path
   Chapters 14, 15, 16 after finishing the main line, or earlier if the reader specifically wants the Berkeley/Stanford 2024-2025 expansion.

## Merge Notes For The Final Editor

- Add a short transition at the end of each chapter pointing to the next chapter's problem shift.
- Avoid repeating generic course logistics or evaluation boilerplate inside parts.
- Preserve lecture-level technical detail, examples, formulas, and failure modes; remove only redundant framing.
- Keep chapter provenance in sidecars even if the main text is smoothed into a more continuous textbook voice.
