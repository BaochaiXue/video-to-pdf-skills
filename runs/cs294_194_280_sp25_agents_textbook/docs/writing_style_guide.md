# Writing Style Guide

## Target Style

Write as a Chinese textbook for self-study, not as a lecture recap.

The reader should be able to understand the main technical story without watching the video.

## Language Rules

- Main prose is in Chinese.
- Keep English for paper titles, benchmark names, model names, algorithm names, code, and math.
- On first appearance, give important concepts in bilingual form, for example:
  `推理时计算（inference-time computation）`
  `自动形式化（autoformalization）`

## Section Expectations

Each lecture chapter should include:

1. learning goals
2. background and problem setup
3. core concepts and terminology
4. main technical sections
5. formulas, algorithms, or code explained step by step
6. connections to official readings
7. examples, counterexamples, failure modes, and caveats
8. links to earlier and later lectures
9. a chapter summary
10. review questions
11. deeper thinking questions
12. extended reading

## Explanation Standard

For every major concept, answer:

- what problem it solves
- why a naive approach is insufficient
- what the core mechanism is
- what the formal definition is
- what a small example looks like
- where it fails
- how it connects back to LLM agents

## Formula Rules

- Use display math for important formulas.
- Explain every symbol immediately after the formula.
- Give intuition, not just notation.
- If the lecture contains a derivation, unfold the derivation step by step instead of skipping to the final equation.

## Algorithm And Code Rules

- Do not paste code or pseudocode without explaining why it exists.
- State inputs, outputs, loop structure, stopping condition, and complexity intuition.
- For security or coding-agent material, explicitly discuss bug surfaces, exploitability, and safety failure modes.

## Reading Integration Rules

- Each major supplemental reading must appear in a paragraph titled or framed as its relation to the lecture.
- Core readings deserve contribution, method, experimental setup, and limitation coverage.
- Do not turn the chapter into a reading list. Readings must support the lecture narrative.

## Course-Specific Distinctions To Preserve

- reasoning vs planning vs search vs inference-time compute
- post-training vs RLHF vs DPO vs PPO vs GRPO
- agent vs tool use vs function calling vs workflow orchestration
- formalization vs autoformalization vs theorem proving vs proof search vs verification
- multimodal agent vs web agent vs GUI agent vs OS agent
- privilege separation vs prompt injection vs memory poisoning

## Provenance Rules

- Facts from recordings, slides, and readings should be presented as source-backed.
- Any extension or synthesis not literally stated in the sources should be marked as interpretation, inference, or extended explanation.
- Never invent missing details in low-confidence spans.
