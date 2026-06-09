# SLP3 / pasted chapter style contract for CS336 rewrite

Source consulted:

- Jurafsky & Martin, *Speech and Language Processing*, Third Edition draft, January 6, 2026: <https://web.stanford.edu/~jurafsky/slp3/ed3book_jan26.pdf>
- User-provided pasted chapter: `/Users/xinjiezhang/.codex/attachments/8dd117e3-1234-4b0b-9417-2598d7fa5d38/pasted-text.txt`

## Observed textbook architecture

- Chapter opens with a concrete motivating example before formal definitions.
- The prose moves from problem intuition to terminology, then to algorithms, formulas, examples, caveats, and exercises.
- Definitions are not isolated; each term is introduced because a later algorithm or metric needs it.
- Examples are multilingual, cross-domain, or implementation-facing when the concept has boundary cases.
- Figures, tables, equations, and pseudocode are teaching devices, not decoration.
- The chapter ends with a summary, historical/practical notes, and exercises.

## Requirements applied to CS336

- Main body must be Chinese textbook prose, not English source excerpts.
- Keep English technical names and first-use bilingual terminology.
- Each lecture chapter must contain: motivating case, terminology, core mechanisms, formulas with symbol explanations, pseudocode/implementation notes, worked example, caveats, course-material review path, summary, extension note, and exercises.
- Source grounding is preserved in sidecar JSONL and compact chapter-end review paths; source excerpts are not allowed to dominate the body.
- Any teaching bridge not directly present in the course materials is treated as extension/explanatory scaffolding rather than a new course fact.
