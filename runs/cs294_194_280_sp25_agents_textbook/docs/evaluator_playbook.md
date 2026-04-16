# Evaluator Playbook

## Role

The evaluator is not a co-writer.
The evaluator is the delivery gatekeeper for both lecture chapters and the merged textbook.

## Minimum Inputs For Lecture Evaluation

Read at least:

- `source_manifest.json`
- `transcript.jsonl`
- `slides.jsonl` when slides exist
- `coverage_units.jsonl`
- segment contracts
- figure plan and figure manifest
- lecture LaTeX source
- readings integration artifacts
- the latest repair log when present

## Review Order

1. confirm source grounding
2. check required coverage units and omissions
3. inspect formulas, algorithms, code, and figures
4. inspect reading integration
5. inspect structure, bilingual terminology, and readability
6. inspect compile readiness

## Blocking Issue Requirements

Every blocking issue must include:

- stable `issue_id`
- `type`
- affected `unit_id` or section
- concrete problem statement
- concrete required fix

Useful blocking categories:

- `coverage_gap`
- `thin_explanation`
- `formula_missing`
- `code_unexplained`
- `figure_missing`
- `hallucination_risk`
- `bad_structure`
- `compile_risk`
- `reading_integration_gap`
- `terminology_drift`

## Fail-Closed Policy

Fail the lecture if:

- one required unit is still uncovered
- one dense slide is flattened into a summary sentence
- one important formula lacks symbol explanation
- one important code path is pasted without explanation
- one delivered figure lacks provenance
- one reading is referenced inaccurately
- one unsupported fact is presented as source-backed

## Repair Handoff

The evaluator should describe repairs so a repair writer can act without guessing.

Bad:

- `The explanation is weak.`

Good:

- `Expand section 3.2 to explain why verifier-guided search is needed, map it to coverage units lec08_u0012 and lec08_u0013, and add symbol explanations for the displayed objective.`

## Course-Specific Checks

- preserve official time exceptions for L08 and L11
- preserve no-class schedule entries at book level
- do not conflate theorem proving with autoformalization or verification
- do not conflate multimodal web agents with GUI or OS agents
- do not treat blog posts, project pages, and magazine articles as if they were arXiv papers

## Repair Loop Limit

- maximum 3 repair passes per lecture
- if the third pass still fails, require `unresolved_issues.md` and a course-level omission entry before the lecture can be marked incomplete
