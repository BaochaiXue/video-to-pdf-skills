# Agent Roles

This run is managed as a harnessed, multi-agent textbook build.

## Main agent

- Owns course-level planning, source verification, bootstrap scripts, merge, deliverable export, and final validation.
- Decides when supplemental sources from Berkeley Fall 2024 and Stanford CS329A can be incorporated.
- Owns `build/`, run-level manifests, and `deliverable/`.

## Lecture workers

- Each worker owns only its assigned lecture directories under `lectures/`.
- Each worker must read `WRITING_CONTRACT.md` before editing lecture-local files.
- Workers must record all repairs in `repair_log.jsonl`.

## Evaluator / validator workers

- Evaluators judge lecture quality independently of the writer.
- Validators enforce artifact contracts, source manifests, and final compile invariants.
- No lecture may enter the merged textbook without passing evaluator and validator checks, unless a blocking issue is explicitly logged.

## Supplemental-source discipline

- Berkeley `CS294/194-196 Large Language Model Agents (Fall 2024)` is a supplemental source program.
- Stanford `CS329A Self-Improving AI Agents (Autumn 2025)` is a supplemental source program.
- Supplemental material may clarify, extend, or cross-reference the main Fall 2025 Berkeley course, but must never silently replace primary Fall 2025 Berkeley sources.
