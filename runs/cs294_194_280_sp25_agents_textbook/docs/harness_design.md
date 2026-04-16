# Harness Design

## Goal

Run this course as a harness-managed textbook build rather than a one-shot prompt.
The workspace itself is the record system.

Core principles:

- plan first
- write structured artifacts before prose
- keep lectures isolated in their own workspaces
- gate delivery on evaluator and validator results
- make failures explicit rather than silent

## Course-Level Stages

1. `course planner`
   seeds `COURSE_SPEC.md`, `COURSE_SOURCE_MANIFEST.json`, course-level coverage, omissions, and docs.
2. `lecture pipeline`
   runs one harness-managed lecture workflow per `lectures/lecXX_*`.
3. `book pipeline`
   merges only lectures that passed evaluator and validator, then runs cross-lecture editing and final textbook evaluation.

## Per-Lecture Agent Graph

Each lecture should run at least these agents:

1. `Source Curator Agent`
2. `Transcript & Slide Parser Agent`
3. `Coverage Planner Agent`
4. `Figure / Visual Provenance Agent`
5. `Lecture Writer Agent`
6. `Reading Integrator Agent`
7. `Skeptical Evaluator Agent`
8. `Repair Writer Agent`

Recommended handoff order:

`source acquisition -> parsing/alignment -> coverage ledger + segment contracts -> figure plan -> lecture writing -> reading integration -> evaluation -> repair -> validation`

## Lecture Artifact Contract

Every lecture workspace should eventually contain:

- `source_manifest.json`
- `transcript_raw.srt` or `transcript_raw.vtt`
- `transcript.jsonl`
- `slides.pdf` when available
- `slides.jsonl`
- `aligned_units.jsonl`
- `slide_transcript_alignment.jsonl`
- `formulas.jsonl`
- `code_units.jsonl`
- `paper_mentions.jsonl`
- `low_confidence_spans.jsonl`
- `coverage_units.jsonl`
- `segment_plan.md`
- `segment_contracts/segment_XX_contract.md`
- `figure_plan.jsonl`
- `figure_manifest.json`
- `lecture.tex` or `lecture_XX_note.tex`
- `lecture_notes.md`
- `lecture_summary.md`
- `exercises.md`
- `glossary_delta.md`
- `notation_delta.md`
- `readings_integration.md`
- `paper_summaries.jsonl`
- `reading_coverage_units.jsonl`
- `eval_report.json`
- `eval_report.md`
- `repair_log.jsonl`
- `unresolved_issues.md` only when repair still fails

## Delivery Gate

A lecture may enter `book/` only if:

- required source artifacts exist
- required coverage units are no longer `planned` or `unclassified`
- omission reasons are explicit
- every figure has provenance
- evaluator report is `pass`
- validator succeeds
- LaTeX compiles

The merged textbook may be called complete only if:

- `book/main.tex` and `book/textbook.pdf` exist
- textbook validator passes
- the final user-facing `.tex` and `.pdf` are copied into the run-local `deliverable/book/` folder
- the `deliverable/` copy is treated as the stable handoff artifact, while `book/` remains the canonical build workspace

Do not stop at “compiled in `book/`”.
For this run, the task is incomplete until the exported textbook exists under `deliverable/book/` and the validator accepts that exported handoff.

## Course-Specific Requirements

- Use the lecture slugs fixed in `COURSE_SPEC.md`.
- Preserve official special-time notes for L08 and L11.
- Preserve official no-class schedule entries for the final textbook appendix.
- Treat L01 intro deck as an official source alongside the main lecture slides.
- Preserve heterogeneous reading provenance, including arXiv, blogs, project pages, magazine articles, and YouTube talks.

## Local Constraint To Carry Forward

The current `build/` scripts in this run still assume lecture directory names beginning with digits.
This course spec intentionally uses `lecXX_*` names.
Do not change the course spec to fit the helper; patch or shim the helper in the build phase instead.
