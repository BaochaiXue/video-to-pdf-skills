# UCB Agentic AI Fall 2025 Textbook Run

This run is the harness-managed textbook build for:

- course: `UCB CS294/194-196: Agentic AI`
- term: `Fall 2025`
- official Berkeley RDI page: <https://rdi.berkeley.edu/agentic-ai/f25>
- public MOOC syllabus: <https://agenticai-learning.org/f25>
- public YouTube playlist: <https://www.youtube.com/playlist?list=PLS01nW3RtgoqGkm4UeqNeZLccW-OGc1fJ>

## Scope

- Primary coverage target: the full Fall 2025 Berkeley Agentic AI course.
- Supplemental source programs:
  - `UCB CS294/194-196: Large Language Model Agents (Fall 2024)` when official public recordings or slides can be verified.
  - `Stanford CS329A: Self-Improving AI Agents (Autumn 2025)` using official course slides and course-page materials; recordings are supplemental if publicly verifiable.

## Canonical workflow

1. `build/bootstrap_course.py`
   Downloads or refreshes official lecture metadata, slide bundles, reading assets, thumbnails, subtitle bundles, and lecture-local workspaces.

2. `build/bootstrap_harness.py`
   Builds the harness layer for each lecture:
   - `transcript.jsonl`
   - `slides.jsonl`
   - `segments.jsonl`
   - `lecture_plan.json`
   - `figure_plan.json`
   - `contracts/segment_##_contract.md`
   - `repair_log.jsonl`

3. Lecture workers write lecture-local outputs:
   - `coverage_units.jsonl`
   - `omission_log.jsonl`
   - `figure_manifest.json`
   - `lecture_XX_note.tex`
   - `lecture_XX_note.pdf`
   - `eval_reports/pass_##.json`

4. `build/build_course_manifest.py`
   Rebuilds course-level status and artifact pointers.

5. `build/compile_all_lecture_notes.py`
   Compiles lecture notes through the validator path.

6. `build/merge_course_notes.py`
   Merges lecture PDFs into one course textbook after evaluator-gated acceptance, then copies the final `.tex` and `.pdf` into `deliverable/`.

## Policy

- Coverage-first: preserve detailed source coverage before compression.
- Source-grounded: all substantive content must come from the course page, videos, slides, readings, or clearly marked extension.
- Best effort on source gaps: missing recordings, missing slides, or broken reading links must be logged in `omission_log.jsonl`; they do not block the rest of the course run.
- Lecture-first: each lecture becomes a standalone chapter before any course-level merge is attempted.
- Supplemental-source discipline: Berkeley Fall 2024 and Stanford CS329A materials may enrich explanation, but they must be explicitly recorded in manifests and never silently replace primary Fall 2025 Berkeley sources.
- Final-hand-off-first: once the merged textbook is generated, copy the final `.tex` and `.pdf` into `deliverable/`.
