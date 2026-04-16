# CS231N Spring 2025 Textbook Run

This run is the harness-managed textbook build for:

- course: `CS231N: Deep Learning for Computer Vision`
- term: `Spring 2025`
- official course page: <https://cs231n.stanford.edu/2025/>
- official schedule: <https://cs231n.stanford.edu/2025/schedule.html>
- official public playlist: <https://www.youtube.com/playlist?list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16>

## Canonical workflow

1. `build/bootstrap_course.py`
   Downloads or refreshes official slides, reading assets, lecture metadata, thumbnails, and subtitle bundles, then writes lecture workspaces.

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
- Source-grounded: all substantive content must come from the course page, slides, readings, subtitles, or clearly marked extension.
- Best effort on source gaps: missing slides, broken reading links, or unavailable assets must be logged in `omission_log.jsonl`; they do not block the rest of the course run.
- Lecture-first: each lecture becomes a standalone chapter before any course-level merge is attempted.
- Final-hand-off-first: once the merged textbook is actually generated, copy the final `.tex` and `.pdf` into `deliverable/` rather than treating `build/` as the user-facing handoff location.

## Current notes

- Lecture 18 has no official slide PDF on the published Spring 2025 schedule page.
- Some schedule-linked reading URLs return `404`; these are preserved in reading manifests and mirrored into lecture omission logs.
