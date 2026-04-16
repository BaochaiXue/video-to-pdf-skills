# MIT RES.9-009 Writing Contract

This run is a harness-managed textbook build. The workspace is the record system.

## Ownership

- Lecture workers may edit only `lectures/NN_*` workspaces with zero-padded numeric prefixes.
- Book workers may edit only `book/`.
- Harness scripts may edit only `build/`.
- Do not overwrite or revert files created by other workers.

## Required Lecture Artifacts

Each lecture workspace must eventually produce:

- `transcript.jsonl`
- `slides.jsonl`
- `segments.jsonl`
- `lecture_plan.json`
- `coverage_units.jsonl`
- `omission_log.jsonl`
- `figure_plan.json`
- `figure_manifest.json`
- `eval_reports/pass_##.json`
- `repair_log.jsonl`
- `lecture_XX_note.tex`
- `lecture_XX_note.pdf`
- `source_manifest.json`

Use `XX` as the zero-padded lecture index.

## Source Rules

- Treat `transcript.jsonl`, `slides.jsonl`, and `segments.jsonl` as the primary structured evidence layer.
- Treat the video, official slides, readings, and course page as co-equal sources.
- Preserve substantive teaching content from both speech and slides.
- If something cannot be covered cleanly, record it in `omission_log.jsonl` instead of silently dropping it.

## Coverage Gate

- Generate `lecture_plan.json` before prose writing.
- Generate `coverage_units.jsonl` before writing or revising `lecture_XX_note.tex`.
- Keep every required unit in exactly one status: `covered`, `partial`, `duplicate`, `omitted`, or `unclassified`.
- `partial` units must explain the gap in `notes`.
- `omitted` and `duplicate` units must be justified in `omission_log.jsonl`.
- Required units may not remain `unclassified` when the lecture is ready for merge.

## Segmentation

- Segment every lecture workspace before writing the chapter note.
- Use the segment contract files under `contracts/segment_##_contract.md`.
- If parallel subagents are not available, process segments serially and keep the coverage ledger explicit.

## Figure Policy

- Emit `figure_plan.json` before finalizing `figure_manifest.json`.
- Prefer official slide figures, page crops, and video frames over decorative images.
- Record provenance for every delivered figure.
- If a figure is frame-derived, include time provenance.

## Writing Policy

- Write in Chinese.
- Keep English terminology, paper titles, algorithms, model names, and benchmark names in their standard form.
- Use `\section{}` and `\subsection{}`.
- End every major section with `\subsection{本章小结}`.
- End every chapter with `\section{总结与延伸}`.
- Expand formulas, derivations, and tradeoffs instead of compressing them into short summaries.

## Evaluation and Repair

- The latest report under `eval_reports/` must have `overall = pass` before the chapter can enter `book/`.
- Record fixes in `repair_log.jsonl`.
- A chapter is not deliverable if required coverage remains unresolved or the evaluator reports blocking issues.

## Book Merge

- `build/build_course_manifest.py` assembles the course manifest from `lectures/`.
- `build/merge_book.py` builds `book/main.tex` and `book/textbook_source_manifest.json`.
- `build/compile.sh` is the canonical book build entry point.
- `build/validate_textbook.py` is the gate before merge or delivery.

## Final Deliverable

- If a lecture is the final requested handoff artifact, copy its final `.tex` and `.pdf` into `deliverable/lectures/<lecture_slug>/`.
- If the merged textbook is the final requested handoff artifact, copy its final `.tex` and `.pdf` into `deliverable/book/`.
- Keep the canonical workspace outputs in `lectures/`, `book/`, and `build/`; `deliverable/` is the user-facing export area.
- Do not treat the run as complete if the final handoff files were generated but not exported into `deliverable/`.

## Failure Discipline

- Keep omissions explicit.
- Keep evaluator failures explicit.
- Do not fake completion by deleting unresolved rows or suppressing reports.
