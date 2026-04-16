# Writing Contract: S294-277 Robots That Learn (Fall 2024)

This run is building a textbook-grade Chinese course note for `CS/S294-277: Robots That Learn (Fall 2024)`.

## Non-Negotiables

- Coverage-first over elegance.
- Source-grounded over speculative completion.
- The repo is the record system: every planning, coverage, omission, figure, evaluation, and repair decision must be written to disk.
- No lecture enters the final merged textbook unless it passes evaluator and validator, or has an explicit blocking note and omission reason.
- Each lecture is its own workspace and its own delivery gate.
- Once a lecture note or merged textbook is actually deliverable, copy the final `.tex` and `.pdf` into the run-local `deliverable/` folder.

## Source Scope

Use all relevant official materials when available:

- official course page snapshot in `meta/course_page.html`
- official playlist metadata in `meta/playlist_flat.json`
- shared course notes PDF
- per-lecture YouTube metadata, thumbnail, and subtitles
- per-lecture official slides when downloaded
- official reading list and downloaded reading PDFs under `materials/readings/`
- official backup video links listed on the course page

If a source is inaccessible, too large, or absent on the official page, keep that gap explicit in:

- `source_manifest.json`
- `omission_log.jsonl`
- the final appendix of the merged textbook

## Known Bootstrap Gaps

These are already known from source acquisition and should not be “papered over”:

- `01_introduction`: no official video or subtitle; `slides.pdf` is currently unavailable in the workspace, but shared course notes are available.
- `02_biomechanics_of_walking_and_running`: no official video or subtitle in the playlist; use slides plus shared course notes and readings.
- `04_human_hand_and_dexterous_object_manipulation`: no official slide or video file on the course page; use shared course notes plus the explicitly linked supplemental human-hand video.
- `backup_materials_pdf`: the shared backup package was marked `skipped_too_large` during bootstrap; if a later worker needs it, re-acquire it deliberately and log the reason.
- Playlist segmentation does not exactly match the course-page schedule. Do not assume title-level 1:1 alignment without reading `meta/course_schedule.json`.

## Lecture Workflow

For each lecture workspace:

1. Read `meta.json`, `source_manifest.json`, `lecture_plan.json`, and `contracts/`.
2. Treat `transcript.jsonl`, `slides.jsonl`, and `segments.jsonl` as the canonical evidence layer.
3. Update `coverage_units.jsonl` so every required unit becomes one of:
   - `covered`
   - `partial`
   - `duplicate`
   - `omitted`
4. Add `omission_log.jsonl` entries for every substantive omission.
5. Write `lecture_XX_note.tex` from the evidence layer, not from `transcript.txt` alone.
6. Record figure selection in `figure_plan.json` and final figures in `figure_manifest.json`.
7. Run evaluator, repair only against evaluator findings, and append `repair_log.jsonl`.
8. Validate before considering the lecture deliverable.
9. If the lecture or merged textbook is the final handoff artifact, place the final `.tex/.pdf` in `deliverable/`.

## Writing Standard

- Main language: Chinese.
- Keep English terms, paper titles, model names, benchmark names, and algorithm names.
- On first mention of important concepts, use Chinese plus English, such as `推理时计算 (inference-time computation)`.
- Expand dense formula slides and derivations step by step.
- When code or algorithm logic appears, explain role, mechanism, and consequences; do not only quote snippets.
- End every major section with `\subsection{本章小结}`.
- End every lecture with a final synthesis section and, where supported, extended reading discussion.

## Worker Split

Preferred split for future subagents:

- `1` course integrator: terminology, chapter ordering, cross-lecture synthesis.
- `N` lecture workers: each owns one lecture directory and does not rewrite others.
- `1` figure worker: frame extraction, crop decisions, and figure provenance.
- `1` evaluator/repair worker: runs evaluator, fixes blocking issues, and records repair logs.

Workers are not alone in the repo. Do not revert other lectures. Keep ownership scoped to the assigned lecture directory or build file.
