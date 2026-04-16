# S11-751/18-781 Writing Contract

All workers and subagents must follow this contract.

## Ownership

- Each worker owns only its assigned lecture directories under `lectures/`.
- Do not edit other workers' lecture directories.
- Do not revert or overwrite files created by other workers.

## Required outputs per lecture

For each owned lecture directory, produce or update:

- `lecture_plan.json`
- `contracts/seg_##_contract.md`
- `figure_plan.json`
- `eval_reports/pass_##.json`
- `repair_log.jsonl`
- `transcript.jsonl`
- `slides.jsonl`
- `segments.jsonl`
- `lecture_XX_note.tex`
- `lecture_XX_note.pdf`
- `source_manifest.json`
- `coverage_units.jsonl`
- `omission_log.jsonl`
- `figure_manifest.json`

Use `XX` as the zero-padded session index in this run, not the public YouTube lecture number.

If a lecture passes evaluator and validator gates, also export final user-facing copies to:

- `deliverable/lectures/<lecture_slug>/lecture_XX_note.tex`
- `deliverable/lectures/<lecture_slug>/lecture_XX_note.pdf`

If a requested lecture deliverable has not been exported there yet, the lecture is not finished.

## Source usage

- Treat `transcript.jsonl`, `slides.jsonl`, `segments.jsonl`, and `source_manifest.json` as the primary structured evidence layer.
- Treat the official WAVLab course page, public YouTube metadata, public subtitles, and any fetched slides/PDFs/readings as co-equal sources.
- Treat `supplement/cs224s_spring2025/source_manifest.json` as supplemental evidence only. It can enrich missing explanations, broaden coverage, and add newer context, but it must be labeled as supplementation or extension when it is not part of the CMU source set.
- Preserve substantive content; do not silently drop slide-heavy or caveat-heavy material.
- If content cannot be covered, record it in `omission_log.jsonl`.

## Coverage gate

- Generate or refresh `lecture_plan.json` before prose writing.
- Generate segment contracts under `contracts/` before prose writing.
- Generate non-empty `coverage_units.jsonl` before writing or revising `lecture_XX_note.tex`.
- Every required coverage unit must end in exactly one status: `covered`, `partial`, `duplicate`, `omitted`, or `unclassified`.
- `covered` and `partial` units must have `mapped_section`.
- `partial` units must explain the gap in `notes`.
- `duplicate` and `omitted` units must have matching entries in `omission_log.jsonl`.
- `required=true` units may not remain `unclassified` when a lecture is marked deliverable.

## Segmentation

- If a lecture is over 20 minutes, has more than 300 subtitle spans, or is in `course mode`, `segments.jsonl` is mandatory.
- If a lecture has no public video or no public slides, keep segmentation and coverage artifacts anyway, using schedule or course-page evidence as fallback.
- When source gaps remain, keep the lecture in `blocked` state with explicit omission reasons instead of forcing unsupported prose.

## Figure policy

- Emit `figure_plan.json` before finalizing `figure_manifest.json`.
- Every deliverable lecture note should include explanatory figures when official material permits.
- Prefer slide diagrams, architecture illustrations, tables, plots, and process figures over decorative images.
- Record all included figures in `figure_manifest.json`.
- If a figure comes from a video frame, record time provenance.

## Writing policy

- Write in Chinese.
- Use `skills/youtube-render-pdf/assets/notes-template.tex` as the base structure.
- Use `\section{}` and `\subsection{}`.
- End every major section with `本章小结`.
- End the lecture with `总结与延伸`.
- Mark non-CMU supplementation clearly, such as `CS224S 补充` or `延伸解释`.
- Explain motivation before mechanism.
- Expand formulas, comparisons, and workflow diagrams instead of collapsing them into short summaries.

## Validation

- The latest evaluator report under `eval_reports/` must be `pass` before the lecture is marked deliverable.
- Record repairs in `repair_log.jsonl`.
- Compile with `xelatex -interaction=nonstopmode -halt-on-error`.
- Fix broken relative paths before finishing.
- Ensure `figure_manifest.json` matches the figures actually present in the `.tex`.
- If a lecture cannot pass because sources are missing, record the blocking reason and exclude it from the final merged textbook until fixed.

## Deliverable export

- Do not move or delete the canonical lecture workspace outputs under `lectures/`; copy deliverable artifacts into `deliverable/`.
- Only export lecture notes to `deliverable/lectures/<lecture_slug>/` after the lecture has passed evaluator and validator gates.
- Only export the merged textbook to `deliverable/book/` after the merged build is considered deliverable.
- Keep `deliverable/` clean: final `.tex`, final `.pdf`, and other explicitly user-facing final assets only.
- If a requested final deliverable file is missing from `deliverable/`, the task remains unfinished and must continue until export succeeds or an explicit blocking issue is recorded.
