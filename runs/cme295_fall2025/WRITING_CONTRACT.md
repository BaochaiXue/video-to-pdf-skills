# CME 295 Writing Contract

All lecture-note workers must follow this contract.

## Output ownership

- Each worker owns only its assigned lecture directories under `lectures/`.
- Do not edit other workers' lecture directories.
- Do not revert or overwrite files created by other workers.

## Required outputs per lecture

For each owned lecture directory, produce:

- `lecture_plan.json`
- `contracts/segment_##_contract.md`
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

Use `XX` as the zero-padded lecture index.

## Source usage

- Treat `transcript.jsonl`, `slides.jsonl`, and `segments.jsonl` as the primary structured evidence layer.
- Treat the video subtitle, slide PDF, slide page renders, syllabus topics, and yt-dlp metadata as co-equal sources.
- Preserve substantive content; do not silently drop slide-heavy material.
- If content cannot be covered, record it in `omission_log.jsonl`.
- Keep `transcript.txt` and `official.txt` only as debug artifacts; do not treat them as the authoritative source for writing.

## Coverage gate

- Generate or refresh `lecture_plan.json` before prose writing.
- Generate segment contracts under `contracts/` before prose writing.
- Generate non-empty `coverage_units.jsonl` before writing or revising `lecture_XX_note.tex`.
- Every required coverage unit must end in exactly one status: `covered`, `partial`, `duplicate`, `omitted`, or `unclassified`.
- `covered` and `partial` units must have `mapped_section`.
- `partial` units must explain the gap in `notes`.
- `duplicate` and `omitted` units must have matching entries in `omission_log.jsonl`.
- `required=true` units may not remain `unclassified` when the lecture note is considered complete.

## Segmentation

- If a lecture is over 20 minutes, has more than 300 subtitle spans, or is in `course mode`, `segments.jsonl` is mandatory.
- Segmentation is mandatory even when no parallel subagents are authorized.
- If subagents are not explicitly authorized, process segments serially.
- If subagents are explicitly authorized, split work into segment coverage workers plus one integration pass.

## Figure policy

- Emit `figure_plan.json` before finalizing `figure_manifest.json`.
- Every lecture note must include explanatory figures from the slide deck or video-derived assets.
- Prefer slide diagrams, architecture illustrations, tables, plots, and process figures over decorative images.
- Aim for at least 2 meaningful figures per lecture when source material permits.
- Record all included figures in `figure_manifest.json`.
- If a figure comes from a video frame, record time provenance.

## Writing policy

- Write in Chinese.
- Use `skills/youtube-render-pdf/assets/notes-template.tex` as the base structure.
- Use `\section{}` and `\subsection{}`.
- End every major section with `\subsection{本章小结}`.
- End the lecture with `\section{总结与延伸}`.
- Explain motivation before mechanism.
- Expand formulas, comparisons, and workflow diagrams instead of collapsing them into short summaries.

## Validation

- The latest evaluator report under `eval_reports/` must be `pass` before the lecture is considered deliverable.
- Record repairs in `repair_log.jsonl`.
- Run `build/validate_youtube_note.py` before accepting a lecture as complete.
- Compile with `xelatex -interaction=nonstopmode -halt-on-error`.
- Fix broken relative paths before finishing.
- Ensure `figure_manifest.json` matches the figures actually present in the `.tex`.
