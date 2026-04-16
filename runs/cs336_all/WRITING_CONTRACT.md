# CS336 Writing Contract

All lecture-note workers must follow this contract.

## Output ownership

- Each worker owns only its assigned lecture directories under `lectures/`.
- Do not edit lecture directories owned by other workers.
- Do not revert or overwrite files created by other workers.

## Required outputs per owned lecture

For each owned lecture directory, produce or preserve:

- `transcript.jsonl`
- `slides.jsonl`
- `segments.jsonl`
- `lecture_plan.json`
- `contracts/segment_##_contract.md`
- `figure_plan.json`
- `eval_reports/pass_##.json`
- `repair_log.jsonl`
- `lecture_XX_note.tex`
- `lecture_XX_note.pdf`
- `source_manifest.json`
- `coverage_units.jsonl`
- `omission_log.jsonl`
- `figure_manifest.json`

## Source usage

- Treat `transcript.jsonl`, `slides.jsonl`, and `segments.jsonl` as the primary structured evidence layer.
- Treat subtitles, official lecture traces, official script extracts, slide-like markdown, and yt-dlp metadata as co-equal sources.
- Preserve substantive technical content from both spoken explanation and official materials.
- If a source gap remains, log it in `omission_log.jsonl` instead of silently dropping it.

## Writing policy

- Write in Chinese.
- Preserve textbook-grade pedagogical depth rather than shortening into a summary.
- End every major section with `\subsection{本章小结}`.
- End the lecture with `\section{总结与延伸}`.

## Validation

- The latest evaluator report under `eval_reports/` must be `pass` before the lecture is considered deliverable.
- Record repairs in `repair_log.jsonl`.
- Run `build/validate_youtube_note.py` before accepting a lecture as complete.
- Compile with `xelatex -interaction=nonstopmode -halt-on-error`.

## Final deliverable

- If the merged textbook is successfully generated, place the final exported `.tex` and `.pdf` in the run-local `deliverable/` folder.

