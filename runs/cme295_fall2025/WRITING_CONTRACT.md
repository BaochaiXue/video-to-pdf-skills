# CME 295 Writing Contract

All lecture-note workers must follow this contract.

## Output ownership

- Each worker owns only its assigned lecture directories under `lectures/`.
- Do not edit other workers' lecture directories.
- Do not revert or overwrite files created by other workers.

## Required outputs per lecture

For each owned lecture directory, produce:

- `lecture_XX_note.tex`
- `lecture_XX_note.pdf`
- `source_manifest.json`
- `coverage_units.jsonl`
- `omission_log.jsonl`
- `figure_manifest.json`

Use `XX` as the zero-padded lecture index.

## Source usage

- Treat the video subtitle, slide PDF, slide page renders, syllabus topics, and yt-dlp metadata as co-equal sources.
- Preserve substantive content; do not silently drop slide-heavy material.
- If content cannot be covered, record it in `omission_log.jsonl`.

## Figure policy

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

- Compile with `xelatex -interaction=nonstopmode -halt-on-error`.
- Fix broken relative paths before finishing.
- Ensure `figure_manifest.json` matches the figures actually present in the `.tex`.
