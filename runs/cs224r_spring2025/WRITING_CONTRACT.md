# CS224R Writing Contract

All lecture-note workers must follow this contract.

## Output ownership

- Each worker owns only its assigned lecture directories under `lectures/`.
- Do not edit lecture directories owned by other workers.
- Do not revert or overwrite files created by other workers.

## Required outputs per owned lecture

For each owned lecture directory, produce:

- `lecture_XX_note.tex`
- `lecture_XX_note.pdf`
- `source_manifest.json`
- `coverage_units.jsonl`
- `omission_log.jsonl`
- `figure_manifest.json`

Use `XX` as the zero-padded playlist index.

## Source usage

- Treat the subtitle, transcript, official slide PDF, slide page renders, course-page lecture title, and yt-dlp metadata as co-equal sources.
- Preserve substantive technical content from both spoken explanation and slides.
- If something cannot be covered cleanly, log it in `omission_log.jsonl` instead of silently dropping it.

## Figure policy

- Every lecture note must include explanatory figures from the slide deck or video-derived assets.
- Prefer diagrams, algorithms, tables, plots, architecture schematics, and training-pipeline illustrations over decorative images.
- Aim for at least 3 meaningful figures per lecture when the slides support it.
- Record every included figure in `figure_manifest.json`.
- If a figure comes from a video frame, record time provenance.

## Writing policy

- Write in Chinese.
- Use `skills/youtube-render-pdf/assets/notes-template.tex` as the base structure.
- Use `\section{}` and `\subsection{}`.
- End every major section with `\subsection{本章小结}`.
- End the lecture with `\section{总结与延伸}`.
- Explain motivation before mechanism.
- Expand formulas, derivations, and tradeoffs instead of collapsing them into brief bullet summaries.

## Validation

- Compile with `xelatex -interaction=nonstopmode -halt-on-error`.
- Fix broken relative paths before finishing.
- Ensure `figure_manifest.json` matches the figures actually present in the `.tex`.
