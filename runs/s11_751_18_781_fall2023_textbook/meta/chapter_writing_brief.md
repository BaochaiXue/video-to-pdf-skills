# Chapter Writing Brief

Use this brief when writing `lecture_XX_note.tex`.

## Required structure

- Start from `skills/youtube-render-pdf/assets/notes-template.tex`
- Fill title-page metadata from `meta.json`
- Use Chinese as the main language
- Keep important English terms, algorithm names, model names, paper titles, and benchmarks
- First mention format: `中文（English）`
- Use `\section{}` and `\subsection{}`
- End every major section with `\subsection{本章小结}`
- End the lecture with `\section{总结与延伸}`

## Grounding

- Primary evidence: `transcript.jsonl`, `slides.jsonl`, `segments.jsonl`, `source_manifest.json`, `coverage_units.jsonl`
- Use `omission_log.jsonl` for anything blocked or partially covered
- If a point comes from `CS224S` or any non-CMU source, label it `CS224S 补充` or `延伸解释`
- Do not present supplementation as if it were part of the original CMU lecture

## Figures

- Each deliverable lecture should include at least:
  - `figures/coverage_map.png`
  - `figures/timeline.png`
- These can be generated with `python3 build/generate_summary_figures.py <lecture_slug>`
- Additional figures are welcome if source-backed
- After writing the note, run `python3 build/sync_figure_manifest.py <lecture_slug>`

## Writing depth

- Expand dense algorithmic material step by step
- For formulas:
  - show display math
  - explain what the formula means
  - explain every symbol right after the formula
- For dynamic programming lectures, do not compress recurrences into one sentence
- For modeling lectures, distinguish:
  - task definition
  - assumptions
  - model factorization or architecture
  - training objective
  - decoding or inference
  - strengths, failure modes, and tradeoffs

## Minimum chapter expectations

- 5 or more substantive sections for public lectures
- 2 or more included figure assets
- No unresolved `[cite]` or `TODO`
- Coverage units should map to real sections, not only `本章小结`
