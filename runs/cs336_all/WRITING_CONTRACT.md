# CS336 Spring 2026 Writing Contract

## Required Evidence

- Each lecture must include `meta.json`, `source_manifest.json`, `transcript.jsonl`, `slides.jsonl`, `segments.jsonl`, `coverage_units.jsonl`, `omission_log.jsonl`, `figure_plan.json`, `figure_manifest.json`, `lecture_plan.json`, and a passing `eval_reports/pass_##.json`.
- Each `lecture_plan.json` must set `textbook_mode: true`.
- The latest eval report must include a `textbook_chapter_style` score.
- Final delivery must copy `cs336_complete_notes.tex` and `cs336_complete_notes.pdf` into `deliverable/`.
- Missing public videos or official materials must be logged in `omission_log.jsonl` and surfaced in the final appendix.

## Writing Rules

- Main prose is Chinese.
- Important terms keep English and use bilingual first mentions.
- Formulas, algorithm names, model names, benchmark names, and paper names keep standard English notation.
- Core sections must be written according to the corresponding video, PPT/PDF, Python lecture script, and official course materials. Each core section must include `本节来源依据` mapping prose back to video windows and official source groups.
- Source-rich textbook-mode chapters must include at least two non-cover instructional figures. Prefer official PPT/PDF pages; use generated source-grounded figures only when official image assets are unavailable.
- Textbook explanations not directly present in source materials must be labeled `延伸解释`.

## Delivery Gate

No lecture enters the merged textbook unless its latest evaluator report passes and the shared validator accepts the lecture workspace with `python3 build/validate_youtube_note.py --compile`.
