# S294-277 Robots That Learn (Fall 2024)

Harness-managed course textbook run for UC Berkeley `CS/S294-277: Robots That Learn (Fall 2024)`.

This run treats the course page, official YouTube playlist, course notes, backup notes, lecture slides, and official readings as the record system for a coverage-first Chinese textbook build.

## Layout

- `build/`: bootstrap, manifest, evaluation, validation, and merge scripts
- `deliverable/`: final handoff `.tex/.pdf` after a lecture or merged textbook passes the delivery gate
- `lectures/`: per-lecture workspaces with evidence, coverage ledgers, note drafts, and evaluator reports
- `materials/`: shared course assets such as downloaded readings and slide PDFs
- `meta/`: course-level raw metadata and source inventories
- `raw/`: raw YouTube metadata, thumbnails, and subtitle artifacts
- `text/`: debug text extractions and course bundle sidecars

## Core Commands

Bootstrap course sources and lecture workspaces:

```bash
python3 runs/s294_277_fall2024_robots_that_learn_textbook/build/bootstrap_course.py
```

Evaluate all lectures that already have `lecture_XX_note.tex`:

```bash
python3 runs/s294_277_fall2024_robots_that_learn_textbook/build/evaluate_notes.py
```

Compile and validate a subset of lectures:

```bash
python3 runs/s294_277_fall2024_robots_that_learn_textbook/build/compile_all_lecture_notes.py --lectures 10 11
```

Rebuild the course manifest:

```bash
python3 runs/s294_277_fall2024_robots_that_learn_textbook/build/build_course_manifest.py
```

Merge passing lecture PDFs into a single course PDF:

```bash
python3 runs/s294_277_fall2024_robots_that_learn_textbook/build/merge_course_notes.py
```

## Delivery Discipline

- Every lecture must keep `meta.json`, `source_manifest.json`, `transcript.jsonl`, `slides.jsonl`, `segments.jsonl`, `coverage_units.jsonl`, `omission_log.jsonl`, `figure_plan.json`, `figure_manifest.json`, `repair_log.jsonl`, and `eval_reports/`.
- Missing or inaccessible sources must be explicit in `source_manifest.json` and `omission_log.jsonl`.
- Final textbook merge should only include lectures that have compiled PDFs and a passing evaluator report.
- If a final lecture note or merged course textbook is considered deliverable, place the exported `.tex` and `.pdf` in `deliverable/`.
