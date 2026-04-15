# Coverage Schema

This file defines the canonical machine-readable artifacts used by the lecture note harness.

## `transcript.jsonl`

One subtitle-aligned span per row.

Required fields:

- `unit_id`
- `source_type`
- `source_id`
- `loc.start`
- `loc.end`
- `text`
- `required`

Recommended values:

- `source_type = "subtitle_span"`
- `source_id = "subtitle_srt"`

## `slides.jsonl`

One slide page per row.

Required fields:

- `unit_id`
- `source_type`
- `source_id`
- `loc.page`
- `text`
- `asset_path`
- `required`

Recommended values:

- `source_type = "slide_page"`
- `source_id = "slides_pdf"`

## `segments.jsonl`

One lecture segment per row.

Required fields:

- `segment_id`
- `start`
- `end`
- `source_unit_ids`
- `target_section_hint`

`source_unit_ids` should reference ids from `transcript.jsonl` and `slides.jsonl`.

## `coverage_units.jsonl`

Coverage ledger rows produced before prose writing.

Required fields:

- `unit_id`
- `source_type`
- `source_id`
- `loc`
- `kind`
- `summary`
- `required`
- `status`
- `mapped_section`
- `figure_ids`
- `notes`

Allowed `status` values:

- `covered`
- `partial`
- `duplicate`
- `omitted`
- `unclassified`

## `repair_log.jsonl`

One repair action per row.

Required fields:

- `pass`
- `issue_id`
- `action`
- `status`
- `notes`

Recommended `status` values:

- `open`
- `fixed`
- `accepted`
- `wont_fix`

## `lecture_plan.json`

Lecture-level planning artifact.

Required fields:

- `lecture_id`
- `title`
- `course_mode`
- `source_inventory`
- `segment_ids`
- `must_cover_kinds`
- `must_emit_artifacts`
- `evaluator_thresholds`

## `figure_plan.json`

Figure-planning artifact.

Each entry must contain:

- `figure_id`
- `source_unit_ids`
- `asset_candidates`
- `selection_reason`
- `required`
- `provenance_type`
- `time_provenance`

## `eval_reports/pass_##.json`

One evaluator pass report.

Required fields:

- `pass`
- `target`
- `overall`
- `scores`
- `blocking_issues`
- `warnings`
- `repair_required`

`scores` should be a dictionary of numeric values in `[0, 1]`.
