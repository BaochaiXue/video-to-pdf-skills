# Evaluator Playbook

This file tells the evaluator how to critique a lecture note.

## Job

The evaluator is not a co-writer.
The evaluator is the gatekeeper for delivery.

The evaluator must:

- score the note with the rubric
- surface blocking issues
- produce repair-ready output

## Inputs

Read at minimum:

- `lecture_plan.json`
- `coverage_units.jsonl`
- `figure_plan.json`
- `figure_manifest.json`
- latest `lecture_XX_note.tex`
- `transcript.jsonl`
- `slides.jsonl`
- `segments.jsonl`

## Blocking Issue Format

Each blocking issue should contain:

- `issue_id`
- `type`
- `unit_id` or another stable target
- `problem`
- `required_fix`

Example categories:

- `coverage_gap`
- `figure_provenance_missing`
- `formula_explained_insufficiently`
- `code_logic_missing`
- `hallucination_risk`
- `coherence_break`

## Evaluation Style

- Be concrete.
- Prefer observable failures over vague impressions.
- Do not say “overall good” when a blocking issue exists.

## Repair Handoff

The evaluator should describe fixes so the repair writer can act without guessing.

Bad:

- “The derivation is weak.”

Good:

- “Expand the update rule in section 2.3 into step-by-step math and explain each symbol after the display equation.”

## Pass Policy

If one blocking issue exists, the lecture fails.

`repair_required` should be:

- `true` when `overall = fail`
- `false` when `overall = pass`
