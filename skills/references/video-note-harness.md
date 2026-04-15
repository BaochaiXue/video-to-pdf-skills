# Shared Video Note Harness

Read this file together with `video-render-pdf-common.md` whenever `youtube-render-pdf` or `bilibili-render-pdf` triggers.

This document defines the harness-managed orchestration layer for lecture note generation.

The common workflow still defines coverage-first writing behavior and delivery standards.
This file defines how the work is staged, handed off, evaluated, repaired, and finally validated.

## Goal

Treat lecture-note generation as a long-running, harness-managed build pipeline instead of a one-shot writing prompt.

The harness must make the work:

- agent-readable
- stage-oriented
- restartable
- evaluator-gated
- repairable
- auditable from local artifacts

The human remains the driver of intent and acceptance.
Agents execute within explicit contracts and feedback loops.

## Fixed Stages

Every lecture note build should follow these stages:

0. `source acquisition`
1. `planner`
2. `coverage extractor`
3. `writer`
4. `figure agent`
5. `evaluator`
6. `repair writer`
7. `final validator`

Do not skip directly from acquisition to note writing in normal operation.

## Stage Contracts

### 0. Source acquisition

Purpose:

- collect canonical local evidence
- normalize source layout
- make the lecture workspace agent-readable

Required inputs:

- metadata
- subtitles or ASR output
- official slides or equivalent non-video material
- cover image
- rendered slide pages or figure candidates when practical

Required canonical outputs:

- `source_manifest.json`
- `transcript.jsonl`
- `slides.jsonl`
- `segments.jsonl`

Debug-only artifacts may also exist:

- `transcript.txt`
- `official.txt`

### 1. Planner

Purpose:

- turn raw evidence into a lecture-level execution plan
- declare what must be covered before any prose is written

Required outputs:

- `lecture_plan.json`

The planner must define:

- lecture identity and title
- source inventory
- segment ids
- must-cover kinds
- required output artifacts
- evaluator thresholds

The planner must not write note prose.

### 2. Coverage extractor

Purpose:

- convert evidence into coverage units
- classify what is required, duplicated, uncertain, or ignorable

Required outputs:

- `coverage_units.jsonl`
- `contracts/segment_##_contract.md`

Coverage extraction must happen before note writing.
If subagents are available, they should own bounded segments and produce coverage ledgers instead of polished prose.

### 3. Writer

Purpose:

- write the lecture note from contracts plus evidence

The writer must consume:

- `lecture_plan.json`
- `coverage_units.jsonl`
- `contracts/segment_##_contract.md`
- `transcript.jsonl`
- `slides.jsonl`
- `segments.jsonl`

The writer must not treat `transcript.txt` or `official.txt` as the authoritative source when the structured evidence exists.

Required outputs:

- `lecture_XX_note.tex`
- `lecture_XX_note.pdf`

### 4. Figure agent

Purpose:

- decide which figures are necessary for teaching value
- record provenance before delivery

Required outputs:

- `figure_plan.json`
- `figure_manifest.json`

The figure agent may use:

- slide page renders
- video frames
- crops
- generated teaching diagrams

Every delivered figure must be explainable and provenance-backed.

### 5. Evaluator

Purpose:

- grade the lecture note against a stable rubric
- surface blocking issues independently of the writer

Required outputs:

- `eval_reports/pass_##.json`

The evaluator must be allowed to fail the note even when the note is fluent and visually polished.
The evaluator is responsible for saying “not done yet”.

### 6. Repair writer

Purpose:

- repair only the issues raised by the evaluator
- record what was changed and what remains unresolved

Required outputs:

- `repair_log.jsonl`

The repair writer must work from evaluator issues, not from a fresh, unconstrained rewrite.

### 7. Final validator

Purpose:

- enforce hard delivery invariants
- prevent shipping notes that failed evaluation or broke artifact contracts

The final validator may block delivery even if PDF compilation succeeds.

## Delivery Gate

Delivery is allowed only when all of the following are true:

- canonical evidence artifacts exist
- harness artifacts exist
- evaluator `overall` is `pass`
- evaluator `blocking_issues` is empty
- validator invariants pass
- `lecture_XX_note.tex` compiles

If any condition fails, the lecture is not deliverable.

## Handoff Rules

When one stage hands off to another, it must prefer structured artifacts over prose summaries.

Examples:

- planner hands off `lecture_plan.json`
- coverage extractor hands off `coverage_units.jsonl` and segment contracts
- figure agent hands off `figure_plan.json`
- evaluator hands off `eval_reports/pass_##.json`
- repair writer appends `repair_log.jsonl`

Do not rely on conversational memory alone.
The workspace itself must be the record system.

## Failure Discipline

When the harness fails:

- record the failure in an artifact
- do not silently continue
- prefer explicit partial states over fake success

If a lecture cannot pass:

- keep the latest evaluator report
- keep the repair log
- keep omissions explicit

## Defaults

- Use the newest evaluator report as the gate for delivery.
- Prefer failing closed rather than shipping low-confidence notes.
- Prefer repairing in bounded passes over doing a wholesale rewrite after evaluation.
