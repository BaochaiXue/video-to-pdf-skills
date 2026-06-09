---
name: youtube-render-pdf
description: >-
  Generate a professional, coverage-first, detailed, figure-rich LaTeX course
  note and final PDF from a YouTube lecture, tutorial, technical talk, or course
  playlist. Use when the user provides a YouTube URL and wants structured Chinese
  notes that jointly cover the video, subtitles, official slides or PDFs, lecture
  code, course pages, and other linked materials when available, prioritizing
  complete source coverage over concise summarization, including the original
  cover image on the front page, auditable coverage or figure manifests, and a
  rendered PDF. Prefer platform subtitles first, and when suitable subtitles are
  unavailable, fall back to the same device-aware ASR backend policy used by the
  Bilibili skill.
---

# YouTube Render PDF

Use this skill to turn a YouTube video into a complete, compileable `.tex` note and a rendered PDF.
This skill should be executed as a harness-managed lecture note pipeline, not as a one-shot writing prompt.

## Read First

Before doing anything else, read:

1. [../references/video-render-pdf-common.md](../references/video-render-pdf-common.md)
2. [../references/video-note-harness.md](../references/video-note-harness.md)
3. [../references/note-quality-rubric.md](../references/note-quality-rubric.md)
4. [../references/coverage-schema.md](../references/coverage-schema.md)
5. [../references/figure-provenance.md](../references/figure-provenance.md)
6. [../references/evaluator-playbook.md](../references/evaluator-playbook.md)

The common workflow defines coverage-first note behavior.
The harness reference defines stage orchestration, evaluator gating, and repair loops.
When this file conflicts with the shared references, follow this file only for YouTube-specific acquisition behavior.

## Platform-Specific Goal

Produce a professional Chinese lecture note from a YouTube URL.

In addition to the shared workflow, this skill must:

- prefer YouTube's official subtitle tracks before ASR
- treat playlists, linked course pages, and description-linked lecture materials as strong `course mode` signals
- prefer linked official materials over ad hoc inference when a lecture page, slide deck, notebook, or repo exists
- treat `transcript.jsonl`, `slides.jsonl`, and `segments.jsonl` as the canonical machine-readable evidence layer whenever they exist
- create harness artifacts such as `lecture_plan.json`, `contracts/segment_##_contract.md`, `figure_plan.json`, `eval_reports/pass_##.json`, and `repair_log.jsonl` when local lecture workspaces are available
- for playlists, long videos, or `course mode`, proactively recommend that the user explicitly request parallel subagents so `spawn_agent` can be used

## YouTube-Specific Source Acquisition

1. Inspect metadata before writing prose.
   Prefer title, chapters, duration, description, playlist context, linked resources, thumbnail availability, and subtitle availability.

2. Enter `course mode` automatically when the video is clearly part of:
   - a course playlist
   - a semester lecture series
   - a university channel lecture sequence
   - a description that links to official lecture materials, course pages, slides, notebooks, repos, or schedules

3. In `course mode`, treat the YouTube video as only one source among many.
   Collect and log every official resource you can access instead of defaulting to a video-only reconstruction.

4. Prefer the best matching subtitle track.
   - use manual subtitles over auto-generated subtitles when both are available
   - prefer the default language that best matches the video or the user's requested language
   - keep timestamps intact for figure provenance and coverage accounting
   - if you generate local lecture artifacts, preserve this information in `transcript.jsonl` rather than flattening it away too early

5. If no suitable subtitle track is available, use a device-aware ASR backend.
   - on `CUDA / NVIDIA`, default to `Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B`
   - on `Apple Silicon Mac`, default to a Whisper backend, prioritizing `whisper.cpp`
   - normalize the result into timestamped `SRT` or `segments`

6. Acquire the original cover image before writing the `.tex`.
   Prefer the highest-resolution official thumbnail exposed by metadata.

7. Prefer the best usable video source for figure extraction.
   Probe formats and choose the highest resolution that is actually downloadable in the current environment.

8. Build a structured evidence layer before writing prose when local lecture workspaces are available.
   Prefer:
   - `transcript.jsonl` for subtitle-aligned spans
   - `slides.jsonl` for page-aligned slide evidence
   - `segments.jsonl` for required segment boundaries and source-unit assignment
   Keep `transcript.txt` and `official.txt` only as debug artifacts.

## Playlist and Course Handling

- If the user gives a playlist URL, preserve playlist order unless the user asks for a different order.
- When the playlist is a university course or lecture series, default to lecture-by-lecture processing rather than one giant prose pass.
- When official lecture materials exist per lecture, build per-lecture source inventories rather than a single playlist-level inventory.
- If some playlist entries are unavailable, private, or missing official materials, record that gap explicitly instead of silently skipping it.
- If the workload is large and the user did not explicitly ask for parallel agent work, recommend phrasing the request with wording such as `请 spawn 多个 subagents 并行执行`.
- If the lecture is longer than 20 minutes, has more than 300 subtitle spans, or enters `course mode`, segmentation is mandatory even without subagents; fall back to serial segment processing instead of one monolithic pass.

## Teaching Signal Inventory

Build the note from all high-signal teaching sources when available:

- video title and chapter structure
- the video's original cover image and key metadata
- on-screen diagrams, formulas, tables, plots, and architecture slides
- subtitle explanations, examples, and verbal emphasis
- short high-signal original dialogue segments in interview, panel, podcast, or conversation videos, when exact wording adds presence, humor, intuition, or unusually compact information
- code snippets shown or described in the talk

## Harness Expectations

- The planner must emit `lecture_plan.json` before any prose writing starts.
- Segment contracts under `contracts/` must exist before the writer produces `lecture_XX_note.tex`.
- The writer should consume contracts plus structured evidence, not flattened debug files.
- The evaluator must emit `eval_reports/pass_##.json` and is allowed to fail the lecture.
- Delivery should be blocked when the latest evaluator report is not `pass`.

## YouTube-Specific Non-Teaching Content

Skip content that does not contribute to the actual lesson, such as:

- greetings
- channel housekeeping
- small talk
- routine back-and-forth that does not add information, tension, humor, intuition, or teaching value
- sponsorship
- routine subscribe or like reminders
- closing pleasantries

Keep the speaker's closing discussion when it carries actual teaching value, such as synthesis, limitations, future work, tradeoffs, advice, or open questions.

For shared writing rules, figure handling, visualization, final checklist, and delivery requirements, follow the shared references listed in `Read First`.
Those references include the high-recall frame selection policy, figure time provenance rules, and `dialoguebox` usage constraints.

## Asset

- `assets/notes-template.tex`: default LaTeX template to copy and fill
