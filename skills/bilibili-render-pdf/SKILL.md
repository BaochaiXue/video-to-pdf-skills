---
name: bilibili-render-pdf
description: >-
  Generate a professional, coverage-first, detailed, figure-rich LaTeX course
  note and final PDF from a Bilibili lecture, tutorial, technical talk, or
  course series. Use when the user provides a Bilibili URL (BV number) and
  wants structured Chinese notes that jointly cover the video, subtitles,
  official slides or PDFs, lecture code, course pages, repos, and other linked
  materials when available, prioritizing complete source coverage over concise
  summarization, including the original cover image on the front page, auditable
  coverage or figure manifests, and a rendered PDF. Falls back to a device-aware
  ASR backend when no CC subtitles are available: default to Qwen3-ASR-1.7B plus
  Qwen3-ForcedAligner-0.6B on CUDA machines, and default to a Whisper backend on
  Apple Silicon Macs.
---

# Bilibili Render PDF

Use this skill to turn a Bilibili video into a complete, compileable `.tex` note and a rendered PDF.
This skill should be executed as a harness-managed lecture note pipeline, not as a one-shot writing prompt.

This skill extends the `youtube-render-pdf` workflow with Bilibili-specific adaptations for subtitle scarcity, login-gated high resolution, multi-part (分P) videos, and platform-specific non-teaching content.

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
When this file conflicts with the shared references, follow this file only for Bilibili-specific acquisition behavior.

## Bilibili vs YouTube: Key Differences

| Aspect | Handling |
|--------|----------|
| Subtitle scarcity | Try CC subtitles first, then ASR, then visual-only mode |
| Login-gated HD | 1080P+ often requires cookies |
| Multi-part videos | Detect 分P and ask the user which parts to process |
| URL formats | Support `bilibili.com/video/BV...` and `b23.tv` |
| Danmaku | Never use danmaku as a teaching source |

## Platform-Specific Goal

Produce a professional Chinese lecture note from a Bilibili URL.

In addition to the shared workflow, this skill must:

- respect 分P structure when building inventories and processing order
- fall back more aggressively from CC subtitles to ASR because platform subtitles are often sparse
- use description links, pinned comments, uploader notes, 网盘材料, and linked repos as strong `course mode` signals
- ignore danmaku as teaching evidence
- create harness artifacts such as `lecture_plan.json`, `contracts/segment_##_contract.md`, `figure_plan.json`, `eval_reports/pass_##.json`, and `repair_log.jsonl` when local lecture workspaces are available
- for long videos, 分P workloads, or `course mode`, proactively recommend that the user explicitly request parallel subagents so `spawn_agent` can be used

## Bilibili-Specific Source Acquisition

### Metadata and Course Detection

1. Inspect metadata before writing prose.
   Prefer title, duration, chapters if present, description, pinned resources, uploader notes, cover availability, subtitle availability, and whether the video belongs to a course, lecture series, or 合集.

2. Detect multi-part (分P) videos and `course mode`.
   - list all parts and ask the user which parts to process before downloading
   - enter `course mode` automatically when the video is clearly part of a course, semester lecture series, uploader合集, or when the description or pinned comment points to official lecture materials

3. In `course mode`, treat the Bilibili video as only one source among many.
   Collect and log official course pages, schedules, slide decks, notebooks, scripts, repos, 网盘材料, or uploader-provided notes whenever accessible.

### Subtitle Acquisition

Priority order:

1. CC subtitles
2. device-aware ASR backend
3. visual-only mode

Prefer manual subtitles over auto-generated subtitles when both are available.
Prefer `zh-Hans`, `zh-CN`, `zh`, or `ai-zh`.
Keep timestamps intact for figure provenance and coverage accounting.

If CC subtitles are unavailable or poor:

- on `CUDA / NVIDIA`, default to `Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B`
- on `Apple Silicon Mac`, default to a Whisper backend, prioritizing `whisper.cpp`
- normalize output into timestamped `SRT` or `segments`

If audio quality is too poor, switch to visual-only mode and record the limitation in the source manifest and omission log.

### Video and Cover Download

1. Acquire the original cover image before writing the `.tex`.
   Prefer the highest-resolution thumbnail exposed by metadata.

2. Prefer the best usable video source for figure extraction.
   Probe formats and choose the highest usable resolution.
   Note that 1080P+ on Bilibili often requires login cookies. If needed, prompt the user to use something like:

```bash
yt-dlp --cookies-from-browser chrome "<URL>"
```

3. Keep all source artifacts local when practical.
   Typical working artifacts include metadata, cover image, subtitles or ASR output, official materials, local video, extracted frames, and generated figures.

## Multi-Part and Course Handling

- Preserve 分P order unless the user asks for a different order.
- In course or lecture-series settings, default to lecture-by-lecture processing rather than one monolithic prose pass.
- Build per-part or per-lecture source inventories rather than one coarse inventory when the parts have distinct materials.
- If some parts are unavailable or missing official materials, record that explicitly instead of silently skipping them.
- If the workload is large and the user did not explicitly ask for parallel agent work, recommend phrasing the request with wording such as `请 spawn 多个 subagents 并行执行`.

## Harness Expectations

- The planner must emit `lecture_plan.json` before any prose writing starts.
- Segment contracts under `contracts/` must exist before the writer produces `lecture_XX_note.tex`.
- The writer should consume contracts plus structured evidence, not flattened debug files.
- The evaluator must emit `eval_reports/pass_##.json` and is allowed to fail the lecture.
- Delivery should be blocked when the latest evaluator report is not `pass`.

## Bilibili-Specific Non-Teaching Content

Skip content that does not contribute to the actual lesson, such as:

- greetings
- small talk
- 一键三连、关注投币、评论区互动引导
- sponsorship
- routine closing pleasantries

Do not use danmaku as a teaching content source.

## Delivery Addendum

In addition to the shared delivery list:

- include the ASR-generated `SRT` or normalized timestamped segments if speech-to-text was used

## Asset

- `assets/notes-template.tex`: default LaTeX template to copy and fill
