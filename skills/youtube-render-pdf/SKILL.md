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

## Read First

Before doing anything else, read:

- [../references/video-render-pdf-common.md](../references/video-render-pdf-common.md)

That shared reference contains the common coverage-first workflow, long-video two-pass strategy, writing rules, coverage artifacts, figure handling, validation, and delivery requirements.

When this file conflicts with the shared reference, follow this file for YouTube-specific behavior.

## Platform-Specific Goal

Produce a professional Chinese lecture note from a YouTube URL.

In addition to the shared workflow, this skill must:

- prefer YouTube's official subtitle tracks before ASR
- treat playlists, linked course pages, and description-linked lecture materials as strong `course mode` signals
- prefer linked official materials over ad hoc inference when a lecture page, slide deck, notebook, or repo exists

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

5. If no suitable subtitle track is available, use a device-aware ASR backend.
   - on `CUDA / NVIDIA`, default to `Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B`
   - on `Apple Silicon Mac`, default to a Whisper backend, prioritizing `whisper.cpp`
   - normalize the result into timestamped `SRT` or `segments`

6. Acquire the original cover image before writing the `.tex`.
   Prefer the highest-resolution official thumbnail exposed by metadata.

7. Prefer the best usable video source for figure extraction.
   Probe formats and choose the highest resolution that is actually downloadable in the current environment.

## Playlist and Course Handling

- If the user gives a playlist URL, preserve playlist order unless the user asks for a different order.
- When the playlist is a university course or lecture series, default to lecture-by-lecture processing rather than one giant prose pass.
- When official lecture materials exist per lecture, build per-lecture source inventories rather than a single playlist-level inventory.
- If some playlist entries are unavailable, private, or missing official materials, record that gap explicitly instead of silently skipping it.

## YouTube-Specific Non-Teaching Content

Skip content that does not contribute to the actual lesson, such as:

- greetings
- channel housekeeping
- sponsorship
- routine subscribe or like reminders
- closing pleasantries

Keep closing discussion when it carries actual teaching value.

## Asset

- `assets/notes-template.tex`: default LaTeX template to copy and fill
