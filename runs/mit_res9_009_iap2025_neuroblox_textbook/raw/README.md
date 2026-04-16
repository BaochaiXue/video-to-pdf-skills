# Raw Acquisition Log

This directory is the scratch area for source acquisition in the MIT RES.9-009 Neuroblox textbook run.

## Current State

- `raw/videos/01_2dbAePEmbhQ`
- `raw/videos/02_XlBRJps84zE`
- `raw/videos/03_ekwu47RHCHE`

Those three lecture packages already exist and contain metadata, thumbnails, and subtitles.

## Acquisition Rule

- Prefer metadata, thumbnails, and subtitles first.
- Do not download large MP4s until a later coverage pass actually needs frame extraction.
- Keep the lecture numbering from the public course site.
- Treat the two `Introduction to Julia` reference videos as supplementary material, not core lecture sources.
- Those two links are official page references rather than lesson iframes, but they are still public YouTube sources exposed by the course site.

## What Was Resolved

- 11 core lecture videos are present across the Neuroblox course pages.
- 2 additional public YouTube links are present on `Introduction to Julia`.
- Manual subtitles are available for the lecture videos.
- No manual subtitles were exposed for the two supplementary Julia reference videos.

## Next Acquisition Step

When a later pass needs more local assets, fetch per-video:

- `*.info.json`
- subtitle files
- thumbnail image

Only fetch video media after that, and only if a figure or frame-based task requires it.
