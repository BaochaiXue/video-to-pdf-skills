# Final Sign-Off

Date: 2026-04-16

## Delivery Status

This run is complete at the deliverable level.

- `build/validate_lecture.py`: all 14 lecture workspaces pass
- `build/validate_textbook.py --require-book-pdf --require-deliverable`: pass
- merged all-in-one PDF exists under `deliverable/book/`
- lecture-level final `.tex` / `.pdf` exports exist under `deliverable/lectures/`

## Does The Textbook Follow The Course Videos?

Yes, with an explicit evidence distinction:

- `video-backed` chapters: `01`, `02`, `03`, `04`, `05`, `06`, `07`, `10`, `11`, `13`, `14`
- `page-backed` chapters because no official embedded video was available: `08`, `09`, `12`

For the video-backed chapters, the textbook was reconstructed from official YouTube subtitles plus official course-page materials.
For the page-backed chapters, the textbook was reconstructed from official course-page HTML, figures, code blocks, and linked readings.

Therefore the textbook is aligned with the course videos wherever official videos exist, and explicitly falls back to official non-video materials only where the course itself does not expose an embedded lecture video.

## Public YouTube List

The public playlist recorded for this run is:

- `MIT Video Productions / IAP 2025`
- `https://www.youtube.com/playlist?list=PLKHPCGvTwsmGEplF2c_Y8WEZ-Woqx-dlN`

It contains the 11 numbered `IntroCompNeuroNeuroblox` videos used by the run.

## Remaining Caveats

- Some units still lack a standalone official slide PDF, but those gaps are explicitly tracked in lecture omission logs.
- The all-in-one PDF uses a unified continuous `Page N` overlay at the top-right of every page to avoid ambiguity with lecture-local pagination preserved inside the merged chapter PDFs.
