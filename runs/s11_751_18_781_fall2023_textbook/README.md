# CMU S11-751/18-781 Fall 2023 Textbook Run

This run converts the public materials for `S11-751/18-781: Speech Recognition and Understanding (Fall 2023)` into a harness-managed Chinese textbook build.

## Grounding policy

- Primary sources: official WAVLab course page, public WAVLab YouTube recordings, and any official slides/readings that can be fetched.
- Supplemental sources: public official materials from `CS224S: Spoken Language Processing (Spring 2025)`.
- Missing or inaccessible official materials must be logged, not silently skipped.

## Directory map

- `build/`: bootstrap, validation, merge, and compile scripts
- `deliverable/`: user-facing final exports only; contains lecture-level deliverables and the merged textbook once they pass gates
- `meta/`: course schedule, source inventories, planning records, and catalog snapshots
- `lectures/`: one lecture/session workspace per official course session
- `raw/`: yt-dlp metadata, thumbnails, and subtitles
- `text/`: debug text bundles and course-page excerpts
- `materials/`: downloaded slide decks and other official artifacts
- `supplement/cs224s_spring2025/`: official supplemental source manifest and local copies
- `book/`: final merged textbook assets

## Delivery policy

- Each lecture is processed independently first.
- Only lectures that pass evaluator/validator gates should enter the final merged textbook.
- Blocked lectures remain in the repo with explicit omission/blocking records.
- When a lecture becomes deliverable, export its final `lecture_XX_note.tex` and `lecture_XX_note.pdf` into `deliverable/lectures/<lecture_slug>/`.
- When the merged textbook becomes deliverable, export the final merged `.tex` and `.pdf` into `deliverable/book/`.
- `deliverable/` should contain only final user-facing outputs, not intermediate harness artifacts.
- If a requested final deliverable has not been exported into `deliverable/`, the run is not complete yet.
- A lecture or merged book is considered finished only when its final deliverable files exist in `deliverable/` or an explicit blocking reason has been recorded and accepted.
