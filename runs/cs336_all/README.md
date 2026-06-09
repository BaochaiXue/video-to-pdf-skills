# CS336 Spring 2026 Textbook Run

This run is the current harness-managed rebuild for `CS336: Language Modeling from Scratch`.

- term: `Spring 2026`
- official course page: <https://cs336.stanford.edu/>
- official public playlist: <https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV>
- public videos in deliverable: `18`
- final textbook PDF: `deliverable/cs336_complete_notes.pdf`
- final textbook TeX: `deliverable/cs336_complete_notes.tex`
- current revision: `slp3-style-textbook-v3`

## Source Policy

The generated chapters are grounded in the Spring 2026 public YouTube videos, downloaded English VTT subtitles, official lecture scripts/PDFs from `stanford-cs336/lectures`, and the official course page schedule. Each core section includes a `本节来源依据` table that maps textbook prose back to video windows and official PPT/PDF/Python-code source groups. Explanatory textbook expansions are marked as `延伸解释`.

## Textbook Style

The active deliverable follows the SLP3 / pasted-chapter style contract in `reference_style/slp3_style_contract.md`: motivating opening, terminology, source-grounded mechanisms, formulas with symbol explanations, pseudocode, worked examples, caveats, source alignment, chapter summary, and exercises.

## Known Gaps

- The course page lists a Daniel Selsam guest lecture on 2026-06-01, but the public playlist snapshot used here has no corresponding video.
- The Dan Fu guest lecture has a public video and subtitles, but no official slide/script link on the schedule.

## Rebuild

Run:

```bash
python3 build/rebuild_spring2026_slp3_style_textbook.py
python3 build/validate_youtube_note.py --compile
```
