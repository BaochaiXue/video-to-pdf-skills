# Book Status

Current deliverable assets:
- `s294_277_complete_textbook.tex`
- `s294_277_complete_textbook.pdf`
- `course_manifest.json`
- main lectures: 18
- supplement appendices: 4

Harness status:
- evaluator summary: all lectures pass
- validator summary: run-level validator previously completed successfully for all lecture workspaces in the latest repair pass.

Official YouTube sources:
- channel: https://www.youtube.com/@robots-that-learn
- playlist: https://www.youtube.com/playlist?list=PLPaC96j0xdLcYLTSoSk9PO1Yg-1udJd-S

Lectures without a corresponding public YouTube video in the current official playlist:
- Lecture 1A: Introduction
- Lecture 1B: Biomechanics of Walking and Running
- Lecture 2B: The Human Hand and Dexterous Object Manipulation

Regeneration rule:
- if lecture sources or merged textbook content changes, rerun `build/build_course_manifest.py` and `build/merge_course_notes.py` so `deliverable/` stays in sync.
