# CS336 Spring 2025 Textbook Run

This run upgrades the legacy `cs336_all` lecture-note pipeline into the stricter harness-managed textbook workflow.

- course: `CS336: Language Modeling from Scratch`
- term: `Spring 2025`
- official public playlist: <https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_>

## Canonical workflow

1. `build/bootstrap_course.py`
   Rehydrates legacy lecture workspaces into canonical harness inputs such as `subtitle.srt`, `transcript.jsonl`, `slides.jsonl`, seeded coverage, and normalized figure provenance.

2. `build/bootstrap_harness.py`
   Generates `lecture_plan.json`, `figure_plan.json`, `contracts/segment_##_contract.md`, and other harness artifacts.

3. Lecture workers repair coverage, omission, figure, and evaluator artifacts until delivery gates pass.

4. `build/build_course_manifest.py`
   Rebuilds course-level artifact pointers.

5. `build/compile_all_lecture_notes.py`
   Compiles notes through the shared delivery validator.

6. `build/merge_course_notes.py`
   Merges the lecture PDFs into a final textbook and copies the merged `.tex/.pdf` into `deliverable/`.

