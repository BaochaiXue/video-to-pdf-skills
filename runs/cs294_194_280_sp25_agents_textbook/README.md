# CS294/194-280 Spring 2025 Agents Textbook Build

This run converts the Berkeley RDI course `CS294/194-280: Advanced Large Language Model Agents` into a harness-managed Chinese textbook build.

Current state:

- course-level planning artifacts are complete and synchronized with the official syllabus page plus the official Berkeley RDI YouTube playlist
- all `12` lecture workspaces under `lectures/` have completed evaluator and validator gates
- `3` supplement workspaces under `supplements/` extend the main textbook with Berkeley Fall 2024, Berkeley Fall 2025, and Stanford CS329A official/public materials
- the merged textbook under `book/` and the final handoff copy under `deliverable/` have both been generated

Primary source of truth:

- official course page: `https://rdi.berkeley.edu/adv-llm-agents/sp25`

Course-level sources discovered from the official page:

- 12 official lecture rows with recording URLs, slide URLs, and supplemental reading URLs
- 2 no-class schedule entries: `2025-02-17` and `2025-03-24`
- 2 special-time lecture entries: `2025-03-31` and `2025-04-21` at `10:00AM-12:00PM PT`
- 1 extra official asset for L01: an intro deck in addition to the main lecture slides

Supplemental course sources folded into the final textbook:

- Berkeley Fall 2024 `CS294/194-196: Large Language Model Agents`
  - official course page: `https://rdi.berkeley.edu/llm-agents/f24`
  - public MOOC page: `https://llmagents-learning.org/f24`
  - official playlist: `https://www.youtube.com/playlist?list=PLS01nW3RtgopsNLeM936V4TNSsvvVglLc`
- Berkeley Fall 2025 `CS294/194-196: Agentic AI`
  - official course page: `https://rdi.berkeley.edu/agentic-ai/f25`
  - public MOOC page: `https://agenticai-learning.org/f25`
  - official playlist: `https://www.youtube.com/playlist?list=PLS01nW3RtgoqGkm4UeqNeZLccW-OGc1fJ`
  - official unlisted public recording for `2025-10-06`: `https://www.youtube.com/watch?v=VfOA2a0dj4w`
- Stanford `CS329A: Self-Improving AI Agents (Autumn 2025)`
  - official course page: `https://cs329a.stanford.edu/`
  - official schedule: `https://cs329a.stanford.edu/#schedule`
  - no official public video page or official slide index was found; the supplement remains schedule-and-readings grounded

## Source Grounding Status

This run is not uniformly “video-only”. Its grounding policy is:

- Spring 2025 Berkeley main textbook:
  - yes, this part is built from the official course videos plus official slides and official readings
  - all `12` official Spring 2025 lectures are video-grounded
  - the official playlist used is `https://www.youtube.com/playlist?list=PLS01nW3RtgorL3AW8REU9nGkzhvtn6Egn`
- Berkeley Fall 2024 supplement:
  - mostly yes, this supplement is grounded in the official/public edited videos, official slides, and official readings
  - it is not grounded in the official `bCourses` original recordings, because those are official but not public
  - the official playlist also contains `2` private entries that are logged as omissions rather than treated as covered
- Berkeley Fall 2025 supplement:
  - mostly yes, this supplement is grounded in the official/public recordings, official slides when public, and official readings
  - `Sep 8` is slides-only
  - `Oct 6` is official and public but unlisted, so it is ingested directly rather than via playlist expansion
- Stanford CS329A supplement:
  - no, this supplement is not a video-grounded reconstruction
  - it is grounded in the official Stanford course page, official schedule, and official reading links
  - an instructor-affiliated public playlist exists, but it is not treated as canonical source-of-truth because the official course site does not publish it as the official video entry

Short answer to “is our textbook based on the course videos?”:

- `yes` for the Spring 2025 Berkeley main book
- `yes, with explicit public-source gaps` for the Berkeley 2024 and Berkeley 2025 supplements
- `no, schedule/readings grounded only` for the Stanford CS329A supplement

## Directory Map

- `AGENTS.md`: short map for all agents entering this run
- `COURSE_SPEC.md`: course scope, lecture slugs, schedule, and artifact contract
- `COURSE_SOURCE_MANIFEST.json`: machine-readable syllabus and source inventory
- `COURSE_COVERAGE_INDEX.jsonl`: seeded course-level coverage ledger
- `COURSE_OMISSION_LOG.jsonl`: explicit gaps and pending source-acquisition issues
- `docs/`: harness, rubric, style, glossary, evaluator, and failure-mode references
- `lectures/`: one workspace per lecture, using `lecXX_*` slugs from `COURSE_SPEC.md`
- `supplements/`: one workspace per supplemental course extension chapter
- `book/`: final textbook integration outputs
- `deliverable/`: final user-facing handoff copy of the merged textbook
- `eval_reports/`: book-level evaluator outputs
- `repair_logs/`: book-level repair logs
- `build/`: local scripts for lecture validation, manifest building, merge, and compilation

## Harness Flow

1. Course planner seeds the course-level record system.
2. Per lecture, agents run:
   `Source Curator -> Transcript & Slide Parser -> Coverage Planner -> Figure Agent -> Lecture Writer -> Reading Integrator -> Skeptical Evaluator -> Repair Writer`.
3. Supplemental-course agents build course-level extension chapters with explicit source manifests, coverage indexes, omission logs, and supplement evaluators.
4. Only lectures or supplements that pass their gates may enter the textbook merge.
5. Book-level agents add frontmatter, consistency, glossary, exercises, final merge, and final evaluation.

## Agent Responsibilities

- Course planner owns only the top-level files listed above plus `docs/`.
- Lecture agents own only their assigned `lectures/lecXX_*` workspace.
- Supplement agents own only their assigned `supplements/<course_slug>/` workspace.
- Book-level agents own only `book/`, `eval_reports/`, and `repair_logs/`.
- No agent should overwrite another lecture workspace without an explicit repair handoff.

## Validation And Build

Lecture-level validation, once lecture workspaces exist:

```bash
python3 build/validate_lecture.py --compile lec01_inference_time_reasoning
```

Course manifest regeneration:

```bash
python3 build/build_course_manifest.py
```

Lecture-note compilation:

```bash
python3 build/compile_all_lecture_notes.py
```

Course merge:

```bash
python3 build/merge_course_notes.py
```

Final handoff export:

```bash
./build/compile.sh
```

The final merged `.tex` and `.pdf` are copied to:

- `deliverable/book/cs294_194_280_sp25_agents_textbook_complete_notes.tex`
- `deliverable/book/cs294_194_280_sp25_agents_textbook_complete_notes.pdf`

## Known Limitations

- Readings are recorded primarily as remote URLs; local mirroring is deferred to lecture-level or supplement-level source acquisition.
- Berkeley Fall 2024 has two official but private playlist entries and official bCourses-only original recordings; those gaps are logged and not silently filled.
- Berkeley Fall 2025 includes a slides-only introduction and one official but unlisted public recording.
- Stanford CS329A official pages did not expose an official public video page or slide index; its supplement is therefore schedule-and-readings grounded rather than media-grounded.

## Unresolved Omissions

See `COURSE_OMISSION_LOG.jsonl` plus each supplement workspace's `COURSE_OMISSION_LOG.jsonl` for the explicit source gaps.

## Final Delivery Verification

As of `2026-04-16`, the user-facing deliverable is confirmed to be the latest merged result for this run.

- textbook validator:
  - `python3 build/validate_textbook.py --require-book-pdf`
  - result: `ok textbook`
- deliverable sync:
  - `book/main.tex == deliverable/book/cs294_194_280_sp25_agents_textbook_complete_notes.tex`
  - `book/textbook.pdf == deliverable/book/cs294_194_280_sp25_agents_textbook_complete_notes.pdf`
- SHA-256:
  - PDF: `1fdd4e7f59b3525892670add4aec484c5bc2d3ab04d9571d82adc02052d41f1e`
  - TeX: `755686be7993b8e020030b03be073ac455e16ba8f151a74c15e0a47a32717582`
- final deliverable PDF stats:
  - pages: `258`
  - size: `34,999,561` bytes
  - creation date: `Thu Apr 16 10:11:21 2026 EDT`

## Incremental Update Procedure

1. Re-open the official course page and compare against `COURSE_SOURCE_MANIFEST.json`.
2. Update `COURSE_SPEC.md` and the manifest if lectures, readings, or time notes changed.
3. Run or repair one lecture workspace at a time under `lectures/`.
4. Re-run lecture validation before allowing a chapter into `book/`.
5. Re-run course merge and final evaluation only after all required lectures and supplemental extension chapters pass.
