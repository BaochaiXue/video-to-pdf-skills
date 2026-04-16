# CS294/194-280 Spring 2025 Agents Textbook Build

This run converts the Berkeley RDI course `CS294/194-280: Advanced Large Language Model Agents` into a harness-managed Chinese textbook build.

Current state:

- course-level planning artifacts are complete and synchronized with the official syllabus page plus the official Berkeley RDI YouTube playlist
- all `12` lecture workspaces under `lectures/` have completed evaluator and validator gates
- the merged textbook under `book/` and the final handoff copy under `deliverable/` have both been generated

Primary source of truth:

- official course page: `https://rdi.berkeley.edu/adv-llm-agents/sp25`

Course-level sources discovered from the official page:

- 12 official lecture rows with recording URLs, slide URLs, and supplemental reading URLs
- 2 no-class schedule entries: `2025-02-17` and `2025-03-24`
- 2 special-time lecture entries: `2025-03-31` and `2025-04-21` at `10:00AM-12:00PM PT`
- 1 extra official asset for L01: an intro deck in addition to the main lecture slides

## Directory Map

- `AGENTS.md`: short map for all agents entering this run
- `COURSE_SPEC.md`: course scope, lecture slugs, schedule, and artifact contract
- `COURSE_SOURCE_MANIFEST.json`: machine-readable syllabus and source inventory
- `COURSE_COVERAGE_INDEX.jsonl`: seeded course-level coverage ledger
- `COURSE_OMISSION_LOG.jsonl`: explicit gaps and pending source-acquisition issues
- `docs/`: harness, rubric, style, glossary, evaluator, and failure-mode references
- `lectures/`: one workspace per lecture, using `lecXX_*` slugs from `COURSE_SPEC.md`
- `book/`: final textbook integration outputs
- `deliverable/`: final user-facing handoff copy of the merged textbook
- `eval_reports/`: book-level evaluator outputs
- `repair_logs/`: book-level repair logs
- `build/`: local scripts for lecture validation, manifest building, merge, and compilation

## Harness Flow

1. Course planner seeds the course-level record system.
2. Per lecture, agents run:
   `Source Curator -> Transcript & Slide Parser -> Coverage Planner -> Figure Agent -> Lecture Writer -> Reading Integrator -> Skeptical Evaluator -> Repair Writer`.
3. Only lectures that pass evaluator and validator may enter the textbook merge.
4. Book-level agents add frontmatter, consistency, glossary, exercises, final merge, and final evaluation.

## Agent Responsibilities

- Course planner owns only the top-level files listed above plus `docs/`.
- Lecture agents own only their assigned `lectures/lecXX_*` workspace.
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

- The official syllabus page exposes per-lecture recording URLs; the official Berkeley RDI YouTube channel additionally exposes a canonical playlist for this course: `https://www.youtube.com/playlist?list=PLS01nW3RtgorL3AW8REU9nGkzhvtn6Egn`.
- Readings are currently recorded as remote URLs only; local mirroring is deferred to lecture-level source acquisition.
- The existing local `build/` helpers currently enumerate lecture directories whose names start with digits. This run is specified to use `lecXX_*` workspace names, so lecture execution must add a selector shim or update the local helpers before bulk validation and merge.

## Unresolved Omissions

See `COURSE_OMISSION_LOG.jsonl` for the current course-level open items.

## Incremental Update Procedure

1. Re-open the official course page and compare against `COURSE_SOURCE_MANIFEST.json`.
2. Update `COURSE_SPEC.md` and the manifest if lectures, readings, or time notes changed.
3. Run or repair one lecture workspace at a time under `lectures/`.
4. Re-run lecture validation before allowing a chapter into `book/`.
5. Re-run course merge and final evaluation only after all required lectures pass.
