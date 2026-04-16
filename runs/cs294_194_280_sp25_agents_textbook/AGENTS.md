# Agents Map

Read these in order before touching this run:

1. `COURSE_SPEC.md`
2. `COURSE_SOURCE_MANIFEST.json`
3. `docs/harness_design.md`
4. `docs/quality_rubric.md`
5. `docs/writing_style_guide.md`
6. `docs/notation_and_glossary.md`
7. `docs/evaluator_playbook.md`
8. `docs/known_failure_modes.md`

Agent boundaries:

- Top-level course planner owns only course-level files in the run root and `docs/`.
- Lecture agents own only their assigned `lectures/lecXX_*` directory.
- Book-level agents own only `book/`, `eval_reports/`, and `repair_logs/`.

Operating rules:

- Treat the workspace as the record system. Write structured artifacts before prose.
- Do not skip the coverage ledger, evaluator, or validator.
- Keep omissions explicit in JSONL logs rather than hidden in prose.
- Use the `lecXX_*` lecture slugs defined in `COURSE_SPEC.md`.
- If a local helper assumes `NN_*` lecture directories, record the mismatch and patch the helper in the build phase instead of renaming the course spec ad hoc.
