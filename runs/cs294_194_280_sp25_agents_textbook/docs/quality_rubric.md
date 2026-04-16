# Quality Rubric

This course uses a fail-closed rubric for both lecture-level and book-level delivery.

## Lecture-Level Passing Thresholds

- `coverage >= 0.95`
- `pedagogical_depth >= 0.85`
- `derivation_fidelity >= 0.80` when formulas or derivations exist
- `code_algorithm_fidelity >= 0.80` when code or algorithms exist
- `figure_usefulness >= 0.80`
- `reading_integration >= 0.80`
- `coherence >= 0.85`
- `hallucination_control >= 0.90`
- `readability >= 0.85`
- `overall` must be `pass`

## Lecture-Level Hard Fail Conditions

Any of the following forces `overall = fail`:

- a required coverage unit remains `planned` or `unclassified`
- a dense slide is collapsed into a one-line takeaway
- a displayed formula is missing symbol explanation
- code or algorithm appears without explaining role, inputs, outputs, loop logic, or failure points
- a figure lacks provenance
- a reading is name-dropped but not connected to the lecture argument
- unsupported claims are presented as source-backed facts
- LaTeX is not compilation-ready

## Book-Level Passing Thresholds

- `course_coverage >= 0.98`
- `textbook_coherence >= 0.85`
- `chapter_depth_avg >= 0.85`
- `hallucination_control >= 0.90`
- no missing required lecture
- no failed lecture validator
- final PDF must compile

## Scored Dimensions

Every lecture evaluator should score at least:

- coverage
- pedagogical_depth
- derivation_fidelity
- code_algorithm_fidelity
- figure_usefulness
- reading_integration
- coherence
- hallucination_control
- readability

Book-level evaluators should additionally score:

- course_coverage
- textbook_coherence
- chapter_depth_avg

## Blocking Issue Policy

If one blocking issue exists, the lecture fails.

Blocking issue categories should include:

- `coverage_gap`
- `thin_explanation`
- `formula_missing`
- `code_unexplained`
- `figure_missing`
- `hallucination_risk`
- `bad_structure`
- `compile_risk`
- `reading_integration_gap`
- `terminology_drift`

Every blocking issue must tell the repair writer what to change and where.
