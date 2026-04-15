# Note Quality Rubric

This rubric is used by the evaluator stage.

The evaluator must score each lecture note independently of the writer.
A fluent note can still fail.

## Required Fields

Every evaluator report must include:

- `pass`
- `target`
- `overall`
- `scores`
- `blocking_issues`
- `warnings`
- `repair_required`

## Scored Dimensions

All scores are normalized to `[0, 1]`.

### 1. Coverage completeness

Question:

- Did the note classify and address all required coverage units?

Passing threshold:

- `>= 0.90`

Fail examples:

- required units remain unclassified
- dense source material is silently skipped
- omissions are undocumented

### 2. Pedagogical depth

Question:

- Does the note teach, not merely summarize?

Passing threshold:

- `>= 0.80`

Fail examples:

- concepts are named but not motivated
- transitions are abrupt
- sections collapse into bullet-point recaps without explanation

### 3. Derivation fidelity

Question:

- When derivations or equations exist, are they expanded faithfully and explained symbol by symbol?

Passing threshold:

- `>= 0.80`

Applicability:

- only when derivation or formula units exist

Fail examples:

- dense formula slides reduced to one-sentence paraphrases
- symbols appear without explanation
- intermediate derivation steps are dropped

### 4. Code fidelity

Question:

- When code exists, does the note explain the logic rather than merely quote or mention it?

Passing threshold:

- `>= 0.80`

Applicability:

- only when code units exist

Fail examples:

- code shown with no role explanation
- key branches or update rules omitted
- code unit mapped only to a summary section

### 5. Figure usefulness

Question:

- Do figures genuinely improve teaching value, and are they provenance-backed?

Passing threshold:

- `>= 0.80`

Fail examples:

- decorative figures with no explanatory role
- figure captions that do not explain why the figure matters
- missing provenance for frame-derived figures

### 6. Coherence

Question:

- Are terminology, symbols, and section transitions internally consistent?

Passing threshold:

- `>= 0.85`

Fail examples:

- same concept renamed across sections
- earlier definitions contradicted later
- section ordering breaks the learning flow

### 7. Hallucination control

Question:

- Does the note distinguish supported facts from inference and avoid unsupported claims?

Passing threshold:

- `>= 0.90`

Fail examples:

- fabricated lecture content
- unsupported implementation details
- invented figures, definitions, or examples presented as source facts

## Blocking Issue Policy

Any of the following must create a blocking issue:

- required coverage remains unclassified
- evaluator detects unsupported factual claims
- figure provenance is missing for frame-derived figures
- formulas or code units are materially misrepresented
- note structure prevents delivery, such as missing major section summaries or missing final synthesis section

If `blocking_issues` is non-empty:

- `overall` must be `fail`
- `repair_required` must be `true`

## Overall Decision

`overall = pass` only when:

- every applicable threshold is met
- `blocking_issues` is empty

Otherwise:

- `overall = fail`

## Evaluator Tone

The evaluator should be concrete and corrective.
Do not use vague praise to soften real failures.
Each blocking issue should tell the repair writer exactly what to fix.
