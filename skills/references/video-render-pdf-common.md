# Shared Video Render PDF Workflow

Read this file immediately after `youtube-render-pdf` or `bilibili-render-pdf` triggers.

This is the shared coverage-first workflow for both platforms. Platform-specific acquisition rules in the skill-local `SKILL.md` take precedence when they conflict with this file.

## Goal

Produce a professional Chinese lecture note and final PDF from a lecture, tutorial, technical talk, or course session.

The output must:

- prioritize complete source coverage before elegant condensation
- jointly use the video, subtitles, official slides or PDFs, code, repos, and course materials when available
- reconstruct the lecture as a teachable note rather than a high-level summary
- place the original cover image on the front page when available
- include necessary figures, formulas, code, and tables
- keep omissions explicit rather than silent
- be a complete `.tex` document and a compiled PDF

## Pedagogical Standard

The note should read like a strong human teacher is guiding the reader through the material.

- explain motivation before mechanism
- make transitions explicit
- introduce intuition before formalism when possible
- break dense material into smaller subsections instead of compressing it into one summary paragraph
- do not trade away source coverage for elegance

## Coverage-First Standard

Coverage comes before compression.

- Reordering is allowed; silent omission is not.
- The goal is a reconstructed lecture note, not a polished summary.
- Treat all accessible official materials as co-equal sources with the video.
- Merge exact duplicates across sources, but do not discard non-duplicate detail just because another source states the same idea more tersely.
- Keep an internal coverage ledger while working.
- Every substantive source unit must end up in exactly one of:
  - final note section
  - exact duplicate collapse
  - omission log

A substantive source unit can be:

- a subtitle span
- a slide bullet
- a derivation step
- a code block or code logic unit
- a figure region
- a table
- an example
- a caveat or limitation
- a transition that carries teaching meaning

Only omit material that is clearly:

- non-teaching
- exactly duplicated
- inaccessible after reasonable effort

Log every substantive omission.

## Common Source Workflow

1. Build a source inventory before writing prose.
   Inspect title, duration, chapters, description, playlist or series context, linked resources, cover availability, subtitle availability, and any official course linkage.

2. Enter `course mode` automatically when the content is clearly part of a course, semester lecture sequence, playlist, or linked official lecture series.
   In `course mode`, collect and log every relevant source you can access, including:
   - official course page
   - schedule page
   - per-lecture slide deck or PDF
   - notebook or script
   - lecture code
   - linked repo
   - preview or trace assets
   - reading notes or linked handouts

3. Acquire official non-video materials before prose writing whenever practical.
   Download or save them locally.
   If they are expected but unavailable, record the gap in `source_manifest.json`.

4. Acquire the original cover image before writing the `.tex`.
   Prefer the highest-resolution official thumbnail exposed by metadata.

5. Keep all source artifacts local when practical.
   Typical working artifacts include metadata, cover image, subtitles or ASR output, official materials, local video, extracted frames, and any generated figures.

6. For long videos and always in `course mode`, create `source_manifest.json`.
   At minimum record:
   - `source_id`
   - `source_type`
   - `origin_url`
   - `local_path`
   - `required_for_coverage`
   - `status`
   - `notes`

## Long Video Strategy

For longer videos or multi-source lectures, use a two-pass coverage workflow rather than a single summarization pass.

- If the video is longer than 20 minutes, the subtitle file contains more than 300 entries, or `course mode` is active, split the work into segments.
- Prefer chapter boundaries first. If those are unavailable or too uneven, split by coherent time windows, subtitle ranges, slide page ranges, or code regions.

### Pass 1: Extraction and Coverage Accounting

For each segment, extract coverage units rather than polished prose.

Coverage units should capture:

- definitions
- derivation steps
- slide bullets
- figure regions
- examples
- code logic blocks
- caveats
- transitions

When subagents are available, their primary deliverable must be a coverage ledger, not a high-level summary.

Each segment result should include:

- coverage units
- required figures with provenance
- unresolved ambiguities
- candidate omissions

### Pass 2: Writing and Integration

- Reconstruct a coherent teaching narrative from the coverage units.
- Reordering is allowed, but every required unit must map to a final section or `omission_log.jsonl`.
- Before delivery, verify that no substantive subtitle span, slide page, official source chunk, or code block remains unclassified.

## Teaching Content Rules

Build the note from all of the following when available:

- official course page and lecture materials
- video title and chapter structure
- cover image and metadata
- diagrams, formulas, tables, plots, and architecture slides
- subtitle explanations, examples, and verbal emphasis
- code shown or described in the lecture

Skip only content that does not contribute to the lesson:

- greetings
- small talk
- sponsorship
- routine channel logistics
- routine sign-off language

Keep substantive closing remarks when they contain synthesis, limitations, tradeoffs, future work, advice, or open questions.

## Writing Rules

1. Write in Chinese unless the user explicitly requests another language.

2. Organize the note with `\section{...}` and `\subsection{...}`.
   Reconstruct the teaching flow when needed, but do not silently drop substantive source content.

3. Maintain an internal coverage ledger while writing.
   Every substantive source unit must map to a final section, duplicate collapse, or `omission_log.jsonl`.

4. Start from `assets/notes-template.tex`.
   Fill in metadata, cover path, and replace the body with the generated note.

5. Use figures whenever they materially improve explanation.
   Optimize for explanatory coverage and readability, not low figure count.

6. Do not place images inside custom message boxes.

7. When a derivation or dense slide appears, expand it layer by layer.
   Do not collapse a derivation into one sentence.
   Do not replace a dense slide or PDF page with a few takeaways when it contains unique definitions, steps, examples, tables, or caveats.

8. When a mathematical formula appears:
   - explain in plain Chinese what it expresses and why it appears
   - show it in display math
   - immediately explain every symbol in a flat list

9. When code appears:
   - explain the role of the code before the listing
   - cover every logically distinct code block that materially contributes to the lecture, even if you do not quote every line verbatim
   - summarize the behavior after the listing when useful
   - wrap code in `lstlisting`
   - include a descriptive `caption`

10. Use `importantbox`, `knowledgebox`, and `warningbox` deliberately.
    They may reinforce detailed explanation, but must not replace detailed explanation.

11. End every major section with `\subsection{本章小结}`.
    Add `\subsection{拓展阅读}` when warranted.

12. End the document with `\section{总结与延伸}` or an equivalent final section.
    Include:
    - substantive closing discussion from the source
    - your own structured distillation
    - expanded synthesis across sections
    - concrete takeaways or next steps when supported

13. If official materials and the video disagree, note the discrepancy instead of silently choosing one side.

14. Do not emit `[cite]` placeholders in the LaTeX.

## Coverage Artifacts

For long videos and always in `course mode`, create these sidecars when practical:

- `source_manifest.json`
- `coverage_units.jsonl`
- `omission_log.jsonl`
- `figure_manifest.json`

Recommended fields:

- `coverage_units.jsonl`
  - `unit_id`
  - `source_id`
  - `loc`
  - `unit_type`
  - `summary`
  - `required`
  - `mapped_section`
  - `status`

- `omission_log.jsonl`
  - `unit_id` or `source_id`
  - `reason`
  - `impact`
  - `user_visible_note`

- `figure_manifest.json`
  - `figure_id`
  - `source_id`
  - `loc`
  - `asset_path`
  - `caption`
  - `crop`
  - `used_in_section`
  - `time_provenance`

If structured JSON or JSONL sidecars are impractical in the current run, keep the same information in clearly labeled local notes, but prefer machine-readable files.

## Figure Handling

Select figures by necessity and teaching value, not by arbitrary quotas.

- bias strongly toward recall before precision when searching candidate frames
- use direct visual inspection for final keep or reject decisions
- do not use OCR as a substitute for visual understanding
- do not infer image meaning only from nearby subtitles or filenames
- when slides or animations reveal progressively, search for the final fully revealed readable state
- prefer several necessary figures over one overloaded unreadable figure

### Figure Time Provenance

Whenever the note references a specific video frame, or a crop derived from a video frame, record its source time interval on the same page as a bottom footnote.

- use a concrete interval such as `00:12:31--00:12:46`
- derive the interval from subtitle-aligned search, not a vague chapter estimate
- keep the figure and provenance footnote on the same page

## Visualization

For concepts that remain hard to explain with screenshots and prose alone, add accurate visualizations.

Preferred routes:

- LaTeX-native diagrams with TikZ or PGFPlots
- externally generated teaching figures, preferably vector output such as PDF

Use visualizations for:

- process flows
- architecture overviews
- curves and charts
- distributions and correlations
- geometric or algorithmic intuition
- summary diagrams that compress a mechanism

Do not add decorative graphics.

## Coverage Validation

Before delivery, verify all of the following:

- no substantive source remains unclassified
- no required coverage unit remains unmapped to a final section or `omission_log.jsonl`
- no dense subtitle interval, slide page, or official code block has been collapsed into a high-level takeaway without checking for unique detail
- no official slide or PDF page carrying unique teaching content has been silently skipped
- every figure appears in `figure_manifest.json`
- every video-frame figure has time provenance
- the final PDF compiles successfully

## Final Checklist

Before delivery, verify:

- no important teaching content has been dropped
- the text and figures are aligned
- the document is visually rich enough for teaching
- the coverage ledger, omission log, and figure manifest are internally consistent

## Delivery

Deliver all of the following:

- the final `.tex` file
- the downloaded cover image
- any extracted or generated figure assets
- `source_manifest.json`
- `coverage_units.jsonl`
- `omission_log.jsonl`
- `figure_manifest.json`
- the subtitle or ASR artifact used for coverage accounting when available
- the compiled PDF

## Asset

- `assets/notes-template.tex`: default LaTeX template to copy and fill
