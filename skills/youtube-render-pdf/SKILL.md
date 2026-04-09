---
name: youtube-render-pdf
description: >-
  Generate a professional, coverage-first, detailed, figure-rich LaTeX course
  note and final PDF from a YouTube lecture, tutorial, technical talk, or course
  playlist. Use when the user provides a YouTube URL and wants structured Chinese
  notes that jointly cover the video, subtitles, official slides or PDFs, lecture
  code, course pages, and other linked materials when available, prioritizing
  complete source coverage over concise summarization, including the original
  cover image on the front page, auditable coverage or figure manifests, and a
  rendered PDF. Prefer platform subtitles first, and when suitable subtitles are
  unavailable, fall back to the same device-aware ASR backend policy used by the
  Bilibili skill.
---

# YouTube Render PDF

Use this skill to turn a YouTube video into a complete, compileable `.tex` note and a rendered PDF.

## Goal

Produce a professional Chinese lecture note from a YouTube URL.

The output must:

- prioritize complete source coverage before elegant condensation
- use the video's actual teaching content rather than subtitle transcription alone
- when official course materials, slides, PDFs, notebooks, repos, or linked notes exist, jointly cover them rather than treating the video as the only source
- place the video's original cover image on the front page of the `.tex` and rendered PDF whenever available
- include all necessary high-value key frames as figures, without adding redundant screenshots
- end with a final synthesis section that includes the speaker's substantive closing discussion and your own distilled takeaways
- be structurally organized with `\section{...}` and `\subsection{...}`
- make omissions explicit rather than silent, and keep an omission log for substantive skipped material
- keep auditable source and coverage manifests for long videos and always in course mode
- be a complete `.tex` document from `\documentclass` to `\end{document}`
- be compiled successfully to PDF as part of the final delivery

## Pedagogical Standard

The notes must read like a strong human teacher is guiding the reader through the material.

- organize each major section so the reader first understands the motivation, then the main idea, then the mechanism, then the example or evidence, and finally the takeaway
- be patient and explicit about logical transitions; make it clear why the speaker introduces a concept, what problem it solves, and how the next idea follows
- aim for deep-but-accessible explanations: keep the technical depth, but introduce formalism only after giving intuition in plain language
- when a section is dense, break it into smaller subsections that progressively build understanding rather than compressing everything into one long derivation
- do not dump subtitle content in chronological order; rewrite it into a teaching sequence with clear intent, contrast, and buildup
- do not trade away source coverage for elegance; teaching quality matters, but not at the cost of silently dropping substantive material

## Coverage-First Standard

Coverage comes before compression.

- Reordering is allowed; silent omission is not.
- The goal is a reconstructed lecture note, not a polished summary.
- Treat the video, subtitles, official slides or PDFs, lecture code, linked repositories, and course pages as co-equal sources when they exist.
- Merge duplicate material across sources, but do not discard non-duplicate detail just because another source covers the same topic at a higher level.
- Keep an internal coverage ledger while working. Every substantive source unit must end up in exactly one of: final note section, exact duplicate collapse, or omission log.
- A substantive source unit can be a subtitle span, slide bullet, derivation step, code block, figure region, table, example, caveat, or transition that carries teaching content.
- Omit only content that is clearly non-teaching, exactly duplicated, or inaccessible after reasonable effort. Log any substantive omission.

## Source Acquisition

1. Build a source inventory before writing prose.
   Inspect title, chapters, duration, description, playlist context, linked resources, thumbnail availability, subtitle availability, and whether the video belongs to a course or lecture series.

2. Enter `course mode` automatically when the video is clearly part of a course, playlist, semester, university lecture series, or when the description links to official lecture materials.
   In `course mode`, collect and log every relevant source you can access, including: official course page, schedule page, per-lecture PDF or slide deck, lecture code, notebook, script, linked repo, preview or trace assets, reading notes, and the video itself.
   Do not treat the YouTube video as the complete source when official materials exist.

3. Acquire official non-video materials before prose writing whenever practical.
   Download or save slide decks, PDFs, linked repos, notebooks, and lecture scripts locally when available.
   When these materials are expected but unavailable, record the gap in the source manifest instead of silently ignoring it.

4. Acquire the video's original cover image before writing the `.tex`.
   Prefer the highest-resolution thumbnail exposed by the platform metadata.
   Save the selected cover locally and reference that local asset from the front page.
   Do not substitute a random video frame when an official cover image is available.

5. Prefer the best matching subtitle track.
   Use manual subtitles over auto-generated subtitles when both are available.
   Prefer the default language that best matches the video or the user's requested language.
   Fall back to the closest available subtitle track only when needed.
   Preserve the subtitle timestamps; do not flatten subtitles into plain text too early if figures still need to be located.

6. If no suitable subtitle track is available, use a device-aware ASR backend.
   On `CUDA / NVIDIA`, default to `Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B`.
   On `Apple Silicon Mac`, default to a Whisper backend, prioritizing `whisper.cpp`; MLX and `openai-whisper` are acceptable alternatives.
   Normalize the result into a timestamped `SRT` file or a `segments` structure with at least `start`, `end`, and `text`.
   If you use Qwen forced alignment, chunk longer audio into short windows first and then merge the aligned output back into one continuous transcript.

7. Prefer the best usable video source for figure extraction.
   Probe formats and choose the highest resolution that is actually downloadable in the current environment.

8. Keep all source artifacts local when practical.
   Typical working artifacts are metadata, the downloaded cover image, timestamped subtitles or ASR output, official slides or PDFs, code or notebooks, optional cleaned transcript text, a local video file, and extracted frames.

9. Create `source_manifest.json` for long videos and always in `course mode`.
   At minimum record `source_id`, `source_type`, `origin_url`, `local_path`, `required_for_coverage`, and `notes`.

## Long Video Strategy

For longer videos or multi-source lectures, use a two-pass coverage workflow rather than a single summarization pass.

- If the video is longer than 20 minutes, the subtitle file contains more than 300 subtitle entries, or `course mode` is active, split the work into smaller segments.
- Prefer chapter boundaries first. If chapters are unavailable or too uneven, split by coherent time windows, subtitle ranges, slide page ranges, or code regions.
- **Pass 1: extraction and coverage accounting.** For each segment, extract coverage units rather than polished prose. Coverage units should capture definitions, derivation steps, slide bullets, figure regions, examples, code logic blocks, caveats, and transitions.
- When subagents are available, each subagent must return a coverage ledger for its assigned slice: `coverage units`, required figures with provenance, unresolved ambiguities, and candidate omissions. Do not ask subagents for high-level summaries as the primary deliverable.
- Keep a small overlap between neighboring segments when explanations cross boundaries, then deduplicate during integration.
- **Pass 2: writing and integration.** Reconstruct a coherent teaching narrative from the coverage units. Reordering is allowed, but every required unit must map to a final section or to the omission log.
- Before delivery, verify that no substantive subtitle span, slide page, code block, or official source chunk remains unclassified.

## Teaching Content Rules

Build the notes from all of the following when available:

- official course page, schedule, slide decks or PDFs, notebooks, scripts, repos, and linked lecture materials
- video title and chapter structure
- the video's original cover image and key metadata
- on-screen diagrams, formulas, tables, plots, and architecture slides
- subtitle explanations, examples, and verbal emphasis
- code snippets shown or described in the talk

Skip content that does not contribute to the actual lesson:

- greetings
- small talk
- sponsorship
- channel logistics
- closing pleasantries

Keep the speaker's closing discussion when it carries actual teaching value, such as synthesis, limitations, future work, tradeoffs, advice, or open questions.

## Writing Rules

1. Write the notes in Chinese unless the user explicitly requests another language.

2. Coverage first, elegance second.
   Reconstruct the teaching flow when needed, but do not silently drop substantive content from subtitles, slides, code, or official course materials.

3. Organize the document with `\section{...}` and `\subsection{...}`.
   Each section should answer, in order when applicable: what problem is being solved, why simpler views are insufficient, what the core idea is, how it works, and what the reader should retain.

4. Maintain an internal coverage ledger while writing.
   Every substantive source unit must map to a final section, an exact-duplicate collapse, or `omission_log.jsonl`.

5. Start from `assets/notes-template.tex`.
   Fill in the metadata block, including the local cover image path, and replace the body content block with the generated notes.

6. The front page must include the video's original cover image when available.
   Place it on the first page rather than burying it later in the document.
   Keep it visually distinct from in-body teaching figures.

7. Use figures whenever they materially improve explanation.
   Include as many figures as are necessary for teaching clarity, even if that means many figures across the document.
   Do not optimize for a small figure count; optimize for explanatory coverage and readability.
   Good figures are key formulas, diagrams, tables, plots, visual comparisons, pipeline schedules, architecture views, and stage-by-stage visual progressions.

8. Do not place images inside custom message boxes.

9. When a derivation or dense slide appears, expand it layer by layer.
   Do not collapse a derivation into one sentence.
   Do not replace a dense slide or PDF page with three takeaways when the slide itself contains definitions, bullets, tables, caveats, or intermediate steps that materially matter.

10. When a mathematical formula appears:
   first explain in plain Chinese what the formula is trying to express and why it appears
   show it in display math using `$$...$$`
   then immediately follow with a flat list that explains every symbol

11. When code examples appear:
   explain the role of the code before the listing
   cover every logically distinct code block that materially contributes to the lecture, even if you do not quote every line verbatim
   summarize the expected behavior after it when useful
   wrap them in `lstlisting`
   include a descriptive `caption`

12. Highlight teaching signals deliberately and repeatedly when the content justifies it:
   use `importantbox` for core concepts the reader must walk away with, including formal definitions, central claims, key mechanism summaries, theorem-like statements, critical algorithm steps, and compact restatements of the main idea after a dense explanation
   use `knowledgebox` for background and side knowledge that improves understanding without being the main thread, including prerequisite reminders, historical lineage, engineering context, design tradeoffs, terminology comparisons, and intuition-building analogies
   use `warningbox` for common misunderstandings and failure points, including notation overload, hidden assumptions, misleading heuristics, easy-to-make implementation mistakes, causal confusions, off-by-one style reasoning errors, and places where the speaker contrasts a wrong intuition with the correct one
   there is no quota of one box per section; add multiple boxes in a section when the material contains multiple distinct teaching signals
   each box should carry a specific pedagogical payload rather than generic emphasis
   prefer placing a box immediately after the paragraph, derivation, or example that motivates it
   routine exposition should stay in normal prose; boxes are for high-signal takeaways, not decoration
   `importantbox`, `knowledgebox`, and `warningbox` may reinforce detailed explanation, but must not replace the detailed explanation itself
   figures must stay outside `importantbox`, `knowledgebox`, and `warningbox`

13. End every major section with `\subsection{本章小结}`.
   Add `\subsection{拓展阅读}` when there are one or two worthwhile external links.

14. End the document with a final top-level section such as `\section{总结与延伸}`.
   That final section must include:
   - the speaker's substantive closing discussion, excluding routine sign-off language
   - your own structured distillation of the core claims, mechanisms, and practical implications
   - your expanded synthesis, including conceptual compression, cross-links between sections, and any careful generalization that stays faithful to the video
   - concrete takeaways, open questions, or next steps when the material supports them

15. If official course materials and the video disagree, note the discrepancy instead of silently choosing one side.

16. Do not emit `[cite]`-style placeholders anywhere in the LaTeX.

## Coverage Artifacts

For long videos and always in `course mode`, create these sidecars when practical:

- `source_manifest.json`
  At minimum include `source_id`, `source_type`, `origin_url`, `local_path`, `required_for_coverage`, `status`, and `notes`.
- `coverage_units.jsonl`
  One record per substantive source unit. Recommended fields: `unit_id`, `source_id`, `loc`, `unit_type`, `summary`, `required`, `mapped_section`, and `status`.
- `omission_log.jsonl`
  Record any substantive omitted material. Recommended fields: `unit_id` or `source_id`, `reason`, `impact`, and `user_visible_note`.
- `figure_manifest.json`
  Record every kept figure. Recommended fields: `figure_id`, `source_id`, `loc`, `asset_path`, `caption`, `crop`, `used_in_section`, and `time_provenance` when applicable.

If you cannot create structured JSON or JSONL files in the current run, keep the same information in clearly labeled local notes, but prefer machine-readable files.

## Figure Handling

Select figures by necessity and teaching value, not by an arbitrary quota or a bias toward keeping the document visually sparse.

When locating candidate frames, bias strongly toward recall before precision.
It is better to inspect too many nearby candidates first than to miss the one frame where the slide, formula, table, or diagram is finally fully revealed and readable.

Frame understanding must come from direct visual inspection.

- Use the `view image` tool to inspect candidate frames and crops before deciding what they show, how they should be described, and whether they are complete enough to include.
- Do not use OCR tools such as `tesseract` as a substitute for visual understanding of a frame.
- Do not infer a frame's semantic content only from nearby subtitles, filenames, or timestamps without checking the image itself.
- Contact sheets, montages, and tiled strips are good for recall, but final keep-or-reject decisions and semantic naming must be based on actual image inspection with `view image`.

### Frame Selection Checklist

Before inserting any video frame, first inspect several nearby candidates from the same subtitle-aligned interval and apply this checklist. If any item fails, reject the frame and keep searching nearby rather than forcing an approximate match.

- Relevance: the frame must directly support the exact concept discussed in the surrounding paragraph or subsection, not just the same broad topic.
- Required content visible: every visual element referenced in the text must already be visible in the frame.
- Fully revealed state: when slides, whiteboards, animations, or dashboards build progressively, use the final fully populated readable state rather than an intermediate state.
- Best nearby candidate: compare multiple nearby frames and prefer the one that is both most complete and most readable.
- Readability: text, formulas, labels, and diagram structure must be legible enough to justify inclusion.

### Frame Naming

- Use neutral timestamp-based names for raw candidate frames. Do not assign semantic names before inspecting the actual frame content.
- Rename a frame semantically only after visually confirming what is fully visible in the image.
- The semantic filename must describe the frame's actual visible content, not a guess based on subtitles, nearby narration, or the intended paragraph topic.
- If the frame is partially revealed, transitional, or ambiguous, keep searching and do not lock in a semantic name yet.

- Use the timestamped subtitle file as the primary locator for key-frame search.
- First identify the subtitle span that corresponds to the concept, example, formula, or visual explanation being discussed.
- Then search within that subtitle-aligned time interval, and slightly around its boundaries when needed, to find the best readable frame.
- Do not jump directly from one guessed timestamp to one extracted frame.
  First generate a dense candidate set across the relevant interval, then inspect and down-select.
- Prefer tools that help you inspect many nearby candidates at once, such as `magick montage`, contact sheets, tiled frame strips, or equivalent workflows.
  Use them to maximize recall and avoid missing the frame where the visual content is fully present.
- When the visual is a progressive PPT reveal, animation build, whiteboard accumulation, or dashboard state change, explicitly search for the final fully populated state.
  Do not stop at the first frame that seems approximately correct.
- If several nearby candidates differ only by progressive reveal state, keep checking until you find the frame with the most complete readable information.
- When in doubt between a sparse early frame and a denser later frame from the same explanation window, prefer the later frame if it is materially more complete and still readable.
- Include every figure that is necessary to explain the content well.
- It is acceptable, and often desirable, to include several figures within one section or subsection when the video builds an idea in stages.
- Omit repetitive or low-information frames.
- Extract frames near chapter boundaries and explanation peaks when chapters exist, but still validate them against subtitle timing.
- Search nearby timestamps when the first extracted frame catches an animation transition.
- Crop, enlarge, or isolate the relevant region when the full frame is too loose.
- When a slide reveals content progressively, capture the final readable state and add intermediate frames only when they teach a genuinely different step.
- For dense visual sections, it is acceptable to over-sample first and discard later.
  Do not optimize candidate count so early that key visual states are never inspected.
- Prefer a sequence of necessary figures over one overloaded figure with unreadable labels.
- Preserve readability of formulas and labels.

## Figure Time Provenance

Whenever the `.tex` or PDF references a specific video frame, or a crop derived from a video frame, record its source time interval on the same page as a bottom footnote.

- The footnote must show the concrete time interval, for example `00:12:31--00:12:46`.
- The interval should come from the subtitle-aligned segment used to locate the figure, not from a vague chapter-level estimate.
- If the figure is a crop, the footnote still refers to the original video time interval of the source frame or subtitle span.
- If several nearby frames in one figure all come from the same subtitle interval, one clear footnote is enough.
- Keep the figure and its time footnote anchored to the same page; prefer layouts such as `[H]`, a non-floating block, or another stable placement when ordinary floats would separate them.

## Coverage Validation

Before delivery, verify all of the following:

- no substantive source remains unclassified
- no required coverage unit remains unmapped to a final section or `omission_log.jsonl`
- no dense subtitle interval, slide page, or official code block has been collapsed into a high-level takeaway without checking for definitions, steps, examples, or caveats
- no official slide or PDF page carrying unique teaching content has been silently skipped
- every figure is represented in `figure_manifest.json`, and every video frame figure has time provenance
- the final PDF compiles successfully

## Visualization

For concepts that remain hard to explain with only screenshots and prose, add accurate visualizations.

Two acceptable routes:

- generate LaTeX-native visualizations with TikZ or PGFPlots
- generate figures ahead of time with scripts and include them as images

For script-generated illustrations, prefer Python tools such as `matplotlib` and `seaborn` when they are the clearest way to produce an accurate teaching figure.

When a visualization is generated externally rather than drawn natively in LaTeX:

- export the figure as `pdf` so it can be inserted into the `.tex` without rasterization loss
- prefer vector output for plots, charts, and schematic illustrations
- avoid `png` or `jpg` for script-generated teaching figures unless the content is inherently raster

When the source material contains relationships, results, or equations that would be clearer when redrawn than when shown as a screenshot, prefer rebuilding them with LaTeX-native tools or with `matplotlib` / `seaborn`.

Use visualizations for:

- process flows, pipelines, and architecture overviews
- curves and charts such as scaling laws, training curves, benchmark results, and ablation comparisons
- distributions, correlations, heatmaps, and other plots that explain data relationships
- complex functions, surfaces, contour plots, and geometric intuition figures
- tables or comparisons that become clearer when redrawn as charts
- summary diagrams that compress a section's core mechanism or takeaway into one figure

Do not add decorative graphics that do not teach anything.

## Final Checklist

Before delivery, verify all of the following:

- no important teaching content has been dropped, and no concrete but critical detail has been lost during restructuring or summarization
- the text and figures are aligned: each inserted frame supports the surrounding explanation, necessary crops have been applied, and the chosen frame shows the fullest relevant information rather than a transitional or incomplete state
- the document is visually rich enough for teaching: check whether more high-information key frames should be added, and whether additional LaTeX-native or Python-script-generated illustrations would improve clarity
- the coverage ledger, omission log, and figure manifest are internally consistent

## Delivery

Deliver all of the following:

- the final `.tex` file
- the downloaded cover image referenced on the front page
- any extracted or generated figure assets referenced by the document
- `source_manifest.json`
- `coverage_units.jsonl`
- `omission_log.jsonl`
- `figure_manifest.json`
- the subtitle or ASR artifact used for coverage accounting when available
- the compiled PDF

## Asset

- `assets/notes-template.tex`: default LaTeX template to copy and fill
