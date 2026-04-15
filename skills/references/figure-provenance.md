# Figure Provenance

This file defines how lecture-note figures should be selected and documented.

## Principle

Figures are evidence-bearing teaching assets, not decoration.

Each delivered figure must answer:

- what concept it teaches
- why this figure was chosen over alternatives
- where it came from

## Preferred Sources

Prefer figures in this order:

1. official slide diagrams, plots, tables, and architecture figures
2. official PDF page crops when the full page is too dense
3. video frames when the lecture demonstrates something not preserved in the slides
4. generated diagrams only when no official figure expresses the concept clearly

## `figure_plan.json`

Before finalizing figures, create or update `figure_plan.json`.

Each entry should state:

- candidate assets considered
- the chosen asset
- why the chosen asset improves teaching value
- which source units require the figure

## `figure_manifest.json`

Each delivered figure must appear in `figure_manifest.json`.

Required fields:

- `figure_id`
- `source_id`
- `loc`
- `asset_path`
- `caption`
- `crop`
- `used_in_section`
- `time_provenance`

## Provenance Rules

### Slide-derived figures

Use:

- `source_id = "slide_or_external_asset"` or a more specific slide-based source id
- `loc.page` when the figure corresponds to a known slide page

### Frame-derived figures

Use:

- `source_id = "video_frame_or_crop"`
- concrete `time_provenance`

Time provenance must be a real interval, for example:

- `00:12:31--00:12:46`

## Caption Rules

Captions must explain the figure's role in the lecture.
Do not use file names or generic captions such as “Figure 1”.

## Validation Expectations

Validation should fail when:

- a figure in the `.tex` is missing from `figure_manifest.json`
- a frame-derived figure has no `time_provenance`
- captions are empty
