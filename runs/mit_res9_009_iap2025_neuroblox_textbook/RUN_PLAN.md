# MIT RES.9-009 Neuroblox Textbook Run Plan

This run follows the harness-managed workflow defined in:

- `skills/references/video-render-pdf-common.md`
- `skills/references/video-note-harness.md`
- `skills/references/coverage-schema.md`
- `skills/references/evaluator-playbook.md`
- `skills/references/note-quality-rubric.md`
- `skills/references/figure-provenance.md`

## Build Goal

Generate a textbook-grade Chinese course book for:

- MIT `RES.9-009: Introduction to Computational Neuroscience with Neuroblox`
- `January IAP 2025`

The book must be:

- coverage-first
- source-grounded
- harness-managed
- evaluator-gated
- auditable from repo artifacts

## Canonical Source Families

- MIT OCW course home
- Neuroblox course home and all official unit pages
- MIT Video Productions `IAP 2025` YouTube playlist containing the full 11-video Neuroblox lecture set
- public YouTube videos embedded from the official Neuroblox course site
- official course-page code blocks, figures, references, and challenge sections

## Working Assumptions

- The official Neuroblox course site is the primary non-video evidence layer.
- Several important units do not expose an embedded video but still contain substantial official teaching content.
- No official slide PDF bundle has been found so far; official page text and images must serve as the non-video evidence layer.
- Missing or inaccessible items must be logged explicitly rather than silently skipped.

## Planned Chapter Mapping

1. `01_intro_to_neuroblox`
   Sources: course home, video `2dbAePEmbhQ`
2. `02_neuroblox_gui`
   Sources: course home, video `XlBRJps84zE`
3. `03_course_structure_and_julia`
   Sources: `getting_started`, `intro_julia`, video `ekwu47RHCHE`
4. `04_differential_equations_and_plotting`
   Sources: `intro_diffeq`, `intro_plot`, video `6EaolLVhnug`
5. `05_blox_and_connections`
   Sources: `blox_connections`, video `8XcN9j5njgg`
6. `06_neurons_neural_masses_and_sources`
   Sources: `neuron_mass`, video `Ptqv16fhOtg`
7. `07_circuit_models_overview`
   Sources: `circuits`, video `ih7IELQ5W50`
8. `08_corticostriatal_microassemblies`
   Sources: `CS_circuit`
9. `09_ping_network`
   Sources: `PING_circuit`
10. `10_decision_making_in_circuit_models`
   Sources: `decision_making`, video `ULAe2VvQgms`
11. `11_synaptic_plasticity_and_reinforcement_learning`
   Sources: `learning`, video `pBvgcIHK6GY`
12. `12_parameter_fitting_with_optimization`
   Sources: `optimization`
13. `13_spectral_dynamic_causal_modeling`
   Sources: `DCM`, video `OEeyks_HIMI`
14. `14_experimental_design`
   Sources: `experimental_design`, video `NU9K8l-gg-Q`

## Known Gaps To Track

- Slide PDFs: not yet found.
- Separate downloadable programming-assignment bundle: not yet found beyond course-page challenge/code materials.

These gaps must be recorded again in source manifests or omission logs if they remain unresolved.
