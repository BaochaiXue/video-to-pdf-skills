# Course Spec

## Identity

- Course: `CS294/194-280: Advanced Large Language Model Agents`
- Term: `Spring 2025`
- Institution: `UC Berkeley / Berkeley RDI`
- Official course page: `https://rdi.berkeley.edu/adv-llm-agents/sp25`
- Default meeting time: `Mondays 4:00PM-6:00PM PT`
- Location: `Anthro/Art Building 160`

This file defines the lecture slugs and top-level execution contract for the textbook build.

## Textbook Goal

Produce a Chinese textbook-quality course note that is:

- coverage-first rather than summary-first
- grounded in official course page, recordings, slides, and supplemental readings
- lecture-by-lecture and evaluator-gated
- merged only after lecture validation passes

## Required Lecture Workspace Contract

Every lecture workspace under `lectures/` must use the exact slug listed below and eventually contain:

- `source_manifest.json`
- `transcript.jsonl`
- `slides.jsonl` when slides exist
- `segments.jsonl`
- `coverage_units.jsonl`
- `figure_manifest.json`
- `lecture.tex` or `lecture_XX_note.tex`
- `eval_report.json` or pass report under `eval_reports/`
- omission and repair logs when applicable

Per-lecture execution is harness-managed:

`Source Curator -> Transcript & Slide Parser -> Coverage Planner -> Figure / Visual Provenance Agent -> Lecture Writer -> Reading Integrator -> Skeptical Evaluator -> Repair Writer`

## Official Lecture List

| ID | Date | Lecture Slug | Title | Speaker | Time Note |
| --- | --- | --- | --- | --- | --- |
| L01 | 2025-01-27 | `lec01_inference_time_reasoning` | Inference-Time Techniques for LLM Reasoning | Xinyun Chen | default time |
| L02 | 2025-02-03 | `lec02_learning_to_reason` | Learning to reason with LLMs | Jason Weston | default time |
| L03 | 2025-02-10 | `lec03_reasoning_memory_planning` | On Reasoning, Memory, and Planning of Language Agents | Yu Su | default time |
| L04 | 2025-02-24 | `lec04_open_training_recipes_reasoning` | Open Training Recipes for Reasoning in Language Models | Hanna Hajishirzi | default time |
| L05 | 2025-03-03 | `lec05_coding_agents_vulnerability_detection` | Coding Agents and AI for Vulnerability Detection | Charles Sutton | default time |
| L06 | 2025-03-10 | `lec06_multimodal_autonomous_agents` | Multimodal Autonomous AI Agents | Ruslan Salakhutdinov | default time |
| L07 | 2025-03-17 | `lec07_multimodal_agents_perception_to_action` | Multimodal Agents – From Perception to Action | Caiming Xiong | default time |
| L08 | 2025-03-31 | `lec08_alphaproof_formal_mathematics` | AlphaProof: when reinforcement learning meets formal mathematics | Thomas Hubert | `10:00AM-12:00PM PT` |
| L09 | 2025-04-07 | `lec09_autoformalization_theorem_proving` | Language models for autoformalization and theorem proving | Kaiyu Yang | default time |
| L10 | 2025-04-14 | `lec10_advanced_theorem_proving` | Advanced topics in theorem proving | Sean Welleck | default time |
| L11 | 2025-04-21 | `lec11_abstraction_discovery_llm_agents` | Abstraction and Discovery with Large Language Model Agents | Swarat Chaudhuri | `10:00AM-12:00PM PT` |
| L12 | 2025-04-28 | `lec12_safe_secure_agentic_ai` | Towards building safe and secure agentic AI | Dawn Song | default time |

All URLs for recordings, slides, and readings are recorded in `COURSE_SOURCE_MANIFEST.json`.

## Non-Lecture Schedule Entries

| Date | Status | Note |
| --- | --- | --- |
| 2025-02-17 | no class | Presidents’ Day |
| 2025-03-24 | no class | Spring Recess |

These dates should not produce lecture workspaces, but they should remain visible in the final textbook appendix or schedule notes.

## Course-Specific Notes

- L01 has an extra official intro deck in addition to the main lecture slides.
- The official course page itself links lectures separately, but the official Berkeley RDI YouTube channel also exposes a canonical course playlist: `https://www.youtube.com/playlist?list=PLS01nW3RtgorL3AW8REU9nGkzhvtn6Egn`.
- Reading formats vary across arXiv abstracts, PDFs, blog posts, and YouTube talks. Lecture-level reading integrators must preserve provenance rather than flattening them into generic paper citations.
- The current local build helpers still assume `NN_*` lecture folder names. This course spec intentionally uses `lecXX_*`; build-phase agents must adapt helpers instead of mutating the lecture slug contract.
