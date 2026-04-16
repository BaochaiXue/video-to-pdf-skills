# Cross-Lecture Consistency Report

## Scope

This report reviews the 12 lecture outputs under `lectures/` and focuses on four classes of cross-chapter issues:

- terminology drift
- symbol or concept boundary drift
- repeated material that should move to part-level exposition
- distinctions that are locally correct but not yet globally stabilized

The lecture set is strong locally. Most chapters already define their own terms carefully. The remaining work is book-level normalization: the textbook needs one canonical vocabulary layer above the lecture-level explanations.

## Priority 1: normalize the `inference-time computation / test-time scaling / search / planning` ladder

**Why this matters**

Right now the lectures are individually coherent but use overlapping terms at different abstraction levels. A reader going straight from L01 to L03/L06/L08/L10 can easily read `planning`, `search`, `test-time scaling`, `inference-time compute`, and `test-time RL` as near-synonyms, which they are not.

**Evidence**

- `lectures/lec01_inference_time_reasoning/lecture.tex:89-155` defines `推理时计算（inference-time computation）` as the deployment-time budget umbrella and treats width/depth/verifier/refinement as different spending patterns.
- `lectures/lec03_reasoning_memory_planning/lecture_repaired.tex:243-307` contrasts reactive execution, tree search, and model-based planning, and explicitly argues that planning is broader than tree search.
- `lectures/lec04_open_training_recipes_reasoning/lecture.tex:63-73` and `179-206` use `test-time scaling` as the deployment-stage budget-allocation phenomenon/recipe.
- `lectures/lec05_coding_agents_vulnerability_detection/lecture.tex:116-129` uses `test-time compute` to mean budget allocation across dynamic versus procedural control stages.
- `lectures/lec06_multimodal_autonomous_agents/lecture.tex:165-210` uses `Tree Search for Language Model Agents` as explicit inference-time search over environment states.
- `lectures/lec08_alphaproof_formal_mathematics/lecture.tex:88-96` introduces `test-time RL`, which is a stronger adaptation mechanism than plain search or sampling.
- `lectures/lec10_advanced_theorem_proving/lecture.tex:174-221` and `347-351` use theorem-proving examples to show that more test-time budget helps only when proof-environment feedback makes search structured.

**Problem**

The book currently lacks one canonical ladder that says:

| Level | Canonical meaning |
| --- | --- |
| inference-time computation | umbrella term for extra deployment-time budget |
| test-time scaling | the empirical/per-design phenomenon that performance changes with more such budget |
| search | explicit branching/exploration over candidate states, trajectories, or proofs |
| planning | choosing and organizing future actions or internal simulations; may use search, but is not reducible to search |
| test-time RL | online or local policy improvement/adaptation using feedback; not interchangeable with search |

**Required editorial action**

- Add this ladder once in the Part I introduction, then refer back to it instead of re-deriving it in later chapters.
- In L03, keep the current planning-vs-search distinction; make that chapter the canonical reference for `planning`.
- In L06, explicitly say `tree search is one inference-time compute mechanism for planning under environment feedback`.
- In L08, explicitly say `test-time RL` is a specialization/update mechanism, not merely wider search.
- In L10, use `proof search` as the domain-specific instance of the general `search` definition already established in Part I.

## Priority 2: stabilize the post-training taxonomy across L02 and L04

**Why this matters**

The textbook currently has two different entry points into post-training:

- L02 is judge-centric and reward-centric.
- L04 is recipe-centric and systems-centric.

Both are correct, but the book still needs one canonical hierarchy so readers do not infer false equivalences.

**Evidence**

- `lectures/lec02_learning_to_reason/lecture_repaired.tex:119-136` frames `SFT -> RLHF -> DPO` as the post-training scaffold for Weston’s reasoning-learning story.
- `lectures/lec02_learning_to_reason/lecture_repaired.tex:249-267` further specializes DPO into IRPO for reasoning-specific preference structure.
- `lectures/lec04_open_training_recipes_reasoning/lecture.tex:63-73` frames `pre-training -> post-training -> test-time inference`.
- `lectures/lec04_open_training_recipes_reasoning/lecture.tex:117-177` carefully separates DPO, PPO, RLHF structure, and RLVR.
- `lectures/lec04_open_training_recipes_reasoning/lecture.tex:153` correctly says better preference data often matters more than the `DPO vs PPO` label.

**Problem**

Without a single cross-chapter taxonomy, a reader may incorrectly conclude one of the following:

- `DPO` is the same thing as `RLHF`.
- `PPO` is the definition of `RLHF`.
- `IRPO` is a generic replacement for `DPO`, rather than a reasoning-specific construction over preference data.
- `RLVR` is just another name for preference tuning.

There is also one explicit gap: `GRPO` does not seem to be defined anywhere in the 12 lecture outputs. That is fine if it remains out of scope, but the final textbook should either omit it entirely or mark it as a non-covered neighboring method in an appendix. It should not suddenly appear in a part introduction without definition.

**Required editorial action**

- Add one part-level taxonomy box with this hierarchy:

| Layer | Canonical placement |
| --- | --- |
| SFT | supervised instruction/behavior shaping |
| RLHF | umbrella pipeline using preference supervision and policy optimization |
| DPO | direct preference objective, usually presented as a simpler alternative to reward-model-plus-RL pipelines |
| PPO | one policy-optimization mechanism often used inside RLHF/RL-style pipelines |
| IRPO | reasoning-specific preference construction/objective design on top of preference optimization |
| RLVR | verifier-backed RL for tasks with externally checkable outcomes |

- Keep L02 as the canonical `judge/evaluator` chapter.
- Keep L04 as the canonical `open recipe / systems pipeline` chapter.
- Add a brief editor note that `GRPO` is not a lecture-backed core topic in this course run unless you decide to add it in an appendix as outside-course context.

## Priority 3: separate `agent`, `tool use`, `workflow`, and `function calling`

**Why this matters**

The book uses `agent` productively, but at the moment the same word is stretched across:

- language agents
- dynamic coding agents
- theorem-proving agents
- discovery agents
- secure agentic systems

This is workable only if the book also pins down how `tool use`, `workflow`, and `function calling` relate to `agent`.

**Evidence**

- `lectures/lec03_reasoning_memory_planning/lecture_repaired.tex:101-135` defines the language-agent framing and treats internal reasoning as an internal action.
- `lectures/lec05_coding_agents_vulnerability_detection/lecture.tex:66-129` distinguishes dynamic tool-using agents from procedural pipelines and makes `control flow` a first-class design choice.
- `lectures/lec10_advanced_theorem_proving/lecture.tex:271-281` explicitly calls LeanHammer more agent-like because it combines retrieval, execution, feedback, and route switching.
- `lectures/lec11_abstraction_discovery_llm_agents/lecture.tex:161-205` argues COPRA is an agent because it maintains stateful search, history, external resources, and environment feedback.
- `lectures/lec12_safe_secure_agentic_ai/lecture.tex:160-190` and `367-444` make the tool-call boundary central to security.

**Problem**

`Function calling` is not visibly stabilized as a term. `Tool use` and `tool call boundary` are present, but the book still needs one explicit rule:

- a tool or function call is one action interface
- a workflow is control logic outside the model
- an agent is the closed-loop decision system operating over environment state, memory, tools, and feedback

Without this, some readers will flatten `function calling` into `agent`, and others will flatten `workflow` into `agent`.

**Required editorial action**

- Add a global glossary entry set:

| Term | Canonical use in the book |
| --- | --- |
| agent | a closed-loop decision system with state, actions, feedback, and budget |
| tool use | the act of invoking an external capability as one action in that loop |
| function calling | a structured subtype of tool use, typically API/schema-constrained |
| workflow | externally scripted control flow, possibly containing one or more model calls |
| agent workflow | a compound system mixing model policy, tools, memory, and controller logic |

- In L05, keep the strongest distinction between dynamic and procedural control.
- In L10/L11, avoid making every proof-environment interaction sound identical to generic consumer-LLM `function calling`; it is better described as environment-mediated action plus feedback.
- In L12, explicitly inherit the earlier tool-use definition so the security chapter does not feel like it introduces a different ontology.

## Priority 4: the formal stack is strong locally, but needs one canonical ladder across L08-L11

**Why this matters**

This is the best-defined part of the current lecture set, but it is also the part with the most repeated definitions. The local chapters are correct; the cross-book issue is duplication plus small framing shifts.

**Evidence**

- `lectures/lec08_alphaproof_formal_mathematics/lecture.tex:138-153` defines `formal specification / verification / theorem proving / proof search / autoformalization` and explicitly says they must not be collapsed.
- `lectures/lec08_alphaproof_formal_mathematics/lecture.tex:246-260` states that the competition protocol still used manual formalization before proof search.
- `lectures/lec09_autoformalization_theorem_proving/lecture.tex:134-146` and `267-380` gives the sharpest textbook decomposition of the stack and the best explanation of theorem-level versus proof-level autoformalization.
- `lectures/lec10_advanced_theorem_proving/lecture.tex:98-110` inherits that stack and shifts the focus to proof-search system design.
- `lectures/lec11_abstraction_discovery_llm_agents/lecture.tex:149-159` and `207-245` then broadens theorem proving into formal verification and discovery workflows.

**Problem**

The chapters are currently close enough that a final editor may be tempted to leave them alone, but the book still needs one ladder that says:

| Layer | Canonical role |
| --- | --- |
| informal reasoning | intuition, sketch, decomposition, explanation |
| formal specification | writing the object/problem/property precisely |
| autoformalization | automatically constructing the formal object from informal material |
| theorem proving | constructing a formal proof once the object is given |
| proof search | the algorithmic search procedure used inside theorem proving |
| verification | checking a proof or artifact against a formal specification |

The current risk is not contradiction. The current risk is reader drift:

- L08 can be misread as if AlphaProof solved the whole stack end-to-end, unless the manual-formalization protocol is foregrounded.
- L11 can be misread as if `formal verification` is simply a synonym for `theorem proving`, whereas the chapter itself correctly argues that verification additionally requires specification modeling and property coverage.

**Required editorial action**

- Make L09 the canonical concept chapter for the formal stack.
- In L08, keep the current environment framing but add an explicit forward pointer: `this chapter is about why formal mathematics is an attractive environment; L09-L10 unpack the representation and proof-search layers`.
- In L10, trim repeated base definitions and focus on `thought`, `sketch`, `retrieval`, `proof optimization`, and `context`.
- In L11, keep `formal verification` as the broader workflow concept and add one sentence explicitly distinguishing it from `theorem proving` in the narrow sense.

## Priority 5: stabilize the multimodal taxonomy across L06 and L07

**Why this matters**

L06 and L07 are individually very good, but they describe adjacent slices of the multimodal-agent landscape. The final textbook should make the hierarchy explicit once, then let each chapter specialize.

**Evidence**

- `lectures/lec06_multimodal_autonomous_agents/lecture.tex:101-126` distinguishes Mind2Web, WebArena, and VisualWebArena.
- `lectures/lec06_multimodal_autonomous_agents/lecture.tex:165-210` centers web-agent search over environment states.
- `lectures/lec06_multimodal_autonomous_agents/lecture.tex:253-319` expands from web agents to physical agents.
- `lectures/lec07_multimodal_agents_perception_to_action/lecture_repaired.tex:121-129` explicitly says the benchmark distinction must not be flattened.
- `lectures/lec07_multimodal_agents_perception_to_action/lecture_repaired.tex:183-228` treats GUI agents as a particular problem formulation around unified visual observation and action grounding.
- `lectures/lec07_multimodal_agents_perception_to_action/lecture_repaired.tex:230-268` treats long-video memory/grounding as another systems dimension, not a separate top-level agent class.

**Problem**

The current text still needs one stable taxonomy:

| Term | Canonical placement |
| --- | --- |
| multimodal agent | umbrella class |
| web agent | multimodal agent operating in web environments |
| computer-use / OS agent | broader than web; crosses apps and OS state |
| GUI agent | formulation centered on screen observation and grounded actions |
| physical / robotic agent | embodied branch with perception-action-control coupling |
| long-video grounding | a memory/observation subproblem, not a separate agent family |

**Required editorial action**

- Add this taxonomy table at the start of Part III.
- In L06, keep the emphasis on benchmark realism, search, and data pipelines for web/physical settings.
- In L07, keep the emphasis on OS/computer-use benchmarks, GUI grounding, action-call training, and long-horizon memory.
- Add one transition paragraph between L06 and L07 saying: `L06 defines the environment/search/data bottlenecks for web-centered multimodal agents; L07 widens that frame to computer-use, GUI, and long-video memory settings`.

## Priority 6: deduplicate the repeated `environment feedback / verifier / grounded workflow` thesis

**Why this matters**

This thesis is correct and should remain central. The issue is not factual inconsistency; the issue is repeated explanatory boilerplate that the final textbook should compress.

**Repeated pattern**

The same general lesson appears in:

- L01: external feedback is what makes longer reasoning useful.
- L05: evaluator and verifier shape coding/security agents.
- L08: formal mathematics is valuable because checker feedback is grounded.
- L09: proof assistants and LeanDojo provide structured environment feedback.
- L10: test-time compute matters when proof environments provide strong signals.
- L12: tool and system boundaries matter because model outputs get upgraded into actions.

**Required editorial action**

- Move the abstract thesis to part-level introductions:
  - Part I for reasoning/inference-time compute
  - Part II for workflow/tool/evaluator design
  - Part IV for formal environments and proof search
  - Part V for security boundaries
- Then trim chapter-local recaps that restate the same general principle in near-identical terms.
- Keep chapter-local versions only where the domain-specific consequence differs:
  - L05: verifier changes workflow design
  - L08/L10: verifier changes proof search
  - L12: verifier/policy layer changes safety guarantees

## Priority 7: keep `formal verification` in L12 aligned with the narrower formal stack from L09-L11

**Why this matters**

L12 is right to invoke formal verification, but it uses the term at the security-systems level rather than the proof-assistant level. That is fine, but the book should say so explicitly.

**Evidence**

- `lectures/lec12_safe_secure_agentic_ai/lecture.tex:95` says formal verification must enter agent pipeline design.
- `lectures/lec12_safe_secure_agentic_ai/lecture.tex:454-480` broadens the question to formal specifications for agentic systems, capability models, and information-flow policies.
- `lectures/lec11_abstraction_discovery_llm_agents/lecture.tex:207-245` uses `formal verification` in the compiler-correctness sense, where theorem proving is one mechanism inside a larger specification workflow.

**Problem**

A reader may mistakenly import the theorem-proving-specific meaning from Part IV into L12, or vice versa. The correct relation is:

- in Part IV, theorem proving is a core mechanism inside formal reasoning/verification workflows
- in L12, formal verification refers more broadly to proving system-level security properties under explicit capability and information-flow models

**Required editorial action**

- Add one editorial sentence near the first formal-verification mention in L12:
  `Here formal verification means proving security or policy properties of the agentic system design, not only proving mathematical theorems inside a proof assistant.`
- Add a backward pointer from L12 to the formal stack in Part IV so the reader sees continuity rather than a new overloaded term.

## Lower-priority cleanup opportunities

- `inner monologue` appears in L03 as internal action/state organization and in L07 as a bridge between high-level reasoning and GUI grounding. Keep both, but add one glossary note that `inner monologue` is a role, not a single fixed algorithm.
- `language agent` in L03 is a useful framing term, but later chapters mostly say `LLM agent`. Pick one default book-level label and treat the other as a specialized emphasis. My recommendation is:
  - use `LLM agent` as the book-wide default
  - reserve `language agent` for L03 when emphasizing language as the organizing medium
- The formal chapters reintroduce the same base definitions multiple times. That repetition is locally helpful, but the final book should move the base glossary to Part IV and shorten the repeated chapter openings.

## Suggested global insertion points

- Add one global taxonomy box in `Part I` for `inference-time compute / scaling / search / planning`.
- Add one post-training taxonomy box spanning `L02 + L04`.
- Add one `agent / workflow / tool use / function calling` glossary table before `L05`.
- Add one formal-stack ladder at the start of `Part IV`, then reduce repeated definitions in `L08-L11`.
- Add one multimodal benchmark-and-agent taxonomy table at the start of `Part III`.
- Add one explicit note in `L12` that the chapter’s use of `formal verification` is system-level and security-property-oriented.

## Bottom line

The lectures do **not** have major factual contradictions. The main remaining issue is that the book still reads like 12 strong chapters that each define their own local ontology. The final editor should preserve the local detail, but impose one global ontology above it.
