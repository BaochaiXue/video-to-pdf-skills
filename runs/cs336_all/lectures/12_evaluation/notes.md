**Evaluation**: given a **fixed model**, how "**good**" is it?

## Benchmark scores

<img src="../../materials/spring2025-lectures/images/deepseek-r1-benchmarks.png" width="800">

[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/pdf/2501.12948.pdf)

<img src="../../materials/spring2025-lectures/images/llama4-benchmarks.png" width="800">

[Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)

<img src="../../materials/spring2025-lectures/var/files/image-514a4fe0690116810f840ff30c6cdf3f-https_www_datocms-assets_com_64837_1741887109-instruct-1_png" width="800">

[OLMo 2 (32B)](https://allenai.org/blog/olmo2-32B)

Recent language models are evaluated on similar, but not entirely identical, benchmarks (MMLU, MATH, etc.).

What are these benchmarks?

What do these numbers mean?

<img src="../../materials/spring2025-lectures/images/helm-capabilities-leaderboard.png" width="1000">

[[HELM capabilities]](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard)

Pay close attention to the costs!

<img src="../../materials/spring2025-lectures/images/artificial-analysis.png" width="800">

[[Artificial Analysis]](https://artificialanalysis.ai/)

Maybe a model is good if people choose to use it (and pay for it)...

<img src="../../materials/spring2025-lectures/images/openrouter.png" width="600">

[[OpenRouter]](https://openrouter.ai/rankings)

<img src="../../materials/spring2025-lectures/images/chatbot-arena-leaderboard.png" width="800">

[[Chatbot Arena]](https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard)

## Vibes

[[X]](https://x.com/demishassabis/status/1919779362980692364)

<img src="../../materials/spring2025-lectures/images/demis-gemini-2.5.png" width="500">

A crisis...

<img src="../../materials/spring2025-lectures/images/karpathy-crisis.png" width="600">

You might think evaluation is a mechanical process (take existing model, throw prompts at it, average some numbers)...

Actually, evaluation is a profound and rich topic...

...and it determines the future of language models.

What's the point of evaluation?

There is no one true evaluation; it depends on what question you're trying to answer.

1. User or company wants to make a purchase decision (model A or model B) for their use case (e.g., customer service chatbots).

2. Researchers want to measure the raw capabilities of a model (e.g., intelligence).

3. We want to understand the benefits + harms of a model (for business and policy reasons).

4. Model developers want to get feedback to improve the model.

In each case, there is an abstract **goal** that needs to be translated into a concrete evaluation.

Framework

1. What are the **inputs**?

2. How do **call** the language model?

3. How do you evaluate the **outputs**?

4. How to **interpret** the results?

What are the inputs?

1. What use cases are **covered**?

2. Do we have representation of **difficult** inputs in the tail?

3. Are the inputs **adapted** to the model (e.g., multi-turn)?

How do you call the language model?

1. How do you prompt the language model?

2. Does the language model use chain-of-thought, tools, RAG, etc.?

3. Are we evaluating the language model or an agentic system (model developer wants former, user wants latter)?

How do you evaluate the outputs?

1. Are the reference outputs used for evaluation error-free?

2. What metrics do you use (e.g., pass@k)?

3. How do you factor in cost (e.g., inference + training)?

4. How do you factor in asymmetric errors (e.g., hallucinations in a medical setting)?

5. How do you handle open-ended generation (no ground truth)?

How do you inteprret the metrics?

1. How do you interpret a number (e.g., 91%) - is it ready for deployment?

2. How do we assess generalization in the face of train-test overlap?

3. Are we evaluating the final model or the method?

Summary: lots of questions to think through when doing evaluation

Recall: that a language model is a probability distribution **p(x)** over sequences of tokens.

Perplexity (1/p(D))^(1/|D|) measures whether p assigns high probability to some dataset D.

In pre-training, you minimize perplexity on the training set.

The obvious thing is to measure perplexity on the test set.

Standard datasets: Penn Treebank (WSJ), WikiText-103 (Wikipedia), One Billion Word Benchmark (from machine translation WMT11 - EuroParl, UN, news)

Papers trained on a dataset (training split) and evaluated on the same dataset (test split)

Pure CNNs+LSTMs on the One Billion Word Benchmark (perplexity 51.3 -> 30.0) [Exploring the Limits of Language Modeling](https://arxiv.org/abs/1602.02410)

GPT-2 trained on WebText (40GB text, websites linked from Reddit), zero-shot on standard datasets

This is out-of-distribution evaluation (but idea is that training covers a lot)

<img src="../../materials/spring2025-lectures/images/gpt2-perplexity.png" width="800">

Works better on small datasets (transfer is helpful), but not larger datasets (1BW)

Since GPT-2 and GPT-3, language modeling papers have shifted more towards downstream task accuracy.

But reasons why perplexity is still useful:

- Smoother than downstream task accuracy (for fitting scaling laws)

- Is universal (why we use it for training) whereas task accuracy might miss some nuances

- Note: can measure conditional perplexity on downstream task too (used for scaling laws) [Establishing Task Scaling Laws via Compute-Efficient Model Ladders](https://arxiv.org/abs/2412.04403)

Warning (if you're running a leaderboard): evaluator needs to trust the language model

For task accuracy, can just take output generated from a blackbox model and compute the desired metrics

For perplexity, need LM to generate probabilities and trust that they sum to 1 (even worse with UNKs back in the day)

The perplexity maximalist view:

- Your true distribution is t, model is p

- Best possible perplexity is H(t) obtained iff p = t

- If have t, then solve all the tasks

- So by pushing down on perplexity, will eventually reach AGI

- Caveat: this might not be the most efficient way to get there (pushing down on parts of the distribution that don't matter)

Things that are spiritually perplexity:

Similar idea: cloze tasks like LAMBADA [The LAMBADA dataset: Word prediction requiring a broad discourse context](https://arxiv.org/abs/1606.06031)

<img src="../../materials/spring2025-lectures/images/lambada.png" width="800">

HellaSwag [HellaSwag: Can a Machine Really Finish Your Sentence?](https://arxiv.org/pdf/1905.07830)

<img src="../../materials/spring2025-lectures/images/hellaswag.png" width="600">

### Massive Multitask Language Understanding (MMLU)

[Measuring Massive Multitask Language Understanding](https://arxiv.org/pdf/2009.03300.pdf)

- 57 subjects (e.g., math, US history, law, morality), multiple-choice

- "collected by graduate and undergraduate students from freely available sources online"

- Really about testing knowledge, not language understanding

- Evaluated on GPT-3 using few-shot prompting

<img src="../../materials/spring2025-lectures/images/mmlu.png" width="800">

[[HELM MMLU for visualizing predictions]](https://crfm.stanford.edu/helm/mmlu/latest/)

### MMLU-Pro

[MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark](https://arxiv.org/abs/2406.01574)

- Removed noisy/trivial questions from MMLU

- Expanded 4 choices to 10 choices

- Evaluated using chain of thought (gives model more of a chance)

- Accuracy of models drop by 16% to 33% (not as saturated)

<img src="../../materials/spring2025-lectures/images/mmlu-pro.png" width="800">

[[HELM MMLU-Pro for visualizing predictions]](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/mmlu_pro)

### Graduate-Level Google-Proof Q&A (GPQA)

[GPQA: A Graduate-Level Google-Proof Q&A Benchmark](https://arxiv.org/abs/2311.12022)

- Questions written by 61 PhD contractors from Upwork

<img src="../../materials/spring2025-lectures/images/gpqa.png" width="800">

- PhD experts achieve 65% accuracy

- Non-experts achieve 34% over 30 minutes with access to Google

- GPT-4 achieves 39%

[[HELM GPQA for visualizing predictions]](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/gpqa)

### Humanity's Last Exam

[Humanity's Last Exam](https://arxiv.org/abs/2501.14249)

- 2500 questions: multimodal, many subjects, multiple-choice + short-answer

<img src="../../materials/spring2025-lectures/images/hle-examples.png" width="800">

- Awarded $500K prize pool + co-authorship to question creators

- Filtered by frontier LLMs, multiple stages of review

<img src="../../materials/spring2025-lectures/images/hle-pipeline.png" width="800">

<img src="../../materials/spring2025-lectures/images/hle-results.png" width="800">

[[latest leaderboard]](https://agi.safe.ai/)

So far, we've been evaluating on fairly structured tasks.

Instruction following (as popularized by ChatGPT): just follow the instructions.

Challenge: how to evaluate an open-ended response?

### Chatbot Arena

[Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](https://arxiv.org/abs/2403.04132)

How it works:

- Random person from the Internet types in prompt

- They get response from two random (anonymized) models

- They rate which one is better

- ELO scores are computed based on the pairwise comparisons

- Features: live (not static) inputs, can accomodate new models

<img src="../../materials/spring2025-lectures/images/chatbot-arena-leaderboard.png" width="800">

[[Chatbot Arena]](https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard)

### Instruction-Following Eval (IFEval)

[Instruction-Following Evaluation for Large Language Models](https://arxiv.org/abs/2311.07911)

<img src="../../materials/spring2025-lectures/images/ifeval-categories.png" width="600">

- Add simple synthetic constraints to instructions

- Constraints can be automatically verified, but not the semantics of the response

- Fairly simple instructions, constraints are a bit artificial

[[HELM IFEval for visualizing predictions]](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/ifeval)

### AlpacaEval

[https://tatsu-lab.github.io/alpaca_eval/](https://tatsu-lab.github.io/alpaca_eval/)

- 805 instructions from various sources

- Metric: win rate against GPT-4 preview as judged by GPT-4 preview (potential bias)

<img src="../../materials/spring2025-lectures/images/alpacaeval-leaderboard.png" width="600">

### WildBench

[WildBench: Benchmarking LLMs with Challenging Tasks from Real Users in the Wild](https://arxiv.org/pdf/2406.04770)

- Sourced 1024 examples from 1M human-chatbot conversations

- Uses GPT-4 turbo as a judge with a checklist (like CoT for judging) + GPT-4 as a judge

- Well-correlated (0.95) with Chatbot Arena (seems to be the de facto sanity check for benchmarks)

<img src="../../materials/spring2025-lectures/images/wildbench.png" width="800">

[[HELM WildBench for visualizing predictions]](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/wildbench)

Consider tasks that require tool use (e.g., running code) and iterating over a period of time

Agent = language model + agent scaffolding (logic for deciding how to use the LM)

### SWEBench

[SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770)

- 2294 tasks across 12 Python repositories

- Given codebase + issue description, submit a PR

- Evaluation metric: unit tests

<img src="../../materials/spring2025-lectures/images/swebench.png" width="800">

### CyBench

[Cybench: A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models](https://arxiv.org/abs/2408.08926)

- 40 Capture the Flag (CTF) tasks

- Use first-solve time as a measure of difficulty

<img src="../../materials/spring2025-lectures/images/cybench.png" width="800">

<img src="../../materials/spring2025-lectures/images/cybench-agent.png" width="800">

<img src="../../materials/spring2025-lectures/images/cybench-results.png" width="800">

### MLEBench

[MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering](https://arxiv.org/abs/2410.07095)

- 75 Kaggle competitions (require training models, processing data, etc.)

<img src="../../materials/spring2025-lectures/images/mlebench.png" width="800">

<img src="../../materials/spring2025-lectures/images/mlebench-results.png" width="800">

All of the tasks so far require linguistic and world knowledge

Can we isolate reasoning from knowledge?

Arguably, reasoning captures a more pure form of intelligence (isn't just about memorizing facts)

[ARC-AGI](https://arcprize.org/arc-agi)

Introduced in 2019 by Francois Chollet

ARC-AGI-1

<img src="../../materials/spring2025-lectures/var/files/image-d1a33e9159cfdb77197551bbbecc6a76-https_arcprize_org_media_images_arc-task-grids_jpg" width="800">

<img src="../../materials/spring2025-lectures/var/files/image-cb3ca481cdf3a72e48fabc7953f49867-https_arcprize_org_media_images_oseriesleaderboard_png" width="800">

ARC-AGI-2: harder

<img src="../../materials/spring2025-lectures/var/files/image-a0338c9fb72d1163cfc8ac66fea4e4ed-https_arcprize_org_media_images_blog_arc-agi-2-unsolved-1_png" width="800">

<img src="../../materials/spring2025-lectures/var/files/image-a375cd28c372458baf4135c081a1ce8b-https_www_team-bhp_com_forum_attachments_road-safety_2173645d1625144681-will-crash-test-rating-change-if-higher-variant-chosen-images-30_jpeg" width="500">

What does safety mean for AI?

[[HELM safety: curated set of benchmarks]](https://crfm.stanford.edu/helm/safety/latest/#/leaderboard)

### HarmBench

[HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal](https://arxiv.org/abs/2402.04249)

- Based on 510 harmful behaviors that violate laws or norms

[[HarmBench on HELM]](https://crfm.stanford.edu/helm/safety/latest/#/leaderboard/harm_bench)

[[Example of safety failure]](https://crfm.stanford.edu/helm/safety/latest/#/runs/harm_bench:model=anthropic_claude-3-7-sonnet-20250219?instancesPage=4)

### AIR-Bench

[AIR-Bench 2024: A Safety Benchmark Based on Risk Categories from Regulations and Policies](https://arxiv.org/abs/2407.17436)

- Based on regulatory frameworks and company policies

- Taxonomized into 314 risk categories, 5694 prompts

<img src="../../materials/spring2025-lectures/var/files/image-0195288103aa687af7faedeaca538056-https_crfm_stanford_edu_helm_assets_air-overview-d2e6c49f_png" width="800">

[[HELM AIR-Bench]](https://crfm.stanford.edu/helm/air-bench/latest/#/leaderboard)

### Jailbreaking

- Language models are trained to refuse harmful instructions

- Greedy Coordinate Gradient (GCG) automatically optimizes prompts to bypass safety [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/pdf/2307.15043)

- Transfers from open-weight models (Llama) to closed models (GPT-4)

<img src="../../materials/spring2025-lectures/images/gcg-examples.png" width="800">

### Pre-deployment testing

- US Safety Institute + UK AI Safety Institute working together

- Company gives safety institutes access to model before release (currently voluntary)

- Safety institutes run evaluations and produce a report to company

[[report]](https://www.nist.gov/system/files/documents/2024/12/18/US_UK_AI%20Safety%20Institute_%20December_Publication-OpenAIo1.pdf)

### But what is safety?

- Many aspects of safety are strongly contextual (politics, law, social norms - which vary across countries)

- Naively, one might think safety is about refusal and is at odds with capability, but there's more...

- Hallucinations in a medical setting makes systems more capable and more safe

Two aspects of a model that reduce safety: capabilities + propensity

- A system could be capable of doing something, but refuse to do it

- For API models, propensity matters

- For open weight models, capability matters (since can easily fine-tune safety away)

**Dual-use**: capable cybersecurity agents (do well on CyBench) can be used to hack into a system or to do penetration testing

CyBench is used by the safety institute as a safety evaluation, but is it really a capability evaluation?

Language models are used heavily in practice:

<img src="../../materials/spring2025-lectures/images/openai-100b-tokens.png" width="600">

[[tweet]](https://x.com/sama/status/1756089361609981993)

<img src="../../materials/spring2025-lectures/images/cursor-1b-lines.png" width="600">

[[tweet]](https://x.com/amanrsanger/status/1916968123535880684)

However, most existing benchmarks (e.g., MMLU) are far away from real-world use.

Live traffic from real people contain garbage, that's not always what we want either.

Two types of prompts:

1. Quizzing: User knows the answer and trying to test the system (think standardized exams).

2. Asking: User doesn't know the answer is trying to use the system to get it.

Asking is more realistic and produces value for the user.

### Clio (Anthropic)

[Clio: Privacy-Preserving Insights into Real-World AI Use](https://arxiv.org/abs/2412.13678)

- Use language models to analyze real user data

- Share general patterns of what people are asking

<img src="../../materials/spring2025-lectures/images/clio-table4.png" width="700">

### MedHELM

[Clio: Privacy-Preserving Insights into Real-World AI Use](https://arxiv.org/abs/2412.13678)

- Previous medical benchmarks were based on standardized exams

- 121 clinical tasks sourced from 29 clinicians, mixture of private and public datasets

<img src="../../materials/spring2025-lectures/var/files/image-dd0716ead405df19ab6676c1baa73632-https_crfm_stanford_edu_helm_assets_medhelm-overview-3ddfcd65_png" width="700">

[[MedHELM]](https://crfm.stanford.edu/helm/medhelm/latest/#/leaderboard)

Unfortunately, realism and privacy are sometimes at odds with each other.

How do we know our evaluations are valid?

### Train-test overlap

- Machine learning 101: don't train on your test set

- Pre-foundation models (ImageNet, SQuAD): well-defined train-test splits

- Nowadays: train on the Internet and don't tell people about your data

Route 1: try to infer train-test overlap from model

- Exploit exchangeability of data points [Proving Test Set Contamination in Black Box Language Models](https://arxiv.org/pdf/2310.17623)

<img src="../../materials/spring2025-lectures/images/contamination-exchangeability.png" width="600">

Route 2: encourage reporting norms (e.g., people report confidence intervals)

- Model providers should report train-test overlap [Language model developers should report train-test overlap](https://arxiv.org/abs/2410.08385)

### Dataset quality

- Fixed up SWE-Bench to produce SWE-Bench Verified [[blog]](https://openai.com/index/introducing-swe-bench-verified/)

- Create Platinum versions of benchmarks [Do Large Language Model Benchmarks Test Reliability?](https://arxiv.org/abs/2502.03461)

<img src="../../materials/spring2025-lectures/var/files/image-a1149b095a48ea306dcdea342363230c-https_pbs_twimg_com_media_GjICXQlWkAAYnDS_format_jpg_name_4096x4096" width="700">

<img src="../../materials/spring2025-lectures/var/files/image-306ae3f862cee7c7842c0b29af3a2f5c-https_pbs_twimg_com_media_GjICcGQXYAAM4o1_format_jpg_name_4096x4096" width="800">

What are we even evaluating?

In other words, what are the rules of a game?

Pre-foundation models, we evaluated **methods** (standardized train-test splits).

Today, we're evaluating **models/systems** (anything goes).

There are some exceptions...

nanogpt speedrun: fixed data, compute time to get to a particular validation loss

<img src="../../materials/spring2025-lectures/images/karpathy-nanogpt-speedrun.png" width="600">

[[X]](https://x.com/karpathy/status/1846790537262571739)

DataComp-LM: given a raw dataset, get the best accuracy using standard training pipeline [DataComp-LM: In search of the next generation of training sets for language models](https://arxiv.org/abs/2406.11794)

Evaluating methods encourage algorithmic innovation from researchers.

Evaluating models/systems is useful for downstream users.

Either way, we need to define the rules of the game!

Takeaways

- There is no one true evaluation; choose the evaluation depending on what you're trying to measure.

- Always look at the individual instances and the predictions.

- There are many aspects to consider: capabilities, safety, costs, realism.

- Clearly state the rules of the game (methods versus models/systems).
