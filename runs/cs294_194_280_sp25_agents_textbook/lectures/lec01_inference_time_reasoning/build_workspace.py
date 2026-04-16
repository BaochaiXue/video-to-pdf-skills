#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parent

VIDEO_URL = "https://www.youtube.com/live/g0Dwtf3BH-0"
COURSE_PAGE = "https://rdi.berkeley.edu/adv-llm-agents/sp25"
INTRO_SLIDES_URL = "https://rdi.berkeley.edu/adv-llm-agents/slides/llm-agents-berkeley-intro-sp25.pdf"
MAIN_SLIDES_URL = "https://rdi.berkeley.edu/adv-llm-agents/slides/inference_time_techniques_lecture_sp25.pdf"


READINGS = [
    {
        "paper_id": "reading_01",
        "paper_title": "Large Language Models as Optimizers",
        "url": "https://arxiv.org/abs/2309.03409",
        "main_question": "Can an LLM act as an optimizer by proposing candidate instructions or solutions from the history of past trials and scores?",
        "core_method": "Optimization by PROmpting (OPRO): keep a trajectory of candidate texts with scalar scores, ask the LLM to propose a new candidate conditioned on this optimization history, then evaluate and append the result.",
        "key_result": "On prompt optimization tasks such as GSM8K and BBH, LLM-generated prompts improve substantially over human-written baselines and can approach few-shot CoT quality without manual exemplar annotation.",
        "limitations": "The loop still needs a trustworthy evaluator and is task-specific; optimization can plateau and may overfit small evaluation sets.",
        "connection_to_lecture": "This is the lecture's canonical example for turning prompt design into an explicit inference-time search loop over instructions.",
        "should_appear_in_sections": ["3.4"],
        "abstract": "Optimization is ubiquitous. While derivative-based algorithms have been powerful tools for various problems, the absence of gradient imposes challenges on many real-world applications. In this work, we propose Optimization by PROmpting (OPRO), a simple and effective approach to leverage large language models (LLMs) as optimizers, where the optimization task is described in natural language. In each optimization step, the LLM generates new solutions from the prompt that contains previously generated solutions with their values, then the new solutions are evaluated and added to the prompt for the next optimization step.",
    },
    {
        "paper_id": "reading_02",
        "paper_title": "Large Language Models Cannot Self-Correct Reasoning Yet",
        "url": "https://arxiv.org/abs/2310.01798",
        "main_question": "Can an LLM improve reasoning quality by revising its own answer without any reliable external feedback?",
        "core_method": "Study intrinsic self-correction on reasoning tasks, comparing original answers, self-corrected answers, and oracle-feedback variants.",
        "key_result": "Without oracle or task-grounded external feedback, self-correction often fails to improve and can even hurt reasoning performance.",
        "limitations": "The result is strongest for the investigated prompting/evaluation setups; better evaluators or process supervision may change the picture.",
        "connection_to_lecture": "This reading directly grounds the lecture's warning that iterative self-improvement only works when the feedback signal is good.",
        "should_appear_in_sections": ["5.3"],
        "abstract": "Large Language Models (LLMs) have emerged as a groundbreaking technology with their unparalleled text generation capabilities across various applications. Nevertheless, concerns persist regarding the accuracy and appropriateness of their generated content. Central to our investigation is the notion of intrinsic self-correction, whereby an LLM attempts to correct its initial responses based solely on its inherent capabilities, without the crutch of external feedback. In the context of reasoning, our research indicates that LLMs struggle to self-correct their responses without external feedback, and at times, their performance even degrades after self-correction.",
    },
    {
        "paper_id": "reading_03",
        "paper_title": "Teaching Large Language Models to Self-Debug",
        "url": "https://arxiv.org/abs/2304.05128",
        "main_question": "Can LLMs debug their own generated programs more effectively when they are asked to explain, test, and revise code?",
        "core_method": "Self-Debugging: use few-shot demonstrations to teach the model to inspect execution results, explain code, and refine its program.",
        "key_result": "Self-debugging improves code generation accuracy on Spider, TransCoder, and MBPP, especially when execution feedback or explanatory traces are available.",
        "limitations": "The gains rely on task settings where execution or meaningful program inspection is available; this does not automatically transfer to open-ended reasoning tasks without feedback.",
        "connection_to_lecture": "This is the lecture's strongest positive case for iterative self-improvement, because code tasks expose external signals such as tests and traces.",
        "should_appear_in_sections": ["5.2"],
        "abstract": "Large language models (LLMs) have achieved impressive performance on code generation. However, for complex programming tasks, generating the correct solution in one go becomes challenging. In this work, we propose Self-Debugging, which teaches a large language model to debug its predicted program via few-shot demonstrations. Self-Debugging achieves the state-of-the-art performance on several code generation benchmarks, and by leveraging feedback messages and reusing failed predictions, it notably improves sample efficiency.",
    },
]


SEGMENTS = [
    {
        "segment_id": "segment_01",
        "title": "课程定位与推理模型为何重要",
        "start": "00:00:00,000",
        "end": "00:08:30,000",
        "slide_refs": [("intro", 5), ("intro", 8), ("intro", 9), ("main", 2), ("main", 3), ("main", 6)],
        "target_section": "1",
        "required_figures": ["lec01_fig_001", "lec01_fig_002", "lec01_fig_003"],
        "required_formulas": [],
        "required_code": [],
    },
    {
        "segment_id": "segment_02",
        "title": "单轨推理的起点：Standard Prompting、Few-shot CoT 与 Zero-shot CoT",
        "start": "00:08:30,000",
        "end": "00:20:30,000",
        "slide_refs": [("main", 9), ("main", 10), ("main", 11), ("main", 12), ("main", 13), ("main", 14)],
        "target_section": "2.1",
        "required_figures": ["lec01_fig_004"],
        "required_formulas": ["formula_self_consistency_vote"],
        "required_code": ["code_analogy_prompt"],
    },
    {
        "segment_id": "segment_03",
        "title": "Analogical Prompting：让模型先回忆示例再求解",
        "start": "00:20:30,000",
        "end": "00:31:30,000",
        "slide_refs": [("main", 15), ("main", 16), ("main", 17), ("main", 18), ("main", 19), ("main", 20), ("main", 21), ("main", 22)],
        "target_section": "2.2",
        "required_figures": ["lec01_fig_005"],
        "required_formulas": [],
        "required_code": ["code_analogy_prompt"],
    },
    {
        "segment_id": "segment_04",
        "title": "把提示工程变成优化：OPRO、Least-to-Most 与 Self-Discover",
        "start": "00:31:30,000",
        "end": "00:45:00,000",
        "slide_refs": [("main", 23), ("main", 24), ("main", 25), ("main", 26), ("main", 27), ("main", 28), ("main", 29), ("main", 30), ("main", 31), ("main", 32), ("main", 33), ("main", 34), ("main", 35), ("main", 36), ("main", 37)],
        "target_section": "2.3",
        "required_figures": ["lec01_fig_006", "lec01_fig_007"],
        "required_formulas": ["formula_opro_update"],
        "required_code": ["code_opro_loop"],
    },
    {
        "segment_id": "segment_05",
        "title": "宽搜索：Self-Consistency 与结果聚合",
        "start": "00:45:00,000",
        "end": "00:56:30,000",
        "slide_refs": [("main", 39), ("main", 40), ("main", 41), ("main", 42), ("main", 43), ("main", 44), ("main", 45), ("main", 46), ("main", 47), ("main", 48), ("main", 49), ("main", 50), ("main", 51)],
        "target_section": "3.1",
        "required_figures": ["lec01_fig_008", "lec01_fig_009"],
        "required_formulas": ["formula_self_consistency_vote"],
        "required_code": ["code_self_consistency"],
    },
    {
        "segment_id": "segment_06",
        "title": "评分器与部分解搜索：Verifier、PRM/ORM 与 Tree of Thoughts",
        "start": "00:56:30,000",
        "end": "01:04:30,000",
        "slide_refs": [("main", 52), ("main", 53), ("main", 54), ("main", 55), ("main", 56), ("main", 57), ("main", 58)],
        "target_section": "3.2",
        "required_figures": ["lec01_fig_010", "lec01_fig_011"],
        "required_formulas": ["formula_verifier_rank"],
        "required_code": ["code_tot_search"],
    },
    {
        "segment_id": "segment_07",
        "title": "深度自改进：Reflexion、Self-Refine 与 Self-Debugging",
        "start": "01:04:30,000",
        "end": "01:12:15,000",
        "slide_refs": [("main", 60), ("main", 61), ("main", 62), ("main", 63), ("main", 64), ("main", 65)],
        "target_section": "4.1",
        "required_figures": ["lec01_fig_012"],
        "required_formulas": [],
        "required_code": ["code_self_debug"],
    },
    {
        "segment_id": "segment_08",
        "title": "为什么很多自我修正会失败",
        "start": "01:12:15,000",
        "end": "01:16:30,000",
        "slide_refs": [("main", 66), ("main", 67), ("main", 68), ("main", 69)],
        "target_section": "4.2",
        "required_figures": ["lec01_fig_013"],
        "required_formulas": [],
        "required_code": [],
    },
    {
        "segment_id": "segment_09",
        "title": "推理时算力分配与 Bitter Lesson",
        "start": "01:16:30,000",
        "end": "01:21:32,790",
        "slide_refs": [("main", 70), ("main", 71), ("main", 72), ("main", 73)],
        "target_section": "5",
        "required_figures": ["lec01_fig_014", "lec01_fig_015"],
        "required_formulas": ["formula_budget_tradeoff"],
        "required_code": [],
    },
]


FIGURES = [
    {"figure_id": "lec01_fig_001", "deck": "intro", "page": 5, "used_for": "解释 agent 与 environment 的闭环关系", "target_section": "1.1", "caption": "课程导入页中的 agent-environment 闭环：LLM agent 不是只生成文本，而是通过推理、工具、记忆、行动与反馈形成外部交互回路。"},
    {"figure_id": "lec01_fig_002", "deck": "intro", "page": 8, "used_for": "说明 2024-2025 reasoning model 的发展背景", "target_section": "1.1", "caption": "课程导入页给出的 reasoning model 时间线：o1、Gemini 2.0 Flash Thinking、o3、DeepSeek-R1 等模型让 test-time reasoning 成为课程的起点。"},
    {"figure_id": "lec01_fig_003", "deck": "main", "page": 6, "used_for": "概括长链式思维的核心想法", "target_section": "1.2", "caption": "讲义主线的核心图：通过触发更长的 chain-of-thought，让模型在推理时消耗更多 token budget。"},
    {"figure_id": "lec01_fig_004", "deck": "main", "page": 12, "used_for": "展示 zero-shot CoT 的典型触发方式", "target_section": "2.1", "caption": "Zero-shot CoT 的经典提示方式：一句 “Let's think step by step” 就能显著改变推理轨迹。"},
    {"figure_id": "lec01_fig_005", "deck": "main", "page": 16, "used_for": "解释 analogical prompting 的输入输出结构", "target_section": "2.2", "caption": "Analogical prompting 的结构：先让模型回忆相关 exemplars，再求解当前问题。"},
    {"figure_id": "lec01_fig_006", "deck": "main", "page": 25, "used_for": "说明 OPRO 的 optimizer/evaluator 分工", "target_section": "2.3", "caption": "OPRO 用语言模型本身充当 optimizer：根据既有候选与得分轨迹提出新 instruction。"},
    {"figure_id": "lec01_fig_007", "deck": "main", "page": 35, "used_for": "解释 Self-Discover 如何组合 reasoning structure", "target_section": "2.3", "caption": "Self-Discover 不再固定一种提示模板，而是让模型为当前任务自组装 reasoning structure。"},
    {"figure_id": "lec01_fig_008", "deck": "main", "page": 40, "used_for": "解释 self-consistency 的投票思想", "target_section": "3.1", "caption": "Self-Consistency 的基本思想：保留多条 reasoning path，只在最终答案层面做聚合。"},
    {"figure_id": "lec01_fig_009", "deck": "main", "page": 47, "used_for": "解释 AlphaCode 的执行一致性聚类", "target_section": "3.1", "caption": "AlphaCode 中的代码一致性选择：按程序在生成测试输入上的执行结果聚类。"},
    {"figure_id": "lec01_fig_010", "deck": "main", "page": 52, "used_for": "区分 outcome-level 与 process-level verifier", "target_section": "3.2", "caption": "Lecture 中对 verifier 的两种范式区分：ORM 在整题级别打分，PRM 在步骤级别打分。"},
    {"figure_id": "lec01_fig_011", "deck": "main", "page": 55, "used_for": "展示 Tree of Thoughts 在 partial solutions 上搜索", "target_section": "3.2", "caption": "Tree of Thoughts 的代表性示意：在部分思路节点上进行 thought generation 与 state evaluation。"},
    {"figure_id": "lec01_fig_012", "deck": "main", "page": 64, "used_for": "比较 self-debugging 的不同反馈形式", "target_section": "4.1", "caption": "Self-Debugging 的反馈形式对比：简短反馈、unit test 结果、代码解释与 execution trace 的信息密度不同。"},
    {"figure_id": "lec01_fig_013", "deck": "main", "page": 67, "used_for": "强调无外部反馈时 self-correction 的风险", "target_section": "4.2", "caption": "Lecture 的负面结果页：没有 oracle 或可靠外部反馈时，self-correction 可能降低推理性能。"},
    {"figure_id": "lec01_fig_014", "deck": "main", "page": 70, "used_for": "说明 token budget 的分配问题", "target_section": "5.1", "caption": "推理时 token budget 不只是“多想一点”这么简单，而是要在并行采样、串行修正和验证上做任务依赖的分配。"},
    {"figure_id": "lec01_fig_015", "deck": "main", "page": 73, "used_for": "总结 lecture 的总原则", "target_section": "5.2", "caption": "Bitter Lesson 在本讲中的含义：应偏好那些能随着计算量增长持续扩展的通用 reasoning 方法。"},
]


FORMULAS = [
    {
        "formula_id": "formula_self_consistency_vote",
        "name": "Self-Consistency 投票",
        "latex": r"\\hat{y} = \\arg\\max_{y} \\sum_{i=1}^{N} \\mathbf{1}[\\operatorname{extract}(a_i)=y]",
        "symbols": {
            "\\hat{y}": "最终输出的聚合答案",
            "a_i": "第 i 条 sampled reasoning path 的完整回答",
            "\\operatorname{extract}(a_i)": "从回答中抽取出的最终答案",
            "N": "采样得到的候选答案数量",
        },
        "source_basis": "Lecture pages 39-44 on self-consistency; the equation is a note-side formalization of the slide concept.",
        "target_section": "3.1",
    },
    {
        "formula_id": "formula_opro_update",
        "name": "OPRO 更新视角",
        "latex": r"x_{t+1} \\sim \\operatorname{LLM}\\left(\\{(x_i, s_i)\\}_{i=1}^{t}, \\mathcal{E}\\right)",
        "symbols": {
            "x_i": "第 i 轮已有的 instruction 或候选解",
            "s_i": "对应候选在小验证集上的分数",
            "\\mathcal{E}": "任务 exemplars 或应用说明",
            "x_{t+1}": "由语言模型提出的新候选",
        },
        "source_basis": "Lecture pages 24-28 and reading Large Language Models as Optimizers.",
        "target_section": "2.3",
    },
    {
        "formula_id": "formula_verifier_rank",
        "name": "Verifier 打分与搜索优先级",
        "latex": r"\\tau^{\\star} = \\arg\\max_{\\tau \\in \\mathcal{T}} r_{\\phi}(\\tau), \\qquad r_{\\phi}(\\tau)=\\sum_{t} r_{\\phi}(s_t, a_t)",
        "symbols": {
            "\\tau": "一条完整或部分 reasoning trajectory",
            "\\mathcal{T}": "待比较的 candidate trajectories",
            "r_{\\phi}": "verifier 或 reward model 的评分函数",
            "(s_t, a_t)": "第 t 步的状态与动作/推理步",
        },
        "source_basis": "Lecture pages 52-57 on ORM, PRM, and tree search; the equation makes the scoring intuition explicit.",
        "target_section": "3.2",
    },
    {
        "formula_id": "formula_budget_tradeoff",
        "name": "推理时预算分配",
        "latex": r"\\max_{M, N, D} \\ \\operatorname{Acc}(M, N, D) \\quad \\text{s.t.} \\quad C(M, N, D) \\le B",
        "symbols": {
            "M": "模型规模或模型家族的选择",
            "N": "并行采样的宽度",
            "D": "串行修正/搜索的深度",
            "C": "总 inference cost",
            "B": "给定的 test-time compute budget",
        },
        "source_basis": "Lecture pages 70-71 on token budget and model-size tradeoffs; the optimization form is a note-side abstraction.",
        "target_section": "5.1",
    },
]


CODE_UNITS = [
    {
        "code_id": "code_analogy_prompt",
        "title": "Analogical prompting 的两阶段提示模板",
        "kind": "pseudocode",
        "target_section": "2.2",
        "snippet": "Instruction:\\n1. Recall relevant exemplars for the current task.\\n2. Solve the initial problem using the recalled exemplars.",
        "source_basis": "Lecture pages 15-16.",
    },
    {
        "code_id": "code_opro_loop",
        "title": "OPRO 的 optimizer-evaluator 循环",
        "kind": "pseudocode",
        "target_section": "2.3",
        "snippet": "Initialize instruction x0\\nfor t in 0..T-1:\\n    score xt on a validation set\\n    append (xt, score) to the optimization history\\n    ask the LLM to propose xt+1 from the sorted history",
        "source_basis": "Lecture pages 24-28 and the OPRO reading.",
    },
    {
        "code_id": "code_self_consistency",
        "title": "Self-Consistency 解码",
        "kind": "pseudocode",
        "target_section": "3.1",
        "snippet": "Sample N reasoning paths with diverse decoding\\nExtract final answers\\nCount answer frequencies\\nReturn the majority answer",
        "source_basis": "Lecture pages 40-44.",
    },
    {
        "code_id": "code_tot_search",
        "title": "Tree-of-Thought 搜索流程",
        "kind": "pseudocode",
        "target_section": "3.2",
        "snippet": "Initialize frontier with the empty state\\nrepeat:\\n    expand candidate next thoughts\\n    score partial states\\n    keep the best frontier states\\nuntil a complete solution is found or budget is exhausted",
        "source_basis": "Lecture pages 54-57.",
    },
    {
        "code_id": "code_self_debug",
        "title": "Self-Debugging 迭代",
        "kind": "pseudocode",
        "target_section": "4.1",
        "snippet": "Generate code\\nRun tests or inspect execution results\\nAsk the model to explain/debug the program\\nRevise the code using the observed feedback",
        "source_basis": "Lecture pages 63-65 and the Self-Debug reading.",
    },
]


PAPER_MENTIONS = [
    "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
    "Show Your Work: Scratchpads for Intermediate Computation with Language Models",
    "Large Language Models are Zero-Shot Reasoners",
    "Large Language Models as Analogical Reasoners",
    "Large Language Models are Human-Level Prompt Engineers",
    "Large Language Models as Optimizers",
    "Least-to-Most Prompting Enables Complex Reasoning in Large Language Models",
    "Compositional Semantic Parsing with Large Language Models",
    "SELF-DISCOVER: Large Language Models Self-Compose Reasoning Structures",
    "Self-Consistency Improves Chain of Thought Reasoning in Language Models",
    "Competition-level Code Generation with AlphaCode",
    "Universal Self-Consistency for Large Language Model Generation",
    "Training Verifiers to Solve Math Word Problems",
    "Let's Verify Step by Step",
    "Tree of Thoughts: Deliberate Problem Solving with Large Language Models",
    "Reflexion: Language Agents with Verbal Reinforcement Learning",
    "Self-Refine: Iterative Refinement with Self-Feedback",
    "Teaching Large Language Models to Self-Debug",
    "Large Language Models Cannot Self-Correct Reasoning Yet",
    "Improving Factuality and Reasoning in Language Models through Multiagent Debate",
    "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters",
    "Inference Scaling Laws: An Empirical Analysis of Compute-Optimal Inference for LLM Problem-Solving",
]


def parse_srt(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n")
    blocks = re.split(r"\n\s*\n", text.strip())
    rows = []
    idx = 1
    for block in blocks:
        lines = [line for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        time_line = lines[1] if lines[0].isdigit() else lines[0]
        if "-->" not in time_line:
            continue
        start, end = [part.strip() for part in time_line.split("-->", 1)]
        payload = lines[2:] if lines[0].isdigit() else lines[1:]
        text = " ".join(line.strip() for line in payload).strip()
        text = re.sub(r"\s+", " ", text)
        speaker = None
        if ":" in text:
            maybe_speaker, rest = text.split(":", 1)
            if maybe_speaker.isupper() and 1 <= len(maybe_speaker.split()) <= 4:
                speaker = maybe_speaker.title()
                text = rest.strip()
        rows.append(
            {
                "unit_id": f"transcript_{idx:06d}",
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": text,
                "confidence": "high",
                "source": "youtube_caption",
            }
        )
        idx += 1
    return rows


def page_title(text: str) -> str:
    for raw in text.splitlines():
        line = raw.strip()
        if line:
            return line[:160]
    return "Untitled"


def extract_pages(pdf_path: Path, source: str, prefix: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    rows = []
    for idx, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        dense = len(text.split()) > 45 or text.count("•") + text.count("●") >= 4
        rows.append(
            {
                "unit_id": f"{prefix}_{idx:03d}",
                "page": idx,
                "title": page_title(text),
                "text": text,
                "figures": [],
                "dense": dense,
                "source": source,
                "deck": prefix,
                "kind": "slide_page",
            }
        )
    return rows


def render_figure(fig: dict) -> str:
    figures_dir = ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)
    pdf_name = "slides_intro.pdf" if fig["deck"] == "intro" else "slides_main.pdf"
    doc = fitz.open(ROOT / pdf_name)
    page = doc.load_page(fig["page"] - 1)
    pix = page.get_pixmap(matrix=fitz.Matrix(1.8, 1.8), alpha=False)
    asset_name = f"{fig['figure_id']}.png"
    asset_path = figures_dir / asset_name
    pix.save(asset_path)
    return str(Path("figures") / asset_name)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")


def main() -> None:
    transcript_source = ROOT / "recording.en-j3PyPqV-e1s.srt"
    transcript_raw = ROOT / "transcript_raw.srt"
    if transcript_source.exists():
        shutil.copyfile(transcript_source, transcript_raw)

    info = json.loads((ROOT / "recording.info.json").read_text(encoding="utf-8"))
    transcript_rows = parse_srt(transcript_raw)
    slides_rows = extract_pages(ROOT / "slides_intro.pdf", "slides_intro.pdf", "intro_slide") + extract_pages(
        ROOT / "slides_main.pdf", "slides_main.pdf", "main_slide"
    )

    write_jsonl(ROOT / "transcript.jsonl", transcript_rows)
    write_jsonl(ROOT / "slides.jsonl", slides_rows)

    segments_rows = []
    for seg in SEGMENTS:
        source_unit_ids = []
        for transcript_row in transcript_rows:
            if seg["start"] <= transcript_row["start"] <= seg["end"]:
                source_unit_ids.append(transcript_row["unit_id"])
        for deck, page in seg["slide_refs"]:
            source_unit_ids.append(f"{'intro_slide' if deck == 'intro' else 'main_slide'}_{page:03d}")
        segments_rows.append(
            {
                "segment_id": seg["segment_id"],
                "title": seg["title"],
                "start": seg["start"],
                "end": seg["end"],
                "target_section": seg["target_section"],
                "source_unit_ids": source_unit_ids,
            }
        )
    write_jsonl(ROOT / "segments.jsonl", segments_rows)

    lecture_plan = {
        "lecture_id": "L01",
        "title": "Inference-Time Techniques for LLM Reasoning",
        "speaker": "Xinyun Chen",
        "course_mode": True,
        "source_inventory": [
            {"source_id": "course_page", "source_type": "course_page", "required_for_coverage": True, "status": "available"},
            {"source_id": "recording_info", "source_type": "youtube_metadata", "required_for_coverage": True, "status": "available"},
            {"source_id": "transcript_raw", "source_type": "youtube_caption", "required_for_coverage": True, "status": "available"},
            {"source_id": "slides_intro", "source_type": "official_slide_pdf", "required_for_coverage": True, "status": "available"},
            {"source_id": "slides_main", "source_type": "official_slide_pdf", "required_for_coverage": True, "status": "available"},
            {"source_id": "readings_manifest", "source_type": "supplemental_readings", "required_for_coverage": True, "status": "available"},
        ],
        "segment_ids": [seg["segment_id"] for seg in SEGMENTS],
        "must_cover_kinds": [
            "motivation",
            "definition",
            "algorithm",
            "example",
            "paper_summary",
            "caveat",
            "open_problem",
        ],
        "must_emit_artifacts": [
            "source_manifest.json",
            "transcript.jsonl",
            "slides.jsonl",
            "coverage_units.jsonl",
            "figure_manifest.json",
            "lecture.tex",
            "lecture.pdf",
            "eval_report.json",
            "repair_log.jsonl",
            "lecture_quality_report.md",
        ],
        "evaluator_thresholds": {
            "coverage": 0.95,
            "pedagogical_depth": 0.85,
            "hallucination_control": 0.90,
            "reading_integration": 0.80,
        },
    }
    write_json(ROOT / "lecture_plan.json", lecture_plan)

    write_json(
        ROOT / "readings_manifest.json",
        {
            "lecture_id": "L01",
            "lecture_title": "Inference-Time Techniques for LLM Reasoning",
            "readings": READINGS,
        },
    )
    write_jsonl(ROOT / "paper_summaries.jsonl", READINGS)

    reading_units = []
    for item in READINGS:
        reading_units.append(
            {
                "unit_id": item["paper_id"],
                "paper_title": item["paper_title"],
                "url": item["url"],
                "importance": "required",
                "connection_to_lecture": item["connection_to_lecture"],
                "should_appear_in_sections": item["should_appear_in_sections"],
                "status": "covered",
            }
        )
    write_jsonl(ROOT / "reading_coverage_units.jsonl", reading_units)

    aligned_units = []
    slide_transcript_alignment = []
    for seg in SEGMENTS:
        slide_ids = [
            f"{'intro_slide' if deck == 'intro' else 'main_slide'}_{page:03d}"
            for deck, page in seg["slide_refs"]
        ]
        transcript_ids = [
            row["unit_id"] for row in transcript_rows if seg["start"] <= row["start"] <= seg["end"]
        ]
        aligned_units.append(
            {
                "aligned_unit_id": seg["segment_id"],
                "segment_title": seg["title"],
                "transcript_unit_ids": transcript_ids[:80],
                "slide_unit_ids": slide_ids,
                "start": seg["start"],
                "end": seg["end"],
                "target_section": seg["target_section"],
                "alignment_confidence": "medium",
            }
        )
        slide_transcript_alignment.append(
            {
                "segment_id": seg["segment_id"],
                "slide_unit_ids": slide_ids,
                "transcript_range": {"start": seg["start"], "end": seg["end"]},
                "method": "manual-lecture-outline alignment based on slide sequence and topic keywords in captions",
                "confidence": "medium",
            }
        )
    write_jsonl(ROOT / "aligned_units.jsonl", aligned_units)
    write_jsonl(ROOT / "slide_transcript_alignment.jsonl", slide_transcript_alignment)

    write_jsonl(ROOT / "formulas.jsonl", FORMULAS)
    write_jsonl(ROOT / "code_units.jsonl", CODE_UNITS)

    paper_rows = []
    for idx, title in enumerate(PAPER_MENTIONS, start=1):
        paper_rows.append(
            {
                "mention_id": f"paper_{idx:03d}",
                "paper_title": title,
                "source": "slides_or_readings",
                "lecture_relevance": "Supports a technique or comparison point mentioned in the lecture.",
            }
        )
    write_jsonl(ROOT / "paper_mentions.jsonl", paper_rows)

    low_confidence_rows = [
        {
            "unit_id": "transcript_lowconf_001",
            "start": "01:20:56,440",
            "end": "01:20:59,320",
            "text": "… methods that continue to scale with increased computation.",
            "reason": "Official caption contains a partially unclear phrase before the quoted Bitter Lesson sentence.",
            "action": "The note quotes only the stable semantic core and logs the uncertainty here.",
        }
    ]
    write_jsonl(ROOT / "low_confidence_spans.jsonl", low_confidence_rows)

    coverage_rows = [
        {
            "unit_id": "lec01_u0001",
            "source_refs": [{"source_type": "slide", "source_id": "intro_slide_005", "loc": {"page": 5}}, {"source_type": "slide", "source_id": "main_slide_002", "loc": {"page": 2}}],
            "kind": ["motivation", "history"],
            "importance": "required",
            "must_explain": ["为什么 2024-2025 年的 reasoning model 让 inference-time compute 成为核心议题", "LLM agent 与 environment feedback 的闭环关系"],
            "target_section": "1.1",
            "status": "covered",
            "covered_by": "1.1",
            "omission_reason": None,
        },
        {
            "unit_id": "lec01_u0002",
            "source_refs": [{"source_type": "slide", "source_id": "main_slide_003", "loc": {"page": 3}}, {"source_type": "transcript", "source_id": "transcript_000140", "loc": {"start": "00:06:20,000", "end": "00:07:10,000"}}],
            "kind": ["motivation", "caveat"],
            "importance": "required",
            "must_explain": ["性能为何随着更多 inference-time compute 提升", "性能提升伴随显著成本"],
            "target_section": "1.2",
            "status": "covered",
            "covered_by": "1.2",
            "omission_reason": None,
        },
        {
            "unit_id": "lec01_u0003",
            "source_refs": [{"source_type": "slide", "source_id": "main_slide_009", "loc": {"page": 9}}],
            "kind": ["definition", "motivation"],
            "importance": "required",
            "must_explain": ["standard prompting 的局限", "为什么只给 final format 不足以支撑 reasoning"],
            "target_section": "2.1",
            "status": "covered",
            "covered_by": "2.1",
            "omission_reason": None,
        },
        {
            "unit_id": "lec01_u0004",
            "source_refs": [{"source_type": "slide", "source_id": "main_slide_010", "loc": {"page": 10}}, {"source_type": "slide", "source_id": "main_slide_011", "loc": {"page": 11}}],
            "kind": ["definition", "algorithm"],
            "importance": "required",
            "must_explain": ["few-shot CoT 的工作机制", "模型规模为何影响 CoT 增益"],
            "target_section": "2.1",
            "status": "covered",
            "covered_by": "2.1",
            "omission_reason": None,
        },
        {
            "unit_id": "lec01_u0005",
            "source_refs": [{"source_type": "slide", "source_id": "main_slide_012", "loc": {"page": 12}}, {"source_type": "slide", "source_id": "main_slide_013", "loc": {"page": 13}}],
            "kind": ["algorithm", "example"],
            "importance": "required",
            "must_explain": ["zero-shot CoT 的触发方式", "它比 vanilla zero-shot 强在哪里、又弱在哪里"],
            "target_section": "2.1",
            "status": "covered",
            "covered_by": "2.1",
            "omission_reason": None,
        },
        {
            "unit_id": "lec01_u0006",
            "source_refs": [{"source_type": "slide", "source_id": "main_slide_015", "loc": {"page": 15}}, {"source_type": "slide", "source_id": "main_slide_016", "loc": {"page": 16}}],
            "kind": ["algorithm", "example"],
            "importance": "required",
            "must_explain": ["analogical prompting 的两阶段结构", "为什么 self-generated exemplars 比固定 exemplars 更灵活"],
            "target_section": "2.2",
            "status": "covered",
            "covered_by": "2.2",
            "omission_reason": None,
        },
        {
            "unit_id": "lec01_u0007",
            "source_refs": [{"source_type": "slide", "source_id": "main_slide_021", "loc": {"page": 21}}, {"source_type": "slide", "source_id": "main_slide_022", "loc": {"page": 22}}],
            "kind": ["experiment", "caveat"],
            "importance": "required",
            "must_explain": ["analogical prompting 的结果趋势", "强模型比弱模型更吃到该方法的收益"],
            "target_section": "2.2",
            "status": "covered",
            "covered_by": "2.2",
            "omission_reason": None,
        },
        {
            "unit_id": "lec01_u0008",
            "source_refs": [{"source_type": "slide", "source_id": "main_slide_024", "loc": {"page": 24}}, {"source_type": "slide", "source_id": "main_slide_025", "loc": {"page": 25}}],
            "kind": ["algorithm", "paper_summary"],
            "importance": "required",
            "must_explain": ["为什么 prompt engineering 可以被重写成 optimization loop", "optimizer 与 evaluator 的角色分工"],
            "target_section": "2.3",
            "status": "covered",
            "covered_by": "2.3",
            "omission_reason": None,
        },
        {
            "unit_id": "lec01_u0009",
            "source_refs": [{"source_type": "reading", "source_id": "reading_01", "loc": {"url": READINGS[0]["url"]}}],
            "kind": ["paper_summary", "experiment"],
            "importance": "required",
            "must_explain": ["OPRO reading 的核心问题、方法与实验结论", "它与 lecture 中 inference-time search 的联系"],
            "target_section": "2.3",
            "status": "covered",
            "covered_by": "2.3",
            "omission_reason": None,
        },
        {
            "unit_id": "lec01_u0010",
            "source_refs": [{"source_type": "slide", "source_id": "main_slide_030", "loc": {"page": 30}}, {"source_type": "slide", "source_id": "main_slide_031", "loc": {"page": 31}}],
            "kind": ["algorithm", "example"],
            "importance": "required",
            "must_explain": ["least-to-most 为什么能帮助 compositional generalization", "easy-to-hard decomposition 的含义"],
            "target_section": "2.4",
            "status": "covered",
            "covered_by": "2.4",
            "omission_reason": None,
        },
        {
            "unit_id": "lec01_u0011",
            "source_refs": [{"source_type": "slide", "source_id": "main_slide_033", "loc": {"page": 33}}, {"source_type": "slide", "source_id": "main_slide_035", "loc": {"page": 35}}],
            "kind": ["algorithm", "open_problem"],
            "importance": "required",
            "must_explain": ["dynamic least-to-most 与 Self-Discover 如何把 reasoning structure task-specific 化", "为什么固定 prompt 模板不够"],
            "target_section": "2.4",
            "status": "covered",
            "covered_by": "2.4",
            "omission_reason": None,
        },
        {
            "unit_id": "lec01_u0012",
            "source_refs": [{"source_type": "slide", "source_id": "main_slide_040", "loc": {"page": 40}}, {"source_type": "slide", "source_id": "main_slide_041", "loc": {"page": 41}}],
            "kind": ["algorithm", "definition"],
            "importance": "required",
            "must_explain": ["self-consistency 的投票机制", "为什么它 marginalize reasoning paths 而不是比较单一路径概率"],
            "target_section": "3.1",
            "status": "covered",
            "covered_by": "3.1",
            "omission_reason": None,
        },
        {
            "unit_id": "lec01_u0013",
            "source_refs": [{"source_type": "slide", "source_id": "main_slide_042", "loc": {"page": 42}}, {"source_type": "slide", "source_id": "main_slide_043", "loc": {"page": 43}}, {"source_type": "slide", "source_id": "main_slide_044", "loc": {"page": 44}}],
            "kind": ["experiment", "caveat"],
            "importance": "required",
            "must_explain": ["更多 samples 与更多 diversity 为什么重要", "为什么 beam search 不等价于 self-consistency"],
            "target_section": "3.1",
            "status": "covered",
            "covered_by": "3.1",
            "omission_reason": None,
        },
        {
            "unit_id": "lec01_u0014",
            "source_refs": [{"source_type": "slide", "source_id": "main_slide_045", "loc": {"page": 45}}, {"source_type": "slide", "source_id": "main_slide_047", "loc": {"page": 47}}],
            "kind": ["code", "example"],
            "importance": "required",
            "must_explain": ["AlphaCode 的 execution-consistency clustering", "为什么代码任务能借助 test execution 获得更强选择信号"],
            "target_section": "3.1",
            "status": "covered",
            "covered_by": "3.1",
            "omission_reason": None,
        },
        {
            "unit_id": "lec01_u0015",
            "source_refs": [{"source_type": "slide", "source_id": "main_slide_050", "loc": {"page": 50}}, {"source_type": "slide", "source_id": "main_slide_051", "loc": {"page": 51}}],
            "kind": ["algorithm", "caveat"],
            "importance": "required",
            "must_explain": ["Universal Self-Consistency 解决了什么问题", "为什么它受 long-context 能力约束"],
            "target_section": "3.1",
            "status": "covered",
            "covered_by": "3.1",
            "omission_reason": None,
        },
        {
            "unit_id": "lec01_u0016",
            "source_refs": [{"source_type": "slide", "source_id": "main_slide_052", "loc": {"page": 52}}, {"source_type": "slide", "source_id": "main_slide_053", "loc": {"page": 53}}],
            "kind": ["definition", "algorithm"],
            "importance": "required",
            "must_explain": ["ORM 与 PRM 的区别", "为什么强 verifier 可以超过 simple consistency voting"],
            "target_section": "3.2",
            "status": "covered",
            "covered_by": "3.2",
            "omission_reason": None,
        },
        {
            "unit_id": "lec01_u0017",
            "source_refs": [{"source_type": "slide", "source_id": "main_slide_054", "loc": {"page": 54}}, {"source_type": "slide", "source_id": "main_slide_055", "loc": {"page": 55}}, {"source_type": "slide", "source_id": "main_slide_057", "loc": {"page": 57}}],
            "kind": ["algorithm", "example"],
            "importance": "required",
            "must_explain": ["Tree of Thoughts 如何在 partial solutions 上搜索", "为什么 partial-state evaluation 比 full-response reranking 更细粒度"],
            "target_section": "3.2",
            "status": "covered",
            "covered_by": "3.2",
            "omission_reason": None,
        },
        {
            "unit_id": "lec01_u0018",
            "source_refs": [{"source_type": "slide", "source_id": "main_slide_061", "loc": {"page": 61}}, {"source_type": "slide", "source_id": "main_slide_062", "loc": {"page": 62}}],
            "kind": ["algorithm", "caveat"],
            "importance": "required",
            "must_explain": ["Reflexion 与 Self-Refine 的一般模板", "为什么这类方法依赖外部评价或 heuristic feedback"],
            "target_section": "4.1",
            "status": "covered",
            "covered_by": "4.1",
            "omission_reason": None,
        },
        {
            "unit_id": "lec01_u0019",
            "source_refs": [{"source_type": "slide", "source_id": "main_slide_063", "loc": {"page": 63}}, {"source_type": "slide", "source_id": "main_slide_064", "loc": {"page": 64}}, {"source_type": "slide", "source_id": "main_slide_065", "loc": {"page": 65}}, {"source_type": "reading", "source_id": "reading_03", "loc": {"url": READINGS[2]["url"]}}],
            "kind": ["code", "paper_summary", "example"],
            "importance": "required",
            "must_explain": ["为什么 code generation 是 self-improvement 的天然试验场", "execution results、trace 和 code explanation 分别提供什么反馈"],
            "target_section": "4.1",
            "status": "covered",
            "covered_by": "4.1",
            "omission_reason": None,
        },
        {
            "unit_id": "lec01_u0020",
            "source_refs": [{"source_type": "slide", "source_id": "main_slide_066", "loc": {"page": 66}}, {"source_type": "slide", "source_id": "main_slide_067", "loc": {"page": 67}}, {"source_type": "slide", "source_id": "main_slide_068", "loc": {"page": 68}}, {"source_type": "slide", "source_id": "main_slide_069", "loc": {"page": 69}}, {"source_type": "reading", "source_id": "reading_02", "loc": {"url": READINGS[1]["url"]}}],
            "kind": ["caveat", "paper_summary"],
            "importance": "required",
            "must_explain": ["没有 oracle/external feedback 时 self-correction 为什么常常失败", "为什么 multi-agent debate 也无法凭空产生可靠 evaluator"],
            "target_section": "4.2",
            "status": "covered",
            "covered_by": "4.2",
            "omission_reason": None,
        },
        {
            "unit_id": "lec01_u0021",
            "source_refs": [{"source_type": "slide", "source_id": "main_slide_070", "loc": {"page": 70}}, {"source_type": "slide", "source_id": "main_slide_071", "loc": {"page": 71}}],
            "kind": ["open_problem", "caveat"],
            "importance": "required",
            "must_explain": ["如何在多样采样、串行修正和模型规模之间分配 budget", "为什么最优策略依赖模型与任务"],
            "target_section": "5.1",
            "status": "covered",
            "covered_by": "5.1",
            "omission_reason": None,
        },
        {
            "unit_id": "lec01_u0022",
            "source_refs": [{"source_type": "slide", "source_id": "main_slide_073", "loc": {"page": 73}}],
            "kind": ["history", "open_problem"],
            "importance": "required",
            "must_explain": ["Bitter Lesson 在 lecture 中被怎样重新解释为 reasoning-method design principle", "为什么要偏好可扩展的通用方法"],
            "target_section": "5.2",
            "status": "covered",
            "covered_by": "5.2",
            "omission_reason": None,
        },
        {
            "unit_id": "lec01_u0023",
            "source_refs": [{"source_type": "slide", "source_id": "intro_slide_011", "loc": {"page": 11}}, {"source_type": "slide", "source_id": "intro_slide_012", "loc": {"page": 12}}, {"source_type": "slide", "source_id": "intro_slide_013", "loc": {"page": 13}}, {"source_type": "slide", "source_id": "intro_slide_014", "loc": {"page": 14}}, {"source_type": "slide", "source_id": "intro_slide_015", "loc": {"page": 15}}, {"source_type": "slide", "source_id": "intro_slide_016", "loc": {"page": 16}}],
            "kind": ["transition"],
            "importance": "recommended",
            "must_explain": ["课程作业、评分和时间线属于 lecture 录制中的行政信息"],
            "target_section": "appendix",
            "status": "omitted",
            "covered_by": None,
            "omission_reason": "Course logistics are not central to the technical content of this lecture note and are logged explicitly instead of being expanded in the textbook body.",
        },
        {
            "unit_id": "lec01_u0024",
            "source_refs": [{"source_type": "transcript", "source_id": "transcript_000001", "loc": {"start": "00:00:00,000", "end": "00:00:21,570"}}],
            "kind": ["transition"],
            "importance": "optional",
            "must_explain": ["开场问候和 teaching team 介绍"],
            "target_section": "appendix",
            "status": "omitted",
            "covered_by": None,
            "omission_reason": "Opening greetings and housekeeping are logged but not expanded in the textbook body.",
        },
    ]
    write_jsonl(ROOT / "coverage_units.jsonl", coverage_rows)

    omission_rows = [
        {
            "unit_id": "lec01_u0023",
            "reason": "course_logistics",
            "user_visible_note": "课程作业、评分与时间线属于行政信息，未并入本讲技术主体。",
        },
        {
            "unit_id": "lec01_u0024",
            "reason": "non_teaching_opening",
            "user_visible_note": "开场寒暄与教学团队介绍保留在 source artifacts 中，但不展开写入技术讲义。",
        },
        {
            "unit_id": "transcript_lowconf_001",
            "reason": "caption_uncertainty",
            "user_visible_note": "结尾引用 Bitter Lesson 时字幕有一小段短语不清晰，讲义只保留稳定语义并在 low_confidence_spans.jsonl 中记录。",
        },
    ]
    write_jsonl(ROOT / "omission_log.jsonl", omission_rows)

    segment_plan_lines = [
        "# Segment Plan",
        "",
        "本讲按照“问题动机 -> 单轨推理 -> 宽搜索 -> 深度自改进 -> 预算分配原则”的顺序组织。",
        "",
    ]
    contracts_dir = ROOT / "segment_contracts"
    contracts_dir.mkdir(exist_ok=True)
    for seg in SEGMENTS:
        segment_plan_lines.append(f"- {seg['segment_id']}: {seg['title']} ({seg['start']} -- {seg['end']}) -> {seg['target_section']}")
        contract = [
            f"# {seg['segment_id']} Contract",
            "",
            "Source range:",
            f"- transcript: {seg['start']} -- {seg['end']}",
            f"- slide refs: {', '.join(f'{deck}:{page}' for deck, page in seg['slide_refs'])}",
            "",
            "Must-cover units:",
        ]
        for row in coverage_rows:
            if row["target_section"].startswith(seg["target_section"]):
                contract.append(f"- {row['unit_id']}")
        contract.extend(
            [
                "",
                "Expected section/subsection:",
                f"- {seg['target_section']}",
                "",
                "Required figures:",
            ]
        )
        contract.extend([f"- {item}" for item in seg["required_figures"]] or ["- none"])
        contract.extend(["", "Required formulas:"])
        contract.extend([f"- {item}" for item in seg["required_formulas"]] or ["- none"])
        contract.extend(["", "Required code snippets:"])
        contract.extend([f"- {item}" for item in seg["required_code"]] or ["- none"])
        contract.extend(
            [
                "",
                "Evaluator checks:",
                "- all required units are concretely explained, not merely name-dropped",
                "- dense slide content is unpacked layer by layer",
                "- any figure used in this segment has provenance in figure_manifest.json",
                "",
                "Done definition:",
                "- the section is textbook-style and self-contained",
                "- formulas explain symbols",
                "- algorithms explain inputs, outputs, and failure modes",
            ]
        )
        (contracts_dir / f"{seg['segment_id']}_contract.md").write_text("\n".join(contract) + "\n", encoding="utf-8")
    (ROOT / "segment_plan.md").write_text("\n".join(segment_plan_lines) + "\n", encoding="utf-8")

    figure_plan_rows = []
    figure_manifest_rows = []
    for fig in FIGURES:
        asset_path = render_figure(fig)
        entry = {
            "figure_id": fig["figure_id"],
            "source_type": "slide",
            "source_ref": {
                "url": INTRO_SLIDES_URL if fig["deck"] == "intro" else MAIN_SLIDES_URL,
                "page": fig["page"],
                "timestamp": None,
            },
            "asset_path": asset_path,
            "used_for": fig["used_for"],
            "target_section": fig["target_section"],
            "caption_draft": fig["caption"],
            "source_unit_ids": [
                row["unit_id"]
                for row in coverage_rows
                if any(
                    ref["source_id"] == f"{'intro_slide' if fig['deck'] == 'intro' else 'main_slide'}_{fig['page']:03d}"
                    for ref in row["source_refs"]
                    if ref["source_type"] == "slide"
                )
            ],
        }
        figure_plan_rows.append(entry)
        figure_manifest_rows.append(
            {
                "figure_id": fig["figure_id"],
                "source_ref": entry["source_ref"],
                "asset_path": asset_path,
                "caption": fig["caption"],
                "used_in_section": fig["target_section"],
                "source_unit_ids": entry["source_unit_ids"],
                "provenance_type": "slide",
                "time_provenance": None,
            }
        )
    write_jsonl(ROOT / "figure_plan.jsonl", figure_plan_rows)
    write_json(ROOT / "figure_manifest.json", figure_manifest_rows)

    source_manifest = {
        "course_id": "cs294_194_280_sp25_agents",
        "lecture_id": "L01",
        "lecture_slug": "lec01_inference_time_reasoning",
        "title": "Inference-Time Techniques for LLM Reasoning",
        "speaker": "Xinyun Chen",
        "origin_url": VIDEO_URL,
        "course_page": COURSE_PAGE,
        "sources": [
            {"source_id": "course_page", "source_type": "course_page", "origin_url": COURSE_PAGE, "local_path": None, "required_for_coverage": True, "status": "available", "notes": "Official Berkeley RDI course page."},
            {"source_id": "recording_info", "source_type": "youtube_metadata", "origin_url": VIDEO_URL, "local_path": "recording.info.json", "required_for_coverage": True, "status": "available", "notes": "yt-dlp metadata JSON."},
            {"source_id": "cover_image", "source_type": "youtube_thumbnail", "origin_url": info.get("thumbnail"), "local_path": "cover.jpg", "required_for_coverage": True, "status": "available", "notes": "Converted from downloaded YouTube thumbnail."},
            {"source_id": "transcript_raw", "source_type": "youtube_caption", "origin_url": VIDEO_URL, "local_path": "transcript_raw.srt", "required_for_coverage": True, "status": "available", "notes": "Canonical subtitle track copied from recording.en-j3PyPqV-e1s.srt."},
            {"source_id": "transcript_jsonl", "source_type": "structured_transcript_evidence", "origin_url": VIDEO_URL, "local_path": "transcript.jsonl", "required_for_coverage": True, "status": "available", "notes": "Timestamped lecture spans for harness consumption."},
            {"source_id": "slides_intro", "source_type": "official_slide_pdf", "origin_url": INTRO_SLIDES_URL, "local_path": "slides_intro.pdf", "required_for_coverage": True, "status": "available", "notes": "Course intro slide deck used for lecture framing."},
            {"source_id": "slides_main", "source_type": "official_slide_pdf", "origin_url": MAIN_SLIDES_URL, "local_path": "slides_main.pdf", "required_for_coverage": True, "status": "available", "notes": "Main lecture slide deck."},
            {"source_id": "slides_pdf", "source_type": "canonical_slide_pdf", "origin_url": MAIN_SLIDES_URL, "local_path": "slides.pdf", "required_for_coverage": True, "status": "available", "notes": "Canonical lecture slide PDF copy."},
            {"source_id": "slides_jsonl", "source_type": "structured_slide_evidence", "origin_url": None, "local_path": "slides.jsonl", "required_for_coverage": True, "status": "available", "notes": "Per-page slide text extraction for both intro and main decks."},
            {"source_id": "readings_manifest", "source_type": "supplemental_readings", "origin_url": COURSE_PAGE, "local_path": "readings_manifest.json", "required_for_coverage": True, "status": "available", "notes": "Lecture readings plus grounded summaries from the provided URLs."},
        ],
    }
    write_json(ROOT / "source_manifest.json", source_manifest)

    acquisition_log = [
        "# Source Acquisition Log",
        "",
        f"- Recording URL: {VIDEO_URL}",
        f"- Official course page: {COURSE_PAGE}",
        f"- Main slide deck downloaded to `slides_main.pdf` from `{MAIN_SLIDES_URL}`.",
        f"- Intro slide deck downloaded to `slides_intro.pdf` from `{INTRO_SLIDES_URL}`.",
        "- Canonical subtitle track: `recording.en-j3PyPqV-e1s.srt` -> `transcript_raw.srt`.",
        "- Auto-caption tracks were preserved locally for debugging but are not the canonical evidence layer.",
        "- Readings were recorded in `readings_manifest.json` using the URLs provided in the lecture entry.",
        "- No video frames were required for this pilot because the slide decks already contain the key explanatory visuals.",
    ]
    (ROOT / "source_acquisition_log.md").write_text("\n".join(acquisition_log) + "\n", encoding="utf-8")

    (ROOT / "repair_log.jsonl").write_text("", encoding="utf-8")


if __name__ == "__main__":
    main()
