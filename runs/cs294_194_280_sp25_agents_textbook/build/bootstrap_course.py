#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


RUN_ROOT = Path(__file__).resolve().parents[1]
BOOK_DIR = RUN_ROOT / "book"
DOCS_DIR = RUN_ROOT / "docs"
LECTURES_DIR = RUN_ROOT / "lectures"
BUILD_DIR = RUN_ROOT / "build"
COURSE_TITLE = "CS294/194-280: Advanced Large Language Model Agents, Spring 2025"
COURSE_SLUG = "cs294_194_280_sp25_agents_textbook"
COURSE_PAGE = "https://rdi.berkeley.edu/adv-llm-agents/sp25"

LECTURES = [
    {
        "index": 1,
        "lecture_id": "L01",
        "slug": "lec01_inference_time_reasoning",
        "date": "2025-01-27",
        "display_date": "Jan 27",
        "title": "Inference-Time Techniques for LLM Reasoning",
        "speaker": "Xinyun Chen",
        "affiliation": "Google DeepMind",
        "recording_url": "https://www.youtube.com/live/g0Dwtf3BH-0",
        "special_time_notes": None,
        "slide_urls": [
            {
                "label": "Intro",
                "url": "https://rdi.berkeley.edu/adv-llm-agents/slides/llm-agents-berkeley-intro-sp25.pdf",
            },
            {
                "label": "Slides",
                "url": "https://rdi.berkeley.edu/adv-llm-agents/slides/inference_time_techniques_lecture_sp25.pdf",
            },
        ],
        "readings": [
            {
                "title": "Large Language Models as Optimizers",
                "url": "https://arxiv.org/abs/2309.03409",
                "type": "paper",
            },
            {
                "title": "Large Language Models Cannot Self-Correct Reasoning Yet",
                "url": "https://arxiv.org/abs/2310.01798",
                "type": "paper",
            },
            {
                "title": "Teaching Large Language Models to Self-Debug",
                "url": "https://arxiv.org/abs/2304.05128",
                "type": "paper",
            },
        ],
        "topics": [
            "inference-time computation",
            "search and verification",
            "self-correction and self-debugging",
            "reasoning-time scaling",
        ],
    },
    {
        "index": 2,
        "lecture_id": "L02",
        "slug": "lec02_learning_to_reason",
        "date": "2025-02-03",
        "display_date": "Feb 3",
        "title": "Learning to reason with LLMs",
        "speaker": "Jason Weston",
        "affiliation": "Meta",
        "recording_url": "https://www.youtube.com/live/_MNlLhU33H0",
        "special_time_notes": None,
        "slide_urls": [
            {
                "label": "Slides",
                "url": "https://rdi.berkeley.edu/adv-llm-agents/slides/Jason-Weston-Reasoning-Alignment-Berkeley-Talk.pdf",
            }
        ],
        "readings": [
            {
                "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
                "url": "https://arxiv.org/abs/2305.18290",
                "type": "paper",
            },
            {
                "title": "Iterative Reasoning Preference Optimization",
                "url": "https://arxiv.org/abs/2404.19733",
                "type": "paper",
            },
            {
                "title": "Chain-of-Verification Reduces Hallucination in Large Language Models",
                "url": "https://arxiv.org/abs/2309.11495",
                "type": "paper",
            },
        ],
        "topics": [
            "learning to reason",
            "preference optimization",
            "verification-guided reasoning",
            "alignment for reasoning behavior",
        ],
    },
    {
        "index": 3,
        "lecture_id": "L03",
        "slug": "lec03_reasoning_memory_planning",
        "date": "2025-02-10",
        "display_date": "Feb 10",
        "title": "On Reasoning, Memory, and Planning of Language Agents",
        "speaker": "Yu Su",
        "affiliation": "Ohio State University",
        "recording_url": "https://www.youtube.com/live/zvI4UN2_i-w",
        "special_time_notes": None,
        "slide_urls": [
            {
                "label": "Slides",
                "url": "https://rdi.berkeley.edu/adv-llm-agents/slides/language_agents_YuSu_Berkeley.pdf",
            }
        ],
        "readings": [
            {
                "title": "Grokked Transformers are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization",
                "url": "https://arxiv.org/abs/2405.15071",
                "type": "paper",
            },
            {
                "title": "HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models",
                "url": "https://arxiv.org/abs/2405.14831",
                "type": "paper",
            },
            {
                "title": "Is Your LLM Secretly a World Model of the Internet? Model-Based Planning for Web Agents",
                "url": "https://arxiv.org/abs/2411.06559",
                "type": "paper",
            },
        ],
        "topics": [
            "reasoning",
            "memory systems",
            "planning",
            "web agents and world models",
        ],
    },
    {
        "index": 4,
        "lecture_id": "L04",
        "slug": "lec04_open_training_recipes_reasoning",
        "date": "2025-02-24",
        "display_date": "Feb 24",
        "title": "Open Training Recipes for Reasoning in Language Models",
        "speaker": "Hanna Hajishirzi",
        "affiliation": "University of Washington",
        "recording_url": "https://www.youtube.com/live/cMiu3A7YBks",
        "special_time_notes": None,
        "slide_urls": [
            {
                "label": "Slides",
                "url": "https://rdi.berkeley.edu/adv-llm-agents/slides/OLMo-Tulu-Reasoning-Hanna.pdf",
            }
        ],
        "readings": [
            {
                "title": "Tulu 3: Pushing Frontiers in Open Language Model Post-Training",
                "url": "https://arxiv.org/abs/2411.15124",
                "type": "paper",
            },
            {
                "title": "Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preference Feedback",
                "url": "https://arxiv.org/abs/2406.09279",
                "type": "paper",
            },
            {
                "title": "OpenScholar: Synthesizing Scientific Literature with Retrieval-augmented LMs",
                "url": "https://arxiv.org/abs/2411.14199",
                "type": "paper",
            },
        ],
        "topics": [
            "open post-training recipes",
            "reasoning post-training",
            "DPO and PPO",
            "open model ecosystems",
        ],
    },
    {
        "index": 5,
        "lecture_id": "L05",
        "slug": "lec05_coding_agents_vulnerability_detection",
        "date": "2025-03-03",
        "display_date": "Mar 3",
        "title": "Coding Agents and AI for Vulnerability Detection",
        "speaker": "Charles Sutton",
        "affiliation": "Google DeepMind",
        "recording_url": "https://www.youtube.com/live/JCk6qJtaCSU",
        "special_time_notes": None,
        "slide_urls": [
            {
                "label": "Slides",
                "url": "https://rdi.berkeley.edu/adv-llm-agents/slides/Code%20Agents%20and%20AI%20for%20Vulnerability%20Detection.pdf",
            }
        ],
        "readings": [
            {
                "title": "Interactive Tools Substantially Assist LM Agents in Finding Security Vulnerabilities",
                "url": "https://arxiv.org/abs/2409.16165",
                "type": "paper",
            }
        ],
        "topics": [
            "coding agents",
            "tool use for software engineering",
            "AI-assisted vulnerability detection",
            "security failure analysis",
        ],
    },
    {
        "index": 6,
        "lecture_id": "L06",
        "slug": "lec06_multimodal_autonomous_agents",
        "date": "2025-03-10",
        "display_date": "Mar 10",
        "title": "Multimodal Autonomous AI Agents",
        "speaker": "Ruslan Salakhutdinov",
        "affiliation": "CMU / Meta",
        "recording_url": "https://www.youtube.com/live/RPINOYM12RU",
        "special_time_notes": None,
        "slide_urls": [
            {
                "label": "Slides",
                "url": "https://rdi.berkeley.edu/adv-llm-agents/slides/ruslan-multimodal.pdf",
            }
        ],
        "readings": [
            {
                "title": "Mind2Web: Towards a Generalist Agent for the Web",
                "url": "https://arxiv.org/abs/2306.06070",
                "type": "paper",
            },
            {
                "title": "WebArena: A Realistic Web Environment for Building Autonomous Agents",
                "url": "https://arxiv.org/abs/2307.13854",
                "type": "paper",
            },
        ],
        "topics": [
            "multimodal autonomous agents",
            "web agents",
            "interactive environments",
            "perception-action feedback",
        ],
    },
    {
        "index": 7,
        "lecture_id": "L07",
        "slug": "lec07_multimodal_agents_perception_to_action",
        "date": "2025-03-17",
        "display_date": "Mar 17",
        "title": "Multimodal Agents – From Perception to Action",
        "speaker": "Caiming Xiong",
        "affiliation": "Salesforce AI Research",
        "recording_url": "https://www.youtube.com/live/n__Tim8K2IY",
        "special_time_notes": None,
        "slide_urls": [
            {
                "label": "Slides",
                "url": "https://rdi.berkeley.edu/adv-llm-agents/slides/Multimodal_Agent_caiming.pdf",
            }
        ],
        "readings": [
            {
                "title": "OSWORLD: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments",
                "url": "https://arxiv.org/pdf/2404.07972",
                "type": "paper",
            },
            {
                "title": "AGUVIS: Unified Pure Vision Agents For Autonomous GUI Interaction",
                "url": "https://arxiv.org/pdf/2412.04454",
                "type": "paper",
            },
        ],
        "topics": [
            "multimodal perception",
            "GUI grounding",
            "OS and GUI benchmarks",
            "perception-to-action loops",
        ],
    },
    {
        "index": 8,
        "lecture_id": "L08",
        "slug": "lec08_alphaproof_formal_mathematics",
        "date": "2025-03-31",
        "display_date": "Mar 31",
        "title": "AlphaProof: when reinforcement learning meets formal mathematics",
        "speaker": "Thomas Hubert",
        "affiliation": "Google DeepMind",
        "recording_url": "https://www.youtube.com/live/3gaEMscOMAU",
        "special_time_notes": "10am-noon PT",
        "slide_urls": [
            {
                "label": "Slides",
                "url": "https://rdi.berkeley.edu/adv-llm-agents/slides/alphaproof.pdf",
            }
        ],
        "readings": [
            {
                "title": "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm",
                "url": "https://arxiv.org/pdf/1712.01815",
                "type": "paper",
            },
            {
                "title": "The Future of Mathematics?",
                "url": "https://www.youtube.com/watch?v=Dp-mQ3HxgDE",
                "type": "video",
            },
        ],
        "topics": [
            "reinforcement learning for formal mathematics",
            "proof search",
            "AlphaProof",
            "formal reasoning environments",
        ],
    },
    {
        "index": 9,
        "lecture_id": "L09",
        "slug": "lec09_autoformalization_theorem_proving",
        "date": "2025-04-07",
        "display_date": "Apr 7",
        "title": "Language models for autoformalization and theorem proving",
        "speaker": "Kaiyu Yang",
        "affiliation": "Meta FAIR",
        "recording_url": "https://www.youtube.com/live/cLhWEyMQ4mQ",
        "special_time_notes": None,
        "slide_urls": [
            {
                "label": "Slides",
                "url": "https://rdi.berkeley.edu/adv-llm-agents/slides/mathverification.pdf",
            }
        ],
        "readings": [
            {
                "title": "LeanDojo: Theorem Proving with Retrieval-Augmented Language Models",
                "url": "https://arxiv.org/abs/2306.15626",
                "type": "paper",
            },
            {
                "title": "Autoformalization with Large Language Models",
                "url": "https://arxiv.org/abs/2205.12615",
                "type": "paper",
            },
            {
                "title": "Autoformalizing Euclidean Geometry",
                "url": "https://arxiv.org/abs/2405.17216",
                "type": "paper",
            },
        ],
        "topics": [
            "autoformalization",
            "retrieval-augmented theorem proving",
            "formal specification",
            "Lean ecosystems",
        ],
    },
    {
        "index": 10,
        "lecture_id": "L10",
        "slug": "lec10_advanced_theorem_proving",
        "date": "2025-04-14",
        "display_date": "Apr 14",
        "title": "Advanced topics in theorem proving",
        "speaker": "Sean Welleck",
        "affiliation": "CMU",
        "recording_url": "https://www.youtube.com/live/Gy5Nm17l9oo",
        "special_time_notes": None,
        "slide_urls": [
            {
                "label": "Slides",
                "url": "https://rdi.berkeley.edu/adv-llm-agents/slides/welleck2025_berkeley_bridging.pdf",
            }
        ],
        "readings": [
            {
                "title": "Draft, Sketch, and Prove: Guiding Formal Theorem Provers with Informal Proofs",
                "url": "https://arxiv.org/abs/2210.12283",
                "type": "paper",
            },
            {
                "title": "miniCTX: Neural Theorem Proving with Long-Contexts",
                "url": "https://www.arxiv.org/pdf/2408.03350",
                "type": "paper",
            },
            {
                "title": "Lean-STaR: Learning to Interleave Thinking and Proving",
                "url": "https://arxiv.org/abs/2407.10040",
                "type": "paper",
            },
            {
                "title": "ImProver: Agent-Based Automated Proof Optimization",
                "url": "https://arxiv.org/abs/2410.04753",
                "type": "paper",
            },
        ],
        "topics": [
            "advanced proof search",
            "interleaving thinking and proving",
            "long-context theorem proving",
            "proof optimization agents",
        ],
    },
    {
        "index": 11,
        "lecture_id": "L11",
        "slug": "lec11_abstraction_discovery_llm_agents",
        "date": "2025-04-21",
        "display_date": "Apr 21",
        "title": "Abstraction and Discovery with Large Language Model Agents",
        "speaker": "Swarat Chaudhuri",
        "affiliation": "UT Austin",
        "recording_url": "https://www.youtube.com/live/IHc0TEMrEdY",
        "special_time_notes": "10am-noon PT",
        "slide_urls": [
            {
                "label": "Slides",
                "url": "https://rdi.berkeley.edu/adv-llm-agents/slides/swarat.pdf",
            }
        ],
        "readings": [
            {
                "title": "An In-Context Learning Agent for Formal Theorem-Proving",
                "url": "https://arxiv.org/abs/2310.04353",
                "type": "paper",
            },
            {
                "title": "Symbolic Regression with a Learned Concept Library",
                "url": "https://arxiv.org/abs/2409.09359",
                "type": "paper",
            },
        ],
        "topics": [
            "abstraction",
            "discovery",
            "concept library learning",
            "LLM agents for scientific reasoning",
        ],
    },
    {
        "index": 12,
        "lecture_id": "L12",
        "slug": "lec12_safe_secure_agentic_ai",
        "date": "2025-04-28",
        "display_date": "Apr 28",
        "title": "Towards building safe and secure agentic AI",
        "speaker": "Dawn Song",
        "affiliation": "UC Berkeley",
        "recording_url": "https://www.youtube.com/live/ti6yPE2VPZc",
        "special_time_notes": None,
        "slide_urls": [
            {
                "label": "Slides",
                "url": "https://rdi.berkeley.edu/adv-llm-agents/slides/dawn-agentic-ai.pdf",
            }
        ],
        "readings": [
            {
                "title": "DataSentinel: A Game-Theoretic Detection of Prompt Injection Attacks",
                "url": "https://arxiv.org/abs/2504.11358",
                "type": "paper",
            },
            {
                "title": "AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases",
                "url": "https://arxiv.org/abs/2407.12784",
                "type": "paper",
            },
            {
                "title": "Progent: Programmable Privilege Control for LLM Agents",
                "url": "https://arxiv.org/html/2504.11703v1",
                "type": "paper",
            },
        ],
        "topics": [
            "agentic AI safety",
            "security",
            "prompt injection",
            "privilege control",
        ],
    },
]


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(payload + ("\n" if payload else ""))


def maybe_write_text(path: Path, content: str) -> None:
    if not path.exists():
        path.write_text(content)


def maybe_write_json(path: Path, payload: object) -> None:
    if not path.exists():
        write_json(path, payload)


def maybe_write_jsonl(path: Path, rows: list[dict]) -> None:
    if not path.exists():
        write_jsonl(path, rows)


def ensure_dirs() -> None:
    for path in [
        RUN_ROOT,
        DOCS_DIR,
        LECTURES_DIR,
        BOOK_DIR,
        BOOK_DIR / "chapters",
        BOOK_DIR / "frontmatter",
        BOOK_DIR / "appendices",
        BOOK_DIR / "figures",
        RUN_ROOT / "eval_reports",
        RUN_ROOT / "repair_logs",
    ]:
        path.mkdir(parents=True, exist_ok=True)
    for lecture in LECTURES:
        lecture_dir = LECTURES_DIR / lecture["slug"]
        for path in [lecture_dir, lecture_dir / "contracts", lecture_dir / "eval_reports", lecture_dir / "figures"]:
            path.mkdir(parents=True, exist_ok=True)


def lecture_meta(lecture: dict) -> dict:
    return {
        "course_id": COURSE_SLUG,
        "playlist_index": lecture["index"],
        "lecture_id": lecture["lecture_id"],
        "slug": lecture["slug"],
        "date": lecture["date"],
        "title": lecture["title"],
        "title_short": lecture["title"],
        "speaker": lecture["speaker"],
        "affiliation": lecture["affiliation"],
        "recording_url": lecture["recording_url"],
        "slide_urls": lecture["slide_urls"],
        "readings": lecture["readings"],
        "special_time_notes": lecture["special_time_notes"],
        "topics": lecture["topics"],
        "course_mode": True,
        "segmentation_required": True,
    }


def build_course_source_manifest() -> dict:
    return {
        "course_id": COURSE_SLUG,
        "course_title": COURSE_TITLE,
        "term": "Spring 2025",
        "institution": "UC Berkeley / Berkeley RDI",
        "official_course_page": COURSE_PAGE,
        "last_synced": "2026-04-15",
        "source_policy": {
            "coverage_first": True,
            "source_grounded": True,
            "repository_as_record_system": True,
            "evaluator_gated_delivery": True,
        },
        "differences_from_initial_request": [],
        "course_level_sources": [
            {
                "source_id": "official_course_page",
                "source_type": "course_page",
                "url": COURSE_PAGE,
                "required": True,
                "status": "available",
            }
        ],
        "lectures": [
            {
                "lecture_id": lecture["lecture_id"],
                "lecture_slug": lecture["slug"],
                "date": lecture["date"],
                "title": lecture["title"],
                "speaker": lecture["speaker"],
                "special_time_notes": lecture["special_time_notes"],
                "recording": {
                    "url": lecture["recording_url"],
                    "required": True,
                    "status": "pending_local_acquisition",
                },
                "slides": [
                    {
                        "label": slide["label"],
                        "url": slide["url"],
                        "required": True,
                        "status": "pending_local_acquisition",
                    }
                    for slide in lecture["slide_urls"]
                ],
                "readings": lecture["readings"],
            }
            for lecture in LECTURES
        ],
    }


def build_course_coverage_index() -> list[dict]:
    rows = []
    for lecture in LECTURES:
        lecture_dir = LECTURES_DIR / lecture["slug"]
        validator_status = "pending"
        if (lecture_dir / "lecture_quality_report.md").exists():
            validator_status = "pass"
        elif any((lecture_dir / "eval_reports").glob("pass_*.json")):
            validator_status = "evaluator_pass_pending_validator"
        elif any(lecture_dir.iterdir()):
            validator_status = "in_progress"
        rows.append(
            {
                "lecture_id": lecture["lecture_id"],
                "lecture_slug": lecture["slug"],
                "title": lecture["title"],
                "coverage_status": "planned" if validator_status == "pending" else "in_progress",
                "validator_status": validator_status,
                "book_status": "not_merged",
                "required_sources": ["recording", "slides", "readings"],
            }
        )
    return rows


def render_readme() -> str:
    lecture_lines = "\n".join(
        [
            f"- `{lecture['lecture_id']}` `{lecture['display_date']}` `{lecture['title']}` / {lecture['speaker']}"
            for lecture in LECTURES
        ]
    )
    return dedent(
        f"""\
        # {COURSE_TITLE}

        该 run 使用 harness-managed workflow，把 Berkeley RDI 的整门公开课转换为教材级中文讲义。

        ## 课程来源

        - Official course page: {COURSE_PAGE}
        - Institution: UC Berkeley / Berkeley RDI
        - Public lectures discovered: {len(LECTURES)}

        ## 当前讲次列表

        {lecture_lines}

        ## 生成流程

        1. `build/bootstrap_course.py` 固化课程 spec、docs、lecture workspaces 和初始 manifests。
        2. 每讲按 `source acquisition -> planner -> coverage extractor -> writer -> figure agent -> evaluator -> repair -> validator` 执行。
        3. 通过 `build/validate_lecture.py` 逐讲 gate。
        4. 全部讲次通过后，由 `build/merge_book.py` 与 `build/compile.sh` 组装整本教材。
        5. 通过 `build/validate_textbook.py` 做全书 gate。

        ## Agents 分工

        - Course planner agents 负责课程页、讲次、sources 与 docs。
        - Lecture-level agents 负责每讲的 source curation、coverage、writing、evaluation 与 repair。
        - Book-level agents 负责跨讲一致性、术语表、习题与总编排。

        ## Validator 使用方法

        - 单讲校验：`python3 build/validate_lecture.py lec01`
        - 全部讲次校验：`python3 build/validate_lecture.py`
        - 全书校验：`python3 build/validate_textbook.py`
        - 编译全书：`./build/compile.sh`

        ## Known limitations

        - YouTube 自动字幕可能有重复 span、断词或专有名词错误，需要在 `low_confidence_spans.jsonl` 和 omission artifacts 中显式记录。
        - 某些 slides/readings 可能在后续站点更新后失效；source manifest 会记录当次抓取状态。
        - 在单讲未通过 evaluator/validator 之前，不应把该讲并入 `book/`。

        ## Unresolved omissions

        - 初始 bootstrap 阶段仅创建课程级 scaffold，不表示任何讲次已经完成。
        - 当前所有未完成讲次都必须在 `COURSE_OMISSION_LOG.jsonl` 或 lecture-level omission logs 中可见。

        ## 如何继续增量更新某一讲

        1. 在 `lectures/<lecture_slug>/` 内补齐 canonical evidence。
        2. 生成 coverage / figure / eval / repair sidecars。
        3. 运行 `python3 build/validate_lecture.py <lecture_slug_or_prefix>`。
        4. 通过后再更新 `COURSE_COVERAGE_INDEX.jsonl` 并合并入 `book/`。
        """
    )


def render_agents_md() -> str:
    return dedent(
        """\
        # AGENTS Map

        先读这些文件，不要把本文件扩写成巨型手册：

        1. `docs/harness_design.md`
        2. `docs/quality_rubric.md`
        3. `docs/writing_style_guide.md`
        4. `docs/notation_and_glossary.md`
        5. `docs/known_failure_modes.md`
        6. `docs/evaluator_playbook.md`

        Lecture-level workspaces are the record system.
        Prefer updating structured artifacts over relying on conversational summaries.
        """
    )


def render_course_spec() -> str:
    rows = []
    for lecture in LECTURES:
        slide_block = "\n".join([f"  - {slide['label']}: {slide['url']}" for slide in lecture["slide_urls"]])
        reading_block = "\n".join([f"  - {reading['title']} ({reading['url']})" for reading in lecture["readings"]])
        special = lecture["special_time_notes"] or "None"
        rows.append(
            dedent(
                f"""\
                ## {lecture['lecture_id']} {lecture['title']}

                - Date: {lecture['date']}
                - Speaker: {lecture['speaker']} ({lecture['affiliation']})
                - Recording: {lecture['recording_url']}
                - Special time notes: {special}
                - Slides:
        {slide_block}
                - Supplemental readings:
        {reading_block}
                """
            )
        )
    body = "\n".join(rows)
    return dedent(
        f"""\
        # COURSE SPEC

        - Course: {COURSE_TITLE}
        - Institution: UC Berkeley / Berkeley RDI
        - Official page: {COURSE_PAGE}
        - Output mode: harness-managed Chinese textbook with lecture-by-lecture gates
        - Source policy: official page + public recordings + official slides + official readings
        - Last synced from course page: 2026-04-15

        {body}
        """
    )


def render_harness_design() -> str:
    return dedent(
        """\
        # Harness Design

        本 run 把“视频转讲义”作为可恢复、可审计的多阶段构建，而不是一次性生成任务。

        Fixed stages:

        0. source acquisition
        1. planner
        2. coverage extractor
        3. writer
        4. figure / visual provenance
        5. skeptical evaluator
        6. repair writer
        7. lecture validator
        8. book assembly
        9. final evaluator / textbook validator

        Key rules:

        - coverage before prose
        - sidecars before final PDF
        - evaluator can fail fluent writing
        - dense slides must be expanded, not compressed into a takeaway sentence
        - repository is the record system; agents should read and write artifacts, not rely on chat state
        - a lecture does not enter `book/` unless evaluator and validator both pass
        """
    )


def render_quality_rubric() -> str:
    return dedent(
        """\
        # Quality Rubric

        Lecture-level passing thresholds:

        - coverage >= 0.95
        - pedagogical_depth >= 0.85
        - hallucination_control >= 0.90
        - reading_integration >= 0.80
        - any required coverage unit left unaddressed => fail
        - any dense slide reduced to a one-line takeaway => fail
        - any formula without symbol explanation => fail
        - any code or algorithm pasted without explanation => fail
        - any figure without provenance => fail

        Book-level passing thresholds:

        - course_coverage >= 0.98
        - textbook_coherence >= 0.85
        - chapter_depth_avg >= 0.85
        - hallucination_control >= 0.90
        - no missing required lecture
        - no failed lecture validator
        - final PDF must compile

        Evaluator posture:

        - not collaborative with the writer
        - must enumerate blocking issues concretely
        - must prefer fail-closed behavior to ambiguous praise
        """
    )


def render_writing_style_guide() -> str:
    return dedent(
        """\
        # Writing Style Guide

        - 主体使用中文，但保留重要英文术语、论文标题、算法名、模型名、benchmark 名称。
        - 第一次出现关键术语时，使用中英双语，例如“推理时计算（inference-time computation）”。
        - 不要只复述 slide bullet；每个重要概念都要回答“问题、朴素方法为何不够、机制、形式化定义、例子、失败模式、与 LLM agents 的关系”。
        - 对公式必须给 display math、符号解释和直觉说明；若源里有推导，要逐步展开。
        - 对算法必须说明输入、输出、循环结构、停止条件和复杂度直觉。
        - 对代码必须说明每个逻辑块解决什么问题，并点出潜在 bug、安全风险或评估盲点。
        - 对 readings 的融入要以教学为目的，不要把章节写成论文列表。
        - 对 multimodal / GUI / web / OS agents，必须解释 perception-action loop、grounding、environment feedback 和 benchmark 局限。
        - 对 formal math / theorem proving / verification 相关内容，必须区分 informal reasoning、formal specification、autoformalization、proof search 与 verification。
        """
    )


def render_notation_and_glossary() -> str:
    return dedent(
        """\
        # Notation And Glossary

        Canonical terminology for this run:

        - 推理时计算（inference-time computation）
        - 搜索（search）
        - 规划（planning）
        - 记忆（memory）
        - 后训练（post-training）
        - 偏好优化（preference optimization）
        - 验证（verification）
        - 定理证明（theorem proving）
        - 自动形式化（autoformalization）
        - 多模态智能体（multimodal agent）
        - GUI grounding
        - Web agent
        - OS agent

        Consistency requirements:

        - reasoning / planning / search / inference-time compute 不混用
        - post-training / RLHF / DPO / PPO / GRPO 关系要明确
        - formalization / autoformalization / theorem proving / proof search 要区分
        - agent / tool use / function calling / workflow 要按上下文精确定义
        """
    )


def render_known_failure_modes() -> str:
    return dedent(
        """\
        # Known Failure Modes

        - 把 dense slide 压成一段“takeaway”，没有逐层解释。
        - 公式只翻译，不解释符号和直觉。
        - 代码只贴出片段，不解释控制流和风险。
        - 读过 readings 但没有明确写出与 lecture 的关系。
        - 把推断当成 source fact，没有标注为“推断/延伸解释”。
        - 图像只有装饰作用，没有教学作用或 provenance。
        - 自动字幕中的术语错误没有进入 low-confidence 或 omission artifacts。
        - 跨讲术语漂移，例如把 search、planning、reasoning 混成同义词。
        """
    )


def render_evaluator_playbook() -> str:
    return dedent(
        """\
        # Evaluator Playbook

        The evaluator is the gatekeeper, not the writer's partner.

        Required checks:

        1. coverage completeness
        2. pedagogical depth
        3. derivation fidelity
        4. code / algorithm fidelity
        5. figure usefulness and provenance
        6. reading integration
        7. coherence with previous lectures
        8. hallucination risk
        9. Chinese textbook readability
        10. LaTeX compile readiness

        Blocking issue style:

        - identify `type`
        - point to `unit_id` or section
        - state the concrete problem
        - prescribe a required fix that a repair writer can execute

        Repair loop discipline:

        - prefer bounded fixes over full rewrites
        - keep `repair_log.jsonl` aligned to evaluator issue ids
        - after 3 failed repair rounds, escalate to `unresolved_issues.md` and course-level omission logs
        """
    )


def render_book_outline() -> str:
    return dedent(
        """\
        # BOOK OUTLINE

        Part I: Foundations of Advanced LLM Agents
        - Course overview
        - Reasoning-time computation
        - Learning to reason
        - Memory and planning

        Part II: Agentic Workflows, Tools, and Code
        - Tool use and workflows
        - Coding agents
        - Vulnerability detection
        - Program verification

        Part III: Multimodal and Interactive Agents
        - Multimodal autonomous agents
        - Perception-to-action agents
        - Web, OS, and GUI environments

        Part IV: Mathematical Reasoning and Theorem Proving
        - RL for formal mathematics
        - Autoformalization
        - Theorem proving
        - Advanced proof search

        Part V: Abstraction, Discovery, Safety, and Security
        - Abstraction and discovery
        - Safe and secure agentic AI
        """
    )


def render_preface() -> str:
    return dedent(
        r"""\
        \section*{前言}
        本书来自 UC Berkeley / Berkeley RDI 课程《CS294/194-280: Advanced Large Language Model Agents, Spring 2025》的公开视频、官方 slides、课程页和 supplemental readings。

        本教材采用 harness-managed workflow：先逐讲构建结构化证据和 coverage ledger，再经过 evaluator 与 validator gate，最后合并成书。
        """
    )


def render_how_to_use() -> str:
    return dedent(
        r"""\
        \section*{如何使用本书}
        建议先按课程顺序阅读，也可以根据主题从推理、工具使用、多模态、形式化数学、以及安全与安全性章节中选择路径。

        每章都保留 source-grounded sidecars，以便回查视频、slides、读物和可能的 omission。
        """
    )


def render_book_main() -> str:
    return dedent(
        r"""\
        \documentclass[a4paper]{article}
        \usepackage[fontset=fandol]{ctex}
        \usepackage[margin=2.5cm]{geometry}
        \usepackage{hyperref}
        \title{CS294/194-280: Advanced Large Language Model Agents\\教材级中文讲义}
        \author{Codex Harness-Managed Build}
        \date{\today}
        \begin{document}
        \maketitle
        \input{frontmatter/preface.tex}
        \input{frontmatter/how_to_use_this_book.tex}
        \tableofcontents
        \newpage
        % Chapters are injected by build/merge_book.py after lecture validation passes.
        \end{document}
        """
    )


def build_textbook_source_manifest() -> dict:
    return {
        "course_id": COURSE_SLUG,
        "book_title": f"{COURSE_TITLE} 教材级中文讲义",
        "chapter_sources": [],
        "status": "bootstrapped_pending_lecture_merge",
    }


def write_placeholder_book_files() -> None:
    maybe_write_text(BOOK_DIR / "BOOK_OUTLINE.md", render_book_outline())
    maybe_write_text(BOOK_DIR / "frontmatter" / "preface.tex", render_preface())
    maybe_write_text(BOOK_DIR / "frontmatter" / "how_to_use_this_book.tex", render_how_to_use())
    maybe_write_text(BOOK_DIR / "appendices" / "glossary.tex", "\\section*{Glossary}\\n待生成。\\n")
    maybe_write_text(BOOK_DIR / "appendices" / "notation.tex", "\\section*{Notation}\\n待生成。\\n")
    maybe_write_text(BOOK_DIR / "main.tex", render_book_main())
    maybe_write_json(BOOK_DIR / "textbook_source_manifest.json", build_textbook_source_manifest())


def write_top_level_files() -> None:
    maybe_write_text(RUN_ROOT / "README.md", render_readme())
    maybe_write_text(RUN_ROOT / "AGENTS.md", render_agents_md())
    maybe_write_text(RUN_ROOT / "COURSE_SPEC.md", render_course_spec())
    maybe_write_json(RUN_ROOT / "COURSE_SOURCE_MANIFEST.json", build_course_source_manifest())
    maybe_write_jsonl(RUN_ROOT / "COURSE_COVERAGE_INDEX.jsonl", build_course_coverage_index())
    maybe_write_text(RUN_ROOT / "COURSE_OMISSION_LOG.jsonl", "")
    maybe_write_text(DOCS_DIR / "harness_design.md", render_harness_design())
    maybe_write_text(DOCS_DIR / "quality_rubric.md", render_quality_rubric())
    maybe_write_text(DOCS_DIR / "writing_style_guide.md", render_writing_style_guide())
    maybe_write_text(DOCS_DIR / "notation_and_glossary.md", render_notation_and_glossary())
    maybe_write_text(DOCS_DIR / "known_failure_modes.md", render_known_failure_modes())
    maybe_write_text(DOCS_DIR / "evaluator_playbook.md", render_evaluator_playbook())


def write_seed_and_meta() -> None:
    seed = {
        "course_id": COURSE_SLUG,
        "course_title": COURSE_TITLE,
        "official_course_page": COURSE_PAGE,
        "lecture_count": len(LECTURES),
        "lectures": [],
    }
    for lecture in LECTURES:
        lecture_dir = LECTURES_DIR / lecture["slug"]
        maybe_write_json(lecture_dir / "meta.json", lecture_meta(lecture))
        seed["lectures"].append(
            {
                "lecture_id": lecture["lecture_id"],
                "lecture_slug": lecture["slug"],
                "title": lecture["title"],
                "date": lecture["date"],
                "speaker": lecture["speaker"],
                "recording_url": lecture["recording_url"],
            }
        )
    maybe_write_json(BUILD_DIR / "course_manifest_seed.json", seed)


def main() -> None:
    ensure_dirs()
    write_top_level_files()
    write_seed_and_meta()
    write_placeholder_book_files()
    print(RUN_ROOT)


if __name__ == "__main__":
    main()
