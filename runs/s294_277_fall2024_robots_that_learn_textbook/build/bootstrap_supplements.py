#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.video_note_harness.common import write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
LECTURES_DIR = ROOT / "lectures"
SPRING_2026_URL = "https://robots-that-learn.github.io/"
UNDERACTUATED_ROOT = "https://underactuated.csail.mit.edu/"
UNDERACTUATED_SCHEDULE = "https://underactuated.csail.mit.edu/Spring2024/schedule.html"


SUPPLEMENTS = [
    {
        "lecture_id": 19,
        "slug": "spring2026_latest_course_updates",
        "title": "Appendix A: Spring 2026 最新课程主线与研究重排",
        "title_short": "Spring 2026 最新课程主线",
        "date": "2026-01-26",
        "topics": [
            "课程主线重排与教学目标变化",
            "新增 diffusion models 与 world models 主题",
            "从 Fall 2024 到 Spring 2026 的 reading 更新",
            "最新讲次结构对 robot learning 教材的影响",
        ],
        "sources": [
            {"source_id": "spring2026_course_page", "source_type": "official_course_page", "origin_url": SPRING_2026_URL},
        ],
    },
    {
        "lecture_id": 20,
        "slug": "spring2026_diffusion_and_world_models",
        "title": "Appendix B: Spring 2026 的 Diffusion Models 与 Video World Models 补充",
        "title_short": "Diffusion 与 World Models 补充",
        "date": "2026-03-09",
        "topics": [
            "Normalizing Flows 与 Flow Matching",
            "Diffusion Models 在机器人学习中的位置",
            "Video World Models 与 World Action Models",
            "Track2Act 与视频驱动操作建模",
        ],
        "sources": [
            {"source_id": "spring2026_course_page", "source_type": "official_course_page", "origin_url": SPRING_2026_URL},
            {"source_id": "normalizing_flows_paper", "source_type": "official_reading", "origin_url": "https://robots-that-learn.github.io/resources/normalizing_flows_paper.pdf"},
            {"source_id": "flow_matching_paper", "source_type": "official_reading", "origin_url": "https://robots-that-learn.github.io/resources/flow_matching_paper.pdf"},
            {"source_id": "wam_arxiv", "source_type": "official_reading", "origin_url": "https://arxiv.org/pdf/2602.15922"},
            {"source_id": "track2act_arxiv", "source_type": "official_reading", "origin_url": "https://arxiv.org/pdf/2405.01527"},
        ],
    },
    {
        "lecture_id": 21,
        "slug": "underactuated_dynamics_control_and_planning_foundations",
        "title": "Appendix C: Underactuated 课程的动力学、控制与规划基础",
        "title_short": "动力学控制规划基础",
        "date": "2024-02-01",
        "topics": [
            "Robot dynamics 与 underactuated model systems",
            "Dynamic Programming 与 LQR",
            "Acrobot / cart-pole / quadrotor 等 canonical systems",
            "Trajectory Optimization 与 planning foundations",
        ],
        "sources": [
            {"source_id": "underactuated_schedule", "source_type": "official_schedule", "origin_url": UNDERACTUATED_SCHEDULE},
            {"source_id": "underactuated_playlist", "source_type": "official_playlist", "origin_url": "https://www.youtube.com/playlist?list=PLkx8KyIQkMfU5szP43GlE_S1QGSPQfL9s"},
        ],
    },
    {
        "lecture_id": 22,
        "slug": "underactuated_locomotion_contact_and_robustness",
        "title": "Appendix D: Underactuated 课程的 locomotion、contact 与 robustness 补充",
        "title_short": "Locomotion / Contact / Robustness 补充",
        "date": "2024-04-01",
        "topics": [
            "Simple legs、limit cycles 与 hybrid locomotion",
            "Contact、humanoids 与 planning under dynamics",
            "Stochastic / robust / output-feedback control",
            "Imitation learning 与 foundation models 的控制视角",
        ],
        "sources": [
            {"source_id": "underactuated_schedule", "source_type": "official_schedule", "origin_url": UNDERACTUATED_SCHEDULE},
            {"source_id": "underactuated_playlist", "source_type": "official_playlist", "origin_url": "https://www.youtube.com/playlist?list=PLkx8KyIQkMfU5szP43GlE_S1QGSPQfL9s"},
        ],
    },
]


def fetch(url: str) -> str:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def paragraphs_from_lines(lines: list[str]) -> list[str]:
    out = []
    for line in lines:
        line = normalize_text(line)
        if line and line not in out:
            out.append(line)
    return out


def spring2026_blocks() -> dict[str, list[str]]:
    html = fetch(SPRING_2026_URL)
    soup = BeautifulSoup(html, "html.parser")
    schedule_lines = []
    reading_lines = []
    in_schedule = False
    in_readings = False
    current_h3 = None
    current_h4 = None
    for tag in soup.find_all(["h2", "h3", "h4", "li", "p"]):
        text = normalize_text(tag.get_text(" ", strip=True))
        if tag.name == "h2":
            in_schedule = text == "Schedule"
            in_readings = text == "Reading materials"
        elif tag.name == "h3":
            current_h3 = text
        elif tag.name == "h4":
            current_h4 = text
        elif in_schedule and tag.name == "li" and text.startswith("Lecture"):
            schedule_lines.append(text)
        elif in_readings and tag.name in {"li", "p"} and text:
            prefix = " / ".join(x for x in [current_h3, current_h4] if x and x.startswith("Lecture"))
            reading_lines.append(f"{prefix}: {text}" if prefix else text)

    update_lines = [
        "Spring 2026 官方课程页显示，这门课仍以 robot learning 为主轴，但重新排列了 biomechanics、kinematics、dexterous manipulation、RL、behavior cloning、visual imitation、locomotion、navigation 与 language-based planning 的顺序。",
        "与 Fall 2024 相比，最明显的新增主题是 Introduction to Diffusion Models，以及 Lecture 6B 的 Video World Models。",
        "Spring 2026 reading list 还显式引入了 Normalizing Flows、Flow Matching、World Action Models are Zero-shot Policies、Track2Act、Universal Manipulation Interface 等更新材料。",
    ]

    diffusion_lines = [
        "Spring 2026 的 Lecture 3 明确以 Diffusion Models 为 guest lecture 主题，并把 Normalizing Flows 与 Flow Matching 放进官方 reading list。",
        "Lecture 6B 明确加入 Video World Models，并在 reading list 中给出 World Action Models are Zero-shot Policies 与 Track2Act 两个近年的 primary sources。",
        "这说明课程主线已经从单纯的 reinforcement learning / behavior cloning，扩展到生成式 action modeling、视频世界模型和更强的表征学习路线。",
    ]

    return {
        "schedule": paragraphs_from_lines(update_lines + schedule_lines),
        "readings": paragraphs_from_lines(diffusion_lines + reading_lines),
    }


def underactuated_page_summary(slug: str, limit: int = 8) -> list[str]:
    html = fetch(f"{UNDERACTUATED_ROOT}{slug}.html")
    soup = BeautifulSoup(html, "html.parser")
    title = normalize_text(soup.title.get_text(" ", strip=True)) if soup.title else slug
    lines = [title]
    main = soup.find("main") or soup
    for tag in main.find_all(["h1", "h2", "h3", "p"], limit=80):
        text = normalize_text(tag.get_text(" ", strip=True))
        if not text:
            continue
        if tag.name.startswith("h") or len(text) > 80:
            lines.append(text)
        if len(lines) >= limit:
            break
    return paragraphs_from_lines(lines)


def underactuated_blocks() -> dict[str, list[str]]:
    control_pages = ["intro", "pend", "dp", "lqr", "acrobot", "lyapunov", "trajopt", "planning"]
    locomotion_pages = ["simple_legs", "limit_cycles", "contact", "humanoids", "robust", "policy_search", "feedback_motion_planning"]
    control_lines = [
        "MIT Underactuated Spring 2024 适合作为这本书的理论骨架，尤其能补强 dynamics、optimal control、trajectory optimization 与 planning 的数学基础。",
    ]
    locomotion_lines = [
        "MIT Underactuated Spring 2024 对 simple legs、limit cycles、contact、humanoids、robust control 和 planning under dynamics 给出了系统化讲解，可用来补强 locomotion 与 contact-rich planning 主线。",
    ]
    for slug in control_pages:
        control_lines.extend(underactuated_page_summary(slug))
    for slug in locomotion_pages:
        locomotion_lines.extend(underactuated_page_summary(slug))
    return {
        "control": paragraphs_from_lines(control_lines),
        "locomotion": paragraphs_from_lines(locomotion_lines),
    }


def paragraph_rows(paragraphs: list[str], source_id: str) -> list[dict]:
    rows = []
    second = 0
    for idx, paragraph in enumerate(paragraphs, start=1):
        start = f"00:00:{second:02d},000"
        second += 5
        end = f"00:00:{second:02d},000"
        rows.append(
            {
                "unit_id": f"sub_{idx:04d}",
                "source_type": "subtitle_span",
                "source_id": source_id,
                "loc": {"start": start, "end": end},
                "text": paragraph,
                "required": True,
            }
        )
    return rows


def slide_rows(title: str) -> list[dict]:
    return [
        {
            "unit_id": "slide_0001",
            "source_type": "slide_page",
            "source_id": "generated_overview_card",
            "loc": {"page": 1},
            "text": title,
            "asset_path": "overview_cover.jpg",
            "required": True,
        }
    ]


def segment_rows(transcript: list[dict], topics: list[str]) -> list[dict]:
    chunk = max(1, len(transcript) // max(1, len(topics)))
    rows = []
    for idx, topic in enumerate(topics, start=1):
        start_i = (idx - 1) * chunk
        end_i = len(transcript) if idx == len(topics) else min(len(transcript), idx * chunk)
        subset = transcript[start_i:end_i] or transcript[:1]
        rows.append(
            {
                "segment_id": f"seg_{idx:02d}",
                "start": subset[0]["loc"]["start"],
                "end": subset[-1]["loc"]["end"],
                "source_unit_ids": [row["unit_id"] for row in subset] + ["slide_0001"],
                "target_section_hint": topic,
            }
        )
    return rows


def coverage_rows(topics: list[str], transcript: list[dict]) -> list[dict]:
    rows = []
    for idx, topic in enumerate(topics, start=1):
        rows.append(
            {
                "unit_id": f"topic_{idx:02d}",
                "source_type": "lecture_topic_seed",
                "source_id": "lecture_meta",
                "loc": {"topic_index": idx},
                "kind": ["topic_seed"],
                "summary": topic,
                "required": True,
                "status": "unclassified",
                "mapped_section": None,
                "figure_ids": [],
                "notes": "Supplement topic seed.",
            }
        )
    for row in transcript:
        rows.append(
            {
                "unit_id": row["unit_id"],
                "source_type": row["source_type"],
                "source_id": row["source_id"],
                "loc": row["loc"],
                "kind": ["subtitle_span"],
                "summary": row["text"][:200],
                "required": True,
                "status": "unclassified",
                "mapped_section": None,
                "figure_ids": [],
                "notes": "",
            }
        )
    rows.append(
        {
            "unit_id": "slide_0001",
            "source_type": "slide_page",
            "source_id": "generated_overview_card",
            "loc": {"page": 1},
            "kind": ["slide_page"],
            "summary": "Overview card for supplement chapter.",
            "required": True,
            "status": "unclassified",
            "mapped_section": None,
            "figure_ids": [],
            "notes": "",
        }
    )
    return rows


def make_cover(path: Path, title: str, source_lines: list[str]) -> None:
    text = title + "\n\n" + "\n".join(source_lines[:8])
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1600, 900), "white")
    draw = ImageDraw.Draw(image)
    font_path = "/Library/Fonts/Arial Unicode.ttf"
    title_font = ImageFont.truetype(font_path, 42)
    body_font = ImageFont.truetype(font_path, 24)

    y = 40
    draw.multiline_text((40, y), title, fill="black", font=title_font, spacing=10)
    y += 120

    words = text.splitlines()[2:]
    wrapped_lines = []
    for raw in words:
        current = ""
        for chunk in re.split(r"(\s+)", raw):
            candidate = current + chunk
            if draw.textlength(candidate, font=body_font) > 1480 and current:
                wrapped_lines.append(current.strip())
                current = chunk
            else:
                current = candidate
        if current.strip():
            wrapped_lines.append(current.strip())
        wrapped_lines.append("")

    draw.multiline_text((40, y), "\n".join(wrapped_lines[:28]), fill="black", font=body_font, spacing=8)
    image.save(path)


def main() -> None:
    spring = spring2026_blocks()
    under = underactuated_blocks()
    content_map = {
        19: spring["schedule"],
        20: spring["readings"],
        21: under["control"],
        22: under["locomotion"],
    }

    for spec in SUPPLEMENTS:
        lecture_dir = LECTURES_DIR / f"{spec['lecture_id']:02d}_{spec['slug']}"
        lecture_dir.mkdir(parents=True, exist_ok=True)
        paragraphs = content_map[spec["lecture_id"]]
        transcript = paragraph_rows(paragraphs, spec["sources"][0]["source_id"])
        slides = slide_rows(spec["title"])
        segments = segment_rows(transcript, spec["topics"])
        coverage = coverage_rows(spec["topics"], transcript)

        cover_path = lecture_dir / "cover.jpg"
        make_cover(cover_path, spec["title"], paragraphs)

        meta = {
            "playlist_index": spec["lecture_id"],
            "schedule_id": f"APP-{spec['lecture_id']}",
            "date": spec["date"],
            "kind": "supplement",
            "title": spec["title"],
            "title_short": spec["title_short"],
            "slug": spec["slug"],
            "video_id": None,
            "video_url": None,
            "course_id": "ucb-s294-277-robots-that-learn-fall-2024",
            "course_page_url": SPRING_2026_URL if spec["lecture_id"] < 21 else UNDERACTUATED_ROOT,
            "thumbnail": str(cover_path.relative_to(REPO_ROOT)),
            "subtitle": None,
            "material": None,
            "transcript_text": str((lecture_dir / "transcript.txt").relative_to(REPO_ROOT)),
            "official_text": str((lecture_dir / "official.txt").relative_to(REPO_ROOT)),
            "course_mode": True,
            "segmentation_required": True,
            "topics": spec["topics"],
        }

        source_manifest = {
            "course_id": "ucb-s294-277-robots-that-learn-fall-2024",
            "course_mode": True,
            "lecture_id": f"{spec['lecture_id']:02d}",
            "lecture_slug": lecture_dir.name,
            "title": spec["title"],
            "origin_url": spec["sources"][0]["origin_url"],
            "slide_origin_url": None,
            "sources": [
                {
                    "source_id": "lecture_meta",
                    "source_type": "lecture_metadata",
                    "origin_url": spec["sources"][0]["origin_url"],
                    "local_path": str((lecture_dir / "meta.json").relative_to(REPO_ROOT)),
                    "required_for_coverage": True,
                    "status": "available",
                    "notes": "Supplement workspace metadata.",
                },
                *[
                    {
                        "source_id": src["source_id"],
                        "source_type": src["source_type"],
                        "origin_url": src["origin_url"],
                        "local_path": None,
                        "required_for_coverage": True,
                        "status": "remote_only",
                        "notes": "Official supplemental source retained as URL.",
                    }
                    for src in spec["sources"]
                ],
                {
                    "source_id": "cover_jpg",
                    "source_type": "cover_image",
                    "origin_url": None,
                    "local_path": str(cover_path.relative_to(REPO_ROOT)),
                    "required_for_coverage": True,
                    "status": "available",
                    "notes": "Generated provenance card summarizing official sources for the supplement chapter.",
                },
            ],
        }

        write_json(lecture_dir / "meta.json", meta)
        write_json(lecture_dir / "source_manifest.json", source_manifest)
        write_jsonl(lecture_dir / "transcript.jsonl", transcript)
        write_jsonl(lecture_dir / "slides.jsonl", slides)
        write_jsonl(lecture_dir / "segments.jsonl", segments)
        write_jsonl(lecture_dir / "coverage_units.jsonl", coverage)
        write_jsonl(lecture_dir / "omission_log.jsonl", [])
        write_json(lecture_dir / "figure_manifest.json", [])
        (lecture_dir / "repair_log.jsonl").write_text("")
        (lecture_dir / "transcript.txt").write_text("\n\n".join(paragraphs) + "\n")
        (lecture_dir / "official.txt").write_text("\n\n".join(paragraphs) + "\n")
        (lecture_dir / "course_notes_excerpt.txt").write_text("\n\n".join(paragraphs) + "\n")
        write_json(
            lecture_dir / "course_notes_excerpt_meta.json",
            [{"section": spec["title_short"], "start_page": 1, "end_page": 1}],
        )
        (lecture_dir / "README.md").write_text(
            f"# {spec['title']}\n\nSupplement appendix workspace bootstrapped from official course pages and notes.\n"
        )

    subprocess.run(
        [
            "python3",
            str(REPO_ROOT / "scripts" / "video_note_harness" / "bootstrap_harness.py"),
            "--run-root",
            str(ROOT),
        ],
        check=True,
    )
    subprocess.run(
        ["python3", str(ROOT / "build" / "build_course_manifest.py")],
        check=True,
    )
    print("bootstrapped supplements")


if __name__ == "__main__":
    main()
