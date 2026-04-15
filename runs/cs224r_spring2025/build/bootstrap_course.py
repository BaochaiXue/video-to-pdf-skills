#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.request
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RUN_ROOT.parents[1]

COURSE_ID = "stanford-cs224r-spring-2025"
PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLoROMvodv4rPwxE0ONYRa_itZFdaKCylL"
COURSE_PAGE_URL = "https://cs224r.stanford.edu/spring_2025/"

LECTURES = [
    {"playlist_index": 1, "date": "2025-04-02", "kind": "lecture", "title": "Lecture 1: Class Intro", "title_short": "Class Intro", "slug": "class_intro", "video_id": "EvHRQhMX7_w", "video_url": "https://www.youtube.com/watch?v=EvHRQhMX7_w", "slide_title": "Course intro + MDPs", "slide_url": "https://cs224r.stanford.edu/spring_2025/slides/01_cs224r_intro_2025.pdf"},
    {"playlist_index": 2, "date": "2025-04-04", "kind": "lecture", "title": "Lecture 2: Imitation Learning", "title_short": "Imitation Learning", "slug": "imitation_learning", "video_id": "WxRDyObrm_M", "video_url": "https://www.youtube.com/watch?v=WxRDyObrm_M", "slide_title": "Imitation Learning", "slide_url": "https://cs224r.stanford.edu/spring_2025/slides/02_cs224r_imitation_2025.pdf"},
    {"playlist_index": 3, "date": "2025-04-09", "kind": "lecture", "title": "Lecture 3: Policy Gradients", "title_short": "Policy Gradients", "slug": "policy_gradients", "video_id": "KCAOXd4IO9o", "video_url": "https://www.youtube.com/watch?v=KCAOXd4IO9o", "slide_title": "Policy Gradients", "slide_url": "https://cs224r.stanford.edu/spring_2025/slides/03_cs224r_policy_gradients_2025.pdf"},
    {"playlist_index": 4, "date": "2025-04-11", "kind": "lecture", "title": "Lecture 4: Actor-Critic Methods", "title_short": "Actor-Critic Methods", "slug": "actor_critic_methods", "video_id": "oejFZShW9hU", "video_url": "https://www.youtube.com/watch?v=oejFZShW9hU", "slide_title": "Actor-Critic Methods", "slide_url": "https://cs224r.stanford.edu/spring_2025/slides/04_cs224r_actor_critic_2025.pdf"},
    {"playlist_index": 5, "date": "2025-04-16", "kind": "lecture", "title": "Lecture 5: Off-Policy Actor Critic", "title_short": "Off-Policy Actor Critic", "slug": "off_policy_actor_critic", "video_id": "cRGKc-nAWho", "video_url": "https://www.youtube.com/watch?v=cRGKc-nAWho", "slide_title": "Off-Policy Actor Critic", "slide_url": "https://cs224r.stanford.edu/spring_2025/slides/05_cs224r_offpolicy_actor_critic_2025.pdf"},
    {"playlist_index": 6, "date": "2025-04-18", "kind": "lecture", "title": "Lecture 6: Q-Learning", "title_short": "Q-Learning", "slug": "q_learning", "video_id": "-7kv6jf0isQ", "video_url": "https://www.youtube.com/watch?v=-7kv6jf0isQ", "slide_title": "Q-learning", "slide_url": "https://cs224r.stanford.edu/spring_2025/slides/06_cs224r_qlearning_2025.pdf"},
    {"playlist_index": 7, "date": "2025-04-23", "kind": "lecture", "title": "Lecture 7: Offline RL", "title_short": "Offline RL", "slug": "offline_rl", "video_id": "lRDaXnPIzks", "video_url": "https://www.youtube.com/watch?v=lRDaXnPIzks", "slide_title": "Offline RL", "slide_url": "https://cs224r.stanford.edu/spring_2025/slides/07_cs224r_offline_rl_2025.pdf"},
    {"playlist_index": 8, "date": "2025-04-25", "kind": "lecture", "title": "Lecture 8: Reward Learning", "title_short": "Reward Learning", "slug": "reward_learning", "video_id": "PDIxDhA9Z6Y", "video_url": "https://www.youtube.com/watch?v=PDIxDhA9Z6Y", "slide_title": "Reward Learning", "slide_url": "https://cs224r.stanford.edu/spring_2025/slides/08_cs224r_reward_learning_2025.pdf"},
    {"playlist_index": 9, "date": "2025-04-30", "kind": "guest_lecture", "title": "Lecture 9: RL for LLMs", "title_short": "RL for LLMs", "slug": "rl_for_llms", "video_id": "XKLGuwvSKvI", "video_url": "https://www.youtube.com/watch?v=XKLGuwvSKvI", "slide_title": "RL for LLMs: Preference Optimization", "slide_url": "https://cs224r.stanford.edu/spring_2025/slides/09_cs224r-2025-rlhf.pdf"},
    {"playlist_index": 10, "date": "2025-05-02", "kind": "guest_lecture", "title": "Lecture 10: RL for LLM Reasoning", "title_short": "RL for LLM Reasoning", "slug": "rl_for_llm_reasoning", "video_id": "O2VpNnwB4lM", "video_url": "https://www.youtube.com/watch?v=O2VpNnwB4lM", "slide_title": "RL for LLMs: Reasoning", "slide_url": "https://cs224r.stanford.edu/spring_2025/slides/10_cs224r-rl_for_reasoning_lecture.pdf"},
    {"playlist_index": 11, "date": "2025-05-07", "kind": "lecture", "title": "Lecture 11: Model-Based RL", "title_short": "Model-Based RL", "slug": "model_based_rl", "video_id": "PvqyGnOirgA", "video_url": "https://www.youtube.com/watch?v=PvqyGnOirgA", "slide_title": "Model-based RL", "slide_url": "https://cs224r.stanford.edu/spring_2025/slides/11_cs224r_mbrl_2025.pdf"},
    {"playlist_index": 12, "date": "2025-05-09", "kind": "lecture", "title": "Lecture 12: Multi-Task RL", "title_short": "Multi-Task RL", "slug": "multi_task_rl", "video_id": "qNdsI_4AQJw", "video_url": "https://www.youtube.com/watch?v=qNdsI_4AQJw", "slide_title": "Multi-Task and Goal-Conditioned RL", "slide_url": "https://cs224r.stanford.edu/spring_2025/slides/12_cs224r_mtrl_gcrl_2025.pdf"},
    {"playlist_index": 13, "date": "2025-05-14", "kind": "lecture", "title": "Lecture 13: Meta RL", "title_short": "Meta RL", "slug": "meta_rl", "video_id": "wSiyEpvoGkA", "video_url": "https://www.youtube.com/watch?v=wSiyEpvoGkA", "slide_title": "Meta-RL", "slide_url": "https://cs224r.stanford.edu/spring_2025/slides/13_cs224r_metarl_2025.pdf"},
    {"playlist_index": 14, "date": "2025-05-16", "kind": "lecture", "title": "Lecture 14: Exploration", "title_short": "Exploration", "slug": "exploration", "video_id": "4tlSKdi8teU", "video_url": "https://www.youtube.com/watch?v=4tlSKdi8teU", "slide_title": "Exploration", "slide_url": "https://cs224r.stanford.edu/spring_2025/slides/14_cs224r_exploration_2025.pdf"},
    {"playlist_index": 15, "date": "2025-05-21", "kind": "lecture", "title": "Lecture 15: Hierarchical RL and IL", "title_short": "Hierarchical RL and IL", "slug": "hierarchical_rl_and_il", "video_id": "iKWYLSVAtfM", "video_url": "https://www.youtube.com/watch?v=iKWYLSVAtfM", "slide_title": "Hierarchical RL and IL", "slide_url": "https://cs224r.stanford.edu/spring_2025/slides/15_cs224r_hierarchy_2025.pdf"},
    {"playlist_index": 16, "date": "2025-05-23", "kind": "lecture", "title": "Lecture 16: RL for Robots", "title_short": "RL for Robots", "slug": "rl_for_robots", "video_id": "rbaWQQLrzl0", "video_url": "https://www.youtube.com/watch?v=rbaWQQLrzl0", "slide_title": "RL for Robots: Autonomous Learning", "slide_url": "https://cs224r.stanford.edu/spring_2025/slides/16_cs224r_autonomy_2025.pdf"},
    {"playlist_index": 17, "date": "2025-05-28", "kind": "guest_lecture", "title": "Lecture 17: Advancing Robot Intelligence", "title_short": "Advancing Robot Intelligence", "slug": "advancing_robot_intelligence", "video_id": "Hp1WBWghrak", "video_url": "https://www.youtube.com/watch?v=Hp1WBWghrak", "slide_title": "RL for Robots: Sim-to-Real Transfer", "slide_url": "https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf"},
    {"playlist_index": 18, "date": "2025-05-30", "kind": "lecture", "title": "Lecture 18: Frontiers", "title_short": "Frontiers", "slug": "frontiers", "video_id": "FacJ_1tTSx4", "video_url": "https://www.youtube.com/watch?v=FacJ_1tTSx4", "slide_title": "Frontiers", "slide_url": "https://cs224r.stanford.edu/spring_2025/slides/18_cs224r_frontiers_how_to_research.pdf"},
    {"playlist_index": 19, "date": "2025-04-18", "kind": "ta_session", "title": "Tutorial Session: Review of Q-Learning", "title_short": "Review of Q-Learning", "slug": "q_learning_tutorial", "video_id": "07MQNMcxhZU", "video_url": "https://www.youtube.com/watch?v=07MQNMcxhZU", "slide_title": "Extra section on Q-learning", "slide_url": "https://cs224r.stanford.edu/spring_2025/slides/section_q_learning_tutorial.pdf"},
]


def ensure_dirs() -> None:
    for dirname in ["build", "lectures", "materials/slides", "meta", "raw", "text"]:
        (RUN_ROOT / dirname).mkdir(parents=True, exist_ok=True)


def download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    urllib.request.urlretrieve(url, dest)


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, check=check, text=True)


def clear_dir(path: Path) -> None:
    if not path.exists():
        return
    for child in sorted(path.iterdir()):
        if child.is_dir():
            clear_dir(child)
            child.rmdir()
        else:
            child.unlink()


def raw_dir(item: dict) -> Path:
    return RUN_ROOT / "raw" / f"{item['playlist_index']:02d}_{item['video_id']}"


def text_dir(item: dict) -> Path:
    return RUN_ROOT / "text" / f"{item['playlist_index']:02d}_{item['video_id']}"


def lecture_dir(item: dict) -> Path:
    return RUN_ROOT / "lectures" / f"{item['playlist_index']:02d}_{item['slug']}"


def lecture_slide_pdf(item: dict) -> Path:
    return RUN_ROOT / "materials" / "slides" / f"{item['playlist_index']:02d}_{item['slug']}.pdf"


def best_subtitle_path(item: dict) -> Path | None:
    candidates = sorted(raw_dir(item).glob("*.srt"))
    if not candidates:
        return None
    preferred = []
    for path in candidates:
        name = path.name
        score = 100
        if ".en-US.srt" in name:
            score = 10
        elif ".en.srt" in name:
            score = 20
        elif ".en-GB.srt" in name:
            score = 30
        elif ".en-orig.srt" in name:
            score = 40
        preferred.append((score, name, path))
    preferred.sort()
    return preferred[0][2]


def srt_to_text(srt_path: Path) -> str:
    text = srt_path.read_text(errors="ignore")
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    lines: list[str] = []
    for block in blocks:
        raw_lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(raw_lines) < 2:
            continue
        maybe_ts = raw_lines[1] if raw_lines[0].isdigit() else raw_lines[0]
        if "-->" not in maybe_ts:
            continue
        payload = raw_lines[2:] if raw_lines[0].isdigit() else raw_lines[1:]
        payload = [re.sub(r"<[^>]+>", "", line).strip() for line in payload]
        payload = [line for line in payload if line]
        if payload:
            lines.append(f"[{maybe_ts}] {' '.join(payload)}")
    return "\n".join(lines).strip() + "\n"


def extract_slide_text(pdf_path: Path, out_path: Path) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    run(["pdftotext", str(pdf_path), str(out_path)])


def render_slide_pages(pdf_path: Path, out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.glob("page-*.png")):
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "page"
    run(["pdftoppm", "-png", str(pdf_path), str(prefix)])


def write_text_bundle(item: dict) -> None:
    tdir = text_dir(item)
    tdir.mkdir(parents=True, exist_ok=True)
    subtitle = best_subtitle_path(item)
    transcript_path = tdir / "transcript.txt"
    if subtitle and not transcript_path.exists():
        transcript_path.write_text(srt_to_text(subtitle))

    official_text = tdir / "official.txt"
    extract_slide_text(lecture_slide_pdf(item), official_text)


def ensure_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(os.path.relpath(src, dst.parent))


def write_lecture_dir(item: dict) -> None:
    ldir = lecture_dir(item)
    ldir.mkdir(parents=True, exist_ok=True)

    covers = list(raw_dir(item).glob("*.jpg"))
    subtitle = best_subtitle_path(item)
    meta = {
        **item,
        "course_id": COURSE_ID,
        "playlist_url": PLAYLIST_URL,
        "course_page_url": COURSE_PAGE_URL,
        "thumbnail": str(covers[0].relative_to(RUN_ROOT)) if covers else None,
        "subtitle": str(subtitle.relative_to(RUN_ROOT)) if subtitle else None,
        "material": str(lecture_slide_pdf(item).relative_to(RUN_ROOT)),
        "transcript_text": str((text_dir(item) / "transcript.txt").relative_to(RUN_ROOT)) if (text_dir(item) / "transcript.txt").exists() else None,
        "official_text": str((text_dir(item) / "official.txt").relative_to(RUN_ROOT)),
        "slide_pages_dir": str((ldir / "pdf_pages").relative_to(RUN_ROOT)),
    }
    (ldir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    if meta["thumbnail"]:
        ensure_symlink(RUN_ROOT / meta["thumbnail"], ldir / "cover.jpg")
    if meta["subtitle"]:
        ensure_symlink(RUN_ROOT / meta["subtitle"], ldir / "subtitle.srt")
    if meta["transcript_text"]:
        ensure_symlink(RUN_ROOT / meta["transcript_text"], ldir / "transcript.txt")
    ensure_symlink(RUN_ROOT / meta["official_text"], ldir / "official.txt")
    ensure_symlink(RUN_ROOT / meta["material"], ldir / "slides.pdf")
    render_slide_pages(lecture_slide_pdf(item), ldir / "pdf_pages")

    for name in ["coverage_units.jsonl", "omission_log.jsonl"]:
        path = ldir / name
        if not path.exists():
            path.write_text("")
    if not (ldir / "figure_manifest.json").exists():
        (ldir / "figure_manifest.json").write_text("[]\n")

    summary_lines = [
        f"# {item['title']}",
        "",
        f"- Date: `{item['date']}`",
        f"- Kind: `{item['kind']}`",
        f"- Video: [YouTube]({item['video_url']})",
        f"- Slides: [slides.pdf](slides.pdf)",
        f"- Cover: [cover.jpg](cover.jpg)" if (ldir / "cover.jpg").exists() else "- Cover: unavailable",
        f"- Subtitle: [subtitle.srt](subtitle.srt)" if (ldir / "subtitle.srt").exists() else "- Subtitle: unavailable",
        f"- Transcript: [transcript.txt](transcript.txt)" if (ldir / "transcript.txt").exists() else "- Transcript: unavailable",
        "- Official text: [official.txt](official.txt)",
        "- Slide pages: `pdf_pages/page-*.png`",
        "",
        "## Writing requirements",
        "",
        "- Use Chinese.",
        "- Preserve coverage of both spoken explanation and slides.",
        "- Include diagrams, tables, and process figures from slides or video as note figures.",
        "- End each major section with `本章小结` and end the document with `总结与延伸`.",
    ]
    (ldir / "README.md").write_text("\n".join(summary_lines) + "\n")


def download_course_pages() -> None:
    download(COURSE_PAGE_URL, RUN_ROOT / "meta" / "course_page.html")
    download("https://cs224r.stanford.edu/", RUN_ROOT / "meta" / "root_page.html")


def download_slides() -> None:
    for item in LECTURES:
        download(item["slide_url"], lecture_slide_pdf(item))


def fetch_playlist_metadata() -> None:
    out = RUN_ROOT / "meta" / "playlist_full.json"
    if out.exists():
        return
    output = subprocess.check_output(
        ["yt-dlp", "--flat-playlist", "--dump-single-json", PLAYLIST_URL],
        text=True,
    )
    out.write_text(output)


def fetch_video_assets(item: dict) -> None:
    rdir = raw_dir(item)
    rdir.mkdir(parents=True, exist_ok=True)
    target_prefix = rdir / f"{item['playlist_index']:02d}_{item['video_id']}"
    info_json = target_prefix.with_suffix(".info.json")
    existing_srts = list(rdir.glob("*.srt"))
    existing_jpgs = list(rdir.glob("*.jpg"))
    if info_json.exists():
        info = json.loads(info_json.read_text())
        if info.get("id") == item["video_id"] and info.get("_type") != "playlist" and existing_srts and existing_jpgs:
            return
        if info.get("id") != item["video_id"] or info.get("_type") == "playlist":
            clear_dir(rdir)

    base_cmd = [
        "yt-dlp",
        "--no-playlist",
        "--skip-download",
        "-o",
        str(target_prefix) + ".%(ext)s",
    ]

    run(
        base_cmd
        + [
            "--write-info-json",
            "--write-thumbnail",
            "--convert-thumbnails",
            "jpg",
            item["video_url"],
        ]
    )

    run(
        base_cmd
        + [
            "--write-subs",
            "--sub-langs",
            "en-US,en,en-GB",
            "--sub-format",
            "srt/vtt/best",
            "--convert-subs",
            "srt",
            item["video_url"],
        ],
        check=False,
    )
    if not list(rdir.glob("*.srt")):
        run(
            base_cmd
            + [
                "--write-auto-subs",
                "--sub-langs",
                "en-US,en,en-GB",
                "--sub-format",
                "srt/vtt/best",
                "--convert-subs",
                "srt",
                item["video_url"],
            ],
            check=False,
        )


def write_course_bundle() -> None:
    bundle = []
    for item in LECTURES:
        meta_path = lecture_dir(item) / "meta.json"
        if meta_path.exists():
            bundle.append(json.loads(meta_path.read_text()))
    (RUN_ROOT / "text" / "course_bundle.json").write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n")


def write_course_manifest_seed() -> None:
    manifest = {
        "course_id": COURSE_ID,
        "title": "Stanford CS224R Deep Reinforcement Learning",
        "playlist_origin_url": PLAYLIST_URL,
        "course_page_url": COURSE_PAGE_URL,
        "course_mode": True,
        "lecture_count": len(LECTURES),
        "lectures": [
            {
                "lecture_id": f"{item['playlist_index']:02d}",
                "lecture_slug": f"{item['playlist_index']:02d}_{item['slug']}",
                "title": item["title"],
                "date": item["date"],
                "video_url": item["video_url"],
                "slide_url": item["slide_url"],
                "kind": item["kind"],
            }
            for item in LECTURES
        ],
    }
    (RUN_ROOT / "build" / "course_manifest_seed.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def write_lectures_index() -> None:
    lines = ["# CS224R Lecture Folders", ""]
    for item in LECTURES:
        slug = f"{item['playlist_index']:02d}_{item['slug']}"
        lines.append(f"- [{item['playlist_index']:02d} {item['title']}](./{slug}/README.md)")
    (RUN_ROOT / "lectures" / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ensure_dirs()
    download_course_pages()
    download_slides()
    fetch_playlist_metadata()
    for item in LECTURES:
        fetch_video_assets(item)
        write_text_bundle(item)
        write_lecture_dir(item)
    write_course_bundle()
    write_course_manifest_seed()
    write_lectures_index()
    subprocess.run(
        [
            "python3",
            str(REPO_ROOT / "scripts" / "video_note_harness" / "bootstrap_harness.py"),
            "--run-root",
            str(RUN_ROOT),
        ],
        check=True,
    )
    print(f"bootstrapped={len(LECTURES)}")
    print(RUN_ROOT)


if __name__ == "__main__":
    main()
