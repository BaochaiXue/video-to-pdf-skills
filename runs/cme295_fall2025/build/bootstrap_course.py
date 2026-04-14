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
TEMPLATE_TEX = REPO_ROOT / "skills" / "youtube-render-pdf" / "assets" / "notes-template.tex"

COURSE_ID = "stanford-cme295-autumn-2025"
PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy"
SYLLABUS_URL = "https://cme295.stanford.edu/syllabus/"

LECTURES = [
    {
        "playlist_index": 1,
        "date": "2025-09-26",
        "title": "Lecture 1: Transformer",
        "title_short": "Transformer",
        "slug": "transformer",
        "video_id": "Ub3GoFaUcds",
        "video_url": "https://www.youtube.com/watch?v=Ub3GoFaUcds&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy&index=1",
        "slide_url": "https://cme295.stanford.edu/slides/fall25-cme295-lecture1.pdf",
        "topics": [
            "Background on NLP and tasks",
            "Tokenization",
            "Embeddings",
            "Word2vec, RNN, LSTM",
            "Attention mechanism",
            "Transformer architecture",
        ],
    },
    {
        "playlist_index": 2,
        "date": "2025-10-03",
        "title": "Lecture 2: Transformer-based models & tricks",
        "title_short": "Transformer-based models & tricks",
        "slug": "transformer_models_and_tricks",
        "video_id": "yT84Y5zCnaA",
        "video_url": "https://www.youtube.com/watch?v=yT84Y5zCnaA&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy&index=2",
        "slide_url": "https://cme295.stanford.edu/slides/fall25-cme295-lecture2.pdf",
        "topics": [
            "Attention approximation",
            "MHA, MQA, GQA",
            "Position embeddings (regular, learned)",
            "RoPE and applications",
            "Transformer-based architectures",
            "BERT and its derivatives",
        ],
    },
    {
        "playlist_index": 3,
        "date": "2025-10-10",
        "title": "Lecture 3: Large Language Models",
        "title_short": "Large Language Models",
        "slug": "large_language_models",
        "video_id": "Q5baLehv5So",
        "video_url": "https://www.youtube.com/watch?v=Q5baLehv5So&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy&index=3",
        "slide_url": "https://cme295.stanford.edu/slides/fall25-cme295-lecture3.pdf",
        "topics": [
            "Definition and architecture",
            "Mixture of experts",
            "Context length, temperature",
            "Sampling strategies",
            "Prompting, in-context learning",
            "Chain of thought",
            "Self-consistency",
        ],
    },
    {
        "playlist_index": 4,
        "date": "2025-10-17",
        "title": "Lecture 4: LLM training",
        "title_short": "LLM training",
        "slug": "llm_training",
        "video_id": "VlA_jt_3Qc4",
        "video_url": "https://www.youtube.com/watch?v=VlA_jt_3Qc4&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy&index=4",
        "slide_url": "https://cme295.stanford.edu/slides/fall25-cme295-lecture4.pdf",
        "topics": [
            "Pretraining",
            "Quantization",
            "Hardware optimization",
            "Supervised finetuning (SFT)",
            "Parameter-efficient finetuning (LoRA)",
        ],
    },
    {
        "playlist_index": 5,
        "date": "2025-10-31",
        "title": "Lecture 5: LLM tuning",
        "title_short": "LLM tuning",
        "slug": "llm_tuning",
        "video_id": "PmW_TMQ3l0I",
        "video_url": "https://www.youtube.com/watch?v=PmW_TMQ3l0I&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy&index=5",
        "slide_url": "https://cme295.stanford.edu/slides/fall25-cme295-lecture5.pdf",
        "topics": [
            "Preference tuning",
            "RLHF overview",
            "Reward modeling",
            "RL approaches (PPO and variants)",
            "DPO",
        ],
    },
    {
        "playlist_index": 6,
        "date": "2025-11-07",
        "title": "Lecture 6: LLM reasoning",
        "title_short": "LLM reasoning",
        "slug": "llm_reasoning",
        "video_id": "k5Fh-UgTuCo",
        "video_url": "https://www.youtube.com/watch?v=k5Fh-UgTuCo&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy&index=6",
        "slide_url": "https://cme295.stanford.edu/slides/fall25-cme295-lecture6.pdf",
        "topics": [
            "Reasoning models",
            "RL for reasoning",
            "GRPO",
            "Scaling",
        ],
    },
    {
        "playlist_index": 7,
        "date": "2025-11-14",
        "title": "Lecture 7: Agentic LLMs",
        "title_short": "Agentic LLMs",
        "slug": "agentic_llms",
        "video_id": "h-7S6HNq0Vg",
        "video_url": "https://www.youtube.com/watch?v=h-7S6HNq0Vg&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy&index=7",
        "slide_url": "https://cme295.stanford.edu/slides/fall25-cme295-lecture7.pdf",
        "topics": [
            "Retrieval-augmented generation",
            "Advanced RAG techniques",
            "Function calling",
            "Agents",
            "ReAct framework",
        ],
    },
    {
        "playlist_index": 8,
        "date": "2025-11-21",
        "title": "Lecture 8: LLM evaluation",
        "title_short": "LLM evaluation",
        "slug": "llm_evaluation",
        "video_id": "8fNP4N46RRo",
        "video_url": "https://www.youtube.com/watch?v=8fNP4N46RRo&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy&index=8",
        "slide_url": "https://cme295.stanford.edu/slides/fall25-cme295-lecture8.pdf",
        "topics": [
            "LLM-as-a-judge overview",
            "Best practices and benefits",
            "Biases and pitfalls",
        ],
    },
    {
        "playlist_index": 9,
        "date": "2025-12-05",
        "title": "Lecture 9: Current trends",
        "title_short": "Current trends",
        "slug": "current_trends",
        "video_id": "Q86qzJ1K1Ss",
        "video_url": "https://www.youtube.com/watch?v=Q86qzJ1K1Ss&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy&index=9",
        "slide_url": "https://cme295.stanford.edu/slides/fall25-cme295-lecture9.pdf",
        "topics": [
            "Recap",
            "Trending topics",
            "Closing thoughts",
        ],
    },
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
        if ".en.srt" in name:
            score = 10
        elif ".en-orig.srt" in name:
            score = 20
        elif ".en-US.srt" in name or ".en-GB.srt" in name:
            score = 30
        elif ".en." in name:
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
    generated = sorted(out_dir.glob("page-*.png"))
    for idx, path in enumerate(generated, start=1):
        target = out_dir / f"page-{idx:02d}.png"
        if path != target:
            path.rename(target)


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

    meta = {
        **item,
        "course_id": COURSE_ID,
        "playlist_url": PLAYLIST_URL,
        "syllabus_url": SYLLABUS_URL,
        "thumbnail": str(next(raw_dir(item).glob("*.jpg")).relative_to(RUN_ROOT)) if list(raw_dir(item).glob("*.jpg")) else None,
        "subtitle": str(best_subtitle_path(item).relative_to(RUN_ROOT)) if best_subtitle_path(item) else None,
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
    pages_src = ldir / "pdf_pages"
    render_slide_pages(lecture_slide_pdf(item), pages_src)

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
        f"- Video: [YouTube]({item['video_url']})",
        f"- Slides: [slides.pdf](slides.pdf)",
        f"- Cover: [cover.jpg](cover.jpg)" if (ldir / "cover.jpg").exists() else "- Cover: unavailable",
        f"- Subtitle: [subtitle.srt](subtitle.srt)" if (ldir / "subtitle.srt").exists() else "- Subtitle: unavailable",
        f"- Transcript: [transcript.txt](transcript.txt)" if (ldir / "transcript.txt").exists() else "- Transcript: unavailable",
        "- Official text: [official.txt](official.txt)",
        "- Slide pages: `pdf_pages/page-*.png`",
        "",
        "## Topics",
        "",
    ]
    for topic in item["topics"]:
        summary_lines.append(f"- {topic}")
    summary_lines.extend(
        [
            "",
            "## Writing requirements",
            "",
            "- Use Chinese.",
            "- Preserve coverage of both video and slides.",
            "- Include diagrams or slide figures as note figures.",
            "- End each major section with `本章小结` and end the document with `总结与延伸`.",
        ]
    )
    (ldir / "README.md").write_text("\n".join(summary_lines) + "\n")


def download_syllabus() -> None:
    download(SYLLABUS_URL, RUN_ROOT / "meta" / "syllabus.html")


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
    video_url = f"https://www.youtube.com/watch?v={item['video_id']}"
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
            video_url,
        ]
    )

    manual_result = run(
        base_cmd
        + [
            "--write-subs",
            "--sub-langs",
            "en-US,en,en-GB",
            "--sub-format",
            "srt/vtt/best",
            "--convert-subs",
            "srt",
            video_url,
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
                video_url,
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
        "title": "Stanford CME 295: Transformers and Large Language Models I",
        "playlist_origin_url": PLAYLIST_URL,
        "syllabus_url": SYLLABUS_URL,
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
            }
            for item in LECTURES
        ],
    }
    (RUN_ROOT / "build" / "course_manifest_seed.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def write_lectures_index() -> None:
    lines = ["# CME 295 Lecture Folders", ""]
    for item in LECTURES:
        slug = f"{item['playlist_index']:02d}_{item['slug']}"
        lines.append(f"- [{item['playlist_index']:02d} {item['title']}](./{slug}/README.md)")
    (RUN_ROOT / "lectures" / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ensure_dirs()
    download_syllabus()
    download_slides()
    fetch_playlist_metadata()
    for item in LECTURES:
        fetch_video_assets(item)
        write_text_bundle(item)
        write_lecture_dir(item)
    write_course_bundle()
    write_course_manifest_seed()
    write_lectures_index()
    print(f"bootstrapped={len(LECTURES)}")
    print(RUN_ROOT)


if __name__ == "__main__":
    main()
