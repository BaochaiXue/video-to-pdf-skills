#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import math
import re
import subprocess
import urllib.request
from pathlib import Path
from typing import TypeVar


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


def clean_subtitle_payload(lines: list[str]) -> list[str]:
    payload = [re.sub(r"<[^>]+>", "", line).strip() for line in lines]
    return [line for line in payload if line]


def parse_srt_entries(srt_path: Path) -> list[dict]:
    text = srt_path.read_text(errors="ignore")
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    entries: list[dict] = []
    for block in blocks:
        raw_lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(raw_lines) < 2:
            continue
        maybe_ts = raw_lines[1] if raw_lines[0].isdigit() else raw_lines[0]
        if "-->" not in maybe_ts:
            continue
        payload = raw_lines[2:] if raw_lines[0].isdigit() else raw_lines[1:]
        payload = clean_subtitle_payload(payload)
        if payload:
            start, end = [part.strip() for part in maybe_ts.split("-->", 1)]
            entries.append(
                {
                    "start": start,
                    "end": end,
                    "text": " ".join(payload),
                }
            )
    return entries


def srt_to_text(srt_path: Path) -> str:
    lines = []
    for entry in parse_srt_entries(srt_path):
        lines.append(f"[{entry['start']} --> {entry['end']}] {entry['text']}")
    return "\n".join(lines).strip() + "\n"


def extract_slide_text(pdf_path: Path, out_path: Path) -> None:
    run(["pdftotext", str(pdf_path), str(out_path)])


def extract_slide_pages(pdf_path: Path) -> list[str]:
    proc = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    pages = proc.stdout.replace("\r\n", "\n").split("\f")
    cleaned: list[str] = []
    for page in pages:
        page = re.sub(r"[ \t]+\n", "\n", page)
        page = re.sub(r"\n{3,}", "\n\n", page.strip())
        cleaned.append(page)
    while cleaned and not cleaned[-1]:
        cleaned.pop()
    return cleaned


def write_jsonl(path: Path, rows: list[dict]) -> None:
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(payload + ("\n" if payload else ""))


def build_transcript_units(subtitle_path: Path) -> list[dict]:
    entries = parse_srt_entries(subtitle_path)
    return [
        {
            "unit_id": f"sub_{idx:04d}",
            "source_type": "subtitle_span",
            "source_id": "subtitle_srt",
            "loc": {
                "start": entry["start"],
                "end": entry["end"],
            },
            "text": entry["text"],
            "required": True,
        }
        for idx, entry in enumerate(entries, start=1)
    ]


def build_slide_units(slide_pages: list[str], lecture_dir_path: Path) -> list[dict]:
    units: list[dict] = []
    for idx, page_text in enumerate(slide_pages, start=1):
        asset_path = lecture_dir_path / "pdf_pages" / f"page-{idx:02d}.png"
        units.append(
            {
                "unit_id": f"slide_{idx:04d}",
                "source_type": "slide_page",
                "source_id": "slides_pdf",
                "loc": {
                    "page": idx,
                },
                "text": page_text,
                "asset_path": str(asset_path.relative_to(lecture_dir_path)),
                "required": bool(page_text.strip()),
            }
        )
    return units


T = TypeVar("T")


def slice_evenly(items: list[T], parts: int) -> list[list[T]]:
    if parts <= 0:
        return [items]
    if not items:
        return [[] for _ in range(parts)]
    chunk_size = math.ceil(len(items) / parts)
    chunks = [items[idx : idx + chunk_size] for idx in range(0, len(items), chunk_size)]
    while len(chunks) < parts:
        chunks.append([])
    return chunks[:parts]


def build_segments(item: dict, transcript_units: list[dict], slide_units: list[dict]) -> list[dict]:
    topic_hints = item.get("topics") or []
    segment_count = max(1, len(topic_hints), math.ceil(max(1, len(transcript_units)) / 150))

    transcript_chunks = slice_evenly(transcript_units, segment_count)
    slide_chunks = slice_evenly([unit for unit in slide_units if unit["required"]], segment_count)

    segments: list[dict] = []
    for idx in range(segment_count):
        transcript_chunk = transcript_chunks[idx] if idx < len(transcript_chunks) else []
        slide_chunk = slide_chunks[idx] if idx < len(slide_chunks) else []
        source_unit_ids = [unit["unit_id"] for unit in transcript_chunk] + [unit["unit_id"] for unit in slide_chunk]
        if transcript_chunk:
            start = transcript_chunk[0]["loc"]["start"]
            end = transcript_chunk[-1]["loc"]["end"]
        else:
            start = None
            end = None
        target_hint = topic_hints[idx] if idx < len(topic_hints) else f"Segment {idx + 1}"
        segments.append(
            {
                "segment_id": f"seg_{idx + 1:02d}",
                "start": start,
                "end": end,
                "source_unit_ids": source_unit_ids,
                "target_section_hint": target_hint,
            }
        )
    return segments


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
    if subtitle:
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

    subtitle_path = best_subtitle_path(item)
    course_mode = True
    segmentation_required = True

    meta = {
        **item,
        "course_id": COURSE_ID,
        "course_mode": course_mode,
        "segmentation_required": segmentation_required,
        "playlist_url": PLAYLIST_URL,
        "syllabus_url": SYLLABUS_URL,
        "thumbnail": str(next(raw_dir(item).glob("*.jpg")).relative_to(RUN_ROOT)) if list(raw_dir(item).glob("*.jpg")) else None,
        "subtitle": str(subtitle_path.relative_to(RUN_ROOT)) if subtitle_path else None,
        "material": str(lecture_slide_pdf(item).relative_to(RUN_ROOT)),
        "transcript_text": str((text_dir(item) / "transcript.txt").relative_to(RUN_ROOT)) if (text_dir(item) / "transcript.txt").exists() else None,
        "official_text": str((text_dir(item) / "official.txt").relative_to(RUN_ROOT)),
        "transcript_jsonl": str((ldir / "transcript.jsonl").relative_to(RUN_ROOT)),
        "slides_jsonl": str((ldir / "slides.jsonl").relative_to(RUN_ROOT)),
        "segments_jsonl": str((ldir / "segments.jsonl").relative_to(RUN_ROOT)),
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

    transcript_units = build_transcript_units(RUN_ROOT / meta["subtitle"]) if meta["subtitle"] else []
    slide_units = build_slide_units(extract_slide_pages(lecture_slide_pdf(item)), ldir)
    segments = build_segments(item, transcript_units, slide_units)

    write_jsonl(ldir / "transcript.jsonl", transcript_units)
    write_jsonl(ldir / "slides.jsonl", slide_units)
    write_jsonl(ldir / "segments.jsonl", segments)

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
        "- Structured transcript: [transcript.jsonl](transcript.jsonl)",
        "- Structured slides: [slides.jsonl](slides.jsonl)",
        "- Segment plan: [segments.jsonl](segments.jsonl)",
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
            "- Treat `transcript.jsonl`, `slides.jsonl`, and `segments.jsonl` as the primary structured evidence layer.",
            "- Generate non-empty `coverage_units.jsonl` before writing or revising `lecture_XX_note.tex`.",
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
