#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse

import requests
from bs4 import BeautifulSoup


RUN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RUN_ROOT.parents[2]

COURSE_ID = "ucb-cs294-194-196-agentic-ai-fall-2025"
COURSE_TITLE = "UCB CS294/194-196: Agentic AI (Fall 2025)"
COURSE_PAGE_URL = "https://rdi.berkeley.edu/agentic-ai/f25"
MOOC_PAGE_URL = "https://agenticai-learning.org/f25"
PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLS01nW3RtgoqGkm4UeqNeZLccW-OGc1fJ"


def ensure_dirs() -> None:
    for dirname in ["build", "lectures", "materials/slides", "materials/readings", "meta", "raw", "text"]:
        (RUN_ROOT / dirname).mkdir(parents=True, exist_ok=True)


def slugify(text: str) -> str:
    text = text.lower().replace("&", "and").replace("/", "_")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload + ("\n" if payload else ""))


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, check=check, text=True)


def download_text(url: str, dest: Path) -> str:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(response.text)
    return response.text


def download_binary(url: str, dest: Path) -> None:
    if dest.exists():
        return
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)


def lecture_dir(item: dict) -> Path:
    return RUN_ROOT / "lectures" / f"{item['lecture_id']}_{item['slug']}"


def raw_dir(item: dict) -> Path:
    if not item.get("video_id"):
        return RUN_ROOT / "raw" / f"{item['lecture_id']}_no_video"
    return RUN_ROOT / "raw" / f"{item['lecture_id']}_{item['video_id']}"


def text_dir(item: dict) -> Path:
    if not item.get("video_id"):
        return RUN_ROOT / "text" / f"{item['lecture_id']}_no_video"
    return RUN_ROOT / "text" / f"{item['lecture_id']}_{item['video_id']}"


def lecture_slide_pdf(item: dict) -> Path:
    return RUN_ROOT / "materials" / "slides" / f"{item['lecture_id']}_{item['slug']}.pdf"


def reading_dir(item: dict) -> Path:
    return RUN_ROOT / "materials" / "readings" / f"{item['lecture_id']}_{item['slug']}"


def first_or_none(values: list[str] | None) -> str | None:
    return values[0] if values else None


def iso_date_from_text(text: str) -> str:
    months = {
        "jan": "01",
        "feb": "02",
        "mar": "03",
        "apr": "04",
        "may": "05",
        "jun": "06",
        "jul": "07",
        "aug": "08",
        "sep": "09",
        "oct": "10",
        "nov": "11",
        "dec": "12",
    }
    match = re.search(r"([A-Za-z]{3})\s+(\d{1,2})", text)
    if not match:
        return text
    month = months[match.group(1).lower()]
    day = int(match.group(2))
    return f"2025-{month}-{day:02d}"


def parse_playlist_id(url: str) -> str | None:
    parsed = urlparse(url)
    values = parse_qs(parsed.query).get("list")
    return values[0] if values else None


def parse_video_id(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.netloc.endswith("youtu.be"):
        return parsed.path.strip("/") or None
    if parsed.netloc.endswith("youtube.com") and parsed.path.startswith("/live/"):
        parts = [part for part in parsed.path.split("/") if part]
        return parts[1] if len(parts) >= 2 else None
    values = parse_qs(parsed.query).get("v")
    return values[0] if values else None


def collect_speaker_names(soup: BeautifulSoup) -> list[str]:
    names: list[str] = []
    for img in soup.find_all("img"):
        alt = (img.get("alt") or "").strip()
        if alt and alt not in names:
            names.append(alt)
    for anchor in soup.find_all("a"):
        text = anchor.get_text(" ", strip=True)
        if text and len(text.split()) <= 4 and any(part[:1].isupper() for part in text.split()):
            if text not in names and not text.startswith("Slides") and not text.startswith("Recording"):
                names.append(text)
    return sorted(names, key=len, reverse=True)


def split_title_and_speaker(text: str, speaker_names: list[str]) -> tuple[str, str | None, str | None]:
    cleaned = re.sub(r"\s*\[\s*(Slides|Recording|Livestream|Quiz)\s*\]\s*", " ", text).strip()
    for name in speaker_names:
        marker = f" {name}"
        idx = cleaned.find(marker)
        if idx != -1:
            title = cleaned[:idx].strip(" -")
            rest = cleaned[idx + 1 + len(name) :].strip(" ,")
            return title or cleaned, name, rest or None
    return cleaned, None, None


def parse_schedule_rows() -> tuple[list[dict], dict]:
    rdi_html = download_text(COURSE_PAGE_URL, RUN_ROOT / "meta" / "course_page.html")
    mooc_html = download_text(MOOC_PAGE_URL, RUN_ROOT / "meta" / "mooc_page.html")

    rdi_soup = BeautifulSoup(rdi_html, "html.parser")
    mooc_soup = BeautifulSoup(mooc_html, "html.parser")
    speaker_names = collect_speaker_names(rdi_soup)

    table = rdi_soup.find_all("table")[2]
    mooc_table = mooc_soup.find_all("table")[2]

    mooc_by_date: dict[str, dict] = {}
    for row in mooc_table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if len(cells) != 3:
            continue
        date_text = cells[0].get_text(" ", strip=True)
        links = [
            {"label": a.get_text(" ", strip=True), "url": urljoin(MOOC_PAGE_URL, a.get("href") or "")}
            for a in row.find_all("a")
        ]
        mooc_by_date[date_text] = {"links": links}

    lectures: list[dict] = []
    course_gaps: list[dict] = []
    lecture_counter = 1
    for row in table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if len(cells) != 3:
            continue
        date_text = cells[0].get_text(" ", strip=True)
        lecture_text = cells[1].get_text(" ", strip=True)
        reading_links = [
            {"label": a.get_text(" ", strip=True), "url": urljoin(COURSE_PAGE_URL, a.get("href") or "")}
            for a in cells[2].find_all("a")
        ]
        lecture_links = [
            {"label": a.get_text(" ", strip=True), "url": urljoin(COURSE_PAGE_URL, a.get("href") or "")}
            for a in cells[1].find_all("a")
        ]
        mooc_links = mooc_by_date.get(date_text, {}).get("links", [])
        combined_links = {row["url"]: row for row in lecture_links}
        for row_link in mooc_links:
            combined_links.setdefault(row_link["url"], row_link)
        lecture_links = list(combined_links.values())

        if lecture_text.lower().startswith("no lecture"):
            course_gaps.append(
                {
                    "date": iso_date_from_text(date_text),
                    "date_text": date_text,
                    "description": lecture_text,
                    "reason": "Scheduled no-lecture week.",
                }
            )
            continue

        title_short, speaker, affiliation = split_title_and_speaker(lecture_text, speaker_names)
        slide_urls = [row["url"] for row in lecture_links if row["label"].lower() == "slides"]
        recording_url = next((row["url"] for row in lecture_links if row["label"].lower() in {"recording", "livestream"}), None)
        quiz_urls = [row["url"] for row in lecture_links if row["label"].lower() == "quiz"]
        video_id = parse_video_id(recording_url) if recording_url else None

        item = {
            "lecture_id": f"{lecture_counter:02d}",
            "playlist_index": lecture_counter,
            "date": iso_date_from_text(date_text),
            "date_text": date_text,
            "title": title_short,
            "title_short": title_short,
            "slug": slugify(title_short),
            "speaker": speaker,
            "affiliation": affiliation,
            "slide_urls": slide_urls,
            "reading_links": reading_links,
            "quiz_urls": quiz_urls,
            "video_url": recording_url,
            "video_id": video_id,
            "course_page_url": COURSE_PAGE_URL,
            "mooc_page_url": MOOC_PAGE_URL,
            "public_video_available": bool(recording_url),
        }
        if not recording_url:
            course_gaps.append(
                {
                    "date": item["date"],
                    "date_text": date_text,
                    "description": title_short,
                    "reason": "No public recording URL was listed on the official course pages.",
                }
            )
        lectures.append(item)
        lecture_counter += 1

    metadata = {
        "course_id": COURSE_ID,
        "course_title": COURSE_TITLE,
        "course_page_url": COURSE_PAGE_URL,
        "mooc_page_url": MOOC_PAGE_URL,
        "playlist_url": PLAYLIST_URL,
        "course_gaps": course_gaps,
    }
    return lectures, metadata


def fetch_playlist_metadata() -> dict:
    out = RUN_ROOT / "meta" / "playlist_full.json"
    if out.exists():
        return json.loads(out.read_text())
    output = subprocess.check_output(["yt-dlp", "--flat-playlist", "--dump-single-json", PLAYLIST_URL], text=True)
    out.write_text(output)
    return json.loads(output)


def attach_playlist_context(lectures: list[dict], playlist_meta: dict) -> tuple[list[dict], list[dict]]:
    entries = playlist_meta.get("entries") or []
    entries_by_id = {entry.get("id"): entry for entry in entries if entry.get("id")}
    playlist_gaps: list[dict] = []
    attached: list[dict] = []
    for item in lectures:
        entry = entries_by_id.get(item.get("video_id")) if item.get("video_id") else None
        if item.get("video_id") and entry is None:
            playlist_gaps.append(
                {
                    "lecture_id": item["lecture_id"],
                    "title": item["title"],
                    "video_url": item["video_url"],
                    "reason": "Recording URL exists on the official course page but is absent from the public playlist.",
                }
            )
        attached.append(
            {
                **item,
                "playlist_url": PLAYLIST_URL,
                "playlist_title": playlist_meta.get("title"),
                "playlist_channel": playlist_meta.get("channel") or playlist_meta.get("uploader"),
                "playlist_entry_title": entry.get("title") if entry else None,
            }
        )
    return attached, playlist_gaps


def fetch_video_assets(item: dict) -> None:
    if not item.get("video_url") or not item.get("video_id"):
        return
    rdir = raw_dir(item)
    rdir.mkdir(parents=True, exist_ok=True)
    prefix = rdir / f"{item['lecture_id']}_{item['video_id']}"
    info_json = prefix.with_suffix(".info.json")
    existing_srts = list(rdir.glob("*.srt"))
    existing_jpgs = list(rdir.glob("*.jpg"))
    if info_json.exists() and existing_srts and existing_jpgs:
        return

    base_cmd = [
        "yt-dlp",
        "--no-playlist",
        "--skip-download",
        "-o",
        str(prefix) + ".%(ext)s",
    ]
    subprocess.run(
        base_cmd + ["--write-info-json", "--write-thumbnail", "--convert-thumbnails", "jpg", item["video_url"]],
        check=True,
        text=True,
    )
    subprocess.run(
        base_cmd
        + [
            "--write-subs",
            "--sub-langs",
            "en,en-US,en-GB",
            "--sub-format",
            "srt/vtt/best",
            "--convert-subs",
            "srt",
            item["video_url"],
        ],
        check=False,
        text=True,
    )
    if not list(rdir.glob("*.srt")):
        subprocess.run(
            base_cmd
            + [
                "--write-auto-subs",
                "--sub-langs",
                "en,en-US,en-GB",
                "--sub-format",
                "srt/vtt/best",
                "--convert-subs",
                "srt",
                item["video_url"],
            ],
            check=False,
            text=True,
        )


def best_subtitle_path(item: dict) -> Path | None:
    if not item.get("video_url"):
        return None
    candidates = sorted(raw_dir(item).glob("*.srt"))
    if not candidates:
        return None
    scored = []
    for path in candidates:
        name = path.name
        score = 100
        if ".en.srt" in name:
            score = 10
        elif ".en-orig.srt" in name:
            score = 20
        elif ".en-US.srt" in name or ".en-GB.srt" in name:
            score = 30
        scored.append((score, name, path))
    scored.sort()
    return scored[0][2]


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
    if not pdf_path.exists():
        out_path.write_text("No official slide PDF was posted for this lecture.\n")
        return
    run(["pdftotext", str(pdf_path), str(out_path)])


def render_slide_pages(pdf_path: Path, out_dir: Path) -> None:
    if not pdf_path.exists():
        return
    if out_dir.exists() and any(out_dir.glob("page-*.png")):
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "page"
    run(["pdftoppm", "-png", str(pdf_path), str(prefix)])
    for idx, path in enumerate(sorted(out_dir.glob("page-*.png")), start=1):
        target = out_dir / f"page-{idx:02d}.png"
        if path != target:
            path.rename(target)


def download_slide_assets(item: dict) -> Path | None:
    slide_urls = item.get("slide_urls") or []
    if not slide_urls:
        return None
    bundle = lecture_slide_pdf(item)
    if bundle.exists():
        return bundle
    download_binary(slide_urls[0], bundle)
    return bundle


def reading_extension(url: str, content_type: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix:
        return suffix
    if "pdf" in content_type:
        return ".pdf"
    if "html" in content_type:
        return ".html"
    return ".bin"


def download_readings(item: dict) -> Path:
    rdir = reading_dir(item)
    rdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for idx, reading in enumerate(item.get("reading_links") or [], start=1):
        url = reading["url"]
        label = reading["label"]
        row = {
            "reading_id": f"reading_{idx:02d}",
            "label": label,
            "url": url,
            "local_path": None,
            "status": "missing",
            "notes": "",
        }
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            ext = reading_extension(url, response.headers.get("content-type", ""))
            dest = rdir / f"{idx:02d}_{slugify(label) or 'reading'}{ext}"
            if not dest.exists():
                if "pdf" in response.headers.get("content-type", "") or ext == ".pdf":
                    dest.write_bytes(response.content)
                else:
                    dest.write_text(response.text)
            row["local_path"] = str(dest.relative_to(RUN_ROOT))
            row["status"] = "available"
        except Exception as exc:  # noqa: BLE001
            row["notes"] = f"download_failed: {exc}"
        rows.append(row)
    manifest = rdir / "manifest.json"
    write_json(manifest, rows)
    return manifest


def write_text_bundle(item: dict, slide_pdf: Path | None) -> None:
    tdir = text_dir(item)
    tdir.mkdir(parents=True, exist_ok=True)
    subtitle = best_subtitle_path(item)
    transcript_path = tdir / "transcript.txt"
    if subtitle:
        transcript_path.write_text(srt_to_text(subtitle))
    else:
        transcript_path.write_text("No public subtitle track is available for this lecture.\n")
    extract_slide_text(slide_pdf or Path("__missing__.pdf"), tdir / "official.txt")


def ensure_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(os.path.relpath(src, dst.parent))


def placeholder_slides_jsonl(lecture_dir_path: Path) -> None:
    rows = [
        {
            "unit_id": "slide_missing_0001",
            "source_type": "missing_slide_notice",
            "source_id": "schedule_notice",
            "loc": {"page": None},
            "text": "No official slide PDF was posted for this lecture.",
            "asset_path": None,
            "required": False,
        }
    ]
    write_jsonl(lecture_dir_path / "slides.jsonl", rows)


def build_source_manifest(item: dict, lecture_dir_path: Path, reading_manifest: Path) -> dict:
    sources = [
        {
            "source_id": "lecture_meta",
            "source_type": "lecture_metadata",
            "origin_url": item.get("course_page_url"),
            "local_path": str((lecture_dir_path / "meta.json").relative_to(RUN_ROOT)),
            "required_for_coverage": True,
            "status": "available",
            "notes": "Normalized lecture metadata for harness-managed note generation.",
        },
        {
            "source_id": "course_page_rdi",
            "source_type": "official_course_page",
            "origin_url": COURSE_PAGE_URL,
            "local_path": "meta/course_page.html",
            "required_for_coverage": True,
            "status": "available",
            "notes": "Primary Berkeley RDI course page.",
        },
        {
            "source_id": "course_page_mooc",
            "source_type": "official_mooc_page",
            "origin_url": MOOC_PAGE_URL,
            "local_path": "meta/mooc_page.html",
            "required_for_coverage": True,
            "status": "available",
            "notes": "Public MOOC syllabus page with duplicated lecture links.",
        },
        {
            "source_id": "playlist_metadata",
            "source_type": "official_playlist_metadata",
            "origin_url": PLAYLIST_URL,
            "local_path": "meta/playlist_full.json",
            "required_for_coverage": False,
            "status": "available",
            "notes": "Public YouTube playlist metadata when available.",
        },
        {
            "source_id": "transcript_txt",
            "source_type": "debug_transcript_text",
            "origin_url": item.get("video_url"),
            "local_path": str((lecture_dir_path / "transcript.txt").relative_to(RUN_ROOT)),
            "required_for_coverage": False,
            "status": "available",
            "notes": "Human-readable transcript or placeholder.",
        },
        {
            "source_id": "official_txt",
            "source_type": "debug_slide_text",
            "origin_url": first_or_none(item.get("slide_urls")),
            "local_path": str((lecture_dir_path / "official.txt").relative_to(RUN_ROOT)),
            "required_for_coverage": False,
            "status": "available",
            "notes": "Human-readable official slide extract or placeholder.",
        },
        {
            "source_id": "reading_manifest",
            "source_type": "official_reading_manifest",
            "origin_url": item.get("course_page_url"),
            "local_path": str(reading_manifest.relative_to(RUN_ROOT)),
            "required_for_coverage": False,
            "status": "available",
            "notes": "Downloaded supplemental reading links and local copies.",
        },
    ]

    if item.get("video_url") and item.get("video_id"):
        info_files = list(raw_dir(item).glob("*.info.json"))
        if info_files:
            sources.append(
                {
                    "source_id": "raw_info_json",
                    "source_type": "platform_metadata",
                    "origin_url": item["video_url"],
                    "local_path": str(info_files[0].relative_to(RUN_ROOT)),
                    "required_for_coverage": True,
                    "status": "available",
                    "notes": "Original yt-dlp metadata dump.",
                }
            )
    subtitle = lecture_dir_path / "subtitle.srt"
    sources.append(
        {
            "source_id": "subtitle_srt",
            "source_type": "platform_subtitle",
            "origin_url": item.get("video_url"),
            "local_path": str(subtitle.relative_to(RUN_ROOT)) if subtitle.exists() else None,
            "required_for_coverage": bool(item.get("video_url")),
            "status": "available" if subtitle.exists() else "missing",
            "notes": "Best available subtitle track or missing if the recording has none.",
        }
    )

    slides_pdf = lecture_dir_path / "slides.pdf"
    sources.append(
        {
            "source_id": "slides_pdf",
            "source_type": "official_slide_pdf",
            "origin_url": first_or_none(item.get("slide_urls")),
            "local_path": str(slides_pdf.relative_to(RUN_ROOT)) if slides_pdf.exists() else None,
            "required_for_coverage": bool(item.get("slide_urls")),
            "status": "available" if slides_pdf.exists() else "missing",
            "notes": "Official lecture slides if posted.",
        }
    )
    return {
        "course_id": COURSE_ID,
        "course_mode": True,
        "lecture_id": item["lecture_id"],
        "lecture_slug": lecture_dir_path.name,
        "title": item["title"],
        "origin_url": item.get("video_url") or item.get("course_page_url"),
        "slide_origin_urls": item.get("slide_urls") or [],
        "sources": sources,
    }


def seed_source_gap_omissions(item: dict, lecture_dir_path: Path, reading_manifest: Path, playlist_gaps: dict[str, str]) -> None:
    omission_path = lecture_dir_path / "omission_log.jsonl"
    rows: list[dict] = []
    if not item.get("video_url"):
        rows.append(
            {
                "unit_id": "source_gap_recording_missing",
                "reason": "No public recording URL was posted on the official course pages.",
                "impact": "moderate",
                "user_visible_note": "本讲只有官方 slides / readings，没有公开视频，因此讲义会以课件和课程页为主做 best effort 重建。",
            }
        )
    elif item["lecture_id"] in playlist_gaps:
        rows.append(
            {
                "unit_id": "source_gap_playlist_missing",
                "reason": playlist_gaps[item["lecture_id"]],
                "impact": "low",
                "user_visible_note": "本讲有公开视频，但它不在公开 playlist 中，因此教材直接按课程页挂出的 recording URL 抓取。",
            }
        )
    if not item.get("slide_urls"):
        rows.append(
            {
                "unit_id": "source_gap_slides_missing",
                "reason": "No official slide PDF was posted on the course pages.",
                "impact": "moderate",
                "user_visible_note": "本讲无官方 slides，因此教材将更多依赖视频、字幕和官方 readings。",
            }
        )
    reading_rows = json.loads(reading_manifest.read_text()) if reading_manifest.exists() else []
    for row in reading_rows:
        if row.get("status") == "available":
            continue
        rows.append(
            {
                "unit_id": f"source_gap_{row.get('reading_id')}",
                "reason": row.get("notes") or "official reading download failed",
                "impact": "low",
                "user_visible_note": f"官方 reading “{row.get('label', 'reading')}” 无法自动下载，讲义保留链接并继续执行 best effort 重建。",
            }
        )
    write_jsonl(omission_path, rows)


def write_lecture_dir(item: dict, reading_manifest: Path, playlist_gaps: dict[str, str]) -> None:
    ldir = lecture_dir(item)
    ldir.mkdir(parents=True, exist_ok=True)
    slide_pdf = lecture_slide_pdf(item)
    subtitle = best_subtitle_path(item)
    covers = list(raw_dir(item).glob("*.jpg")) if item.get("video_url") else []

    meta = {
        **item,
        "course_id": COURSE_ID,
        "course_mode": True,
        "segmentation_required": bool(item.get("video_url")) or bool(item.get("slide_urls")),
        "course_page_url": COURSE_PAGE_URL,
        "mooc_page_url": MOOC_PAGE_URL,
        "playlist_url": PLAYLIST_URL,
        "thumbnail": str(covers[0].relative_to(RUN_ROOT)) if covers else None,
        "subtitle": str(subtitle.relative_to(RUN_ROOT)) if subtitle else None,
        "transcript_text": str((text_dir(item) / "transcript.txt").relative_to(RUN_ROOT)),
        "official_text": str((text_dir(item) / "official.txt").relative_to(RUN_ROOT)),
        "slide_pages_dir": str((ldir / "pdf_pages").relative_to(RUN_ROOT)),
        "reading_manifest": str(reading_manifest.relative_to(RUN_ROOT)),
    }
    write_json(ldir / "meta.json", meta)

    if meta["thumbnail"]:
        ensure_symlink(RUN_ROOT / meta["thumbnail"], ldir / "cover.jpg")
    if meta["subtitle"]:
        ensure_symlink(RUN_ROOT / meta["subtitle"], ldir / "subtitle.srt")
    ensure_symlink(RUN_ROOT / meta["transcript_text"], ldir / "transcript.txt")
    ensure_symlink(RUN_ROOT / meta["official_text"], ldir / "official.txt")
    if slide_pdf.exists():
        ensure_symlink(slide_pdf, ldir / "slides.pdf")
        render_slide_pages(slide_pdf, ldir / "pdf_pages")
    else:
        placeholder_slides_jsonl(ldir)

    for name in ["coverage_units.jsonl", "repair_log.jsonl"]:
        path = ldir / name
        if not path.exists():
            path.write_text("")
    if not (ldir / "figure_manifest.json").exists():
        (ldir / "figure_manifest.json").write_text("[]\n")

    write_json(ldir / "source_manifest.json", build_source_manifest(item, ldir, reading_manifest))
    seed_source_gap_omissions(item, ldir, reading_manifest, playlist_gaps)

    summary = [
        f"# {item['title']}",
        "",
        f"- Date: `{item['date']}`",
        f"- Speaker: `{item['speaker']}`" if item.get("speaker") else "- Speaker: unavailable",
        f"- Affiliation: `{item['affiliation']}`" if item.get("affiliation") else "- Affiliation: unavailable",
        f"- Recording: [YouTube]({item['video_url']})" if item.get("video_url") else "- Recording: unavailable on official pages",
        f"- Playlist: [Agentic AI MOOC Fall 2025]({PLAYLIST_URL})",
        f"- Berkeley RDI page: [course page]({COURSE_PAGE_URL})",
        f"- MOOC page: [syllabus]({MOOC_PAGE_URL})",
        f"- Slides: [slides.pdf](slides.pdf)" if (ldir / "slides.pdf").exists() else "- Slides: unavailable",
        f"- Readings: [manifest.json]({os.path.relpath(reading_manifest, ldir)})",
        "",
        "## Supplemental Readings",
    ]
    if item.get("reading_links"):
        summary.extend([f"- [{row['label']}]({row['url']})" for row in item["reading_links"]])
    else:
        summary.append("- none")
    (ldir / "README.md").write_text("\n".join(summary) + "\n")


def write_course_bundle(lectures: list[dict]) -> None:
    bundle = []
    for item in lectures:
        meta_path = lecture_dir(item) / "meta.json"
        if meta_path.exists():
            bundle.append(json.loads(meta_path.read_text()))
    write_json(RUN_ROOT / "text" / "course_bundle.json", bundle)


def write_course_manifest_seed(lectures: list[dict], metadata: dict, playlist_meta: dict, playlist_gaps: list[dict]) -> None:
    payload = {
        "course_id": COURSE_ID,
        "title": COURSE_TITLE,
        "course_page_url": COURSE_PAGE_URL,
        "mooc_page_url": MOOC_PAGE_URL,
        "playlist_origin_url": PLAYLIST_URL,
        "playlist_title": playlist_meta.get("title"),
        "playlist_channel": playlist_meta.get("channel") or playlist_meta.get("uploader"),
        "scheduled_session_count": len(lectures),
        "public_playlist_count": len(playlist_meta.get("entries") or []),
        "missing_public_sessions": metadata.get("course_gaps", []) + playlist_gaps,
        "course_mode": True,
        "lecture_count": len(lectures),
        "lectures": [
            {
                "lecture_id": item["lecture_id"],
                "lecture_slug": f"{item['lecture_id']}_{item['slug']}",
                "title": item["title"],
                "date": item["date"],
                "video_url": item.get("video_url"),
                "slide_urls": item.get("slide_urls") or [],
                "reading_links": item.get("reading_links") or [],
            }
            for item in lectures
        ],
    }
    write_json(RUN_ROOT / "build" / "course_manifest_seed.json", payload)


def write_lectures_index(lectures: list[dict]) -> None:
    lines = ["# Agentic AI Fall 2025 Lecture Folders", ""]
    for item in lectures:
        lines.append(f"- [{item['lecture_id']} {item['title']}](./{item['lecture_id']}_{item['slug']}/README.md)")
    (RUN_ROOT / "lectures" / "README.md").write_text("\n".join(lines) + "\n")


def write_run_metadata(lectures: list[dict], metadata: dict, playlist_meta: dict, playlist_gaps: list[dict]) -> None:
    write_json(RUN_ROOT / "meta" / "lectures.json", lectures)
    write_json(
        RUN_ROOT / "meta" / "course_sources.json",
        {
            **metadata,
            "playlist_title": playlist_meta.get("title"),
            "playlist_channel": playlist_meta.get("channel") or playlist_meta.get("uploader"),
            "playlist_gaps": playlist_gaps,
        },
    )
    omission_rows = [
        {
            "unit_id": f"course_gap_{idx + 1:02d}",
            "reason": row["reason"],
            "impact": "moderate",
            "user_visible_note": f"{row.get('date_text', row.get('date'))}: {row.get('description', row.get('title', 'session'))}",
        }
        for idx, row in enumerate(metadata.get("course_gaps", []) + playlist_gaps)
    ]
    write_jsonl(RUN_ROOT / "omission_log.jsonl", omission_rows)


def main() -> None:
    ensure_dirs()
    lectures, metadata = parse_schedule_rows()
    playlist_meta = fetch_playlist_metadata()
    lectures, playlist_gaps = attach_playlist_context(lectures, playlist_meta)

    for item in lectures:
        slide_pdf = download_slide_assets(item)
        reading_manifest = download_readings(item)
        if item.get("video_url"):
            fetch_video_assets(item)
        write_text_bundle(item, slide_pdf)
        write_lecture_dir(
            item,
            reading_manifest,
            {row["lecture_id"]: row["reason"] for row in playlist_gaps if row.get("lecture_id")},
        )

    write_course_bundle(lectures)
    write_course_manifest_seed(lectures, metadata, playlist_meta, playlist_gaps)
    write_run_metadata(lectures, metadata, playlist_meta, playlist_gaps)
    write_lectures_index(lectures)
    print(f"bootstrapped={len(lectures)}")
    print(RUN_ROOT)


if __name__ == "__main__":
    main()
