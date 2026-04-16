#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


RUN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RUN_ROOT.parents[1]

COURSE_ID = "stanford-cs231n-spring-2025"
COURSE_TITLE = "Stanford CS231N: Deep Learning for Computer Vision (Spring 2025)"
PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16"
COURSE_PAGE_URL = "https://cs231n.stanford.edu/2025/"
SCHEDULE_URL = "https://cs231n.stanford.edu/2025/schedule.html"


def ensure_dirs() -> None:
    for dirname in ["build", "lectures", "materials/slides", "materials/readings", "meta", "raw", "text"]:
        (RUN_ROOT / dirname).mkdir(parents=True, exist_ok=True)


def slugify(text: str) -> str:
    text = text.lower().replace("&", "and").replace("/", "_")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(payload + ("\n" if payload else ""))


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


def lecture_slide_parts(item: dict) -> list[Path]:
    parts: list[Path] = []
    urls = item.get("slide_urls") or []
    for idx, url in enumerate(urls, start=1):
        suffix = f"_part{idx:02d}.pdf" if len(urls) > 1 else ".pdf"
        parts.append(RUN_ROOT / "materials" / "slides" / f"{item['playlist_index']:02d}_{item['slug']}{suffix}")
    return parts


def reading_dir(item: dict) -> Path:
    return RUN_ROOT / "materials" / "readings" / f"{item['playlist_index']:02d}_{item['slug']}"


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


def parse_schedule_rows() -> list[dict]:
    html = requests.get(SCHEDULE_URL, timeout=30).text
    (RUN_ROOT / "meta" / "schedule.html").write_text(html)
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        raise RuntimeError("schedule table not found")

    lectures: list[dict] = []
    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        desc_text = cells[1].get_text(" ", strip=True)
        if not desc_text.startswith("Lecture "):
            continue

        date = cells[0].get_text(" ", strip=True)
        desc_strings = [text.strip() for text in cells[1].stripped_strings if text.strip()]
        title = next((text for text in desc_strings if text.startswith("Lecture ")), desc_text)
        lecture_num_match = re.match(r"Lecture\s+(\d+):\s*(.+)", title)
        if not lecture_num_match:
            continue
        lecture_num = int(lecture_num_match.group(1))
        title_short = lecture_num_match.group(2).strip()
        topics = [
            text
            for text in desc_strings[1:]
            if not text.lower().startswith("slides")
            and text not in {"[", "]", "[ slides ]"}
        ]
        if not topics:
            topics = [title_short]

        slide_urls = [
            urljoin(SCHEDULE_URL, anchor.get("href"))
            for anchor in cells[1].find_all("a")
            if "slide" in anchor.get_text(" ", strip=True).lower()
        ]
        reading_links = [
            {
                "label": anchor.get_text(" ", strip=True),
                "url": urljoin(SCHEDULE_URL, anchor.get("href")),
            }
            for anchor in cells[3].find_all("a")
        ] if len(cells) > 3 else []

        lectures.append(
            {
                "playlist_index": lecture_num,
                "date": f"2025-{date.split('/')[0]}-{date.split('/')[1]}",
                "kind": "lecture",
                "title": title,
                "title_short": title_short,
                "slug": slugify(title_short),
                "topics": topics,
                "lecturer": cells[2].get_text(" ", strip=True) if len(cells) > 2 else "",
                "slide_urls": slide_urls,
                "reading_links": reading_links,
            }
        )
    return lectures


def fetch_playlist_metadata() -> dict:
    out = RUN_ROOT / "meta" / "playlist_full.json"
    if not out.exists():
        output = subprocess.check_output(
            ["yt-dlp", "--flat-playlist", "--dump-single-json", PLAYLIST_URL],
            text=True,
        )
        out.write_text(output)
    return json.loads(out.read_text())


def attach_playlist_videos(lectures: list[dict], playlist_meta: dict) -> list[dict]:
    entries = playlist_meta.get("entries") or []
    parsed_by_num: dict[int, dict] = {}
    for entry in entries:
        title = entry.get("title") or ""
        match = re.search(r"Lecture\s+(\d+)", title)
        if match:
            parsed_by_num[int(match.group(1))] = entry

    attached: list[dict] = []
    for lecture in lectures:
        entry = parsed_by_num.get(lecture["playlist_index"])
        if entry is None and 0 <= lecture["playlist_index"] - 1 < len(entries):
            entry = entries[lecture["playlist_index"] - 1]
        if entry is None:
            raise RuntimeError(f"playlist entry missing for lecture {lecture['playlist_index']:02d}")
        lecture = {
            **lecture,
            "video_id": entry["id"],
            "video_url": f"https://www.youtube.com/watch?v={entry['id']}",
            "playlist_url": PLAYLIST_URL,
        }
        attached.append(lecture)
    return attached


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
        out_path.write_text("No official slides were posted for this lecture on the Spring 2025 schedule page.\n")
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
    generated = sorted(out_dir.glob("page-*.png"))
    for idx, path in enumerate(generated, start=1):
        target = out_dir / f"page-{idx:02d}.png"
        if path != target:
            path.rename(target)


def download_slide_assets(item: dict) -> Path | None:
    slide_urls = item.get("slide_urls") or []
    if not slide_urls:
        return None

    bundle_path = lecture_slide_pdf(item)
    part_paths = lecture_slide_parts(item)
    for url, part_path in zip(slide_urls, part_paths):
        download(url, part_path)

    if len(part_paths) == 1:
        if part_paths[0] != bundle_path:
            shutil.copyfile(part_paths[0], bundle_path)
        return bundle_path

    if not bundle_path.exists():
        run(["pdfunite", *[str(path) for path in part_paths], str(bundle_path)])
    return bundle_path


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
    manifest_rows: list[dict] = []
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
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            ext = reading_extension(url, response.headers.get("content-type", ""))
            dest = rdir / f"{idx:02d}_{slugify(label) or 'reading'}{ext}"
            if not dest.exists():
                if ext == ".pdf" or "application/pdf" in response.headers.get("content-type", ""):
                    dest.write_bytes(response.content)
                else:
                    dest.write_text(response.text)
            row["local_path"] = str(dest.relative_to(RUN_ROOT))
            row["status"] = "available"
        except Exception as exc:  # noqa: BLE001
            row["notes"] = f"download_failed: {exc}"
        manifest_rows.append(row)
    manifest_path = rdir / "manifest.json"
    write_json(manifest_path, manifest_rows)
    return manifest_path


def write_text_bundle(item: dict) -> None:
    tdir = text_dir(item)
    tdir.mkdir(parents=True, exist_ok=True)
    subtitle = best_subtitle_path(item)
    transcript_path = tdir / "transcript.txt"
    if subtitle:
        transcript_path.write_text(srt_to_text(subtitle))
    else:
        transcript_path.write_text("No subtitle track was downloaded for this lecture.\n")

    official_text = tdir / "official.txt"
    slide_bundle = lecture_slide_pdf(item)
    extract_slide_text(slide_bundle, official_text)


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
            "text": "No official slides were posted for this lecture on the Spring 2025 schedule page.",
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
            "origin_url": item["video_url"],
            "local_path": str((lecture_dir_path / "meta.json").relative_to(RUN_ROOT)),
            "required_for_coverage": True,
            "status": "available",
            "notes": "Normalized lecture metadata for course-note generation.",
        },
        {
            "source_id": "schedule_page",
            "source_type": "official_course_schedule",
            "origin_url": SCHEDULE_URL,
            "local_path": "meta/schedule.html",
            "required_for_coverage": True,
            "status": "available",
            "notes": "Official Spring 2025 schedule page.",
        },
        {
            "source_id": "raw_info_json",
            "source_type": "platform_metadata",
            "origin_url": item["video_url"],
            "local_path": str(next(raw_dir(item).glob("*.info.json")).relative_to(RUN_ROOT)),
            "required_for_coverage": True,
            "status": "available",
            "notes": "Original yt-dlp metadata dump for the lecture video.",
        },
        {
            "source_id": "subtitle_srt",
            "source_type": "platform_subtitle",
            "origin_url": None,
            "local_path": str((lecture_dir_path / "subtitle.srt").relative_to(RUN_ROOT)) if (lecture_dir_path / "subtitle.srt").exists() else None,
            "required_for_coverage": True,
            "status": "available" if (lecture_dir_path / "subtitle.srt").exists() else "missing",
            "notes": "",
        },
        {
            "source_id": "transcript_txt",
            "source_type": "debug_transcript_text",
            "origin_url": None,
            "local_path": str((lecture_dir_path / "transcript.txt").relative_to(RUN_ROOT)),
            "required_for_coverage": False,
            "status": "available",
            "notes": "Human-readable debug transcript.",
        },
        {
            "source_id": "official_txt",
            "source_type": "debug_slide_text",
            "origin_url": None,
            "local_path": str((lecture_dir_path / "official.txt").relative_to(RUN_ROOT)),
            "required_for_coverage": False,
            "status": "available",
            "notes": "Human-readable debug slide extract.",
        },
        {
            "source_id": "reading_manifest",
            "source_type": "official_reading_manifest",
            "origin_url": COURSE_PAGE_URL,
            "local_path": str(reading_manifest.relative_to(RUN_ROOT)),
            "required_for_coverage": False,
            "status": "available",
            "notes": "Downloaded official reading links and local copies.",
        },
    ]

    slides_pdf = lecture_dir_path / "slides.pdf"
    if slides_pdf.exists():
        sources.extend(
            [
                {
                    "source_id": "slides_pdf",
                    "source_type": "official_slide_pdf",
                    "origin_url": item["slide_urls"][0] if item.get("slide_urls") else None,
                    "local_path": str(slides_pdf.relative_to(RUN_ROOT)),
                    "required_for_coverage": True,
                    "status": "available",
                    "notes": "Official lecture slides bundled into one PDF.",
                },
                {
                    "source_id": "pdf_pages",
                    "source_type": "derived_slide_renders",
                    "origin_url": None,
                    "local_path": str((lecture_dir_path / "pdf_pages").relative_to(RUN_ROOT)),
                    "required_for_coverage": False,
                    "status": "available",
                    "notes": "Per-page slide renders for figure selection.",
                },
            ]
        )
    else:
        sources.append(
            {
                "source_id": "slides_pdf",
                "source_type": "official_slide_pdf",
                "origin_url": None,
                "local_path": None,
                "required_for_coverage": False,
                "status": "missing",
                "notes": "No official lecture slides were posted for this lecture.",
            }
        )

    return {
        "course_id": COURSE_ID,
        "course_mode": True,
        "lecture_id": f"{item['playlist_index']:02d}",
        "lecture_slug": lecture_dir_path.name,
        "title": item["title"],
        "origin_url": item["video_url"],
        "slide_origin_urls": item.get("slide_urls") or [],
        "sources": sources,
    }


def seed_source_gap_omissions(item: dict, lecture_dir_path: Path, reading_manifest: Path) -> None:
    omission_path = lecture_dir_path / "omission_log.jsonl"
    existing: list[dict] = []
    if omission_path.exists():
        for raw_line in omission_path.read_text().splitlines():
            line = raw_line.strip()
            if line:
                existing.append(json.loads(line))
    existing_ids = {row.get("unit_id") for row in existing}
    seeded = list(existing)

    if not item.get("slide_urls"):
        unit_id = "source_gap_slides_missing"
        if unit_id not in existing_ids:
            seeded.append(
                {
                    "unit_id": unit_id,
                    "reason": "No official lecture slides were posted on the Spring 2025 schedule page.",
                    "impact": "moderate",
                    "user_visible_note": "本讲无官方 slides，因此教材只能依据公开视频、字幕、课程页上下文做 best effort 重建。",
                }
            )
            existing_ids.add(unit_id)

    manifest_rows = json.loads(reading_manifest.read_text()) if reading_manifest.exists() else []
    for row in manifest_rows:
        if row.get("status") == "available":
            continue
        unit_id = f"source_gap_{row.get('reading_id')}"
        if unit_id in existing_ids:
            continue
        seeded.append(
            {
                "unit_id": unit_id,
                "reason": row.get("notes") or "official reading download failed",
                "impact": "low",
                "user_visible_note": f"官方 reading “{row.get('label', 'reading')}” 无法自动下载，本讲保留链接并继续执行 best effort 讲义生成。",
            }
        )
        existing_ids.add(unit_id)

    if seeded:
        write_jsonl(omission_path, seeded)


def write_lecture_dir(item: dict, reading_manifest: Path) -> None:
    ldir = lecture_dir(item)
    ldir.mkdir(parents=True, exist_ok=True)

    covers = list(raw_dir(item).glob("*.jpg"))
    subtitle = best_subtitle_path(item)
    slide_bundle = lecture_slide_pdf(item)
    meta = {
        **item,
        "course_id": COURSE_ID,
        "course_mode": True,
        "segmentation_required": True,
        "playlist_url": PLAYLIST_URL,
        "course_page_url": COURSE_PAGE_URL,
        "schedule_url": SCHEDULE_URL,
        "thumbnail": str(covers[0].relative_to(RUN_ROOT)) if covers else None,
        "subtitle": str(subtitle.relative_to(RUN_ROOT)) if subtitle else None,
        "material": str(slide_bundle.relative_to(RUN_ROOT)) if slide_bundle.exists() else None,
        "reading_manifest": str(reading_manifest.relative_to(RUN_ROOT)),
        "transcript_text": str((text_dir(item) / "transcript.txt").relative_to(RUN_ROOT)),
        "official_text": str((text_dir(item) / "official.txt").relative_to(RUN_ROOT)),
        "slide_pages_dir": str((ldir / "pdf_pages").relative_to(RUN_ROOT)),
    }
    write_json(ldir / "meta.json", meta)

    if meta["thumbnail"]:
        ensure_symlink(RUN_ROOT / meta["thumbnail"], ldir / "cover.jpg")
    if meta["subtitle"]:
        ensure_symlink(RUN_ROOT / meta["subtitle"], ldir / "subtitle.srt")
    ensure_symlink(RUN_ROOT / meta["transcript_text"], ldir / "transcript.txt")
    ensure_symlink(RUN_ROOT / meta["official_text"], ldir / "official.txt")
    if meta["material"]:
        ensure_symlink(RUN_ROOT / meta["material"], ldir / "slides.pdf")
        render_slide_pages(slide_bundle, ldir / "pdf_pages")
    else:
        placeholder_slides_jsonl(ldir)

    for name in ["coverage_units.jsonl", "omission_log.jsonl"]:
        path = ldir / name
        if not path.exists():
            path.write_text("")
    if not (ldir / "figure_manifest.json").exists():
        (ldir / "figure_manifest.json").write_text("[]\n")

    write_json(ldir / "source_manifest.json", build_source_manifest(item, ldir, reading_manifest))
    seed_source_gap_omissions(item, ldir, reading_manifest)

    summary_lines = [
        f"# {item['title']}",
        "",
        f"- Date: `{item['date']}`",
        f"- Lecturer: `{item['lecturer']}`" if item.get("lecturer") else "- Lecturer: unavailable",
        f"- Video: [YouTube]({item['video_url']})",
        f"- Playlist: [YouTube playlist]({PLAYLIST_URL})",
        f"- Course page: [CS231N Spring 2025]({COURSE_PAGE_URL})",
        f"- Schedule: [schedule.html](../../meta/schedule.html)",
        f"- Slides: [slides.pdf](slides.pdf)" if (ldir / "slides.pdf").exists() else "- Slides: unavailable on official schedule",
        f"- Cover: [cover.jpg](cover.jpg)" if (ldir / "cover.jpg").exists() else "- Cover: unavailable",
        f"- Subtitle: [subtitle.srt](subtitle.srt)" if (ldir / "subtitle.srt").exists() else "- Subtitle: unavailable",
        f"- Transcript: [transcript.txt](transcript.txt)",
        "- Official text: [official.txt](official.txt)",
        f"- Readings: [manifest.json]({os.path.relpath(reading_manifest, ldir)})",
        "- Slide pages: `pdf_pages/page-*.png`" if (ldir / "pdf_pages").exists() else "- Slide pages: unavailable",
        "",
        "## Topics",
        "",
    ]
    for topic in item["topics"]:
        summary_lines.append(f"- {topic}")
    summary_lines.extend(
        [
            "",
            "## Reading Links",
            "",
        ]
    )
    if item.get("reading_links"):
        summary_lines.extend([f"- [{reading['label']}]({reading['url']})" for reading in item["reading_links"]])
    else:
        summary_lines.append("- none explicitly listed on the official schedule page")
    summary_lines.extend(
        [
            "",
            "## Writing requirements",
            "",
            "- Use Chinese.",
            "- Preserve coverage of video, slides, course page context, and official readings.",
            "- Include diagrams, tables, experiments, and algorithmic figures when they materially improve explanation.",
            "- End each major section with `本章小结` and end the document with `总结与延伸`.",
            "- If a source is missing or unavailable, record it in `omission_log.jsonl` and the final appendix.",
        ]
    )
    (ldir / "README.md").write_text("\n".join(summary_lines) + "\n")


def download_course_pages() -> None:
    download(COURSE_PAGE_URL, RUN_ROOT / "meta" / "course_page.html")
    download(SCHEDULE_URL, RUN_ROOT / "meta" / "schedule.html")


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


def write_course_bundle(lectures: list[dict]) -> None:
    bundle = []
    for item in lectures:
        meta_path = lecture_dir(item) / "meta.json"
        if meta_path.exists():
            bundle.append(json.loads(meta_path.read_text()))
    write_json(RUN_ROOT / "text" / "course_bundle.json", bundle)


def write_course_manifest_seed(lectures: list[dict]) -> None:
    manifest = {
        "course_id": COURSE_ID,
        "title": COURSE_TITLE,
        "playlist_origin_url": PLAYLIST_URL,
        "course_page_url": COURSE_PAGE_URL,
        "schedule_url": SCHEDULE_URL,
        "course_mode": True,
        "lecture_count": len(lectures),
        "lectures": [
            {
                "lecture_id": f"{item['playlist_index']:02d}",
                "lecture_slug": f"{item['playlist_index']:02d}_{item['slug']}",
                "title": item["title"],
                "date": item["date"],
                "video_url": item["video_url"],
                "slide_urls": item.get("slide_urls") or [],
                "reading_links": item.get("reading_links") or [],
            }
            for item in lectures
        ],
    }
    write_json(RUN_ROOT / "build" / "course_manifest_seed.json", manifest)


def write_lectures_index(lectures: list[dict]) -> None:
    lines = ["# CS231N Lecture Folders", ""]
    for item in lectures:
        slug = f"{item['playlist_index']:02d}_{item['slug']}"
        lines.append(f"- [{item['playlist_index']:02d} {item['title']}](./{slug}/README.md)")
    (RUN_ROOT / "lectures" / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ensure_dirs()
    download_course_pages()
    playlist_meta = fetch_playlist_metadata()
    lectures = attach_playlist_videos(parse_schedule_rows(), playlist_meta)
    write_json(RUN_ROOT / "meta" / "lectures.json", lectures)

    for item in lectures:
        download_slide_assets(item)
        reading_manifest = download_readings(item)
        fetch_video_assets(item)
        write_text_bundle(item)
        write_lecture_dir(item, reading_manifest)

    write_course_bundle(lectures)
    write_course_manifest_seed(lectures)
    write_lectures_index(lectures)
    print(f"bootstrapped={len(lectures)}")
    print(RUN_ROOT)


if __name__ == "__main__":
    main()
