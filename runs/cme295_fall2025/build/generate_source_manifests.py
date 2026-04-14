#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LECTURES_DIR = ROOT / "lectures"


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def source_entry(
    source_id: str,
    source_type: str,
    local_path: Path | None,
    required: bool,
    origin_url: str | None = None,
    status: str = "available",
    notes: str = "",
) -> dict:
    return {
        "source_id": source_id,
        "source_type": source_type,
        "origin_url": origin_url,
        "local_path": rel(local_path) if local_path else None,
        "required_for_coverage": required,
        "status": status,
        "notes": notes,
    }


def figure_manifest_from_tex(lecture_dir: Path) -> list[dict]:
    tex_files = sorted(lecture_dir.glob("lecture_*_note.tex"))
    if not tex_files:
        return []
    tex = tex_files[0].read_text()
    figures: list[dict] = []
    pending_path: str | None = None
    current_section: str | None = None
    figure_id = 1
    for raw_line in tex.splitlines():
        line = raw_line.strip()
        if line.startswith(r"\section{") or line.startswith(r"\subsection{"):
            start = line.find("{")
            end = line.rfind("}")
            if start != -1 and end != -1 and end > start:
                current_section = line[start + 1 : end]
        if line.startswith(r"\includegraphics"):
            start = line.rfind("{")
            end = line.rfind("}")
            if start != -1 and end != -1 and end > start:
                pending_path = line[start + 1 : end]
        elif line.startswith(r"\caption{") and pending_path:
            caption = line[len(r"\caption{") : -1]
            source_type = "video_frame_or_crop" if "frames/" in pending_path else "slide_or_external_asset"
            loc = None
            page_match = re.search(r"pdf_pages/page-(\d+)\.png$", pending_path)
            if page_match:
                loc = {"page": int(page_match.group(1))}
            figures.append(
                {
                    "figure_id": f"figure_{figure_id:02d}",
                    "source_id": source_type,
                    "loc": loc,
                    "asset_path": pending_path,
                    "caption": caption,
                    "crop": False,
                    "used_in_section": current_section,
                    "time_provenance": None,
                }
            )
            figure_id += 1
            pending_path = None
    return figures


def build_manifest(lecture_dir: Path) -> dict:
    meta = json.loads((lecture_dir / "meta.json").read_text())
    sources: list[dict] = []

    sources.append(
        source_entry(
            "lecture_meta",
            "lecture_metadata",
            lecture_dir / "meta.json",
            True,
            origin_url=meta.get("video_url"),
            notes="Normalized lecture metadata for course-note generation.",
        )
    )

    for name, source_type, required in [
        ("cover.jpg", "cover_image", True),
        ("subtitle.srt", "platform_subtitle", True),
        ("transcript.jsonl", "structured_transcript_evidence", True),
        ("slides.jsonl", "structured_slide_evidence", True),
        ("segments.jsonl", "segment_plan", bool(meta.get("segmentation_required"))),
        ("transcript.txt", "debug_transcript_text", False),
        ("official.txt", "debug_slide_text", False),
        ("slides.pdf", "official_slide_pdf", True),
        ("pdf_pages", "derived_slide_renders", False),
    ]:
        path = lecture_dir / name
        if path.exists():
            sources.append(source_entry(name.replace(".", "_"), source_type, path, required))
        else:
            sources.append(
                source_entry(
                    name.replace(".", "_"),
                    source_type,
                    None,
                    required,
                    status="missing",
                    notes="Expected local lecture artifact was not found.",
                )
            )

    raw_dir = ROOT / "raw" / f"{meta['playlist_index']:02d}_{meta['video_id']}"
    info_candidates = sorted(raw_dir.glob("*.info.json"))
    if info_candidates:
        sources.append(
            source_entry(
                "raw_info_json",
                "platform_metadata",
                info_candidates[0],
                True,
                origin_url=meta.get("video_url"),
                notes="Original yt-dlp metadata dump for the video.",
            )
        )
    else:
        sources.append(
            source_entry(
                "raw_info_json",
                "platform_metadata",
                None,
                True,
                origin_url=meta.get("video_url"),
                status="missing",
                notes="yt-dlp metadata JSON is missing.",
            )
        )

    return {
        "course_id": meta["course_id"],
        "course_mode": bool(meta.get("course_mode", True)),
        "lecture_id": f"{meta['playlist_index']:02d}",
        "lecture_slug": lecture_dir.name,
        "title": meta["title"],
        "origin_url": meta["video_url"],
        "slide_origin_url": meta["slide_url"],
        "sources": sources,
        "outputs": [
            {
                "path": rel(path),
                "type": "latex_note" if path.suffix == ".tex" else "rendered_pdf",
                "status": "generated",
            }
            for path in sorted(lecture_dir.glob("lecture_*_note.tex")) + sorted(lecture_dir.glob("lecture_*_note.pdf"))
        ],
    }


def main() -> None:
    lecture_dirs = sorted(p for p in LECTURES_DIR.iterdir() if p.is_dir() and p.name[:2].isdigit())
    for lecture_dir in lecture_dirs:
        manifest = build_manifest(lecture_dir)
        (lecture_dir / "source_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        figures = figure_manifest_from_tex(lecture_dir)
        (lecture_dir / "figure_manifest.json").write_text(json.dumps(figures, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
