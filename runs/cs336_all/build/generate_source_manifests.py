#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
LECTURES_DIR = ROOT / "lectures"
RAW_DIR = ROOT / "raw"
MATERIALS_DIR = ROOT / "materials" / "spring2025-lectures"
NONEXEC_DIR = MATERIALS_DIR / "nonexecutable"


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def find_raw_dir(lecture_num: str) -> Path | None:
    matches = sorted(RAW_DIR.glob(f"{lecture_num}_*"))
    return matches[0] if matches else None


def find_official_material(lecture_num: str) -> Path | None:
    py_path = MATERIALS_DIR / f"lecture_{int(lecture_num)}.py"
    if py_path.exists():
        return py_path

    py_path_padded = MATERIALS_DIR / f"lecture_{lecture_num}.py"
    if py_path_padded.exists():
        return py_path_padded

    pdf_matches = sorted(NONEXEC_DIR.glob(f"2025 Lecture {int(lecture_num)} - *.pdf"))
    if pdf_matches:
        return pdf_matches[0]
    return None


def source_entry(
    source_id: str,
    source_type: str,
    local_path: Path | None,
    required: bool,
    status: str = "available",
    origin_url: str | None = None,
    notes: str = "",
) -> Dict:
    return {
        "source_id": source_id,
        "source_type": source_type,
        "origin_url": origin_url,
        "local_path": rel(local_path) if local_path else None,
        "required_for_coverage": required,
        "status": status,
        "notes": notes,
    }


def figure_manifest_from_tex(lecture_dir: Path) -> List[Dict]:
    tex_files = sorted(lecture_dir.glob("lecture_*_note.tex"))
    if not tex_files:
        return []

    tex = tex_files[0].read_text()
    figures: List[Dict] = []
    current_caption = ""
    pending_path = None
    figure_id = 1
    for raw_line in tex.splitlines():
        line = raw_line.strip()
        if line.startswith(r"\includegraphics"):
            start = line.rfind("{")
            end = line.rfind("}")
            if start != -1 and end != -1 and end > start:
                pending_path = line[start + 1 : end]
        elif line.startswith(r"\caption{") and pending_path:
            current_caption = line[len(r"\caption{") : -1]
            source_type = "video_frame_or_crop" if pending_path.startswith("frames/") else "slide_or_external_asset"
            figures.append(
                {
                    "figure_id": f"figure_{figure_id:02d}",
                    "source_id": source_type,
                    "loc": None,
                    "asset_path": pending_path,
                    "caption": current_caption,
                    "crop": False,
                    "used_in_section": None,
                    "time_provenance": None,
                }
            )
            figure_id += 1
            pending_path = None
            current_caption = ""
    return figures


def build_manifest(lecture_dir: Path) -> Dict:
    lecture_num = lecture_dir.name.split("_", 1)[0]
    meta_path = lecture_dir / "meta.json"
    meta = load_json(meta_path)
    raw_dir = find_raw_dir(lecture_num)
    official_material = find_official_material(lecture_num)

    sources: List[Dict] = []

    sources.append(
        source_entry(
            "lecture_meta",
            "lecture_metadata",
            meta_path,
            True,
            origin_url=meta.get("webpage_url"),
            notes="Local per-lecture metadata normalized for note generation.",
        )
    )
    for name, source_type, required in [
        ("official.txt", "official_text_material", True),
        ("transcript.txt", "normalized_transcript", True),
        ("notes.md", "intermediate_notes", False),
        ("cover.jpg", "cover_image", True),
    ]:
        path = lecture_dir / name
        if path.exists():
            sources.append(source_entry(name.replace(".", "_"), source_type, path, required))

    for dirname, source_type in [
        ("pdf_pages", "derived_slide_renders"),
        ("frames", "derived_frame_assets"),
    ]:
        path = lecture_dir / dirname
        if path.exists():
            sources.append(
                source_entry(
                    dirname,
                    source_type,
                    path,
                    False,
                    notes="Derived asset directory used during note reconstruction.",
                )
            )

    if raw_dir:
        info_path = next(raw_dir.glob("*.info.json"), None)
        if info_path:
            info = load_json(info_path)
            sources.append(
                source_entry(
                    "raw_info_json",
                    "platform_metadata",
                    info_path,
                    True,
                    origin_url=info.get("webpage_url"),
                    notes="Original yt-dlp metadata dump.",
                )
            )

        for srt in sorted(raw_dir.glob("*.srt")):
            kind = "platform_subtitle"
            note = "Original subtitle track downloaded from platform."
            if srt.name.endswith(".en-orig.srt"):
                note = "Original subtitle track before normalization."
            sources.append(source_entry(srt.stem, kind, srt, True, notes=note))

        raw_cover = next(raw_dir.glob("*.jpg"), None)
        if raw_cover:
            sources.append(
                source_entry(
                    "raw_thumbnail",
                    "platform_thumbnail",
                    raw_cover,
                    False,
                    notes="Original downloaded platform thumbnail.",
                )
            )

    if official_material and official_material.exists():
        material_type = "official_lecture_script" if official_material.suffix == ".py" else "official_slide_pdf"
        notes = (
            "Executable official lecture source; treat as a co-equal source with the video."
            if official_material.suffix == ".py"
            else "Official slide PDF; treat as a co-equal source with the video."
        )
        sources.append(
            source_entry(
                "official_material",
                material_type,
                official_material,
                True,
                notes=notes,
            )
        )
    else:
        sources.append(
            source_entry(
                "official_material",
                "official_material",
                None,
                True,
                status="missing",
                notes="No local official lecture script or slide PDF detected.",
            )
        )

    return {
        "course_id": "stanford-cs336-spring-2025",
        "course_mode": True,
        "lecture_id": lecture_num,
        "lecture_slug": lecture_dir.name,
        "title": meta.get("title"),
        "origin_url": meta.get("webpage_url"),
        "sources": sources,
    }


def main() -> None:
    lecture_dirs = sorted(p for p in LECTURES_DIR.iterdir() if p.is_dir() and p.name[:2].isdigit())
    course_manifest = {
        "course_id": "stanford-cs336-spring-2025",
        "playlist_origin_url": "https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_",
        "course_mode": True,
        "lecture_count": len(lecture_dirs),
        "lectures": [],
    }

    for lecture_dir in lecture_dirs:
        manifest = build_manifest(lecture_dir)
        (lecture_dir / "source_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

        figure_manifest = figure_manifest_from_tex(lecture_dir)
        (lecture_dir / "figure_manifest.json").write_text(json.dumps(figure_manifest, indent=2, ensure_ascii=False) + "\n")

        coverage_path = lecture_dir / "coverage_units.jsonl"
        if not coverage_path.exists():
            coverage_path.write_text("")

        omission_path = lecture_dir / "omission_log.jsonl"
        if not omission_path.exists():
            omission_path.write_text("")

        course_manifest["lectures"].append(
            {
                "lecture_id": manifest["lecture_id"],
                "lecture_slug": manifest["lecture_slug"],
                "title": manifest["title"],
                "source_manifest": rel(lecture_dir / "source_manifest.json"),
                "figure_manifest": rel(lecture_dir / "figure_manifest.json"),
            }
        )

    (ROOT / "build" / "course_manifest.json").write_text(
        json.dumps(course_manifest, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
