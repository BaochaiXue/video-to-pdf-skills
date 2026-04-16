#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LECTURES_DIR = ROOT / "lectures"


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def main() -> None:
    lecture_dirs = sorted(p for p in LECTURES_DIR.iterdir() if p.is_dir() and p.name[:2].isdigit())
    seed = json.loads((ROOT / "build" / "course_manifest_seed.json").read_text())
    final = {
        **seed,
        "lectures": [],
    }

    for lecture_dir in lecture_dirs:
        meta = json.loads((lecture_dir / "meta.json").read_text())
        final["lectures"].append(
            {
                "lecture_id": f"{meta['playlist_index']:02d}",
                "lecture_slug": lecture_dir.name,
                "title": meta["title"],
                "date": meta["date"],
                "lecturer": meta.get("lecturer"),
                "course_page_url": meta.get("course_page_url"),
                "schedule_url": meta.get("schedule_url"),
                "video_url": meta.get("webpage_url"),
                "official_material_urls": meta.get("official_material_urls") or [],
                "course_mode": bool(meta.get("course_mode", True)),
                "segmentation_required": bool(meta.get("segmentation_required", False)),
                "source_manifest": rel(lecture_dir / "source_manifest.json") if (lecture_dir / "source_manifest.json").exists() else None,
                "transcript_jsonl": rel(lecture_dir / "transcript.jsonl") if (lecture_dir / "transcript.jsonl").exists() else None,
                "slides_jsonl": rel(lecture_dir / "slides.jsonl") if (lecture_dir / "slides.jsonl").exists() else None,
                "segments_jsonl": rel(lecture_dir / "segments.jsonl") if (lecture_dir / "segments.jsonl").exists() else None,
                "lecture_plan": rel(lecture_dir / "lecture_plan.json") if (lecture_dir / "lecture_plan.json").exists() else None,
                "figure_plan": rel(lecture_dir / "figure_plan.json") if (lecture_dir / "figure_plan.json").exists() else None,
                "latest_eval_report": rel(sorted((lecture_dir / "eval_reports").glob("pass_*.json"))[-1]) if (lecture_dir / "eval_reports").exists() and list((lecture_dir / "eval_reports").glob("pass_*.json")) else None,
                "repair_log": rel(lecture_dir / "repair_log.jsonl") if (lecture_dir / "repair_log.jsonl").exists() else None,
                "coverage_units": rel(lecture_dir / "coverage_units.jsonl"),
                "omission_log": rel(lecture_dir / "omission_log.jsonl"),
                "figure_manifest": rel(lecture_dir / "figure_manifest.json"),
                "lecture_tex": rel(next(iter(sorted(lecture_dir.glob("lecture_*_note.tex"))), lecture_dir / "missing.tex")),
                "lecture_pdf": rel(next(iter(sorted(lecture_dir.glob("lecture_*_note.pdf"))), lecture_dir / "missing.pdf")),
            }
        )

    (ROOT / "build" / "course_manifest.json").write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
