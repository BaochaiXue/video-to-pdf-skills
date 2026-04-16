#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LECTURES_DIR = ROOT / "lectures"


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def lecture_dirs() -> list[Path]:
    return sorted(p for p in LECTURES_DIR.iterdir() if p.is_dir() and p.name.startswith("lec"))


def latest_eval_report(lecture_dir: Path) -> Path | None:
    direct = lecture_dir / "eval_report.json"
    if direct.exists():
        return direct
    reports = sorted((lecture_dir / "eval_reports").glob("pass_*.json"))
    return reports[-1] if reports else None


def select_lecture_tex(lecture_dir: Path) -> Path | None:
    for candidate in [lecture_dir / "lecture_repaired.tex", lecture_dir / "lecture.tex"]:
        if candidate.exists():
            return candidate
    tex_files = sorted(lecture_dir.glob("lecture_*_note.tex"))
    return tex_files[0] if tex_files else None


def select_lecture_pdf(lecture_dir: Path) -> Path | None:
    for candidate in [lecture_dir / "lecture_repaired.pdf", lecture_dir / "lecture.pdf"]:
        if candidate.exists():
            return candidate
    pdf_files = sorted(lecture_dir.glob("lecture_*_note.pdf"))
    return pdf_files[0] if pdf_files else None


def main() -> None:
    seed = json.loads((ROOT / "build" / "course_manifest_seed.json").read_text())
    final = {
        **seed,
        "lectures": [],
    }

    for lecture_dir in lecture_dirs():
        meta = json.loads((lecture_dir / "meta.json").read_text())
        eval_report = latest_eval_report(lecture_dir)
        lecture_tex = select_lecture_tex(lecture_dir)
        lecture_pdf = select_lecture_pdf(lecture_dir)
        final["lectures"].append(
            {
                "lecture_id": meta.get("lecture_id", f"{meta['playlist_index']:02d}"),
                "lecture_slug": lecture_dir.name,
                "title": meta["title"],
                "date": meta["date"],
                "speaker": meta.get("speaker"),
                "course_mode": bool(meta.get("course_mode", True)),
                "segmentation_required": bool(meta.get("segmentation_required", False)),
                "source_manifest": rel(lecture_dir / "source_manifest.json") if (lecture_dir / "source_manifest.json").exists() else None,
                "transcript_jsonl": rel(lecture_dir / "transcript.jsonl") if (lecture_dir / "transcript.jsonl").exists() else None,
                "slides_jsonl": rel(lecture_dir / "slides.jsonl") if (lecture_dir / "slides.jsonl").exists() else None,
                "segments_jsonl": rel(lecture_dir / "segments.jsonl") if (lecture_dir / "segments.jsonl").exists() else None,
                "lecture_plan": rel(lecture_dir / "lecture_plan.json") if (lecture_dir / "lecture_plan.json").exists() else None,
                "figure_plan": rel(lecture_dir / "figure_plan.json") if (lecture_dir / "figure_plan.json").exists() else (rel(lecture_dir / "figure_plan.jsonl") if (lecture_dir / "figure_plan.jsonl").exists() else None),
                "latest_eval_report": rel(eval_report) if eval_report is not None else None,
                "repair_log": rel(lecture_dir / "repair_log.jsonl") if (lecture_dir / "repair_log.jsonl").exists() else None,
                "coverage_units": rel(lecture_dir / "coverage_units.jsonl") if (lecture_dir / "coverage_units.jsonl").exists() else None,
                "omission_log": rel(lecture_dir / "omission_log.jsonl") if (lecture_dir / "omission_log.jsonl").exists() else None,
                "figure_manifest": rel(lecture_dir / "figure_manifest.json") if (lecture_dir / "figure_manifest.json").exists() else None,
                "lecture_tex": rel(lecture_tex) if lecture_tex is not None else None,
                "lecture_pdf": rel(lecture_pdf) if lecture_pdf is not None else None,
                "lecture_quality_report": rel(lecture_dir / "lecture_quality_report.md") if (lecture_dir / "lecture_quality_report.md").exists() else None,
            }
        )

    (ROOT / "build" / "course_manifest.json").write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
