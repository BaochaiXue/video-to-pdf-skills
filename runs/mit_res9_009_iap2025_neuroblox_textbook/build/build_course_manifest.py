#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = RUN_ROOT / "build"
LECTURES_DIR = RUN_ROOT / "lectures"


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def read_seed() -> dict:
    seed_path = BUILD_DIR / "course_manifest_seed.json"
    if seed_path.exists():
        seed = load_json(seed_path)
        if isinstance(seed, dict):
            return seed
    return {
        "course_id": "MIT RES.9-009",
        "course_slug": "mit_res9_009_iap2025_neuroblox_textbook",
        "title": "MIT RES.9-009: Introduction to Computational Neuroscience with Neuroblox (January IAP 2025)",
        "term": "IAP 2025",
        "course_mode": True,
        "lectures": [],
    }


def lecture_sort_key(path: Path) -> tuple[int, str]:
    match = re.match(r"^(\d+)_", path.name)
    if match:
        return int(match.group(1)), path.name
    return 10**9, path.name


def select(path: Path, pattern: str) -> str | None:
    candidates = sorted(path.glob(pattern))
    if not candidates:
        return None
    return str(candidates[0].relative_to(RUN_ROOT))


def select_note_tex(lecture_dir: Path) -> str | None:
    preferred = lecture_dir / "lecture_XX_note.tex"
    if preferred.exists():
        return str(preferred.relative_to(RUN_ROOT))
    return select(lecture_dir, "lecture_*_note.tex")


def select_note_pdf(lecture_dir: Path) -> str | None:
    preferred = lecture_dir / "lecture_XX_note.pdf"
    if preferred.exists():
        return str(preferred.relative_to(RUN_ROOT))
    return select(lecture_dir, "lecture_*_note.pdf")


def latest_eval_report(lecture_dir: Path) -> str | None:
    reports = sorted((lecture_dir / "eval_reports").glob("pass_*.json"))
    if not reports:
        return None
    return str(reports[-1].relative_to(RUN_ROOT))


def main() -> None:
    seed = read_seed()
    lectures: list[dict] = []
    lecture_dirs = sorted((path for path in LECTURES_DIR.iterdir() if path.is_dir() and re.match(r"^\d+_", path.name)), key=lecture_sort_key)

    for lecture_dir in lecture_dirs:
        lecture_id_match = re.match(r"^(\d+)_", lecture_dir.name)
        lecture_id = lecture_id_match.group(1) if lecture_id_match else lecture_dir.name
        plan_path = lecture_dir / "lecture_plan.json"
        title = lecture_dir.name
        if plan_path.exists():
            plan = load_json(plan_path)
            if isinstance(plan, dict):
                title = str(plan.get("title") or title)
        eval_report = latest_eval_report(lecture_dir)
        eval_state = "unknown"
        if eval_report:
            report = load_json(RUN_ROOT / eval_report)
            if isinstance(report, dict):
                eval_state = str(report.get("overall") or "unknown")

        lectures.append(
            {
                "lecture_id": f"{int(lecture_id):02d}" if lecture_id.isdigit() else lecture_id,
                "lecture_slug": lecture_dir.name,
                "title": title,
                "lecture_plan": str(plan_path.relative_to(RUN_ROOT)) if plan_path.exists() else None,
                "source_manifest": select(lecture_dir, "source_manifest.json"),
                "transcript_jsonl": select(lecture_dir, "transcript.jsonl"),
                "slides_jsonl": select(lecture_dir, "slides.jsonl"),
                "segments_jsonl": select(lecture_dir, "segments.jsonl"),
                "coverage_units": select(lecture_dir, "coverage_units.jsonl"),
                "omission_log": select(lecture_dir, "omission_log.jsonl"),
                "figure_plan": select(lecture_dir, "figure_plan.json"),
                "figure_manifest": select(lecture_dir, "figure_manifest.json"),
                "repair_log": select(lecture_dir, "repair_log.jsonl"),
                "latest_eval_report": eval_report,
                "evaluation_state": eval_state,
                "lecture_tex": select_note_tex(lecture_dir),
                "lecture_pdf": select_note_pdf(lecture_dir),
            }
        )

    manifest = {**seed, "lectures": lectures}
    (BUILD_DIR / "course_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(BUILD_DIR / "course_manifest.json")


if __name__ == "__main__":
    main()
