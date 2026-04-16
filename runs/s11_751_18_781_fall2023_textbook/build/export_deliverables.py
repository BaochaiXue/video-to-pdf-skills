#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = RUN_ROOT / "build"
LECTURES_DIR = RUN_ROOT / "lectures"
DELIVERABLE_DIR = RUN_ROOT / "deliverable"
DELIVERABLE_LECTURES_DIR = DELIVERABLE_DIR / "lectures"
DELIVERABLE_BOOK_DIR = DELIVERABLE_DIR / "book"


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def latest_eval_pass(lecture: dict) -> bool:
    ref = lecture.get("latest_eval_report")
    if not ref:
        return False
    path = RUN_ROOT / ref
    if not path.exists():
        return False
    report = json.loads(path.read_text())
    return report.get("overall") == "pass"


def export_lecture_deliverables(manifest: dict) -> int:
    exported = 0
    for lecture in manifest.get("lectures", []):
        if not latest_eval_pass(lecture):
            continue
        lecture_dir = LECTURES_DIR / lecture["lecture_slug"]
        tex_files = sorted(lecture_dir.glob("lecture_*_note.tex"))
        pdf_files = sorted(lecture_dir.glob("lecture_*_note.pdf"))
        if not tex_files or not pdf_files:
            continue
        target_dir = DELIVERABLE_LECTURES_DIR / lecture["lecture_slug"]
        copied_tex = copy_if_exists(tex_files[0], target_dir / tex_files[0].name)
        copied_pdf = copy_if_exists(pdf_files[0], target_dir / pdf_files[0].name)
        if copied_tex and copied_pdf:
            exported += 1
    return exported


def export_book_deliverable(manifest: dict) -> bool:
    status_path = BUILD_DIR / "textbook_status.json"
    pdf_path = BUILD_DIR / "speech_recognition_understanding_fall2023_textbook.pdf"
    tex_path = BUILD_DIR / "speech_recognition_understanding_fall2023_textbook.tex"
    if not status_path.exists() or not pdf_path.exists() or not tex_path.exists():
        return False
    status = json.loads(status_path.read_text())
    if not status.get("deliverable_lectures"):
        return False
    copied_tex = copy_if_exists(tex_path, DELIVERABLE_BOOK_DIR / tex_path.name)
    copied_pdf = copy_if_exists(pdf_path, DELIVERABLE_BOOK_DIR / pdf_path.name)
    return copied_tex and copied_pdf


def main() -> None:
    manifest_path = BUILD_DIR / "course_manifest.json"
    if not manifest_path.exists():
        raise SystemExit("build/course_manifest.json missing")
    manifest = json.loads(manifest_path.read_text())
    DELIVERABLE_DIR.mkdir(parents=True, exist_ok=True)
    lecture_count = export_lecture_deliverables(manifest)
    book_exported = export_book_deliverable(manifest)
    print(
        json.dumps(
            {
                "lecture_exports": lecture_count,
                "book_exported": book_exported,
                "deliverable_dir": str(DELIVERABLE_DIR),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
