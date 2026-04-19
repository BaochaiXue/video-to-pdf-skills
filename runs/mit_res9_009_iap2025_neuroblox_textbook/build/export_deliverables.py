#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
LECTURES_DIR = RUN_ROOT / "lectures"
BOOK_DIR = RUN_ROOT / "book"
DELIVERABLE_DIR = RUN_ROOT / "deliverable"
DELIVERABLE_LECTURES_DIR = DELIVERABLE_DIR / "lectures"
DELIVERABLE_BOOK_DIR = DELIVERABLE_DIR / "book"
DELIVERABLE_SCRATCH_SUFFIXES = {
    ".aux",
    ".log",
    ".out",
    ".toc",
    ".fls",
    ".fdb_latexmk",
}


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return True


def clean_deliverable_book_dir() -> None:
    if not DELIVERABLE_BOOK_DIR.exists():
        return
    for path in DELIVERABLE_BOOK_DIR.iterdir():
        if path.is_dir():
            continue
        if path.suffix in DELIVERABLE_SCRATCH_SUFFIXES:
            path.unlink()


def export_lectures() -> list[str]:
    exported: list[str] = []
    for lecture_dir in sorted(path for path in LECTURES_DIR.iterdir() if path.is_dir()):
        tex_files = sorted(lecture_dir.glob("lecture_*_note.tex"))
        pdf_files = sorted(lecture_dir.glob("lecture_*_note.pdf"))
        if not tex_files or not pdf_files:
            continue
        target_dir = DELIVERABLE_LECTURES_DIR / lecture_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tex_files[0], target_dir / tex_files[0].name)
        shutil.copyfile(pdf_files[0], target_dir / pdf_files[0].name)
        exported.append(lecture_dir.name)
    return exported


def export_book() -> list[str]:
    exported: list[str] = []
    DELIVERABLE_BOOK_DIR.mkdir(parents=True, exist_ok=True)
    clean_deliverable_book_dir()
    manifest_path = BOOK_DIR / "textbook_source_manifest.json"
    merged_tex = BOOK_DIR / "main.tex"
    merged_pdf = BOOK_DIR / "textbook.pdf"

    course_slug = "course"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        if isinstance(manifest, dict):
            course = manifest.get("course")
            if isinstance(course, dict):
                course_slug = str(course.get("course_slug") or course.get("course_id") or course_slug)

    tex_name = f"{course_slug}_complete_notes.tex"
    pdf_name = f"{course_slug}_complete_notes.pdf"

    if copy_if_exists(merged_tex, DELIVERABLE_BOOK_DIR / tex_name):
        exported.append(tex_name)
    if copy_if_exists(merged_pdf, DELIVERABLE_BOOK_DIR / pdf_name):
        exported.append(pdf_name)
    return exported


def main() -> None:
    lectures = export_lectures()
    book = export_book()
    print(json.dumps({"lectures": lectures, "book": book}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
