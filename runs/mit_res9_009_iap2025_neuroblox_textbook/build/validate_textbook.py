#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from validate_lecture import lecture_dirs, validate_lecture


RUN_ROOT = Path(__file__).resolve().parents[1]
BOOK_DIR = RUN_ROOT / "book"
DELIVERABLE_DIR = RUN_ROOT / "deliverable"


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def compile_tex(tex_path: Path) -> None:
    for _ in range(2):
        subprocess.run(
            ["xelatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tex_path.parent,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )


def active_lecture_dirs(selectors: list[str] | None) -> list[Path]:
    dirs = lecture_dirs(selectors)
    active: list[Path] = []
    for lecture_dir in dirs:
        markers = [
            "lecture_plan.json",
            "source_manifest.json",
            "transcript.jsonl",
            "slides.jsonl",
            "segments.jsonl",
            "coverage_units.jsonl",
            "lecture_XX_note.tex",
            "lecture_XX_note.pdf",
            "eval_reports",
        ]
        if any((lecture_dir / marker).exists() for marker in markers):
            active.append(lecture_dir)
    return active


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("lectures", nargs="*")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--require-book-pdf", action="store_true")
    parser.add_argument("--require-deliverable", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []
    for name in ["main.tex", "textbook_source_manifest.json"]:
        path = BOOK_DIR / name
        if not path.exists():
            errors.append(f"book/{name} missing")
    for name in [
        "frontmatter/preface.tex",
        "frontmatter/how_to_use_this_book.tex",
        "appendices/glossary.tex",
        "appendices/notation.tex",
    ]:
        path = BOOK_DIR / name
        if not path.exists():
            errors.append(f"book/{name} missing")

    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)

    manifest = load_json(BOOK_DIR / "textbook_source_manifest.json")
    if not isinstance(manifest, dict) or "course" not in manifest or "chapters" not in manifest:
        errors.append("book/textbook_source_manifest.json must contain course and chapters")

    for lecture_dir in active_lecture_dirs(args.lectures):
        errors.extend(validate_lecture(lecture_dir, compile_pdf=args.compile))

    main_tex = BOOK_DIR / "main.tex"
    if args.compile:
        try:
            compile_tex(main_tex)
        except subprocess.CalledProcessError as exc:
            errors.append("book/main.tex failed to compile")
            if exc.stdout:
                errors.append(exc.stdout[-4000:])

    if args.require_book_pdf and not (BOOK_DIR / "textbook.pdf").exists():
        errors.append("book/textbook.pdf missing")

    if args.require_deliverable:
        deliverable_book_dir = DELIVERABLE_DIR / "book"
        if not deliverable_book_dir.exists():
            errors.append("deliverable/book missing")
        else:
            tex_files = sorted(deliverable_book_dir.glob("*_complete_notes.tex"))
            pdf_files = sorted(deliverable_book_dir.glob("*_complete_notes.pdf"))
            if not tex_files:
                errors.append("deliverable/book missing merged textbook .tex")
            if not pdf_files:
                errors.append("deliverable/book missing merged textbook .pdf")
        deliverable_lectures_dir = DELIVERABLE_DIR / "lectures"
        for lecture_dir in active_lecture_dirs(args.lectures):
            tex_files = sorted(lecture_dir.glob("lecture_*_note.tex"))
            pdf_files = sorted(lecture_dir.glob("lecture_*_note.pdf"))
            if not tex_files or not pdf_files:
                continue
            target_dir = deliverable_lectures_dir / lecture_dir.name
            if not target_dir.exists():
                errors.append(f"deliverable/lectures/{lecture_dir.name} missing")
                continue
            if not list(target_dir.glob("lecture_*_note.tex")):
                errors.append(f"deliverable/lectures/{lecture_dir.name} missing lecture .tex")
            if not list(target_dir.glob("lecture_*_note.pdf")):
                errors.append(f"deliverable/lectures/{lecture_dir.name} missing lecture .pdf")

    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)

    print("ok textbook")


if __name__ == "__main__":
    main()
