#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
LECTURES_DIR = RUN_ROOT / "lectures"
BOOK_DIR = RUN_ROOT / "book"
DELIVERABLE_DIR = RUN_ROOT / "deliverable" / "book"
SUPPLEMENTS_DIR = RUN_ROOT / "supplements"


def require(condition: bool, errors: list[str], message: str) -> None:
    if not condition:
        errors.append(message)


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def lecture_dirs() -> list[Path]:
    return sorted(path for path in LECTURES_DIR.iterdir() if path.is_dir())


def supplement_dirs() -> list[Path]:
    if not SUPPLEMENTS_DIR.exists():
        return []
    return sorted(path for path in SUPPLEMENTS_DIR.iterdir() if path.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-book-pdf", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []
    course_manifest = RUN_ROOT / "COURSE_SOURCE_MANIFEST.json"
    coverage_index = RUN_ROOT / "COURSE_COVERAGE_INDEX.jsonl"
    omission_log = RUN_ROOT / "COURSE_OMISSION_LOG.jsonl"
    require(course_manifest.exists(), errors, "COURSE_SOURCE_MANIFEST.json missing")
    require(coverage_index.exists(), errors, "COURSE_COVERAGE_INDEX.jsonl missing")
    require(omission_log.exists(), errors, "COURSE_OMISSION_LOG.jsonl missing")

    for lecture_dir in lecture_dirs():
        manifest = lecture_dir / "source_manifest.json"
        coverage = lecture_dir / "coverage_units.jsonl"
        figure_manifest = lecture_dir / "figure_manifest.json"
        lecture_quality = lecture_dir / "lecture_quality_report.md"
        lecture_pdf = lecture_dir / "lecture.pdf"
        note_pdf_candidates = sorted(lecture_dir.glob("lecture_*_note.pdf"))
        if not lecture_pdf.exists() and note_pdf_candidates:
            lecture_pdf = note_pdf_candidates[0]
        eval_report = lecture_dir / "eval_report.json"
        if not eval_report.exists() and (lecture_dir / "eval_reports").exists():
            reports = sorted((lecture_dir / "eval_reports").glob("pass_*.json"))
            if reports:
                eval_report = reports[-1]
        require(manifest.exists() and manifest.stat().st_size > 0, errors, f"{lecture_dir.name}: missing source_manifest.json")
        require(coverage.exists() and coverage.stat().st_size > 0, errors, f"{lecture_dir.name}: missing coverage_units.jsonl")
        require(figure_manifest.exists(), errors, f"{lecture_dir.name}: missing figure_manifest.json")
        require(eval_report.exists(), errors, f"{lecture_dir.name}: missing eval_report.json")
        require(lecture_pdf.exists(), errors, f"{lecture_dir.name}: missing lecture.pdf")
        require(lecture_quality.exists(), errors, f"{lecture_dir.name}: missing lecture_quality_report.md")
        if eval_report.exists():
            report = load_json(eval_report)
            if isinstance(report, dict):
                require(report.get("overall") == "pass", errors, f"{lecture_dir.name}: evaluator did not pass")
        if lecture_quality.exists():
            text = lecture_quality.read_text()
            require("validator_status: pass" in text, errors, f"{lecture_dir.name}: lecture validator did not pass")

    for supplement_dir in supplement_dirs():
        manifest = supplement_dir / "COURSE_SOURCE_MANIFEST.json"
        coverage = supplement_dir / "COURSE_COVERAGE_INDEX.jsonl"
        omission = supplement_dir / "COURSE_OMISSION_LOG.jsonl"
        tex = supplement_dir / "course_extension.tex"
        eval_report = supplement_dir / "supplement_eval.json"
        require(manifest.exists() and manifest.stat().st_size > 0, errors, f"{supplement_dir.name}: missing COURSE_SOURCE_MANIFEST.json")
        require(coverage.exists() and coverage.stat().st_size > 0, errors, f"{supplement_dir.name}: missing COURSE_COVERAGE_INDEX.jsonl")
        require(omission.exists(), errors, f"{supplement_dir.name}: missing COURSE_OMISSION_LOG.jsonl")
        require(tex.exists() and tex.stat().st_size > 0, errors, f"{supplement_dir.name}: missing course_extension.tex")
        require(eval_report.exists(), errors, f"{supplement_dir.name}: missing supplement_eval.json")
        if eval_report.exists():
            report = load_json(eval_report)
            if isinstance(report, dict):
                require(report.get("overall") == "pass", errors, f"{supplement_dir.name}: supplement evaluator did not pass")

    textbook_manifest = BOOK_DIR / "textbook_source_manifest.json"
    main_tex = BOOK_DIR / "main.tex"
    outline = BOOK_DIR / "BOOK_OUTLINE.md"
    require(main_tex.exists(), errors, "book/main.tex missing")
    require(textbook_manifest.exists(), errors, "book/textbook_source_manifest.json missing")
    require(outline.exists(), errors, "book/BOOK_OUTLINE.md missing")
    require((BOOK_DIR / "frontmatter" / "preface.tex").exists(), errors, "book/frontmatter/preface.tex missing")
    require((BOOK_DIR / "frontmatter" / "how_to_use_this_book.tex").exists(), errors, "book/frontmatter/how_to_use_this_book.tex missing")
    require((BOOK_DIR / "appendices" / "glossary.tex").exists(), errors, "book/appendices/glossary.tex missing")
    require((BOOK_DIR / "appendices" / "notation.tex").exists(), errors, "book/appendices/notation.tex missing")
    require((BOOK_DIR / "appendices" / "exercises.tex").exists(), errors, "book/appendices/exercises.tex missing")
    require((BOOK_DIR / "appendices" / "paper_map.tex").exists(), errors, "book/appendices/paper_map.tex missing")
    require((BOOK_DIR / "appendices" / "benchmark_map.tex").exists(), errors, "book/appendices/benchmark_map.tex missing")
    require((BOOK_DIR / "appendices" / "algorithm_index.tex").exists(), errors, "book/appendices/algorithm_index.tex missing")
    require((BOOK_DIR / "appendices" / "figure_provenance.tex").exists(), errors, "book/appendices/figure_provenance.tex missing")
    require((BOOK_DIR / "appendices" / "omission_log.tex").exists(), errors, "book/appendices/omission_log.tex missing")
    require((BOOK_DIR / "appendices" / "suggested_reading_paths.tex").exists(), errors, "book/appendices/suggested_reading_paths.tex missing")
    require((DELIVERABLE_DIR / "cs294_194_280_sp25_agents_textbook_complete_notes.tex").exists(), errors, "deliverable/book final tex missing")
    require((DELIVERABLE_DIR / "cs294_194_280_sp25_agents_textbook_complete_notes.pdf").exists(), errors, "deliverable/book final pdf missing")
    if args.require_book_pdf:
        require((BOOK_DIR / "textbook.pdf").exists(), errors, "book/textbook.pdf missing")

    if errors:
        for message in errors:
            print(message, file=sys.stderr)
        raise SystemExit(1)

    print("ok textbook")


if __name__ == "__main__":
    main()
