#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = RUN_ROOT / "build"


def require(condition: bool, errors: list[str], message: str) -> None:
    if not condition:
        errors.append(message)


def lecture_status(lecture: dict) -> tuple[str, str]:
    eval_ref = lecture.get("latest_eval_report")
    pdf_ref = lecture.get("lecture_pdf")
    eval_ok = False
    reason = "lecture PDF not available"
    if eval_ref:
        eval_path = RUN_ROOT / eval_ref
        if eval_path.exists():
            report = json.loads(eval_path.read_text())
            eval_ok = report.get("overall") == "pass"
            if not eval_ok:
                issues = report.get("blocking_issues") or []
                if issues:
                    reason = "; ".join(issue.get("problem", "blocking issue") for issue in issues[:3] if isinstance(issue, dict))
                else:
                    reason = "latest evaluator report did not pass"
            else:
                reason = ""
    pdf_ok = bool(pdf_ref) and not str(pdf_ref).endswith("missing.pdf") and (RUN_ROOT / pdf_ref).exists()
    if eval_ok and pdf_ok:
        return "deliverable", ""
    if not pdf_ok and not reason:
        reason = "lecture PDF not available"
    return "blocked", reason or "lecture not deliverable"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-book-pdf", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []
    manifest_path = BUILD_DIR / "course_manifest.json"
    require(manifest_path.exists(), errors, "build/course_manifest.json missing")
    if errors:
        for message in errors:
            print(message, file=sys.stderr)
        raise SystemExit(1)

    manifest = json.loads(manifest_path.read_text())
    summary = {
        "course_id": manifest.get("course_id"),
        "title": manifest.get("title"),
        "deliverable_lectures": [],
        "blocked_lectures": [],
    }
    for lecture in manifest.get("lectures", []):
        lecture_dir = RUN_ROOT / "lectures" / lecture["lecture_slug"]
        require((lecture_dir / "source_manifest.json").exists(), errors, f"{lecture['lecture_slug']}: missing source_manifest.json")
        require((lecture_dir / "coverage_units.jsonl").exists(), errors, f"{lecture['lecture_slug']}: missing coverage_units.jsonl")
        require((lecture_dir / "omission_log.jsonl").exists(), errors, f"{lecture['lecture_slug']}: missing omission_log.jsonl")
        status, reason = lecture_status(lecture)
        row = {
            "lecture_id": lecture["lecture_id"],
            "lecture_slug": lecture["lecture_slug"],
            "title": lecture["title"],
            "date": lecture["date"],
            "status": status,
            "reason": reason,
        }
        if status == "deliverable":
            summary["deliverable_lectures"].append(row)
        else:
            summary["blocked_lectures"].append(row)

    if args.require_book_pdf:
        require((BUILD_DIR / "speech_recognition_understanding_fall2023_textbook.pdf").exists(), errors, "merged textbook PDF missing")

    (BUILD_DIR / "textbook_status.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    if errors:
        for message in errors:
            print(message, file=sys.stderr)
        raise SystemExit(1)

    print(
        f"ok textbook deliverable={len(summary['deliverable_lectures'])} blocked={len(summary['blocked_lectures'])}"
    )


if __name__ == "__main__":
    main()
