#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
LECTURES_DIR = RUN_ROOT / "lectures"
ALLOWED_COVERAGE_STATUSES = {"covered", "partial", "duplicate", "omitted", "unclassified"}


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def lecture_sort_key(path: Path) -> tuple[int, str]:
    match = re.match(r"^(\d+)_", path.name)
    if match:
        return int(match.group(1)), path.name
    return 10**9, path.name


def lecture_dirs(selectors: list[str] | None) -> list[Path]:
    dirs = sorted((path for path in LECTURES_DIR.iterdir() if path.is_dir() and re.match(r"^\d+_", path.name)), key=lecture_sort_key)
    if not selectors:
        return dirs
    selected: list[Path] = []
    for token in selectors:
        match = None
        for path in dirs:
            if path.name == token or path.name.startswith(token) or path.name.split("_", 1)[0] == token:
                match = path
                break
        if match is None:
            raise SystemExit(f"unknown lecture selector: {token}")
        selected.append(match)
    return selected


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


def latest_eval_report(lecture_dir: Path) -> Path | None:
    reports = sorted((lecture_dir / "eval_reports").glob("pass_*.json"))
    return reports[-1] if reports else None


def select_tex(lecture_dir: Path) -> Path | None:
    preferred = lecture_dir / "lecture_XX_note.tex"
    if preferred.exists():
        return preferred
    candidates = sorted(lecture_dir.glob("lecture_*_note.tex"))
    return candidates[0] if candidates else None


def select_pdf(lecture_dir: Path) -> Path | None:
    preferred = lecture_dir / "lecture_XX_note.pdf"
    if preferred.exists():
        return preferred
    candidates = sorted(lecture_dir.glob("lecture_*_note.pdf"))
    return candidates[0] if candidates else None


def validate_lecture(lecture_dir: Path, compile_pdf: bool = False) -> list[str]:
    errors: list[str] = []

    required_paths = [
        "source_manifest.json",
        "transcript.jsonl",
        "slides.jsonl",
        "segments.jsonl",
        "lecture_plan.json",
        "coverage_units.jsonl",
        "omission_log.jsonl",
        "figure_plan.json",
        "figure_manifest.json",
        "repair_log.jsonl",
    ]
    for name in required_paths:
        path = lecture_dir / name
        if not path.exists():
            errors.append(f"{lecture_dir.name}: missing {name}")

    tex_path = select_tex(lecture_dir)
    if tex_path is None:
        errors.append(f"{lecture_dir.name}: missing lecture_XX_note.tex")
    pdf_path = select_pdf(lecture_dir)
    if pdf_path is None:
        errors.append(f"{lecture_dir.name}: missing lecture_XX_note.pdf")
    eval_path = latest_eval_report(lecture_dir)
    if eval_path is None:
        errors.append(f"{lecture_dir.name}: missing eval_reports/pass_##.json")

    if errors:
        return errors

    if compile_pdf and tex_path is not None:
        try:
            compile_tex(tex_path)
        except subprocess.CalledProcessError as exc:
            errors.append(f"{lecture_dir.name}: xelatex failed for {tex_path.name}")
            if exc.stdout:
                errors.append(exc.stdout[-4000:])
            return errors

    source_manifest = load_json(lecture_dir / "source_manifest.json")
    transcript_rows = load_jsonl(lecture_dir / "transcript.jsonl")
    slide_rows = load_jsonl(lecture_dir / "slides.jsonl")
    segment_rows = load_jsonl(lecture_dir / "segments.jsonl")
    plan = load_json(lecture_dir / "lecture_plan.json")
    coverage_rows = load_jsonl(lecture_dir / "coverage_units.jsonl")
    omission_rows = load_jsonl(lecture_dir / "omission_log.jsonl")
    figure_plan = load_json(lecture_dir / "figure_plan.json")
    figure_manifest = load_json(lecture_dir / "figure_manifest.json")
    repair_rows = load_jsonl(lecture_dir / "repair_log.jsonl")
    eval_report = load_json(eval_path)

    if not isinstance(source_manifest, dict):
        errors.append(f"{lecture_dir.name}: source_manifest.json must be an object")
    if not isinstance(plan, dict):
        errors.append(f"{lecture_dir.name}: lecture_plan.json must be an object")
    if not isinstance(figure_plan, list):
        errors.append(f"{lecture_dir.name}: figure_plan.json must be an array")
    if not isinstance(figure_manifest, list):
        errors.append(f"{lecture_dir.name}: figure_manifest.json must be an array")
    if not isinstance(eval_report, dict):
        errors.append(f"{lecture_dir.name}: latest eval report must be an object")

    for field in ["lecture_id", "title", "course_mode", "source_inventory", "segment_ids", "must_cover_kinds", "must_emit_artifacts", "evaluator_thresholds"]:
        if not isinstance(plan, dict) or field not in plan:
            errors.append(f"{lecture_dir.name}: lecture_plan.json missing {field}")

    for row in transcript_rows:
        if not all(key in row for key in ["unit_id", "source_type", "source_id", "loc", "text", "required"]):
            errors.append(f"{lecture_dir.name}: invalid transcript.jsonl row")
            break
        loc = row.get("loc")
        if not isinstance(loc, dict) or "start" not in loc or "end" not in loc:
            errors.append(f"{lecture_dir.name}: transcript row missing loc.start/end")
            break

    for row in slide_rows:
        if not all(key in row for key in ["unit_id", "source_type", "source_id", "loc", "text", "asset_path", "required"]):
            errors.append(f"{lecture_dir.name}: invalid slides.jsonl row")
            break

    for row in segment_rows:
        if not all(key in row for key in ["segment_id", "start", "end", "source_unit_ids", "target_section_hint"]):
            errors.append(f"{lecture_dir.name}: invalid segments.jsonl row")
            break

    for row in coverage_rows:
        if not all(key in row for key in ["unit_id", "source_type", "source_id", "loc", "kind", "summary", "required", "status", "mapped_section", "figure_ids", "notes"]):
            errors.append(f"{lecture_dir.name}: invalid coverage_units.jsonl row")
            break
        if row.get("status") not in ALLOWED_COVERAGE_STATUSES:
            errors.append(f"{lecture_dir.name}: invalid coverage status {row.get('status')!r}")
        if row.get("required") is True and row.get("status") == "unclassified":
            errors.append(f"{lecture_dir.name}: required unit {row.get('unit_id')} remains unclassified")
        if row.get("status") == "partial" and not str(row.get("notes", "")).strip():
            errors.append(f"{lecture_dir.name}: partial unit {row.get('unit_id')} missing notes")

    omission_by_unit = {row.get("unit_id"): row for row in omission_rows if isinstance(row, dict)}
    for row in coverage_rows:
        if row.get("status") == "omitted" and row.get("unit_id") not in omission_by_unit:
            errors.append(f"{lecture_dir.name}: omitted unit {row.get('unit_id')} missing omission_log entry")

    for row in figure_plan:
        if not all(key in row for key in ["figure_id", "source_unit_ids", "asset_candidates", "selection_reason", "required", "provenance_type", "time_provenance"]):
            errors.append(f"{lecture_dir.name}: invalid figure_plan.json row")
            break

    for row in figure_manifest:
        if not all(key in row for key in ["figure_id", "source_id", "loc", "asset_path", "caption", "crop", "used_in_section", "time_provenance"]):
            errors.append(f"{lecture_dir.name}: invalid figure_manifest.json row")
            break
        if "frames/" in str(row.get("asset_path", "")) and not row.get("time_provenance"):
            errors.append(f"{lecture_dir.name}: frame-derived figure missing time_provenance")

    if not isinstance(eval_report, dict):
        errors.append(f"{lecture_dir.name}: evaluator report must be a JSON object")
    else:
        if eval_report.get("overall") != "pass":
            errors.append(f"{lecture_dir.name}: evaluator overall != pass")
        if eval_report.get("blocking_issues"):
            errors.append(f"{lecture_dir.name}: evaluator blocking_issues not empty")

    tex = tex_path.read_text() if tex_path is not None else ""
    if "[cite]" in tex:
        errors.append(f"{lecture_dir.name}: unresolved [cite] placeholder in tex")
    if "\\section{总结与延伸}" not in tex:
        errors.append(f"{lecture_dir.name}: missing 总结与延伸 section")
    if "\\subsection{本章小结}" not in tex:
        errors.append(f"{lecture_dir.name}: missing 本章小结 subsection")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("lectures", nargs="*")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    failures = 0
    for lecture_dir in lecture_dirs(args.lectures):
        errors = validate_lecture(lecture_dir, compile_pdf=args.compile)
        if errors:
            failures += 1
            print(f"{lecture_dir.name}: fail")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"{lecture_dir.name}: pass")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
