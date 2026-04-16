#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
LECTURES_DIR = RUN_ROOT / "lectures"


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def lecture_dirs(selectors: list[str] | None) -> list[Path]:
    dirs = sorted(path for path in LECTURES_DIR.iterdir() if path.is_dir() and path.name.startswith("lec"))
    if not selectors:
        return dirs
    resolved: list[Path] = []
    for selector in selectors:
        match = None
        for path in dirs:
            if path.name == selector or path.name.startswith(selector):
                match = path
                break
        if match is None:
            raise SystemExit(f"unknown lecture selector: {selector}")
        resolved.append(match)
    return resolved


def select_tex(lecture_dir: Path) -> Path | None:
    candidates = [
        lecture_dir / "lecture_repaired.tex",
        lecture_dir / "lecture.tex",
    ]
    note_candidates = sorted(lecture_dir.glob("lecture_*_note.tex"))
    candidates.extend(note_candidates)
    for path in candidates:
        if path.exists():
            return path
    return None


def select_eval_report(lecture_dir: Path) -> Path | None:
    direct = lecture_dir / "eval_report.json"
    if direct.exists():
        return direct
    reports = sorted((lecture_dir / "eval_reports").glob("pass_*.json"))
    return reports[-1] if reports else None


def compile_tex(tex_path: Path) -> tuple[bool, str]:
    try:
        for _ in range(2):
            subprocess.run(
                ["xelatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
                cwd=tex_path.parent,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        return True, "xelatex pass"
    except subprocess.CalledProcessError as exc:
        return False, exc.stdout[-4000:] if exc.stdout else str(exc)


def validator_status(lecture_dir: Path) -> tuple[bool, list[str]]:
    errors: list[str] = []

    source_manifest = lecture_dir / "source_manifest.json"
    transcript_jsonl = lecture_dir / "transcript.jsonl"
    slides_jsonl = lecture_dir / "slides.jsonl"
    coverage_jsonl = lecture_dir / "coverage_units.jsonl"
    omission_jsonl = lecture_dir / "omission_log.jsonl"
    figure_manifest_path = lecture_dir / "figure_manifest.json"
    tex_path = select_tex(lecture_dir)
    eval_report_path = select_eval_report(lecture_dir)

    if not source_manifest.exists() or source_manifest.stat().st_size == 0:
        errors.append("missing or empty source_manifest.json")
    if not transcript_jsonl.exists() or transcript_jsonl.stat().st_size == 0:
        errors.append("missing or empty transcript.jsonl")
    if not coverage_jsonl.exists() or coverage_jsonl.stat().st_size == 0:
        errors.append("missing or empty coverage_units.jsonl")
    if not figure_manifest_path.exists():
        errors.append("missing figure_manifest.json")
    if tex_path is None:
        errors.append("missing lecture tex source")
    if eval_report_path is None:
        errors.append("missing evaluator report")

    slides_required = (lecture_dir / "slides.pdf").exists() or (lecture_dir / "slides_main.pdf").exists()
    if slides_required and (not slides_jsonl.exists() or slides_jsonl.stat().st_size == 0):
        errors.append("slides are available but slides.jsonl is missing or empty")

    if errors:
        return False, errors

    transcript_rows = load_jsonl(transcript_jsonl)
    for row in transcript_rows[:20]:
        has_ts = ("start" in row and "end" in row) or (
            isinstance(row.get("loc"), dict)
            and row["loc"].get("start")
            and row["loc"].get("end")
        )
        if not has_ts:
            errors.append("transcript.jsonl does not preserve timestamps")
            break

    coverage_rows = load_jsonl(coverage_jsonl)
    omission_rows = load_jsonl(omission_jsonl) if omission_jsonl.exists() else []
    omission_by_unit = {
        row.get("unit_id"): row
        for row in omission_rows
        if isinstance(row, dict) and row.get("unit_id")
    }
    if not coverage_rows:
        errors.append("coverage_units.jsonl is empty")
    for row in coverage_rows:
        required = row.get("importance") == "required" or row.get("required") is True
        status = row.get("status")
        if required and status in {"planned", "unclassified", None}:
            errors.append(f"required unit left unresolved: {row.get('unit_id')}")
        if status == "omitted" and not (row.get("omission_reason") or omission_by_unit.get(row.get("unit_id"))):
            errors.append(f"omitted unit missing omission reason: {row.get('unit_id')}")

    if figure_manifest_path.exists():
        figure_manifest = load_json(figure_manifest_path)
        if not isinstance(figure_manifest, list):
            errors.append("figure_manifest.json must be a list")
        else:
            for figure in figure_manifest:
                if not isinstance(figure, dict):
                    errors.append("figure_manifest.json contains non-object entry")
                    continue
                if not figure.get("source_ref"):
                    errors.append(f"figure missing source_ref: {figure.get('figure_id') or figure.get('asset_path')}")

    if eval_report_path is not None:
        report = load_json(eval_report_path)
        if not isinstance(report, dict):
            errors.append("evaluator report must be a JSON object")
        elif report.get("overall") != "pass":
            errors.append("evaluator overall != pass")

    if tex_path is not None:
        ok, detail = compile_tex(tex_path)
        if not ok:
            errors.append(f"LaTeX compile failed: {detail}")

    return not errors, errors


def write_quality_report(lecture_dir: Path, passed: bool, errors: list[str]) -> None:
    lines = [
        f"# Lecture Quality Report: {lecture_dir.name}",
        "",
        f"- validator_status: {'pass' if passed else 'fail'}",
        f"- evaluator_report: {(select_eval_report(lecture_dir) or Path('missing')).name}",
        f"- tex_source: {(select_tex(lecture_dir) or Path('missing')).name}",
        "",
    ]
    if errors:
        lines.append("## Errors")
        lines.extend([f"- {error}" for error in errors])
    else:
        lines.append("## Result")
        lines.append("- All required lecture-level checks passed.")
    (lecture_dir / "lecture_quality_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="Accepted for compatibility; validation always compiles the lecture tex.")
    parser.add_argument("lectures", nargs="*")
    args = parser.parse_args()

    failures = 0
    for lecture_dir in lecture_dirs(args.lectures):
        passed, errors = validator_status(lecture_dir)
        write_quality_report(lecture_dir, passed, errors)
        print(f"{lecture_dir.name}: {'pass' if passed else 'fail'}")
        if errors:
            for error in errors:
                print(f"  - {error}")
            failures += 1

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
