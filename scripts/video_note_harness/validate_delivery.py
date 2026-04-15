#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.video_note_harness.common import (
    ALLOWED_COVERAGE_STATUSES,
    find_latest_eval_report,
    infer_code_units,
    infer_formula_units,
    load_json,
    load_jsonl,
    parse_tex_figure_assets,
)


def has_math_markup(tex: str) -> bool:
    if any(token in tex for token in [r"\[", r"\begin{equation}", r"\begin{align}", "$$"]):
        return True
    return bool(re.search(r"(?<!\\)\$[^$]+\$", tex))


def has_symbol_explanation(tex: str) -> bool:
    if ("符号" in tex) or ("其中" in tex):
        return True
    if re.search(r"transition\s*\([^)]*\)", tex, re.I):
        return True
    if re.search(r"状态.*动作|动作.*价值|回报.*折扣|折扣.*回报", tex):
        return True
    return False


def require(condition: bool, errors: list[str], message: str) -> None:
    if not condition:
        errors.append(message)


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


def lecture_dirs(run_root: Path, selectors: list[str] | None) -> list[Path]:
    dirs = sorted(path for path in (run_root / "lectures").iterdir() if path.is_dir() and path.name[:2].isdigit())
    if not selectors:
        return dirs
    resolved: list[Path] = []
    for token in selectors:
        matched = None
        for path in dirs:
            if path.name == token or path.name.startswith(token + "_") or path.name.startswith(token):
                matched = path
                break
        if matched is None:
            raise SystemExit(f"unknown lecture selector: {token}")
        resolved.append(matched)
    return resolved


def is_legacy_lecture(lecture_dir: Path) -> bool:
    return not (lecture_dir / "lecture_plan.json").exists()


def validate_legacy_lecture(lecture_dir: Path, compile_pdf: bool) -> list[str]:
    errors: list[str] = []
    required_paths = {
        "source_manifest.json": False,
        "coverage_units.jsonl": False,
        "omission_log.jsonl": False,
        "figure_manifest.json": False,
    }
    for name, must_be_non_empty in required_paths.items():
        path = lecture_dir / name
        require(path.exists(), errors, f"{lecture_dir.name}: missing {name}")
        if path.exists() and must_be_non_empty:
            require(path.stat().st_size > 0, errors, f"{lecture_dir.name}: empty {name}")

    tex_files = sorted(lecture_dir.glob("lecture_*_note.tex"))
    require(bool(tex_files), errors, f"{lecture_dir.name}: missing lecture_XX_note.tex")
    if errors:
        return errors

    tex_path = tex_files[0]
    if compile_pdf:
        try:
            compile_tex(tex_path)
        except subprocess.CalledProcessError as exc:
            errors.append(f"{lecture_dir.name}: xelatex failed for {tex_path.name}: {exc}")
            return errors

    coverage_rows = load_jsonl(lecture_dir / "coverage_units.jsonl") if (lecture_dir / "coverage_units.jsonl").exists() else []
    omission_rows = load_jsonl(lecture_dir / "omission_log.jsonl") if (lecture_dir / "omission_log.jsonl").exists() else []
    figure_manifest = load_json(lecture_dir / "figure_manifest.json") if (lecture_dir / "figure_manifest.json").exists() else []
    tex_assets = parse_tex_figure_assets(tex_path)

    if coverage_rows:
        omission_by_unit = {row.get("unit_id"): row for row in omission_rows if isinstance(row, dict)}
        for row in coverage_rows:
            status = row.get("status", "covered")
            if status == "included":
                status = "covered"
            require(status in ALLOWED_COVERAGE_STATUSES or status == "covered", errors, f"{lecture_dir.name}: invalid coverage status {status!r}")
            if row.get("required") is True:
                require(status != "unclassified", errors, f"{lecture_dir.name}: required unit {row.get('unit_id')} remains unclassified")
            if status == "omitted":
                require(row.get("unit_id") in omission_by_unit, errors, f"{lecture_dir.name}: omitted unit {row.get('unit_id')} missing omission_log entry")

    manifest_assets = {
        row.get("asset_path")
        for row in figure_manifest
        if isinstance(row, dict) and row.get("asset_path")
    } if isinstance(figure_manifest, list) else set()
    for asset in tex_assets:
        require(asset in manifest_assets, errors, f"{lecture_dir.name}: figure_manifest.json missing asset {asset}")
    return errors


def validate_harness_lecture(lecture_dir: Path, compile_pdf: bool) -> list[str]:
    errors: list[str] = []
    required_paths = {
        "source_manifest.json": True,
        "transcript.jsonl": True,
        "slides.jsonl": True,
        "segments.jsonl": True,
        "lecture_plan.json": True,
        "coverage_units.jsonl": True,
        "omission_log.jsonl": False,
        "figure_plan.json": False,
        "figure_manifest.json": False,
        "repair_log.jsonl": False,
    }
    for name, must_be_non_empty in required_paths.items():
        path = lecture_dir / name
        require(path.exists(), errors, f"{lecture_dir.name}: missing {name}")
        if path.exists() and must_be_non_empty:
            require(path.stat().st_size > 0, errors, f"{lecture_dir.name}: empty {name}")

    tex_files = sorted(lecture_dir.glob("lecture_*_note.tex"))
    require(bool(tex_files), errors, f"{lecture_dir.name}: missing lecture_XX_note.tex")
    eval_report_path = find_latest_eval_report(lecture_dir)
    require(eval_report_path is not None, errors, f"{lecture_dir.name}: missing eval_reports/pass_##.json")
    if errors:
        return errors

    tex_path = tex_files[0]
    if compile_pdf:
        try:
            compile_tex(tex_path)
        except subprocess.CalledProcessError as exc:
            errors.append(f"{lecture_dir.name}: xelatex failed for {tex_path.name}: {exc}")
            return errors

    source_manifest = load_json(lecture_dir / "source_manifest.json")
    transcript_units = load_jsonl(lecture_dir / "transcript.jsonl")
    slide_units = load_jsonl(lecture_dir / "slides.jsonl")
    segments = load_jsonl(lecture_dir / "segments.jsonl")
    lecture_plan = load_json(lecture_dir / "lecture_plan.json")
    coverage_rows = load_jsonl(lecture_dir / "coverage_units.jsonl")
    omission_rows = load_jsonl(lecture_dir / "omission_log.jsonl")
    figure_plan = load_json(lecture_dir / "figure_plan.json")
    figure_manifest = load_json(lecture_dir / "figure_manifest.json")
    repair_rows = load_jsonl(lecture_dir / "repair_log.jsonl")
    eval_report = load_json(eval_report_path)

    require(isinstance(source_manifest, dict), errors, f"{lecture_dir.name}: source_manifest.json must be an object")
    require(isinstance(lecture_plan, dict), errors, f"{lecture_dir.name}: lecture_plan.json must be an object")
    require(isinstance(figure_plan, list), errors, f"{lecture_dir.name}: figure_plan.json must be an array")
    require(isinstance(figure_manifest, list), errors, f"{lecture_dir.name}: figure_manifest.json must be an array")
    require(isinstance(eval_report, dict), errors, f"{lecture_dir.name}: eval report must be an object")

    for field in ["lecture_id", "title", "course_mode", "source_inventory", "segment_ids", "must_cover_kinds", "must_emit_artifacts", "evaluator_thresholds"]:
        require(field in lecture_plan, errors, f"{lecture_dir.name}: lecture_plan.json missing {field}")
    for row in transcript_units:
        require("unit_id" in row and "loc" in row and "text" in row, errors, f"{lecture_dir.name}: invalid transcript.jsonl row")
    for row in slide_units:
        require("unit_id" in row and "loc" in row and "asset_path" in row, errors, f"{lecture_dir.name}: invalid slides.jsonl row")
    for row in segments:
        for field in ["segment_id", "source_unit_ids", "target_section_hint"]:
            require(field in row, errors, f"{lecture_dir.name}: segments.jsonl row missing {field}")
    for row in figure_plan:
        for field in ["figure_id", "source_unit_ids", "asset_candidates", "selection_reason", "required", "provenance_type", "time_provenance"]:
            require(field in row, errors, f"{lecture_dir.name}: figure_plan.json row missing {field}")

    for field in ["pass", "target", "overall", "scores", "blocking_issues", "warnings", "repair_required"]:
        require(field in eval_report, errors, f"{lecture_dir.name}: eval report missing {field}")

    tex = tex_path.read_text()
    tex_assets = parse_tex_figure_assets(tex_path)
    omission_by_unit = {row.get("unit_id"): row for row in omission_rows if isinstance(row, dict)}
    for row in coverage_rows:
        status = row.get("status")
        require(status in ALLOWED_COVERAGE_STATUSES, errors, f"{lecture_dir.name}: invalid coverage status {status!r}")
        if row.get("required") is True:
            require(status != "unclassified", errors, f"{lecture_dir.name}: required unit {row.get('unit_id')} remains unclassified")
        if status == "partial":
            require(bool(str(row.get("notes", "")).strip()), errors, f"{lecture_dir.name}: partial unit {row.get('unit_id')} missing notes")
        if status == "omitted":
            omission = omission_by_unit.get(row.get("unit_id"))
            require(omission is not None, errors, f"{lecture_dir.name}: omitted unit {row.get('unit_id')} missing omission_log entry")
            if omission is not None:
                require(bool(omission.get("user_visible_note")), errors, f"{lecture_dir.name}: omitted unit {row.get('unit_id')} missing user_visible_note")
        mapped = str(row.get("mapped_section") or "")
        require(mapped != "本章小结", errors, f"{lecture_dir.name}: coverage unit {row.get('unit_id')} maps only to 本章小结")

    formula_units = infer_formula_units(coverage_rows)
    if formula_units:
        has_display_math = has_math_markup(tex)
        symbol_explanation = has_symbol_explanation(tex)
        require(has_display_math, errors, f"{lecture_dir.name}: formula-oriented coverage exists but no display math detected")
        require(symbol_explanation, errors, f"{lecture_dir.name}: formula-oriented coverage exists but no symbol explanation detected")

    code_units = infer_code_units(coverage_rows)
    if code_units:
        has_listing = r"\begin{lstlisting}" in tex
        has_code_explanation = ("代码" in tex) or ("实现" in tex) or ("伪代码" in tex)
        require(has_listing or has_code_explanation, errors, f"{lecture_dir.name}: code-oriented coverage exists but no listing/explanation detected")

    manifest_assets = {
        row.get("asset_path")
        for row in figure_manifest
        if isinstance(row, dict) and row.get("asset_path")
    }
    for asset in tex_assets:
        require(asset in manifest_assets, errors, f"{lecture_dir.name}: figure_manifest.json missing asset {asset}")

    for row in figure_manifest:
        if not isinstance(row, dict):
            errors.append(f"{lecture_dir.name}: figure_manifest.json contains non-object entry")
            continue
        if "frames/" in str(row.get("asset_path", "")):
            require(bool(row.get("time_provenance")), errors, f"{lecture_dir.name}: frame asset {row.get('asset_path')} missing time_provenance")

    require(eval_report.get("overall") == "pass", errors, f"{lecture_dir.name}: latest evaluator report did not pass")
    blocking_issues = eval_report.get("blocking_issues") or []
    if blocking_issues:
        repair_statuses = {row.get("status") for row in repair_rows if isinstance(row, dict)}
        require(bool({"fixed", "accepted"} & repair_statuses), errors, f"{lecture_dir.name}: blocking issues exist without accepted/fixed repair log entries")

    require("[cite]" not in tex, errors, f"{lecture_dir.name}: note contains unresolved citation placeholder")
    require(r"\section{总结与延伸}" in tex, errors, f"{lecture_dir.name}: note missing 总结与延伸 section")
    require(r"\subsection{本章小结}" in tex, errors, f"{lecture_dir.name}: note missing 本章小结 subsection")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("lectures", nargs="*")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = (REPO_ROOT / args.run_root).resolve()

    all_errors: list[str] = []
    for lecture_dir in lecture_dirs(run_root, args.lectures):
        errors = validate_legacy_lecture(lecture_dir, args.compile) if is_legacy_lecture(lecture_dir) else validate_harness_lecture(lecture_dir, args.compile)
        if errors:
            all_errors.extend(errors)
        else:
            print(f"ok {lecture_dir.name}")

    if all_errors:
        for message in all_errors:
            print(message, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
