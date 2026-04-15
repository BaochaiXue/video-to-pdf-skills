#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.video_note_harness.common import (
    DEFAULT_THRESHOLDS,
    find_latest_eval_report,
    infer_code_units,
    infer_formula_units,
    load_json,
    load_jsonl,
    parse_tex_figure_assets,
    write_json,
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


def latest_eval_pass(lecture_dir: Path) -> int:
    latest = find_latest_eval_report(lecture_dir)
    if not latest:
        return 1
    stem = latest.stem
    try:
        return int(stem.split("_")[-1]) + 1
    except ValueError:
        return 1


def score_pedagogical_depth(tex: str) -> float:
    score = 0.65
    if tex.count(r"\section{") >= 3:
        score += 0.10
    if tex.count(r"\subsection{本章小结}") >= 1:
        score += 0.10
    if r"\section{总结与延伸}" in tex:
        score += 0.10
    if len(tex) > 5000:
        score += 0.05
    return min(score, 1.0)


def score_derivation_fidelity(tex: str, coverage_rows: list[dict[str, Any]]) -> float:
    formula_units = infer_formula_units(coverage_rows)
    if not formula_units:
        return 1.0
    has_display_math = has_math_markup(tex)
    symbol_explanation = has_symbol_explanation(tex)
    score = 0.55
    if has_display_math:
        score += 0.3
    if symbol_explanation:
        score += 0.15
    return min(score, 1.0)


def score_code_fidelity(tex: str, coverage_rows: list[dict[str, Any]]) -> float:
    code_units = infer_code_units(coverage_rows)
    if not code_units:
        return 1.0
    has_listing = r"\begin{lstlisting}" in tex
    has_code_explanation = ("代码" in tex) or ("实现" in tex) or ("伪代码" in tex)
    score = 0.4
    if has_listing:
        score += 0.3
    if has_code_explanation:
        score += 0.3
    return min(score, 1.0)


def score_figure_usefulness(tex_assets: list[str], figure_manifest: list[dict[str, Any]]) -> float:
    if not tex_assets:
        return 0.0
    covered_assets = {row.get("asset_path") for row in figure_manifest if isinstance(row, dict)}
    captioned = sum(1 for row in figure_manifest if isinstance(row, dict) and row.get("caption"))
    asset_coverage = sum(1 for asset in tex_assets if asset in covered_assets) / max(1, len(tex_assets))
    caption_ratio = captioned / max(1, len(tex_assets))
    return min(1.0, 0.5 * asset_coverage + 0.5 * caption_ratio)


def score_coherence(tex: str) -> float:
    score = 0.6
    if tex.count(r"\subsection{本章小结}") >= 1:
        score += 0.1
    if r"\section{总结与延伸}" in tex:
        score += 0.15
    if tex.count(r"\section{") >= 3:
        score += 0.15
    return min(score, 1.0)


def score_hallucination_control(tex: str, coverage_rows: list[dict[str, Any]], omission_rows: list[dict[str, Any]]) -> float:
    score = 0.7
    if "[cite]" not in tex and "TODO" not in tex:
        score += 0.1
    if all(row.get("status") != "unclassified" for row in coverage_rows):
        score += 0.1
    if all(row.get("user_visible_note") or row.get("reason") for row in omission_rows):
        score += 0.1
    return min(score, 1.0)


def build_blocking_issues(
    tex: str,
    coverage_rows: list[dict[str, Any]],
    omission_rows: list[dict[str, Any]],
    figure_manifest: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    omission_by_unit = {row.get("unit_id"): row for row in omission_rows if isinstance(row, dict)}
    for row in coverage_rows:
        if row.get("required") and row.get("status") == "unclassified":
            issues.append(
                {
                    "issue_id": f"issue_{len(issues)+1:02d}",
                    "type": "coverage_gap",
                    "unit_id": row.get("unit_id"),
                    "problem": "required coverage unit remains unclassified",
                    "required_fix": "map the unit to a section or log it explicitly as omitted/duplicate/partial",
                }
            )
        if row.get("status") == "partial" and not str(row.get("notes", "")).strip():
            issues.append(
                {
                    "issue_id": f"issue_{len(issues)+1:02d}",
                    "type": "partial_without_note",
                    "unit_id": row.get("unit_id"),
                    "problem": "partial coverage has no explanatory notes",
                    "required_fix": "add concrete notes explaining what remains uncovered",
                }
            )
        if row.get("status") == "omitted" and row.get("unit_id") not in omission_by_unit:
            issues.append(
                {
                    "issue_id": f"issue_{len(issues)+1:02d}",
                    "type": "omission_without_log",
                    "unit_id": row.get("unit_id"),
                    "problem": "omitted unit has no omission_log entry",
                    "required_fix": "add omission log entry with reason and user-visible note",
                }
            )
    if "[cite]" in tex:
        issues.append(
            {
                "issue_id": f"issue_{len(issues)+1:02d}",
                "type": "hallucination_risk",
                "unit_id": None,
                "problem": "note still contains unresolved citation placeholders",
                "required_fix": "remove placeholders and replace with grounded text",
            }
        )
    for row in figure_manifest:
        if isinstance(row, dict) and "frames/" in str(row.get("asset_path", "")) and not row.get("time_provenance"):
            issues.append(
                {
                    "issue_id": f"issue_{len(issues)+1:02d}",
                    "type": "figure_provenance_missing",
                    "unit_id": row.get("figure_id"),
                    "problem": "frame-derived figure lacks time provenance",
                    "required_fix": "record concrete time interval for the frame-derived figure",
                }
            )
    return issues


def evaluate_lecture(lecture_dir: Path) -> dict[str, Any]:
    tex_path = sorted(lecture_dir.glob("lecture_*_note.tex"))[0]
    tex = tex_path.read_text()
    coverage_rows = load_jsonl(lecture_dir / "coverage_units.jsonl")
    omission_rows = load_jsonl(lecture_dir / "omission_log.jsonl")
    figure_manifest = load_json(lecture_dir / "figure_manifest.json")
    if not isinstance(figure_manifest, list):
        figure_manifest = []
    tex_assets = parse_tex_figure_assets(tex_path)

    required_rows = [row for row in coverage_rows if row.get("required")]
    classified_required = [
        row
        for row in required_rows
        if row.get("status") in {"covered", "partial", "duplicate", "omitted"}
    ]
    coverage_score = len(classified_required) / max(1, len(required_rows))
    scores = {
        "coverage_completeness": round(coverage_score, 3),
        "pedagogical_depth": round(score_pedagogical_depth(tex), 3),
        "derivation_fidelity": round(score_derivation_fidelity(tex, coverage_rows), 3),
        "code_fidelity": round(score_code_fidelity(tex, coverage_rows), 3),
        "figure_usefulness": round(score_figure_usefulness(tex_assets, figure_manifest), 3),
        "coherence": round(score_coherence(tex), 3),
        "hallucination_control": round(score_hallucination_control(tex, coverage_rows, omission_rows), 3),
    }
    blocking_issues = build_blocking_issues(tex, coverage_rows, omission_rows, figure_manifest)

    failed_thresholds = [
        name
        for name, threshold in DEFAULT_THRESHOLDS.items()
        if scores.get(name, 0.0) < threshold
    ]
    overall = "pass" if not blocking_issues and not failed_thresholds else "fail"
    warnings = []
    if not tex_assets:
        warnings.append("No instructional figure assets were detected in the note tex.")
    if not infer_formula_units(coverage_rows):
        warnings.append("No formula-oriented coverage units inferred for this lecture.")
    if not infer_code_units(coverage_rows):
        warnings.append("No code-oriented coverage units inferred for this lecture.")

    report = {
        "pass": latest_eval_pass(lecture_dir),
        "target": lecture_dir.name,
        "overall": overall,
        "scores": scores,
        "blocking_issues": blocking_issues,
        "warnings": warnings,
        "repair_required": overall != "pass",
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = (REPO_ROOT / args.run_root).resolve()
    lecture_dirs = sorted(path for path in (run_root / "lectures").iterdir() if path.is_dir() and path.name[:2].isdigit())
    for lecture_dir in lecture_dirs:
        report = evaluate_lecture(lecture_dir)
        report_dir = lecture_dir / "eval_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"pass_{report['pass']:02d}.json"
        write_json(report_path, report)
        print(f"{lecture_dir.name}: {report['overall']}")


if __name__ == "__main__":
    main()
