#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LECTURES_DIR = ROOT / "lectures"
REGRESSION_LECTURES = [
    "01_transformer",
    "04_llm_training",
    "07_agentic_llms",
]
ALLOWED_STATUSES = {"covered", "partial", "duplicate", "omitted", "unclassified"}


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


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


def resolve_lecture_dirs(tokens: list[str] | None, regression_only: bool) -> list[Path]:
    lecture_dirs = sorted(path for path in LECTURES_DIR.iterdir() if path.is_dir() and path.name[:2].isdigit())
    if regression_only:
        wanted = set(REGRESSION_LECTURES)
        return [path for path in lecture_dirs if path.name in wanted]
    if not tokens:
        return lecture_dirs

    resolved: list[Path] = []
    for token in tokens:
        matched = None
        for path in lecture_dirs:
            if path.name == token or path.name.startswith(token + "_") or path.name.startswith(token):
                matched = path
                break
        if matched is None:
            raise SystemExit(f"unknown lecture selector: {token}")
        resolved.append(matched)
    return resolved


def parse_tex_figure_assets(tex_path: Path) -> set[str]:
    text = tex_path.read_text()
    assets = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text))
    return {
        asset
        for asset in assets
        if asset
        and not asset.startswith("\\")
        and asset != "cover.jpg"
    }


def require(condition: bool, errors: list[str], message: str) -> None:
    if not condition:
        errors.append(message)


def validate_lecture(lecture_dir: Path, compile_pdf: bool) -> list[str]:
    errors: list[str] = []
    required_paths = {
        "source_manifest.json": True,
        "transcript.jsonl": True,
        "slides.jsonl": True,
        "coverage_units.jsonl": True,
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

    source_manifest = load_json(lecture_dir / "source_manifest.json")
    transcript_units = load_jsonl(lecture_dir / "transcript.jsonl")
    slide_units = load_jsonl(lecture_dir / "slides.jsonl")
    coverage_units = load_jsonl(lecture_dir / "coverage_units.jsonl")
    omission_log = load_jsonl(lecture_dir / "omission_log.jsonl")
    figure_manifest = load_json(lecture_dir / "figure_manifest.json")
    if not isinstance(source_manifest, dict):
        errors.append(f"{lecture_dir.name}: source_manifest.json must be a JSON object")
        return errors
    if not isinstance(figure_manifest, list):
        errors.append(f"{lecture_dir.name}: figure_manifest.json must be a JSON array")
        return errors

    source_ids = {
        source.get("source_id")
        for source in source_manifest.get("sources", [])
        if isinstance(source, dict)
    }
    require("transcript_jsonl" in source_ids, errors, f"{lecture_dir.name}: source_manifest.json missing transcript_jsonl source")
    require("slides_jsonl" in source_ids, errors, f"{lecture_dir.name}: source_manifest.json missing slides_jsonl source")
    if source_manifest.get("course_mode") or len(transcript_units) > 300:
        segments_path = lecture_dir / "segments.jsonl"
        require(segments_path.exists(), errors, f"{lecture_dir.name}: missing segments.jsonl")
        require(segments_path.stat().st_size > 0, errors, f"{lecture_dir.name}: empty segments.jsonl")
        require("segments_jsonl" in source_ids, errors, f"{lecture_dir.name}: source_manifest.json missing segments_jsonl source")

    for row in transcript_units:
        for key in ["unit_id", "source_type", "source_id", "loc", "text", "required"]:
            require(key in row, errors, f"{lecture_dir.name}: transcript.jsonl row missing {key}")
    for row in slide_units:
        for key in ["unit_id", "source_type", "source_id", "loc", "text", "asset_path", "required"]:
            require(key in row, errors, f"{lecture_dir.name}: slides.jsonl row missing {key}")

    omission_by_unit = {
        row.get("unit_id"): row
        for row in omission_log
        if isinstance(row, dict) and row.get("unit_id")
    }
    required_coverage_fields = [
        "unit_id",
        "source_type",
        "source_id",
        "loc",
        "kind",
        "summary",
        "required",
        "status",
        "mapped_section",
        "figure_ids",
        "notes",
    ]
    for row in coverage_units:
        for key in required_coverage_fields:
            require(key in row, errors, f"{lecture_dir.name}: coverage row {row.get('unit_id', '<unknown>')} missing {key}")
        status = row.get("status")
        require(status in ALLOWED_STATUSES, errors, f"{lecture_dir.name}: coverage row {row.get('unit_id')} has invalid status {status!r}")
        if row.get("required") is True:
            require(status != "unclassified", errors, f"{lecture_dir.name}: required unit {row.get('unit_id')} remains unclassified")
        if status in {"covered", "partial"}:
            require(bool(row.get("mapped_section")), errors, f"{lecture_dir.name}: {status} unit {row.get('unit_id')} missing mapped_section")
        if status == "partial":
            require(bool(str(row.get("notes", "")).strip()), errors, f"{lecture_dir.name}: partial unit {row.get('unit_id')} missing explanatory notes")
        if status in {"duplicate", "omitted"}:
            require(row.get("unit_id") in omission_by_unit, errors, f"{lecture_dir.name}: {status} unit {row.get('unit_id')} missing omission_log entry")

    manifest_assets = {
        row.get("asset_path")
        for row in figure_manifest
        if isinstance(row, dict) and row.get("asset_path")
    }
    tex_assets = parse_tex_figure_assets(tex_path)
    missing_assets = sorted(asset for asset in tex_assets if asset not in manifest_assets)
    if missing_assets:
        errors.append(f"{lecture_dir.name}: figure_manifest.json missing asset entries for {', '.join(missing_assets)}")

    for row in figure_manifest:
        if not isinstance(row, dict):
            errors.append(f"{lecture_dir.name}: figure_manifest.json contains non-object entry")
            continue
        if "frames/" in str(row.get("asset_path", "")):
            require(bool(row.get("time_provenance")), errors, f"{lecture_dir.name}: frame asset {row.get('asset_path')} missing time_provenance")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("lectures", nargs="*", help="lecture ids or lecture directory names")
    parser.add_argument("--compile", action="store_true", help="compile lecture_XX_note.tex before validation")
    parser.add_argument("--regression-set", action="store_true", help="validate the fixed 3-lecture regression set")
    args = parser.parse_args()

    lecture_dirs = resolve_lecture_dirs(args.lectures, args.regression_set)
    all_errors: list[str] = []
    for lecture_dir in lecture_dirs:
        errors = validate_lecture(lecture_dir, args.compile)
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
