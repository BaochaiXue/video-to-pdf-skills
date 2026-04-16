#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.video_note_harness.common import (
    DEFAULT_THRESHOLDS,
    ALLOWED_COVERAGE_STATUSES,
    build_segments,
    build_slide_units,
    build_transcript_units,
    classify_figure_provenance,
    detect_kind_hints,
    extract_slide_pages,
    infer_code_units,
    infer_formula_units,
    load_json,
    load_jsonl,
    parse_tex_figure_assets,
    write_json,
    write_jsonl,
)


def lecture_dirs(run_root: Path) -> list[Path]:
    return sorted(path for path in (run_root / "lectures").iterdir() if path.is_dir() and path.name[:2].isdigit())


def ensure_structured_evidence(lecture_dir: Path) -> None:
    meta = load_json(lecture_dir / "meta.json")

    transcript_jsonl = lecture_dir / "transcript.jsonl"
    if not transcript_jsonl.exists() and (lecture_dir / "subtitle.srt").exists():
        rows = build_transcript_units(lecture_dir / "subtitle.srt")
        write_jsonl(transcript_jsonl, rows)

    slides_jsonl = lecture_dir / "slides.jsonl"
    if not slides_jsonl.exists() and (lecture_dir / "slides.pdf").exists():
        pages = extract_slide_pages(lecture_dir / "slides.pdf")
        write_jsonl(slides_jsonl, build_slide_units(pages, lecture_dir))

    segments_jsonl = lecture_dir / "segments.jsonl"
    if not segments_jsonl.exists():
        topic_hints = meta.get("topics") or [meta.get("title_short") or meta.get("title") or lecture_dir.name]
        transcript_rows = load_jsonl(transcript_jsonl) if transcript_jsonl.exists() else []
        slide_rows = load_jsonl(slides_jsonl) if slides_jsonl.exists() else []
        write_jsonl(segments_jsonl, build_segments(topic_hints, transcript_rows, slide_rows))

    if isinstance(meta, dict):
        meta["course_mode"] = bool(meta.get("course_mode", True))
        meta["segmentation_required"] = bool(meta.get("segmentation_required", segments_jsonl.exists()))
        if transcript_jsonl.exists():
            meta["transcript_jsonl"] = str(transcript_jsonl.relative_to(REPO_ROOT))
        if slides_jsonl.exists():
            meta["slides_jsonl"] = str(slides_jsonl.relative_to(REPO_ROOT))
        if segments_jsonl.exists():
            meta["segments_jsonl"] = str(segments_jsonl.relative_to(REPO_ROOT))
        write_json(lecture_dir / "meta.json", meta)


def build_lecture_plan(lecture_dir: Path) -> dict[str, Any]:
    meta = load_json(lecture_dir / "meta.json")
    source_manifest = load_json(lecture_dir / "source_manifest.json") if (lecture_dir / "source_manifest.json").exists() else {"sources": []}
    if not isinstance(source_manifest, dict):
        source_manifest = {"sources": source_manifest if isinstance(source_manifest, list) else []}
    coverage_rows = load_jsonl(lecture_dir / "coverage_units.jsonl") if (lecture_dir / "coverage_units.jsonl").exists() else []
    segments = load_jsonl(lecture_dir / "segments.jsonl") if (lecture_dir / "segments.jsonl").exists() else []
    source_inventory = []
    for source in source_manifest.get("sources", []):
        if not isinstance(source, dict):
            continue
        source_inventory.append(
            {
                "source_id": source.get("source_id"),
                "source_type": source.get("source_type"),
                "required_for_coverage": source.get("required_for_coverage", False),
                "status": source.get("status", "available"),
            }
        )

    return {
        "lecture_id": f"{meta.get('playlist_index', lecture_dir.name[:2]):02d}",
        "title": meta.get("title", lecture_dir.name),
        "course_mode": bool(meta.get("course_mode", True)),
        "source_inventory": source_inventory,
        "segment_ids": [segment.get("segment_id") for segment in segments],
        "must_cover_kinds": detect_kind_hints(coverage_rows),
        "must_emit_artifacts": [
            "lecture_XX_note.tex",
            "lecture_XX_note.pdf",
            "coverage_units.jsonl",
            "omission_log.jsonl",
            "figure_manifest.json",
            "figure_plan.json",
            "eval_reports/pass_01.json",
            "repair_log.jsonl",
        ],
        "evaluator_thresholds": DEFAULT_THRESHOLDS,
    }


def build_figure_plan(lecture_dir: Path) -> list[dict[str, Any]]:
    manifest_path = lecture_dir / "figure_manifest.json"
    tex_files = sorted(lecture_dir.glob("lecture_*_note.tex"))
    existing = load_json(manifest_path) if manifest_path.exists() else []
    if not isinstance(existing, list):
        existing = []
    manifest_by_asset = {row.get("asset_path"): row for row in existing if isinstance(row, dict)}
    referenced_assets = []
    if tex_files:
        referenced_assets = parse_tex_figure_assets(tex_files[0])
    else:
        referenced_assets = [row.get("asset_path") for row in existing if isinstance(row, dict) and row.get("asset_path")]

    plan: list[dict[str, Any]] = []
    for idx, asset_path in enumerate(dict.fromkeys(referenced_assets), start=1):
        row = manifest_by_asset.get(asset_path, {})
        plan.append(
            {
                "figure_id": row.get("figure_id") or f"figure_{idx:02d}",
                "source_unit_ids": row.get("source_unit_ids") or [],
                "asset_candidates": [asset_path],
                "selection_reason": row.get("selection_reason") or "Promoted from delivered figure manifest during harness migration.",
                "required": True,
                "provenance_type": row.get("provenance_type") or classify_figure_provenance(asset_path),
                "time_provenance": row.get("time_provenance"),
            }
        )
    return plan


def build_segment_contracts(lecture_dir: Path) -> None:
    segments = load_jsonl(lecture_dir / "segments.jsonl")
    coverage_rows = load_jsonl(lecture_dir / "coverage_units.jsonl") if (lecture_dir / "coverage_units.jsonl").exists() else []
    figure_plan = load_json(lecture_dir / "figure_plan.json") if (lecture_dir / "figure_plan.json").exists() else []
    formula_units = set(infer_formula_units(coverage_rows))
    code_units = set(infer_code_units(coverage_rows))
    contracts_dir = lecture_dir / "contracts"
    contracts_dir.mkdir(parents=True, exist_ok=True)

    for segment in segments:
        segment_id = segment.get("segment_id")
        source_unit_ids = segment.get("source_unit_ids") or []
        required_figures = [
            figure.get("figure_id")
            for figure in figure_plan
            if isinstance(figure, dict)
            and set(figure.get("source_unit_ids") or []).intersection(source_unit_ids)
        ]
        formula_hits = [unit_id for unit_id in source_unit_ids if unit_id in formula_units]
        code_hits = [unit_id for unit_id in source_unit_ids if unit_id in code_units]

        lines = [
            f"# {segment_id} Contract",
            "",
            "Source range:",
            f"- transcript: {segment.get('start') or 'unknown'} -- {segment.get('end') or 'unknown'}",
            f"- segment hint: {segment.get('target_section_hint') or 'Segment'}",
            "",
            "Must cover unit ids:",
        ]
        if source_unit_ids:
            lines.extend([f"- {unit_id}" for unit_id in source_unit_ids])
        else:
            lines.append("- none explicitly assigned")
        lines.extend(
            [
                "",
                "Required formulas:",
            ]
        )
        lines.extend([f"- {unit_id}" for unit_id in formula_hits] or ["- none explicitly inferred"])
        lines.extend(
            [
                "",
                "Required code units:",
            ]
        )
        lines.extend([f"- {unit_id}" for unit_id in code_hits] or ["- none explicitly inferred"])
        lines.extend(
            [
                "",
                "Required figures:",
            ]
        )
        lines.extend([f"- {figure_id}" for figure_id in required_figures] or ["- none explicitly inferred"])
        lines.extend(
            [
                "",
                "Done checks:",
                "- no required unit remains unclassified",
                "- formulas are expanded, not compressed into takeaways",
                "- figures, if used, are provenance-backed",
                "- the segment output maps into a concrete section rather than only a summary subsection",
                "",
            ]
        )
        (contracts_dir / f"{segment_id}_contract.md").write_text("\n".join(lines) + "\n")


def ensure_repair_log(lecture_dir: Path) -> None:
    repair_log = lecture_dir / "repair_log.jsonl"
    if not repair_log.exists():
        repair_log.write_text("")


def infer_source_type(source_id: str) -> str:
    lowered = source_id.lower()
    if "official" in lowered or "slides" in lowered:
        return "slide_evidence"
    if "subtitle" in lowered or "transcript" in lowered:
        return "subtitle_evidence"
    if source_id == "lecture_meta":
        return "lecture_metadata"
    return "source_unit"


def normalize_coverage_rows(lecture_dir: Path) -> None:
    path = lecture_dir / "coverage_units.jsonl"
    if not path.exists():
        return
    rows = load_jsonl(path)
    normalized: list[dict[str, Any]] = []
    changed = False
    for row in rows:
        row = dict(row)
        status = str(row.get("status", "covered"))
        if status == "included":
            status = "covered"
        if status not in ALLOWED_COVERAGE_STATUSES:
            status = "covered"
        kind = row.get("kind")
        if isinstance(kind, str):
            kind = [kind]
        if not kind:
            unit_type = row.get("unit_type")
            kind = [unit_type] if unit_type else ["coverage_unit"]
        normalized_row = {
            "unit_id": row.get("unit_id"),
            "source_type": row.get("source_type") or infer_source_type(str(row.get("source_id", ""))),
            "source_id": row.get("source_id"),
            "loc": row.get("loc"),
            "kind": kind,
            "summary": row.get("summary", ""),
            "required": row.get("required", True),
            "status": status,
            "mapped_section": row.get("mapped_section"),
            "figure_ids": row.get("figure_ids") or [],
            "notes": row.get("notes", ""),
        }
        if row.get("unit_type") is not None:
            normalized_row["unit_type"] = row.get("unit_type")
        if normalized_row != row:
            changed = True
        normalized.append(normalized_row)
    if changed:
        write_jsonl(path, normalized)


def enrich_source_manifest(lecture_dir: Path) -> None:
    manifest = load_json(lecture_dir / "source_manifest.json") if (lecture_dir / "source_manifest.json").exists() else {}
    if not isinstance(manifest, dict):
        manifest = {"sources": manifest if isinstance(manifest, list) else []}
    sources = manifest.setdefault("sources", [])
    existing_ids = {row.get("source_id") for row in sources if isinstance(row, dict)}
    additions = [
        ("transcript_jsonl", "structured_transcript_evidence", lecture_dir / "transcript.jsonl", True, "Canonical subtitle-aligned evidence."),
        ("slides_jsonl", "structured_slide_evidence", lecture_dir / "slides.jsonl", True, "Canonical slide-page evidence."),
        ("segments_jsonl", "segment_plan", lecture_dir / "segments.jsonl", True, "Canonical segment plan for harness-managed note writing."),
        ("lecture_plan_json", "harness_lecture_plan", lecture_dir / "lecture_plan.json", True, "Harness lecture plan artifact."),
        ("figure_plan_json", "harness_figure_plan", lecture_dir / "figure_plan.json", False, "Harness figure planning artifact."),
        ("repair_log_jsonl", "harness_repair_log", lecture_dir / "repair_log.jsonl", False, "Repair actions recorded after evaluator feedback."),
        ("eval_report_pass_01", "harness_eval_report", lecture_dir / "eval_reports" / "pass_01.json", False, "First evaluator pass report."),
    ]
    for source_id, source_type, path, required, notes in additions:
        if source_id in existing_ids:
            continue
        sources.append(
            {
                "source_id": source_id,
                "source_type": source_type,
                "origin_url": None,
                "local_path": str(path.relative_to(REPO_ROOT)) if path.exists() else None,
                "required_for_coverage": required,
                "status": "available" if path.exists() else "missing",
                "notes": notes,
            }
        )
    write_json(lecture_dir / "source_manifest.json", manifest)


def bootstrap_lecture(lecture_dir: Path) -> None:
    ensure_structured_evidence(lecture_dir)
    normalize_coverage_rows(lecture_dir)
    ensure_repair_log(lecture_dir)
    (lecture_dir / "eval_reports").mkdir(parents=True, exist_ok=True)
    enrich_source_manifest(lecture_dir)
    write_json(lecture_dir / "lecture_plan.json", build_lecture_plan(lecture_dir))
    write_json(lecture_dir / "figure_plan.json", build_figure_plan(lecture_dir))
    build_segment_contracts(lecture_dir)
    enrich_source_manifest(lecture_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, help="run root such as runs/cme295_fall2025")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = (REPO_ROOT / run_root).resolve()
    for lecture_dir in lecture_dirs(run_root):
        bootstrap_lecture(lecture_dir)
        print(f"bootstrapped {lecture_dir.name}")


if __name__ == "__main__":
    main()
