#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LECTURES_DIR = ROOT / "lectures"


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(payload + ("\n" if payload else ""))


def infer_source_type(source_id: str) -> str:
    if "+" in source_id:
        return "multi_source"
    if "official" in source_id or "slides" in source_id:
        return "slide_evidence"
    if "transcript" in source_id or "subtitle" in source_id:
        return "subtitle_evidence"
    if source_id == "lecture_meta":
        return "lecture_metadata"
    return "source_unit"


def partial_note(existing_notes: str, omission_rows: list[dict]) -> str:
    if existing_notes.strip():
        return existing_notes
    for row in omission_rows:
        reason = str(row.get("reason", "")).lower()
        visible = str(row.get("user_visible_note", ""))
        if "partial" in reason or "尚未" in visible or "只在" in visible:
            return visible or str(row.get("reason", ""))
    return "Partial coverage requires follow-up expansion before final acceptance."


def normalize_lecture(lecture_dir: Path) -> bool:
    coverage_path = lecture_dir / "coverage_units.jsonl"
    omission_path = lecture_dir / "omission_log.jsonl"
    if not coverage_path.exists():
        return False

    coverage_rows = load_jsonl(coverage_path)
    omission_rows = load_jsonl(omission_path) if omission_path.exists() else []
    changed = False
    normalized: list[dict] = []
    for row in coverage_rows:
        row = dict(row)
        source_id = str(row.get("source_id", ""))
        status = str(row.get("status", "covered"))
        notes = str(row.get("notes", ""))
        normalized_row = {
            "unit_id": row.get("unit_id"),
            "source_type": row.get("source_type") or infer_source_type(source_id),
            "source_id": source_id,
            "loc": row.get("loc"),
            "kind": row.get("kind") or ([row["unit_type"]] if row.get("unit_type") else ["coverage_unit"]),
            "summary": row.get("summary", ""),
            "required": row.get("required", True),
            "status": status,
            "mapped_section": row.get("mapped_section"),
            "figure_ids": row.get("figure_ids") or [],
            "notes": partial_note(notes, omission_rows) if status == "partial" else notes,
        }
        for key in ["unit_type"]:
            if key in row:
                normalized_row[key] = row[key]
        if normalized_row != row:
            changed = True
        normalized.append(normalized_row)

    if changed:
        write_jsonl(coverage_path, normalized)
    return changed


def main() -> None:
    changed_count = 0
    for lecture_dir in sorted(path for path in LECTURES_DIR.iterdir() if path.is_dir() and path.name[:2].isdigit()):
        if normalize_lecture(lecture_dir):
            changed_count += 1
            print(f"normalized {lecture_dir.name}")
    print(f"changed={changed_count}")


if __name__ == "__main__":
    main()
