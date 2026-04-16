#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
LECTURES_DIR = RUN_ROOT / "lectures"


FIGURE_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
CAPTION_RE = re.compile(r"\\caption\{([^}]*)\}")


def lecture_dirs(selectors: list[str] | None) -> list[Path]:
    dirs = sorted(path for path in LECTURES_DIR.iterdir() if path.is_dir() and path.name[:2].isdigit())
    if not selectors:
        return dirs
    resolved = []
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


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else []


def extract_entries(tex_path: Path) -> list[dict]:
    lines = tex_path.read_text().splitlines()
    entries = []
    pending_asset = None
    for raw in lines:
        asset_match = FIGURE_RE.search(raw)
        if asset_match:
            pending_asset = asset_match.group(1)
            continue
        caption_match = CAPTION_RE.search(raw)
        if caption_match and pending_asset:
            entries.append({"asset_path": pending_asset, "caption": caption_match.group(1)})
            pending_asset = None
    return entries


def process_lecture(lecture_dir: Path) -> None:
    tex_files = sorted(lecture_dir.glob("lecture_*_note.tex"))
    if not tex_files:
        return
    tex_path = tex_files[0]
    figure_plan = load_json(lecture_dir / "figure_plan.json")
    existing = load_json(lecture_dir / "figure_manifest.json")
    existing_by_asset = {
        row.get("asset_path"): row for row in existing if isinstance(row, dict) and row.get("asset_path")
    } if isinstance(existing, list) else {}
    plan_by_asset = {}
    if isinstance(figure_plan, list):
        for row in figure_plan:
            if not isinstance(row, dict):
                continue
            for asset in row.get("asset_candidates") or []:
                plan_by_asset[asset] = row

    rows = []
    for idx, entry in enumerate(extract_entries(tex_path), start=1):
        asset = entry["asset_path"]
        base = existing_by_asset.get(asset, {})
        plan = plan_by_asset.get(asset, {})
        rows.append(
            {
                "figure_id": base.get("figure_id") or plan.get("figure_id") or f"figure_{idx:02d}",
                "source_id": base.get("source_id") or ("generated_summary_figure" if "figures/" in asset else "slide_or_external_asset"),
                "loc": base.get("loc"),
                "asset_path": asset,
                "caption": entry["caption"],
                "crop": bool(base.get("crop", False)),
                "used_in_section": base.get("used_in_section") or plan.get("selection_reason"),
                "time_provenance": base.get("time_provenance") or plan.get("time_provenance"),
            }
        )
    (lecture_dir / "figure_manifest.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    print(lecture_dir.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("lectures", nargs="*")
    args = parser.parse_args()
    for lecture_dir in lecture_dirs(args.lectures):
        process_lecture(lecture_dir)


if __name__ == "__main__":
    main()
