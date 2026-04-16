#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import textwrap
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
LECTURES_DIR = RUN_ROOT / "lectures"
MAGICK = shutil.which("magick")
FONT = "/System/Library/Fonts/Supplemental/Verdana.ttf"


def load_json(path: Path):
    return json.loads(path.read_text())


def load_jsonl(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


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


def pick_units(coverage_rows: list[dict], limit: int = 8) -> list[dict]:
    rows = [row for row in coverage_rows if row.get("summary")]
    rows.sort(key=lambda row: str(row.get("unit_id")))
    return rows[:limit]


def wrap_lines(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False))


def timeline_label(row: dict) -> str:
    loc = row.get("loc")
    if isinstance(loc, dict):
        if loc.get("start") and loc.get("end"):
            return f"{loc['start']} - {loc['end']}"
        if loc.get("page"):
            return f"page {loc['page']}"
    if isinstance(loc, str):
        return loc
    return "course page"


def render_blocks(out: Path, blocks: list[dict]) -> None:
    if not MAGICK:
        raise SystemExit("ImageMagick `magick` not found")
    cmd = [MAGICK]
    for idx, block in enumerate(blocks):
        cmd.extend(
            [
                "-background",
                block["background"],
                "-fill",
                block["fill"],
                "-font",
                FONT,
                "-size",
                block["size"],
                f"caption:{block['text']}",
            ]
        )
        if idx < len(blocks) - 1:
            cmd.append("(")
            cmd.extend(
                [
                    "-size",
                    block["size"].split("x")[0] + "x18",
                    "xc:white",
                ]
            )
            cmd.append(")")
    cmd.extend(["-append", str(out)])
    subprocess.run(cmd, check=True)


def draw_coverage_map(lecture_dir: Path, meta: dict, rows: list[dict]) -> Path:
    fig_dir = lecture_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "coverage_map.png"
    blocks = [
        {
            "background": "#e8f1f8",
            "fill": "#14324a",
            "size": "1500x170",
            "text": f"{meta['title']}\nCoverage Map",
        },
        {
            "background": "white",
            "fill": "#51606f",
            "size": "1500x90",
            "text": "Coverage-first lecture roadmap generated from coverage_units.jsonl",
        },
    ]
    for row in rows:
        section = str(row.get("mapped_section") or "待映射章节")
        summary = wrap_lines(str(row.get("summary") or ""), 44)
        blocks.append(
            {
                "background": "#f1faee",
                "fill": "#1d3557",
                "size": "1500x180",
                "text": f"{row.get('unit_id', '')} | {section}\n{summary}",
            }
        )
    render_blocks(out, blocks)
    return out


def draw_timeline(lecture_dir: Path, meta: dict, rows: list[dict]) -> Path:
    fig_dir = lecture_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "timeline.png"
    blocks = [
        {
            "background": "#faedcd",
            "fill": "#3d405b",
            "size": "1500x170",
            "text": f"{meta['title']}\nSegment Timeline",
        },
        {
            "background": "white",
            "fill": "#51606f",
            "size": "1500x90",
            "text": "Time-aligned topic summary derived from transcript and coverage units",
        },
    ]
    for row in rows:
        label = timeline_label(row)
        summary = wrap_lines(str(row.get("summary") or ""), 48)
        blocks.append(
            {
                "background": "#fff7eb",
                "fill": "#2f3e46",
                "size": "1500x170",
                "text": f"{label}\n{summary}",
            }
        )
    render_blocks(out, blocks)
    return out


def process_lecture(lecture_dir: Path) -> None:
    meta = load_json(lecture_dir / "meta.json")
    coverage_rows = load_jsonl(lecture_dir / "coverage_units.jsonl")
    rows = pick_units(coverage_rows)
    draw_coverage_map(lecture_dir, meta, rows)
    draw_timeline(lecture_dir, meta, rows)
    print(lecture_dir.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("lectures", nargs="*")
    args = parser.parse_args()
    for lecture_dir in lecture_dirs(args.lectures):
        process_lecture(lecture_dir)


if __name__ == "__main__":
    main()
