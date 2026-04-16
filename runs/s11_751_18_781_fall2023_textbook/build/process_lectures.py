#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RUN_ROOT.parents[1]
LECTURES_DIR = RUN_ROOT / "lectures"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.video_note_harness.evaluate_note import evaluate_lecture


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


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def evaluate_one(lecture_dir: Path) -> Path:
    report = evaluate_lecture(lecture_dir)
    report_dir = lecture_dir / "eval_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"pass_{report['pass']:02d}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return path


def compile_note(lecture_dir: Path, tex_path: Path) -> None:
    for _ in range(2):
        run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name], cwd=lecture_dir)


def process_lecture(lecture_dir: Path) -> None:
    run(["python3", str(RUN_ROOT / "build" / "draft_notes_from_coverage.py"), lecture_dir.name])
    run(["python3", str(RUN_ROOT / "build" / "sync_figure_manifest.py"), lecture_dir.name])
    tex_files = sorted(lecture_dir.glob("lecture_*_note.tex"))
    if not tex_files:
        raise SystemExit(f"{lecture_dir.name}: lecture_XX_note.tex not generated")
    compile_note(lecture_dir, tex_files[0])
    evaluate_one(lecture_dir)
    selector = lecture_dir.name[:2]
    run(["python3", str(RUN_ROOT / "build" / "validate_lecture.py"), selector, "--compile"])
    print(f"ok {lecture_dir.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("lectures", nargs="*")
    args = parser.parse_args()
    for lecture_dir in lecture_dirs(args.lectures):
        process_lecture(lecture_dir)


if __name__ == "__main__":
    main()
