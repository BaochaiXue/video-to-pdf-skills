#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LECTURES_DIR = ROOT / "lectures"
VALIDATOR = ROOT / "build" / "validate_youtube_note.py"


def lecture_dirs(selected: set[str] | None = None) -> list[Path]:
    lecture_paths = sorted(p for p in LECTURES_DIR.iterdir() if p.is_dir() and p.name[:2].isdigit())
    if selected is None:
        return lecture_paths
    return [path for path in lecture_paths if path.name in selected or path.name.split("_", 1)[0] in selected]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lectures", nargs="*", help="lecture ids or lecture directory names")
    args = parser.parse_args()

    selected = set(args.lectures) if args.lectures else None
    for lecture_dir in lecture_dirs(selected):
        subprocess.run(
            ["python3", str(VALIDATOR), "--compile", lecture_dir.name],
            cwd=ROOT,
            check=True,
        )
        print(lecture_dir.name)


if __name__ == "__main__":
    main()
