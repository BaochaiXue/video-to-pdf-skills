#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
LECTURES_DIR = RUN_ROOT / "lectures"


def lecture_sort_key(path: Path) -> tuple[int, str]:
    match = re.match(r"^(\d+)_", path.name)
    if match:
        return int(match.group(1)), path.name
    return 10**9, path.name


def lecture_dirs(selectors: list[str] | None) -> list[Path]:
    dirs = sorted((path for path in LECTURES_DIR.iterdir() if path.is_dir() and re.match(r"^\d+_", path.name)), key=lecture_sort_key)
    if not selectors:
        return dirs
    selected: list[Path] = []
    for token in selectors:
        match = None
        for path in dirs:
            if path.name == token or path.name.startswith(token) or path.name.split("_", 1)[0] == token:
                match = path
                break
        if match is None:
            raise SystemExit(f"unknown lecture selector: {token}")
        selected.append(match)
    return selected


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


def select_tex(lecture_dir: Path) -> Path | None:
    preferred = lecture_dir / "lecture_XX_note.tex"
    if preferred.exists():
        return preferred
    candidates = sorted(lecture_dir.glob("lecture_*_note.tex"))
    return candidates[0] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("lectures", nargs="*")
    args = parser.parse_args()

    for lecture_dir in lecture_dirs(args.lectures):
        tex_path = select_tex(lecture_dir)
        if tex_path is None:
            continue
        compile_tex(tex_path)
        print(lecture_dir.name)


if __name__ == "__main__":
    main()
