#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LECTURES_DIR = ROOT / "lectures"


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


def main() -> None:
    lecture_dirs = sorted(p for p in LECTURES_DIR.iterdir() if p.is_dir() and p.name[:2].isdigit())
    for lecture_dir in lecture_dirs:
        tex_files = sorted(lecture_dir.glob("lecture_*_note.tex"))
        if not tex_files:
            continue
        compile_tex(tex_files[0])
        print(lecture_dir.name)


if __name__ == "__main__":
    main()
