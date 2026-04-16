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


def lecture_dirs() -> list[Path]:
    return sorted(p for p in LECTURES_DIR.iterdir() if p.is_dir() and p.name.startswith("lec"))


def select_lecture_tex(lecture_dir: Path) -> Path | None:
    for candidate in [lecture_dir / "lecture_repaired.tex", lecture_dir / "lecture.tex"]:
        if candidate.exists():
            return candidate
    tex_files = sorted(lecture_dir.glob("lecture_*_note.tex"))
    return tex_files[0] if tex_files else None


def main() -> None:
    for lecture_dir in lecture_dirs():
        tex_file = select_lecture_tex(lecture_dir)
        if tex_file is None:
            continue
        compile_tex(tex_file)
        print(lecture_dir.name)


if __name__ == "__main__":
    main()
