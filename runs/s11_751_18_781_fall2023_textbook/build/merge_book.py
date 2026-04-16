#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
LECTURES_DIR = RUN_ROOT / "lectures"
BOOK_DIR = RUN_ROOT / "book"
CHAPTERS_DIR = BOOK_DIR / "chapters"


def main() -> None:
    CHAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    lecture_entries = []
    include_lines = []
    for lecture_dir in sorted(path for path in LECTURES_DIR.iterdir() if path.is_dir()):
        repaired = lecture_dir / "lecture_repaired.tex"
        primary = lecture_dir / "lecture.tex"
        source = repaired if repaired.exists() else primary
        if not source.exists():
            continue
        chapter_name = f"{lecture_dir.name}.tex"
        target = CHAPTERS_DIR / chapter_name
        target.write_text(source.read_text())
        lecture_entries.append(
            {
                "lecture_slug": lecture_dir.name,
                "chapter_path": str(target.relative_to(RUN_ROOT)),
                "source_tex": str(source.relative_to(RUN_ROOT)),
                "source_manifest": str((lecture_dir / "source_manifest.json").relative_to(RUN_ROOT)) if (lecture_dir / "source_manifest.json").exists() else None,
            }
        )
        include_lines.append(f"\\input{{chapters/{chapter_name}}}")

    frontmatter = [
        r"\documentclass[a4paper]{article}",
        r"\usepackage[fontset=fandol]{ctex}",
        r"\usepackage[margin=2.5cm]{geometry}",
        r"\usepackage{hyperref}",
        r"\title{CS294/194-280: Advanced Large Language Model Agents}",
        r"\author{Codex Harness-Managed Textbook Build}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
        r"\tableofcontents",
        r"\newpage",
    ]
    ending = [r"\end{document}"]
    (BOOK_DIR / "main.tex").write_text("\n".join(frontmatter + include_lines + ending) + "\n")
    (BOOK_DIR / "textbook_source_manifest.json").write_text(json.dumps({"lectures": lecture_entries}, indent=2, ensure_ascii=False) + "\n")
    print(BOOK_DIR / "main.tex")


if __name__ == "__main__":
    main()
