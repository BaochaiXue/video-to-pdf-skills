#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build"


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


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
    manifest = json.loads((BUILD_DIR / "course_manifest.json").read_text())
    tex_path = BUILD_DIR / "cs224r_complete_notes.tex"
    title = latex_escape(manifest.get("title", "Stanford CS224R 全课程讲义"))

    lines = [
        r"\documentclass[a4paper]{article}",
        r"\usepackage[fontset=fandol]{ctex}",
        r"\usepackage[margin=2cm]{geometry}",
        r"\usepackage{pdfpages}",
        r"\usepackage{hyperref}",
        r"\usepackage{longtable}",
        f"\\title{{{title}}}",
        r"\author{Codex based on course page, slides, subtitles, and lecture videos}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
        r"\tableofcontents",
        r"\newpage",
        r"\section*{课程说明}",
        r"本讲义先逐讲生成，再按播放列表顺序合并。",
        r"\addcontentsline{toc}{section}{课程说明}",
        r"\section*{课程目录}",
        r"\addcontentsline{toc}{section}{课程目录}",
        r"\begin{longtable}{p{0.08\textwidth}p{0.67\textwidth}p{0.18\textwidth}}",
        r"\textbf{讲次} & \textbf{主题} & \textbf{日期}\\",
        r"\hline",
    ]

    for lecture in manifest["lectures"]:
        lines.append(f"{lecture['lecture_id']} & {latex_escape(lecture['title'])} & {lecture['date']}\\\\")
    lines.extend([r"\end{longtable}", ""])

    for lecture in manifest["lectures"]:
        pdf_rel = lecture["lecture_pdf"]
        if not pdf_rel or pdf_rel.endswith("missing.pdf"):
            continue
        pdf_path = ROOT / pdf_rel
        include_path = os.path.relpath(pdf_path, BUILD_DIR)
        section_title = latex_escape(lecture["title"])
        lines.extend(
            [
                f"\\section{{{section_title}}}",
                f"\\includepdf[pages=-,pagecommand={{\\thispagestyle{{plain}}}}]{{{include_path}}}",
                "",
            ]
        )

    lines.append(r"\end{document}")
    tex_path.write_text("\n".join(lines) + "\n")
    compile_tex(tex_path)
    print(tex_path)


if __name__ == "__main__":
    main()
