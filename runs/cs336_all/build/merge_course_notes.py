#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build"
DELIVERABLE_DIR = ROOT / "deliverable"


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


def clean_title(text: str) -> str:
    text = text.split("|")[-1].strip()
    text = text.replace("Lec.", "Lecture")
    return text


def compile_tex(tex_path: Path) -> None:
    for suffix in [".aux", ".out", ".toc"]:
        stale = tex_path.with_suffix(suffix)
        if stale.exists():
            stale.unlink()
    for _ in range(2):
        subprocess.run(
            ["xelatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tex_path.parent,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def publish_deliverables(*paths: Path) -> None:
    DELIVERABLE_DIR.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.exists():
            shutil.copy2(path, DELIVERABLE_DIR / path.name)


def main() -> None:
    manifest = json.loads((BUILD_DIR / "course_manifest.json").read_text())
    tex_path = BUILD_DIR / "cs336_complete_notes.tex"
    title = latex_escape(manifest.get("title", "Stanford CS336 全课程讲义"))
    course_page_url = manifest.get("course_page_url")
    playlist_url = manifest.get("playlist_origin_url") or manifest.get("playlist_url")
    playlist_channel = latex_escape(manifest.get("playlist_channel", "unknown"))
    scheduled_count = manifest.get("scheduled_session_count")
    public_count = manifest.get("public_playlist_count")
    missing_public = manifest.get("missing_public_sessions") or []

    lines = [
        r"\documentclass[a4paper]{article}",
        r"\usepackage[fontset=fandol]{ctex}",
        r"\usepackage[margin=2cm]{geometry}",
        r"\usepackage{pdfpages}",
        r"\usepackage{hyperref}",
        r"\usepackage{longtable}",
        f"\\title{{{title}}}",
        r"\author{Codex based on course materials, official traces, subtitles, and lecture videos}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
        r"\tableofcontents",
        r"\newpage",
        r"\section*{课程说明}",
        r"本讲义先逐讲生成，再按播放列表顺序合并。",
        r"\addcontentsline{toc}{section}{课程说明}",
        r"\section*{来源说明}",
        r"\addcontentsline{toc}{section}{来源说明}",
    ]
    if course_page_url and playlist_url:
        lines.append(
            f"本书基于 Stanford CS336 Spring 2025 课程页（\\href{{{course_page_url}}}{{course page}}）和 Stanford Online 官方公开 playlist（\\href{{{playlist_url}}}{{playlist}}）重建。"
        )
    if scheduled_count and public_count:
        lines.append(
            f"官方课程表共有 {scheduled_count} 次排课，Stanford Online 公开 playlist 提供其中 {public_count} 个公开视频；频道为 {playlist_channel}。"
        )
    if (ROOT / "meta" / "latest_offering_2026.json").exists():
        lines.append(
            r"此外，本书第 1-6 讲还显式对照了当前 Spring 2026 官方 course page / slides / scripts，并在不改写 2025 主视频主线的前提下吸收了新增教学内容。"
        )
    if missing_public:
        lines.append(r"\begin{itemize}")
        for row in missing_public:
            desc = latex_escape(row.get("description", "missing public session"))
            date = latex_escape(row.get("date", "unknown date"))
            lines.append(f"\\item 未公开视频的课程表讲次：{row.get('schedule_index')}（{date}，{desc}）。")
        lines.append(r"\end{itemize}")
    lines.extend(
        [
            r"\section*{课程目录}",
            r"\addcontentsline{toc}{section}{课程目录}",
            r"\begin{longtable}{p{0.08\textwidth}p{0.67\textwidth}p{0.18\textwidth}}",
            r"\textbf{讲次} & \textbf{主题} & \textbf{日期}\\",
            r"\hline",
        ]
    )

    for lecture in manifest["lectures"]:
        lines.append(f"{lecture['lecture_id']} & {latex_escape(clean_title(lecture['title']))} & {lecture['date']}\\\\")
    lines.extend([r"\end{longtable}", ""])

    for lecture in manifest["lectures"]:
        pdf_rel = lecture["lecture_pdf"]
        if not pdf_rel or pdf_rel.endswith("missing.pdf"):
            continue
        pdf_path = ROOT / pdf_rel
        include_path = os.path.relpath(pdf_path, BUILD_DIR)
        section_title = latex_escape(clean_title(lecture["title"]))
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
    publish_deliverables(tex_path, tex_path.with_suffix(".pdf"))
    print(tex_path)


if __name__ == "__main__":
    main()
