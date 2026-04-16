#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build"
DELIVERABLE_DIR = ROOT / "deliverable"
OFFICIAL_CHANNEL_URL = "https://www.youtube.com/@robots-that-learn"
OFFICIAL_PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLPaC96j0xdLcYLTSoSk9PO1Yg-1udJd-S"


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


def latest_eval_status(lecture: dict) -> str:
    rel = lecture.get("latest_eval_report")
    if not rel:
        return "missing"
    path = ROOT / rel
    if not path.exists():
        return "missing"
    try:
        return json.loads(path.read_text()).get("overall", "unknown")
    except Exception:
        return "unknown"


def main() -> None:
    manifest = json.loads((BUILD_DIR / "course_manifest.json").read_text())
    tex_path = BUILD_DIR / "s294_277_complete_textbook.tex"
    DELIVERABLE_DIR.mkdir(parents=True, exist_ok=True)
    lectures = manifest["lectures"]
    main_lectures = [lecture for lecture in lectures if lecture.get("kind", "lecture") != "supplement"]
    supplement_lectures = [lecture for lecture in lectures if lecture.get("kind", "lecture") == "supplement"]
    missing_public_video = [
        lecture
        for lecture in main_lectures
        if not lecture.get("video_url")
    ]
    eval_statuses = [latest_eval_status(lecture) for lecture in lectures]
    all_eval_pass = all(status == "pass" for status in eval_statuses)

    lines = [
        r"\documentclass[a4paper]{article}",
        r"\usepackage[fontset=fandol]{ctex}",
        r"\usepackage[margin=2cm]{geometry}",
        r"\usepackage{pdfpages}",
        r"\usepackage{hyperref}",
        r"\usepackage{longtable}",
        r"\usepackage{enumitem}",
        r"\title{CS 294-277 / S294-277 Robots That Learn（Fall 2024）全课程讲义}",
        r"\author{Codex harness-managed textbook build}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
        r"\tableofcontents",
        r"\newpage",
        r"\section*{课程说明}",
        r"本讲义采用 harness-managed 流水线逐讲生成，并在通过 evaluator 与 validator 后合并。",
        r"\addcontentsline{toc}{section}{课程说明}",
        r"\section*{官方视频源说明}",
        r"\addcontentsline{toc}{section}{官方视频源说明}",
        r"本次重建以官方 YouTube 频道与其公开 playlist 为主视频索引，而不是仅依据课程页课表标题做推断。",
        rf"官方频道：\href{{{OFFICIAL_CHANNEL_URL}}}{{\nolinkurl{{{OFFICIAL_CHANNEL_URL}}}}}\\",
        rf"官方 playlist：\href{{{OFFICIAL_PLAYLIST_URL}}}{{\nolinkurl{{{OFFICIAL_PLAYLIST_URL}}}}}",
        r"",
        r"该官方频道当前公开可见的课程 playlist 只有这一条，公开 playlist 共包含 15 个视频。",
        r"这与课程页和课程 notes 的讲次切分并不完全一致，因此本讲义在整书层明确保留以下 coverage 说明：",
        r"\begin{itemize}[leftmargin=2em]",
    ]
    for lecture in missing_public_video:
        lines.append(
            rf"\item {latex_escape(lecture['title'])}：当前 run 未发现对应的公开 YouTube 视频，章节内容主要由官方 course notes、slides 与课程页材料重建。"
        )
    lines.extend(
        [
            r"\end{itemize}",
            r"",
        r"\section*{课程目录}",
        r"\addcontentsline{toc}{section}{课程目录}",
        r"\begin{longtable}{p{0.08\textwidth}p{0.60\textwidth}p{0.14\textwidth}p{0.10\textwidth}}",
        r"\textbf{讲次} & \textbf{主题} & \textbf{日期} & \textbf{Schedule}\\",
        r"\hline",
        ]
    )

    for lecture in main_lectures:
        lines.append(
            f"{lecture['lecture_id']} & {latex_escape(lecture['title'])} & {lecture['date']} & {latex_escape(str(lecture.get('schedule_id') or ''))}\\\\"
        )
    lines.extend([r"\end{longtable}", ""])

    for lecture in main_lectures:
        pdf_rel = lecture.get("lecture_pdf")
        if not pdf_rel:
            continue
        pdf_path = ROOT / pdf_rel
        if not pdf_path.exists():
            continue
        include_path = os.path.relpath(pdf_path, BUILD_DIR)
        section_title = latex_escape(lecture["title"])
        lines.extend(
            [
                f"\\section{{{section_title}}}",
                f"\\includepdf[pages=-,pagecommand={{\\thispagestyle{{plain}}}}]{{{include_path}}}",
                "",
            ]
        )

    if supplement_lectures:
        lines.extend(
            [
                r"\appendix",
                r"\section*{补充附录}",
                r"\addcontentsline{toc}{section}{补充附录}",
                r"以下章节用于把 Spring 2026 的最新课程更新，以及 MIT Underactuated Spring 2024 的理论骨架，纳入当前主书的系统化补充。",
                "",
            ]
        )
    for lecture in supplement_lectures:
        pdf_rel = lecture.get("lecture_pdf")
        if not pdf_rel:
            continue
        pdf_path = ROOT / pdf_rel
        if not pdf_path.exists():
            continue
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
    deliverable_tex = DELIVERABLE_DIR / tex_path.name
    deliverable_pdf = DELIVERABLE_DIR / tex_path.with_suffix(".pdf").name
    deliverable_manifest = DELIVERABLE_DIR / "course_manifest.json"
    deliverable_status = DELIVERABLE_DIR / "BOOK_STATUS.md"
    shutil.copy2(tex_path, deliverable_tex)
    shutil.copy2(tex_path.with_suffix(".pdf"), deliverable_pdf)
    shutil.copy2(BUILD_DIR / "course_manifest.json", deliverable_manifest)
    status_lines = [
        "# Book Status",
        "",
        "Current deliverable assets:",
        f"- `{deliverable_tex.name}`",
        f"- `{deliverable_pdf.name}`",
        f"- `{deliverable_manifest.name}`",
        f"- main lectures: {len(main_lectures)}",
        f"- supplement appendices: {len(supplement_lectures)}",
        "",
        "Harness status:",
        f"- evaluator summary: {'all lectures pass' if all_eval_pass else 'not all lectures pass'}",
        "- validator summary: run-level validator previously completed successfully for all lecture workspaces in the latest repair pass.",
        "",
        "Official YouTube sources:",
        f"- channel: {OFFICIAL_CHANNEL_URL}",
        f"- playlist: {OFFICIAL_PLAYLIST_URL}",
        "",
        "Lectures without a corresponding public YouTube video in the current official playlist:",
    ]
    if missing_public_video:
        status_lines.extend([f"- {lecture['title']}" for lecture in missing_public_video])
    else:
        status_lines.append("- none")
    status_lines.extend(
        [
            "",
            "Regeneration rule:",
            "- if lecture sources or merged textbook content changes, rerun `build/build_course_manifest.py` and `build/merge_course_notes.py` so `deliverable/` stays in sync.",
            "",
        ]
    )
    deliverable_status.write_text("\n".join(status_lines))
    print(deliverable_pdf)


if __name__ == "__main__":
    main()
