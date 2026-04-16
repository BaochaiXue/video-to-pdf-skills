#!/usr/bin/env python3
from __future__ import annotations

import json
import os
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
    log_path = tex_path.with_suffix(".log")
    for _ in range(2):
        with log_path.open("a") as handle:
            subprocess.run(
                ["xelatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
                cwd=tex_path.parent,
                check=True,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )


def latest_eval_pass(lecture: dict) -> bool:
    ref = lecture.get("latest_eval_report")
    if not ref or ref.endswith("null"):
        return False
    path = ROOT / ref
    if not path.exists():
        return False
    report = json.loads(path.read_text())
    return report.get("overall") == "pass"


def blocked_reason(lecture: dict) -> str:
    ref = lecture.get("latest_eval_report")
    if ref:
        path = ROOT / ref
        if path.exists():
            report = json.loads(path.read_text())
            issues = report.get("blocking_issues") or []
            if issues:
                return "; ".join(issue.get("problem", "blocking issue") for issue in issues[:3] if isinstance(issue, dict))
            if report.get("overall") != "pass":
                return "latest evaluator report did not pass"
    pdf_ref = lecture.get("lecture_pdf")
    if not pdf_ref or pdf_ref.endswith("missing.pdf"):
        return "lecture PDF not available"
    return "lecture not marked deliverable"


def main() -> None:
    manifest = json.loads((BUILD_DIR / "course_manifest.json").read_text())
    tex_path = BUILD_DIR / "speech_recognition_understanding_fall2023_textbook.tex"
    title = latex_escape(manifest.get("title", "Speech Recognition and Understanding 全课程讲义"))

    deliverable = []
    blocked = []
    for lecture in manifest["lectures"]:
        pdf_ref = lecture.get("lecture_pdf")
        pdf_ok = bool(pdf_ref) and not pdf_ref.endswith("missing.pdf") and (ROOT / pdf_ref).exists()
        if pdf_ok and latest_eval_pass(lecture):
            deliverable.append(lecture)
        else:
            blocked.append({**lecture, "blocked_reason": blocked_reason(lecture)})

    lines = [
        r"\documentclass[a4paper]{article}",
        r"\usepackage[fontset=fandol]{ctex}",
        r"\usepackage[margin=2cm]{geometry}",
        r"\usepackage{pdfpages}",
        r"\usepackage{hyperref}",
        r"\usepackage{longtable}",
        f"\\title{{{title}}}",
        r"\author{Codex harness-managed build from course page, public videos, and supplemental official materials}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
        r"\tableofcontents",
        r"\newpage",
        r"\section*{课程说明}",
        r"本教材按 harness-managed 流水线逐讲生成，并只纳入通过 evaluator gate 的章节。",
        r"\addcontentsline{toc}{section}{课程说明}",
        r"\section*{课程目录}",
        r"\addcontentsline{toc}{section}{课程目录}",
        r"\begin{longtable}{p{0.08\textwidth}p{0.55\textwidth}p{0.15\textwidth}p{0.15\textwidth}}",
        r"\textbf{讲次} & \textbf{主题} & \textbf{日期} & \textbf{状态}\\",
        r"\hline",
    ]

    deliverable_ids = {lecture["lecture_id"] for lecture in deliverable}
    for lecture in manifest["lectures"]:
        status = "deliverable" if lecture["lecture_id"] in deliverable_ids else "blocked"
        lines.append(
            f"{lecture['lecture_id']} & {latex_escape(lecture['title'])} & {lecture['date']} & {status}\\\\"
        )
    lines.extend([r"\end{longtable}", ""])

    for lecture in deliverable:
        pdf_rel = lecture["lecture_pdf"]
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

    if blocked:
        lines.extend(
            [
                r"\section{Blocked Sessions And Omissions}",
                r"以下讲次由于缺少公开源、章节尚未通过 evaluator，或尚未生成可交付 PDF，因此没有进入当前合并版教材。",
                r"\begin{longtable}{p{0.08\textwidth}p{0.35\textwidth}p{0.17\textwidth}p{0.30\textwidth}}",
                r"\textbf{讲次} & \textbf{主题} & \textbf{日期} & \textbf{原因}\\",
                r"\hline",
            ]
        )
        for lecture in blocked:
            lines.append(
                f"{lecture['lecture_id']} & {latex_escape(lecture['title'])} & {lecture['date']} & {latex_escape(lecture['blocked_reason'])}\\\\"
            )
        lines.append(r"\end{longtable}")

    lines.append(r"\end{document}")
    tex_path.write_text("\n".join(lines) + "\n")
    compile_tex(tex_path)
    print(tex_path)


if __name__ == "__main__":
    main()
