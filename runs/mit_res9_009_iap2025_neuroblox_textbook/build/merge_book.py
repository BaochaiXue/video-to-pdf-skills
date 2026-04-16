#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = RUN_ROOT / "build"
BOOK_DIR = RUN_ROOT / "book"
PDFS_DIR = BOOK_DIR / "compiled_pdfs"


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


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


def manifest() -> dict:
    manifest_path = BUILD_DIR / "course_manifest.json"
    if manifest_path.exists():
        loaded = load_json(manifest_path)
        if isinstance(loaded, dict):
            return loaded
    seed_path = BUILD_DIR / "course_manifest_seed.json"
    if seed_path.exists():
        seed = load_json(seed_path)
        if isinstance(seed, dict):
            return {**seed, "lectures": seed.get("lectures", [])}
    return {
        "course_id": "MIT RES.9-009",
        "course_slug": "mit_res9_009_iap2025_neuroblox_textbook",
        "title": "MIT RES.9-009: Introduction to Computational Neuroscience with Neuroblox (January IAP 2025)",
        "term": "IAP 2025",
        "lectures": [],
    }


def main() -> None:
    BOOK_DIR.mkdir(parents=True, exist_ok=True)
    (BOOK_DIR / "chapters").mkdir(parents=True, exist_ok=True)
    PDFS_DIR.mkdir(parents=True, exist_ok=True)
    course = manifest()
    title = latex_escape(str(course.get("title") or "MIT RES.9-009 Neuroblox Textbook"))
    lectures = course.get("lectures") if isinstance(course.get("lectures"), list) else []

    chapter_rows: list[str] = []
    chapter_sections: list[str] = []
    chapter_manifest: list[dict] = []

    for lecture in lectures:
        if not isinstance(lecture, dict):
            continue
        lecture_id = str(lecture.get("lecture_id") or "")
        lecture_slug = str(lecture.get("lecture_slug") or "")
        lecture_title = latex_escape(str(lecture.get("title") or lecture_slug or lecture_id))
        lecture_pdf_rel = lecture.get("lecture_pdf")
        lecture_tex_rel = lecture.get("lecture_tex")
        lecture_state = str(lecture.get("evaluation_state") or "unknown")
        chapter_rows.append(
            f"{latex_escape(lecture_id)} & {lecture_title} & {latex_escape(lecture_state)}\\\\"
        )
        chapter_manifest.append(
            {
                "lecture_id": lecture_id,
                "lecture_slug": lecture_slug,
                "title": lecture.get("title") or lecture_slug,
                "lecture_tex": lecture_tex_rel,
                "lecture_pdf": lecture_pdf_rel,
                "latest_eval_report": lecture.get("latest_eval_report"),
                "evaluation_state": lecture_state,
            }
        )
        if lecture_pdf_rel:
            pdf_path = RUN_ROOT / str(lecture_pdf_rel)
            if pdf_path.exists():
                staged_name = f"lec{lecture_id}.pdf"
                staged_pdf = PDFS_DIR / staged_name
                if staged_pdf.exists() or staged_pdf.is_symlink():
                    staged_pdf.unlink()
                shutil.copyfile(pdf_path, staged_pdf)
                include_path = os.path.relpath(staged_pdf, BOOK_DIR)
                chapter_sections.extend(
                    [
                        f"\\section{{{lecture_id} {lecture_title}}}",
                        f"\\includepdf[pages=-,pagecommand={{\\thispagestyle{{plain}}}}]{{\\detokenize{{{include_path}}}}}",
                        "",
                    ]
                )

    frontmatter_inputs = [
        r"\input{frontmatter/preface}",
        r"\input{frontmatter/how_to_use_this_book}",
    ]
    appendix_inputs = [
        r"\input{appendices/glossary}",
        r"\input{appendices/notation}",
    ]

    lines = [
        r"\documentclass[a4paper]{article}",
        r"\usepackage[fontset=fandol]{ctex}",
        r"\usepackage[margin=2cm]{geometry}",
        r"\usepackage{hyperref}",
        r"\usepackage{longtable}",
        r"\usepackage{pdfpages}",
        f"\\title{{{title}\\\\教材级中文讲义}}",
        r"\author{基于 MIT OpenCourseWare、Neuroblox 官方课程页与 MIT Video Productions 公开录播重建}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
        r"\tableofcontents",
        r"\newpage",
        r"\section*{课程说明}",
        r"本书由 `lectures/` 下通过 evaluator 与 validator 的章节工作区合并而成；视频侧的公开播放入口以 MIT Video Productions 的 `IAP 2025` playlist 为准，章节边界则以官方 Neuroblox 课程页单元结构为准。",
        r"\addcontentsline{toc}{section}{课程说明}",
    ]
    lines.extend(frontmatter_inputs)
    lines.extend(
        [
            r"\section*{章节目录}",
            r"\addcontentsline{toc}{section}{章节目录}",
            r"\begin{longtable}{p{0.10\textwidth}p{0.68\textwidth}p{0.14\textwidth}}",
            r"\textbf{章号} & \textbf{章节标题} & \textbf{状态}\\",
            r"\hline",
        ]
    )
    if chapter_rows:
        lines.extend(chapter_rows)
    else:
        lines.append(r"\multicolumn{3}{l}{尚未生成任何章节；请先完成 `lectures/` 中的章节工作区。}\\")
    lines.extend([r"\end{longtable}", ""])
    lines.extend(chapter_sections)
    lines.extend(appendix_inputs)
    lines.append(r"\end{document}")

    (BOOK_DIR / "main.tex").write_text("\n".join(lines) + "\n")

    frontmatter_lines = [
        r"\documentclass[a4paper]{article}",
        r"\usepackage[fontset=fandol]{ctex}",
        r"\usepackage[margin=2cm]{geometry}",
        r"\usepackage{hyperref}",
        r"\usepackage{longtable}",
        f"\\title{{{title}\\\\教材级中文讲义}}",
        r"\author{基于 MIT OpenCourseWare、Neuroblox 官方课程页与 MIT Video Productions 公开录播重建}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
        r"\tableofcontents",
        r"\newpage",
        r"\section*{课程说明}",
        r"本书由 `lectures/` 下通过 evaluator 与 validator 的章节工作区合并而成；视频侧的公开播放入口以 MIT Video Productions 的 `IAP 2025` playlist 为准，章节边界则以官方 Neuroblox 课程页单元结构为准。",
        r"\addcontentsline{toc}{section}{课程说明}",
    ]
    frontmatter_lines.extend(frontmatter_inputs)
    frontmatter_lines.extend(
        [
            r"\section*{章节目录}",
            r"\addcontentsline{toc}{section}{章节目录}",
            r"\begin{longtable}{p{0.10\textwidth}p{0.68\textwidth}p{0.14\textwidth}}",
            r"\textbf{章号} & \textbf{章节标题} & \textbf{状态}\\",
            r"\hline",
        ]
    )
    if chapter_rows:
        frontmatter_lines.extend(chapter_rows)
    else:
        frontmatter_lines.append(r"\multicolumn{3}{l}{尚未生成任何章节；请先完成 `lectures/` 中的章节工作区。}\\")
    frontmatter_lines.extend([r"\end{longtable}", r"\end{document}"])
    (BOOK_DIR / "frontmatter.tex").write_text("\n".join(frontmatter_lines) + "\n")

    appendix_lines = [
        r"\documentclass[a4paper]{article}",
        r"\usepackage[fontset=fandol]{ctex}",
        r"\usepackage[margin=2cm]{geometry}",
        r"\usepackage{hyperref}",
        r"\usepackage{longtable}",
        r"\begin{document}",
    ]
    appendix_lines.extend(appendix_inputs)
    appendix_lines.append(r"\end{document}")
    (BOOK_DIR / "appendix.tex").write_text("\n".join(appendix_lines) + "\n")

    (BOOK_DIR / "textbook_source_manifest.json").write_text(json.dumps({"course": course, "chapters": chapter_manifest}, indent=2, ensure_ascii=False) + "\n")
    print(BOOK_DIR / "main.tex")


if __name__ == "__main__":
    main()
