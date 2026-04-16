#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
LECTURES_DIR = RUN_ROOT / "lectures"
BOOK_DIR = RUN_ROOT / "book"
CHAPTERS_DIR = BOOK_DIR / "chapters"

PARTS = [
    (
        "Foundations of Advanced LLM Agents",
        ["course_overview", "lec01_inference_time_reasoning", "lec02_learning_to_reason", "lec03_reasoning_memory_planning"],
    ),
    (
        "Agentic Workflows, Tools, and Code",
        ["lec04_open_training_recipes_reasoning", "lec05_coding_agents_vulnerability_detection"],
    ),
    (
        "Multimodal and Interactive Agents",
        ["lec06_multimodal_autonomous_agents", "lec07_multimodal_agents_perception_to_action"],
    ),
    (
        "Formal Mathematics, Verification, and Theorem Proving",
        ["lec08_alphaproof_formal_mathematics", "lec09_autoformalization_theorem_proving", "lec10_advanced_theorem_proving"],
    ),
    (
        "Abstraction, Discovery, Safety, and Security",
        ["lec11_abstraction_discovery_llm_agents", "lec12_safe_secure_agentic_ai"],
    ),
]


def lecture_dirs() -> list[Path]:
    return sorted(path for path in LECTURES_DIR.iterdir() if path.is_dir() and path.name.startswith("lec"))


def select_source_tex(lecture_dir: Path) -> Path | None:
    for candidate in [lecture_dir / "lecture_repaired.tex", lecture_dir / "lecture.tex"]:
        if candidate.exists():
            return candidate
    tex_files = sorted(lecture_dir.glob("lecture_*_note.tex"))
    return tex_files[0] if tex_files else None


def extract_chapter_body(source: Path) -> str:
    text = source.read_text()
    if r"\begin{document}" in text:
        text = text.split(r"\begin{document}", 1)[1]
    if r"\end{document}" in text:
        text = text.rsplit(r"\end{document}", 1)[0]
    text = re.sub(r"\\begin\{titlepage\}.*?\\end\{titlepage\}", "", text, flags=re.S)
    text = re.sub(r"\\tableofcontents", "", text)
    text = re.sub(r"\\newpage\s*", "", text)
    first_section = text.find(r"\section{")
    if first_section != -1:
        text = text[first_section:]
    return text.strip() + "\n"


def rewrite_asset_paths(body: str, lecture_dir: Path) -> str:
    lecture_rel = f"../lectures/{lecture_dir.name}/"
    return body.replace("{figures/", "{" + lecture_rel + "figures/")


def main() -> None:
    CHAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    lecture_entries = []
    chapter_inputs: list[str] = []
    for lecture_dir in lecture_dirs():
        source = select_source_tex(lecture_dir)
        if source is None:
            continue
        meta_path = lecture_dir / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"title": lecture_dir.name}
        chapter_name = f"{lecture_dir.name}.tex"
        target = CHAPTERS_DIR / chapter_name
        chapter_body = rewrite_asset_paths(extract_chapter_body(source), lecture_dir)
        target.write_text(
            "\n".join(
                [
                    f"\\chapter{{{meta['title']}}}",
                    f"\\label{{chap:{lecture_dir.name}}}",
                    chapter_body,
                ]
            )
        )
        lecture_entries.append(
            {
                "lecture_slug": lecture_dir.name,
                "lecture_title": meta.get("title"),
                "speaker": meta.get("speaker"),
                "date": meta.get("date"),
                "chapter_path": str(target.relative_to(RUN_ROOT)),
                "source_tex": str(source.relative_to(RUN_ROOT)),
                "source_manifest": str((lecture_dir / "source_manifest.json").relative_to(RUN_ROOT)) if (lecture_dir / "source_manifest.json").exists() else None,
                "eval_report": str((lecture_dir / "eval_report.json").relative_to(RUN_ROOT)) if (lecture_dir / "eval_report.json").exists() else None,
                "lecture_pdf": str((lecture_dir / "lecture.pdf").relative_to(RUN_ROOT)) if (lecture_dir / "lecture.pdf").exists() else None,
            }
        )
        chapter_inputs.append(lecture_dir.name)

    eval_scores_by_slug: dict[str, dict] = {}
    for entry in lecture_entries:
        eval_path = entry.get("eval_report")
        if not eval_path:
            continue
        full = RUN_ROOT / eval_path
        if not full.exists():
            continue
        report = json.loads(full.read_text())
        eval_scores_by_slug[entry["lecture_slug"]] = {
            "scores": report.get("scores", {}),
            "overall": report.get("overall"),
        }

    include_lines = []
    seen = set(chapter_inputs)
    for part_title, slugs in PARTS:
        materialized = [slug for slug in slugs if slug == "course_overview" or slug in seen]
        if not materialized:
            continue
        include_lines.append(f"\\part{{{part_title}}}")
        for slug in materialized:
            include_lines.append(f"\\input{{chapters/{slug}.tex}}")

    frontmatter = [
        r"\documentclass[a4paper]{ctexbook}",
        r"\usepackage[fontset=fandol]{ctex}",
        r"\usepackage[margin=2.5cm]{geometry}",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage{graphicx}",
        r"\usepackage[most]{tcolorbox}",
        r"\usepackage{listings}",
        r"\usepackage{hyperref}",
        r"\usepackage{xurl}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage{float}",
        r"\usepackage{xcolor}",
        r"\lstset{language=Python,basicstyle=\ttfamily\small,keywordstyle=\color{blue},stringstyle=\color{red!60!black},commentstyle=\color{green!50!black},breaklines=true,frame=single,numbers=left,numberstyle=\tiny\color{gray}}",
        r"\newtcolorbox{knowledgebox}[1]{enhanced,colback=blue!5!white,colframe=blue!70!black,colbacktitle=blue!70!black,coltitle=white,fonttitle=\bfseries,title=#1,sharp corners}",
        r"\newtcolorbox{importantbox}[1]{enhanced,colback=yellow!10!white,colframe=yellow!70!black,colbacktitle=yellow!70!black,coltitle=black,fonttitle=\bfseries,title=#1,sharp corners}",
        r"\newtcolorbox{warningbox}[1]{enhanced,colback=red!5!white,colframe=red!70!black,colbacktitle=red!70!black,coltitle=white,fonttitle=\bfseries,title=#1,sharp corners}",
        r"\title{CS294/194-280: Advanced Large Language Model Agents\\教材级中文讲义}",
        r"\author{Codex Harness-Managed Textbook Build}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
        r"\frontmatter",
        r"\input{frontmatter/preface.tex}",
        r"\input{frontmatter/how_to_use_this_book.tex}",
        r"\tableofcontents",
        r"\newpage",
        r"\mainmatter",
    ]
    ending = [
        r"\appendix",
        r"\input{appendices/exercises.tex}",
        r"\input{appendices/glossary.tex}",
        r"\input{appendices/notation.tex}",
        r"\input{appendices/paper_map.tex}",
        r"\input{appendices/benchmark_map.tex}",
        r"\input{appendices/algorithm_index.tex}",
        r"\input{appendices/figure_provenance.tex}",
        r"\input{appendices/omission_log.tex}",
        r"\input{appendices/suggested_reading_paths.tex}",
        r"\end{document}",
    ]
    (BOOK_DIR / "main.tex").write_text("\n".join(frontmatter + include_lines + ending) + "\n")
    textbook_manifest = {
        "course_id": "cs294_194_280_sp25_agents_textbook",
        "official_youtube_playlist": "https://www.youtube.com/playlist?list=PLS01nW3RtgorL3AW8REU9nGkzhvtn6Egn",
        "chapter_count": len(lecture_entries),
        "lectures": [],
    }
    for idx, entry in enumerate(lecture_entries, start=1):
        payload = {
            **entry,
            "chapter_label": f"Chapter {idx}",
        }
        payload.update(eval_scores_by_slug.get(entry["lecture_slug"], {}))
        textbook_manifest["lectures"].append(payload)
    (BOOK_DIR / "textbook_source_manifest.json").write_text(json.dumps(textbook_manifest, indent=2, ensure_ascii=False) + "\n")
    print(BOOK_DIR / "main.tex")


if __name__ == "__main__":
    main()
