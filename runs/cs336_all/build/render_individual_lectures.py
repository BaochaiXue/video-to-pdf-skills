#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RUN_ROOT.parents[1]
LECTURES_DIR = RUN_ROOT / "lectures"
MASTER_TEX = RUN_ROOT / "build" / "cs336_all_in_one.tex"
TEMPLATE_TEX = REPO_ROOT / "skills" / "youtube-render-pdf" / "assets" / "notes-template.tex"


LECTURE_TITLES_CN = {
    1: "课程全景与 Tokenization",
    2: "PyTorch 与资源核算",
    3: "架构与超参数",
    4: "Mixture of Experts",
    5: "GPU 基础与硬件直觉",
    6: "Kernel 优化与 Triton",
    7: "并行训练 I",
    8: "并行训练 II",
    9: "Scaling Laws I",
    10: "推理系统",
    11: "Scaling Laws II",
    12: "评估体系",
    13: "数据工程 I",
    14: "数据工程 II",
    15: "对齐：SFT / RLHF",
    16: "对齐：RL I",
    17: "对齐：RL II",
}


@dataclass
class LectureBlock:
    start_lecture: int
    end_lecture: int
    title: str
    content: str


@dataclass
class SectionBlock:
    title: str
    lecture_blocks: list[LectureBlock]


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


def replace_command(tex: str, name: str, value: str) -> str:
    pattern = re.compile(rf"\\newcommand\{{\\{name}\}}\{{.*?\}}")
    return pattern.sub(lambda _m: f"\\newcommand{{\\{name}}}{{{value}}}", tex, count=1)


def format_upload_date(raw: str | None) -> str:
    if not raw:
        return "未注明"
    dt = datetime.strptime(raw, "%Y%m%d")
    return dt.strftime("%Y年%m月%d日")


def clean_lecture_heading(title: str) -> str:
    return title.strip()


def adjust_paths(content: str) -> str:
    content = content.replace("../materials/", "../../materials/")
    content = content.replace("../raw/", "../../raw/")
    return content


def parse_master_sections() -> list[SectionBlock]:
    text = MASTER_TEX.read_text()
    body = text.split(r"\tableofcontents", 1)[1]
    body = body.split(r"\newpage", 1)[1]
    body = body.rsplit(r"\end{document}", 1)[0]

    section_matches = list(re.finditer(r"\\section\{([^}]*)\}", body))
    sections: list[SectionBlock] = []
    for idx, match in enumerate(section_matches):
        title = match.group(1)
        if title == "总结与延伸":
            break
        start = match.end()
        end = section_matches[idx + 1].start() if idx + 1 < len(section_matches) else len(body)
        section_text = body[start:end]

        subsection_matches = list(re.finditer(r"\\subsection\{([^}]*)\}", section_text))
        lecture_blocks: list[LectureBlock] = []
        for sidx, smatch in enumerate(subsection_matches):
            heading = smatch.group(1)
            block_start = smatch.end()
            block_end = subsection_matches[sidx + 1].start() if sidx + 1 < len(subsection_matches) else len(section_text)
            content = section_text[block_start:block_end].strip()
            lecture_match = re.match(r"讲次 (\d+)(?:--(\d+))?：(.+)", heading)
            if not lecture_match:
                continue
            start_lecture = int(lecture_match.group(1))
            end_lecture = int(lecture_match.group(2) or lecture_match.group(1))
            lecture_blocks.append(
                LectureBlock(
                    start_lecture=start_lecture,
                    end_lecture=end_lecture,
                    title=clean_lecture_heading(lecture_match.group(3)),
                    content=adjust_paths(content),
                )
            )
        sections.append(SectionBlock(title=title, lecture_blocks=lecture_blocks))
    return sections


def build_summary_text(lecture_idx: int, section_title: str, subsection_titles: list[str]) -> str:
    titles = "、".join(f"“{title}”" for title in subsection_titles)
    next_title = LECTURE_TITLES_CN.get(lecture_idx + 1)
    lines = [
        f"本讲围绕{titles}展开，核心目标是把“{LECTURE_TITLES_CN[lecture_idx]}”放回整门课的系统语境里理解。与其把这一讲只当作若干术语、图表或公式的堆叠，更重要的是看清它到底在回答什么问题、通过什么机制作答，以及这些结论会怎样约束后续工程实现。",
        "",
        "如果只抓住三个带走点，可以概括为：",
        r"\begin{itemize}",
        r"\item 先把问题定义、资源约束或目标函数说清楚，再谈具体技巧和实现细节。",
        r"\item 把正文中的结构图、复杂度关系和工程权衡一起读，避免只记住局部结论却忽略适用条件。",
        r"\item 回到整门 CS336 的主线，理解这一讲如何为后面的训练、推理、评估、数据或对齐章节提供前提。",
        r"\end{itemize}",
    ]
    if next_title:
        lines.extend(
            [
                "",
                f"继续阅读下一讲“{next_title}”时，可以特别留意本讲中的哪些假设会被继承、放大或修正；这通常也是把单讲知识真正串成体系的关键。"
            ]
        )
    return "\n".join(lines)


def build_final_section(lecture_idx: int, section_title: str) -> str:
    lines = [
        r"\section{总结与延伸}",
        f"从整门 CS336 的结构来看，第 {lecture_idx} 讲处理的是“{section_title}”这一层面的关键问题。它不是孤立知识点，而是在回答一个更大的系统问题：如何在真实资源、任务目标和实现约束下，把语言模型做得更强、更稳、更高效。",
        "",
        "如果继续精读，本讲最值得反复回看的材料有三类：一是核心图示与结构图，二是涉及复杂度、内存、吞吐或目标函数的公式，三是讲者反复强调的工程权衡。把这三类证据对齐起来，通常就能更准确地把握本讲真正的结论，而不是只记住若干局部术语。",
    ]
    if lecture_idx == 1:
        lines.extend(
            [
                "",
                "作为课程开篇，这一讲还承担了给整门课立问题、立世界观的作用。后续每一讲其实都可以看成是在回答这里埋下的同一个问题：语言模型系统到底由哪些相互制约的部件组成。"
            ]
        )
    elif lecture_idx == 17:
        lines.extend(
            [
                "",
                "作为课程收束，这一讲也把整门课重新指向后训练、可验证奖励和 test-time compute 的前沿问题。到这里再回看课程开头，会更清楚地理解“从零构建语言模型”为什么最终会走到对齐与强化学习。"
            ]
        )
    else:
        next_title = LECTURE_TITLES_CN.get(lecture_idx + 1)
        if next_title:
            lines.extend(
                [
                    "",
                    f"从学习路径上看，最自然的下一步是把本讲结论带到下一讲“{next_title}”里继续验证：看看这里建立的机制、代价模型或行为假设，会如何在后续章节里被继承、挑战或扩展。"
                ]
            )
    return "\n".join(lines)


def read_meta(lecture_dir: Path) -> tuple[dict, dict]:
    meta = json.loads((lecture_dir / "meta.json").read_text())
    raw_dir = RUN_ROOT / "raw" / f"{meta['playlist_index']:02d}_{meta['video_id']}"
    info = json.loads((raw_dir / f"{meta['playlist_index']:02d}_{meta['video_id']}.info.json").read_text())
    return meta, info


def build_body_for_lecture(lecture_idx: int, sections: list[SectionBlock]) -> str:
    body_parts: list[str] = []
    matched_any = False
    for section in sections:
        matched_blocks = [
            block
            for block in section.lecture_blocks
            if block.start_lecture <= lecture_idx <= block.end_lecture
        ]
        if not matched_blocks:
            continue
        matched_any = True
        body_parts.append(f"\\section{{{section.title}}}")
        body_parts.append("")
        subsection_titles: list[str] = []
        for block in matched_blocks:
            subsection_titles.append(block.title)
            body_parts.append(f"\\subsection{{{block.title}}}")
            body_parts.append(block.content)
            body_parts.append("")
        body_parts.append(r"\subsection{本章小结}")
        body_parts.append(build_summary_text(lecture_idx, section.title, subsection_titles))
        body_parts.append("")
        body_parts.append(build_final_section(lecture_idx, section.title))
        body_parts.append("")
    if not matched_any:
        raise ValueError(f"no lecture content found for lecture {lecture_idx}")
    return "\n".join(body_parts).strip() + "\n"


def build_tex(lecture_dir: Path, lecture_idx: int, sections: list[SectionBlock]) -> str:
    meta, info = read_meta(lecture_dir)
    template = TEMPLATE_TEX.read_text()

    note_title = f"CS336 第 {lecture_idx} 讲：{LECTURE_TITLES_CN[lecture_idx]}"
    note_authors = f"Codex 基于 Stanford CS336 2025 第 {lecture_idx} 讲视频、字幕与课程材料整理"
    video_channel = info.get("channel") or info.get("uploader") or "Stanford Online"
    publish_date = format_upload_date(info.get("upload_date"))
    duration = info.get("duration_string") or str(meta.get("duration") or "")
    video_url = info.get("webpage_url") or f"https://www.youtube.com/watch?v={meta['video_id']}"

    template = replace_command(template, "notetitle", latex_escape(note_title))
    template = replace_command(template, "noteauthors", latex_escape(note_authors))
    template = replace_command(template, "notedate", r"\today")
    template = replace_command(template, "videochannel", latex_escape(video_channel))
    template = replace_command(template, "videopublishdate", latex_escape(publish_date))
    template = replace_command(template, "videoduration", latex_escape(duration))
    template = replace_command(template, "videourl", video_url)
    template = replace_command(template, "videocoverpath", "cover.jpg")

    body = build_body_for_lecture(lecture_idx, sections)
    template = re.sub(
        r"%% --- 正文内容开始 --- %%.*?%% --- 正文内容结束 --- %%",
        lambda _m: "%% --- 正文内容开始 --- %%\n\n" + body + "\n%% --- 正文内容结束 --- %%",
        template,
        flags=re.S,
    )
    return template


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


def lecture_dirs(selected: set[int] | None = None) -> list[Path]:
    dirs = sorted(LECTURES_DIR.glob("[0-9][0-9]_*"))
    if selected is None:
        return dirs
    out = []
    for lecture_dir in dirs:
        idx = int(lecture_dir.name.split("_", 1)[0])
        if idx in selected:
            out.append(lecture_dir)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lectures", nargs="*", type=int, help="lecture numbers to render")
    parser.add_argument("--force", action="store_true", help="overwrite existing tex/pdf outputs")
    parser.add_argument("--no-compile", action="store_true", help="generate tex only")
    args = parser.parse_args()

    selected = set(args.lectures) if args.lectures else None
    sections = parse_master_sections()

    rendered: list[str] = []
    skipped: list[str] = []
    for lecture_dir in lecture_dirs(selected):
        lecture_idx = int(lecture_dir.name.split("_", 1)[0])
        tex_path = lecture_dir / f"lecture_{lecture_idx:02d}_note.tex"
        pdf_path = lecture_dir / f"lecture_{lecture_idx:02d}_note.pdf"

        if not args.force and tex_path.exists() and pdf_path.exists():
            skipped.append(lecture_dir.name)
            continue

        tex = build_tex(lecture_dir, lecture_idx, sections)
        tex_path.write_text(tex)
        if not args.no_compile:
            compile_tex(tex_path)
        rendered.append(lecture_dir.name)

    print("rendered=" + ",".join(rendered))
    print("skipped=" + ",".join(skipped))


if __name__ == "__main__":
    main()
