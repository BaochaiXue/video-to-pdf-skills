#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "text" / "course_bundle.json"
LECTURES_DIR = ROOT / "lectures"
MATERIALS_DIR = ROOT / "materials" / "spring2025-lectures"
TRACE_DIR = MATERIALS_DIR / "var" / "traces"


def slugify(text: str) -> str:
    text = text.split("|")[-1].strip()
    if ":" in text:
        text = text.split(":", 1)[1].strip()
    text = text.lower()
    text = text.replace("&", "and")
    text = text.replace("/", "_")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def ensure_symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(os.path.relpath(src, dst.parent))


def link_to_md(rendering: dict) -> str:
    external = rendering.get("external_link")
    internal = rendering.get("internal_link")
    if external:
        title = (external.get("title") or external.get("url") or "link").strip()
        url = external.get("url") or ""
        return f"[{title}]({url})" if url else title
    if internal:
        path = internal.get("path", "")
        line = internal.get("line_number", "")
        label = rendering.get("data") or path
        return f"`{label}` ({path}:{line})"
    return rendering.get("data") or ""


def step_renderings_to_blocks(step: dict, lecture_dir: Path) -> list[str]:
    blocks: list[str] = []
    inline_parts: list[str] = []

    def flush_inline():
        nonlocal inline_parts
        if inline_parts:
            blocks.append(" ".join(part for part in inline_parts if part).strip())
            inline_parts = []

    for rendering in step.get("renderings", []):
        rtype = rendering.get("type")
        if rtype == "markdown":
            data = (rendering.get("data") or "").strip()
            if not data:
                continue
            if data.startswith("#"):
                flush_inline()
                blocks.append(data)
            else:
                inline_parts.append(data)
        elif rtype == "link":
            inline_parts.append(link_to_md(rendering))
        elif rtype == "image":
            flush_inline()
            raw_path = rendering.get("data") or ""
            width = rendering.get("style", {}).get("width")
            src = Path(raw_path)
            if not src.is_absolute():
                src = (MATERIALS_DIR / raw_path).resolve()
            rel = os.path.relpath(src, lecture_dir)
            if width:
                blocks.append(f'<img src="{rel}" width="{width}">')
            else:
                blocks.append(f"![]({rel})")
    flush_inline()
    return [block for block in blocks if block]


def trace_to_markdown(trace_path: Path, lecture_dir: Path) -> str:
    data = json.loads(trace_path.read_text())
    blocks: list[str] = []
    for step in data.get("steps", []):
        blocks.extend(step_renderings_to_blocks(step, lecture_dir))

    deduped: list[str] = []
    prev = None
    for block in blocks:
        if block == prev:
            continue
        deduped.append(block)
        prev = block
    return "\n\n".join(deduped).strip() + "\n"


def official_text_to_markdown(text_path: Path) -> str:
    text = text_path.read_text(errors="ignore")
    text = text.replace("\f", "\n\n")
    lines = [line.rstrip() for line in text.splitlines()]
    out: list[str] = []
    prev_blank = True
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if not prev_blank:
                out.append("")
            prev_blank = True
            continue
        if re.fullmatch(r"Lecture\s+\d+", stripped):
            out.append(f"# {stripped}")
        elif stripped.startswith("###") or stripped.startswith("##"):
            out.append(stripped)
        else:
            out.append(stripped)
        prev_blank = False
    return "\n".join(out).strip() + "\n"


def build_lecture_folder(item: dict):
    idx = int(item["playlist_index"])
    lecture_dir = LECTURES_DIR / f"{idx:02d}_{slugify(item['title'])}"
    lecture_dir.mkdir(parents=True, exist_ok=True)

    bundle_out = {
        **item,
        "title_short": item["title"].split("|")[-1].strip(),
        "lecture_dir": str(lecture_dir.relative_to(ROOT)),
    }
    (lecture_dir / "meta.json").write_text(json.dumps(bundle_out, indent=2))

    cover_src = ROOT / item["thumbnail"]
    transcript_src = ROOT / item["transcript_text"]
    official_src = ROOT / item["official_text"]
    ensure_symlink(cover_src, lecture_dir / "cover.jpg")
    ensure_symlink(transcript_src, lecture_dir / "transcript.txt")
    ensure_symlink(official_src, lecture_dir / "official.txt")

    notes_md = lecture_dir / "notes.md"
    trace_name = f"lecture_{idx:02d}.json"
    trace_src = TRACE_DIR / trace_name
    if trace_src.exists():
        ensure_symlink(trace_src, lecture_dir / "official_trace.json")
        notes_md.write_text(trace_to_markdown(trace_src, lecture_dir))
    else:
        notes_md.write_text(official_text_to_markdown(official_src))

    summary = [
        f"# {bundle_out['title_short']}",
        "",
        f"- Video ID: `{item['video_id']}`",
        f"- Cover: [cover.jpg](cover.jpg)",
        f"- Transcript: [transcript.txt](transcript.txt)",
        f"- Official material: [official.txt](official.txt)",
        f"- Generated notes: [notes.md](notes.md)",
    ]
    if trace_src.exists():
        summary.append("- Official trace: [official_trace.json](official_trace.json)")
    summary.append("")
    summary.append("This folder is the lecture-specific workspace for refining this lecture into its own polished note/PDF.")
    (lecture_dir / "README.md").write_text("\n".join(summary) + "\n")


def main():
    LECTURES_DIR.mkdir(parents=True, exist_ok=True)
    bundle = json.loads(BUNDLE_PATH.read_text())
    for item in bundle:
        build_lecture_folder(item)

    index_lines = ["# CS336 Lecture Folders", ""]
    for item in bundle:
        idx = int(item["playlist_index"])
        folder = f"{idx:02d}_{slugify(item['title'])}"
        title = item["title"].split("|")[-1].strip()
        index_lines.append(f"- [{idx:02d} {title}](./{folder}/README.md)")
    (LECTURES_DIR / "README.md").write_text("\n".join(index_lines) + "\n")
    print(f"created={len(bundle)}")
    print(LECTURES_DIR)


if __name__ == "__main__":
    main()
