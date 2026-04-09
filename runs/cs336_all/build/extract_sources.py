#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "raw"
MATERIALS_DIR = ROOT / "materials" / "spring2025-lectures"
TEXT_DIR = ROOT / "text"
META_PATH = ROOT / "meta" / "playlist_full.json"


def read_json(path: Path):
    return json.loads(path.read_text())


def srt_to_text(path: Path) -> str:
    lines = []
    prev = None
    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.isdigit():
            continue
        if "-->" in line:
            continue
        if line == prev:
            continue
        lines.append(line)
        prev = line
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_calls(py_path: Path) -> tuple[str, list[str]]:
    source = py_path.read_text()
    tree = ast.parse(source)
    chunks: list[str] = []
    images: list[str] = []
    funcs = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    visited_funcs: set[str] = set()

    def literal_str(node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    def visit_node(node: ast.AST):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name == "text" and node.args:
                value = literal_str(node.args[0])
                if value:
                    chunks.append(value)
            elif func_name == "image" and node.args:
                value = literal_str(node.args[0])
                if value:
                    images.append(value)
            elif func_name in funcs:
                visit_function(func_name)
        for child in ast.iter_child_nodes(node):
            visit_node(child)

    def visit_function(name: str):
        if name in visited_funcs:
            return
        visited_funcs.add(name)
        for stmt in funcs[name].body:
            visit_node(stmt)

    if "main" in funcs:
        visit_function("main")
    else:
        for stmt in tree.body:
            visit_node(stmt)
    cleaned = []
    for chunk in chunks:
        chunk = chunk.replace("<br>", "\n")
        chunk = re.sub(r"`([^`]+)`", r"\1", chunk)
        chunk = re.sub(r"\s+\n", "\n", chunk)
        cleaned.append(chunk.strip())
    return "\n".join(x for x in cleaned if x).strip(), images


def pdf_to_text(pdf_path: Path) -> str:
    proc = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        capture_output=True,
        check=True,
        text=True,
    )
    text = proc.stdout.replace("\f", "\n\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main():
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    playlist = read_json(META_PATH)
    lectures_out = []

    for entry in playlist["entries"]:
        idx = int(entry["playlist_index"])
        lecture_dir = TEXT_DIR / f"{idx:02d}_{entry['id']}"
        lecture_dir.mkdir(parents=True, exist_ok=True)

        subtitle = RAW_DIR / f"{idx:02d}_{entry['id']}" / f"{idx:02d}_{entry['id']}.en-orig.srt"
        if not subtitle.exists():
            subtitle = RAW_DIR / f"{idx:02d}_{entry['id']}" / f"{idx:02d}_{entry['id']}.en.srt"

        transcript_txt = lecture_dir / "transcript.txt"
        transcript_txt.write_text(srt_to_text(subtitle))

        py_material = MATERIALS_DIR / f"lecture_{idx:02d}.py"
        pdf_candidates = sorted((MATERIALS_DIR / "nonexecutable").glob(f"2025 Lecture {idx} -*.pdf"))
        official_txt = lecture_dir / "official.txt"
        material_images: list[str] = []
        material_path: str
        if py_material.exists():
            official_text, material_images = extract_text_calls(py_material)
            material_path = str(py_material.relative_to(ROOT))
        elif pdf_candidates:
            official_text = pdf_to_text(pdf_candidates[0])
            material_path = str(pdf_candidates[0].relative_to(ROOT))
        else:
            official_text = ""
            material_path = ""
        official_txt.write_text(official_text)

        lecture_meta = {
            "playlist_index": idx,
            "video_id": entry["id"],
            "title": entry["title"],
            "duration": entry.get("duration"),
            "thumbnail": str((RAW_DIR / f"{idx:02d}_{entry['id']}" / f"{idx:02d}_{entry['id']}.jpg").relative_to(ROOT)),
            "subtitle": str(subtitle.relative_to(ROOT)),
            "material": material_path,
            "images": material_images,
            "transcript_text": str(transcript_txt.relative_to(ROOT)),
            "official_text": str(official_txt.relative_to(ROOT)),
        }
        (lecture_dir / "bundle.json").write_text(json.dumps(lecture_meta, indent=2))
        lectures_out.append(lecture_meta)

    bundle_path = TEXT_DIR / "course_bundle.json"
    bundle_path.write_text(json.dumps(lectures_out, indent=2))
    print(bundle_path)
    print(f"lectures={len(lectures_out)}")


if __name__ == "__main__":
    main()
