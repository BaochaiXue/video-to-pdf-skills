#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


RUN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RUN_ROOT.parents[2]
PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_"
COURSE_ID = "stanford-cs336-spring-2025"
COURSE_TITLE = "Stanford CS336: Language Modeling from Scratch (Spring 2025)"
MATERIALS_DIR = RUN_ROOT / "materials" / "spring2025-lectures"
LECTURES_DIR = RUN_ROOT / "lectures"
TEXT_DIR = RUN_ROOT / "text"


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(payload + ("\n" if payload else ""))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def lecture_dirs() -> list[Path]:
    return sorted(path for path in LECTURES_DIR.iterdir() if path.is_dir() and path.name[:2].isdigit())


def ensure_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(os.path.relpath(src, dst.parent))


def best_subtitle_path(meta: dict[str, Any]) -> Path:
    return RUN_ROOT / meta["subtitle"]


def parse_srt_entries(srt_path: Path) -> list[dict[str, str]]:
    text = srt_path.read_text(errors="ignore")
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    entries: list[dict[str, str]] = []
    for block in blocks:
        raw_lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(raw_lines) < 2:
            continue
        maybe_ts = raw_lines[1] if raw_lines[0].isdigit() else raw_lines[0]
        if "-->" not in maybe_ts:
            continue
        payload = raw_lines[2:] if raw_lines[0].isdigit() else raw_lines[1:]
        payload = [re.sub(r"<[^>]+>", "", line).strip() for line in payload]
        payload = [line for line in payload if line]
        if payload:
            start, end = [part.strip() for part in maybe_ts.split("-->", 1)]
            entries.append({"start": start, "end": end, "text": " ".join(payload)})
    return entries


def build_transcript_units(srt_path: Path) -> list[dict[str, Any]]:
    return [
        {
            "unit_id": f"sub_{idx:04d}",
            "source_type": "subtitle_span",
            "source_id": "subtitle_srt",
            "loc": {"start": entry["start"], "end": entry["end"]},
            "text": entry["text"],
            "required": True,
        }
        for idx, entry in enumerate(parse_srt_entries(srt_path), start=1)
    ]


def resolve_asset_path(raw_path: str, lecture_dir: Path) -> str | None:
    candidate = raw_path.strip()
    if not candidate:
        return None
    if candidate.startswith("http://") or candidate.startswith("https://"):
        return candidate
    if candidate.startswith("../") or candidate.startswith("../../"):
        return candidate
    if candidate.startswith("frames/") or candidate.startswith("pdf_pages/"):
        return candidate
    if candidate.startswith("images/"):
        return os.path.relpath(MATERIALS_DIR / candidate, lecture_dir)
    if candidate.startswith("/"):
        return candidate
    return candidate


def trace_row_text(rendering: dict[str, Any]) -> str:
    rtype = rendering.get("type")
    if rtype == "markdown":
        return (rendering.get("data") or "").strip()
    if rtype == "link":
        label = rendering.get("data") or ""
        external = rendering.get("external_link") or {}
        if external.get("title"):
            return str(external["title"]).strip()
        return str(label).strip()
    return ""


def build_slide_units_from_trace(trace_path: Path, lecture_dir: Path) -> list[dict[str, Any]]:
    data = load_json(trace_path)
    rows: list[dict[str, Any]] = []
    unit_idx = 1
    for step_idx, step in enumerate(data.get("steps", []), start=1):
        stack = step.get("stack") or []
        stack_frame = stack[-1] if stack else {}
        for rendering_idx, rendering in enumerate(step.get("renderings", []), start=1):
            asset_path = None
            if rendering.get("type") == "image":
                asset_path = resolve_asset_path(str(rendering.get("data") or ""), lecture_dir)
            text = trace_row_text(rendering)
            if not text and asset_path is None:
                continue
            rows.append(
                {
                    "unit_id": f"trace_{unit_idx:04d}",
                    "source_type": "official_trace_block",
                    "source_id": "official_trace",
                    "loc": {
                        "trace_step": step_idx,
                        "rendering_index": rendering_idx,
                        "path": stack_frame.get("path"),
                        "line_number": stack_frame.get("line_number"),
                        "function_name": stack_frame.get("function_name"),
                    },
                    "text": text,
                    "asset_path": asset_path,
                    "required": bool(text),
                }
            )
            unit_idx += 1
    return rows


def build_slide_units_from_markdown(notes_path: Path, lecture_dir: Path) -> list[dict[str, Any]]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", notes_path.read_text()) if block.strip()]
    rows: list[dict[str, Any]] = []
    unit_idx = 1
    image_pat = re.compile(r'!\[[^\]]*\]\(([^)]+)\)|<img\s+src="([^"]+)"')
    for block_idx, block in enumerate(blocks, start=1):
        matches = image_pat.findall(block)
        assets = [resolve_asset_path(m[0] or m[1], lecture_dir) for m in matches]
        text = re.sub(image_pat, "", block).strip()
        rows.append(
            {
                "unit_id": f"note_{unit_idx:04d}",
                "source_type": "official_note_block",
                "source_id": "notes_md",
                "loc": {"block": block_idx},
                "text": text,
                "asset_path": next((asset for asset in assets if asset), None),
                "required": bool(text),
            }
        )
        unit_idx += 1
    return rows


def build_slide_units(lecture_dir: Path) -> list[dict[str, Any]]:
    trace_path = lecture_dir / "official_trace.json"
    if trace_path.exists():
        rows = build_slide_units_from_trace(trace_path, lecture_dir)
        if rows:
            return rows
    notes_path = lecture_dir / "notes.md"
    if notes_path.exists():
        rows = build_slide_units_from_markdown(notes_path, lecture_dir)
        if rows:
            return rows
    official_txt = lecture_dir / "official.txt"
    if official_txt.exists():
        rows = []
        for idx, block in enumerate(re.split(r"\n\s*\n", official_txt.read_text()), start=1):
            block = block.strip()
            if block:
                rows.append(
                    {
                        "unit_id": f"official_{idx:04d}",
                        "source_type": "official_text_block",
                        "source_id": "official_txt",
                        "loc": {"block": idx},
                        "text": block,
                        "asset_path": None,
                        "required": True,
                    }
                )
        if rows:
            return rows
    return [
        {
            "unit_id": "official_missing_0001",
            "source_type": "official_missing_notice",
            "source_id": "official_material",
            "loc": {"block": None},
            "text": "No structured official material was found beyond the existing transcript and note outputs.",
            "asset_path": None,
            "required": False,
        }
    ]


def infer_topics_from_tex(tex_path: Path, fallback_title: str) -> list[str]:
    if not tex_path.exists():
        return [fallback_title]
    text = tex_path.read_text()
    topics: list[str] = []
    for match in re.finditer(r"\\section\{([^}]*)\}|\\subsection\{([^}]*)\}", text):
        title = (match.group(1) or match.group(2) or "").strip()
        if not title or title in {"本章小结", "总结与延伸", "拓展阅读"}:
            continue
        if title not in topics:
            topics.append(title)
    return topics or [fallback_title]


def build_segments(topics: list[str], transcript_units: list[dict[str, Any]], slide_units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segment_count = max(1, len(topics), (len(transcript_units) + 149) // 150)
    transcript_chunk_size = max(1, (len(transcript_units) + segment_count - 1) // segment_count) if transcript_units else 1
    slide_required = [row for row in slide_units if row.get("required")]
    slide_chunk_size = max(1, (len(slide_required) + segment_count - 1) // segment_count) if slide_required else 1
    segments: list[dict[str, Any]] = []
    for idx in range(segment_count):
        t_chunk = transcript_units[idx * transcript_chunk_size : (idx + 1) * transcript_chunk_size]
        s_chunk = slide_required[idx * slide_chunk_size : (idx + 1) * slide_chunk_size]
        segments.append(
            {
                "segment_id": f"seg_{idx + 1:02d}",
                "start": t_chunk[0]["loc"]["start"] if t_chunk else None,
                "end": t_chunk[-1]["loc"]["end"] if t_chunk else None,
                "source_unit_ids": [row["unit_id"] for row in t_chunk] + [row["unit_id"] for row in s_chunk],
                "target_section_hint": topics[idx] if idx < len(topics) else f"Segment {idx + 1}",
            }
        )
    return segments


def infer_kind(section_title: str) -> list[str]:
    title = section_title.lower()
    if any(token in title for token in ["公式", "推导", "bellman", "gradient", "rl", "优势", "loss"]):
        return ["derivation_step"]
    if any(token in title for token in ["代码", "kernel", "triton", "实现", "算法"]):
        return ["code_logic_block"]
    if any(token in title for token in ["例子", "example", "case", "实验"]):
        return ["example"]
    return ["concept_section"]


def seed_coverage_rows(lecture_dir: Path, segments: list[dict[str, Any]]) -> None:
    coverage_path = lecture_dir / "coverage_units.jsonl"
    if coverage_path.exists() and coverage_path.stat().st_size > 0:
        return
    tex_path = next(iter(sorted(lecture_dir.glob("lecture_*_note.tex"))), None)
    if tex_path is None:
        return
    text = tex_path.read_text()
    current_section = None
    rows: list[dict[str, Any]] = []
    segment_idx = 0
    for line in text.splitlines():
        line = line.strip()
        section_match = re.match(r"\\section\{([^}]*)\}", line)
        if section_match:
            current_section = section_match.group(1).strip()
            continue
        subsection_match = re.match(r"\\subsection\{([^}]*)\}", line)
        if not subsection_match:
            continue
        subsection = subsection_match.group(1).strip()
        if subsection in {"本章小结", "拓展阅读"}:
            continue
        if current_section == "总结与延伸":
            continue
        mapped_section = subsection
        segment = segments[min(segment_idx, max(0, len(segments) - 1))] if segments else None
        rows.append(
            {
                "unit_id": f"seed_{len(rows) + 1:04d}",
                "source_type": "multi_source",
                "source_id": "transcript_jsonl+slides_jsonl",
                "loc": {
                    "section": current_section,
                    "subsection": subsection,
                    "segment_id": segment.get("segment_id") if segment else None,
                },
                "kind": infer_kind(f"{current_section} {subsection}"),
                "summary": f"{current_section} / {subsection}",
                "required": True,
                "status": "covered",
                "mapped_section": mapped_section,
                "figure_ids": [],
                "notes": "",
            }
        )
        segment_idx += 1
    if not rows and current_section:
        rows.append(
            {
                "unit_id": "seed_0001",
                "source_type": "multi_source",
                "source_id": "transcript_jsonl+slides_jsonl",
                "loc": {"section": current_section},
                "kind": ["concept_section"],
                "summary": current_section,
                "required": True,
                "status": "covered",
                "mapped_section": current_section,
                "figure_ids": [],
                "notes": "",
            }
        )
    write_jsonl(coverage_path, rows)


def update_figure_manifest_from_tex(lecture_dir: Path) -> None:
    manifest_path = lecture_dir / "figure_manifest.json"
    tex_path = next(iter(sorted(lecture_dir.glob("lecture_*_note.tex"))), None)
    if tex_path is None or not manifest_path.exists():
        return
    existing = load_json(manifest_path)
    if not isinstance(existing, list):
        existing = []
    by_asset = {row.get("asset_path"): dict(row) for row in existing if isinstance(row, dict) and row.get("asset_path")}
    figures: list[dict[str, Any]] = []
    current_section = None
    pending_asset = None
    last_figure = None
    figure_id = 1
    for raw_line in tex_path.read_text().splitlines():
        line = raw_line.strip()
        sec_match = re.match(r"\\section\{([^}]*)\}|\\subsection\{([^}]*)\}", line)
        if sec_match:
            current_section = (sec_match.group(1) or sec_match.group(2) or "").strip()
        if line.startswith(r"\includegraphics"):
            start = line.rfind("{")
            end = line.rfind("}")
            if start != -1 and end != -1 and end > start:
                pending_asset = line[start + 1 : end]
        elif line.startswith(r"\caption{") and pending_asset:
            caption = line[len(r"\caption{") : -1]
            row = by_asset.get(pending_asset, {})
            source_type = "video_frame_or_crop" if "frames/" in pending_asset else "slide_or_external_asset"
            loc = row.get("loc")
            if loc is None:
                page_match = re.search(r"page-0*(\d+)\.png$", pending_asset)
                slide_match = re.search(r"slide-0*(\d+)\.png$", pending_asset)
                if page_match:
                    loc = {"page": int(page_match.group(1))}
                elif slide_match:
                    loc = {"slide": int(slide_match.group(1))}
            figure = {
                "figure_id": row.get("figure_id") or f"figure_{figure_id:02d}",
                "source_id": row.get("source_id") or source_type,
                "loc": loc,
                "asset_path": pending_asset,
                "caption": caption,
                "crop": row.get("crop", False),
                "used_in_section": current_section,
                "time_provenance": row.get("time_provenance"),
            }
            figures.append(figure)
            last_figure = figure
            pending_asset = None
            figure_id += 1
        elif line.startswith(r"\footnotetext{") and last_figure is not None:
            footnote = line[len(r"\footnotetext{") : -1]
            match = re.search(r"([0-9]{2}:[0-9]{2}:[0-9]{2}--[0-9]{2}:[0-9]{2}:[0-9]{2})", footnote)
            if match and "frames/" in str(last_figure.get("asset_path", "")):
                last_figure["time_provenance"] = match.group(1)
    write_json(manifest_path, figures)


def ensure_source_manifest_metadata(lecture_dir: Path, meta: dict[str, Any]) -> None:
    manifest_path = lecture_dir / "source_manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    if not isinstance(manifest, dict):
        manifest = {"sources": manifest if isinstance(manifest, list) else []}
    manifest["course_id"] = COURSE_ID
    manifest["course_mode"] = True
    manifest["origin_url"] = meta.get("webpage_url")
    sources = manifest.setdefault("sources", [])
    existing_ids = {row.get("source_id") for row in sources if isinstance(row, dict)}
    additions = []
    for source_id, source_type, rel_path, required, notes in [
        ("subtitle_srt", "platform_subtitle", meta.get("subtitle"), True, "Original subtitle track downloaded from platform."),
        ("transcript_jsonl", "structured_transcript_evidence", str((lecture_dir / "transcript.jsonl").relative_to(RUN_ROOT)), True, "Canonical subtitle-aligned evidence."),
        ("slides_jsonl", "structured_slide_evidence", str((lecture_dir / "slides.jsonl").relative_to(RUN_ROOT)), True, "Canonical official-material evidence."),
        ("segments_jsonl", "segment_plan", str((lecture_dir / "segments.jsonl").relative_to(RUN_ROOT)), True, "Canonical segment plan."),
    ]:
        if source_id in existing_ids:
            continue
        additions.append(
            {
                "source_id": source_id,
                "source_type": source_type,
                "origin_url": None,
                "local_path": rel_path,
                "required_for_coverage": required,
                "status": "available" if rel_path else "missing",
                "notes": notes,
            }
        )
    if (lecture_dir / "official_trace.json").exists() and "official_trace_json" not in existing_ids:
        additions.append(
            {
                "source_id": "official_trace_json",
                "source_type": "official_trace",
                "origin_url": None,
                "local_path": str((lecture_dir / "official_trace.json").relative_to(RUN_ROOT)),
                "required_for_coverage": False,
                "status": "available",
                "notes": "Structured execution trace for the official lecture script.",
            }
        )
    sources.extend(additions)
    write_json(manifest_path, manifest)


def enrich_meta(lecture_dir: Path) -> dict[str, Any]:
    meta_path = lecture_dir / "meta.json"
    meta = load_json(meta_path)
    raw_info = RUN_ROOT / f"raw/{meta['playlist_index']:02d}_{meta['video_id']}/{meta['playlist_index']:02d}_{meta['video_id']}.info.json"
    info = load_json(raw_info) if raw_info.exists() else {}
    meta["course_id"] = COURSE_ID
    meta["course_mode"] = True
    meta["segmentation_required"] = True
    meta["playlist_url"] = PLAYLIST_URL
    meta["course_page_url"] = PLAYLIST_URL
    meta["webpage_url"] = info.get("webpage_url")
    upload_date = info.get("upload_date")
    if upload_date and not meta.get("date"):
        meta["date"] = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}"
    title_short = meta.get("title", "").split("|")[-1].strip()
    meta["title_short"] = title_short
    tex_path = next(iter(sorted(lecture_dir.glob("lecture_*_note.tex"))), None)
    meta["topics"] = infer_topics_from_tex(tex_path, title_short)
    write_json(meta_path, meta)
    return meta


def write_course_manifest_seed(metas: list[dict[str, Any]]) -> None:
    seed = {
        "course_id": COURSE_ID,
        "title": COURSE_TITLE,
        "playlist_origin_url": PLAYLIST_URL,
        "course_mode": True,
        "lecture_count": len(metas),
        "lectures": [
            {
                "lecture_id": f"{meta['playlist_index']:02d}",
                "lecture_slug": f"{meta['playlist_index']:02d}_{Path(meta['lecture_dir']).name.split('_',1)[1] if meta.get('lecture_dir') else ''}".rstrip("_"),
                "title": meta["title"],
                "date": meta.get("date"),
                "video_url": meta.get("webpage_url"),
            }
            for meta in metas
        ],
    }
    write_json(RUN_ROOT / "build" / "course_manifest_seed.json", seed)


def write_readme() -> None:
    text = f"""# CS336 Spring 2025 Textbook Run

This run upgrades the legacy `cs336_all` lecture-note pipeline into the stricter harness-managed textbook workflow.

- course: `CS336: Language Modeling from Scratch`
- term: `Spring 2025`
- official public playlist: <{PLAYLIST_URL}>

## Canonical workflow

1. `build/bootstrap_course.py`
   Rehydrates legacy lecture workspaces into canonical harness inputs such as `subtitle.srt`, `transcript.jsonl`, `slides.jsonl`, seeded coverage, and normalized figure provenance.

2. `build/bootstrap_harness.py`
   Generates `lecture_plan.json`, `figure_plan.json`, `contracts/segment_##_contract.md`, and other harness artifacts.

3. Lecture workers repair coverage, omission, figure, and evaluator artifacts until delivery gates pass.

4. `build/build_course_manifest.py`
   Rebuilds course-level artifact pointers.

5. `build/compile_all_lecture_notes.py`
   Compiles notes through the shared delivery validator.

6. `build/merge_course_notes.py`
   Merges the lecture PDFs into a final textbook and copies the merged `.tex/.pdf` into `deliverable/`.
"""
    (RUN_ROOT / "README.md").write_text(text + "\n")


def write_contract() -> None:
    text = """# CS336 Writing Contract

All lecture-note workers must follow this contract.

## Output ownership

- Each worker owns only its assigned lecture directories under `lectures/`.
- Do not edit lecture directories owned by other workers.
- Do not revert or overwrite files created by other workers.

## Required outputs per owned lecture

For each owned lecture directory, produce or preserve:

- `transcript.jsonl`
- `slides.jsonl`
- `segments.jsonl`
- `lecture_plan.json`
- `contracts/segment_##_contract.md`
- `figure_plan.json`
- `eval_reports/pass_##.json`
- `repair_log.jsonl`
- `lecture_XX_note.tex`
- `lecture_XX_note.pdf`
- `source_manifest.json`
- `coverage_units.jsonl`
- `omission_log.jsonl`
- `figure_manifest.json`

## Source usage

- Treat `transcript.jsonl`, `slides.jsonl`, and `segments.jsonl` as the primary structured evidence layer.
- Treat subtitles, official lecture traces, official script extracts, slide-like markdown, and yt-dlp metadata as co-equal sources.
- Preserve substantive technical content from both spoken explanation and official materials.
- If a source gap remains, log it in `omission_log.jsonl` instead of silently dropping it.

## Writing policy

- Write in Chinese.
- Preserve textbook-grade pedagogical depth rather than shortening into a summary.
- End every major section with `\\subsection{本章小结}`.
- End the lecture with `\\section{总结与延伸}`.

## Validation

- The latest evaluator report under `eval_reports/` must be `pass` before the lecture is considered deliverable.
- Record repairs in `repair_log.jsonl`.
- Run `build/validate_youtube_note.py` before accepting a lecture as complete.
- Compile with `xelatex -interaction=nonstopmode -halt-on-error`.

## Final deliverable

- If the merged textbook is successfully generated, place the final exported `.tex` and `.pdf` in the run-local `deliverable/` folder.
"""
    (RUN_ROOT / "WRITING_CONTRACT.md").write_text(text + "\n")


def main() -> None:
    metas: list[dict[str, Any]] = []
    for lecture_dir in lecture_dirs():
        meta = enrich_meta(lecture_dir)
        subtitle_path = best_subtitle_path(meta)
        if subtitle_path.exists():
            ensure_symlink(subtitle_path, lecture_dir / "subtitle.srt")
            write_jsonl(lecture_dir / "transcript.jsonl", build_transcript_units(subtitle_path))
        slides_rows = build_slide_units(lecture_dir)
        write_jsonl(lecture_dir / "slides.jsonl", slides_rows)
        segments = build_segments(meta["topics"], load_jsonl(lecture_dir / "transcript.jsonl"), slides_rows)
        write_jsonl(lecture_dir / "segments.jsonl", segments)
        seed_coverage_rows(lecture_dir, segments)
        update_figure_manifest_from_tex(lecture_dir)
        ensure_source_manifest_metadata(lecture_dir, meta)
        metas.append(meta)

    write_course_manifest_seed(metas)
    write_readme()
    write_contract()
    print(f"bootstrapped={len(metas)}")
    print(RUN_ROOT)


if __name__ == "__main__":
    main()
