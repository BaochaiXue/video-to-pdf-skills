from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable


DEFAULT_THRESHOLDS = {
    "coverage_completeness": 0.90,
    "pedagogical_depth": 0.80,
    "derivation_fidelity": 0.80,
    "code_fidelity": 0.80,
    "figure_usefulness": 0.80,
    "coherence": 0.85,
    "hallucination_control": 0.90,
}

ALLOWED_COVERAGE_STATUSES = {"covered", "partial", "duplicate", "omitted", "unclassified"}


def load_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    data = list(rows)
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in data)
    path.write_text(payload + ("\n" if payload else ""))


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


def build_transcript_units(subtitle_path: Path, source_id: str = "subtitle_srt") -> list[dict[str, Any]]:
    entries = parse_srt_entries(subtitle_path)
    return [
        {
            "unit_id": f"sub_{idx:04d}",
            "source_type": "subtitle_span",
            "source_id": source_id,
            "loc": {"start": entry["start"], "end": entry["end"]},
            "text": entry["text"],
            "required": True,
        }
        for idx, entry in enumerate(entries, start=1)
    ]


def extract_slide_pages(pdf_path: Path) -> list[str]:
    proc = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    pages = proc.stdout.replace("\r\n", "\n").split("\f")
    cleaned: list[str] = []
    for page in pages:
        page = re.sub(r"[ \t]+\n", "\n", page)
        page = re.sub(r"\n{3,}", "\n\n", page.strip())
        cleaned.append(page)
    while cleaned and not cleaned[-1]:
        cleaned.pop()
    return cleaned


def build_slide_units(slide_pages: list[str], lecture_dir: Path, source_id: str = "slides_pdf") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, page_text in enumerate(slide_pages, start=1):
        rows.append(
            {
                "unit_id": f"slide_{idx:04d}",
                "source_type": "slide_page",
                "source_id": source_id,
                "loc": {"page": idx},
                "text": page_text,
                "asset_path": f"pdf_pages/page-{idx:02d}.png",
                "required": bool(page_text.strip()),
            }
        )
    return rows


def slice_evenly(items: list[Any], parts: int) -> list[list[Any]]:
    if parts <= 0:
        return [items]
    if not items:
        return [[] for _ in range(parts)]
    chunk_size = math.ceil(len(items) / parts)
    chunks = [items[idx : idx + chunk_size] for idx in range(0, len(items), chunk_size)]
    while len(chunks) < parts:
        chunks.append([])
    return chunks[:parts]


def infer_segment_count(topic_count: int, transcript_count: int) -> int:
    return max(1, topic_count, math.ceil(max(1, transcript_count) / 150))


def build_segments(
    topic_hints: list[str],
    transcript_units: list[dict[str, Any]],
    slide_units: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    segment_count = infer_segment_count(len(topic_hints), len(transcript_units))
    transcript_chunks = slice_evenly(transcript_units, segment_count)
    slide_chunks = slice_evenly([unit for unit in slide_units if unit.get("required")], segment_count)
    segments: list[dict[str, Any]] = []
    for idx in range(segment_count):
        transcript_chunk = transcript_chunks[idx] if idx < len(transcript_chunks) else []
        slide_chunk = slide_chunks[idx] if idx < len(slide_chunks) else []
        source_unit_ids = [unit["unit_id"] for unit in transcript_chunk] + [unit["unit_id"] for unit in slide_chunk]
        start = transcript_chunk[0]["loc"]["start"] if transcript_chunk else None
        end = transcript_chunk[-1]["loc"]["end"] if transcript_chunk else None
        target_hint = topic_hints[idx] if idx < len(topic_hints) else f"Segment {idx + 1}"
        segments.append(
            {
                "segment_id": f"seg_{idx + 1:02d}",
                "start": start,
                "end": end,
                "source_unit_ids": source_unit_ids,
                "target_section_hint": target_hint,
            }
        )
    return segments


def detect_kind_hints(coverage_rows: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for row in coverage_rows:
        kinds = row.get("kind") or []
        if isinstance(kinds, str):
            kinds = [kinds]
        for kind in kinds:
            if kind and kind not in values:
                values.append(kind)
    return values


def classify_figure_provenance(asset_path: str | None) -> str:
    asset = asset_path or ""
    if "frames/" in asset:
        return "video_frame"
    if asset.endswith(".png") or asset.endswith(".jpg") or asset.endswith(".jpeg"):
        return "slide_or_image_asset"
    return "unknown"


def parse_tex_figure_assets(tex_path: Path) -> list[str]:
    text = tex_path.read_text()
    assets = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)
    return [
        asset
        for asset in assets
        if asset
        and not asset.startswith("\\")
        and asset != "cover.jpg"
    ]


def find_latest_eval_report(lecture_dir: Path) -> Path | None:
    reports = sorted((lecture_dir / "eval_reports").glob("pass_*.json"))
    return reports[-1] if reports else None


def infer_formula_units(coverage_rows: list[dict[str, Any]]) -> list[str]:
    hits: list[str] = []
    pattern = re.compile(r"formula|derivation|equation|bellman|policy[_ -]?gradient|value[_ -]?function", re.I)
    for row in coverage_rows:
        text = " ".join([str(row.get("unit_type", "")), " ".join(row.get("kind") or [])])
        if pattern.search(text):
            hits.append(str(row.get("unit_id")))
    return hits


def infer_code_units(coverage_rows: list[dict[str, Any]]) -> list[str]:
    hits: list[str] = []
    pattern = re.compile(r"code|source[_ -]?code|pseudocode|implementation_example|kernel_code|代码|伪代码", re.I)
    for row in coverage_rows:
        text = " ".join([str(row.get("unit_type", "")), " ".join(row.get("kind") or [])])
        if pattern.search(text):
            hits.append(str(row.get("unit_id")))
    return hits


def bool_path(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0
