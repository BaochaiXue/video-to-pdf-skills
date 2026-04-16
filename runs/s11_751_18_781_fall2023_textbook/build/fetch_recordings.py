#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
LECTURES_DIR = RUN_ROOT / "lectures"


def lecture_dirs(selectors: list[str] | None) -> list[Path]:
    dirs = sorted(path for path in LECTURES_DIR.iterdir() if path.is_dir() and path.name[:2].isdigit())
    if not selectors:
        return dirs
    resolved: list[Path] = []
    for token in selectors:
        matched = None
        for path in dirs:
            if path.name == token or path.name.startswith(token + "_") or path.name.startswith(token):
                matched = path
                break
        if matched is None:
            raise SystemExit(f"unknown lecture selector: {token}")
        resolved.append(matched)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("lectures", nargs="*")
    args = parser.parse_args()

    for lecture_dir in lecture_dirs(args.lectures):
        meta = json.loads((lecture_dir / "meta.json").read_text())
        video_url = meta.get("video_url")
        video_id = meta.get("video_id")
        if not video_url or not video_id:
            print(f"skip {lecture_dir.name}: no public video")
            continue
        raw_dir = RUN_ROOT / "raw" / f"{meta['session_index']:02d}_{video_id}"
        raw_dir.mkdir(parents=True, exist_ok=True)
        target = raw_dir / f"{meta['session_index']:02d}_{video_id}.recording.%(ext)s"
        if list(raw_dir.glob("*.recording.*")):
            print(f"exists {lecture_dir.name}")
            continue
        subprocess.run(
            [
                "yt-dlp",
                "--no-playlist",
                "-f",
                "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best",
                "-o",
                str(target),
                video_url,
            ],
            check=True,
        )
        print(f"downloaded {lecture_dir.name}")


if __name__ == "__main__":
    main()
