#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.video_note_harness.common import find_latest_eval_report, load_json


def lecture_dirs(run_root: Path) -> list[Path]:
    return sorted(path for path in (run_root / "lectures").iterdir() if path.is_dir() and path.name[:2].isdigit())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = (REPO_ROOT / args.run_root).resolve()

    counter: Counter[str] = Counter()
    for lecture_dir in lecture_dirs(run_root):
        report_path = find_latest_eval_report(lecture_dir)
        if not report_path:
            continue
        report = load_json(report_path)
        for issue in report.get("blocking_issues", []):
            if isinstance(issue, dict):
                counter[str(issue.get("type", "unknown"))] += 1

    for issue_type, count in counter.most_common():
        print(f"{issue_type}\t{count}")


if __name__ == "__main__":
    main()
