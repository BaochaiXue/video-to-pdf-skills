#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")/.."
python3 build/build_course_manifest.py
python3 build/merge_book.py
cd book
xelatex -interaction=nonstopmode -halt-on-error frontmatter.tex
xelatex -interaction=nonstopmode -halt-on-error frontmatter.tex
xelatex -interaction=nonstopmode -halt-on-error appendix.tex
xelatex -interaction=nonstopmode -halt-on-error appendix.tex

python3 - <<'PY'
import json
import subprocess
from pathlib import Path

book = Path.cwd()
manifest = json.loads((book / "textbook_source_manifest.json").read_text())
inputs = [str((book / "frontmatter.pdf").resolve())]
for chapter in manifest.get("chapters", []):
    path = chapter.get("lecture_pdf")
    if not path:
        continue
    pdf = (book.parent / path).resolve()
    if pdf.exists():
        inputs.append(str(pdf))
inputs.append(str((book / "appendix.pdf").resolve()))
subprocess.run(
    [
        "gs",
        "-q",
        "-dNOPAUSE",
        "-dBATCH",
        "-sDEVICE=pdfwrite",
        "-sOutputFile=textbook.pdf",
        *inputs,
    ],
    check=True,
)
PY
cd ..
python3 build/finalize_textbook_pdf.py >/dev/null
python3 build/export_deliverables.py >/dev/null
cd book
echo "$PWD/textbook.pdf"
