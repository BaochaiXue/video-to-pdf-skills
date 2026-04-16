#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")/.."
python3 build/build_course_manifest.py
python3 build/validate_textbook.py
python3 build/merge_course_notes.py
python3 build/export_deliverables.py
echo "$PWD/deliverable/book/speech_recognition_understanding_fall2023_textbook.pdf"
