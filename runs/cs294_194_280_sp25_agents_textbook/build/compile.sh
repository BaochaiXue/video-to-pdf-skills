#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")/.."
python3 build/merge_book.py
cd book
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
cp -f main.pdf textbook.pdf
mkdir -p ../deliverable/book
cp -f main.tex ../deliverable/book/cs294_194_280_sp25_agents_textbook_complete_notes.tex
cp -f textbook.pdf ../deliverable/book/cs294_194_280_sp25_agents_textbook_complete_notes.pdf
echo "$PWD/textbook.pdf"
