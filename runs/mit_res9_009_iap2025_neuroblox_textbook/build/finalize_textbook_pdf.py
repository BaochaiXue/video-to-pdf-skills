#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import tempfile

import fitz
from pypdf import PdfReader, PdfWriter
from pypdf.constants import PageLabelStyle


RUN_ROOT = Path(__file__).resolve().parents[1]
BOOK_DIR = RUN_ROOT / "book"
TEXTBOOK_PDF = BOOK_DIR / "textbook.pdf"


def stamp_visible_page_numbers(src: Path, dst: Path) -> None:
    doc = fitz.open(src)
    total_pages = len(doc)
    for idx, page in enumerate(doc, start=1):
        rect = page.rect
        box = fitz.Rect(rect.width - 120, 18, rect.width - 18, 34)
        page.draw_rect(box, color=(1, 1, 1), fill=(1, 1, 1), overlay=True)
        page.insert_textbox(
            box,
            f"Page {idx}",
            fontname="helv",
            fontsize=9,
            color=(0.15, 0.15, 0.15),
            align=fitz.TEXT_ALIGN_RIGHT,
            overlay=True,
        )
    doc.save(dst, garbage=4, deflate=True)
    doc.close()


def apply_page_labels(src: Path, dst: Path) -> None:
    reader = PdfReader(str(src))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    if reader.pages:
        writer.set_page_label(
            0,
            len(reader.pages) - 1,
            style=PageLabelStyle.DECIMAL,
            start=1,
        )
    with dst.open("wb") as fh:
        writer.write(fh)


def main() -> None:
    if not TEXTBOOK_PDF.exists():
        raise SystemExit(f"missing merged textbook pdf: {TEXTBOOK_PDF}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        stamped = tmpdir_path / "textbook_stamped.pdf"
        labeled = tmpdir_path / "textbook_labeled.pdf"
        stamp_visible_page_numbers(TEXTBOOK_PDF, stamped)
        apply_page_labels(stamped, labeled)
        labeled.replace(TEXTBOOK_PDF)

    print(TEXTBOOK_PDF)


if __name__ == "__main__":
    main()
