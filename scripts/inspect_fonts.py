"""
Print every block on one page with the fonts it uses.

A diagnostic, not part of the pipeline. Detection rules should be built
from what real papers actually contain, not from a guess about what they
might contain — this is how you look.

    python scripts/inspect_fonts.py "path/to/paper.pdf" 2

The page number is 0-based, so 2 is the third page. Leave it off for
page 0.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pymupdf


def inspect(pdf_path: Path, page_index: int) -> None:
    doc = pymupdf.open(pdf_path)
    try:
        if page_index >= doc.page_count:
            print(f"This PDF has {doc.page_count} pages (0–{doc.page_count - 1}).")
            return

        page = doc[page_index]
        page_dict = page.get_text("dict")

        print(f"{pdf_path.name} — page {page_index} of {doc.page_count}")
        print(f"page size: {page_dict['width']:.0f} x {page_dict['height']:.0f} points")
        print("=" * 78)

        all_fonts: Counter[str] = Counter()

        for i, block in enumerate(page_dict.get("blocks", [])):
            x0, y0, x1, y1 = block["bbox"]
            if block.get("type") == 1:
                print(f"\n[{i:>3}] IMAGE  {x1 - x0:.0f}x{y1 - y0:.0f}pt  at ({x0:.0f},{y0:.0f})")
                continue

            fonts: Counter[str] = Counter()
            text_parts: list[str] = []
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if text.strip():
                        key = f"{span.get('font', '?')} {span.get('size', 0):.1f}pt"
                        fonts[key] += len(text)
                        all_fonts[span.get("font", "?")] += len(text)
                    text_parts.append(text)

            text = "".join(text_parts).strip().replace("\n", " ")
            if not text:
                continue

            print(f"\n[{i:>3}] {x1 - x0:>5.0f}x{y1 - y0:<4.0f}pt at ({x0:>5.0f},{y0:>5.0f})")
            print(f"      fonts: {', '.join(f'{k} x{v}' for k, v in fonts.most_common(4))}")
            print(f"      text : {text[:90]}")

        print("\n" + "=" * 78)
        print("fonts on this page, by how many characters use them:")
        for font, count in all_fonts.most_common():
            print(f"   {count:>6}  {font}")

        vectors = page.get_drawings()
        print(f"\nvector drawings on this page: {len(vectors)}")
    finally:
        doc.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    inspect(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 0)
