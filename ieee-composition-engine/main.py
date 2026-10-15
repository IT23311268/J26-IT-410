"""
main.py - the command you run.

Right now it runs stage 1 only: read a document and show what came out.
As we add stages during the week, this file grows to run the whole pipeline.

Usage:
    python main.py data/input/sample_paper.pdf
    python main.py data/input/sample_paper.pdf --preview 40
"""

from __future__ import annotations

import argparse
import os
import sys

from src.ingest import load_document


def print_summary(doc) -> None:
    print()
    print("=" * 78)
    print("STAGE 1 - INGEST")
    print("=" * 78)
    print(f"  File            : {doc.source_path}")
    print(f"  Format          : {doc.source_format}")
    print(f"  Pages           : {doc.page_count}")
    print(f"  Lines extracted : {len(doc.blocks)}")
    print(f"  Body font size  : {doc.body_font_size} pt   "
          f"(the most common size - our baseline for spotting headings)")
    print()


def print_blocks(doc, limit: int) -> None:
    print(f"  First {min(limit, len(doc.blocks))} lines, with their layout clues:")
    print()
    print(f"  {'#':>3}  {'pg':>2}  {'size':>5}  {'style':<6}  text")
    print(f"  {'-'*3}  {'-'*2}  {'-'*5}  {'-'*6}  {'-'*46}")

    for index, block in enumerate(doc.blocks[:limit], start=1):
        style = ("B" if block.bold else "") + ("I" if block.italic else "")
        style = style or "-"
        text = block.text if len(block.text) <= 46 else block.text[:43] + "..."
        print(f"  {index:>3}  {block.page:>2}  {block.font_size:>5}  {style:<6}  {text}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="IEEE Manuscript Composition Engine - stage 1 (ingest)"
    )
    parser.add_argument("input_file", help="path to a .pdf or .docx file")
    parser.add_argument("--preview", type=int, default=25,
                        help="how many extracted lines to print (default 25)")
    parser.add_argument("--outdir", default="data/output",
                        help="where to write the JSON result")
    args = parser.parse_args()

    try:
        doc = load_document(args.input_file)
    except (FileNotFoundError, ValueError) as error:
        print(f"\n  ERROR: {error}\n")
        return 1

    print_summary(doc)
    print_blocks(doc, args.preview)

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input_file))[0]
    json_path = os.path.join(args.outdir, f"{stem}_stage1_blocks.json")
    doc.to_json(json_path)
    print(f"  Full result written to: {json_path}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
