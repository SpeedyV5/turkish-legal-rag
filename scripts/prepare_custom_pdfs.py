"""Prepare a local PDF collection for the existing corpus build pipeline.

This is intended for evaluator-provided custom documents. It copies PDFs from
an input folder into data/raw/mevzuat/pdfs and writes a matching seed_urls.csv,
so the existing registry/chunking/embedding/index scripts can be reused.

Usage:
  python scripts/prepare_custom_pdfs.py --input-dir path/to/pdfs
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path


RAW_DIR = Path("data/raw/mevzuat")
PDF_DIR = RAW_DIR / "pdfs"
SEED_PATH = RAW_DIR / "seed_urls.csv"


def slugify(value: str) -> str:
    value = value.lower()
    value = value.replace("ı", "i").replace("ğ", "g").replace("ü", "u")
    value = value.replace("ş", "s").replace("ö", "o").replace("ç", "c")
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value or "document"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare custom PDF corpus inputs")
    parser.add_argument("--input-dir", required=True, help="Folder containing evaluator PDFs")
    parser.add_argument("--source-family", default="custom")
    parser.add_argument("--doc-type", default="custom_pdf")
    parser.add_argument("--reset", action="store_true",
                        help="Remove existing PDFs in data/raw/mevzuat/pdfs first")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in: {input_dir}")

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    if args.reset:
        for old_pdf in PDF_DIR.glob("*.pdf"):
            old_pdf.unlink()

    rows = []
    seen_ids: set[str] = set()
    for idx, pdf in enumerate(pdfs, start=1):
        base_id = slugify(pdf.stem)
        doc_id = base_id
        if doc_id in seen_ids:
            doc_id = f"{base_id}_{idx}"
        seen_ids.add(doc_id)

        target = PDF_DIR / f"{doc_id}.pdf"
        shutil.copy2(pdf, target)
        rows.append({
            "doc_id": doc_id,
            "source_family": args.source_family,
            "source_name": "local_custom_pdf",
            "doc_type": args.doc_type,
            "title": pdf.stem,
            "url": str(pdf.resolve()),
            "fallback_url": "",
            "version_status": "custom",
            "language": "tr",
            "jurisdiction": "tr",
        })
        print(f"[OK] {pdf.name} -> {target}")

    with SEED_PATH.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "doc_id", "source_family", "source_name", "doc_type", "title",
            "url", "fallback_url", "version_status", "language", "jurisdiction",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] Wrote seed file: {SEED_PATH}")
    print("[NEXT] Run corpus registry, chunking, embedding and vector-store build commands.")


if __name__ == "__main__":
    main()
