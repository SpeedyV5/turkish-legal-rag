from __future__ import annotations

from pathlib import Path
import fitz  # PyMuPDF


PDF_DIR = Path("data/raw/mevzuat/pdfs")


def extract_preview(pdf_path: Path, max_pages: int = 3) -> str:
    doc = fitz.open(pdf_path)
    texts = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        texts.append(page.get_text())
    doc.close()
    return "\n".join(texts)


def main() -> None:
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {PDF_DIR}")

    for pdf_path in pdf_files:
        print(f"[INFO] Inspecting: {pdf_path.name}")
        preview = extract_preview(pdf_path)
        print(preview[:2000])
        print("=" * 80)


if __name__ == "__main__":
    main()