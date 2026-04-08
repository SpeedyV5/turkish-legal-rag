from __future__ import annotations

from pathlib import Path
import re
import json
import pandas as pd
import fitz  # PyMuPDF
import yaml


SEED_PATH = Path("data/raw/mevzuat/seed_urls.csv")
PDF_DIR = Path("data/raw/mevzuat/pdfs")


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_seed_csv(path: Path) -> pd.DataFrame:
    encodings_to_try = ["utf-8", "utf-8-sig", "cp1254", "latin-1"]
    last_error = None

    for enc in encodings_to_try:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"[INFO] Seed CSV loaded with encoding: {enc}")
            return df
        except Exception as e:
            last_error = e

    raise ValueError(f"Could not read CSV with tried encodings. Last error: {last_error}")


def extract_full_text(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    texts = []
    for page in doc:
        texts.append(page.get_text())
    doc.close()
    return "\n".join(texts)


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_summary(text: str, max_chars: int = 500) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def extract_article_refs(text: str) -> str | None:
    matches = re.findall(r"\bMadde\s+\d+\b|\bMADDE\s+\d+\b", text)
    unique_matches = sorted(set(matches))
    if not unique_matches:
        return None
    return ", ".join(unique_matches[:200])


def main() -> None:
    corpus_cfg = load_yaml("configs/corpus_config.yaml")
    registry_path = Path(corpus_cfg["output"]["registry_path"])
    sample_jsonl_path = Path(corpus_cfg["output"]["sample_jsonl_path"])

    if not registry_path.exists():
        raise FileNotFoundError(
            f"Registry file not found: {registry_path}. Run build_registry first."
        )

    seed_df = load_seed_csv(SEED_PATH)
    registry_df = pd.read_csv(registry_path)

    existing_doc_ids = set(registry_df["doc_id"].dropna().astype(str).tolist()) if len(registry_df) > 0 else set()

    new_rows = []

    for _, row in seed_df.iterrows():
        doc_id = str(row["doc_id"]).strip()

        if doc_id in existing_doc_ids:
            print(f"[SKIP] Already registered: {doc_id}")
            continue

        pdf_path = PDF_DIR / f"{doc_id}.pdf"
        if not pdf_path.exists():
            print(f"[WARN] PDF not found for {doc_id}: {pdf_path}")
            continue

        print(f"[INFO] Extracting text from {pdf_path}")
        raw_text = extract_full_text(pdf_path)
        cleaned_text = clean_text(raw_text)

        if not cleaned_text:
            print(f"[WARN] Empty text extracted for {doc_id}")
            continue

        record = {
            "doc_id": doc_id,
            "source_family": str(row.get("source_family", "mevzuat")).strip(),
            "source_name": str(row.get("source_name", "mevzuat_gov_tr")).strip(),
            "doc_type": str(row.get("doc_type", "unknown")).strip(),
            "title": str(row.get("title", "")).strip(),
            "official_no": None,
            "official_date": None,
            "url": str(row.get("url", "")).strip(),
            "language": str(row.get("language", "tr")).strip(),
            "version_status": str(row.get("version_status", "current")).strip(),
            "jurisdiction": str(row.get("jurisdiction", "tr")).strip(),
            "text": cleaned_text,
            "summary": build_summary(cleaned_text),
            "article_refs": extract_article_refs(cleaned_text),
            "court_chamber": None,
            "tags": "official_pdf,mevzuat",
        }

        new_rows.append(record)
        print(f"[OK] Registered: {doc_id} | text_len={len(cleaned_text)}")

    if not new_rows:
        print("[INFO] No new documents to register.")
        return

    new_df = pd.DataFrame(new_rows)
    updated_df = pd.concat([registry_df, new_df], ignore_index=True)
    updated_df.to_csv(registry_path, index=False, encoding="utf-8")

    sample_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with sample_jsonl_path.open("w", encoding="utf-8") as f:
        for row in updated_df.to_dict(orient="records"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[INFO] Registry updated: {registry_path}")
    print(f"[INFO] JSONL exported: {sample_jsonl_path}")
    print(f"[INFO] Added {len(new_rows)} new document(s).")


if __name__ == "__main__":
    main()