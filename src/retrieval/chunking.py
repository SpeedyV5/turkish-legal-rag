from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ARTICLE_PATTERN = re.compile(
    r"(?=(?:^|\n)(MADDE\s+\d+[-–]?[A-Za-z0-9()\-]*|Madde\s+\d+[-–]?[A-Za-z0-9()\-]*))",
    re.MULTILINE,
)


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_by_article(text: str) -> list[tuple[str | None, str]]:
    matches = list(ARTICLE_PATTERN.finditer(text))

    if not matches:
        return [(None, text.strip())]

    sections: list[tuple[str | None, str]] = []

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk_text = text[start:end].strip()

        header_match = re.search(r"(MADDE\s+\d+[-–]?[A-Za-z0-9()\-]*|Madde\s+\d+[-–]?[A-Za-z0-9()\-]*)", chunk_text)
        article_no = header_match.group(1) if header_match else None

        sections.append((article_no, chunk_text))

    return sections


def split_long_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + max_chars, text_len)
        chunk = text[start:end]

        if end < text_len:
            last_break = max(
                chunk.rfind("\n"),
                chunk.rfind(". "),
                chunk.rfind("; "),
            )
            if last_break > max_chars // 2:
                end = start + last_break + 1
                chunk = text[start:end]

        chunks.append(chunk.strip())

        if end >= text_len:
            break

        start = max(0, end - overlap_chars)

    return chunks


def build_chunks_for_document(
    row: pd.Series,
    min_chunk_chars: int,
    max_chunk_chars: int,
    overlap_chars: int,
) -> list[dict[str, Any]]:
    doc_id = str(row["doc_id"])
    title = str(row["title"])
    doc_type = str(row["doc_type"])
    source_family = str(row["source_family"])
    source_name = str(row["source_name"])
    url = str(row["url"]) if pd.notna(row["url"]) else None
    full_text = str(row["text"])

    article_sections = split_by_article(full_text)

    chunks: list[dict[str, Any]] = []
    chunk_counter = 0

    for article_ref, article_text in article_sections:
        article_text = article_text.strip()
        if not article_text:
            continue

        if len(article_text) < min_chunk_chars:
            sub_chunks = [article_text]
        else:
            sub_chunks = split_long_text(
                text=article_text,
                max_chars=max_chunk_chars,
                overlap_chars=overlap_chars,
            )

        for sub_idx, sub_chunk in enumerate(sub_chunks):
            sub_chunk = sub_chunk.strip()
            if not sub_chunk:
                continue

            chunk_id = f"{doc_id}_chunk_{chunk_counter}"
            chunk_counter += 1

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "title": title,
                    "doc_type": doc_type,
                    "source_family": source_family,
                    "source_name": source_name,
                    "url": url,
                    "article_ref": article_ref,
                    "subchunk_index": sub_idx,
                    "text": sub_chunk,
                    "text_len": len(sub_chunk),
                }
            )

    return chunks


def save_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    cfg = load_yaml("configs/retrieval_config.yaml")

    registry_path = Path(cfg["input"]["registry_path"])
    chunks_path = Path(cfg["output"]["chunks_path"])

    min_chunk_chars = int(cfg["chunking"]["min_chunk_chars"])
    max_chunk_chars = int(cfg["chunking"]["max_chunk_chars"])
    overlap_chars = int(cfg["chunking"]["overlap_chars"])

    if not registry_path.exists():
        raise FileNotFoundError(f"Registry not found: {registry_path}")

    df = pd.read_csv(registry_path)

    all_chunks: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        doc_chunks = build_chunks_for_document(
            row=row,
            min_chunk_chars=min_chunk_chars,
            max_chunk_chars=max_chunk_chars,
            overlap_chars=overlap_chars,
        )
        all_chunks.extend(doc_chunks)
        print(f"[INFO] {row['doc_id']} -> {len(doc_chunks)} chunk(s)")

    save_jsonl(all_chunks, chunks_path)

    print(f"[INFO] Total chunks: {len(all_chunks)}")
    print(f"[INFO] Saved to: {chunks_path}")


if __name__ == "__main__":
    main()