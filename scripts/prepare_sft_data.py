"""Prepare supervised fine-tuning (SFT) data from the train split.

For each question in the train benchmark:
1. Look up gold-relevant chunks from the corpus (by doc_id + article_ref).
2. Build the user prompt using the same prompt_builder as inference.
3. Construct the gold assistant response: expected_answer + "Dayanak: Madde X, Madde Y".
4. Emit a chat-format record: [system, user, assistant].

Outputs:
- data/sft/sft_train.jsonl            -- one JSON per line, chat messages format
- data/sft/sft_train_sharegpt.jsonl   -- ShareGPT-style for axolotl / LLaMA-Factory
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.generation.prompt_builder import build_user_prompt


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_article_ref(ref: str) -> str:
    """Normalize article ref for matching: lowercase, strip dashes, collapse spaces."""
    r = ref.strip().lower()
    r = re.sub(r"[\-–—]+$", "", r)  # trailing dash/hyphen
    r = re.sub(r"\s+", " ", r).strip()
    return r


def build_corpus_index(corpus_path: str | Path) -> dict[str, list[dict]]:
    """Index chunks by (doc_id, normalized_article_ref) for fast lookup."""
    idx: dict[str, list[dict]] = defaultdict(list)
    for chunk in load_jsonl(corpus_path):
        art = chunk.get("article_ref", "")
        key = f"{chunk['doc_id']}||{normalize_article_ref(art)}"
        idx[key].append(chunk)
    return idx


def find_gold_chunks(
    q: dict,
    corpus_idx: dict[str, list[dict]],
    max_chunks: int = 5,
) -> list[dict]:
    """Return corpus chunks that match the question's gold doc_id + article_ref pairs."""
    gold_docs = q.get("relevant_doc_ids", [])
    gold_arts = q.get("relevant_articles", [])

    # Build (doc_id, article_ref) pairs
    pairs = []
    if len(gold_docs) == 1 and len(gold_arts) >= 1:
        # Single doc, multiple articles
        for art in gold_arts:
            pairs.append((gold_docs[0], art))
    elif len(gold_docs) == len(gold_arts):
        pairs = list(zip(gold_docs, gold_arts))
    else:
        # Fallback: cartesian product
        for doc in gold_docs:
            for art in gold_arts:
                pairs.append((doc, art))

    chunks = []
    seen = set()
    for doc_id, art_ref in pairs:
        key = f"{doc_id}||{normalize_article_ref(art_ref)}"
        for c in corpus_idx.get(key, []):
            cid = c["chunk_id"]
            if cid not in seen:
                seen.add(cid)
                chunks.append(c)

    # Fallback: partial match on normalized article_ref
    if not chunks:
        for doc_id, art_ref in pairs:
            norm = normalize_article_ref(art_ref)
            for key, clist in corpus_idx.items():
                if key.startswith(f"{doc_id}||") and norm in key:
                    for c in clist:
                        cid = c["chunk_id"]
                        if cid not in seen:
                            seen.add(cid)
                            chunks.append(c)

    return chunks[:max_chunks]


def build_gold_answer(q: dict) -> str:
    """Build the ideal assistant response: answer + Dayanak line."""
    answer = q["expected_answer"].strip()
    articles = q.get("relevant_articles", [])

    # Normalize article references to "Madde N" format
    refs = []
    for art in articles:
        art = art.strip()
        if art and art not in refs:
            refs.append(art)

    dayanak = ", ".join(refs) if refs else ""
    if dayanak:
        return f"{answer}\nDayanak: {dayanak}"
    return answer


def build_system_prompt(config_path: str = "configs/generation_config.yaml") -> str:
    """Read system prompt from generation config."""
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["prompting"]["system_prompt"].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SFT data from train split")
    parser.add_argument(
        "--train-path",
        default="data/benchmark/gold_benchmark_train.jsonl",
    )
    parser.add_argument(
        "--corpus-path",
        default="data/processed/corpus/chunk_metadata.jsonl",
    )
    parser.add_argument(
        "--generation-config",
        default="configs/generation_config.yaml",
    )
    parser.add_argument("--output-dir", default="data/sft")
    parser.add_argument(
        "--max-context-chunks",
        type=int,
        default=5,
        help="Max gold chunks to include as context.",
    )
    args = parser.parse_args()

    print("[INFO] Loading train split...")
    train = load_jsonl(args.train_path)
    print(f"  {len(train)} questions")

    print("[INFO] Loading corpus index...")
    corpus_idx = build_corpus_index(args.corpus_path)
    print(f"  {len(corpus_idx)} unique (doc_id, article_ref) keys")

    system_prompt = build_system_prompt(args.generation_config)
    print(f"[INFO] System prompt: {system_prompt[:80]}...")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chat_rows = []
    sharegpt_rows = []
    skipped = 0

    for q in train:
        gold_chunks = find_gold_chunks(q, corpus_idx, max_chunks=args.max_context_chunks)
        if not gold_chunks:
            print(f"  [WARN] No gold chunks found for {q['id']} "
                  f"(docs={q.get('relevant_doc_ids')}, arts={q.get('relevant_articles')})")
            skipped += 1
            continue

        user_prompt = build_user_prompt(q["question"], gold_chunks)
        assistant_response = build_gold_answer(q)

        # Chat format (OpenAI-style)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ]
        chat_rows.append({
            "id": q["id"],
            "messages": messages,
            "question_type": q["question_type"],
            "difficulty": q["difficulty"],
            "source_law": q["source_law"],
            "n_gold_chunks": len(gold_chunks),
        })

        # ShareGPT format (for axolotl / LLaMA-Factory)
        sharegpt_rows.append({
            "id": q["id"],
            "system": system_prompt,
            "conversations": [
                {"from": "human", "value": user_prompt},
                {"from": "gpt", "value": assistant_response},
            ],
        })

    # Save chat format
    chat_path = output_dir / "sft_train.jsonl"
    with open(chat_path, "w", encoding="utf-8") as f:
        for row in chat_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Save ShareGPT format
    sgpt_path = output_dir / "sft_train_sharegpt.jsonl"
    with open(sgpt_path, "w", encoding="utf-8") as f:
        for row in sharegpt_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n[DONE] SFT data prepared:")
    print(f"  Total questions:  {len(train)}")
    print(f"  Usable examples:  {len(chat_rows)}")
    print(f"  Skipped (no gold): {skipped}")
    print(f"  Chat format:      {chat_path}")
    print(f"  ShareGPT format:  {sgpt_path}")

    # Quick stats
    from collections import Counter
    types = Counter(r["question_type"] for r in chat_rows)
    diffs = Counter(r["difficulty"] for r in chat_rows)
    print(f"\n  By type:       {dict(types)}")
    print(f"  By difficulty:  {dict(diffs)}")


if __name__ == "__main__":
    main()
