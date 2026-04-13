from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.benchmark.gold_questions import QUESTIONS


OUTPUT_PATH = Path("data/benchmark/gold_benchmark.jsonl")


def validate_questions(questions: list[dict]) -> None:
    ids = [q["id"] for q in questions]
    dupes = [qid for qid, count in Counter(ids).items() if count > 1]
    if dupes:
        raise ValueError(f"Duplicate question IDs found: {dupes}")

    required_fields = {
        "id", "question", "question_type", "expected_answer",
        "relevant_doc_ids", "relevant_articles", "difficulty", "source_law",
    }
    for q in questions:
        missing = required_fields - set(q.keys())
        if missing:
            raise ValueError(f"Question {q.get('id', '?')} missing fields: {missing}")

    print(f"[INFO] Validation passed: {len(questions)} questions")


def print_summary(questions: list[dict]) -> None:
    by_law = Counter(q["source_law"] for q in questions)
    by_type = Counter(q["question_type"] for q in questions)
    by_diff = Counter(q["difficulty"] for q in questions)

    print("\n=== Benchmark Summary ===")
    print(f"Total questions: {len(questions)}")

    print("\nBy source law:")
    for law, count in by_law.most_common():
        print(f"  {law}: {count}")

    print("\nBy question type:")
    for qtype, count in by_type.most_common():
        print(f"  {qtype}: {count}")

    print("\nBy difficulty:")
    for diff, count in by_diff.most_common():
        print(f"  {diff}: {count}")


def main() -> None:
    validate_questions(QUESTIONS)
    print_summary(QUESTIONS)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for q in QUESTIONS:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"\n[INFO] Benchmark saved to: {OUTPUT_PATH}")
    print(f"[INFO] Total: {len(QUESTIONS)} questions")


if __name__ == "__main__":
    main()
