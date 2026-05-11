"""Stratified train/dev/test split of the gold benchmark.

Stratification key: (source_law, question_type, difficulty).
Within each stratum we deterministically (seeded) shuffle and assign
~60/20/20. Tiny strata (<3 items) all go to train to avoid over-thin
dev/test cells.

Outputs:
- data/benchmark/gold_benchmark_train.jsonl
- data/benchmark/gold_benchmark_dev.jsonl
- data/benchmark/gold_benchmark_test.jsonl
- data/benchmark/split_manifest.json   (which qid -> which split)
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


SEED = 13
TRAIN_RATIO = 0.60
DEV_RATIO = 0.20
# remainder = TEST_RATIO

INPUT = Path("data/benchmark/gold_benchmark.jsonl")
OUT_DIR = Path("data/benchmark")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def save_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    rows = load_jsonl(INPUT)
    print(f"[INFO] Loaded {len(rows)} questions")

    # Stratify
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["source_law"], r["question_type"], r["difficulty"])
        strata[key].append(r)

    rng = random.Random(SEED)
    train: list[dict] = []
    dev: list[dict] = []
    test: list[dict] = []

    for key, items in sorted(strata.items()):
        rng.shuffle(items)
        n = len(items)
        if n < 3:
            train.extend(items)
            continue
        n_train = max(1, int(round(n * TRAIN_RATIO)))
        n_dev = max(1, int(round(n * DEV_RATIO)))
        # Ensure at least 1 in test if n>=3
        if n_train + n_dev >= n:
            n_train = max(1, n - n_dev - 1)
        n_test = n - n_train - n_dev
        train.extend(items[:n_train])
        dev.extend(items[n_train:n_train + n_dev])
        test.extend(items[n_train + n_dev:])

    # Sanity-check uniqueness
    all_ids = [r["id"] for r in train + dev + test]
    assert len(all_ids) == len(set(all_ids)) == len(rows), \
        f"Split inconsistent: total={len(all_ids)} unique={len(set(all_ids))} rows={len(rows)}"

    # Sort by id for stability
    train.sort(key=lambda r: r["id"])
    dev.sort(key=lambda r: r["id"])
    test.sort(key=lambda r: r["id"])

    save_jsonl(train, OUT_DIR / "gold_benchmark_train.jsonl")
    save_jsonl(dev, OUT_DIR / "gold_benchmark_dev.jsonl")
    save_jsonl(test, OUT_DIR / "gold_benchmark_test.jsonl")

    manifest = {
        "seed": SEED,
        "ratios": {"train": TRAIN_RATIO, "dev": DEV_RATIO, "test": round(1 - TRAIN_RATIO - DEV_RATIO, 4)},
        "counts": {"train": len(train), "dev": len(dev), "test": len(test), "total": len(rows)},
        "splits": {
            "train": [r["id"] for r in train],
            "dev": [r["id"] for r in dev],
            "test": [r["id"] for r in test],
        },
    }
    (OUT_DIR / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    # Print stratum-level breakdown
    print(f"[OK] train={len(train)} dev={len(dev)} test={len(test)}")
    by_split = {"train": train, "dev": dev, "test": test}
    print("\nBreakdown by question_type:")
    for split, items in by_split.items():
        cnt: dict[str, int] = defaultdict(int)
        for r in items:
            cnt[r["question_type"]] += 1
        print(f"  {split:>5}: " + ", ".join(f"{k}={v}" for k, v in sorted(cnt.items())))

    print("\nBreakdown by difficulty:")
    for split, items in by_split.items():
        cnt = defaultdict(int)
        for r in items:
            cnt[r["difficulty"]] += 1
        print(f"  {split:>5}: " + ", ".join(f"{k}={v}" for k, v in sorted(cnt.items())))

    print("\nBreakdown by source_law:")
    for split, items in by_split.items():
        cnt = defaultdict(int)
        for r in items:
            cnt[r["source_law"]] += 1
        print(f"  {split:>5}: " + ", ".join(f"{k}={v}" for k, v in sorted(cnt.items())))


if __name__ == "__main__":
    main()
