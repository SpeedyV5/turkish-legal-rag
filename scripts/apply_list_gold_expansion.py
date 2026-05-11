"""Apply manually-reviewed list-gold expansions to the benchmark.

Workflow:
1. Run scripts/suggest_list_gold_expansion.py to generate candidates.
2. MANUALLY review data/benchmark/list_gold_expansion_candidates.jsonl
   and for each row, edit the `suggested_extra_articles` field to keep
   only the articles you confirm as actually relevant
   (delete the ones that are noise).
3. Run this script. It merges the (edited) suggestions into the gold
   benchmark and writes data/benchmark/gold_benchmark_v2.jsonl plus
   re-stratified train/dev/test splits.

We never silently drop anything: original gold articles are always kept;
extras are appended.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


CANDIDATES = Path("data/benchmark/list_gold_expansion_candidates.jsonl")
ORIGINAL = Path("data/benchmark/gold_benchmark.jsonl")
OUT = Path("data/benchmark/gold_benchmark_v2.jsonl")


def normalize_num(ref: str | None) -> str | None:
    if not ref:
        return None
    m = re.search(r"(\d+)", ref)
    return m.group(1) if m else None


def main() -> None:
    if not CANDIDATES.exists():
        print(f"[ERROR] {CANDIDATES} not found. Run suggest_list_gold_expansion.py first.")
        sys.exit(1)

    rows = [json.loads(l) for l in ORIGINAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    cands = [json.loads(l) for l in CANDIDATES.read_text(encoding="utf-8").splitlines() if l.strip()]
    cand_by_id = {c["id"]: c for c in cands}

    n_extended = 0
    n_extra_articles = 0
    out_rows = []

    for r in rows:
        cand = cand_by_id.get(r["id"])
        if not cand or not cand.get("suggested_extra_articles"):
            out_rows.append(r)
            continue

        existing_nums = {normalize_num(a) for a in r["relevant_articles"]}
        existing_nums.discard(None)

        merged = list(r["relevant_articles"])
        for art in cand["suggested_extra_articles"]:
            if normalize_num(art) and normalize_num(art) not in existing_nums:
                merged.append(art)
                existing_nums.add(normalize_num(art))
                n_extra_articles += 1

        new_r = dict(r)
        new_r["relevant_articles"] = merged
        if len(merged) != len(r["relevant_articles"]):
            n_extended += 1
        out_rows.append(new_r)

    with OUT.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[OK] Wrote {OUT}")
    print(f"[OK] Extended {n_extended} questions with {n_extra_articles} extra gold articles")
    print(f"[NEXT] Re-run scripts/split_benchmark.py with INPUT pointed at {OUT.name} "
          f"if you want to refresh splits, OR keep the current split and just swap the gold file.")


if __name__ == "__main__":
    main()
