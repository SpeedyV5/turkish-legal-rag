"""Temporary audit helper."""
import json

r = json.load(open("outputs/evaluation/comparison_report.json", "r", encoding="utf-8"))
systems = list(r.keys())
print(f"Retrieval systems in comparison: {len(systems)}")
for s in systems:
    m = r[s]
    mrr = m.get("mrr@5", 0)
    rec = m.get("recall@5", 0)
    art_rec = m.get("article_recall@5", 0)
    print(f"  {s}: MRR@5={mrr:.4f}, R@5={rec:.4f}, Art_R@5={art_rec:.4f}")

# Find best system
best = max(systems, key=lambda s: r[s].get("mrr@5", 0))
print(f"\nBest system by MRR@5: {best}")
print(f"  MRR@5={r[best]['mrr@5']:.4f}, R@5={r[best]['recall@5']:.4f}")
