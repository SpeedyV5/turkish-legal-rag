from __future__ import annotations

from typing import Any

from sentence_transformers import CrossEncoder


class LegalReranker:
    """Cross-encoder reranker for Turkish legal retrieval."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
    ) -> None:
        print(f"[INFO] Loading cross-encoder reranker: {model_name}")
        self.model = CrossEncoder(model_name, max_length=512, device=device)
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)

        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)
            candidate["original_score"] = candidate.get("score", 0.0)
            candidate["score"] = float(score)
            candidate["retrieval_method"] = candidate.get("retrieval_method", "unknown") + "+reranked"

        reranked = sorted(candidates, key=lambda x: x["score"], reverse=True)

        if top_k is not None:
            reranked = reranked[:top_k]

        return reranked


class RerankedRetriever:
    """Wraps any retriever and applies cross-encoder reranking."""

    def __init__(
        self,
        base_retriever: Any,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        candidate_multiplier: int = 5,
        device: str | None = None,
    ) -> None:
        self.base = base_retriever
        self.reranker = LegalReranker(model_name=reranker_model, device=device)
        self.candidate_multiplier = candidate_multiplier

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        k = top_k if top_k is not None else 5
        candidate_k = k * self.candidate_multiplier

        candidates = self.base.search(query, top_k=candidate_k)
        reranked = self.reranker.rerank(query, candidates, top_k=k)

        return reranked


def main() -> None:
    from src.retrieval.hybrid_retriever import HybridRetriever

    base = HybridRetriever(fusion_method="rrf")
    retriever = RerankedRetriever(base)

    sample_queries = [
        "Cumhuriyetin nitelikleri nelerdir?",
        "Kasten öldürme suçunun cezası nedir?",
        "Boşanma sebepleri nelerdir?",
    ]

    for q in sample_queries:
        print(f"\n{'#' * 80}")
        print(f"QUERY: {q}")
        print("#" * 80)
        results = retriever.search(q, top_k=5)
        for i, item in enumerate(results, 1):
            print(
                f"  [{i}] score={item['score']:.4f} | {item['title']} "
                f"| {item.get('article_ref')} | {item['retrieval_method']}"
            )


if __name__ == "__main__":
    main()
