from __future__ import annotations

import argparse

from src.generation.generator import LocalGenerator
from src.generation.prompt_builder import build_user_prompt


def build_retriever(system: str = "e5large_reranked_ml"):
    """Build retriever based on system name. Models are forced to CPU to leave GPU for the LLM."""
    if system == "baseline":
        from src.retrieval.retriever import LegalRetriever
        return LegalRetriever()

    if system == "e5large_reranked_ml":
        from src.retrieval.hybrid_retriever import DenseRetriever
        from src.retrieval.reranker import RerankedRetriever
        base = DenseRetriever("configs/retrieval_config_e5large.yaml")
        base.model = base.model.to("cpu")
        return RerankedRetriever(
            base,
            reranker_model="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            device="cpu",
        )

    if system == "e5large_dense":
        from src.retrieval.hybrid_retriever import DenseRetriever
        retriever = DenseRetriever("configs/retrieval_config_e5large.yaml")
        retriever.model = retriever.model.to("cpu")
        return retriever

    if system == "dense_reranked_ml":
        from src.retrieval.hybrid_retriever import DenseRetriever
        from src.retrieval.reranker import RerankedRetriever
        base = DenseRetriever("configs/retrieval_config.yaml")
        base.model = base.model.to("cpu")
        return RerankedRetriever(
            base,
            reranker_model="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            device="cpu",
        )

    raise ValueError(f"Unknown system: {system}")


class TurkishLegalRAGPipeline:
    def __init__(self, system: str = "e5large_reranked_ml") -> None:
        print(f"[INFO] Retrieval system: {system}")
        self.retriever = build_retriever(system)
        print("[INFO] Loading LLM generator...")
        self.generator = LocalGenerator()

    def answer(self, question: str, top_k: int = 5) -> dict:
        retrieved = self.retriever.search(question, top_k=top_k)
        prompt = build_user_prompt(question, retrieved)
        answer = self.generator.generate(prompt)

        return {
            "question": question,
            "retrieved": retrieved,
            "answer": answer,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--system", default="e5large_reranked_ml",
        choices=["baseline", "e5large_reranked_ml", "e5large_dense", "dense_reranked_ml"],
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    pipeline = TurkishLegalRAGPipeline(system=args.system)

    print("\n" + "=" * 60)
    print("Turkish Legal RAG - Interactive QA")
    print(f"Retrieval: {args.system} | Top-K: {args.top_k}")
    print("=" * 60)

    while True:
        question = input("\nSoru gir ('q' ile çık): ").strip()
        if question.lower() in {"q", "quit", "exit"}:
            break
        if not question:
            continue

        print("\n[Aranıyor...]")
        result = pipeline.answer(question, top_k=args.top_k)

        print("\n[CEVAP]\n")
        print(result["answer"])

        print("\n[KAYNAKLAR]\n")
        for i, item in enumerate(result["retrieved"][:5], 1):
            print(
                f"  [{i}] {item['title']} | {item.get('article_ref')} "
                f"| score={item['score']:.4f} | {item.get('retrieval_method', '')}"
            )


if __name__ == "__main__":
    main()