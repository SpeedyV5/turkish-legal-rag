from __future__ import annotations

import argparse

from src.generation.generator import LocalGenerator
from src.generation.prompt_builder import build_user_prompt


def build_retriever(system: str = "e5large_reranked_bge"):
    """Build retriever based on system name. Models are forced to CPU to leave GPU for the LLM."""
    if system == "baseline":
        # Clean dense baseline (e5-base, no keyword bonuses).
        # The legacy LegalRetriever lives in src/retrieval/legacy/ and must NOT
        # be used for evaluation or in this pipeline.
        from src.retrieval.hybrid_retriever import DenseRetriever
        retriever = DenseRetriever("configs/retrieval_config.yaml")
        retriever.model = retriever.model.to("cpu")
        return retriever

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

    if system == "e5large_reranked_bge":
        from src.retrieval.hybrid_retriever import DenseRetriever
        from src.retrieval.reranker import RerankedRetriever
        base = DenseRetriever("configs/retrieval_config_e5large.yaml")
        base.model = base.model.to("cpu")
        return RerankedRetriever(
            base,
            reranker_model="BAAI/bge-reranker-v2-m3",
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
    def __init__(self, system: str = "e5large_reranked_bge", lora_adapter: str | None = None) -> None:
        print(f"[INFO] Retrieval system: {system}")
        self.retriever = build_retriever(system)
        print("[INFO] Loading LLM generator...")
        self.generator = LocalGenerator(lora_adapter_path=lora_adapter)

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
        "--system", default="e5large_reranked_bge",
        choices=["baseline", "e5large_reranked_ml", "e5large_reranked_bge", "e5large_dense", "dense_reranked_ml"],
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--lora-adapter", default=None,
                        help="Path to LoRA adapter for fine-tuned model.")
    parser.add_argument("--question", default=None,
                        help="Single question for non-interactive mode.")
    args = parser.parse_args()

    pipeline = TurkishLegalRAGPipeline(system=args.system, lora_adapter=args.lora_adapter)

    print("\n" + "=" * 60)
    print("Turkish Legal RAG - Interactive QA")
    print(f"Retrieval: {args.system} | Top-K: {args.top_k}")
    print("=" * 60)

    def print_result(result):
        print("\n[CEVAP]\n")
        print(result["answer"])
        print("\n[KAYNAKLAR]\n")
        for i, item in enumerate(result["retrieved"][:5], 1):
            print(
                f"  [{i}] {item['title']} | {item.get('article_ref')} "
                f"| score={item['score']:.4f} | {item.get('retrieval_method', '')}"
            )

    if args.question:
        import time
        t0 = time.time()
        result = pipeline.answer(args.question, top_k=args.top_k)
        latency = time.time() - t0
        print_result(result)
        print(f"\n[Latency: {latency:.1f}s]")
        return

    while True:
        question = input("\nSoru gir ('q' ile çık): ").strip()
        if question.lower() in {"q", "quit", "exit"}:
            break
        if not question:
            continue

        print("\n[Aranıyor...]")
        result = pipeline.answer(question, top_k=args.top_k)
        print_result(result)


if __name__ == "__main__":
    main()