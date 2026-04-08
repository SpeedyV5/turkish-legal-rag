from __future__ import annotations

from src.generation.generator import LocalGenerator
from src.generation.prompt_builder import build_user_prompt
from src.retrieval.retriever import LegalRetriever


class TurkishLegalRAGPipeline:
    def __init__(self) -> None:
        self.retriever = LegalRetriever()
        self.generator = LocalGenerator()

    def answer(self, question: str, top_k: int = 3) -> dict:
        retrieved = self.retriever.search(question, top_k=top_k)
        prompt = build_user_prompt(question, retrieved)
        answer = self.generator.generate(prompt)

        return {
            "question": question,
            "retrieved": retrieved,
            "answer": answer,
        }


def main() -> None:
    pipeline = TurkishLegalRAGPipeline()

    while True:
        question = input("\nSoru gir ('q' ile çık): ").strip()
        if question.lower() in {"q", "quit", "exit"}:
            break
        if not question:
            continue

        result = pipeline.answer(question, top_k=3)

        print("\n[CEVAP]\n")
        print(result["answer"])

        print("\n[KAYNAKLAR]\n")
        for item in result["retrieved"][:3]:
            print(f"- {item['title']} | {item.get('article_ref')} | score={item['score']:.4f}")


if __name__ == "__main__":
    main()