from __future__ import annotations

from typing import Any


def format_context(results: list[dict[str, Any]], max_chars_per_chunk: int = 350) -> str:
    parts = []

    for i, item in enumerate(results, start=1):
        title = item.get("title", "")
        article_ref = item.get("article_ref", "")
        doc_id = item.get("doc_id", "")
        text = item.get("text", "").strip()[:max_chars_per_chunk]

        block = (
            f"[Kaynak {i}]\n"
            f"Belge: {title}\n"
            f"Belge ID: {doc_id}\n"
            f"Madde: {article_ref}\n"
            f"Metin:\n{text}\n"
        )
        parts.append(block)

    return "\n\n".join(parts)


def detect_question_type(question: str) -> str:
    q = question.lower()

    if any(x in q for x in ["nelerdir", "hangileridir", "şartları nelerdir", "sebepleri nelerdir"]):
        return "list"
    if any(x in q for x in ["nedir", "ne demektir", "tanımı"]):
        return "definition"
    return "general"


def build_user_prompt(question: str, results: list[dict[str, Any]]) -> str:
    context = format_context(results)
    qtype = detect_question_type(question)

    if qtype == "list":
        task_instruction = (
            "Bu bir listeleme sorusudur. "
            "Bağlamdaki ilgili tüm maddeleri mümkün olduğunca birlikte kullan. "
            "Eksik liste verme. "
            "Cevabı madde madde yaz."
        )
    elif qtype == "definition":
        task_instruction = (
            "Bu bir tanım sorusudur. "
            "En ilgili maddeyi merkeze alarak kısa ve net cevap ver."
        )
    else:
        task_instruction = (
            "Bağlamdaki en ilgili parçaları kullanarak kısa ve net cevap ver."
        )

    return f"""Aşağıda Türk hukuk metinlerinden getirilen bağlam parçaları verilmiştir.

Bağlam:
{context}

Soru:
{question}

Görev:
- Soruyu yalnızca verilen bağlama dayanarak cevapla.
- {task_instruction}
- Bağlamda olmayan bilgi ekleme.
- Cevabın sonunda 'Dayanak:' satırı aç ve kullandığın madde numaralarını yaz.
"""