from __future__ import annotations

import re
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


INSUFFICIENT_CONTEXT_FALLBACK = "Bağlamda yeterli bilgi yoktur."


def is_penalty_question(question: str) -> bool:
    q = question.lower()
    penalty_terms = ["cezası", "hapis", "adli para", "yaptırım"]
    return any(term in q for term in penalty_terms)


def _article_num(ref: str | None) -> int | None:
    if not ref:
        return None
    match = re.search(r"(\d+)", str(ref))
    return int(match.group(1)) if match else None


def _normalize_article_ref(ref: str | None) -> str | None:
    if not ref:
        return None
    match = re.search(r"(\d+)", str(ref))
    if not match:
        return None
    return f"Madde {match.group(1)}"


def prioritize_demo_results(question: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reorder demo context without changing retrieval scores.

    For penalty questions, rerankers often place qualified-offense or reduction
    articles before the base offense article. When the top hits are from the
    same law, showing lower article numbers first usually surfaces the base
    penalty before qualified cases.
    """
    if not is_penalty_question(question) or not results:
        return results

    top_doc_id = results[0].get("doc_id")
    top_article_num = _article_num(results[0].get("article_ref"))
    if top_article_num is None:
        return results

    same_doc_nearby = [
        item for item in results
        if item.get("doc_id") == top_doc_id
        and (num := _article_num(item.get("article_ref"))) is not None
        and abs(num - top_article_num) <= 2
    ]
    remaining = [item for item in results if item not in same_doc_nearby]
    same_doc_nearby_sorted = sorted(
        same_doc_nearby,
        key=lambda item: (
            _article_num(item.get("article_ref")) is None,
            _article_num(item.get("article_ref")) or 10**9,
        ),
    )
    return same_doc_nearby_sorted + remaining


def build_fallback_dayanak(results: list[dict[str, Any]], max_refs: int = 3) -> str:
    top_doc_id = results[0].get("doc_id") if results else None
    refs = []
    seen = set()
    for item in results:
        if top_doc_id and item.get("doc_id") != top_doc_id:
            continue
        ref = _normalize_article_ref(item.get("article_ref"))
        if not ref or ref in seen:
            continue
        refs.append(ref)
        seen.add(ref)
        if len(refs) >= max_refs:
            break
    return ", ".join(refs)


def _finish_at_word_boundary(text: str) -> str:
    text = text.rstrip(" ,;:-\n")
    if not text:
        return text
    if re.search(r"[\s.!?]$", text):
        return text.strip()
    last_space = text.rfind(" ")
    if last_space > len(text) * 0.75:
        return text[:last_space].rstrip(" ,;:-")
    return text.strip()


def _sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _strip_dayanak(answer: str) -> str:
    return re.sub(r"(?ims)^\s*dayanak\s*[:\-].*$", "", answer).strip()


def _deduplicate_lines(answer: str) -> str:
    cleaned = []
    previous_norm = None
    for line in answer.splitlines():
        norm = re.sub(r"\s+", " ", line).strip().lower()
        if not norm:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        if norm == previous_norm:
            continue
        cleaned.append(line.rstrip())
        previous_norm = norm
    return "\n".join(cleaned).strip()


def _word_spans(text: str) -> list[tuple[str, int, int]]:
    return [
        (match.group(0).lower(), match.start(), match.end())
        for match in re.finditer(r"\w+", text, flags=re.UNICODE)
    ]


def _find_repeated_prefix_start(answer: str, min_words: int = 6) -> int | None:
    words = _word_spans(answer)
    if len(words) < min_words * 2:
        return None
    prefix = [word for word, _, _ in words[:min_words]]
    starts = []
    for idx in range(0, len(words) - min_words + 1):
        window = [word for word, _, _ in words[idx:idx + min_words]]
        if window == prefix:
            starts.append(words[idx][1])
    return starts[-1] if len(starts) >= 2 else None


def _collapse_progressive_repetition(answer: str) -> str:
    """Keep the latest expansion when a model restarts the same answer prefix."""
    text = answer.strip()
    compact_text = re.sub(r"\s+", " ", text)
    prefix = compact_text[:45].strip()
    if len(prefix) >= 25:
        prefix_pattern = re.escape(prefix).replace(r"\ ", r"\s+")
        starts = [match.start() for match in re.finditer(prefix_pattern, text, flags=re.IGNORECASE)]
        if len(starts) >= 2:
            text = text[starts[-1]:].lstrip()

    for _ in range(3):
        start = _find_repeated_prefix_start(text)
        if start is None or start <= 0:
            break
        text = text[start:].lstrip()
    return text


def _truncate_on_repeated_ngram(answer: str, n: int = 5) -> str:
    words = _word_spans(answer)
    if len(words) < n * 2:
        return answer.strip()

    seen: set[tuple[str, ...]] = set()
    for idx in range(0, len(words) - n + 1):
        ngram = tuple(word for word, _, _ in words[idx:idx + n])
        if ngram in seen:
            cut_at = words[idx][1]
            trimmed = answer[:cut_at].rstrip(" ,;:-\n")
            if len(trimmed) > 80:
                return trimmed.strip()
            return answer.strip()
        seen.add(ngram)
    return answer.strip()


def _truncate_long_answer(answer: str, max_chars: int = 650) -> str:
    if len(answer) <= max_chars:
        return answer.strip()
    prefix = answer[:max_chars].rstrip()
    last_sentence = max(prefix.rfind("."), prefix.rfind("!"), prefix.rfind("?"))
    if last_sentence > max_chars * 0.45:
        return prefix[:last_sentence + 1].strip()
    return prefix.strip()


def _compact_demo_answer(answer: str, max_sentences: int = 2, max_chars: int = 520) -> str:
    answer = _strip_dayanak(answer)
    answer = _deduplicate_lines(answer)
    answer = _collapse_progressive_repetition(answer)
    answer = _truncate_on_repeated_ngram(answer, n=5)

    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    if len(lines) > 1:
        answer = " ".join(lines)

    sentences = _sentence_split(answer)
    if sentences:
        answer = " ".join(sentences[:max_sentences])

    if len(answer) > max_chars:
        answer = answer[:max_chars].rstrip(" ,;:-")
        sentence_end = max(answer.rfind("."), answer.rfind("!"), answer.rfind("?"))
        if sentence_end > 80:
            answer = answer[:sentence_end + 1]
        else:
            answer = _finish_at_word_boundary(answer)

    return answer.strip()


def postprocess_answer(answer: str, results: list[dict[str, Any]]) -> str:
    """Ensure interactive/demo answers keep the required citation line.

    This does not invent a new legal claim; it only cites the retrieved source
    articles when the generator forgets or corrupts the required Dayanak line.
    """
    answer = _deduplicate_lines(answer.strip())
    citation_match = re.search(r"(?im)^\s*dayanak\s*[:\-]\s*(.+)$", answer)
    citation_text = citation_match.group(1).strip() if citation_match else ""
    has_numeric_citation = bool(re.search(r"\d+", citation_text))

    answer_body = answer if not citation_match else _strip_dayanak(answer)
    answer_body = _collapse_progressive_repetition(answer_body)
    answer_body = _truncate_on_repeated_ngram(answer_body)
    answer_body = _truncate_long_answer(answer_body)

    if citation_match and has_numeric_citation:
        return f"{answer_body}\nDayanak: {citation_text}".strip()

    fallback = build_fallback_dayanak(results)
    if not fallback:
        return answer_body.strip()
    return f"{answer_body.strip()}\nDayanak: {fallback}"


def ensure_dayanak(answer: str, results: list[dict[str, Any]]) -> str:
    return postprocess_answer(answer, results)


def postprocess_demo_safe_answer(answer: str, results: list[dict[str, Any]]) -> str:
    answer_body = _compact_demo_answer(answer)
    if not answer_body:
        answer_body = INSUFFICIENT_CONTEXT_FALLBACK
    fallback = build_fallback_dayanak(results, max_refs=1)
    if not fallback:
        return answer_body
    return f"{answer_body}\nDayanak: {fallback}"


def build_user_prompt(question: str, results: list[dict[str, Any]]) -> str:
    context = format_context(results)
    qtype = detect_question_type(question)

    if qtype == "list":
        task_instruction = (
            "Bu bir LİSTE sorusudur. Bağlamdaki ilgili TÜM maddeleri kullan, "
            "cevabı numaralı madde madde yaz. Eksik liste verme."
        )
    elif qtype == "definition":
        task_instruction = (
            "Bu bir TANIM sorusudur. En ilgili maddeyi merkeze alarak "
            "1-3 cümlede kısa ve net cevap ver."
        )
    else:
        task_instruction = (
            "Bağlamdaki en ilgili parçayı kullanarak kısa ve net cevap ver."
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
- Cevabın sonunda yeni bir satıra geç ve "Dayanak:" yaz; ardından virgülle ayırarak kullandığın madde numaralarını "Madde N" formatında listele (parantez, açıklama veya başka kelime kullanma).
"""


def build_demo_safe_prompt(question: str, results: list[dict[str, Any]]) -> str:
    results = prioritize_demo_results(question, results)
    context = format_context(results[:3], max_chars_per_chunk=320)
    qtype = detect_question_type(question)

    if qtype == "list":
        answer_shape = "En fazla 3 kısa madde yaz; her madde tek cümle olsun."
    else:
        answer_shape = "En fazla 2 kısa cümle yaz; cevabı tamamlanmış cümlelerle bitir."
    penalty_instruction = ""
    if is_penalty_question(question):
        penalty_instruction = (
            "\n- Soru temel ceza miktarını soruyorsa önce temel suç maddesindeki "
            "cezayı yaz; nitelikli hal veya indirim maddelerini ikinci planda tut."
        )

    return f"""Aşağıda Türk hukuk metinlerinden getirilen bağlam parçaları verilmiştir.

Bağlam:
{context}

Soru:
{question}

Görev:
- Soruyu yalnızca verilen bağlama dayanarak cevapla.
- {answer_shape}
- Bağlamda açıkça olmayan bilgi ekleme.
- Tekrar eden cümle veya kelime grubu yazma.
- Dayanak satırı yazma; kaynak maddeleri sistem ayrıca ekleyecek.
- Bağlam cevap için yetersizse sadece "{INSUFFICIENT_CONTEXT_FALLBACK}" yaz.
{penalty_instruction}
"""