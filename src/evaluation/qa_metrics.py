"""QA-side metrics for the Turkish Legal RAG project.

Components:
- Turkish-aware text normalization
- Token-level EM and F1
- Citation parser ("Dayanak:" line)
- Citation precision / recall / F1 vs gold articles
- Lightweight extractive faithfulness (lexical support of answer tokens
  in the retrieved context). This is a heuristic, NOT a substitute for
  proper NLI/LLM-as-judge faithfulness; treat it as a first-pass signal.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any


# ----------------------------- Normalization ----------------------------- #

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def turkish_lower(s: str) -> str:
    """Lowercase that handles I/İ correctly for Turkish."""
    s = s.replace("İ", "i").replace("I", "ı")
    return s.lower()


def normalize_text(s: str) -> str:
    s = turkish_lower(s)
    s = unicodedata.normalize("NFC", s)
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(s: str) -> list[str]:
    return [t for t in normalize_text(s).split() if t]


# ------------------------------ EM and F1 ------------------------------- #

def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0


def token_f1(prediction: str, reference: str) -> dict[str, float]:
    pred_toks = tokenize(prediction)
    ref_toks = tokenize(reference)

    if not pred_toks or not ref_toks:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    # Multiset intersection (so repeated tokens are counted properly)
    from collections import Counter
    pred_counts = Counter(pred_toks)
    ref_counts = Counter(ref_toks)
    common = pred_counts & ref_counts
    n_common = sum(common.values())

    if n_common == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    precision = n_common / len(pred_toks)
    recall = n_common / len(ref_toks)
    f1 = 2 * precision * recall / (precision + recall)
    return {"f1": f1, "precision": precision, "recall": recall}


# --------------------------- Citation parser ---------------------------- #

# "Madde 12", "MADDE 81-", "madde 102/A", "m. 12", with optional ranges
_ARTICLE_RE = re.compile(
    r"\b(?:madde|m\.?)\s*[:\-]?\s*(\d+(?:\s*[/\-]\s*[A-Za-zÇĞİÖŞÜçğıöşü0-9]+)?)\b",
    flags=re.IGNORECASE,
)

# "Dayanak:" line: capture the rest of that line and any continuation up to
# the next blank line / EOF.
_DAYANAK_RE = re.compile(
    r"(?im)^\s*dayanak\s*[:\-]\s*(.+?)(?=\n\s*\n|\Z)",
    flags=re.DOTALL,
)


def parse_citation_block(answer: str) -> str | None:
    """Return the raw text after 'Dayanak:' (everything until blank line / EOF),
    or None if no such block is found.
    """
    m = _DAYANAK_RE.search(answer)
    if not m:
        return None
    return m.group(1).strip()


def extract_article_numbers(text: str) -> list[str]:
    """Extract article numeric refs from arbitrary text. Returns sorted unique
    article numbers as strings (e.g. ['1','2','138']).
    """
    nums = []
    for m in _ARTICLE_RE.finditer(text):
        raw = m.group(1)
        n = re.match(r"\d+", raw.strip()).group(0)
        nums.append(n)
    return sorted(set(nums), key=lambda x: int(x))


def extract_cited_articles(answer: str) -> dict[str, Any]:
    """Look for 'Dayanak:' block first; if not found, fall back to scanning
    the whole answer for article references."""
    block = parse_citation_block(answer)
    if block is not None:
        nums = extract_article_numbers(block)
        return {"has_dayanak": True, "block": block, "articles": nums}
    nums = extract_article_numbers(answer)
    return {"has_dayanak": False, "block": None, "articles": nums}


def gold_article_numbers(gold_articles: list[str]) -> list[str]:
    return extract_article_numbers(" ".join(gold_articles))


def citation_metrics(
    cited_articles: list[str],
    gold_articles: list[str],
) -> dict[str, float]:
    cited_set = set(cited_articles)
    gold_set = set(gold_article_numbers(gold_articles))

    if not gold_set:
        return {"citation_precision": 0.0, "citation_recall": 0.0,
                "citation_f1": 0.0, "citation_exact": 0.0,
                "n_cited": float(len(cited_set)), "n_gold": 0.0}

    if not cited_set:
        return {"citation_precision": 0.0, "citation_recall": 0.0,
                "citation_f1": 0.0, "citation_exact": 0.0,
                "n_cited": 0.0, "n_gold": float(len(gold_set))}

    inter = cited_set & gold_set
    precision = len(inter) / len(cited_set)
    recall = len(inter) / len(gold_set)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    exact = 1.0 if cited_set == gold_set else 0.0
    return {
        "citation_precision": precision,
        "citation_recall": recall,
        "citation_f1": f1,
        "citation_exact": exact,
        "n_cited": float(len(cited_set)),
        "n_gold": float(len(gold_set)),
    }


# ------------------------ Lexical faithfulness -------------------------- #

# Tokens that should not count when computing support coverage (very common
# function words / connectives / morphological residue).
_STOPLIKE = {
    "ve", "veya", "ya", "ile", "ki", "de", "da", "ise", "için", "bir",
    "bu", "şu", "o", "gibi", "kadar", "olarak", "olan", "eğer", "ama",
    "ancak", "fakat", "çünkü", "her", "hem", "ne", "mi", "mı", "mu", "mü",
    "den", "dan", "dir", "dır", "dur", "dür", "ten", "tan",
    "üzere", "göre", "karşı", "sonra", "önce", "ait", "ilgili",
    "sayılı", "kanunu", "kanun", "madde", "maddesi", "fıkra", "fıkrası",
    "bakımından", "bağlamında",
}


def faithfulness_lexical(
    answer: str,
    contexts: list[str],
    *,
    drop_dayanak: bool = True,
    n_gram: int = 1,
) -> dict[str, float]:
    """Heuristic lexical support of answer tokens in retrieved contexts.

    Strategy:
    - Strip the 'Dayanak:' block from the answer if present (it cites
      article numbers, not factual claims; we don't want to evaluate those
      here).
    - Build the set of content tokens from the rest of the answer
      (lowercase, punctuation-free, minus stop-like tokens, length > 2).
    - Build the set of content tokens from the concatenated contexts.
    - support_ratio = |answer_content ∩ context_content| / |answer_content|.

    This is intentionally permissive (token-overlap, not entailment) and
    is reported as `faithfulness_lexical` to remind the reader it is not a
    proper NLI score. Use LLM-as-judge or NLI for the final verdict.
    """
    text = answer
    if drop_dayanak:
        m = _DAYANAK_RE.search(answer)
        if m:
            text = (answer[:m.start()] + " " + answer[m.end():]).strip()

    ans_tokens = [t for t in tokenize(text) if len(t) > 2 and t not in _STOPLIKE]
    ctx_tokens_set = set()
    for c in contexts:
        ctx_tokens_set.update(t for t in tokenize(c) if len(t) > 2)

    if not ans_tokens:
        return {
            "faithfulness_lexical": 0.0,
            "n_answer_tokens": 0.0,
            "n_unsupported_tokens": 0.0,
        }

    if n_gram == 1:
        supported = sum(1 for t in ans_tokens if t in ctx_tokens_set)
    else:
        # bigram support is stricter; we keep unigram default
        ctx_concat = " ".join(tokenize(c) for c in contexts)
        supported = sum(1 for i in range(len(ans_tokens) - n_gram + 1)
                        if " ".join(ans_tokens[i:i + n_gram]) in ctx_concat)

    n = len(ans_tokens)
    return {
        "faithfulness_lexical": supported / n,
        "n_answer_tokens": float(n),
        "n_unsupported_tokens": float(n - supported),
    }


# --------------------------- Aggregation -------------------------------- #

def aggregate_qa_metrics(per_q: list[dict[str, float]]) -> dict[str, float]:
    if not per_q:
        return {}
    agg: dict[str, float] = {}
    keys = set().union(*[set(p.keys()) for p in per_q])
    for k in sorted(keys):
        vals = [p[k] for p in per_q if k in p]
        agg[k] = round(sum(vals) / len(vals), 4) if vals else 0.0
    return agg
