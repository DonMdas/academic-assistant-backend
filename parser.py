# parser.py

import pdfplumber
from config import FILTER_THRESHOLDS, WINDOW_CONFIG
from ollama_parser import safe_ollama_call


def _new_filter_stats():
    return {
        "page_filter_llm_default_count": 0,
        "page_filter_toc_llm_default_count": 0,
        "page_filter_short_page_llm_default_count": 0,
    }

def extract_pages(pdf_path):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages.append({
                    "page_num": i + 1,
                    "text": text.strip()
                })
    return pages


def detect_document_type(pages):
    sample = pages[:10]

    avg_len = sum(len(p["text"]) for p in sample) / len(sample)
    avg_lines = sum(len(p["text"].split("\n")) for p in sample) / len(sample)

    if avg_len < 300 and avg_lines < 15:
        return "slides"
    elif avg_len > 1500:
        return "textbook"
    else:
        return "notes"


def check_document_size(pages):
    full_text = " ".join(p["text"] for p in pages)
    total_chars = len(full_text)

    return {
        "is_small": total_chars < 50000,
        "total_chars": total_chars,
        "full_text": full_text
    }


def is_table_of_contents(text, aggressive=False):
    lines = text.split("\n")
    lowered = text.lower()

    # Heuristic 1: too many short lines
    short_lines = sum(1 for l in lines if len(l.strip()) < 60)
    
    # Heuristic 2: many dots (..... style)
    dot_ratio = text.count('.') / max(len(text), 1)

    # Heuristic 3: explicit TOC phrases only (avoid matching plain 'contents' in body text)
    toc_keywords = ["table of contents", "contents page"]

    if any(k in lowered for k in toc_keywords):
        return True

    # Allow a standalone 'Contents' heading only when it appears at the top of the page.
    top_lines = [l.strip().lower() for l in lines[:5] if l.strip()]
    if any(l in {"contents", "index"} for l in top_lines):
        return True

    if aggressive:
        if short_lines / max(len(lines), 1) > 0.7:
            return True

        if dot_ratio > 0.05:
            return True

    return False


def is_toc_ai(text, stats=None):
    prompt = f"""
Is this page a table of contents?

Text:
{text}

Answer ONLY:
true or false
"""

    res = safe_ollama_call(prompt)
    if not isinstance(res, dict) or not res:
        if isinstance(stats, dict):
            stats["page_filter_llm_default_count"] = int(stats.get("page_filter_llm_default_count", 0)) + 1
            stats["page_filter_toc_llm_default_count"] = int(stats.get("page_filter_toc_llm_default_count", 0)) + 1
        return False

    return "true" in str(res).lower()


def is_short_page_useful_ai(text, doc_type, stats=None):
    prompt = f"""
You are deciding whether a short document page should be kept for chunking.

Document type: {doc_type}
Word count: {len(text.split())}
Character count: {len(text)}

Keep the page if it contains useful instructional content, examples, formulas,
definitions, exercises, headings, or any meaningful partial content.
Reject only if it is mostly navigation, blank space, copyright, index, or table of contents.

Return JSON only:
{{
  "keep": true,
  "reason": "short explanation"
}}

Text:
{text}
"""

    res = safe_ollama_call(prompt)
    if not isinstance(res, dict):
        res = {}

    keep_value = res.get("keep")
    if keep_value is None:
        if isinstance(stats, dict):
            stats["page_filter_llm_default_count"] = int(stats.get("page_filter_llm_default_count", 0)) + 1
            stats["page_filter_short_page_llm_default_count"] = int(stats.get("page_filter_short_page_llm_default_count", 0)) + 1
        keep = True
    else:
        keep = keep_value

    if isinstance(keep, str):
        keep = keep.lower() in {"true", "yes", "1"}
    return bool(keep), str(res.get("reason", "")).strip()



def rule_based_filter(page, doc_type, stats=None):
    text = page["text"]
    word_count = len(text.split())

    # Relaxed mode: drop TOC only on strong structural signals.
    if is_table_of_contents(text, aggressive=False):
        return False

    # Run expensive/strict AI TOC check only for short pages where TOC confusion is likelier.
    if word_count <= 280 and is_toc_ai(text, stats=stats):
        return False

    thresholds = FILTER_THRESHOLDS[doc_type]

    if len(text) < thresholds["min_chars"] or word_count < max(25, thresholds["min_lines"] * 12):
        keep, reason = is_short_page_useful_ai(text, doc_type, stats=stats)
        if not keep:
            print(f"🧹 Dropping short page ({word_count} words): {reason or 'LLM marked it not useful'}")
        return keep

    if len(text.split("\n")) < thresholds["min_lines"]:
        if doc_type != "slides":
            keep, reason = is_short_page_useful_ai(text, doc_type, stats=stats)
            if not keep:
                print(f"🧹 Dropping low-line page ({word_count} words): {reason or 'LLM marked it not useful'}")
            return keep

    useless_keywords = [
        "copyright", "isbn", "all rights reserved",
        "table of contents", "acknowledgements"
    ]

    if any(k in text.lower() for k in useless_keywords):
        return False

    return True


def filter_pages(pages, doc_type, return_stats=False):
    stats = _new_filter_stats()
    filtered = [p for p in pages if rule_based_filter(p, doc_type, stats=stats)]
    if return_stats:
        return filtered, stats
    return filtered


def create_windows(pages, doc_type):
    cfg = WINDOW_CONFIG[doc_type]
    size, overlap = cfg["window_size"], cfg["overlap"]

    windows = []
    step = size - overlap

    for i in range(0, len(pages), step):
        chunk_pages = pages[i:i+size]
        text = "\n\n".join(p["text"] for p in chunk_pages)

        windows.append({
            "window_id": len(windows),
            "text": text,
            "pages": [p["page_num"] for p in chunk_pages]
        })

    return windows