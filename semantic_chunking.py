# semantic_chunking.py

import difflib
import re

import numpy as np

from ollama_parser import chat_ollama, safe_ollama_call, detect_segments
from rag_engine import model as embedding_model

def _to_percent(value, default):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_char_range(text_len, start_pct, end_pct):
    start = int(text_len * (start_pct / 100.0))
    end = int(text_len * (end_pct / 100.0))

    start = max(0, min(text_len, start))
    end = max(0, min(text_len, end))

    if end < start:
        start, end = end, start

    return start, end


def _normalize_token(token):
    return re.sub(r"[^a-z0-9]+", "", token.lower())


def _tokenize_with_spans(text):
    tokens = []
    for match in re.finditer(r"\S+", text):
        raw = match.group(0)
        normalized = _normalize_token(raw)
        if not normalized:
            continue
        tokens.append({
            "token": normalized,
            "start": match.start(),
            "end": match.end(),
        })
    return tokens


def _phrase_tokens(phrase):
    if not phrase:
        return []
    parts = re.findall(r"\S+", str(phrase))
    out = [_normalize_token(p) for p in parts]
    return [p for p in out if p]


def _find_exact_phrase_index(text_tokens, phrase_tokens, start_search=0):
    if not phrase_tokens:
        return None

    n = len(text_tokens)
    m = len(phrase_tokens)
    if m > n:
        return None

    token_values = [t["token"] for t in text_tokens]
    for i in range(max(0, start_search), n - m + 1):
        if token_values[i:i + m] == phrase_tokens:
            return i
    return None


def _find_fuzzy_phrase_index(text_tokens, phrase_tokens, start_search=0, threshold=0.76):
    if not phrase_tokens:
        return None

    n = len(text_tokens)
    m = len(phrase_tokens)
    if m > n:
        return None

    phrase_text = " ".join(phrase_tokens)
    best_index = None
    best_score = 0.0

    for i in range(max(0, start_search), n - m + 1):
        candidate = " ".join(text_tokens[j]["token"] for j in range(i, i + m))
        score = difflib.SequenceMatcher(None, phrase_text, candidate).ratio()
        if score > best_score:
            best_score = score
            best_index = i

    if best_index is not None and best_score >= threshold:
        return best_index
    return None


def _resolve_anchor_range(text, seg):
    text_tokens = _tokenize_with_spans(text)
    if not text_tokens:
        return None

    start_anchor = seg.get("start_anchor") or seg.get("start_phrase")
    end_anchor = seg.get("end_anchor") or seg.get("end_phrase")

    start_phrase_tokens = _phrase_tokens(start_anchor)
    end_phrase_tokens = _phrase_tokens(end_anchor)

    if not start_phrase_tokens or not end_phrase_tokens:
        return None

    start_idx = _find_exact_phrase_index(text_tokens, start_phrase_tokens)
    start_match_type = "exact"
    if start_idx is None:
        start_idx = _find_fuzzy_phrase_index(text_tokens, start_phrase_tokens)
        start_match_type = "fuzzy"

    if start_idx is None:
        return None

    end_idx = _find_exact_phrase_index(text_tokens, end_phrase_tokens, start_search=start_idx)
    end_match_type = "exact"
    if end_idx is None:
        end_idx = _find_fuzzy_phrase_index(text_tokens, end_phrase_tokens, start_search=start_idx)
        end_match_type = "fuzzy"

    if end_idx is None:
        return None

    start_char = text_tokens[start_idx]["start"]
    end_token_last_index = end_idx + len(end_phrase_tokens) - 1
    if end_token_last_index >= len(text_tokens):
        return None
    end_char = text_tokens[end_token_last_index]["end"]

    if end_char <= start_char:
        return None

    return {
        "start": start_char,
        "end": end_char,
        "method": "anchors",
        "start_match": start_match_type,
        "end_match": end_match_type,
    }


def _resolve_segment_char_range(text, seg):
    anchor_result = _resolve_anchor_range(text, seg)
    if anchor_result is not None:
        return anchor_result

    start_pct = _to_percent(seg.get("start_percent"), 0.0)
    end_pct = _to_percent(seg.get("end_percent"), 100.0)

    start_pct = max(0.0, min(100.0, start_pct))
    end_pct = max(0.0, min(100.0, end_pct))
    if end_pct < start_pct:
        start_pct, end_pct = end_pct, start_pct

    start, end = _to_char_range(len(text), start_pct, end_pct)

    return {
        "start": start,
        "end": end,
        "method": "percent",
        "start_match": "n/a",
        "end_match": "n/a",
    }


def _merge_ranges(ranges):
    if not ranges:
        return []

    ordered = sorted(ranges, key=lambda x: (x[0], x[1]))
    merged = [ordered[0]]

    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


def _chat_text(prompt):
    response = chat_ollama(
        messages=[{"role": "user", "content": prompt}],
        retries=3,
    )

    return response.get("message", {}).get("content", "").strip()


def _normalize_text_list(value):
    if isinstance(value, list):
        cleaned = []
        for item in value:
            text = str(item).strip()
            if text:
                cleaned.append(text)
        return cleaned
    return []


def _normalize_complexity(value, default="intermediate"):
    allowed = {"beginner", "intermediate", "advanced"}
    if not value:
        return default

    lowered = str(value).lower()
    for candidate in allowed:
        if candidate in lowered:
            return candidate

    return default


def _build_topics_covered_summary(topic, subtopics=None, summary_text=""):
    topics = []
    seen = set()

    def _push(value):
        text = str(value or "").strip()
        if not text:
            return
        key = text.lower()
        if key in seen:
            return
        seen.add(key)
        topics.append(text)

    _push(topic)
    if isinstance(subtopics, list):
        for item in subtopics:
            _push(item)

    topic_clause = "Topics covered: " + "; ".join(topics[:12]) if topics else "Topics covered: Unspecified topic"
    detail = str(summary_text or "").strip()

    if not detail:
        return topic_clause

    if detail.lower().startswith("topics covered:"):
        return detail

    return f"{topic_clause}. {detail}"


def _normalize_estimated_time(value, default=30):
    try:
        if value is None:
            return default
        numeric = int(float(value))
        return max(5, min(180, numeric))
    except (TypeError, ValueError):
        return default


def _collapse_whitespace(text, limit=None):
    collapsed = re.sub(r"\s+", " ", str(text or "")).strip()
    if limit is not None and len(collapsed) > limit:
        return collapsed[:limit].rstrip()
    return collapsed


def _fallback_topic_from_text(text):
    collapsed = _collapse_whitespace(text, limit=400)
    if not collapsed:
        return "Untitled Topic"

    words = collapsed.split()
    if len(words) <= 8:
        return collapsed[:120]

    sentences = re.split(r"(?<=[.!?])\s+", collapsed)
    if sentences and sentences[0].strip():
        first_sentence = sentences[0].strip()
        first_words = first_sentence.split()
        if len(first_words) <= 12:
            return first_sentence[:120]

    return " ".join(words[:8]).strip(" .:-") or "Untitled Topic"


def _fallback_summary_from_text(text, previous_summary=None):
    collapsed = _collapse_whitespace(text, limit=700)
    if not collapsed:
        return str(previous_summary or "").strip()

    if previous_summary:
        base = str(previous_summary).strip()
        if base:
            return f"{base} {collapsed[:220]}".strip()

    return collapsed[:260]


def _longest_suffix_prefix_overlap(previous_text, current_text, min_overlap_chars=220):
    previous_text = str(previous_text or "")
    current_text = str(current_text or "")
    max_len = min(len(previous_text), len(current_text))

    if max_len < min_overlap_chars:
        return 0

    for size in range(max_len, min_overlap_chars - 1, -1):
        if previous_text[-size:] == current_text[:size]:
            return size

    return 0


def _build_analysis_text(current_text, previous_window_text=None, min_overlap_chars=220, min_novel_chars=600):
    overlap_chars = _longest_suffix_prefix_overlap(
        previous_window_text,
        current_text,
        min_overlap_chars=min_overlap_chars,
    )

    if overlap_chars <= 0:
        return {
            "analysis_text": current_text,
            "used_novel_text": False,
            "trimmed_overlap_chars": 0,
        }

    novel_text = str(current_text)[overlap_chars:].lstrip()
    if len(novel_text) < min_novel_chars:
        return {
            "analysis_text": current_text,
            "used_novel_text": False,
            "trimmed_overlap_chars": 0,
        }

    return {
        "analysis_text": novel_text,
        "used_novel_text": True,
        "trimmed_overlap_chars": overlap_chars,
    }


def _label_tokens(text):
    return {token for token in re.findall(r"[a-z0-9]{3,}", str(text).lower())}


def _topic_label_similarity(label_a, label_b):
    tokens_a = _label_tokens(label_a)
    tokens_b = _label_tokens(label_b)
    if not tokens_a or not tokens_b:
        return 0.0

    intersection = len(tokens_a.intersection(tokens_b))
    union = len(tokens_a.union(tokens_b))
    if union == 0:
        return 0.0
    return float(intersection / union)


def _cosine_similarity(vector_a, vector_b):
    denominator = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(vector_a, vector_b) / denominator)


def _estimate_topic_transition(previous_state, text, continue_threshold=0.72, break_threshold=0.56):
    if not isinstance(previous_state, dict):
        return None

    previous_topic = _collapse_whitespace(previous_state.get("topic", ""), limit=160)
    previous_summary = _collapse_whitespace(previous_state.get("summary", ""), limit=700)
    if not previous_topic and not previous_summary:
        return None

    current_text = _collapse_whitespace(text, limit=1400)
    if not current_text:
        return None

    previous_context = previous_topic
    if previous_summary:
        previous_context = f"{previous_topic}. {previous_summary}".strip(" .")

    try:
        vectors = embedding_model.encode([previous_context, current_text])
        if len(vectors) < 2:
            return None
        similarity = _cosine_similarity(np.asarray(vectors[0], dtype=np.float32), np.asarray(vectors[1], dtype=np.float32))
    except Exception:
        return None

    if similarity >= continue_threshold:
        return {"same_topic": True, "similarity": similarity, "source": "embedding"}

    if similarity <= break_threshold:
        return {"same_topic": False, "similarity": similarity, "source": "embedding"}

    return {"same_topic": None, "similarity": similarity, "source": "embedding"}


def _analyze_chunk_context(text, previous_state=None, forced_same_topic=None):
    previous_heading = ""
    previous_summary = ""
    decision_note = ""

    if isinstance(previous_state, dict):
        previous_heading = str(previous_state.get("topic", "") or "").strip()
        previous_summary = str(previous_state.get("summary", "") or "").strip()

    if forced_same_topic is True:
        decision_note = "Embedding gate decision: this chunk continues the previous topic."
    elif forced_same_topic is False:
        decision_note = "Embedding gate decision: this chunk starts a new topic."

    prompt = f"""
You are building a hierarchical study outline from fixed sequential chunks.

Previous topic heading:
{previous_heading or 'None'}

Previous running summary:
{previous_summary or 'None'}

Current chunk text:
{_collapse_whitespace(text, limit=6000)}

{decision_note}

If the embedding gate has already decided the relationship, do not contradict it.
If the gate is neutral, decide whether this chunk continues the previous topic.

Return JSON only:
{{
  "same_topic": true,
  "topic": "short topic heading",
  "summary": "updated running summary for the topic after including this chunk",
  "key_concepts": [],
  "subtopics": [],
  "complexity": "beginner|intermediate|advanced",
  "prerequisites": [],
  "estimated_time": 30
}}

Rules:
- If the chunk continues the previous topic, set same_topic to true and keep the topic stable.
- If it starts a new topic, set same_topic to false and give a new concise heading.
- The summary must reflect the whole topic so far, not only the latest chunk.
- If the text introduces a new numbered section/chapter heading, prefer same_topic=false.
- Keep the output factual, concise, and directly grounded in the text.
"""

    try:
        response = safe_ollama_call(prompt)
    except Exception:
        response = {}

    if not isinstance(response, dict):
        return {}

    return response


def _merge_unique_items(existing, incoming):
    merged = []
    seen = set()

    for value in list(existing) + list(incoming):
        text = str(value).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(text)

    return merged


def _decide_fallback_merge(fallback_chunk_text, prev_chunk_text=None, next_chunk_text=None):
    """Use the LLM to decide whether a fallback chunk should merge left, right, or stay separate."""
    if not prev_chunk_text and not next_chunk_text:
        return "keep_separate"

    prev_preview = prev_chunk_text[:250] if prev_chunk_text else "N/A"
    next_preview = next_chunk_text[:250] if next_chunk_text else "N/A"
    fallback_preview = fallback_chunk_text[:250]

    prompt = f"""
You are a document chunking expert.

Choose the best action for this uncovered fallback chunk:
Fallback chunk:
{fallback_preview}

Previous chunk:
{prev_preview}

Next chunk:
{next_preview}

Return exactly one token:
previous
next
keep_separate
"""

    try:
        decision_text = _chat_text(prompt).lower()
    except Exception:
        return "keep_separate"

    if "previous" in decision_text:
        return "previous"
    if "next" in decision_text:
        return "next"
    return "keep_separate"


def _build_uncovered_fallback_chunks(window, covered_ranges):
    text = window["text"]
    text_len = len(text)
    if text_len == 0:
        return []

    merged = _merge_ranges([(s, e) for s, e in covered_ranges if s < e])

    gaps = []
    cursor = 0
    for start, end in merged:
        if cursor < start:
            gaps.append((cursor, start))
        cursor = max(cursor, end)

    if cursor < text_len:
        gaps.append((cursor, text_len))

    # If no LLM ranges exist, preserve full window text as one fallback chunk.
    if not merged and text.strip():
        gaps = [(0, text_len)]

    fallback_chunks = []
    for start, end in gaps:
        chunk_text = text[start:end]
        if not chunk_text.strip():
            continue

        start_pct = round((start / text_len) * 100.0, 2)
        end_pct = round((end / text_len) * 100.0, 2)

        fallback_chunks.append({
            "topic": "Uncovered Segment",
            "text": chunk_text,
            "size": "fallback",
            "pages": window["pages"],
            "window_id": window.get("window_id"),
            "boundary_source": "llm_uncovered_fallback",
            "coverage": {
                "start_char": start,
                "end_char": end,
                "start_percent": start_pct,
                "end_percent": end_pct,
                "window_id": window.get("window_id"),
            },
            "confidence": "low",
            "description": "This chunk was created from text not covered by LLM-provided boundaries, to ensure full document coverage.",
        })

    return fallback_chunks


def split_by_segments(window, segments):
    text = window["text"]
    chunks = []
    covered_ranges = []

    if not text:
        return chunks

    for seg in segments:
        resolved = _resolve_segment_char_range(text, seg)
        start = resolved["start"]
        end = resolved["end"]

        chunk_text = text[start:end]
        if not chunk_text.strip():
            continue

        covered_ranges.append((start, end))

        text_len = max(1, len(text))
        start_pct = round((start / text_len) * 100.0, 2)
        end_pct = round((end / text_len) * 100.0, 2)

        chunks.append({
            "topic": seg.get("topic", "Unknown"),
            "text": chunk_text,
            "size": seg.get("size", "medium"),
            "pages": window["pages"],
            "window_id": window.get("window_id"),
            "boundary_source": "llm_segment",
            "coverage": {
                "start_char": start,
                "end_char": end,
                "start_percent": start_pct,
                "end_percent": end_pct,
                "window_id": window.get("window_id"),
                "boundary_method": resolved.get("method", "percent"),
                "start_match": resolved.get("start_match", "n/a"),
                "end_match": resolved.get("end_match", "n/a"),
                "start_anchor": seg.get("start_anchor") or seg.get("start_phrase"),
                "end_anchor": seg.get("end_anchor") or seg.get("end_phrase"),
            },
            "confidence": "medium",
        })

    chunks.extend(_build_uncovered_fallback_chunks(window, covered_ranges))

    return chunks


def merge_small_chunks(segments):
    merged = []
    buffer = None

    for seg in segments:
        # Keep fallback chunks unchanged to retain exact uncovered-region coverage.
        if seg.get("boundary_source") == "llm_uncovered_fallback":
            if buffer:
                merged.append(buffer)
                buffer = None
            merged.append(seg)
            continue

        # Only merge small chunks from the same window.
        if buffer and seg.get("window_id") != buffer.get("window_id"):
            merged.append(buffer)
            buffer = None

        if seg["size"] in ["tiny", "small"]:
            if buffer:
                buffer["text"] += "\n\n" + seg["text"]
                buffer["topic"] += f", {seg['topic']}"
                cov_a = buffer.get("coverage")
                cov_b = seg.get("coverage")
                if isinstance(cov_a, dict) and isinstance(cov_b, dict):
                    cov_a["start_char"] = min(cov_a.get("start_char", 0), cov_b.get("start_char", 0))
                    cov_a["end_char"] = max(cov_a.get("end_char", 0), cov_b.get("end_char", 0))
                    cov_a["start_percent"] = min(cov_a.get("start_percent", 0.0), cov_b.get("start_percent", 0.0))
                    cov_a["end_percent"] = max(cov_a.get("end_percent", 0.0), cov_b.get("end_percent", 0.0))
            else:
                buffer = dict(seg)
                if isinstance(seg.get("coverage"), dict):
                    buffer["coverage"] = dict(seg["coverage"])
        else:
            if buffer:
                merged.append(buffer)
                buffer = None
            merged.append(seg)

    if buffer:
        merged.append(buffer)

    return merged


def _window_id_sort_key(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 10**9


def _merge_chunk_text(target, source, prepend=False):
    if prepend:
        target["text"] = source["text"] + "\n\n" + target["text"]
    else:
        target["text"] += "\n\n" + source["text"]

    target.setdefault("merged_with_fallback_count", 0)
    target["merged_with_fallback_count"] += 1

    target["boundary_source"] = "llm_segment+fallback"

    target_cov = target.get("coverage")
    source_cov = source.get("coverage")
    if isinstance(target_cov, dict) and isinstance(source_cov, dict):
        if prepend:
            target_cov["start_char"] = source_cov.get("start_char", target_cov.get("start_char"))
            target_cov["start_percent"] = source_cov.get("start_percent", target_cov.get("start_percent"))
        else:
            target_cov["end_char"] = source_cov.get("end_char", target_cov.get("end_char"))
            target_cov["end_percent"] = source_cov.get("end_percent", target_cov.get("end_percent"))


def _finalize_pending_prefix(chunk):
    pending_prefixes = chunk.pop("_pending_prefixes", [])
    pending_fallbacks = chunk.pop("_pending_fallbacks", [])

    if not pending_prefixes:
        return chunk

    prefix_text = "\n\n".join(pending_prefixes)
    chunk["text"] = prefix_text + "\n\n" + chunk["text"]
    chunk.setdefault("merged_with_fallback_count", 0)
    chunk["merged_with_fallback_count"] += len(pending_prefixes)
    chunk["boundary_source"] = "llm_segment+fallback"

    chunk_cov = chunk.get("coverage")
    if isinstance(chunk_cov, dict) and pending_fallbacks:
        first_cov = next((fb.get("coverage") for fb in pending_fallbacks if isinstance(fb.get("coverage"), dict)), None)
        if first_cov:
            chunk_cov["start_char"] = first_cov.get("start_char", chunk_cov.get("start_char"))
            chunk_cov["start_percent"] = first_cov.get("start_percent", chunk_cov.get("start_percent"))

    merged_ids = chunk.setdefault("merged_with_fallback", [])
    for fb in pending_fallbacks:
        merged_ids.append(fb.get("chunk_label", "fallback"))

    return chunk


def _merge_fallbacks_for_window(window_chunks):
    processed = []
    stats = {"fallback_total": 0, "merged_previous": 0, "merged_next": 0, "kept_separate": 0}

    for index, chunk in enumerate(window_chunks):
        if chunk.get("boundary_source") != "llm_uncovered_fallback":
            chunk = _finalize_pending_prefix(chunk)
            processed.append(chunk)
            continue

        stats["fallback_total"] += 1

        previous_chunk = processed[-1] if processed else None
        next_chunk = None
        for look_ahead in window_chunks[index + 1:]:
            if look_ahead.get("boundary_source") != "llm_uncovered_fallback":
                next_chunk = look_ahead
                break

        decision = _decide_fallback_merge(
            chunk.get("text", ""),
            previous_chunk.get("text") if previous_chunk else None,
            next_chunk.get("text") if next_chunk else None,
        )

        if decision == "previous" and previous_chunk is not None:
            _merge_chunk_text(previous_chunk, chunk, prepend=False)
            previous_chunk.setdefault("merged_with_fallback", []).append(chunk.get("chunk_label", "fallback"))
            stats["merged_previous"] += 1
        elif decision == "next" and next_chunk is not None:
            next_chunk.setdefault("_pending_prefixes", []).append(chunk["text"])
            next_chunk.setdefault("_pending_fallbacks", []).append(chunk)
            stats["merged_next"] += 1
        else:
            processed.append(chunk)
            stats["kept_separate"] += 1

    return processed, stats


def build_legacy_segment_windows(windows):
    """
    Build pseudo-windows from legacy LLM segments + uncovered fallback logic.

    Returned windows can be passed to the hierarchical chunk builder to keep
    metadata extraction unified while still using legacy boundaries.
    """
    segmented_windows = []
    stats = {
        "input_windows": len(windows),
        "output_windows": 0,
        "windows_with_llm_segments": 0,
        "windows_with_segment_fallback": 0,
        "segment_fallback_chunks_total": 0,
        "segment_fallback_merged_previous": 0,
        "segment_fallback_merged_next": 0,
        "segment_fallback_kept_separate": 0,
    }

    for index, window in enumerate(windows):
        text = str(window.get("text", "") or "")
        if not text.strip():
            continue

        response = {}
        try:
            response = detect_segments(text)
        except Exception:
            response = {}

        raw_segments = response.get("segments", []) if isinstance(response, dict) else []
        segments = [seg for seg in raw_segments if isinstance(seg, dict)] if isinstance(raw_segments, list) else []

        if segments:
            stats["windows_with_llm_segments"] += 1
        else:
            stats["windows_with_segment_fallback"] += 1

        window_chunks = split_by_segments(window, segments)
        window_chunks = merge_small_chunks(window_chunks)
        window_chunks.sort(
            key=lambda item: (item.get("coverage") or {}).get("start_char", 0)
        )

        window_chunks, merge_stats = _merge_fallbacks_for_window(window_chunks)
        stats["segment_fallback_chunks_total"] += int(merge_stats.get("fallback_total", 0))
        stats["segment_fallback_merged_previous"] += int(merge_stats.get("merged_previous", 0))
        stats["segment_fallback_merged_next"] += int(merge_stats.get("merged_next", 0))
        stats["segment_fallback_kept_separate"] += int(merge_stats.get("kept_separate", 0))

        for seg_index, chunk in enumerate(window_chunks):
            chunk_text = str(chunk.get("text", "") or "")
            if not chunk_text.strip():
                continue

            segmented_windows.append({
                "window_id": f"{window.get('window_id', index)}:{seg_index}",
                "text": chunk_text,
                "pages": list(chunk.get("pages", [])),
                "_legacy_segment_meta": {
                    "boundary_source": chunk.get("boundary_source"),
                    "coverage": chunk.get("coverage"),
                    "confidence": chunk.get("confidence"),
                    "description": chunk.get("description"),
                    "original_window_id": chunk.get("window_id", window.get("window_id", index)),
                },
            })

    stats["output_windows"] = len(segmented_windows)
    return segmented_windows, stats


def build_semantic_chunks(windows, segment_results=None):
    """
    Build fixed chunks, then walk them sequentially to assign and update topic summaries.

    The LLM decides whether each chunk continues the previous topic. If it does,
    the running topic state is updated; otherwise a new topic group starts.
    """
    chunks = []
    stats = {
        "input_windows": len(windows),
        "output_chunks": 0,
        "topic_groups": 0,
        "topic_continuations": 0,
        "topic_resets": 0,
        "llm_failures": 0,
        "analysis_novel_windows": 0,
        "analysis_overlap_trimmed_chars": 0,
        "embedding_gate_evaluations": 0,
        "embedding_gate_hits": 0,
        "embedding_gate_continuations": 0,
        "embedding_gate_breaks": 0,
        "embedding_gate_neutral": 0,
    }

    current_state = None
    current_group_index = -1
    previous_window_text = ""

    for index, window in enumerate(windows):
        text = str(window.get("text", "") or "")
        if not text.strip():
            continue

        print(
            f"   [chunk {index + 1}/{len(windows)}] evaluating window_id={window.get('window_id', index)} chars={len(text)}..."
        )

        analysis_view = _build_analysis_text(text, previous_window_text=previous_window_text)
        analysis_text = analysis_view["analysis_text"]
        trimmed_overlap_chars = int(analysis_view.get("trimmed_overlap_chars", 0))
        if analysis_view.get("used_novel_text"):
            stats["analysis_novel_windows"] += 1
            stats["analysis_overlap_trimmed_chars"] += trimmed_overlap_chars
            print(f"      analysis text: using novel tail (trimmed_overlap_chars={trimmed_overlap_chars})")
        else:
            print("      analysis text: using full window")

        gate = _estimate_topic_transition(current_state, analysis_text)
        forced_same_topic = None
        stats["embedding_gate_evaluations"] += 1
        if gate is not None and gate.get("same_topic") is not None:
            stats["embedding_gate_hits"] += 1
            forced_same_topic = bool(gate["same_topic"])
            if forced_same_topic:
                stats["embedding_gate_continuations"] += 1
                gate_label = "continue"
            else:
                stats["embedding_gate_breaks"] += 1
                gate_label = "new-topic"
            print(
                f"      embedding gate: similarity={gate.get('similarity', 0.0):.3f} -> {gate_label}"
            )
        else:
            stats["embedding_gate_neutral"] += 1
            if gate is not None:
                print(
                    f"      embedding gate: similarity={gate.get('similarity', 0.0):.3f} -> neutral"
                )
            else:
                print("      embedding gate: unavailable -> ask LLM")

        analysis = _analyze_chunk_context(analysis_text, current_state, forced_same_topic=forced_same_topic)
        same_topic = analysis.get("same_topic")
        if forced_same_topic is not None:
            same_topic = forced_same_topic
        if isinstance(same_topic, str):
            same_topic = same_topic.strip().lower() in {"true", "yes", "1"}

        if same_topic is None:
            same_topic = current_state is not None

        if same_topic is True and isinstance(current_state, dict):
            proposed_topic = str(analysis.get("topic") or "").strip()
            current_topic = str(current_state.get("topic") or "").strip()
            gate_similarity = float(gate.get("similarity", 1.0)) if isinstance(gate, dict) else 1.0
            if proposed_topic and current_topic:
                label_similarity = _topic_label_similarity(proposed_topic, current_topic)
                if label_similarity < 0.15 and gate_similarity < 0.62:
                    same_topic = False
                    print(
                        f"      topic consistency: heading drift detected (label_similarity={label_similarity:.2f}, similarity={gate_similarity:.2f}) -> new-topic"
                    )

        # If the model says "new topic" but proposed/current labels are almost the same,
        # treat it as continuation unless the embedding gate explicitly forced a break.
        if same_topic is False and isinstance(current_state, dict) and forced_same_topic is None:
            proposed_topic = str(analysis.get("topic") or "").strip()
            current_topic = str(current_state.get("topic") or "").strip()
            gate_similarity = float(gate.get("similarity", 0.0)) if isinstance(gate, dict) else 0.0
            if proposed_topic and current_topic:
                label_similarity = _topic_label_similarity(proposed_topic, current_topic)
                if label_similarity >= 0.60 or (label_similarity >= 0.35 and gate_similarity >= 0.63):
                    same_topic = True
                    print(
                        f"      topic consistency: labels aligned (label_similarity={label_similarity:.2f}, similarity={gate_similarity:.2f}) -> continue"
                    )

        continuation_of_previous = current_state is not None and same_topic is True

        llm_failed = not bool(analysis)
        if llm_failed:
            stats["llm_failures"] += 1

        metadata_fallback_used = bool(llm_failed)
        metadata_source = "fallback" if llm_failed else "llm"

        if current_state is None or not same_topic:
            current_group_index += 1
            stats["topic_groups"] += 1
            stats["topic_resets"] += 1 if current_state is not None else 0

            topic_raw = str(analysis.get("topic") or "").strip()
            summary_raw = str(analysis.get("summary") or "").strip()
            complexity_raw = analysis.get("complexity")
            estimated_time_raw = analysis.get("estimated_time")

            topic = str(topic_raw or _fallback_topic_from_text(analysis_text)).strip()
            summary = str(summary_raw or _fallback_summary_from_text(analysis_text)).strip()
            key_concepts = _normalize_text_list(analysis.get("key_concepts"))
            subtopics = _normalize_text_list(analysis.get("subtopics"))
            prerequisites = _normalize_text_list(analysis.get("prerequisites"))
            complexity = _normalize_complexity(complexity_raw, default="intermediate")
            estimated_time = _normalize_estimated_time(estimated_time_raw, default=30)
            summary = _build_topics_covered_summary(topic, subtopics, summary)

            metadata_fallback_used = metadata_fallback_used or (not topic_raw) or (not summary_raw)
            metadata_fallback_used = metadata_fallback_used or (complexity_raw in (None, ""))
            metadata_fallback_used = metadata_fallback_used or (estimated_time_raw in (None, ""))
            if not llm_failed and metadata_fallback_used:
                metadata_source = "llm+fallback"

            current_state = {
                "topic": topic,
                "summary": summary,
                "key_concepts": key_concepts,
                "subtopics": subtopics,
                "complexity": complexity,
                "prerequisites": prerequisites,
                "estimated_time": estimated_time,
                "group_index": current_group_index,
            }
            print(f"      topic: start '{topic}'")
        else:
            stats["topic_continuations"] += 1

            summary_raw = str(analysis.get("summary") or "").strip()
            complexity_raw = analysis.get("complexity")
            estimated_time_raw = analysis.get("estimated_time")

            topic = str(current_state.get("topic") or analysis.get("topic") or _fallback_topic_from_text(analysis_text)).strip()
            summary = str(summary_raw or _fallback_summary_from_text(analysis_text, current_state.get("summary"))).strip()
            merged_subtopics = _merge_unique_items(current_state.get("subtopics", []), _normalize_text_list(analysis.get("subtopics")))
            summary = _build_topics_covered_summary(topic, merged_subtopics, summary)

            current_state["topic"] = topic
            current_state["summary"] = summary
            current_state["key_concepts"] = _merge_unique_items(current_state.get("key_concepts", []), _normalize_text_list(analysis.get("key_concepts")))
            current_state["subtopics"] = merged_subtopics
            current_state["prerequisites"] = _merge_unique_items(current_state.get("prerequisites", []), _normalize_text_list(analysis.get("prerequisites")))
            current_state["complexity"] = _normalize_complexity(complexity_raw, default=current_state.get("complexity", "intermediate"))
            current_state["estimated_time"] = max(
                current_state.get("estimated_time", 30),
                _normalize_estimated_time(estimated_time_raw, default=current_state.get("estimated_time", 30))
            )

            metadata_fallback_used = metadata_fallback_used or (not summary_raw)
            metadata_fallback_used = metadata_fallback_used or (complexity_raw in (None, ""))
            metadata_fallback_used = metadata_fallback_used or (estimated_time_raw in (None, ""))
            if not llm_failed and metadata_fallback_used:
                metadata_source = "llm+fallback"

            print(f"      topic: continue '{topic}'")

        text_len = len(text)
        chunk_id = f"chunk_{len(chunks)}"
        coverage = {
            "window_id": window.get("window_id"),
            "start_char": 0,
            "end_char": text_len,
            "start_percent": 0.0,
            "end_percent": 100.0,
            "boundary_method": "fixed_window",
            "start_match": "n/a",
            "end_match": "n/a",
        }

        metadata = {
            "topic": current_state["topic"],
            "summary": current_state["summary"],
            "subtopics": list(current_state.get("subtopics", [])),
            "key_concepts": list(current_state.get("key_concepts", [])),
            "complexity": current_state.get("complexity", "intermediate"),
            "prerequisites": list(current_state.get("prerequisites", [])),
            "estimated_time": current_state.get("estimated_time", 30),
            "pages": list(window.get("pages", [])),
            "window_id": window.get("window_id"),
            "group_index": current_state["group_index"],
            "same_topic_as_previous": continuation_of_previous,
            "topic_continues": continuation_of_previous,
            "metadata_source": metadata_source,
            "llm_failed": bool(llm_failed),
            "metadata_fallback_used": bool(metadata_fallback_used),
            "boundary_source": "fixed_window_hierarchical",
            "coverage": coverage,
        }

        chunks.append({
            "chunk_id": chunk_id,
            "text": text,
            "metadata": metadata,
        })

        previous_window_text = text

    stats["output_chunks"] = len(chunks)

    print(
        "🧩 Hierarchical chunk stats: "
        f"input_windows={stats['input_windows']} | "
        f"output_chunks={stats['output_chunks']} | "
        f"topic_groups={stats['topic_groups']} | "
        f"topic_continuations={stats['topic_continuations']} | "
        f"topic_resets={stats['topic_resets']} | "
        f"llm_failures={stats['llm_failures']} | "
        f"analysis_novel_windows={stats['analysis_novel_windows']} | "
        f"analysis_overlap_trimmed_chars={stats['analysis_overlap_trimmed_chars']} | "
        f"embedding_gate_evaluations={stats['embedding_gate_evaluations']} | "
        f"embedding_gate_hits={stats['embedding_gate_hits']} | "
        f"embedding_gate_continuations={stats['embedding_gate_continuations']} | "
        f"embedding_gate_breaks={stats['embedding_gate_breaks']}"
        f" | embedding_gate_neutral={stats['embedding_gate_neutral']}"
    )

    return chunks, stats
