# main.py

import uuid
import time
import os
import json
import re
import math

import numpy as np

from parser import (
    extract_pages,
    detect_document_type,
    check_document_size,
    filter_pages,
    create_windows
)

from semantic_chunking import build_semantic_chunks, build_legacy_segment_windows
from db import DocumentDB
from rag_engine import RAGEngine, model as embedding_model
from ollama_parser import extract_metadata
from config import USE_LEGACY_SEGMENT_CHUNKING


def _normalize_estimated_time(value, default=30):
    if isinstance(value, (int, float)):
        return int(value)

    if isinstance(value, str):
        nums = [int(n) for n in re.findall(r"\d+", value)]
        if len(nums) >= 2:
            return int(sum(nums[:2]) / 2)
        if len(nums) == 1:
            return nums[0]

    return default


def _normalize_complexity(value, default="intermediate"):
    allowed = {"beginner", "intermediate", "advanced"}
    if not value:
        return default

    lowered = str(value).lower()
    tokens = re.findall(r"[a-zA-Z]+", lowered)
    seen = [t for t in tokens if t in allowed]

    if "advanced" in seen:
        return "advanced"
    if "intermediate" in seen:
        return "intermediate"
    if "beginner" in seen:
        return "beginner"

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


def _estimate_study_time_minutes(text, complexity="intermediate", concept_count=0):
    words = len(str(text).split())
    reading_minutes = words / 115.0

    # Lightweight signal for mathematically dense chunks.
    symbol_hits = len(re.findall(r"[=+\-*/^]|\d", str(text)))
    density_bonus = min(12, symbol_hits // 35)

    complexity_bonus = {
        "beginner": 0,
        "intermediate": 5,
        "advanced": 10,
    }.get(complexity, 5)

    concept_bonus = min(max(int(concept_count), 0) * 2, 20)

    estimate = reading_minutes + complexity_bonus + concept_bonus + density_bonus
    rounded_to_5 = int(math.ceil(max(10, estimate) / 5.0) * 5)
    return min(90, rounded_to_5)


def _count_text_metrics(text):
    text = str(text or "")
    return {
        "chars": len(text),
        "words": len(text.split()),
    }


def _init_fallback_report(legacy_segment_mode=False):
    return {
        "small_doc_fallback_count": 0,
        "chunk_llm_failure_count": 0,
        "page_filter_llm_default_count": 0,
        "page_filter_toc_llm_default_count": 0,
        "page_filter_short_page_llm_default_count": 0,
        "legacy_segment_mode": bool(legacy_segment_mode),
    }


def _chunk_metadata_source_counts(chunks):
    counts = {}
    for chunk in chunks:
        metadata = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
        source = str(metadata.get("metadata_source", "unknown")).strip() or "unknown"
        counts[source] = counts.get(source, 0) + 1
    return counts


def _merge_ranges(ranges):
    valid = sorted((s, e) for s, e in ranges if e > s)
    if not valid:
        return []

    merged = [valid[0]]
    for start, end in valid[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _unique_covered_chars_from_chunks(chunks, windows):
    if not windows:
        return 0

    window_lengths = {w.get("window_id", i): len(str(w.get("text", ""))) for i, w in enumerate(windows)}
    ranges_by_window = {}

    for chunk in chunks:
        meta = chunk.get("metadata", {})
        coverage = meta.get("coverage", {}) if isinstance(meta, dict) else {}
        if not isinstance(coverage, dict):
            continue

        window_id = coverage.get("window_id", meta.get("window_id"))
        if window_id not in window_lengths:
            continue

        start = coverage.get("start_char")
        end = coverage.get("end_char")
        if not isinstance(start, int) or not isinstance(end, int):
            continue

        window_len = window_lengths[window_id]
        start = max(0, min(window_len, start))
        end = max(0, min(window_len, end))
        if end <= start:
            continue

        ranges_by_window.setdefault(window_id, []).append((start, end))

    total = 0
    for window_id, ranges in ranges_by_window.items():
        merged = _merge_ranges(ranges)
        total += sum(end - start for start, end in merged)

    return total


def _summarize_chunk_coverage(source_text, chunks, windows=None):
    source_metrics = _count_text_metrics(source_text)
    combined_text = "\n\n".join(str(chunk.get("text", "")) for chunk in chunks)
    chunk_metrics = _count_text_metrics(combined_text)

    net_char_delta = chunk_metrics["chars"] - source_metrics["chars"]
    net_word_delta = chunk_metrics["words"] - source_metrics["words"]

    char_delta_pct = (net_char_delta / source_metrics["chars"] * 100.0) if source_metrics["chars"] else 0.0
    word_delta_pct = (net_word_delta / source_metrics["words"] * 100.0) if source_metrics["words"] else 0.0

    result = {
        "source_chars": source_metrics["chars"],
        "source_words": source_metrics["words"],
        "chunk_chars": chunk_metrics["chars"],
        "chunk_words": chunk_metrics["words"],
        "net_char_delta": net_char_delta,
        "net_word_delta": net_word_delta,
        "net_char_delta_pct": round(char_delta_pct, 2),
        "net_word_delta_pct": round(word_delta_pct, 2),
    }

    if windows:
        window_source_chars = sum(len(str(w.get("text", ""))) for w in windows)
        unique_covered_chars = _unique_covered_chars_from_chunks(chunks, windows)
        coverage_pct = (unique_covered_chars / window_source_chars * 100.0) if window_source_chars else 0.0
        duplication_chars = max(0, chunk_metrics["chars"] - unique_covered_chars)
        duplication_pct = (duplication_chars / chunk_metrics["chars"] * 100.0) if chunk_metrics["chars"] else 0.0
        expansion_pct = (chunk_metrics["chars"] / window_source_chars * 100.0) if window_source_chars else 0.0

        result.update({
            "window_source_chars": window_source_chars,
            "unique_covered_chars": unique_covered_chars,
            "uncovered_window_chars": max(0, window_source_chars - unique_covered_chars),
            "window_coverage_pct": round(coverage_pct, 2),
            "duplication_chars": duplication_chars,
            "duplication_pct_of_chunk": round(duplication_pct, 2),
            "expansion_pct_vs_window": round(expansion_pct, 2),
        })

    return result


def _summarize_ingest_coverage(raw_text, filtered_text, chunks, windows=None):
    raw_stats = _count_text_metrics(raw_text)
    filtered_stats = _count_text_metrics(filtered_text)
    chunk_stats = _summarize_chunk_coverage(filtered_text, chunks, windows=windows)

    raw_char_retention = (chunk_stats["chunk_chars"] / raw_stats["chars"] * 100.0) if raw_stats["chars"] else 0.0
    raw_word_retention = (chunk_stats["chunk_words"] / raw_stats["words"] * 100.0) if raw_stats["words"] else 0.0
    filtered_char_retention = (filtered_stats["chars"] / raw_stats["chars"] * 100.0) if raw_stats["chars"] else 0.0
    filtered_word_retention = (filtered_stats["words"] / raw_stats["words"] * 100.0) if raw_stats["words"] else 0.0

    return {
        "raw_source_chars": raw_stats["chars"],
        "raw_source_words": raw_stats["words"],
        "filtered_source_chars": filtered_stats["chars"],
        "filtered_source_words": filtered_stats["words"],
        "chunk_chars": chunk_stats["chunk_chars"],
        "chunk_words": chunk_stats["chunk_words"],
        "filtered_char_retention_pct": round(filtered_char_retention, 2),
        "filtered_word_retention_pct": round(filtered_word_retention, 2),
        "raw_char_retention_pct": round(raw_char_retention, 2),
        "raw_word_retention_pct": round(raw_word_retention, 2),
        "net_char_delta_vs_raw": chunk_stats["chunk_chars"] - raw_stats["chars"],
        "net_word_delta_vs_raw": chunk_stats["chunk_words"] - raw_stats["words"],
        "net_char_delta_vs_filtered": chunk_stats["net_char_delta"],
        "net_word_delta_vs_filtered": chunk_stats["net_word_delta"],
        "filtered_coverage": chunk_stats,
    }


def _session_index_path(session_id):
    return f"indexes/sessions/{session_id}"


def _session_index_exists(index_path):
    return os.path.exists(f"{index_path}.index") and os.path.exists(f"{index_path}.pkl")


def _prepare_session_chunks(chunks, source_doc_id, session_id):
    prepared = []
    for chunk in chunks:
        metadata = dict(chunk.get("metadata", {}))
        metadata["source_doc_id"] = source_doc_id
        metadata["session_id"] = session_id
        prepared.append({
            "text": str(chunk.get("text", "")),
            "metadata": metadata,
        })
    return prepared


def _collect_session_topics_and_summary(session_chunk_rows):
    topic_seen = set()
    summary_seen = set()
    topics = []
    summary_parts = []

    for _chunk_id, _doc_id, _content, metadata_raw in session_chunk_rows:
        metadata = {}
        try:
            metadata = json.loads(metadata_raw) if metadata_raw else {}
        except Exception:
            metadata = {}

        topic = str(metadata.get("topic", "")).strip()
        if topic:
            key = topic.lower()
            if key not in topic_seen:
                topic_seen.add(key)
                topics.append(topic)

        summary = str(metadata.get("summary", "")).strip()
        if summary:
            key = summary.lower()
            if key not in summary_seen:
                summary_seen.add(key)
                summary_parts.append(summary)

    topic_clause = "Topics covered: " + "; ".join(topics[:12]) if topics else ""

    if summary_parts:
        summary_body = " ".join(summary_parts[:12]).strip()
        if topic_clause:
            session_summary = f"{topic_clause}. {summary_body}" if summary_body else topic_clause
        else:
            session_summary = summary_body
    elif topic_clause:
        session_summary = topic_clause
    else:
        session_summary = ""

    if len(session_summary) > 2200:
        session_summary = session_summary[:2200].rstrip()

    return topics, session_summary


def _refresh_session_profile(db, session_id, rag_index_path=None):
    session_rows = db.get_session_chunks(session_id)
    topics, summary = _collect_session_topics_and_summary(session_rows)
    db.update_session_profile(
        session_id,
        topics=topics,
        summary=summary,
        rag_index_path=rag_index_path,
    )
    return {
        "topics": topics,
        "summary": summary,
    }


def _update_session_rag(session_id, source_doc_id, chunks):
    if not session_id or not chunks:
        return None

    index_path = _session_index_path(session_id)
    os.makedirs("indexes/sessions", exist_ok=True)

    if _session_index_exists(index_path):
        try:
            session_rag = RAGEngine(index_path=index_path)
        except Exception as exc:
            print(f"⚠️ Session index reload failed, recreating index: {exc}")
            session_rag = RAGEngine()
    else:
        session_rag = RAGEngine()

    session_chunks = _prepare_session_chunks(chunks, source_doc_id=source_doc_id, session_id=session_id)
    session_rag.add_chunks(session_chunks)
    session_rag.save(index_path)
    return index_path


def _pages_are_near(pages_a, pages_b, gap=2):
    if not pages_a or not pages_b:
        return True

    a_min, a_max = min(pages_a), max(pages_a)
    b_min, b_max = min(pages_b), max(pages_b)

    return not (a_max + gap < b_min or b_max + gap < a_min)


def _deduplicate_chunks_semantic(chunks, similarity_threshold=0.88, page_gap=2):
    if len(chunks) < 2:
        return chunks

    texts = [c.get("text", "") for c in chunks]

    try:
        embeddings = np.array(embedding_model.encode(texts), dtype=np.float32)
    except Exception as e:
        print(f"⚠️ Semantic dedup disabled (embedding error): {e}")
        return chunks

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = embeddings / norms

    duplicate_indexes = set()
    merged_with = {}  # Track which chunks were merged into which
    kept = []

    for i, chunk_i in enumerate(chunks):
        if i in duplicate_indexes:
            continue

        meta_i = chunk_i.get("metadata", {})
        pages_i = meta_i.get("pages", [])

        for j in range(i + 1, len(chunks)):
            if j in duplicate_indexes:
                continue

            chunk_j = chunks[j]
            meta_j = chunk_j.get("metadata", {})
            pages_j = meta_j.get("pages", [])

            if not _pages_are_near(pages_i, pages_j, gap=page_gap):
                continue

            similarity = float(np.dot(normalized[i], normalized[j]))

            if similarity >= similarity_threshold:
                # Merge chunk_j's text into chunk_i before discarding
                text_j = chunk_j.get("text", "")
                if text_j and text_j not in chunk_i.get("text", ""):
                    chunk_i["text"] = chunk_i.get("text", "") + "\n\n[MERGED DUPLICATE]\n" + text_j
                
                # Track the merge for audit
                if "merged_chunks" not in meta_i:
                    meta_i["merged_chunks"] = []
                meta_i["merged_chunks"].append(chunk_j.get("id", f"chunk_{j}"))
                
                duplicate_indexes.add(j)
                merged_with[j] = i

        kept.append(chunk_i)

    return kept


def _normalize_topic_signature(topic, subtopics):
    parts = [str(topic or "").lower().strip()]
    if isinstance(subtopics, list):
        parts.extend(str(s).lower().strip() for s in subtopics)

    text = " ".join(parts)
    tokens = re.findall(r"[a-zA-Z]+", text)
    stopwords = {
        "the", "and", "or", "of", "to", "in", "for", "on", "with",
        "a", "an", "by", "from", "is", "are", "their", "its", "value",
        "values", "theory", "introduction"
    }
    return {t for t in tokens if len(t) > 2 and t not in stopwords}


def _jaccard_similarity(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _deduplicate_chunks_by_topic(enriched_chunks, similarity_threshold=0.60, page_gap=3):
    if len(enriched_chunks) < 2:
        return enriched_chunks

    duplicate_indexes = set()
    kept = []

    signatures = []
    topic_documents = []
    text_documents = []
    for c in enriched_chunks:
        meta = c.get("metadata", {})
        topic = meta.get("topic", "")
        subtopics = meta.get("subtopics", [])
        signatures.append(_normalize_topic_signature(topic, subtopics))

        if isinstance(subtopics, list):
            subtopic_text = "; ".join(str(s) for s in subtopics)
        else:
            subtopic_text = ""
        topic_documents.append(f"{topic}. {subtopic_text}".strip())
        text_documents.append(str(c.get("text", "")))

    try:
        topic_emb = np.array(embedding_model.encode(topic_documents), dtype=np.float32)
        topic_norms = np.linalg.norm(topic_emb, axis=1, keepdims=True)
        topic_norms[topic_norms == 0] = 1.0
        topic_emb = topic_emb / topic_norms
    except Exception as e:
        print(f"⚠️ Topic embedding dedup fallback (embedding error): {e}")
        topic_emb = None

    try:
        text_emb = np.array(embedding_model.encode(text_documents), dtype=np.float32)
        text_norms = np.linalg.norm(text_emb, axis=1, keepdims=True)
        text_norms[text_norms == 0] = 1.0
        text_emb = text_emb / text_norms
    except Exception as e:
        print(f"⚠️ Text embedding dedup fallback (embedding error): {e}")
        text_emb = None

    for i, chunk_i in enumerate(enriched_chunks):
        if i in duplicate_indexes:
            continue

        meta_i = chunk_i.get("metadata", {})
        pages_i = meta_i.get("pages", [])
        text_i_len = len(chunk_i.get("text", ""))
        subtopics_i = meta_i.get("subtopics", []) if isinstance(meta_i.get("subtopics", []), list) else []

        for j in range(i + 1, len(enriched_chunks)):
            if j in duplicate_indexes:
                continue

            chunk_j = enriched_chunks[j]
            meta_j = chunk_j.get("metadata", {})
            pages_j = meta_j.get("pages", [])

            if not _pages_are_near(pages_i, pages_j, gap=page_gap):
                continue

            topic_sim = _jaccard_similarity(signatures[i], signatures[j])
            emb_sim = None
            if topic_emb is not None:
                emb_sim = float(np.dot(topic_emb[i], topic_emb[j]))

            text_sim = None
            if text_emb is not None:
                text_sim = float(np.dot(text_emb[i], text_emb[j]))

            similar_by_tokens = topic_sim >= similarity_threshold
            similar_by_embedding = emb_sim is not None and emb_sim >= 0.86
            similar_by_text = text_sim is not None and text_sim >= 0.90
            if not (similar_by_tokens or similar_by_embedding or similar_by_text):
                continue

            text_j_len = len(chunk_j.get("text", ""))
            subtopics_j = meta_j.get("subtopics", []) if isinstance(meta_j.get("subtopics", []), list) else []

            score_i = len(signatures[i]) + len(subtopics_i)
            score_j = len(signatures[j]) + len(subtopics_j)

            # Prefer the more specific chunk; if tied, keep the longer text chunk.
            kept_idx = None
            removed_idx = None
            if score_i > score_j:
                duplicate_indexes.add(j)
                kept_idx = i
                removed_idx = j
            elif score_j > score_i:
                duplicate_indexes.add(i)
                kept_idx = j
                removed_idx = i
            else:
                if text_i_len >= text_j_len:
                    duplicate_indexes.add(j)
                    kept_idx = i
                    removed_idx = j
                else:
                    duplicate_indexes.add(i)
                    kept_idx = j
                    removed_idx = i

            # Merge the removed chunk's text into the kept chunk
            if kept_idx is not None and removed_idx is not None:
                kept_chunk = enriched_chunks[kept_idx]
                removed_chunk = enriched_chunks[removed_idx]
                
                text_removed = removed_chunk.get("text", "")
                if text_removed and text_removed not in kept_chunk.get("text", ""):
                    kept_chunk["text"] = kept_chunk.get("text", "") + "\n\n[MERGED DUPLICATE]\n" + text_removed
                
                # Track the merge for audit
                meta_kept = kept_chunk.get("metadata", {})
                if "merged_chunks" not in meta_kept:
                    meta_kept["merged_chunks"] = []
                meta_kept["merged_chunks"].append(removed_chunk.get("id", f"chunk_{removed_idx}"))

        if i not in duplicate_indexes:
            kept.append(chunk_i)

    return kept


def _normalize_subtopics(value):
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    out.append(text)
            elif isinstance(item, dict):
                text = (
                    item.get("title")
                    or item.get("name")
                    or item.get("subtopic")
                    or item.get("description")
                    or ""
                )
                text = str(text).strip()
                if text:
                    out.append(text)
        return out

    return []


def _normalize_metadata(extracted_meta, fallback_meta, chunk_text):
    meta = extracted_meta if isinstance(extracted_meta, dict) else {}
    fallback = fallback_meta if isinstance(fallback_meta, dict) else {}

    has_llm_meta = bool(meta)

    topic_raw = str(meta.get("topic") or "").strip()
    summary_raw = str(meta.get("summary") or "").strip()
    complexity_raw = meta.get("complexity")
    estimated_time_raw = meta.get("estimated_time")

    topic = str(topic_raw or fallback.get("topic") or "Unknown Topic").strip()
    complexity = _normalize_complexity(complexity_raw, default="intermediate")
    key_concepts = _normalize_subtopics(meta.get("key_concepts"))

    heuristic_time = _estimate_study_time_minutes(
        text=chunk_text,
        complexity=complexity,
        concept_count=len(key_concepts),
    )

    llm_time = _normalize_estimated_time(estimated_time_raw, default=None)
    if llm_time is None:
        estimated_time = heuristic_time
    else:
        llm_time = max(5, min(180, int(llm_time)))
        # Blend model time with deterministic estimate to avoid flat defaults.
        estimated_time = int(round(0.35 * llm_time + 0.65 * heuristic_time))
        estimated_time = int(math.ceil(estimated_time / 5.0) * 5)
        estimated_time = max(10, min(180, estimated_time))

    summary = summary_raw
    if not summary and fallback.get("boundary_source") == "llm_uncovered_fallback":
        summary = "Auto-created fallback chunk from text not covered by LLM boundaries."

    subtopics = _normalize_subtopics(meta.get("subtopics"))
    summary = _build_topics_covered_summary(topic, subtopics, summary)

    metadata_fallback_used = (
        (not has_llm_meta)
        or (not topic_raw)
        or (not summary_raw)
        or (complexity_raw in (None, ""))
        or (estimated_time_raw in (None, ""))
    )
    metadata_source = "fallback" if not has_llm_meta else ("llm+fallback" if metadata_fallback_used else "llm")

    normalized = {
        "topic": topic,
        "summary": summary,
        "subtopics": subtopics,
        "key_concepts": key_concepts,
        "complexity": complexity,
        "prerequisites": _normalize_subtopics(meta.get("prerequisites")),
        "estimated_time": estimated_time,
        "pages": fallback.get("pages", []),
        "metadata_source": metadata_source,
        "llm_failed": not has_llm_meta,
        "metadata_fallback_used": metadata_fallback_used,
    }

    for optional_key in ["window_id", "boundary_source", "coverage", "confidence", "description"]:
        if optional_key in fallback:
            normalized[optional_key] = fallback.get(optional_key)

    return normalized


# -----------------------------
# FULL PIPELINE (Ollama + chunks)
# -----------------------------

def full_pipeline(windows, source_text=None, use_legacy_segment_chunking=False):
    active_windows = windows
    legacy_segment_stats = None

    if use_legacy_segment_chunking:
        print(f"✂️ Legacy segment chunking enabled for {len(windows)} windows...")
        active_windows, legacy_segment_stats = build_legacy_segment_windows(windows)
        print(
            "✂️ Legacy segment prep: "
            f"output_windows={legacy_segment_stats.get('output_windows', 0)} | "
            f"windows_with_llm_segments={legacy_segment_stats.get('windows_with_llm_segments', 0)} | "
            f"windows_with_segment_fallback={legacy_segment_stats.get('windows_with_segment_fallback', 0)}"
        )

    print(f"✂️ Building fixed hierarchical chunks for {len(active_windows)} windows...")
    chunk_result = build_semantic_chunks(active_windows)
    if isinstance(chunk_result, tuple):
        chunks, chunking_stats = chunk_result
    else:
        chunks = chunk_result
        chunking_stats = {}
    print(f"✂️ Hierarchical chunks built: {len(chunks)}")
    if chunking_stats:
        print(
            "🧩 Chunk log: "
            f"input_windows={chunking_stats.get('input_windows', 0)} | "
            f"output_chunks={chunking_stats.get('output_chunks', 0)} | "
            f"topic_groups={chunking_stats.get('topic_groups', 0)} | "
            f"topic_continuations={chunking_stats.get('topic_continuations', 0)} | "
            f"topic_resets={chunking_stats.get('topic_resets', 0)} | "
            f"llm_failures={chunking_stats.get('llm_failures', 0)}"
        )

    source_text = source_text if source_text is not None else "\n\n".join(w.get("text", "") for w in windows)
    coverage_stats = _summarize_chunk_coverage(source_text, chunks, windows=windows)
    print(
        "📏 Chunk stats: "
        f"source_chars={coverage_stats['source_chars']} | source_words={coverage_stats['source_words']} | "
        f"chunk_chars={coverage_stats['chunk_chars']} | chunk_words={coverage_stats['chunk_words']} | "
        f"net_char_delta={coverage_stats['net_char_delta']} ({coverage_stats['net_char_delta_pct']}%) | "
        f"net_word_delta={coverage_stats['net_word_delta']} ({coverage_stats['net_word_delta_pct']}%)"
    )

    if "window_coverage_pct" in coverage_stats:
        print(
            "📏 Coverage quality: "
            f"window_coverage={coverage_stats['window_coverage_pct']}% | "
            f"duplication_in_chunks={coverage_stats['duplication_pct_of_chunk']}% | "
            f"expansion_vs_window={coverage_stats['expansion_pct_vs_window']}%"
        )

    chunking_stats = chunking_stats or {}
    chunking_stats["coverage"] = coverage_stats

    if use_legacy_segment_chunking:
        chunking_stats["legacy_segment_stats"] = legacy_segment_stats or {}

        for idx, chunk in enumerate(chunks):
            if idx >= len(active_windows):
                break

            legacy_meta = active_windows[idx].get("_legacy_segment_meta", {})
            if not isinstance(legacy_meta, dict):
                continue

            metadata = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
            if not isinstance(metadata, dict):
                continue

            for key in ["boundary_source", "coverage", "confidence", "description"]:
                value = legacy_meta.get(key)
                if value is not None:
                    metadata[key] = value

            if "original_window_id" in legacy_meta:
                metadata["window_id"] = legacy_meta.get("original_window_id")

            metadata["legacy_segmented"] = True

    return chunks, chunking_stats


# -----------------------------
# MAIN ROUTER
# -----------------------------
def process_document(pdf_path, session_id=None, session_title=None):
    db = DocumentDB()
    rag = RAGEngine()

    if session_id or session_title:
        session_id = db.ensure_session(session_id=session_id, title=session_title)

    doc_id = str(uuid.uuid4())
    session_rag_index_path = None
    session_profile = None
    fallback_report = _init_fallback_report(
        legacy_segment_mode=USE_LEGACY_SEGMENT_CHUNKING,
    )

    print("📄 Extracting pages...")
    pages = extract_pages(pdf_path)

    if not pages:
        raise ValueError("No text could be extracted from document.")

    # -------------------------
    # 1. SIZE CHECK (FIRST)
    # -------------------------
    size_info = check_document_size(pages)

    if size_info["is_small"]:
        print(f"⚡ Small document detected ({size_info['total_chars']} chars)")

        small_doc_text = str(size_info["full_text"])
        fallback_summary = small_doc_text[:300]
        fallback_metadata = {
            "topic": f"Full Document: {os.path.basename(pdf_path)}",
            "summary": fallback_summary,
            "subtopics": [],
            "key_concepts": [],
            "complexity": "intermediate",
            "prerequisites": [],
            "estimated_time": _estimate_study_time_minutes(small_doc_text),
            "pages": [p["page_num"] for p in pages if "page_num" in p],
            "window_id": 0,
            "boundary_source": "small_doc_full_context",
            "coverage": {
                "window_id": 0,
                "start_char": 0,
                "end_char": len(small_doc_text),
                "start_percent": 0.0,
                "end_percent": 100.0,
                "boundary_method": "full_document",
                "start_match": "n/a",
                "end_match": "n/a",
            },
        }

        print("🧠 Enriching small document metadata with LLM...")
        try:
            extracted_small_meta = extract_metadata(small_doc_text)
        except Exception as exc:
            print(f"⚠️ Small-doc metadata extraction failed, using fallback metadata: {exc}")
            extracted_small_meta = {}

        small_doc_metadata = _normalize_metadata(
            extracted_meta=extracted_small_meta,
            fallback_meta=fallback_metadata,
            chunk_text=small_doc_text,
        )
        small_doc_metadata["summary"] = str(small_doc_metadata.get("summary") or fallback_summary).strip()
        small_doc_metadata["group_index"] = 0
        small_doc_metadata["same_topic_as_previous"] = False
        small_doc_metadata["topic_continues"] = False

        fallback_report["small_doc_fallback_count"] = int(
            small_doc_metadata.get("metadata_source") != "llm"
        )
        fallback_report["chunk_llm_failure_count"] = int(
            bool(small_doc_metadata.get("llm_failed"))
        )

        print(
            "🧾 Fallback report: "
            f"small_doc_fallback_count={fallback_report['small_doc_fallback_count']} | "
            f"chunk_llm_failure_count={fallback_report['chunk_llm_failure_count']} | "
            f"page_filter_llm_default_count={fallback_report['page_filter_llm_default_count']}"
        )

        small_doc_chunk = {
            "chunk_id": "chunk_0",
            "text": small_doc_text,
            "metadata": small_doc_metadata,
        }

        db.store_document(
            doc_id,
            strategy="full_context",
            doc_type="small_doc",
            total_chars=size_info["total_chars"],
            session_id=session_id,
            ingest_report=fallback_report,
        )
        db.store_chunks(doc_id, [small_doc_chunk])

        if session_id:
            session_rag_index_path = _update_session_rag(session_id, source_doc_id=doc_id, chunks=[small_doc_chunk])
            session_profile = _refresh_session_profile(db, session_id, rag_index_path=session_rag_index_path)

        return {
            "doc_id": doc_id,
            "session_id": session_id,
            "strategy": "full_context",
            "text": size_info["full_text"],
            "fallback_report": fallback_report,
            "session_rag_index_path": session_rag_index_path,
            "session_topic_count": len(session_profile.get("topics", [])) if session_profile else 0,
            "session_summary": session_profile.get("summary") if session_profile else "",
        }

    # -------------------------
    # 2. DOCUMENT TYPE
    # -------------------------
    doc_type = detect_document_type(pages)
    print(f"📊 Document type: {doc_type}")

    # -------------------------
    # 3. FILTERING
    # -------------------------
    filtered_pages, page_filter_stats = filter_pages(pages, doc_type, return_stats=True)
    fallback_report.update({
        "page_filter_llm_default_count": int(page_filter_stats.get("page_filter_llm_default_count", 0)),
        "page_filter_toc_llm_default_count": int(page_filter_stats.get("page_filter_toc_llm_default_count", 0)),
        "page_filter_short_page_llm_default_count": int(page_filter_stats.get("page_filter_short_page_llm_default_count", 0)),
    })
    print(f"🧹 Pages after filtering: {len(filtered_pages)} / {len(pages)}")

    if not filtered_pages:
        raise ValueError("All pages filtered out. Check filtering rules.")

    # -------------------------
    # 4. WINDOWING
    # -------------------------
    windows = create_windows(filtered_pages, doc_type)
    print(f"🪟 Created {len(windows)} windows")

    filtered_source_text = "\n\n".join(p["text"] for p in filtered_pages)
    raw_source_text = size_info["full_text"]

    # -------------------------
    # 5. FULL PIPELINE
    # -------------------------
    chunks, chunking_stats = full_pipeline(
        windows,
        source_text=filtered_source_text,
        use_legacy_segment_chunking=USE_LEGACY_SEGMENT_CHUNKING,
    )
    print(f"📦 Generated {len(chunks)} chunks")

    fallback_report["chunk_llm_failure_count"] = int(chunking_stats.get("llm_failures", 0))
    fallback_report["chunk_metadata_source_counts"] = _chunk_metadata_source_counts(chunks)

    ingest_coverage = _summarize_ingest_coverage(raw_source_text, filtered_source_text, chunks, windows=windows)
    print(
        "📊 Ingest coverage: "
        f"raw_words={ingest_coverage['raw_source_words']} -> chunk_words={ingest_coverage['chunk_words']} | "
        f"raw_chars={ingest_coverage['raw_source_chars']} -> chunk_chars={ingest_coverage['chunk_chars']} | "
        f"filter_retention_words={ingest_coverage['filtered_word_retention_pct']}% | "
        f"filter_retention_chars={ingest_coverage['filtered_char_retention_pct']}% | "
        f"raw_net_char_delta={ingest_coverage['net_char_delta_vs_raw']}"
    )

    print(
        "🧾 Fallback report: "
        f"small_doc_fallback_count={fallback_report['small_doc_fallback_count']} | "
        f"chunk_llm_failure_count={fallback_report['chunk_llm_failure_count']} | "
        f"page_filter_llm_default_count={fallback_report['page_filter_llm_default_count']} | "
        f"legacy_segment_mode={fallback_report['legacy_segment_mode']}"
    )

    if not chunks:
        raise ValueError("No chunks generated. Segment detection/metadata extraction returned empty results.")
    # -------------------------
    # 6. STORE IN DB
    # -------------------------
    index_path = f"indexes/{doc_id}"

    os.makedirs("indexes", exist_ok=True)

    if not chunks:
        raise ValueError("No chunks to index.")

    print(f"📚 Building and saving FAISS index at {index_path}...")

    rag.add_chunks(chunks)
    rag.save(index_path)

    db.store_document(
        doc_id,
        strategy="rag",
        doc_type=doc_type,
        total_chars=size_info["total_chars"],
        index_path=index_path,
        session_id=session_id,
        ingest_report=fallback_report,
    )

    db.store_chunks(doc_id, chunks)

    if session_id:
        session_rag_index_path = _update_session_rag(session_id, source_doc_id=doc_id, chunks=chunks)
        session_profile = _refresh_session_profile(db, session_id, rag_index_path=session_rag_index_path)

    # -------------------------
    # 7. BUILD RAG INDEX
    # -------------------------
    
    return {
        "doc_id": doc_id,
        "session_id": session_id,
        "strategy": "rag",
        "doc_type": doc_type,
        "chunks": len(chunks),
        "chunking_stats": chunking_stats,
        "ingest_coverage": ingest_coverage,
        "fallback_report": fallback_report,
        "coverage": chunking_stats.get("coverage", {}),
        "session_rag_index_path": session_rag_index_path,
        "session_topic_count": len(session_profile.get("topics", [])) if session_profile else 0,
        "session_summary": session_profile.get("summary") if session_profile else "",
    }


def process_documents(pdf_paths, session_id=None, session_title=None, continue_on_error=True):
    if not pdf_paths:
        raise ValueError("No PDF paths were provided.")

    db = DocumentDB()
    resolved_session_id = session_id
    if resolved_session_id or session_title:
        resolved_session_id = db.ensure_session(
            session_id=resolved_session_id,
            title=session_title,
        )
    else:
        resolved_session_id = db.create_session(title=session_title or "Study Session")

    processed = []
    failures = []

    total_files = len(pdf_paths)
    print(
        f"🗂️ Starting multi-document ingest: files={total_files} | session_id={resolved_session_id}"
    )

    for idx, path in enumerate(pdf_paths, 1):
        print(f"\n{'=' * 80}")
        print(f"[{idx}/{total_files}] Processing: {path}")
        print(f"{'=' * 80}")

        try:
            result = process_document(
                path,
                session_id=resolved_session_id,
                session_title=session_title,
            )
            processed.append(result)
        except Exception as exc:
            failure = {
                "pdf_path": path,
                "error": str(exc),
            }
            failures.append(failure)
            print(f"❌ Failed: {path} | {exc}")
            if not continue_on_error:
                break

    session_profile = db.get_session_profile(resolved_session_id)

    return {
        "session_id": resolved_session_id,
        "session_title": session_title,
        "requested_files": total_files,
        "processed_files": len(processed),
        "failed_files": len(failures),
        "documents": processed,
        "failures": failures,
        "session_profile": session_profile,
    }


# -----------------------------
# OPTIONAL: QUERY FUNCTION
# -----------------------------
def query_document(query, rag_engine, top_k=5):
    results = rag_engine.retrieve(query, top_k=top_k)

    print("\n🔎 Retrieved Chunks:\n")
    for i, r in enumerate(results):
        print(f"[{i+1}] Topic: {r['metadata'].get('topic', 'N/A')}")
        print(r["text"][:300])
        print("-" * 50)

    return results


# -----------------------------
# RUN SCRIPT
# -----------------------------
if __name__ == "__main__":
    data_dir = r"data\cv"
    pdf_paths = [
        os.path.join(data_dir, name)
        for name in sorted(os.listdir(data_dir))
        if name.lower().endswith(".pdf")
    ]

    if not pdf_paths:
        raise ValueError(f"No PDF files found in '{data_dir}'.")

    print(f"📂 Found {len(pdf_paths)} PDF(s) in '{data_dir}'")
    result = process_documents(pdf_paths, session_title="Default Study Session")

    print("\n✅ Processing Complete:")
    print(result)