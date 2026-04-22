#!/usr/bin/env python
"""
Detailed chunk inspector for hierarchical chunking.

Shows:
- document-level diagnostics (topic groups, transitions, boundary sources)
- per-chunk hierarchical fields (group_index, continuation flags)
- coverage and preview for quick debugging
"""

import json
import re
from collections import Counter

from db import DocumentDB


def _safe_json_loads(value):
    try:
        data = json.loads(value)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _chunk_sort_key(storage_chunk_id):
    # Stored chunk ids are prefixed as "{doc_id}:chunk_{n}".
    match = re.search(r"chunk_(\d+)$", str(storage_chunk_id))
    if match:
        return int(match.group(1))
    return 10**9


def _preview(text, limit=260):
    flat = str(text or "").replace("\n", " ").strip()
    return flat[:limit] + ("..." if len(flat) > limit else "")


def _print_document_summary(rows):
    metas = [row[3] for row in rows]

    source_counts = Counter(str(meta.get("boundary_source", "unknown")) for meta in metas)
    topic_counts = Counter(str(meta.get("topic", "Unknown")).strip() or "Unknown" for meta in metas)
    group_counts = Counter(meta.get("group_index", "n/a") for meta in metas)

    continues_true = sum(1 for meta in metas if meta.get("topic_continues") is True)
    same_prev_true = sum(1 for meta in metas if meta.get("same_topic_as_previous") is True)

    print("Document diagnostics:")
    print(f"  Topic groups (group_index): {len(group_counts)}")
    print(f"  Unique topic labels:        {len(topic_counts)}")
    print(f"  topic_continues=True:       {continues_true}")
    print(f"  same_topic_as_previous=True:{same_prev_true}")

    print("  Boundary sources:")
    for source, count in sorted(source_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"    - {source}: {count}")

    print("  Top topic labels:")
    for topic, count in topic_counts.most_common(6):
        print(f"    - {topic[:70]}: {count}")

    if len(group_counts) == 1 and len(rows) > 3:
        print("  NOTE: All chunks are in one topic group. Inspect continuation flags below.")


def main():
    db = DocumentDB()

    docs = db.list_documents()
    if not docs:
        print("No documents in database.")
        return

    print("\nAvailable Documents:\n")
    for i, (doc_id, strategy, doc_type) in enumerate(docs):
        print(f"[{i}] {doc_id[:8]}... | Strategy: {strategy} | Type: {doc_type}")

    try:
        choice = int(input("\nSelect document (number): "))
        doc_id, strategy, doc_type = docs[choice]
    except Exception:
        print("Invalid selection.")
        return

    print(f"\n{'=' * 90}")
    print(f"DOCUMENT: {doc_id} | Strategy: {strategy} | Type: {doc_type}")
    print(f"{'=' * 90}\n")

    cursor = db.conn.execute(
        "SELECT chunk_id, metadata, content FROM chunks WHERE doc_id=?",
        (doc_id,),
    )

    rows_raw = cursor.fetchall()
    if not rows_raw:
        print("No chunks found for this document.")
        return

    rows = []
    for chunk_id, metadata_json, content in rows_raw:
        meta = _safe_json_loads(metadata_json)
        rows.append((chunk_id, metadata_json, content or "", meta))

    rows.sort(key=lambda item: _chunk_sort_key(item[0]))

    print(f"Total chunks: {len(rows)}\n")
    _print_document_summary(rows)

    previous_group = None
    previous_topic = None

    for idx, (chunk_id, _metadata_json, chunk_text, meta) in enumerate(rows, 1):
        topic = str(meta.get("topic", "N/A"))
        summary = str(meta.get("summary", "N/A"))
        complexity = meta.get("complexity", "N/A")
        estimated_time = meta.get("estimated_time", "N/A")
        pages = meta.get("pages", [])

        boundary_source = str(meta.get("boundary_source", "unknown"))
        group_index = meta.get("group_index", "n/a")
        continues = meta.get("topic_continues", "N/A")
        same_prev = meta.get("same_topic_as_previous", "N/A")

        transition = ""
        if previous_group is not None and group_index != previous_group:
            transition = "<-- GROUP CHANGE"
        elif previous_topic is not None and topic != previous_topic:
            transition = "<-- TOPIC LABEL CHANGE"

        print(f"\n{'-' * 90}")
        print(f"[{idx}] Chunk ID: {chunk_id} {transition}")
        print(f"{'-' * 90}")
        print(f"  Group index:            {group_index}")
        print(f"  Topic:                  {topic}")
        print(f"  Summary:                {summary[:140]}{'...' if len(summary) > 140 else ''}")
        print(f"  topic_continues:        {continues}")
        print(f"  same_topic_as_previous: {same_prev}")
        print(f"  Source:                 {boundary_source}")
        print(f"  Complexity:             {complexity}")
        print(f"  Time (min):             {estimated_time}")
        print(f"  Pages:                  {pages}")

        coverage = meta.get("coverage", {}) if isinstance(meta.get("coverage"), dict) else {}
        if coverage:
            print("  Coverage:")
            print(f"    - boundary_method: {coverage.get('boundary_method', 'N/A')}")
            print(f"    - char range:      {coverage.get('start_char', '?')} - {coverage.get('end_char', '?')}")
            print(f"    - percent:         {coverage.get('start_percent', '?')}% - {coverage.get('end_percent', '?')}%")
            print(f"    - window_id:       {coverage.get('window_id', 'N/A')}")

        key_concepts = meta.get("key_concepts", []) if isinstance(meta.get("key_concepts"), list) else []
        if key_concepts:
            print(f"  Concepts:              {', '.join(str(c) for c in key_concepts[:6])}")

        subtopics = meta.get("subtopics", []) if isinstance(meta.get("subtopics"), list) else []
        if subtopics:
            print(f"  Subtopics:             {', '.join(str(s) for s in subtopics[:4])}")

        print(f"  Preview:               {_preview(chunk_text)}")

        previous_group = group_index
        previous_topic = topic

    print(f"\n{'=' * 90}")


if __name__ == "__main__":
    main()
