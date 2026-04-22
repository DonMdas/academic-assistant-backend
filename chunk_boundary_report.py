#!/usr/bin/env python
"""
Chunk boundary quality report.

Shows per-document totals for:
- anchor_exact_count
- anchor_fuzzy_count
- percent_fallback_count
- uncovered_fallback_chunk_count
"""

import json
from collections import Counter

from db import DocumentDB


def _classify_chunk(meta):
    boundary_source = meta.get("boundary_source", "")

    if boundary_source == "llm_uncovered_fallback":
        return "uncovered_fallback"

    coverage = meta.get("coverage") if isinstance(meta.get("coverage"), dict) else {}
    method = str(coverage.get("boundary_method", "")).lower()
    start_match = str(coverage.get("start_match", "")).lower()
    end_match = str(coverage.get("end_match", "")).lower()

    if method == "anchors":
        if start_match == "exact" and end_match == "exact":
            return "anchor_exact"
        return "anchor_fuzzy"

    if method == "percent":
        return "percent_fallback"

    return "unknown"


def main():
    db = DocumentDB()
    docs = db.list_documents()

    if not docs:
        print("No documents found in database.")
        return

    print("\nChunk Boundary Report\n")

    for doc_id, strategy, doc_type in docs:
        rows = db.conn.execute(
            "SELECT metadata FROM chunks WHERE doc_id=?",
            (doc_id,),
        ).fetchall()

        stats = Counter()
        for (metadata_json,) in rows:
            try:
                meta = json.loads(metadata_json)
            except Exception:
                stats["unknown"] += 1
                continue

            bucket = _classify_chunk(meta)
            stats[bucket] += 1

        total = len(rows)
        anchor_exact = stats["anchor_exact"]
        anchor_fuzzy = stats["anchor_fuzzy"]
        percent_fallback = stats["percent_fallback"]
        uncovered_fallback = stats["uncovered_fallback"]

        print("=" * 80)
        print(f"Document: {doc_id}")
        print(f"Strategy: {strategy} | Type: {doc_type}")
        print(f"Total chunks: {total}")
        print(f"anchor_exact_count: {anchor_exact}")
        print(f"anchor_fuzzy_count: {anchor_fuzzy}")
        print(f"percent_fallback_count: {percent_fallback}")
        print(f"uncovered_fallback_chunk_count: {uncovered_fallback}")
        if stats["unknown"]:
            print(f"unknown_count: {stats['unknown']}")

        covered_chunks = max(total - uncovered_fallback, 0)
        anchor_based = anchor_exact + anchor_fuzzy
        if covered_chunks > 0:
            anchor_rate = (anchor_based / covered_chunks) * 100.0
            print(f"anchor_usage_rate_on_non_fallback: {anchor_rate:.1f}%")

    print("=" * 80)


if __name__ == "__main__":
    main()
