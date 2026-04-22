# interface.py

import os
import json

from db import DocumentDB
from rag_engine import RAGEngine
from gemma_engine import generate_answer

db = DocumentDB()


def show_retrieved_chunks(chunks):
    print("\n📎 Retrieved Chunks Used:\n")

    if not chunks:
        print("No chunks retrieved.")
        return

    for i, c in enumerate(chunks, 1):
        retrieval = c.get("retrieval", {})
        score = retrieval.get("score")
        distance = retrieval.get("distance")
        semantic_score = retrieval.get("semantic_score")
        keyword_score = retrieval.get("keyword_score")
        qwen_score = retrieval.get("qwen_score")
        qwen_fallback = retrieval.get("qwen_fallback")
        rank = retrieval.get("rank", i)

        topic = c.get("metadata", {}).get("topic", "Unknown")
        pages = c.get("metadata", {}).get("pages", [])
        source_doc_id = c.get("source_doc_id") or c.get("metadata", {}).get("source_doc_id")
        preview = c.get("text", "").replace("\n", " ").strip()
        preview = preview[:220] + ("..." if len(preview) > 220 else "")

        score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
        distance_text = f"{distance:.4f}" if isinstance(distance, (int, float)) else "N/A"

        print(f"[{rank}] Topic: {topic}")
        if source_doc_id:
            print(f"    Source Doc: {source_doc_id}")
        print(f"    Score: {score_text} | Distance: {distance_text}")
        if isinstance(semantic_score, (int, float)) or isinstance(keyword_score, (int, float)):
            semantic_text = f"{semantic_score:.4f}" if isinstance(semantic_score, (int, float)) else "N/A"
            keyword_text = f"{keyword_score:.4f}" if isinstance(keyword_score, (int, float)) else "N/A"
            print(f"    Hybrid: semantic={semantic_text} | keyword={keyword_text}")
        if isinstance(qwen_score, (int, float)):
            fallback_note = f" (fallback={qwen_fallback})" if qwen_fallback else ""
            print(f"    Qwen rerank: {qwen_score:.4f}{fallback_note}")
        print(f"    Pages: {pages}")
        print(f"    Preview: {preview}")
        print("-" * 60)


def list_docs():
    docs = db.list_documents()

    print("\n📚 Documents:\n")
    for i, d in enumerate(docs):
        print(f"[{i}] {d[0]} | {d[1]} | {d[2]}")

    return docs


def list_sessions():
    sessions = db.list_sessions()

    print("\n🗂️ Study Sessions:\n")
    if not sessions:
        print("No sessions found.")
        return []

    for i, (session_id, title, created_at, document_count) in enumerate(sessions):
        name = title or "Untitled Session"
        print(f"[{i}] {session_id} | {name} | docs={document_count} | created={created_at}")

    return sessions


def select_doc():
    docs = list_docs()
    choice = int(input("\nSelect doc: "))
    return docs[choice][0]


def select_session():
    sessions = list_sessions()
    if not sessions:
        return None

    choice = int(input("\nSelect session: "))
    return sessions[choice][0]


def choose_mode():
    print("\nModes:")
    print("1. Normal Q&A (Qwen rerank)")
    print("2. Study Plan (Qwen rerank, broader retrieval)")
    print("3. Beginner Only")
    print("4. Time-based Plan")

    return input("Select mode: ")


def _merge_session_results(session_results, top_k):
    merged = sorted(
        session_results,
        key=lambda c: c.get("retrieval", {}).get("score", 0.0),
        reverse=True,
    )
    merged = merged[:top_k]

    for i, chunk in enumerate(merged, 1):
        retrieval = dict(chunk.get("retrieval", {}))
        retrieval["rank"] = i
        chunk["retrieval"] = retrieval

    return merged


def _retrieve_from_session(rag_by_doc, query, top_k=5, filters=None):
    all_results = []
    per_doc_top_k = max(top_k, 5)

    for doc_id, rag in rag_by_doc.items():
        doc_results = rag.retrieve(query, top_k=per_doc_top_k, filters=filters)

        for chunk in doc_results:
            item = {
                "text": chunk.get("text", ""),
                "metadata": chunk.get("metadata", {}),
                "retrieval": dict(chunk.get("retrieval", {})),
                "source_doc_id": doc_id,
            }
            all_results.append(item)

    return _merge_session_results(all_results, top_k=top_k)


def _retrieve_from_session_with_time_budget(rag_by_doc, query, max_total_time=120):
    candidates = _retrieve_from_session(rag_by_doc, query, top_k=20)

    selected = []
    total_time = 0

    for item in candidates:
        estimated_time = item.get("metadata", {}).get("estimated_time", 30)
        if total_time + estimated_time > max_total_time:
            continue
        selected.append(item)
        total_time += estimated_time

    return selected


def query_doc(doc_id):
    doc = db.get_document(doc_id)

    strategy = doc[1]
    index_path = doc[4]

    if strategy == "full_context":
        print("⚡ Full context doc — directly send to Gemma (not implemented here)")
        return

    rag = RAGEngine(index_path=index_path)

    while True:
        query = input("\nAsk (or 'exit'): ")

        if query == "exit":
            break

        mode = choose_mode()

        # -------------------
        # RETRIEVAL LOGIC
        # -------------------
        if mode == "1":
            chunks = rag.retrieve_hybrid(
                query,
                top_k=5,
                rerank_with_qwen=True,
                rerank_top_n=5,
            )

        elif mode == "2":
            chunks = rag.retrieve_hybrid(
                query,
                top_k=8,
                rerank_with_qwen=True,
                rerank_top_n=6,
            )

        elif mode == "3":
            chunks = rag.retrieve(query, filters={"complexity": "beginner"})

        elif mode == "4":
            max_time = int(input("Enter max time (minutes): "))
            chunks = rag.retrieve_with_time_budget(query, max_time)

        else:
            print("Invalid mode")
            continue

        # -------------------
        # GEMMA CALL
        # -------------------
        if mode == "2" or mode == "4":
            answer = generate_answer(query, chunks, mode="plan")
        else:
            answer = generate_answer(query, chunks, mode="qa")

        print("\n🧠 Answer:\n")
        print(answer)
        show_retrieved_chunks(chunks)
        print("\n" + "=" * 60)


def query_session(session_id):
    session_profile = db.get_session_profile(session_id)
    if not session_profile:
        print("Session not found.")
        return

    session_docs = db.get_session_documents(session_id)

    if not session_docs:
        print("No documents found in this session.")
        return

    topics = list(session_profile.get("topics", []))
    summary = str(session_profile.get("summary", "")).strip()
    rag_index_path = session_profile.get("rag_index_path") or f"indexes/sessions/{session_id}"

    has_index = os.path.exists(f"{rag_index_path}.index") and os.path.exists(f"{rag_index_path}.pkl")
    if not has_index:
        print("⚠️ Session index not found. Rebuilding from stored chunks...")
        session_rows = db.get_session_chunks(session_id)
        if not session_rows:
            print("No stored chunks found to rebuild session index.")
            return

        rebuild_chunks = []
        os.makedirs(os.path.dirname(rag_index_path), exist_ok=True)
        for _chunk_id, doc_id, content, metadata_json in session_rows:
            metadata = {}
            try:
                metadata = json.loads(metadata_json) if metadata_json else {}
            except Exception:
                metadata = {}

            metadata["source_doc_id"] = metadata.get("source_doc_id") or doc_id
            metadata["session_id"] = session_id
            rebuild_chunks.append({
                "text": str(content or ""),
                "metadata": metadata,
            })

        session_rag_rebuild = RAGEngine()
        session_rag_rebuild.add_chunks(rebuild_chunks)
        session_rag_rebuild.save(rag_index_path)

        db.update_session_profile(
            session_id,
            topics=topics,
            summary=summary,
            rag_index_path=rag_index_path,
        )

    session_rag = RAGEngine(index_path=rag_index_path)

    print(
        f"Session ready: {session_id} | docs={len(session_docs)} | topics={len(topics)}"
    )
    if summary:
        preview = summary[:220] + ("..." if len(summary) > 220 else "")
        print(f"Session summary: {preview}")

    while True:
        query = input("\nAsk (or 'exit'): ")

        if query == "exit":
            break

        mode = choose_mode()

        if mode == "1":
            chunks = session_rag.retrieve_hybrid(
                query,
                top_k=5,
                rerank_with_qwen=True,
                rerank_top_n=5,
            )
        elif mode == "2":
            chunks = session_rag.retrieve_hybrid(
                query,
                top_k=8,
                rerank_with_qwen=True,
                rerank_top_n=6,
            )
        elif mode == "3":
            chunks = session_rag.retrieve(query, filters={"complexity": "beginner"})
        elif mode == "4":
            max_time = int(input("Enter max time (minutes): "))
            chunks = session_rag.retrieve_with_time_budget(query, max_time)
        else:
            print("Invalid mode")
            continue

        for rank, item in enumerate(chunks, 1):
            metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
            if isinstance(metadata, dict):
                item["source_doc_id"] = item.get("source_doc_id") or metadata.get("source_doc_id")
            retrieval = dict(item.get("retrieval", {}))
            retrieval["rank"] = rank
            item["retrieval"] = retrieval

        if mode in {"2", "4"}:
            answer = generate_answer(query, chunks, mode="plan")
        else:
            answer = generate_answer(query, chunks, mode="qa")

        print("\n🧠 Answer:\n")
        print(answer)
        show_retrieved_chunks(chunks)
        print("\n" + "=" * 60)


def choose_query_target():
    print("\nQuery Scope:")
    print("1. Single document")
    print("2. Session (multiple docs)")

    return input("Select scope: ")


if __name__ == "__main__":
    scope = choose_query_target()

    if scope == "1":
        doc_id = select_doc()
        query_doc(doc_id)
    elif scope == "2":
        session_id = select_session()
        if session_id:
            query_session(session_id)
    else:
        print("Invalid scope")