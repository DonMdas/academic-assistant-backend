import re
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.chat.service import retrieve_schedule_chunks
from backend.config import settings
from backend.db.models import Chunk, Schedule, StudySession

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
}


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]{2,}", str(text or "").lower())
    return [token for token in tokens if token not in _STOPWORDS]


def _keyword_overlap(question_tokens: list[str], text: str) -> float:
    if not question_tokens:
        return 0.0

    text_tokens = set(_tokenize(text))
    if not text_tokens:
        return 0.0

    hits = sum(1 for token in question_tokens if token in text_tokens)
    return float(hits / max(1, len(question_tokens)))


def _focus_chunk_map(study_session: StudySession) -> dict[str, dict]:
    out = {}
    for item in list(study_session.focus_chunks_json or []):
        chunk_id = str(item.get("chunk_id") or "").strip()
        if chunk_id:
            out[chunk_id] = item
    return out


def _local_retrieval(study_session: StudySession, question: str, chunks: list[Chunk]) -> dict:
    try:
        from rag_engine import model as embedding_model
    except Exception:
        return {
            "retrieval_path": "rag_fallback",
            "chunks": [],
            "sources": [],
            "max_similarity": 0.0,
            "keyword_overlap": 0.0,
        }

    texts = [str(chunk.content or "") for chunk in chunks]
    if not texts:
        return {
            "retrieval_path": "rag_fallback",
            "chunks": [],
            "sources": [],
            "max_similarity": 0.0,
            "keyword_overlap": 0.0,
        }

    question_vector = np.array(embedding_model.encode([question]), dtype=np.float32)[0]
    chunk_vectors = np.array(embedding_model.encode(texts), dtype=np.float32)

    q_norm = np.linalg.norm(question_vector)
    c_norms = np.linalg.norm(chunk_vectors, axis=1)
    if q_norm == 0.0:
        similarities = np.zeros(shape=(len(chunks),), dtype=np.float32)
    else:
        denom = np.maximum(c_norms * q_norm, 1e-9)
        similarities = np.dot(chunk_vectors, question_vector) / denom

    focus_map = _focus_chunk_map(study_session)
    question_tokens = _tokenize(question)
    blended_text = "\n".join(texts)
    keyword_overlap = _keyword_overlap(question_tokens, blended_text)

    ranked = []
    for index, chunk in enumerate(chunks):
        score = float(similarities[index])
        focus_payload = focus_map.get(chunk.id, {})
        metadata = dict(chunk.metadata_json or {})
        metadata["chunk_id"] = chunk.id
        metadata["topic"] = focus_payload.get("topic") or metadata.get("topic")
        metadata["focus_points"] = list(focus_payload.get("focus_points") or [])

        ranked.append(
            {
                "text": chunk.content,
                "metadata": metadata,
                "retrieval": {
                    "score": score,
                    "semantic_score": score,
                    "keyword_score": keyword_overlap,
                },
            }
        )

    ranked.sort(key=lambda row: row.get("retrieval", {}).get("score", 0.0), reverse=True)

    max_similarity = float(ranked[0].get("retrieval", {}).get("score", 0.0)) if ranked else 0.0
    use_local = (
        max_similarity >= float(settings.SESSION_EMBEDDING_THRESHOLD)
        or keyword_overlap >= float(settings.SESSION_KEYWORD_OVERLAP_MIN)
    )

    if not use_local:
        return {
            "retrieval_path": "rag_fallback",
            "chunks": [],
            "sources": [],
            "max_similarity": max_similarity,
            "keyword_overlap": keyword_overlap,
        }

    selected = ranked[:4]
    sources = [
        {
            "chunk_id": row.get("metadata", {}).get("chunk_id"),
            "topic": row.get("metadata", {}).get("topic"),
            "score": row.get("retrieval", {}).get("score"),
            "focus_points": row.get("metadata", {}).get("focus_points", []),
        }
        for row in selected
    ]

    return {
        "retrieval_path": "local_chunk",
        "chunks": selected,
        "sources": sources,
        "max_similarity": max_similarity,
        "keyword_overlap": keyword_overlap,
    }


def two_stage_retrieve(db: Session, study_session: StudySession, question: str) -> dict[str, Any]:
    focus_ids = list(_focus_chunk_map(study_session).keys())

    if focus_ids:
        local_chunks = db.scalars(
            select(Chunk).where(
                Chunk.schedule_id == study_session.schedule_id,
                Chunk.id.in_(focus_ids),
            )
        ).all()
    else:
        local_chunks = []

    if local_chunks:
        local = _local_retrieval(study_session, question, local_chunks)
        if local.get("retrieval_path") == "local_chunk":
            return local
        local_similarity = float(local.get("max_similarity", 0.0))
        local_keyword = float(local.get("keyword_overlap", 0.0))
    else:
        local_similarity = 0.0
        local_keyword = 0.0

    schedule = db.get(Schedule, study_session.schedule_id)
    if schedule is None:
        return {
            "retrieval_path": "rag_fallback",
            "chunks": [],
            "sources": [],
            "max_similarity": local_similarity,
            "keyword_overlap": local_keyword,
        }

    rag_chunks, rag_sources, _index_path = retrieve_schedule_chunks(
        db=db,
        schedule=schedule,
        message=question,
        mode="qa",
        max_minutes=None,
    )

    return {
        "retrieval_path": "rag_fallback",
        "chunks": rag_chunks,
        "sources": rag_sources,
        "max_similarity": local_similarity,
        "keyword_overlap": local_keyword,
    }
