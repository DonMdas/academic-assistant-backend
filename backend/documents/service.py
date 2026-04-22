import json
import os
import sqlite3
from pathlib import Path

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.config import schedule_merged_index_base, settings
from backend.db.models import Chunk, Document, Schedule


def serialize_document(document: Document) -> dict:
    return {
        "id": document.id,
        "schedule_id": document.schedule_id,
        "filename": document.filename,
        "file_path": document.file_path,
        "file_size": document.file_size,
        "ingest_status": document.ingest_status,
        "strategy": document.strategy,
        "doc_type": document.doc_type,
        "ingest_report": document.ingest_report or {},
        "created_at": document.created_at.isoformat() if document.created_at else None,
    }


def schedule_upload_dir(schedule_id: str) -> Path:
    return Path(settings.UPLOAD_ROOT) / "schedules" / schedule_id


def get_document_or_404(db: Session, schedule_id: str, document_id: str) -> Document:
    document = db.scalar(
        select(Document).where(
            Document.id == document_id,
            Document.schedule_id == schedule_id,
        )
    )
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return document


def get_chunk_or_404(db: Session, schedule_id: str, chunk_id: str) -> Chunk:
    chunk = db.scalar(
        select(Chunk).where(
            Chunk.id == chunk_id,
            Chunk.schedule_id == schedule_id,
        )
    )
    if chunk is None:
        chunk = _get_legacy_chunk_or_404(db, schedule_id, chunk_id)
    return chunk


def _get_legacy_chunk_or_404(db: Session, schedule_id: str, chunk_id: str) -> Chunk:
    conn = sqlite3.connect(settings.LEGACY_DB_PATH)
    try:
        legacy_row = conn.execute(
            "SELECT chunk_id, doc_id, content, metadata FROM chunks WHERE chunk_id=? LIMIT 1",
            (chunk_id,),
        ).fetchone()
    finally:
        conn.close()

    if legacy_row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chunk not found")

    legacy_chunk_id = str(legacy_row[0] or "").strip()
    legacy_doc_id = str(legacy_row[1] or "").strip()
    legacy_content = str(legacy_row[2] or "")
    legacy_metadata_raw = str(legacy_row[3] or "")

    document = db.scalar(
        select(Document).where(
            Document.schedule_id == schedule_id,
            Document.pipeline_doc_id == legacy_doc_id,
        )
    )
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chunk not found")

    metadata = {}
    if legacy_metadata_raw:
        try:
            parsed = json.loads(legacy_metadata_raw)
            if isinstance(parsed, dict):
                metadata = parsed
        except Exception:
            metadata = {"raw_metadata": legacy_metadata_raw}

    metadata.setdefault("chunk_id", legacy_chunk_id)
    metadata.setdefault("legacy_chunk_id", legacy_chunk_id)

    return Chunk(
        id=legacy_chunk_id,
        document_id=document.id,
        schedule_id=schedule_id,
        content=legacy_content,
        metadata_json=metadata,
        embedding_path=document.index_path,
    )


def list_documents_for_schedule(db: Session, schedule_id: str) -> list[dict]:
    rows = db.scalars(
        select(Document)
        .where(Document.schedule_id == schedule_id)
        .order_by(Document.created_at.desc())
    ).all()
    return [serialize_document(row) for row in rows]


def serialize_chunk_detail(db: Session, chunk: Chunk) -> dict:
    document = db.get(Document, chunk.document_id)
    metadata = dict(chunk.metadata_json or {})
    return {
        "id": chunk.id,
        "schedule_id": chunk.schedule_id,
        "document_id": chunk.document_id,
        "filename": document.filename if document is not None else metadata.get("filename"),
        "content": chunk.content,
        "metadata": metadata,
        "created_at": chunk.created_at.isoformat() if chunk.created_at else None,
    }


def rebuild_schedule_index_from_chunks(db: Session, schedule: Schedule) -> str:
    chunks = db.scalars(
        select(Chunk).where(Chunk.schedule_id == schedule.id).order_by(Chunk.created_at.asc())
    ).all()

    merged_base = schedule_merged_index_base(schedule.id)
    merged_base.parent.mkdir(parents=True, exist_ok=True)

    if not chunks:
        for suffix in (".index", ".pkl"):
            path = f"{merged_base}{suffix}"
            if os.path.exists(path):
                os.remove(path)
        return ""

    try:
        from rag_engine import RAGEngine

        rag = RAGEngine()
        rag.add_chunks(
            [
                {
                    "text": chunk.content,
                    "metadata": chunk.metadata_json or {},
                }
                for chunk in chunks
            ]
        )
        rag.save(str(merged_base))
    except Exception:
        return schedule.index_path

    return str(merged_base)


def _delete_legacy_document(pipeline_doc_id: str) -> None:
    legacy_id = str(pipeline_doc_id or "").strip()
    if not legacy_id:
        return

    conn = sqlite3.connect(settings.LEGACY_DB_PATH)
    try:
        row = conn.execute(
            "SELECT index_path FROM documents WHERE doc_id=?",
            (legacy_id,),
        ).fetchone()
        index_path = str(row[0]) if row and row[0] else ""

        conn.execute("DELETE FROM chunks WHERE doc_id=?", (legacy_id,))
        conn.execute("DELETE FROM session_documents WHERE doc_id=?", (legacy_id,))
        conn.execute("DELETE FROM documents WHERE doc_id=?", (legacy_id,))
        conn.commit()

        if index_path:
            for suffix in (".index", ".pkl"):
                path = f"{index_path}{suffix}"
                if os.path.exists(path):
                    os.remove(path)
    finally:
        conn.close()


def delete_document_with_cleanup(db: Session, schedule: Schedule, document: Document) -> dict:
    db.query(Chunk).filter(Chunk.document_id == document.id).delete(synchronize_session=False)

    if document.pipeline_doc_id:
        _delete_legacy_document(document.pipeline_doc_id)

    file_path = str(document.file_path or "")
    db.delete(document)
    db.commit()

    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception:
            pass

    updated_index_path = rebuild_schedule_index_from_chunks(db, schedule)
    schedule.index_path = updated_index_path
    db.commit()
    db.refresh(schedule)

    return {
        "deleted_document_id": document.id,
        "schedule_id": schedule.id,
        "schedule_index_path": schedule.index_path,
    }
