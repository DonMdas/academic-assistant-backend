import json
import os
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from backend.config import schedule_merged_index_base, settings
from backend.db.models import Chunk, Document, Schedule
from backend.db.session import SessionLocal
from backend.operations.logs import operation_logs
from backend.timezone_utils import iso_now_ist, now_ist, to_ist_aware


def _ist_iso() -> str:
    return iso_now_ist()


def _calc_duration_seconds(report: dict[str, Any]) -> float | None:
    started_at = str(report.get("started_at") or "").strip()
    if not started_at:
        return None
    try:
        start_dt = datetime.fromisoformat(started_at)
        start_dt = to_ist_aware(start_dt)
        seconds = (now_ist() - start_dt).total_seconds()
        return round(max(0.0, float(seconds)), 3)
    except Exception:
        return None


def _append_ingest_stage(
    report: dict[str, Any],
    *,
    stage: str,
    message: str,
    progress_pct: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    ts = _ist_iso()
    report["updated_at"] = ts
    report["current_stage"] = str(stage or "unknown")
    report["last_message"] = str(message or "").strip()

    if progress_pct is not None:
        report["progress_pct"] = max(0, min(100, int(progress_pct)))

    stages = list(report.get("stages") or [])
    entry: dict[str, Any] = {
        "ts": ts,
        "stage": str(stage or "unknown"),
        "message": str(message or "").strip(),
    }
    if isinstance(metadata, dict) and metadata:
        entry["metadata"] = metadata

    stages.append(entry)
    report["stages"] = stages[-120:]


def _build_pipeline_summary(result: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(result or {})
    coverage = dict(payload.get("coverage") or {})
    ingest_coverage = dict(payload.get("ingest_coverage") or {})
    chunking_stats = dict(payload.get("chunking_stats") or {})

    chunking_stats_subset = {
        "input_windows": int(chunking_stats.get("input_windows") or 0),
        "output_chunks": int(chunking_stats.get("output_chunks") or 0),
        "topic_groups": int(chunking_stats.get("topic_groups") or 0),
        "topic_continuations": int(chunking_stats.get("topic_continuations") or 0),
        "topic_resets": int(chunking_stats.get("topic_resets") or 0),
        "llm_failures": int(chunking_stats.get("llm_failures") or 0),
        "analysis_novel_windows": int(chunking_stats.get("analysis_novel_windows") or 0),
        "analysis_overlap_trimmed_chars": int(chunking_stats.get("analysis_overlap_trimmed_chars") or 0),
        "embedding_gate_hits": int(chunking_stats.get("embedding_gate_hits") or 0),
        "embedding_gate_continuations": int(chunking_stats.get("embedding_gate_continuations") or 0),
        "embedding_gate_breaks": int(chunking_stats.get("embedding_gate_breaks") or 0),
    }

    ingest_coverage_subset = {
        "raw_source_chars": int(ingest_coverage.get("raw_source_chars") or 0),
        "raw_source_words": int(ingest_coverage.get("raw_source_words") or 0),
        "filtered_source_chars": int(ingest_coverage.get("filtered_source_chars") or 0),
        "filtered_source_words": int(ingest_coverage.get("filtered_source_words") or 0),
        "chunk_chars": int(ingest_coverage.get("chunk_chars") or 0),
        "chunk_words": int(ingest_coverage.get("chunk_words") or 0),
        "filtered_char_retention_pct": float(ingest_coverage.get("filtered_char_retention_pct") or 0.0),
        "filtered_word_retention_pct": float(ingest_coverage.get("filtered_word_retention_pct") or 0.0),
        "raw_char_retention_pct": float(ingest_coverage.get("raw_char_retention_pct") or 0.0),
        "raw_word_retention_pct": float(ingest_coverage.get("raw_word_retention_pct") or 0.0),
    }

    coverage_subset = {
        "window_coverage_pct": float(coverage.get("window_coverage_pct") or 0.0),
        "duplication_pct_of_chunk": float(coverage.get("duplication_pct_of_chunk") or 0.0),
        "expansion_pct_vs_window": float(coverage.get("expansion_pct_vs_window") or 0.0),
        "chunk_words": int(coverage.get("chunk_words") or 0),
        "source_words": int(coverage.get("source_words") or 0),
    }

    return {
        "strategy": str(payload.get("strategy") or ""),
        "doc_type": str(payload.get("doc_type") or ""),
        "chunks": int(payload.get("chunks") or 0),
        "chunking_stats": chunking_stats_subset,
        "ingest_coverage": ingest_coverage_subset,
        "coverage": coverage_subset,
        "fallback_report": dict(payload.get("fallback_report") or {}),
        "session": {
            "session_id": str(payload.get("session_id") or ""),
            "session_topic_count": int(payload.get("session_topic_count") or 0),
            "session_summary": str(payload.get("session_summary") or ""),
            "session_rag_index_path": str(payload.get("session_rag_index_path") or ""),
        },
    }


def _legacy_chunk_rows(legacy_doc_id: str) -> list[tuple[str, str, str]]:
    conn = sqlite3.connect(settings.LEGACY_DB_PATH)
    try:
        rows = conn.execute(
            "SELECT chunk_id, content, metadata FROM chunks WHERE doc_id=? ORDER BY chunk_id ASC",
            (legacy_doc_id,),
        ).fetchall()
        return [(str(row[0]), str(row[1] or ""), str(row[2] or "")) for row in rows]
    finally:
        conn.close()


def _sync_chunks_from_legacy(db: Session, document: Document) -> int:
    if not document.pipeline_doc_id:
        return 0

    rows = _legacy_chunk_rows(document.pipeline_doc_id)
    if not rows:
        return 0

    db.query(Chunk).filter(Chunk.document_id == document.id).delete(synchronize_session=False)

    synced = 0
    for legacy_chunk_id, content, metadata_raw in rows:
        metadata = {}
        if metadata_raw:
            try:
                payload = json.loads(metadata_raw)
                if isinstance(payload, dict):
                    metadata = payload
            except Exception:
                metadata = {"raw_metadata": metadata_raw}

        chunk = db.get(Chunk, legacy_chunk_id)
        if chunk is None:
            chunk = Chunk(
                id=legacy_chunk_id,
                document_id=document.id,
                schedule_id=document.schedule_id,
                content=content,
                metadata_json=metadata,
                embedding_path=document.index_path,
            )
            db.add(chunk)
        else:
            chunk.document_id = document.id
            chunk.schedule_id = document.schedule_id
            chunk.content = content
            chunk.metadata_json = metadata
            chunk.embedding_path = document.index_path

        synced += 1

    return synced


def _mirror_schedule_index(schedule_id: str, legacy_index_base: str | None) -> str:
    source_base = str(legacy_index_base or "").strip()
    if not source_base:
        source_base = f"indexes/sessions/{schedule_id}"

    source_index = Path(f"{source_base}.index")
    source_pkl = Path(f"{source_base}.pkl")
    if not source_index.exists() or not source_pkl.exists():
        return ""

    target_base = schedule_merged_index_base(schedule_id)
    target_base.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(source_index, Path(f"{target_base}.index"))
    shutil.copy2(source_pkl, Path(f"{target_base}.pkl"))

    return str(target_base)


def run_document_ingestion(
    document_id: str,
    schedule_id: str,
    operation_id: str | None = None,
    user_id: str | None = None,
) -> None:
    db = SessionLocal()
    document = None

    try:
        document = db.get(Document, document_id)
        schedule = db.get(Schedule, schedule_id)

        owner_user_id = str(user_id or (schedule.user_id if schedule is not None else "")).strip()
        if operation_id and owner_user_id:
            operation_logs.start(
                operation_id=operation_id,
                user_id=owner_user_id,
                kind="document_ingestion",
                message="Ingestion worker started",
                metadata={"document_id": document_id, "schedule_id": schedule_id},
            )

        if document is None or schedule is None:
            if operation_id:
                operation_logs.fail(operation_id, "Document or schedule not found for ingestion")
            return

        report = dict(document.ingest_report or {})
        if operation_id:
            report["operation_id"] = operation_id
        if "started_at" not in report:
            report["started_at"] = _ist_iso()
        report["pipeline_entrypoint"] = "main.process_document"
        report["status"] = "processing"
        _append_ingest_stage(
            report,
            stage="worker_started",
            message="Ingestion worker started",
            progress_pct=5,
            metadata={
                "document_id": document_id,
                "schedule_id": schedule_id,
                "filename": str(document.filename or ""),
            },
        )
        document.ingest_report = report
        db.commit()

        document.ingest_status = "processing"
        report["status"] = "processing"
        _append_ingest_stage(
            report,
            stage="processing",
            message="Document status set to processing",
            progress_pct=10,
        )
        document.ingest_report = report
        db.commit()
        if operation_id:
            operation_logs.append(operation_id, "Document status set to processing")

        from main import process_document

        _append_ingest_stage(
            report,
            stage="main_pipeline_started",
            message="Running main ingestion pipeline",
            progress_pct=20,
            metadata={"entrypoint": "main.process_document"},
        )
        document.ingest_report = report
        db.commit()

        if operation_id:
            operation_logs.append(operation_id, "Running main ingestion pipeline")

        result = process_document(
            document.file_path,
            session_id=schedule_id,
            session_title=schedule.name,
        )

        if operation_id:
            operation_logs.append(
                operation_id,
                "Main ingestion pipeline finished",
                metadata={"pipeline_doc_id": str(result.get("doc_id") or "")},
            )

        document.pipeline_doc_id = str(result.get("doc_id") or "")
        document.strategy = str(result.get("strategy") or document.strategy or "rag")
        document.doc_type = str(result.get("doc_type") or document.doc_type or "notes")

        report["pipeline_result"] = result
        report["pipeline_summary"] = _build_pipeline_summary(result)
        _append_ingest_stage(
            report,
            stage="main_pipeline_completed",
            message="Main ingestion pipeline completed",
            progress_pct=70,
            metadata={
                "pipeline_doc_id": str(result.get("doc_id") or ""),
                "strategy": str(result.get("strategy") or ""),
                "doc_type": str(result.get("doc_type") or ""),
                "chunks": int(result.get("chunks") or 0),
            },
        )
        document.ingest_report = report
        db.commit()

        mirrored = _mirror_schedule_index(schedule_id, result.get("session_rag_index_path"))
        if mirrored:
            document.index_path = mirrored
            schedule.index_path = mirrored
            report["mirrored_index_path"] = mirrored
            _append_ingest_stage(
                report,
                stage="schedule_index_mirrored",
                message="Mirrored session index to schedule index",
                progress_pct=82,
                metadata={"index_path": mirrored},
            )
            document.ingest_report = report
            db.commit()
            if operation_id:
                operation_logs.append(operation_id, "Mirrored schedule index", metadata={"index_path": mirrored})
        else:
            _append_ingest_stage(
                report,
                stage="schedule_index_not_mirrored",
                message="No session index files found to mirror",
                progress_pct=82,
            )
            document.ingest_report = report
            db.commit()

        synced_chunks = _sync_chunks_from_legacy(db, document)
        report["synced_chunk_count"] = synced_chunks
        _append_ingest_stage(
            report,
            stage="chunks_synced",
            message="Synchronized chunks from legacy store",
            progress_pct=92,
            metadata={"synced_chunk_count": synced_chunks},
        )
        document.ingest_report = report
        db.commit()
        if operation_id:
            operation_logs.append(operation_id, "Synchronized chunks from legacy store", metadata={"synced_chunk_count": synced_chunks})

        document.ingest_status = "done"
        report["status"] = "done"
        report["completed_at"] = _ist_iso()
        duration_seconds = _calc_duration_seconds(report)
        if duration_seconds is not None:
            report["duration_seconds"] = duration_seconds
        _append_ingest_stage(
            report,
            stage="completed",
            message="Document ingestion completed",
            progress_pct=100,
            metadata={"synced_chunk_count": synced_chunks},
        )
        document.ingest_report = report
        db.commit()
        if operation_id:
            operation_logs.succeed(
                operation_id,
                "Document ingestion completed",
                metadata={"document_id": document.id, "synced_chunk_count": synced_chunks},
            )
    except Exception as exc:
        if document is not None:
            document.ingest_status = "failed"
            previous = dict(document.ingest_report or {})
            previous["status"] = "failed"
            previous["error"] = str(exc)
            previous["completed_at"] = _ist_iso()
            duration_seconds = _calc_duration_seconds(previous)
            if duration_seconds is not None:
                previous["duration_seconds"] = duration_seconds
            _append_ingest_stage(
                previous,
                stage="failed",
                message="Document ingestion failed",
                progress_pct=100,
                metadata={"error": str(exc)},
            )
            document.ingest_report = previous
            db.commit()
        if operation_id:
            operation_logs.fail(operation_id, f"Document ingestion failed: {exc}")
    finally:
        db.close()
