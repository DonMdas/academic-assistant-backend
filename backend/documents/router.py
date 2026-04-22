import asyncio
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Query, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.auth.dependencies import get_current_user
from backend.db.models import Document, User
from backend.db.session import SessionLocal, get_db
from backend.documents.ingest_tasks import run_document_ingestion
from backend.documents.service import (
    get_chunk_or_404,
    delete_document_with_cleanup,
    get_document_or_404,
    list_documents_for_schedule,
    schedule_upload_dir,
    serialize_chunk_detail,
    serialize_document,
)
from backend.operations.logs import operation_logs
from backend.schedules.service import get_schedule_or_404
from backend.sse.utils import format_sse


router = APIRouter(prefix="/schedules/{schedule_id}/documents", tags=["documents"])


async def _save_upload(upload: UploadFile, target_path: Path) -> int:
    data = await upload.read()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "wb") as handle:
        handle.write(data)
    return len(data)


@router.post(
    "",
    summary="Upload and ingest documents",
    description=(
        "Uploads one or more files for a schedule, creates document records, and queues background ingestion. "
        "Each uploaded file immediately enters pending/processing lifecycle."
    ),
    response_description="List of created document records.",
)
async def upload_documents(
    schedule_id: str,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(..., description="One or more files to ingest into this schedule."),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    schedule = get_schedule_or_404(db, current_user.id, schedule_id)

    saved_docs = []
    operation_ids: list[str] = []
    upload_dir = schedule_upload_dir(schedule.id)

    for file in files:
        raw_name = file.filename or "document.pdf"
        operation_id = operation_logs.start(
            user_id=current_user.id,
            kind="document_ingestion",
            message=f"Queued ingestion for {raw_name}",
            metadata={"schedule_id": schedule.id, "filename": raw_name},
        )
        if operation_id:
            operation_ids.append(operation_id)

        unique_name = f"{uuid.uuid4()}_{os.path.basename(file.filename or 'document.pdf')}"
        target = upload_dir / unique_name
        file_size = await _save_upload(file, target)

        ingest_report = {}
        if operation_id:
            ingest_report["operation_id"] = operation_id

        document = Document(
            schedule_id=schedule.id,
            filename=file.filename or unique_name,
            file_path=str(target),
            file_size=file_size,
            ingest_status="pending",
            strategy="rag",
            doc_type="notes",
            ingest_report=ingest_report,
        )
        db.add(document)
        db.commit()
        db.refresh(document)

        background_tasks.add_task(run_document_ingestion, document.id, schedule.id, operation_id, current_user.id)
        saved_docs.append(serialize_document(document))

    return {"documents": saved_docs, "operation_ids": operation_ids}


@router.get(
    "",
    summary="List schedule documents",
    description="Returns all documents for a schedule, including ingestion status and metadata.",
    response_description="Array of document records.",
)
def list_documents(
    schedule_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    get_schedule_or_404(db, current_user.id, schedule_id)
    return list_documents_for_schedule(db, schedule_id)


@router.get(
    "/{doc_id}",
    summary="Get document",
    description="Returns one document record for the selected schedule.",
    response_description="Single document record.",
)
def get_document(
    schedule_id: str,
    doc_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    get_schedule_or_404(db, current_user.id, schedule_id)
    document = get_document_or_404(db, schedule_id, doc_id)
    return serialize_document(document)


@router.get(
    "/{doc_id}/ingest-status",
    summary="Get ingestion status",
    description=(
        "Returns ingestion status as JSON by default, or streams status updates as Server-Sent Events when stream=true. "
        "SSE emits status events and finishes with done."
    ),
    response_description="Current ingestion status or an SSE stream.",
)
async def get_ingest_status(
    schedule_id: str,
    doc_id: str,
    stream: bool = Query(False, description="Set true to receive SSE status updates."),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    get_schedule_or_404(db, current_user.id, schedule_id)

    if not stream:
        document = get_document_or_404(db, schedule_id, doc_id)
        report = dict(document.ingest_report or {})
        operation_id = str(report.get("operation_id") or "").strip() or None
        return {
            "document_id": document.id,
            "ingest_status": document.ingest_status,
            "ingest_report": report,
            "operation_id": operation_id,
        }

    async def event_stream():
        operation_event_sent = False
        while True:
            with SessionLocal() as local_db:
                document = get_document_or_404(local_db, schedule_id, doc_id)
                report = dict(document.ingest_report or {})
                operation_id = str(report.get("operation_id") or "").strip() or None

                if operation_id and not operation_event_sent:
                    yield format_sse("operation", {"operation_id": operation_id})
                    operation_event_sent = True

                payload = {
                    "document_id": document.id,
                    "ingest_status": document.ingest_status,
                    "ingest_report": report,
                    "operation_id": operation_id,
                }
                yield format_sse("status", payload)
                if document.ingest_status in {"done", "failed"}:
                    yield format_sse("done", payload)
                    break
            await asyncio.sleep(1.0)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get(
    "/chunks/{chunk_id}",
    summary="Get chunk detail",
    description="Returns one indexed chunk for the selected schedule, including full content and metadata.",
    response_description="Single chunk detail record.",
)
def get_chunk_detail(
    schedule_id: str,
    chunk_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    get_schedule_or_404(db, current_user.id, schedule_id)
    chunk = get_chunk_or_404(db, schedule_id, chunk_id)
    return serialize_chunk_detail(db, chunk)


@router.delete(
    "/{doc_id}",
    summary="Delete document",
    description=(
        "Deletes a document, removes associated chunks, performs legacy cleanup, "
        "and rebuilds the schedule-level merged index."
    ),
    response_description="Deletion and index-rebuild result payload.",
)
def delete_document(
    schedule_id: str,
    doc_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    schedule = get_schedule_or_404(db, current_user.id, schedule_id)
    document = get_document_or_404(db, schedule.id, doc_id)
    return delete_document_with_cleanup(db, schedule, document)
