from __future__ import annotations

from dataclasses import dataclass, field
import threading
import uuid
from typing import Any

from backend.timezone_utils import iso_now_ist


MAX_OPERATION_ENTRIES = 1500
MAX_OPERATION_COUNT = 400


def _ist_iso() -> str:
    return iso_now_ist()


def _coerce_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}

    normalized: dict[str, Any] = {}
    for key, value in metadata.items():
        safe_key = str(key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            normalized[safe_key] = value
            continue
        if isinstance(value, (list, dict)):
            normalized[safe_key] = value
            continue
        normalized[safe_key] = str(value)
    return normalized


@dataclass
class OperationLogEntry:
    seq: int
    ts: str
    level: str
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OperationState:
    operation_id: str
    user_id: str
    kind: str
    status: str = "running"
    created_at: str = field(default_factory=_ist_iso)
    updated_at: str = field(default_factory=_ist_iso)
    entries: list[OperationLogEntry] = field(default_factory=list)
    next_seq: int = 1


class OperationLogManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ops: dict[str, OperationState] = {}
        self._order: list[str] = []

    def new_operation_id(self) -> str:
        return str(uuid.uuid4())

    def _evict_if_needed_locked(self) -> None:
        while len(self._order) > MAX_OPERATION_COUNT:
            stale_id = self._order.pop(0)
            self._ops.pop(stale_id, None)

    def _append_locked(
        self,
        state: OperationState,
        level: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        entry = OperationLogEntry(
            seq=state.next_seq,
            ts=_ist_iso(),
            level=str(level or "info"),
            message=str(message or "").strip(),
            metadata=_coerce_metadata(metadata),
        )
        state.next_seq += 1

        state.entries.append(entry)
        if len(state.entries) > MAX_OPERATION_ENTRIES:
            state.entries = state.entries[-MAX_OPERATION_ENTRIES:]

        state.updated_at = entry.ts

    def start(
        self,
        user_id: str,
        kind: str,
        message: str,
        operation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        op_id = str(operation_id or self.new_operation_id())

        with self._lock:
            state = self._ops.get(op_id)
            if state is None:
                state = OperationState(operation_id=op_id, user_id=str(user_id), kind=str(kind or "operation"))
                self._ops[op_id] = state
                self._order.append(op_id)
                self._evict_if_needed_locked()
            else:
                if state.user_id != str(user_id):
                    return ""
                state.kind = str(kind or state.kind or "operation")
                if state.status in {"done", "failed"}:
                    state.status = "running"
                    state.updated_at = _ist_iso()

            if message:
                self._append_locked(state, "info", message, metadata)

        return op_id

    def append(
        self,
        operation_id: str,
        message: str,
        level: str = "info",
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        op_id = str(operation_id or "").strip()
        if not op_id:
            return False

        with self._lock:
            state = self._ops.get(op_id)
            if state is None:
                return False
            self._append_locked(state, level, message, metadata)
            return True

    def finish(
        self,
        operation_id: str,
        success: bool,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        op_id = str(operation_id or "").strip()
        if not op_id:
            return False

        with self._lock:
            state = self._ops.get(op_id)
            if state is None:
                return False

            level = "success" if success else "error"
            if message:
                self._append_locked(state, level, message, metadata)
            state.status = "done" if success else "failed"
            state.updated_at = _ist_iso()
            return True

    def succeed(self, operation_id: str, message: str, metadata: dict[str, Any] | None = None) -> bool:
        return self.finish(operation_id=operation_id, success=True, message=message, metadata=metadata)

    def fail(self, operation_id: str, message: str, metadata: dict[str, Any] | None = None) -> bool:
        return self.finish(operation_id=operation_id, success=False, message=message, metadata=metadata)

    def get_logs(
        self,
        operation_id: str,
        user_id: str,
        offset: int = 0,
        limit: int = 200,
    ) -> dict[str, Any] | None:
        op_id = str(operation_id or "").strip()
        if not op_id:
            return None

        safe_offset = max(0, int(offset or 0))
        safe_limit = max(1, min(int(limit or 200), 500))

        with self._lock:
            state = self._ops.get(op_id)
            if state is None or state.user_id != str(user_id):
                return None

            total = len(state.entries)
            start = min(safe_offset, total)
            end = min(total, start + safe_limit)

            entries = [
                {
                    "seq": entry.seq,
                    "ts": entry.ts,
                    "level": entry.level,
                    "message": entry.message,
                    "metadata": entry.metadata,
                }
                for entry in state.entries[start:end]
            ]

            return {
                "operation_id": state.operation_id,
                "kind": state.kind,
                "status": state.status,
                "created_at": state.created_at,
                "updated_at": state.updated_at,
                "entries": entries,
                "next_offset": end,
                "done": state.status in {"done", "failed"},
                "total_entries": total,
            }


operation_logs = OperationLogManager()
