# Academic Assistant API Documentation

## 1. Purpose and Scope

This document describes the FastAPI backend implemented under the `backend/` package.
It covers:

- Every exposed HTTP endpoint
- Authentication and authorization behavior
- Request and response contracts
- Server-Sent Events (SSE) stream formats
- End-to-end application flow from document upload to session chat

The API is designed for a study planning assistant that ingests course documents, generates study plans, and powers schedule-level and session-focused retrieval/chat experiences.

## 2. Runtime and API Basics

- Framework: FastAPI
- App entrypoint: `backend/main.py`
- OpenAPI docs (default FastAPI):
  - `GET /docs`
  - `GET /openapi.json`
- Health check: `GET /healthz`

## 3. Security Model

### 3.1 Access Token

Most endpoints require:

- `Authorization: Bearer <access_token>`

Access tokens are JWTs issued by `/auth/google` and `/auth/refresh`.

### 3.2 Refresh Token

Refresh token is stored in an HTTP-only cookie:

- Cookie name: `refresh_token` (configurable)

`/auth/refresh` validates and rotates the refresh token.
`/auth/logout` revokes it.

### 3.3 User Scoping

All schedule/session resources are user-scoped. Ownership checks are enforced via DB joins (`Schedule.user_id == current_user.id`).

## 4. SSE Response Format

Several endpoints stream via Server-Sent Events.
All events are emitted in this wire format:

```text
event: <event_name>
data: <json_payload>

```

Event payload JSON is ASCII-escaped (`ensure_ascii=True`).

Common event names:

- `operation`
- `status`
- `sources`
- `delta`
- `done`
- `error`

## 5. Endpoint Catalog

## 5.1 System

### GET /healthz

- Auth: No
- Purpose: Liveness probe
- Request body: None
- Response:

```json
{
  "status": "ok"
}
```

## 5.2 Auth (`/auth`)

### POST /auth/google

- Auth: No
- Purpose:
  - Validate Google ID token via Google tokeninfo endpoint
  - Create/update local user
  - Issue access token
  - Set refresh token cookie
- Request body:

```json
{
  "id_token": "<google-id-token>"
}
```

- Response body:

```json
{
  "access_token": "<jwt>",
  "user": {
    "id": "<uuid>",
    "email": "user@example.com",
    "name": "User Name",
    "avatar": "https://..."
  }
}
```

- Notes:
  - Returns `401` if token is invalid, audience mismatched, or email unverified.

### POST /auth/refresh

- Auth: No bearer required
- Purpose:
  - Read refresh cookie
  - Validate and rotate refresh token
  - Return new access token and set new refresh cookie
- Request body: None
- Response body: same shape as `/auth/google`
- Notes:
  - Returns `401` if cookie missing/invalid/revoked/expired.

### POST /auth/logout

- Auth: No bearer required (cookie-driven)
- Purpose:
  - Revoke refresh token in DB
  - Clear refresh cookie
- Request body: None
- Response:

```json
{
  "success": true
}
```

## 5.3 Schedules (`/schedules`)

### GET /schedules

- Auth: Required
- Purpose: List all schedules for current user
- Request body: None
- Response: array of schedule objects

Key fields:

- `id`
- `name`
- `description`
- `status` (`active` or `archived`)
- `index_path`
- `created_at`, `updated_at`

### POST /schedules

- Auth: Required
- Purpose: Create schedule
- Request body:

```json
{
  "name": "Computer Vision Midterm",
  "description": "April study schedule"
}
```

- Response: created schedule object

### GET /schedules/{schedule_id}

- Auth: Required
- Purpose: Get enriched schedule detail
- Includes:
  - Core schedule metadata
  - Document list summary
  - Latest plan summary (if available)

### PATCH /schedules/{schedule_id}

- Auth: Required
- Purpose: Partial update of name/description
- Request body:

```json
{
  "name": "Updated Name",
  "description": "Updated description"
}
```

- Response: updated schedule object

### DELETE /schedules/{schedule_id}

- Auth: Required
- Purpose: Soft delete schedule (archive)
- Response:

```json
{
  "id": "<schedule_id>",
  "status": "archived"
}
```

## 5.4 Documents (`/schedules/{schedule_id}/documents`)

### POST /schedules/{schedule_id}/documents

- Auth: Required
- Purpose:
  - Upload one or more files
  - Create document records
  - Queue ingestion in background
- Content type: `multipart/form-data`
- Form fields:
  - `files`: repeated file parts
- Response:

```json
{
  "operation_ids": ["<operation_id>", "<operation_id>"],
  "documents": [
    {
      "id": "<doc_id>",
      "schedule_id": "<schedule_id>",
      "filename": "notes.pdf",
      "file_path": "uploads/...",
      "file_size": 12345,
      "ingest_status": "pending",
      "strategy": "rag",
      "doc_type": "notes",
      "ingest_report": {},
      "created_at": "..."
    }
  ]
}
```

### GET /schedules/{schedule_id}/documents

- Auth: Required
- Purpose: List documents for schedule
- Response: array of document objects

### GET /schedules/{schedule_id}/documents/{doc_id}

- Auth: Required
- Purpose: Get one document record
- Response: single document object

### GET /schedules/{schedule_id}/documents/{doc_id}/ingest-status

- Auth: Required
- Purpose:
  - Poll ingestion status (JSON mode)
  - Stream ingestion status (SSE mode)
- Query params:
  - `stream` (bool, default `false`)

#### JSON mode (`stream=false`)

Response:

```json
{
  "document_id": "<doc_id>",
  "ingest_status": "processing",
  "ingest_report": {},
  "operation_id": "<operation_id>"
}
```

#### SSE mode (`stream=true`)

Events:

- `operation`: includes `operation_id` when known
- `status`: periodic progress snapshot
- `done`: final status when ingestion finishes or fails

### DELETE /schedules/{schedule_id}/documents/{doc_id}

- Auth: Required
- Purpose:
  - Delete document and chunk mappings
  - Delete legacy DB references if present
  - Rebuild merged schedule index
- Response:

```json
{
  "deleted_document_id": "<doc_id>",
  "schedule_id": "<schedule_id>",
  "schedule_index_path": "indexes/schedules/<schedule_id>/merged"
}
```

## 5.5 Plans (`/schedules/{schedule_id}/plan`)

### POST /schedules/{schedule_id}/plan/generate

- Auth: Required
- Purpose: Generate a fresh draft study plan
- Request body:

```json
{
  "constraints": {
    "start_date": "2026-04-21",
    "end_date": "2026-05-05",
    "timezone_name": "Asia/Kolkata",
    "calendar_mode": "none",
    "daily_max_minutes": 120,
    "buffer_days": 1
  },
  "user_feedback": "No sessions on Friday. Prefer evening slots.",
  "feedback_history": ["Keep weekends light"]
}
```

- Response: plan object
- Long-running metadata: response includes optional `operation_id`

Plan object includes:

- `id`
- `schedule_id`
- `status` (`draft`, `active`, `confirmed`)
- `constraints`
- `review`
- `sessions`
- `session_count`
- timestamps

Planner behavior:

- Runs baseline schedule construction from constraints and calendar availability.
- Runs Gemma planner passes (non-interactive in API mode) to propose constraint/tool updates.
- Runs Qwen review and records severity/approval flags.
- If Qwen severity is critical, applies a safety reset and rebuild.
- If Qwen flags issues, runs one focused Gemma refinement pass.
- If `user_feedback` is provided, runs a feedback revision pass and re-reviews with Qwen.
- Persists model metadata under `constraints.model_notes` and exposes summary review status in `review`.

### POST /schedules/{schedule_id}/plan/generate-async

- Auth: Required
- Purpose: Queue draft plan generation in a background worker and return immediately
- Request body: same as `/generate`
- Response:

```json
{
  "operation_id": "<operation_id>",
  "kind": "plan_generate",
  "status": "running",
  "schedule_id": "<schedule_id>"
}
```

- Notes:
  - Use this endpoint when generation is long-running and you want responsive clients.
  - Poll `GET /schedules/{schedule_id}/plan/logs?operation_id=<operation_id>` for planner-style lines.
  - Or poll `GET /operations/{operation_id}/logs` for structured entries.

### GET /schedules/{schedule_id}/plan/logs

- Auth: Required
- Purpose: Retrieve planner operation logs in terminal-style format (`[Planner] ...`)
- Query params:
  - `operation_id` (required)
  - `offset` (>=0, default 0)
  - `limit` (1..500, default 200)
  - `include_metadata` (boolean, default false)
- Response:

```json
{
  "operation_id": "<operation_id>",
  "kind": "plan_generate",
  "status": "running",
  "entries": [
    {
      "seq": 3,
      "ts": "2026-04-21T18:10:29+00:00",
      "level": "info",
      "message": "Running Qwen review",
      "metadata": {
        "phase": "api_plan_generate"
      }
    }
  ],
  "next_offset": 4,
  "done": false,
  "total_entries": 4,
  "schedule_id": "<schedule_id>",
  "format": "planner_text",
  "lines": [
    "[Planner] Running Qwen review"
  ],
  "text": "[Planner] Running Qwen review"
}
```

### GET /schedules/{schedule_id}/plan

- Auth: Required
- Purpose: Get latest plan for schedule
- Response: plan object or `null`

### GET /schedules/{schedule_id}/plan/all

- Auth: Required
- Purpose: List all plans for schedule
- Behavior:
  - Returns every plan scoped to the schedule
  - Ordered by `updated_at` descending (latest first)
- Response: array of plan objects

### PATCH /schedules/{schedule_id}/plan

- Auth: Required
- Purpose: Revise latest plan using natural-language feedback
- Request body:

```json
{
  "feedback": "No sessions on Friday. Prefer evenings.",
  "feedback_history": ["Shift harder topics earlier"]
}
```

- Behavior:
  - Applies deterministic feedback-derived constraint updates (for reliability even on model fallback)
  - Runs one Gemma feedback revision pass (non-interactive)
  - Rebuilds schedule and runs Qwen feedback validation
  - Applies critical-severity reset when needed
  - Stores updated draft and feedback metadata
- Response: updated plan object
- Long-running metadata: response includes optional `operation_id`

### DELETE /schedules/{schedule_id}/plan/{plan_id}

- Auth: Required
- Purpose:
  - Delete one specific plan in the schedule
  - Delete dependent materialized study sessions for that plan
  - Delete session chat history tied to those removed sessions
- Response:

```json
{
  "deleted_plan_id": "<plan_id>",
  "deleted_plan_status": "draft",
  "schedule_id": "<schedule_id>",
  "deleted_materialized_sessions": 5,
  "deleted_session_chat_messages": 18,
  "operation_id": "<operation_id>"
}
```

### DELETE /schedules/{schedule_id}/plan

- Auth: Required
- Purpose:
  - Delete all plans for the schedule
  - Delete all dependent materialized sessions in the schedule
  - Delete all session chat history tied to those sessions
- Response:

```json
{
  "schedule_id": "<schedule_id>",
  "deleted_plans": 2,
  "deleted_plan_ids": ["<plan_id_1>", "<plan_id_2>"],
  "deleted_materialized_sessions": 10,
  "deleted_session_chat_messages": 34,
  "operation_id": "<operation_id>"
}
```

### POST /schedules/{schedule_id}/plan/confirm

- Auth: Required
- Purpose:
  - Activate latest draft
  - Materialize `study_sessions_api` rows
  - Demote previous active plans to confirmed
- Response:

```json
{
  "plan_id": "<plan_id>",
  "status": "active",
  "materialized_sessions": 5,
  "operation_id": "<operation_id>"
}
```

### GET /schedules/{schedule_id}/plan/sessions

- Auth: Required
- Purpose: List materialized study sessions
- Response: array of session summaries

### GET /schedules/{schedule_id}/plan/sessions/{session_id}

- Auth: Required
- Purpose: Detailed view of one materialized session
- Includes:
  - Focus chunks and previews
  - Prerequisites
  - Briefing text/status
  - Upcoming sessions preview

## 5.6 Schedule Chat (`/schedules/{schedule_id}/chat`)

### POST /schedules/{schedule_id}/chat

- Auth: Required
- Purpose:
  - Retrieve schedule-level chunks
  - Generate answer
  - Persist user + assistant turns
  - Stream answer via SSE
- Request body:

```json
{
  "message": "Explain Sobel vs Canny",
  "mode": "qa",
  "history": [],
  "max_minutes": null
}
```

- `mode` behavior:
  - `qa`: hybrid retrieval + rerank
  - `plan`: broader retrieval set
  - `beginner`: filtered retrieval for beginner complexity
  - `time_budget`: retrieval constrained by estimated time

SSE events:

- `operation` payload includes `operation_id`
- `sources`: retrieval citations
- `delta`: incremental text chunks
- `done`: final answer text

### GET /schedules/{schedule_id}/chat/history

- Auth: Required
- Purpose: Paginated history for schedule chat
- Query params:
  - `limit` (1..200, default 30)
  - `offset` (>=0, default 0)
- Response: array of chat messages

### DELETE /schedules/{schedule_id}/chat/history

- Auth: Required
- Purpose: Delete all schedule chat history for current user
- Response:

```json
{
  "deleted": 12
}
```

## 5.7 Session APIs (`/sessions`)

### POST /sessions/{session_id}/start

- Auth: Required
- Purpose:
  - Mark materialized session as `active`
  - Initialize/reset briefing stream
  - Trigger background briefing generation
- Response:

```json
{
  "id": "<session_id>",
  "plan_id": "<plan_id>",
  "schedule_id": "<schedule_id>",
  "session_number": 2,
  "title": "Study: Edge Detection",
  "scheduled_date": "2026-04-24",
  "start_time": "18:00:00",
  "end_time": "19:00:00",
  "status": "active",
  "briefing_status": "generating",
  "briefing_stream_url": "/sessions/<session_id>/briefing/stream",
  "operation_id": "<operation_id>"
}
```

### GET /sessions/{session_id}/briefing/stream

- Auth: Required
- Purpose: Stream generated briefing content
- Query params:
  - `operation_id` (optional): when provided, stream emits a leading `operation` event

Possible events:

- `operation`
- `status`: current briefing state
- `delta`: briefing text chunk
- `done`: completion signal
- `error`: generation failure

### POST /sessions/{session_id}/chat

- Auth: Required
- Purpose:
  - Two-stage retrieval for focused session chat
  - Save session chat turns
  - Stream answer
- Request body:

```json
{
  "message": "How do derivatives detect edges?",
  "history": []
}
```

Retrieval path logic:

- `local_chunk`: strong match in session focus chunks
- `rag_fallback`: fallback to schedule-level RAG retrieval

SSE events:

- `operation` payload includes `operation_id`
- `sources` payload includes:
  - `sources`
  - `retrieval_path`
  - `max_similarity`
  - `keyword_overlap`
- `delta`
- `done`

## 5.8 Operations (`/operations`)

### GET /operations/{operation_id}/logs

- Auth: Required
- Purpose:
  - Poll incremental logs for long-running operations (ingestion, plan generation/revision/confirm, briefing, chat)
  - Drive UI log windows with offset-based pagination
- Query params:
  - `offset` (>=0, default 0)
  - `limit` (1..500, default 200)
- Response:

```json
{
  "operation_id": "<operation_id>",
  "kind": "document_ingestion",
  "status": "running",
  "created_at": "2026-04-20T10:00:00+00:00",
  "updated_at": "2026-04-20T10:00:02+00:00",
  "entries": [
    {
      "seq": 1,
      "ts": "2026-04-20T10:00:00+00:00",
      "level": "info",
      "message": "Queued ingestion for notes.pdf",
      "metadata": {
        "schedule_id": "<schedule_id>",
        "filename": "notes.pdf"
      }
    }
  ],
  "next_offset": 1,
  "done": false,
  "total_entries": 1
}
```

- Notes:
  - Logs are user-scoped and process-local (in-memory).
  - Poll by reusing `next_offset` from previous response.

### GET /sessions/{session_id}/chat/history

- Auth: Required
- Purpose: Paginated history for one session chat
- Query params:
  - `limit` (1..200)
  - `offset` (>=0)

### POST /sessions/{session_id}/complete

- Auth: Required
- Purpose: Mark session as completed
- Response: updated session state

### GET /sessions/{session_id}/sidebar

- Auth: Required
- Purpose: Sidebar helper payload
- Response includes:
  - `prerequisites`
  - `upcoming_sessions` (up to next 3)

## 6. Data and Storage Flow

The backend uses two data/index domains:

- App DB (`backend_app.db`) via SQLAlchemy models in `backend/db/models.py`
- Legacy ingestion/index artifacts (`documents.db` and index files) reused by existing pipeline modules

Primary app tables:

- `users`
- `refresh_tokens`
- `schedules`
- `documents_api`
- `chunks_api`
- `study_plans_api`
- `study_sessions_api`
- `chat_messages`
- `session_chat_messages`

Index storage:

- Schedule merged index target: `indexes/schedules/{schedule_id}/merged.(index|pkl)`
- Legacy fallback: `indexes/sessions/{schedule_id}.(index|pkl)`

## 7. End-to-End Application Flow

## 7.1 Authentication and Session Establishment

1. Client authenticates with Google and sends ID token to `POST /auth/google`.
2. API validates token, upserts local user, returns access token, sets refresh cookie.
3. Client sends Bearer token for protected APIs.
4. When access token expires, client calls `POST /auth/refresh`.

## 7.2 Schedule Setup and Document Ingestion

1. Create schedule via `POST /schedules`.
2. Upload one or more files via `POST /schedules/{schedule_id}/documents`.
3. Background ingestion task (`run_document_ingestion`) does:
   - Calls legacy `main.process_document(...)`
   - Syncs legacy chunk rows into `chunks_api`
   - Mirrors index into schedule merged index path
4. Client tracks status through `GET .../ingest-status` (JSON or SSE).

## 7.3 Plan Generation and Revision

1. Generate initial draft with `POST /schedules/{schedule_id}/plan/generate`.
2. Plan service builds schedule using:
   - Chunk metadata
   - Constraints
   - Calendar availability (real or null mode)
3. Gemma planner passes can update constraints/tool hints before finalizing a draft.
4. Qwen review feedback is attached in plan constraints metadata and summarized in `review`.
5. User can iterate with `PATCH /schedules/{schedule_id}/plan` feedback.
6. Responses may indicate `review.user_feedback_requested=true` when additional feedback is recommended before confirm.
7. Confirm with `POST /schedules/{schedule_id}/plan/confirm` to materialize sessions.
8. Optional cleanup paths:
  - delete one plan: `DELETE /schedules/{schedule_id}/plan/{plan_id}`
  - delete all schedule plans: `DELETE /schedules/{schedule_id}/plan`
9. Historical inspection path:
  - list all plans: `GET /schedules/{schedule_id}/plan/all`

## 7.4 Study Session Execution

1. Start session with `POST /sessions/{session_id}/start`.
2. Briefing generation runs in background and is streamed by `GET /sessions/{session_id}/briefing/stream`.
3. During session, ask focused questions via `POST /sessions/{session_id}/chat`.
4. Two-stage retrieval tries local session chunks first, then schedule RAG fallback.
5. Complete session with `POST /sessions/{session_id}/complete`.
6. For any long-running action, poll `GET /operations/{operation_id}/logs` to render UI-friendly progress logs.

## 7.5 Schedule-Level Chat

1. Use `POST /schedules/{schedule_id}/chat` for broader schedule Q and A.
2. API retrieves relevant chunks from schedule index, generates answer, and streams SSE.
3. Chat history can be listed or cleared with history endpoints.

## 8. Error Patterns and Operational Notes

Common error classes:

- `401 Unauthorized`:
  - Missing/invalid Bearer token
  - Invalid Google token
  - Invalid refresh cookie/token
- `404 Not Found`:
  - Missing schedule/document/session scoped to user
- `400 Bad Request`:
  - Invalid planning constraints or no ingested chunks before planning
- `500/502`:
  - Upstream model/service issues (RAG index load, Google token endpoint, etc.)

Operational notes:

- SSE implementation is in-memory (`backend/sse/utils.py`), so stream state is process-local.
- Ingestion and briefing generation are background task based; restart can interrupt in-flight tasks.
- Refresh token rotation is enforced on each `/auth/refresh` call.

## 9. Quick Start Sequence (Recommended Client Order)

1. `POST /auth/google`
2. `POST /schedules`
3. `POST /schedules/{schedule_id}/documents`
4. `GET /schedules/{schedule_id}/documents/{doc_id}/ingest-status?stream=true`
5. `POST /schedules/{schedule_id}/plan/generate`
6. `PATCH /schedules/{schedule_id}/plan` (optional revisions)
7. `POST /schedules/{schedule_id}/plan/confirm`
8. `POST /sessions/{session_id}/start`
9. `GET /sessions/{session_id}/briefing/stream`
10. `POST /sessions/{session_id}/chat`
11. `POST /sessions/{session_id}/complete`
