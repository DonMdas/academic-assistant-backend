# Academic Assistant

This workspace contains a FastAPI backend and a React frontend for document ingestion, study-plan generation, and session-focused learning support.

## Key Update

Long-running backend operations now expose operation-level logs that the frontend polls and renders in a dedicated log window.

- New API: `GET /operations/{operation_id}/logs`
- Long-running endpoints now provide or emit `operation_id`
- Frontend tracks operation IDs and polls logs incrementally with offsets

## Project Structure

- `backend/`: FastAPI application and service layers
- `frontend/`: React + TypeScript client
- `api documentation.md`: full endpoint contracts and SSE behavior

## Run Backend

Use your existing backend startup command for this workspace (for example via uvicorn).

## Run Frontend

From `frontend/`:

```bash
npm install
npm run dev
```

The Vite dev server proxies `/api` to `VITE_PROXY_TARGET` (default `http://127.0.0.1:8000`).
