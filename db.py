# db.py

import sqlite3
import json
import uuid

class DocumentDB:
    def __init__(self, db_path="documents.db"):
        self.conn = sqlite3.connect(db_path)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            strategy TEXT,
            doc_type TEXT,
            total_chars INTEGER,
            index_path TEXT,
            session_id TEXT,
            ingest_report_json TEXT
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS study_sessions (
            session_id TEXT PRIMARY KEY,
            title TEXT,
            rag_index_path TEXT,
            session_topics_json TEXT,
            session_summary TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS session_documents (
            session_id TEXT,
            doc_id TEXT,
            added_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (session_id, doc_id)
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT,
            content TEXT,
            metadata TEXT
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS study_plans (
            plan_id TEXT PRIMARY KEY,
            session_id TEXT,
            start_date TEXT,
            end_date TEXT,
            constraints_json TEXT,
            coverage_json TEXT,
            raw_plan_json TEXT,
            status TEXT DEFAULT 'draft',
            calendar_mode TEXT,
            calendar_id TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            approved_at TEXT
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS study_plan_slots (
            slot_id TEXT PRIMARY KEY,
            plan_id TEXT,
            session_id TEXT,
            start_time TEXT,
            end_time TEXT,
            duration_minutes INTEGER,
            difficulty TEXT,
            items_json TEXT,
            prerequisites_json TEXT,
            coverage_chunk_ids_json TEXT,
            calendar_event_id TEXT,
            calendar_status TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS study_chunk_schedule (
            mapping_id TEXT PRIMARY KEY,
            plan_id TEXT,
            session_id TEXT,
            slot_id TEXT,
            chunk_id TEXT,
            topic TEXT,
            prerequisites_json TEXT,
            schedule_date TEXT,
            start_time TEXT,
            end_time TEXT,
            calendar_event_id TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """)

        cursor.execute("PRAGMA table_info(documents)")
        columns = {row[1] for row in cursor.fetchall()}

        if "index_path" not in columns:
            cursor.execute("ALTER TABLE documents ADD COLUMN index_path TEXT")
        if "session_id" not in columns:
            cursor.execute("ALTER TABLE documents ADD COLUMN session_id TEXT")
        if "ingest_report_json" not in columns:
            cursor.execute("ALTER TABLE documents ADD COLUMN ingest_report_json TEXT")

        cursor.execute("PRAGMA table_info(study_sessions)")
        session_columns = {row[1] for row in cursor.fetchall()}

        if "rag_index_path" not in session_columns:
            cursor.execute("ALTER TABLE study_sessions ADD COLUMN rag_index_path TEXT")
        if "session_topics_json" not in session_columns:
            cursor.execute("ALTER TABLE study_sessions ADD COLUMN session_topics_json TEXT")
        if "session_summary" not in session_columns:
            cursor.execute("ALTER TABLE study_sessions ADD COLUMN session_summary TEXT")

        cursor.execute("PRAGMA table_info(study_chunk_schedule)")
        chunk_schedule_columns = {row[1] for row in cursor.fetchall()}
        if "prerequisites_json" not in chunk_schedule_columns:
            cursor.execute("ALTER TABLE study_chunk_schedule ADD COLUMN prerequisites_json TEXT")

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_session_id ON documents(session_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_session_documents_session_id ON session_documents(session_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_session_documents_doc_id ON session_documents(doc_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_study_plans_session_id ON study_plans(session_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_study_plan_slots_plan_id ON study_plan_slots(plan_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_study_plan_slots_session_id ON study_plan_slots(session_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_study_chunk_schedule_plan_id ON study_chunk_schedule(plan_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_study_chunk_schedule_slot_id ON study_chunk_schedule(slot_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_study_chunk_schedule_chunk_id ON study_chunk_schedule(chunk_id)"
        )

        self.conn.commit()

    def create_session(self, title=None, session_id=None):
        resolved_session_id = str(session_id or uuid.uuid4())
        self.conn.execute(
            "INSERT OR IGNORE INTO study_sessions (session_id, title) VALUES (?, ?)",
            (resolved_session_id, title)
        )

        if title:
            self.conn.execute(
                "UPDATE study_sessions SET title=? WHERE session_id=?",
                (title, resolved_session_id)
            )

        self.conn.commit()
        return resolved_session_id

    def ensure_session(self, session_id=None, title=None):
        if session_id:
            row = self.conn.execute(
                "SELECT session_id FROM study_sessions WHERE session_id=?",
                (session_id,)
            ).fetchone()
            if row:
                if title:
                    self.conn.execute(
                        "UPDATE study_sessions SET title=? WHERE session_id=?",
                        (title, session_id)
                    )
                    self.conn.commit()
                return session_id

        return self.create_session(title=title, session_id=session_id)

    def add_document_to_session(self, session_id, doc_id):
        self.conn.execute(
            "INSERT OR IGNORE INTO session_documents (session_id, doc_id) VALUES (?, ?)",
            (session_id, doc_id)
        )
        self.conn.execute(
            "UPDATE documents SET session_id=? WHERE doc_id=?",
            (session_id, doc_id)
        )
        self.conn.commit()

    def list_sessions(self):
        cursor = self.conn.execute(
            """
            SELECT s.session_id, s.title, s.created_at, COUNT(sd.doc_id) AS document_count
            FROM study_sessions s
            LEFT JOIN session_documents sd ON sd.session_id = s.session_id
            GROUP BY s.session_id, s.title, s.created_at
            ORDER BY s.created_at DESC
            """
        )
        return cursor.fetchall()

    def get_session(self, session_id):
        cursor = self.conn.execute(
            "SELECT session_id, title, created_at FROM study_sessions WHERE session_id=?",
            (session_id,)
        )
        return cursor.fetchone()

    def get_session_profile(self, session_id):
        row = self.conn.execute(
            """
            SELECT session_id, title, created_at, rag_index_path, session_topics_json, session_summary
            FROM study_sessions
            WHERE session_id=?
            """,
            (session_id,),
        ).fetchone()

        if not row:
            return None

        topics = []
        raw_topics = row[4]
        if raw_topics:
            try:
                parsed = json.loads(raw_topics)
                if isinstance(parsed, list):
                    topics = [str(item).strip() for item in parsed if str(item).strip()]
            except Exception:
                topics = []

        return {
            "session_id": row[0],
            "title": row[1],
            "created_at": row[2],
            "rag_index_path": row[3],
            "topics": topics,
            "summary": row[5] or "",
        }

    def update_session_profile(self, session_id, topics=None, summary=None, rag_index_path=None):
        topics = list(topics or [])
        cleaned_topics = []
        seen = set()
        for topic in topics:
            text = str(topic).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned_topics.append(text)

        topics_json = json.dumps(cleaned_topics)
        session_summary = str(summary or "").strip()

        self.conn.execute(
            """
            UPDATE study_sessions
            SET rag_index_path=COALESCE(?, rag_index_path),
                session_topics_json=?,
                session_summary=?
            WHERE session_id=?
            """,
            (rag_index_path, topics_json, session_summary, session_id),
        )
        self.conn.commit()

    def get_session_documents(self, session_id):
        cursor = self.conn.execute(
            """
            SELECT d.doc_id, d.strategy, d.doc_type, d.total_chars, d.index_path
            FROM session_documents sd
            JOIN documents d ON d.doc_id = sd.doc_id
            WHERE sd.session_id=?
            ORDER BY sd.added_at ASC
            """,
            (session_id,)
        )
        return cursor.fetchall()

    def get_session_chunks(self, session_id):
        cursor = self.conn.execute(
            """
            SELECT c.chunk_id, c.doc_id, c.content, c.metadata
            FROM chunks c
            JOIN session_documents sd ON sd.doc_id = c.doc_id
            WHERE sd.session_id=?
            ORDER BY c.chunk_id ASC
            """,
            (session_id,)
        )
        return cursor.fetchall()

    def store_document(self, doc_id, strategy, doc_type, total_chars, index_path=None, session_id=None, ingest_report=None):
        ingest_report_json = None
        if ingest_report is not None:
            try:
                ingest_report_json = json.dumps(ingest_report)
            except Exception:
                ingest_report_json = json.dumps({"error": "failed_to_serialize_ingest_report"})

        self.conn.execute(
            "INSERT INTO documents (doc_id, strategy, doc_type, total_chars, index_path, session_id, ingest_report_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (doc_id, strategy, doc_type, total_chars, index_path, session_id, ingest_report_json)
        )

        if session_id:
            self.add_document_to_session(session_id, doc_id)
            return

        self.conn.commit()

    def store_chunks(self, doc_id, chunks):
        for chunk in chunks:
            raw_chunk_id = str(chunk.get("chunk_id", ""))
            if not raw_chunk_id:
                raw_chunk_id = "chunk"

            # Keep chunk_id unique across documents in the current schema.
            storage_chunk_id = f"{doc_id}:{raw_chunk_id}"

            self.conn.execute(
                "INSERT INTO chunks (chunk_id, doc_id, content, metadata) VALUES (?, ?, ?, ?)",
                (
                    storage_chunk_id,
                    doc_id,
                    chunk["text"],
                    json.dumps(chunk["metadata"])
                )
            )
        self.conn.commit()
        
    def list_documents(self, session_id=None):
        if session_id:
            cursor = self.conn.execute(
                """
                SELECT d.doc_id, d.strategy, d.doc_type
                FROM session_documents sd
                JOIN documents d ON d.doc_id = sd.doc_id
                WHERE sd.session_id=?
                ORDER BY sd.added_at ASC
                """,
                (session_id,)
            )
            return cursor.fetchall()

        cursor = self.conn.execute("SELECT doc_id, strategy, doc_type FROM documents")
        return cursor.fetchall()

    def get_document(self, doc_id):
        cursor = self.conn.execute(
            "SELECT * FROM documents WHERE doc_id=?", (doc_id,)
        )
        return cursor.fetchone()

    def create_study_plan(
        self,
        plan_id,
        session_id,
        start_date,
        end_date,
        constraints=None,
        coverage=None,
        raw_plan=None,
        status="draft",
        calendar_mode=None,
        calendar_id=None,
    ):
        constraints_json = json.dumps(constraints or {})
        coverage_json = json.dumps(coverage or {})
        raw_plan_json = json.dumps(raw_plan or {})

        self.conn.execute(
            """
            INSERT INTO study_plans (
                plan_id,
                session_id,
                start_date,
                end_date,
                constraints_json,
                coverage_json,
                raw_plan_json,
                status,
                calendar_mode,
                calendar_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                plan_id,
                session_id,
                start_date,
                end_date,
                constraints_json,
                coverage_json,
                raw_plan_json,
                status,
                calendar_mode,
                calendar_id,
            ),
        )
        self.conn.commit()

    def update_study_plan(
        self,
        plan_id,
        status=None,
        constraints=None,
        coverage=None,
        raw_plan=None,
        approved=False,
        calendar_mode=None,
        calendar_id=None,
    ):
        updates = []
        values = []

        if status is not None:
            updates.append("status=?")
            values.append(status)
        if constraints is not None:
            updates.append("constraints_json=?")
            values.append(json.dumps(constraints))
        if coverage is not None:
            updates.append("coverage_json=?")
            values.append(json.dumps(coverage))
        if raw_plan is not None:
            updates.append("raw_plan_json=?")
            values.append(json.dumps(raw_plan))
        if calendar_mode is not None:
            updates.append("calendar_mode=?")
            values.append(calendar_mode)
        if calendar_id is not None:
            updates.append("calendar_id=?")
            values.append(calendar_id)
        if approved:
            updates.append("approved_at=CURRENT_TIMESTAMP")

        if not updates:
            return

        values.append(plan_id)
        sql = f"UPDATE study_plans SET {', '.join(updates)} WHERE plan_id=?"
        self.conn.execute(sql, tuple(values))
        self.conn.commit()

    def get_study_plan(self, plan_id):
        row = self.conn.execute(
            """
            SELECT
                plan_id,
                session_id,
                start_date,
                end_date,
                constraints_json,
                coverage_json,
                raw_plan_json,
                status,
                calendar_mode,
                calendar_id,
                created_at,
                approved_at
            FROM study_plans
            WHERE plan_id=?
            """,
            (plan_id,),
        ).fetchone()

        if not row:
            return None

        def _parse_json(value, default):
            if not value:
                return default
            try:
                parsed = json.loads(value)
                return parsed
            except Exception:
                return default

        return {
            "plan_id": row[0],
            "session_id": row[1],
            "start_date": row[2],
            "end_date": row[3],
            "constraints": _parse_json(row[4], {}),
            "coverage": _parse_json(row[5], {}),
            "raw_plan": _parse_json(row[6], {}),
            "status": row[7],
            "calendar_mode": row[8],
            "calendar_id": row[9],
            "created_at": row[10],
            "approved_at": row[11],
        }

    def get_latest_study_plan(self, session_id, statuses=None):
        params = [session_id]
        where_status = ""
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            where_status = f" AND status IN ({placeholders})"
            params.extend(list(statuses))

        row = self.conn.execute(
            f"""
            SELECT plan_id
            FROM study_plans
            WHERE session_id=? {where_status}
            ORDER BY created_at DESC
            LIMIT 1
            """,
            tuple(params),
        ).fetchone()

        if not row:
            return None

        return self.get_study_plan(row[0])

    def replace_study_plan_slots(self, plan_id, session_id, slots):
        self.conn.execute("DELETE FROM study_plan_slots WHERE plan_id=?", (plan_id,))

        for index, slot in enumerate(slots, 1):
            slot_id = str(slot.get("slot_id") or uuid.uuid4())
            items = slot.get("items", [])
            prerequisites = slot.get("prerequisites", [])
            coverage_chunk_ids = slot.get("coverage_chunk_ids", [])

            self.conn.execute(
                """
                INSERT INTO study_plan_slots (
                    slot_id,
                    plan_id,
                    session_id,
                    start_time,
                    end_time,
                    duration_minutes,
                    difficulty,
                    items_json,
                    prerequisites_json,
                    coverage_chunk_ids_json,
                    calendar_event_id,
                    calendar_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    slot_id,
                    plan_id,
                    session_id,
                    slot.get("start_time"),
                    slot.get("end_time"),
                    int(slot.get("duration_minutes", 0) or 0),
                    str(slot.get("difficulty", "intermediate")),
                    json.dumps(items),
                    json.dumps(prerequisites),
                    json.dumps(coverage_chunk_ids),
                    slot.get("calendar_event_id"),
                    slot.get("calendar_status"),
                ),
            )

            slot["slot_id"] = slot_id
            slot["slot_index"] = index

        self.conn.commit()

    def list_study_plan_slots(self, plan_id):
        rows = self.conn.execute(
            """
            SELECT
                slot_id,
                start_time,
                end_time,
                duration_minutes,
                difficulty,
                items_json,
                prerequisites_json,
                coverage_chunk_ids_json,
                calendar_event_id,
                calendar_status
            FROM study_plan_slots
            WHERE plan_id=?
            ORDER BY start_time ASC
            """,
            (plan_id,),
        ).fetchall()

        def _safe_json(value, default):
            if not value:
                return default
            try:
                parsed = json.loads(value)
                return parsed
            except Exception:
                return default

        slots = []
        for row in rows:
            slots.append({
                "slot_id": row[0],
                "start_time": row[1],
                "end_time": row[2],
                "duration_minutes": row[3],
                "difficulty": row[4],
                "items": _safe_json(row[5], []),
                "prerequisites": _safe_json(row[6], []),
                "coverage_chunk_ids": _safe_json(row[7], []),
                "calendar_event_id": row[8],
                "calendar_status": row[9],
            })

        return slots

    def update_study_plan_slot_calendar(self, slot_id, calendar_event_id=None, calendar_status=None):
        updates = []
        values = []

        if calendar_event_id is not None:
            updates.append("calendar_event_id=?")
            values.append(calendar_event_id)
        if calendar_status is not None:
            updates.append("calendar_status=?")
            values.append(calendar_status)

        if not updates:
            return

        values.append(slot_id)
        sql = f"UPDATE study_plan_slots SET {', '.join(updates)} WHERE slot_id=?"
        self.conn.execute(sql, tuple(values))
        self.conn.commit()

    def replace_chunk_schedule_refs(self, plan_id, session_id, slots):
        self.conn.execute("DELETE FROM study_chunk_schedule WHERE plan_id=?", (plan_id,))

        for slot in slots:
            slot_id = str(slot.get("slot_id") or "")
            slot_start = str(slot.get("start_time") or "")
            slot_end = str(slot.get("end_time") or "")
            slot_calendar_event_id = slot.get("calendar_event_id")

            for item in slot.get("items", []):
                chunk_id = str(item.get("chunk_id") or "").strip()
                if not chunk_id:
                    continue

                schedule_date = str(item.get("scheduled_date") or "").strip()
                if not schedule_date:
                    schedule_date = slot_start[:10] if len(slot_start) >= 10 else ""

                start_time = str(item.get("scheduled_start_time") or slot_start)
                end_time = str(item.get("scheduled_end_time") or slot_end)

                self.conn.execute(
                    """
                    INSERT INTO study_chunk_schedule (
                        mapping_id,
                        plan_id,
                        session_id,
                        slot_id,
                        chunk_id,
                        topic,
                        prerequisites_json,
                        schedule_date,
                        start_time,
                        end_time,
                        calendar_event_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        plan_id,
                        session_id,
                        slot_id,
                        chunk_id,
                        str(item.get("topic") or ""),
                        json.dumps(item.get("prerequisites", [])),
                        schedule_date,
                        start_time,
                        end_time,
                        slot_calendar_event_id,
                    ),
                )

        self.conn.commit()

    def list_chunk_schedule_refs(self, plan_id):
        rows = self.conn.execute(
            """
            SELECT mapping_id, slot_id, chunk_id, topic, prerequisites_json, schedule_date, start_time, end_time, calendar_event_id
            FROM study_chunk_schedule
            WHERE plan_id=?
            ORDER BY schedule_date ASC, start_time ASC
            """,
            (plan_id,),
        ).fetchall()

        def _safe_json(value, default):
            if not value:
                return default
            try:
                return json.loads(value)
            except Exception:
                return default

        refs = []
        for row in rows:
            refs.append({
                "mapping_id": row[0],
                "slot_id": row[1],
                "chunk_id": row[2],
                "topic": row[3],
                "prerequisites": _safe_json(row[4], []),
                "schedule_date": row[5],
                "start_time": row[6],
                "end_time": row[7],
                "calendar_event_id": row[8],
            })
        return refs

    def update_chunk_schedule_calendar_by_slot(self, slot_id, calendar_event_id=None):
        self.conn.execute(
            "UPDATE study_chunk_schedule SET calendar_event_id=? WHERE slot_id=?",
            (calendar_event_id, slot_id),
        )
        self.conn.commit()

    def list_plan_calendar_event_ids(self, plan_id):
        rows = self.conn.execute(
            """
            SELECT DISTINCT calendar_event_id
            FROM study_plan_slots
            WHERE plan_id=? AND calendar_event_id IS NOT NULL AND TRIM(calendar_event_id) <> ''
            """,
            (plan_id,),
        ).fetchall()
        return [str(row[0]) for row in rows if row and row[0]]

    def purge_plan_references(self, plan_id, delete_plan_row=False):
        self.conn.execute("DELETE FROM study_chunk_schedule WHERE plan_id=?", (plan_id,))
        self.conn.execute("DELETE FROM study_plan_slots WHERE plan_id=?", (plan_id,))

        if delete_plan_row:
            self.conn.execute("DELETE FROM study_plans WHERE plan_id=?", (plan_id,))
        else:
            self.conn.execute(
                "UPDATE study_plans SET status='deleted' WHERE plan_id=?",
                (plan_id,),
            )

        self.conn.commit()