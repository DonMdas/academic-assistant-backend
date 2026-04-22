import argparse

from db import DocumentDB
from calendar_planner_interface import GoogleCalendarService


def _choose_session(db):
    sessions = db.list_sessions()
    if not sessions:
        print("No sessions found.")
        return None

    print("\nAvailable sessions:\n")
    for idx, row in enumerate(sessions):
        session_id, title, created_at, doc_count = row
        print(f"[{idx}] {session_id} | {title or 'Untitled'} | docs={doc_count} | created={created_at}")

    raw = input("Select session index: ").strip()
    try:
        index = int(raw)
    except Exception:
        index = 0

    index = max(0, min(index, len(sessions) - 1))
    return sessions[index][0]


def _resolve_plan(db, session_id=None, plan_id=None):
    if plan_id:
        plan = db.get_study_plan(plan_id)
        if not plan:
            raise ValueError(f"Plan not found: {plan_id}")
        return plan

    if not session_id:
        raise ValueError("session_id or plan_id is required")

    latest = db.get_latest_study_plan(session_id=session_id)
    if not latest:
        raise ValueError("No study plan found for selected session")
    return latest


def main():
    parser = argparse.ArgumentParser(
        description="Delete Google Calendar events for a study plan and purge DB references.",
    )
    parser.add_argument("--session-id", default="", help="Session ID (optional if plan-id provided)")
    parser.add_argument("--plan-id", default="", help="Specific plan ID to delete")
    parser.add_argument("--calendar-id", default="", help="Calendar ID override")
    parser.add_argument("--timezone", default="UTC", help="Timezone for Google client initialization")
    parser.add_argument("--delete-plan-row", action="store_true", help="Delete row from study_plans instead of marking status=deleted")
    args = parser.parse_args()

    db = DocumentDB()

    session_id = args.session_id.strip()
    plan_id = args.plan_id.strip()

    if not session_id and not plan_id:
        session_id = _choose_session(db)
        if not session_id:
            return

    plan = _resolve_plan(db, session_id=session_id, plan_id=plan_id)
    plan_id = plan["plan_id"]

    calendar_id = (args.calendar_id or plan.get("calendar_id") or "primary").strip() or "primary"
    event_ids = db.list_plan_calendar_event_ids(plan_id)

    print(f"\nPlan selected: {plan_id}")
    print(f"Session: {plan.get('session_id')}")
    print(f"Calendar ID: {calendar_id}")
    print(f"Events to delete: {len(event_ids)}")

    confirm = input("Proceed with calendar + DB reference cleanup? (yes/no): ").strip().lower()
    if confirm not in {"yes", "y"}:
        print("Cancelled.")
        return

    deleted = 0
    failed = 0

    if event_ids:
        calendar_service = GoogleCalendarService(timezone_name=args.timezone)
        delete_results = calendar_service.delete_events(calendar_id=calendar_id, event_ids=event_ids)

        for row in delete_results:
            status = str(row.get("status", "")).lower()
            if status == "deleted":
                deleted += 1
            else:
                failed += 1

        print(f"Calendar event deletion: deleted={deleted} failed={failed}")

    db.purge_plan_references(plan_id=plan_id, delete_plan_row=bool(args.delete_plan_row))
    print("DB reference cleanup complete (study_plan_slots + study_chunk_schedule).")

    if args.delete_plan_row:
        print("Plan row deleted from study_plans.")
    else:
        print("Plan status updated to deleted.")


if __name__ == "__main__":
    main()
