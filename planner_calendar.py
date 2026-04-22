import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, time, timezone, timedelta
from zoneinfo import ZoneInfo

from planner_common import (
    CalendarEvent,
    _normalize_bool,
    _normalize_blocked_dates,
    _normalize_blocked_weekdays,
    _normalize_optional_int,
    _normalize_timezone_name,
    _normalize_specific_day_windows,
    _parse_time_hhmm,
    _resolve_timezone,
)

#SCOPES = ["https://www.googleapis.com/auth/calendar"]

SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]


class GoogleCalendarService:
    """
    Real Google Calendar integration only.

    Required files in project root:
    - credentials.json (OAuth desktop client)

    First run opens browser for consent and stores token.json.
    """

    def __init__(
        self,
        timezone_name,
        credentials_path="credentials.json",
        token_path="token.json",
        credentials_info=None,
        allow_local_oauth=True,
    ):
        tz_name = _normalize_timezone_name(timezone_name)
        tz = _resolve_timezone(tz_name)
        if not tz_name or tz_name.upper() in {"UTC", "ETC/UTC", "Z"}:
            tz_name = "UTC"
        else:
            try:
                ZoneInfo(tz_name)
            except Exception:
                tz_name = "UTC"
                tz = timezone.utc

        self.timezone_name = tz_name
        self.tzinfo = tz
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.credentials_info = dict(credentials_info or {})
        self.allow_local_oauth = bool(allow_local_oauth)
        self.credentials = None
        self.service = self._build_service()

    def _build_service(self):
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
        except Exception as exc:
            raise RuntimeError(
                "Google Calendar libraries missing. Install requirements first."
            ) from exc

        creds = None
        if self.credentials_info:
            try:
                creds = Credentials.from_authorized_user_info(self.credentials_info, SCOPES)
            except Exception:
                creds = None

        if creds is None and self.allow_local_oauth and os.path.exists(self.token_path):
            try:
                creds = Credentials.from_authorized_user_file(self.token_path, SCOPES)
            except Exception:
                creds = None

        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())

        if not creds or not creds.valid:
            if not self.allow_local_oauth:
                raise RuntimeError("Google Calendar is not connected for this user.")

            if not os.path.exists(self.credentials_path):
                raise FileNotFoundError(
                    "credentials.json not found. Create OAuth desktop client credentials and place file in project root."
                )

            oauth_host = str(os.getenv("GOOGLE_OAUTH_LOCAL_HOST", "localhost") or "localhost").strip() or "localhost"
            oauth_port_raw = str(os.getenv("GOOGLE_OAUTH_LOCAL_PORT", "8080") or "8080").strip()
            try:
                oauth_port = int(oauth_port_raw)
            except Exception:
                oauth_port = 8080

            client_type = "unknown"
            try:
                with open(self.credentials_path, "r", encoding="utf-8") as handle:
                    creds_payload = json.load(handle)
                if isinstance(creds_payload, dict):
                    if "installed" in creds_payload:
                        client_type = "installed"
                    elif "web" in creds_payload:
                        client_type = "web"
            except Exception:
                client_type = "unknown"

            if client_type == "web":
                print(
                    "OAuth note: credentials.json appears to be a Web client. "
                    "If login fails, add the loopback redirect URI shown in the error message "
                    "or replace with a Desktop App OAuth client."
                )

            print(f"Starting Google OAuth flow on http://{oauth_host}:{oauth_port}/ ...")

            flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, SCOPES)
            try:
                creds = flow.run_local_server(host=oauth_host, port=oauth_port)
            except Exception as exc:
                redirect_uri = f"http://{oauth_host}:{oauth_port}/"
                alt_redirect_uri = f"http://127.0.0.1:{oauth_port}/"
                raise RuntimeError(
                    "Google OAuth login failed. If you are using a Web OAuth client, register "
                    f"these Authorized redirect URIs in Google Cloud Console: {redirect_uri} and {alt_redirect_uri}. "
                    "Or replace credentials.json with a Desktop App OAuth client. "
                    f"Original error: {exc}"
                ) from exc

            with open(self.token_path, "w", encoding="utf-8") as handle:
                handle.write(creds.to_json())

            print("Google OAuth completed. Initializing Calendar API client...")

        self.credentials = creds
        service = build("calendar", "v3", credentials=creds, cache_discovery=False)
        print("Google Calendar client ready.")
        return service

    def get_credentials_payload(self):
        if self.credentials is None:
            return {}
        try:
            return dict(json.loads(self.credentials.to_json()) or {})
        except Exception:
            return {}

    def get_authenticated_identity(self):
        identity = {
            "google_id": "",
            "email": "",
            "primary_calendar_id": "",
        }

        access_token = str(getattr(self.credentials, "token", "") or "").strip()
        if access_token:
            params = urllib.parse.urlencode({"access_token": access_token})
            url = f"https://oauth2.googleapis.com/tokeninfo?{params}"
            try:
                with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310
                    payload = json.loads(response.read().decode("utf-8"))
                if isinstance(payload, dict):
                    identity["google_id"] = str(payload.get("sub") or payload.get("user_id") or "").strip()
                    identity["email"] = str(payload.get("email") or "").strip().lower()
            except Exception:
                pass

        try:
            primary = self.service.calendars().get(calendarId="primary").execute()
            primary_calendar_id = str(dict(primary or {}).get("id") or "").strip().lower()
            if primary_calendar_id:
                identity["primary_calendar_id"] = primary_calendar_id
                if not identity["email"] and "@" in primary_calendar_id:
                    identity["email"] = primary_calendar_id
        except Exception:
            pass

        return identity

    def list_events(self, calendar_id, start_dt, end_dt):
        payload = self.service.events().list(
            calendarId=calendar_id,
            timeMin=start_dt.isoformat(),
            timeMax=end_dt.isoformat(),
            singleEvents=True,
            orderBy="startTime",
        ).execute()

        events = []
        for item in payload.get("items", []):
            start_info = item.get("start", {})
            end_info = item.get("end", {})
            start_dt_raw = start_info.get("dateTime")
            end_dt_raw = end_info.get("dateTime")
            all_day_start = start_info.get("date")
            all_day_end = end_info.get("date")

            parsed_start = None
            parsed_end = None

            if start_dt_raw and end_dt_raw:
                parsed_start = datetime.fromisoformat(start_dt_raw.replace("Z", "+00:00"))
                parsed_end = datetime.fromisoformat(end_dt_raw.replace("Z", "+00:00"))
            elif all_day_start and all_day_end:
                start_day = datetime.strptime(all_day_start, "%Y-%m-%d").date()
                end_day = datetime.strptime(all_day_end, "%Y-%m-%d").date()
                parsed_start = datetime.combine(start_day, time.min, tzinfo=self.tzinfo)
                parsed_end = datetime.combine(end_day, time.min, tzinfo=self.tzinfo)

            if not parsed_start or not parsed_end:
                continue

            events.append(
                CalendarEvent(
                    start=parsed_start,
                    end=parsed_end,
                    summary=str(item.get("summary", "Busy")),
                    event_id=item.get("id"),
                )
            )

        return events

    def create_events(self, calendar_id, slots):
        results = []

        for slot in slots:
            body = {
                "summary": slot.get("title", "Study Session"),
                "description": slot.get("description", ""),
                "start": {
                    "dateTime": slot["start_time"],
                    "timeZone": self.timezone_name,
                },
                "end": {
                    "dateTime": slot["end_time"],
                    "timeZone": self.timezone_name,
                },
            }

            try:
                created = self.service.events().insert(calendarId=calendar_id, body=body).execute()
                results.append({
                    "slot_id": slot.get("slot_id"),
                    "event_id": created.get("id"),
                    "status": "created",
                })
            except Exception as exc:
                results.append({
                    "slot_id": slot.get("slot_id"),
                    "event_id": None,
                    "status": f"failed: {exc}",
                })

        return results

    def delete_events(self, calendar_id, event_ids):
        results = []
        for event_id in list(event_ids or []):
            event_id = str(event_id or "").strip()
            if not event_id:
                continue
            try:
                self.service.events().delete(calendarId=calendar_id, eventId=event_id).execute()
                results.append({
                    "event_id": event_id,
                    "status": "deleted",
                })
            except Exception as exc:
                results.append({
                    "event_id": event_id,
                    "status": f"failed: {exc}",
                })
        return results


def _merge_intervals(intervals):
    ordered = sorted(intervals, key=lambda item: item[0])
    if not ordered:
        return []

    merged = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


def _subtract_interval(base_start, base_end, busy_start, busy_end):
    if busy_end <= base_start or busy_start >= base_end:
        return [(base_start, base_end)]

    out = []
    if busy_start > base_start:
        out.append((base_start, min(busy_start, base_end)))
    if busy_end < base_end:
        out.append((max(busy_end, base_start), base_end))
    return out


def compute_free_blocks(start_date, end_date, constraints, tzinfo, busy_events):
    include_weekends_raw = constraints.get("include_weekends", True)
    include_weekends = _normalize_bool(include_weekends_raw, default=True)
    blocked_weekdays = set(_normalize_blocked_weekdays(constraints.get("blocked_weekdays", [])))
    blocked_dates = set(_normalize_blocked_dates(constraints.get("blocked_dates", [])))
    daily_cap = _normalize_optional_int(constraints.get("daily_max_minutes"), minimum=30, maximum=720)

    start_clock = _parse_time_hhmm(
        constraints.get("study_window_start"),
        time(hour=18, minute=0),
    )
    end_clock = _parse_time_hhmm(
        constraints.get("study_window_end"),
        time(hour=22, minute=30),
    )

    busy_ranges = _merge_intervals([(event.start, event.end) for event in busy_events])
    specific_day_windows = _normalize_specific_day_windows(constraints.get("specific_day_windows", []))

    current = start_date
    free_blocks = []
    while current <= end_date:
        if current.isoformat() in blocked_dates:
            current += timedelta(days=1)
            continue

        if current.weekday() in blocked_weekdays:
            current += timedelta(days=1)
            continue

        if (not include_weekends) and current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        day_start = datetime.combine(current, start_clock, tzinfo=tzinfo)
        day_end = datetime.combine(current, end_clock, tzinfo=tzinfo)

        if day_end <= day_start:
            day_end = day_start + timedelta(hours=2)

        day_key = current.isoformat()
        matching_windows = [row for row in specific_day_windows if str(row.get("date", "")) == day_key]

        hard_windows = []
        soft_windows = []
        for row in matching_windows:
            row_start = _parse_time_hhmm(row.get("start"), start_clock)
            row_end = _parse_time_hhmm(row.get("end"), end_clock)

            block_start = datetime.combine(current, row_start, tzinfo=tzinfo)
            block_end = datetime.combine(current, row_end, tzinfo=tzinfo)
            if block_end <= block_start:
                block_end = block_start + timedelta(hours=2)

            if str(row.get("priority", "hard")).lower() == "hard":
                hard_windows.append((block_start, block_end))
            else:
                soft_windows.append((block_start, block_end))

        if hard_windows:
            day_blocks = list(hard_windows)
        else:
            day_blocks = [(day_start, day_end)] + soft_windows

        day_blocks = _merge_intervals(day_blocks)
        if not day_blocks:
            current += timedelta(days=1)
            continue

        day_min_start = min(block[0] for block in day_blocks)
        day_max_end = max(block[1] for block in day_blocks)

        for busy_start, busy_end in busy_ranges:
            if busy_end <= day_min_start or busy_start >= day_max_end:
                continue

            next_blocks = []
            for block_start, block_end in day_blocks:
                next_blocks.extend(_subtract_interval(block_start, block_end, busy_start, busy_end))
            day_blocks = next_blocks
            if not day_blocks:
                break

        cap_left = daily_cap if daily_cap is not None else 10**9
        for block_start, block_end in day_blocks:
            duration = int((block_end - block_start).total_seconds() // 60)
            if duration <= 0:
                continue

            usable = min(duration, cap_left)
            if usable <= 0:
                break

            free_blocks.append((block_start, block_start + timedelta(minutes=usable)))
            cap_left -= usable
            if cap_left <= 0:
                break

        current += timedelta(days=1)

    return free_blocks
