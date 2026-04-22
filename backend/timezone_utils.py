from datetime import date, datetime
from zoneinfo import ZoneInfo

IST_TZ = ZoneInfo("Asia/Kolkata")


def now_ist() -> datetime:
    return datetime.now(IST_TZ)


def now_ist_naive() -> datetime:
    return now_ist().replace(tzinfo=None)


def today_ist() -> date:
    return now_ist().date()


def iso_now_ist() -> str:
    return now_ist().isoformat()


def to_ist_naive(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(IST_TZ).replace(tzinfo=None)


def to_ist_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=IST_TZ)
    return value.astimezone(IST_TZ)
