import asyncio
import json
from dataclasses import dataclass, field


def format_sse(event: str, payload: dict) -> str:
    body = json.dumps(payload, ensure_ascii=True)
    return f"event: {event}\ndata: {body}\n\n"


@dataclass
class StreamState:
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    closed: bool = False


class InMemorySSEManager:
    def __init__(self) -> None:
        self._streams: dict[str, StreamState] = {}

    def create_or_reset(self, stream_id: str) -> StreamState:
        state = StreamState()
        self._streams[stream_id] = state
        return state

    def exists(self, stream_id: str) -> bool:
        return stream_id in self._streams

    async def publish(self, stream_id: str, event: str, payload: dict) -> None:
        state = self._streams.get(stream_id)
        if state is None or state.closed:
            return
        await state.queue.put((event, payload))

    async def close(self, stream_id: str) -> None:
        state = self._streams.get(stream_id)
        if state is None:
            return
        state.closed = True
        await state.queue.put(None)

    async def stream(self, stream_id: str):
        state = self._streams.get(stream_id)
        if state is None:
            yield format_sse("error", {"message": "stream_not_found"})
            return

        while True:
            item = await state.queue.get()
            if item is None:
                break
            event, payload = item
            yield format_sse(event, payload)

        self._streams.pop(stream_id, None)


sse_manager = InMemorySSEManager()
