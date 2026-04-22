import json
import sys
import time
from urllib import error, request

from config import OLLAMA_BASE_URL
from ollama_parser import MODEL


def _print_header(title):
    print(f"\n== {title} ==")


def _safe_json(value):
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        return repr(value)


def _base_url():
    return str(OLLAMA_BASE_URL).rstrip("/")


def _http_json(method, path, payload=None, timeout=15):
    url = f"{_base_url()}{path}"
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, method=method)
    req.add_header("Accept", "application/json")
    if body is not None:
        req.add_header("Content-Type", "application/json")

    with request.urlopen(req, timeout=timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
        return json.loads(raw) if raw.strip() else {}


def _print_http_probe(title, method, path, payload=None, timeout=15):
    _print_header(title)
    start = time.perf_counter()
    try:
        data = _http_json(method=method, path=path, payload=payload, timeout=timeout)
        elapsed = time.perf_counter() - start
        print("status: ok")
        print(f"elapsed_seconds: {elapsed:.2f}")
        print(_safe_json(data))
        return data
    except Exception as exc:
        elapsed = time.perf_counter() - start
        print("status: error")
        print(f"elapsed_seconds: {elapsed:.2f}")
        print(f"exception_type: {type(exc).__name__}")
        print(f"exception: {repr(exc)}")
        return None


def main():
    _print_header("Environment")
    print(f"python: {sys.executable}")
    print(f"ollama_base_url: {_base_url()}")
    print(f"model: {MODEL}")
    print("client_initialized: not used in this probe")

    _print_http_probe("API Version Probe", "GET", "/api/version", timeout=10)
    _print_http_probe("API Tags Probe", "GET", "/api/tags", timeout=10)

    chat_payload = {
        "model": MODEL,
        "stream": False,
        "messages": [
            {"role": "user", "content": "Reply with only the word pong."},
        ],
        "options": {
            "num_predict": 8,
        },
    }
    _print_http_probe("Chat POST", "POST", "/api/chat", payload=chat_payload, timeout=30)

    _print_header("Metadata Probe")
    sample_text = (
        "Convolutional neural networks use kernels and feature maps to detect edges, "
        "textures, and shapes in images."
    )
    start = time.perf_counter()
    try:
        from ollama_parser import extract_metadata

        metadata = extract_metadata(sample_text)
        elapsed = time.perf_counter() - start
        print(f"status: ok")
        print(f"elapsed_seconds: {elapsed:.2f}")
        print(_safe_json(metadata))
    except Exception as exc:
        elapsed = time.perf_counter() - start
        print(f"status: error")
        print(f"elapsed_seconds: {elapsed:.2f}")
        print(f"exception_type: {type(exc).__name__}")
        print(f"exception: {repr(exc)}")


if __name__ == "__main__":
    main()
