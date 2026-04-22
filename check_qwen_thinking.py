import argparse
import statistics
import time
from collections.abc import Mapping
from typing import Any

import ollama_parser as op


DEFAULT_PROMPT = (
    "Give a one-sentence definition of edge detection. "
    "Answer directly without extra formatting."
)


def _preview(text: str, limit: int = 180) -> str:
    value = str(text or "").replace("\n", " ").strip()
    if len(value) <= limit:
        return value
    return value[:limit] + "..."


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value

    if isinstance(value, Mapping):
        try:
            return dict(value)
        except Exception:
            return {}

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass

    dict_method = getattr(value, "dict", None)
    if callable(dict_method):
        try:
            dumped = dict_method()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass

    extracted = {}
    for key in (
        "message",
        "model",
        "created_at",
        "done",
        "done_reason",
        "error",
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "eval_count",
    ):
        if hasattr(value, key):
            extracted[key] = getattr(value, key)

    return extracted


def _safe_message(response: Any) -> dict:
    payload = _as_dict(response)
    if not payload:
        return {}

    message = payload.get("message", {})
    message = _as_dict(message)
    if not message:
        # Some client shapes may flatten content at top level.
        content = payload.get("content")
        if isinstance(content, str):
            return {"content": content}
        return {}

    return message


def _response_error_reason(response: Any) -> str:
    payload = _as_dict(response)
    if not payload:
        return "empty_or_unrecognized_response"

    if payload.get("error"):
        return f"error_field_present: {payload.get('error')}"

    message = _safe_message(payload)
    if not message:
        return "missing_message"

    content = str(message.get("content", "") or "").strip()
    if not content:
        return "empty_message_content"

    return "ok"


def _has_reasoning_markers(message: dict) -> tuple[bool, list[str]]:
    content = str(message.get("content", "") or "")
    lowered = content.lower()

    markers = []
    if "<think>" in lowered or "</think>" in lowered:
        markers.append("content_has_think_tags")

    for key in ("thinking", "reasoning", "thoughts"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            markers.append(f"message.{key}_present")

    return bool(markers), markers


def _run_probe_case(prompt: str, disable_thinking: bool, runs: int, retries: int) -> dict:
    op.DISABLE_QWEN_THINKING = disable_thinking

    elapsed_values = []
    responses = []
    errors = 0
    error_reasons = []

    for _ in range(runs):
        start = time.perf_counter()
        response = op.chat_ollama(
            messages=[{"role": "user", "content": prompt}],
            retries=retries,
        )
        elapsed = time.perf_counter() - start

        elapsed_values.append(elapsed)
        responses.append(response)
        reason = _response_error_reason(response)
        if reason != "ok":
            errors += 1
            error_reasons.append(reason)

    last_message = _safe_message(responses[-1] if responses else {})
    has_reasoning, markers = _has_reasoning_markers(last_message)
    last_payload = _as_dict(responses[-1]) if responses else {}

    return {
        "disable_thinking": disable_thinking,
        "runs": runs,
        "errors": errors,
        "error_reasons": sorted(set(error_reasons)),
        "avg_seconds": statistics.mean(elapsed_values) if elapsed_values else None,
        "min_seconds": min(elapsed_values) if elapsed_values else None,
        "max_seconds": max(elapsed_values) if elapsed_values else None,
        "has_reasoning_markers": has_reasoning,
        "reasoning_markers": markers,
        "content_preview": _preview(last_message.get("content", "")),
        "response_type": type(responses[-1]).__name__ if responses else "None",
        "response_keys": sorted(last_payload.keys()),
        "message_keys": sorted(last_message.keys()),
    }


def _print_case(label: str, result: dict) -> None:
    print("\n" + "=" * 72)
    print(label)
    print("-" * 72)
    print(f"disable_thinking: {result['disable_thinking']}")
    print(f"runs: {result['runs']} | errors: {result['errors']}")
    print(
        "latency (s): "
        f"avg={result['avg_seconds']:.3f} "
        f"min={result['min_seconds']:.3f} "
        f"max={result['max_seconds']:.3f}"
        if result["avg_seconds"] is not None
        else "latency (s): n/a"
    )
    print(f"response type: {result['response_type']}")
    print(f"reasoning markers: {result['has_reasoning_markers']} -> {result['reasoning_markers']}")
    print(f"error reasons: {result['error_reasons']}")
    print(f"response keys: {result['response_keys']}")
    print(f"message keys: {result['message_keys']}")
    print(f"content preview: {result['content_preview']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check Qwen thinking on/off behavior through the project wrapper.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt used for probing")
    parser.add_argument("--runs", type=int, default=2, help="Number of runs per case")
    parser.add_argument("--retries", type=int, default=1, help="Retries per run")
    parser.add_argument("--model", default="", help="Optional model override (e.g., qwen3.5:9b)")
    parser.add_argument(
        "--mode",
        choices=["compare", "off", "on"],
        default="compare",
        help="compare: test both; off: thinking enabled; on: thinking disabled",
    )
    args = parser.parse_args()

    if args.runs < 1:
        raise ValueError("--runs must be >= 1")

    if args.model:
        op.MODEL = args.model

    print("Qwen Thinking Behavior Check")
    print("=" * 72)
    print(f"model: {op.MODEL}")
    print(f"ollama_base_url: {op.OLLAMA_BASE_URL}")
    print(f"prompt: {_preview(args.prompt, 120)}")

    if args.mode in {"compare", "on"}:
        result_disabled = _run_probe_case(
            prompt=args.prompt,
            disable_thinking=True,
            runs=args.runs,
            retries=args.retries,
        )
        _print_case("Case A: Thinking Disabled", result_disabled)
    else:
        result_disabled = None

    if args.mode in {"compare", "off"}:
        result_enabled = _run_probe_case(
            prompt=args.prompt,
            disable_thinking=False,
            runs=args.runs,
            retries=args.retries,
        )
        _print_case("Case B: Thinking Enabled", result_enabled)
    else:
        result_enabled = None

    if result_disabled and result_enabled:
        delta = result_enabled["avg_seconds"] - result_disabled["avg_seconds"]
        print("\n" + "=" * 72)
        print("Comparison")
        print("-" * 72)
        if delta > 0:
            print(f"Thinking disabled is faster by {delta:.3f}s on average.")
        elif delta < 0:
            print(f"Thinking enabled is faster by {abs(delta):.3f}s on average.")
        else:
            print("Average latency is the same in this sample.")

        print(
            "Reasoning marker difference: "
            f"disabled={result_disabled['has_reasoning_markers']} | "
            f"enabled={result_enabled['has_reasoning_markers']}"
        )


if __name__ == "__main__":
    main()
