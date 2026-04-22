# ollama_parser.py

import ollama
import json
import time
import os
import re
from collections.abc import Mapping
from config import OLLAMA_BASE_URL, DISABLE_OLLAMA_THINKING, QWEN_MODEL

MODEL = QWEN_MODEL


def _env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return bool(default)

    lowered = str(value).strip().lower()
    return lowered in {"1", "true", "yes", "on"}


# Enabled by default to reduce latency for reasoning-capable Qwen variants.
DISABLE_QWEN_THINKING = _env_flag("DISABLE_QWEN_THINKING", True)

BOUNDARY_PROMPT = """
Identify ALL distinct topics in this text.

For each topic, provide:
1) start_anchor: exact short phrase (3-12 words) copied from where the topic starts
2) end_anchor: exact short phrase (3-12 words) copied from where the topic ends
3) start_percent and end_percent as backup values if anchor matching fails

Rules:
- Anchors must be verbatim from the text (no paraphrasing)
- Keep topic segments ordered and non-empty
- Cover the full text as much as possible across segments

Text:
{text}

Return JSON:
{{
 "segments": [
   {{
     "topic": "topic name",
         "start_anchor": "exact phrase from text",
         "end_anchor": "exact phrase from text",
     "start_percent": 0,
     "end_percent": 50,
     "size": "tiny|small|medium|large",
     "summary": "short summary"
   }}
 ]
}}
"""

EXTRACTION_PROMPT = """
Extract structured study info.

Important summary rule:
- The summary must explicitly name the topic(s) covered.
- If a descriptive summary is not possible, still output a short topic-only summary such as
    \"Topics covered: <topic>; <subtopic 1>; <subtopic 2>\".

Text:
{text}

Return JSON:
{{
 "topic": "",
 "subtopics": [],
 "summary": "",
 "key_concepts": [],
 "complexity": "beginner|intermediate|advanced",
 "prerequisites": [],
 "estimated_time": 30
}}
"""


def _build_ollama_client():
    # Newer ollama versions expose Client(host=...), while some older builds may differ.
    if hasattr(ollama, "Client"):
        return ollama.Client(host=OLLAMA_BASE_URL)
    return None


OLLAMA_CLIENT = _build_ollama_client()


def _as_dict(value):
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

    return {}


def _message_to_dict(message):
    if isinstance(message, dict):
        return message

    message_dict = _as_dict(message)
    if isinstance(message_dict, dict) and ("content" in message_dict or "role" in message_dict):
        return message_dict

    content = getattr(message, "content", None)
    role = getattr(message, "role", None)
    images = getattr(message, "images", None)
    tool_calls = getattr(message, "tool_calls", None)

    if content is None and role is None and images is None and tool_calls is None:
        return {}

    result = {}
    if role is not None:
        result["role"] = role
    if content is not None:
        result["content"] = str(content)
    if images is not None:
        result["images"] = images
    if tool_calls is not None:
        result["tool_calls"] = tool_calls
    return result


def _normalize_chat_response(response):
    payload = _as_dict(response)

    if not payload and response is not None:
        # Fallback for SDK response objects that may not serialize cleanly.
        for key in (
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
            if hasattr(response, key):
                payload[key] = getattr(response, key)
        if hasattr(response, "message"):
            payload["message"] = getattr(response, "message")

    message = _message_to_dict(payload.get("message"))
    if message:
        payload["message"] = message
    elif "content" in payload:
        payload["message"] = {
            "content": str(payload.get("content", "") or ""),
        }

    return payload


def _is_qwen_model(model_name):
    return "qwen" in str(model_name or "").lower()


def _prepend_no_think_message(messages):
    cleaned = list(messages or [])
    already_has_no_think = any(
        isinstance(message, dict) and "/no_think" in str(message.get("content", ""))
        for message in cleaned
    )
    if already_has_no_think:
        return cleaned

    return [{"role": "system", "content": "/no_think"}] + cleaned


def _chat_once(messages, output_format=None):
    chat_kwargs = {
        "model": MODEL,
        "messages": list(messages or []),
    }
    if output_format:
        chat_kwargs["format"] = output_format

    should_disable_thinking = bool(DISABLE_OLLAMA_THINKING) or (
        DISABLE_QWEN_THINKING and _is_qwen_model(MODEL)
    )
    if should_disable_thinking:
        # Prefer native flag when available in the installed Ollama client.
        chat_kwargs["think"] = False

    try:
        if OLLAMA_CLIENT is not None:
            response = OLLAMA_CLIENT.chat(**chat_kwargs)
        else:
            response = ollama.chat(**chat_kwargs)
        return _normalize_chat_response(response)
    except TypeError:
        # Older client builds may not support think=False; fall back to prompt-level switch.
        chat_kwargs.pop("think", None)
        if should_disable_thinking:
            chat_kwargs["messages"] = _prepend_no_think_message(chat_kwargs.get("messages", []))

        if OLLAMA_CLIENT is not None:
            response = OLLAMA_CLIENT.chat(**chat_kwargs)
        else:
            response = ollama.chat(**chat_kwargs)
        return _normalize_chat_response(response)


def chat_ollama(messages, output_format=None, retries=3):
    for attempt in range(retries):
        try:
            return _chat_once(messages=messages, output_format=output_format)
        except Exception as e:
            print(f"⚠️ Retry {attempt+1}: {e}")
            time.sleep(1)

    return {}


def _parse_json_content(content):
    text = str(content or "").strip()
    if not text:
        return {}

    # Handle fenced JSON replies.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|JSON)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    # Fallback: parse first JSON object embedded in additional text.
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}

    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def safe_ollama_call(prompt, retries=3):
    response = chat_ollama(
        messages=[{"role": "user", "content": prompt}],
        output_format="json",
        retries=retries,
    )

    if not isinstance(response, dict):
        return {}

    content = str(response.get("message", {}).get("content", "") or "").strip()
    if not content:
        return {}

    return _parse_json_content(content)


def detect_segments(window_text):
    prompt = BOUNDARY_PROMPT.format(text=window_text)
    return safe_ollama_call(prompt)


def extract_metadata(text):
    prompt = EXTRACTION_PROMPT.format(text=text)
    return safe_ollama_call(prompt)