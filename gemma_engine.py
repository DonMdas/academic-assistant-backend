# gemma_engine.py

import os
import re
import json
from dotenv import load_dotenv
import google.generativeai as genai

# Load variables from .env file
load_dotenv()

# Get API key from environment
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# Configure Gemini
genai.configure(api_key=api_key)

model = genai.GenerativeModel(
    "gemma-4-26b-a4b-it",
    system_instruction="You are a strict study assistant. Output ONLY the final answer. No reasoning, no thought process, no drafting, no preamble, no constraints list, no metadata. Just the direct answer.")


def _strip_code_fences(text):
    value = str(text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json|JSON)?\\s*", "", value)
        value = re.sub(r"\\s*```$", "", value)
    return value.strip()


def _extract_json_answer(raw_text):
    text = _strip_code_fences(raw_text)
    candidates = [text]

    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        candidates.insert(0, json_match.group(0))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except Exception:
            continue

        if isinstance(data, dict):
            for key in ("answer", "final_answer", "response"):
                value = data.get(key)
                if value is None:
                    continue
                parsed = str(value).strip()
                if parsed:
                    return parsed

    return None


def _clean_qa_text(raw_text):
    blocked_prefixes = (
        "strict study assistant",
        "output only",
        "no reasoning",
        "no thought process",
        "no drafting",
        "no preamble",
        "no constraints",
        "no metadata",
        "answer (nothing else)",
    )

    lines = [line.strip() for line in str(raw_text or "").splitlines() if line.strip()]
    cleaned = []
    for line in lines:
        normalized = re.sub(r"^[*\\-•]+\\s*", "", line).strip().lower()
        if any(normalized.startswith(prefix) for prefix in blocked_prefixes):
            continue
        cleaned.append(line)

    if not cleaned:
        return ""

    non_bullet = [line for line in cleaned if not re.match(r"^[*\\-•]+\\s+", line)]
    if non_bullet:
        return non_bullet[-1].strip()

    return " ".join(re.sub(r"^[*\\-•]+\\s*", "", line).strip() for line in cleaned).strip()


def generate_answer(query, chunks, mode="qa"):
    context = "\n\n".join([
        f"Topic: {c['metadata'].get('topic', '')}\n"
        f"Summary: {c['metadata'].get('summary', '')}\n"
        f"Content: {c['text']}"
        for c in chunks
    ])

    if mode == "qa":
        prompt = f"""
Context:
{context}

Question:
{query}

Return JSON only:
{{
    "answer": "the direct final answer"
}}

Rules:
- Do not include chain-of-thought.
- Do not include bullets unless the answer itself requires a list.
- Do not include any extra keys, explanations, or metadata.
"""

    elif mode == "plan":
        prompt = f"""
You are a study planner.

Student request:
{query}

Available material:
{context}

Create a structured study plan.
Include:
- order of topics
- time allocation
- difficulty progression
"""

    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.1,
            "max_output_tokens": 600,
        },
    )
    raw_text = str(getattr(response, "text", "") or "").strip()

    if mode == "qa":
        parsed_answer = _extract_json_answer(raw_text)
        if parsed_answer:
            return parsed_answer
        cleaned_fallback = _clean_qa_text(raw_text)
        if cleaned_fallback:
            return cleaned_fallback

    return raw_text