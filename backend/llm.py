import re


def _safe_import_generate_answer():
    try:
        from gemma_engine import generate_answer  # type: ignore

        return generate_answer
    except Exception:
        return None


def _safe_import_model():
    try:
        import gemma_engine  # type: ignore

        return getattr(gemma_engine, "model", None)
    except Exception:
        return None


def _chunk_text_preview(text: str, limit: int = 220) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "..."


def generate_answer_with_chunks(query: str, chunks: list[dict], mode: str = "qa") -> str:
    generator = _safe_import_generate_answer()
    if generator is not None:
        try:
            return str(generator(query, chunks, mode=mode)).strip()
        except Exception:
            pass

    if not chunks:
        return "I do not have enough indexed material for this question yet."

    preview_lines = []
    for idx, chunk in enumerate(chunks[:3], 1):
        topic = str(chunk.get("metadata", {}).get("topic", "General")).strip() or "General"
        text = _chunk_text_preview(chunk.get("text", ""))
        preview_lines.append(f"{idx}. [{topic}] {text}")

    return (
        "Model generation is unavailable right now. Here are the most relevant extracted points:\n"
        + "\n".join(preview_lines)
    )


def generate_text(prompt: str, max_tokens: int = 1024, temperature: float = 0.2) -> str:
    model = _safe_import_model()
    if model is not None:
        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": float(temperature),
                    "max_output_tokens": int(max_tokens),
                },
            )
            text = str(getattr(response, "text", "") or "").strip()
            if text:
                return text
        except Exception:
            pass

    fallback = _chunk_text_preview(prompt, limit=800)
    return (
        "Briefing model is currently unavailable. Use the following session material summary as a guide:\n"
        f"{fallback}"
    )
