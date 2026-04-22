import ollama
import os

from config import OLLAMA_BASE_URL, QWEN_MODEL


MODEL = QWEN_MODEL


def _env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


DISABLE_QWEN_THINKING = _env_flag("DISABLE_QWEN_THINKING", True)


def _is_qwen_model(model_name):
    return "qwen" in str(model_name or "").lower()


def _prepend_no_think_message(messages):
    cleaned = list(messages or [])
    has_no_think = any(
        isinstance(message, dict) and "/no_think" in str(message.get("content", ""))
        for message in cleaned
    )
    if has_no_think:
        return cleaned
    return [{"role": "system", "content": "/no_think"}] + cleaned


def _build_ollama_client():
    if hasattr(ollama, "Client"):
        return ollama.Client(host=OLLAMA_BASE_URL)
    return None


def _chat(client, messages):
    chat_kwargs = {"model": MODEL, "messages": list(messages or [])}
    should_disable_thinking = DISABLE_QWEN_THINKING and _is_qwen_model(MODEL)

    if should_disable_thinking:
        chat_kwargs["think"] = False

    try:
        if client is not None:
            return client.chat(**chat_kwargs)
        return ollama.chat(**chat_kwargs)
    except TypeError:
        chat_kwargs.pop("think", None)
        if should_disable_thinking:
            chat_kwargs["messages"] = _prepend_no_think_message(chat_kwargs.get("messages", []))

        if client is not None:
            return client.chat(**chat_kwargs)
        return ollama.chat(**chat_kwargs)


def main():
    client = _build_ollama_client()
    messages = []

    print(f"Connected to Ollama at: {OLLAMA_BASE_URL}")
    print(f"Using model: {MODEL}")
    print("Type your question and press Enter. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Exiting chat.")
            break

        messages.append({"role": "user", "content": user_input})

        try:
            response = _chat(client, messages)
            assistant_text = response.get("message", {}).get("content", "")
            messages.append({"role": "assistant", "content": assistant_text})
            print(f"Assistant: {assistant_text}\n")
        except Exception as exc:
            print(f"Error while calling Ollama: {exc}\n")


if __name__ == "__main__":
    main()
