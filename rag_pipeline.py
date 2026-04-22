# rag_pipeline.py

import uuid

def create_basic_chunks(windows):
    """
    Minimal version (you will replace with Ollama later)
    """
    chunks = []

    for w in windows:
        chunk_id = str(uuid.uuid4())

        chunks.append({
            "chunk_id": chunk_id,
            "text": w["text"],
            "metadata": {
                "pages": w["pages"],
                "type": "raw_window"
            }
        })

    return chunks


def prepare_for_embedding(chunks):
    """
    What will be embedded later
    """
    prepared = []

    for c in chunks:
        embed_text = c["text"][:500]  # placeholder

        prepared.append({
            "chunk_id": c["chunk_id"],
            "embed_text": embed_text,
            "metadata": c["metadata"]
        })

    return prepared