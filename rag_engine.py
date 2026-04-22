# rag_engine.py

import faiss
import numpy as np
import pickle
import re
from sentence_transformers import SentenceTransformer

from ollama_parser import safe_ollama_call

model = SentenceTransformer("all-MiniLM-L6-v2")

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "to", "was", "were",
    "what", "when", "where", "which", "who", "why", "will", "with", "you", "your",
}


def _tokenize(text):
    tokens = re.findall(r"[a-z0-9]{2,}", str(text or "").lower())
    return [token for token in tokens if token not in _STOPWORDS]


def _keyword_overlap_score(query_tokens, text):
    if not query_tokens:
        return 0.0

    doc_tokens = set(_tokenize(text))
    if not doc_tokens:
        return 0.0

    matched = sum(1 for token in query_tokens if token in doc_tokens)
    return float(matched / max(1, len(query_tokens)))


def _clamp(value, low, high):
    return max(low, min(high, value))


class RAGEngine:
    def __init__(self, index_path=None):
        self.index = faiss.IndexFlatL2(384)
        self.texts = []
        self.metadata = []

        if index_path:
            self.load(index_path)

    def add_chunks(self, chunks):
        if not chunks:
            return

        embeddings = model.encode([c["text"] for c in chunks])
        self.index.add(np.array(embeddings))

        self.texts.extend([c["text"] for c in chunks])
        self.metadata.extend([c["metadata"] for c in chunks])

    def save(self, path):
        faiss.write_index(self.index, f"{path}.index")

        with open(f"{path}.pkl", "wb") as f:
            pickle.dump({
                "texts": self.texts,
                "metadata": self.metadata
            }, f)

    def load(self, path):
        self.index = faiss.read_index(f"{path}.index")

        with open(f"{path}.pkl", "rb") as f:
            data = pickle.load(f)

        self.texts = data["texts"]
        self.metadata = data["metadata"]

    def _passes_filters(self, metadata, filters):
        if not filters:
            return True

        if "complexity" in filters:
            if metadata.get("complexity") != filters["complexity"]:
                return False

        if "max_time" in filters:
            if metadata.get("estimated_time", 0) > filters["max_time"]:
                return False

        return True

    def _qwen_relevance_score(self, query, candidate):
        metadata = candidate.get("metadata", {}) if isinstance(candidate, dict) else {}
        topic = str(metadata.get("topic", "")).strip()
        summary = str(metadata.get("summary", "")).strip()
        text = str(candidate.get("text", "")).strip()

        prompt = f"""
You are reranking retrieval candidates for question answering.

Question:
{query}

Candidate topic:
{topic or 'N/A'}

Candidate summary:
{summary or 'N/A'}

Candidate content:
{text}

Return JSON only:
{{
  "relevance": 0
}}

Rules:
- relevance must be an integer from 0 to 100.
- Higher means more relevant for answering the question.
"""

        try:
            response = safe_ollama_call(prompt)
        except Exception:
            return None

        if not isinstance(response, dict):
            return None

        raw = response.get("relevance")
        if raw is None:
            raw = response.get("score")
        if raw is None:
            return None

        try:
            score = float(raw)
        except (TypeError, ValueError):
            return None

        score = _clamp(score, 0.0, 100.0)
        return float(score / 100.0)

    def retrieve_hybrid(
        self,
        query,
        top_k=5,
        filters=None,
        candidate_multiplier=4,
        semantic_weight=0.78,
        keyword_weight=0.22,
        rerank_with_qwen=False,
        rerank_top_n=6,
        qwen_weight=0.25,
    ):
        if not self.texts:
            return []

        query = str(query or "").strip()
        if not query:
            return []

        semantic_weight = float(_clamp(float(semantic_weight), 0.0, 1.0))
        keyword_weight = float(_clamp(float(keyword_weight), 0.0, 1.0))
        weight_total = semantic_weight + keyword_weight
        if weight_total <= 0.0:
            semantic_weight, keyword_weight = 0.78, 0.22
            weight_total = 1.0
        semantic_weight /= weight_total
        keyword_weight /= weight_total

        q_emb = model.encode([query])
        candidate_count = min(
            len(self.texts),
            max(int(top_k) * max(2, int(candidate_multiplier)), int(top_k) + 4),
        )
        D, I = self.index.search(np.array(q_emb), candidate_count)

        query_tokens = sorted(set(_tokenize(query)))
        results = []

        for rank, idx in enumerate(I[0]):
            if idx < 0 or idx >= len(self.texts):
                continue

            metadata = self.metadata[idx]
            if not self._passes_filters(metadata, filters):
                continue

            text = self.texts[idx]
            distance = float(D[0][rank])
            semantic_score = 1.0 / (1.0 + max(distance, 0.0))

            topic = str(metadata.get("topic", ""))
            summary = str(metadata.get("summary", ""))
            text_kw_score = _keyword_overlap_score(query_tokens, text)
            topic_kw_score = _keyword_overlap_score(query_tokens, topic)
            summary_kw_score = _keyword_overlap_score(query_tokens, summary)
            keyword_score = (
                0.55 * text_kw_score
                + 0.30 * topic_kw_score
                + 0.15 * summary_kw_score
            )

            combined_score = semantic_weight * semantic_score + keyword_weight * keyword_score

            results.append({
                "text": text,
                "metadata": metadata,
                "retrieval": {
                    "rank": rank + 1,
                    "distance": distance,
                    "score": combined_score,
                    "semantic_score": semantic_score,
                    "keyword_score": keyword_score,
                    "keyword_text_score": text_kw_score,
                    "keyword_topic_score": topic_kw_score,
                    "keyword_summary_score": summary_kw_score,
                },
            })

        if not results:
            return []

        results.sort(key=lambda item: item.get("retrieval", {}).get("score", 0.0), reverse=True)

        if rerank_with_qwen:
            rerank_top_n = max(1, min(int(rerank_top_n), len(results)))
            qwen_weight = float(_clamp(float(qwen_weight), 0.0, 1.0))
            target_top_k = max(1, int(top_k))

            def _apply_qwen(candidate):
                retrieval = candidate.get("retrieval", {})
                base_score = float(retrieval.get("score", 0.0))
                qwen_score = self._qwen_relevance_score(query, candidate)
                if qwen_score is None:
                    qwen_score = base_score
                    retrieval["qwen_fallback"] = "base_score"
                else:
                    retrieval.pop("qwen_fallback", None)

                retrieval["qwen_score"] = qwen_score
                retrieval["score"] = (1.0 - qwen_weight) * base_score + qwen_weight * qwen_score

            for candidate in results[:rerank_top_n]:
                _apply_qwen(candidate)

            results.sort(key=lambda item: item.get("retrieval", {}).get("score", 0.0), reverse=True)

            # Rerank any new candidates that rise into the final top-k after resorting.
            for _ in range(3):
                pending = [
                    candidate
                    for candidate in results[:target_top_k]
                    if "qwen_score" not in candidate.get("retrieval", {})
                ]
                if not pending:
                    break
                for candidate in pending:
                    _apply_qwen(candidate)
                results.sort(key=lambda item: item.get("retrieval", {}).get("score", 0.0), reverse=True)

        final_results = results[: max(1, int(top_k))]
        if rerank_with_qwen:
            for item in final_results:
                retrieval = item.get("retrieval", {})
                if "qwen_score" in retrieval:
                    continue
                base_score = float(retrieval.get("score", 0.0))
                retrieval["qwen_score"] = base_score
                retrieval["qwen_fallback"] = "not_reranked"

        for rank, item in enumerate(final_results, 1):
            item["retrieval"]["rank"] = rank

        return final_results

    def retrieve(self, query, top_k=5, filters=None):
        return self.retrieve_hybrid(query, top_k=top_k, filters=filters)

    def retrieve_with_time_budget(self, query, max_total_time=120):
        results = self.retrieve_hybrid(query, top_k=12)

        selected = []
        total_time = 0

        for r in results:
            t = r["metadata"].get("estimated_time", 30)

            if total_time + t > max_total_time:
                break

            selected.append(r)
            total_time += t

        return selected