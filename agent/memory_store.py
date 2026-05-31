"""
memory_store.py
===============
Long-term vector memory for the fraud agent.

Uses FAISS + sentence-transformers to store and retrieve known fraud
patterns via semantic similarity search. No external API is required —
the embedding model runs locally.
"""

from __future__ import annotations

_EMBED_MODEL_NAME = "all-mpnet-base-v2"


class FraudVectorMemory:
    """
    Semantic memory over a corpus of known fraud patterns.

    Usage::

        memory = FraudVectorMemory()
        memory.build_index(KNOWN_FRAUD_PATTERNS)
        results = memory.search("multiple small transactions in quick succession", k=3)
    """

    def __init__(self) -> None:
        self._model = None   # lazy-loaded SentenceTransformer
        self._index = None   # FAISS IndexFlatIP
        self._patterns: list[dict] = []
        self._dimension: int = 0

    # ── Lazy model loader ──────────────────────────────────────────────────────
    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            self._model = SentenceTransformer(_EMBED_MODEL_NAME)
        return self._model

    # ── Index construction ────────────────────────────────────────────────────
    def build_index(self, patterns: list[dict]) -> None:
        """Encode all fraud patterns and build a FAISS inner-product index."""
        import faiss  # noqa: PLC0415

        self._patterns = list(patterns)
        texts = [self._pattern_to_text(p) for p in patterns]

        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(embeddings)

        self._dimension = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(self._dimension)
        self._index.add(embeddings)

    # ── Search ────────────────────────────────────────────────────────────────
    def search(self, query: str, k: int = 3) -> list[dict]:
        """
        Return the top-k most similar fraud patterns to the query.

        Each result dict contains all original pattern fields plus
        ``similarity_score`` (0–1, higher = more similar).
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        import faiss  # noqa: PLC0415

        model = self._get_model()
        q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)

        distances, indices = self._index.search(q_emb, min(k, self._index.ntotal))
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                entry = dict(self._patterns[idx])
                entry["similarity_score"] = round(float(dist), 4)
                results.append(entry)
        return results

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _pattern_to_text(pattern: dict) -> str:
        indicators = "; ".join(pattern.get("indicators", []))
        return (
            f"{pattern['name']}: {pattern['description']} "
            f"Indicators: {indicators}. "
            f"Severity: {pattern.get('severity', 'UNKNOWN')}."
        )
