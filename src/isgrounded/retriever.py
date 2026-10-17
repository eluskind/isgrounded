"""Retriever interface + a torch-free default.

Re-grounding (verifier.py) needs a way to re-search your corpus by the claim text. Any object
with `.search(query, k) -> list[Citation]` works, wrap your LangChain / LlamaIndex / vector-DB
retriever, or use the bundled `TfidfRetriever` for a zero-setup lexical baseline.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import Citation


@runtime_checkable
class Retriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[Citation]:
        ...


class TfidfRetriever:
    """Lexical TF-IDF retriever (needs scikit-learn; extra: isgrounded[tfidf]).

    No embeddings, no torch, no GPU — a good default for examples and tests. For production
    semantic retrieval, pass your own embedding-based retriever instead.
    """

    def __init__(self, passages: list[tuple[str, str]]):
        """passages: list of (source_id, text)."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        if not passages:
            raise ValueError("TfidfRetriever needs at least one passage")
        self._ids = [p[0] for p in passages]
        self._texts = [p[1] for p in passages]
        self._vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._matrix = self._vec.fit_transform(self._texts)

    def search(self, query: str, k: int = 5) -> list[Citation]:
        import numpy as np
        qv = self._vec.transform([query])
        sims = (self._matrix @ qv.T).toarray().ravel()
        order = np.argsort(-sims)[:k]
        return [Citation(source_id=self._ids[i], text=self._texts[i], score=float(sims[i]))
                for i in order]


class SentenceTransformerRetriever:
    """Convenience semantic retriever using sentence-transformers embeddings (extra: isgrounded[embed]).

    A zero-wiring semantic option for examples and the benchmarks: ranks by meaning, not word
    overlap, so it recovers paraphrased support a lexical retriever misses. Needs
    `sentence-transformers` (and torch). For production, bring your own embedding-based retriever
    (any object with `.search(query, k) -> list[Citation]`).
    """

    def __init__(self, passages: list[tuple[str, str]],
                 model: str = "sentence-transformers/all-mpnet-base-v2"):
        """passages: list of (source_id, text)."""
        from sentence_transformers import SentenceTransformer
        import numpy as np
        if not passages:
            raise ValueError("SentenceTransformerRetriever needs at least one passage")
        self._ids = [p[0] for p in passages]
        self._texts = [p[1] for p in passages]
        self._model = SentenceTransformer(model)
        emb = self._model.encode(self._texts, normalize_embeddings=True, show_progress_bar=False)
        self._emb = np.asarray(emb, dtype="float32")

    def search(self, query: str, k: int = 5) -> list[Citation]:
        import numpy as np
        q = np.asarray(
            self._model.encode(query, normalize_embeddings=True, show_progress_bar=False),
            dtype="float32",
        )
        sims = (self._emb @ q).ravel()
        order = np.argsort(-sims)[:k]
        return [Citation(source_id=self._ids[i], text=self._texts[i], score=float(sims[i]))
                for i in order]
