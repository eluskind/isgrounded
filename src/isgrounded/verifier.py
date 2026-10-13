"""GroundedVerifier: the runtime verify-or-abstain decision, with optional re-grounding.

- `verify(claim, evidence)` -> grounded / ungrounded.
- `verify(claim, evidence, reground=True)` with a retriever -> on "ungrounded", re-search the
  corpus by the CLAIM (not the original query), exclude already-cited passages, and re-verify
  with the SAME strict gate. This rescues a MIS-CITATION (support is in the corpus, wrong chunk
  was cited) without rescuing a CONFABULATION (support is nowhere) — because the bar is unchanged.
"""
from __future__ import annotations

from .gate import grounded_gate
from .llm import LLM, llm_from_env
from .retriever import Retriever
from .types import Citation, GroundingResult

RETRIEVAL_FLOOR = 0.05  # never re-ground onto near-zero-overlap passages


class GroundedVerifier:
    def __init__(self, llm: LLM | None = None, retriever: Retriever | None = None, *,
                 strict: bool = False, retrieval_floor: float = RETRIEVAL_FLOOR,
                 reground_k: int = 5, reground_top: int = 3):
        self.llm = llm or llm_from_env()
        self.retriever = retriever
        self.strict = strict
        self.retrieval_floor = retrieval_floor
        self.reground_k = reground_k
        self.reground_top = reground_top

    def verify(self, claim: str, evidence: str | None = None, *, reground: bool = False,
               source_id: str = "provided") -> GroundingResult:
        """Verify a claim. If `evidence` is None, fetch it from the retriever first."""
        if evidence is None:
            if self.retriever is None:
                raise ValueError("verify() needs `evidence`, or a retriever to fetch it.")
            hits = self.retriever.search(claim, k=self.reground_k)[: self.reground_top]
            cites = hits
            evidence = "\n\n".join(h.text for h in hits)
        else:
            cites = [Citation(source_id=source_id, text=evidence)]

        grounded, reason = grounded_gate(self.llm, claim, evidence, strict=self.strict)
        if grounded:
            return GroundingResult(claim, True, "grounded", reason, cites)

        if reground and self.retriever is not None:
            recovered = self._reground(claim, already=cites)
            if recovered is not None:
                return recovered

        return GroundingResult(claim, False, "ungrounded", reason, [])

    def _reground(self, claim: str, already: list[Citation]) -> GroundingResult | None:
        seen = {c.source_id for c in already}
        hits = [h for h in self.retriever.search(claim, k=self.reground_k)
                if h.source_id not in seen
                and (h.score is None or h.score >= self.retrieval_floor)]
        if not hits:
            return None
        top = hits[: self.reground_top]
        new_evidence = "\n\n".join(h.text for h in top)
        grounded, reason = grounded_gate(self.llm, claim, new_evidence, strict=self.strict)
        if grounded:
            return GroundingResult(claim, True, "recovered", f"re-grounded: {reason}", top)
        return None


def is_grounded(claim: str, evidence: str, *, llm: LLM | None = None,
                strict: bool = False) -> bool:
    """Convenience one-liner: is the claim grounded in the evidence? True / False."""
    grounded, _ = grounded_gate(llm or llm_from_env(), claim, evidence, strict=strict)
    return grounded
