"""Core result types for isgrounded."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Verdict = Literal["grounded", "ungrounded", "recovered"]


@dataclass
class Citation:
    """A passage offered as support for a claim."""
    source_id: str          # where it came from: a doc id, chunk id, URL, heading, etc.
    text: str               # the supporting passage text
    score: float | None = None   # retriever score, if any


@dataclass
class GroundingResult:
    """The verdict for one claim against its evidence."""
    claim: str
    grounded: bool
    verdict: Verdict        # "grounded" | "ungrounded" | "recovered"
    reason: str
    citations: list[Citation] = field(default_factory=list)

    def __bool__(self) -> bool:  # `if verifier.verify(...):`
        return self.grounded
