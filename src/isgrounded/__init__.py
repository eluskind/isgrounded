"""isgrounded — ask before you ship an LLM answer: is it grounded?

A runtime verify-or-abstain gate: it credits a claim only when the retrieved evidence actually
supports it (with a citation), and abstains otherwise instead of guessing. Model-agnostic,
retriever-agnostic.
"""
from __future__ import annotations

from .gate import grounded_gate
from .llm import LLM, GeminiLLM, OpenAICompatLLM, llm_from_env
from .retriever import Retriever, SentenceTransformerRetriever, TfidfRetriever
from .types import Citation, GroundingResult, Verdict
from .verifier import GroundedVerifier, is_grounded

__version__ = "0.1.0"
__all__ = [
    "is_grounded",
    "GroundedVerifier",
    "grounded_gate",
    "GroundingResult",
    "Citation",
    "Verdict",
    "Retriever",
    "TfidfRetriever",
    "SentenceTransformerRetriever",
    "LLM",
    "OpenAICompatLLM",
    "GeminiLLM",
    "llm_from_env",
]
