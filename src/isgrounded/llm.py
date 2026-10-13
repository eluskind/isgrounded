"""Provider-agnostic LLM layer.

isgrounded is a wrapper, not a model. It talks to whatever LLM you point it at through a
tiny `complete()` interface. Bring your own: any OpenAI-compatible endpoint (OpenAI, Ollama,
OpenRouter, vLLM, ...) or Google Gemini. Add a provider by implementing `LLM`.
"""
from __future__ import annotations

import os
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLM(Protocol):
    def complete(self, system: str, user: str, *, temperature: float = 0.0,
                 seed: int | None = None) -> str:
        ...


class OpenAICompatLLM:
    """Any OpenAI-compatible chat endpoint. Point `base_url` at OpenAI (default), a local
    Ollama/vLLM server, OpenRouter, etc. Requires the `openai` package (extra: isgrounded[openai])."""

    def __init__(self, model: str = "gpt-4o-mini", *, base_url: str | None = None,
                 api_key: str | None = None, api_key_env: str = "OPENAI_API_KEY"):
        from openai import OpenAI  # lazy import so the core has no hard dependency
        self.model = model
        self._client = OpenAI(base_url=base_url,
                              api_key=api_key or os.environ.get(api_key_env, "not-needed"))

    def complete(self, system: str, user: str, *, temperature: float = 0.0,
                 seed: int | None = None) -> str:
        kwargs: dict = {}
        if seed is not None:
            kwargs["seed"] = seed
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            **kwargs,
        )
        return resp.choices[0].message.content or ""


class GeminiLLM:
    """Google Gemini via the google-genai SDK (extra: isgrounded[gemini])."""

    def __init__(self, model: str = "gemini-2.5-flash", *, api_key: str | None = None,
                 api_key_env: str = "GEMINI_API_KEY"):
        from google import genai  # lazy
        self.model = model
        self._client = genai.Client(api_key=api_key or os.environ.get(api_key_env))

    def complete(self, system: str, user: str, *, temperature: float = 0.0,
                 seed: int | None = None) -> str:
        from google.genai import types as gt
        cfg_kwargs: dict = {"system_instruction": system, "temperature": temperature}
        if seed is not None:
            cfg_kwargs["seed"] = seed
        try:
            cfg = gt.GenerateContentConfig(**cfg_kwargs)
        except TypeError:
            cfg_kwargs.pop("seed", None)  # older SDKs may not accept seed
            cfg = gt.GenerateContentConfig(**cfg_kwargs)
        resp = self._client.models.generate_content(model=self.model, contents=user, config=cfg)
        return resp.text or ""


def llm_from_env() -> LLM:
    """Build an LLM from environment variables:

        ISGROUNDED_LLM      openai (default) | ollama | openrouter | gemini
        ISGROUNDED_MODEL    model name (provider-specific default otherwise)
        ISGROUNDED_BASE_URL override the OpenAI-compatible base URL
    """
    provider = os.environ.get("ISGROUNDED_LLM", "openai").lower()
    model = os.environ.get("ISGROUNDED_MODEL")
    if provider in ("openai", "openai-compat", "ollama", "openrouter", "vllm"):
        base = os.environ.get("ISGROUNDED_BASE_URL")
        if provider == "ollama" and not base:
            base = "http://localhost:11434/v1"
        return OpenAICompatLLM(model or "gpt-4o-mini", base_url=base)
    if provider in ("gemini", "google"):
        return GeminiLLM(model or "gemini-2.5-flash")
    raise ValueError(f"Unknown ISGROUNDED_LLM={provider!r} "
                     "(expected one of: openai, ollama, openrouter, vllm, gemini)")
