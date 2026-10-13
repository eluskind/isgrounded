"""The grounding gate: ask the judge whether evidence supports a claim.

Single-judge by default. `strict=True` runs a multi-judge majority (two-then-tiebreak): a
panel of 3 fixed seeds, decided as soon as the first two agree, so a unanimous panel costs 2
calls and only a 1-1 split spends the third. The decision boundary is identical to a full
majority-of-3; it just skips the redundant call.

Default is single-judge on purpose: on a hard public benchmark the panel changed the verdict
on only ~3% of cases and barely moved accuracy, so it isn't worth being the default. Keep it
as an opt-in for high-stakes paths.

Failure policy: fail CLOSED. If the judge errors or returns something unparseable, treat the
claim as NOT grounded (abstain). For a verify-or-abstain tool, "when in doubt, abstain" is the
safe default.
"""
from __future__ import annotations

import json
import re

from .llm import LLM
from .prompts import GROUNDING_SYSTEM, GROUNDING_USER

DEFAULT_SEEDS: tuple[int, ...] = (7, 17, 29)
DEFAULT_TEMPERATURE = 0.4

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse(out: str) -> tuple[bool, str] | None:
    m = _JSON_RE.search(out or "")
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    if "grounded" not in d:
        return None
    return bool(d["grounded"]), str(d.get("reason", "")).strip()


def _judge_once(llm: LLM, claim: str, evidence: str, *, seed: int,
                temperature: float) -> tuple[bool, str]:
    user = GROUNDING_USER.format(claim=claim, evidence=evidence)
    try:
        out = llm.complete(GROUNDING_SYSTEM, user, temperature=temperature, seed=seed)
    except Exception as e:  # noqa: BLE001 — fail closed: abstain rather than guess
        return False, f"judge error, abstaining: {e!r}"
    parsed = _parse(out)
    if parsed is None:
        return False, "unparseable judge output, abstaining"
    return parsed


def grounded_gate(llm: LLM, claim: str, evidence: str, *, strict: bool = False,
                  seeds: tuple[int, ...] = DEFAULT_SEEDS,
                  temperature: float = DEFAULT_TEMPERATURE) -> tuple[bool, str]:
    """Return (grounded, reason). Single-judge unless strict."""
    if not strict:
        return _judge_once(llm, claim, evidence, seed=seeds[0], temperature=temperature)

    v0 = _judge_once(llm, claim, evidence, seed=seeds[0], temperature=temperature)
    v1 = _judge_once(llm, claim, evidence, seed=seeds[1], temperature=temperature)
    votes = [v0, v1]
    if v0[0] != v1[0]:  # 1-1 split -> the third seed breaks the tie
        votes.append(_judge_once(llm, claim, evidence, seed=seeds[2], temperature=temperature))

    grounded_reasons = [r for g, r in votes if g]
    not_reasons = [r for g, r in votes if not g]
    cast = len(votes)
    # grounded only on a strict majority (2-of-2 short-circuit, else 2-of-3)
    if len(grounded_reasons) > cast // 2 or (cast == 2 and len(grounded_reasons) == 2):
        return True, grounded_reasons[0]
    return False, (not_reasons[0] if not_reasons else "majority not grounded")
