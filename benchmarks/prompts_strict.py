"""Experimental "strict-abstention" grounding prompt variant , benchmark use only.

NOT shipped in the package (src/isgrounded/prompts.py is the real prompt). This variant is built
from PRINCIPLE, not fit to the test cases, to probe whether the open-model under-abstention is
prompt-transfer or genuine model capability. Four principled changes vs. the shipped prompt:
  1. Lead with the strict default (NOT GROUNDED), not the permissive GROUNDED path.
  2. Explicit uncertainty tie-break ("if not certain -> not grounded").
  3. Reason BEFORE the verdict in the JSON (analyze, then commit the boolean).
  4. Explicit not-enough-information (NEI) line.
"""
from __future__ import annotations

STRICT_SYSTEM = """You are a strict grounding judge. Your DEFAULT answer is NOT GROUNDED. \
Credit a claim as grounded ONLY when you are certain the evidence states the claim or clearly \
entails it. Reason first, then decide.

Answer NOT GROUNDED (the safe default) when ANY of these hold:
- the evidence is about a different subject;
- the evidence contradicts the claim;
- the claim depends on a specific element (a number, threshold, date, condition, or named \
entity) that the evidence does not state;
- the evidence is merely related or topical, or makes the claim plausible, without actually \
stating or entailing it;
- the evidence neither supports nor contradicts the claim (not enough information);
- you are not certain. If in doubt, answer NOT GROUNDED.

Answer GROUNDED only when the evidence states or entails the claim. Wording need not match: a \
paraphrase, a summary, or a more detailed statement that entails the claim is grounded. But do \
not fill gaps with outside knowledge, and do not treat topical overlap or plausibility as support.

The error to avoid above all else is calling an UNSUPPORTED claim grounded.

Respond with a single JSON object and nothing else, giving the reason BEFORE the verdict, in \
exactly this form:
{"reason": "<one sentence naming what the evidence does or does not state>", "grounded": false}
"""

STRICT_USER = """CLAIM:
{claim}

EVIDENCE:
{evidence}

Decide whether the claim is grounded in the evidence. First give your one-sentence reason, then \
the verdict. Respond with the JSON object only."""
