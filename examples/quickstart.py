"""Quickstart: verify + re-ground over a tiny corpus.

Needs an LLM configured via env, e.g.:
    export ISGROUNDED_LLM=openai ISGROUNDED_MODEL=gpt-4o-mini OPENAI_API_KEY=sk-...
    pip install "isgrounded[openai,tfidf]"
    python examples/quickstart.py
"""
from isgrounded import GroundedVerifier, TfidfRetriever

# A tiny "corpus": (source_id, passage_text)
CORPUS = [
    ("policy#enc", "All customer records in our data stores are encrypted on disk using AES-256; "
                   "keys are held in a dedicated KMS."),
    ("policy#mfa", "Any connection to a production database must pass multi-factor authentication "
                   "before access is granted, logged per session."),
    ("policy#sec", "Our engineering team values security and reviews access quarterly."),
]

retriever = TfidfRetriever(CORPUS)
v = GroundedVerifier(retriever=retriever)

# 1) grounded: the evidence directly supports the claim
r = v.verify(
    claim="Customer data is encrypted at rest.",
    evidence="All customer records in our data stores are encrypted on disk using AES-256.",
)
print("1)", r.verdict, "-", r.reason)

# 2) ungrounded: related topic, no actual support
r = v.verify(
    claim="Production database access requires multi-factor authentication.",
    evidence="Our engineering team values security and reviews access quarterly.",
)
print("2)", r.verdict, "-", r.reason)

# 3) re-grounding: wrong chunk cited, but the real support is in the corpus -> recovered
r = v.verify(
    claim="Production database access requires multi-factor authentication.",
    evidence="Our engineering team values security and reviews access quarterly.",
    reground=True,
)
print("3)", r.verdict, "-", r.reason,
      "->", [c.source_id for c in r.citations])
