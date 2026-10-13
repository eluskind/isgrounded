# isgrounded

**Before you ship an LLM answer, ask: is it grounded?**

`isgrounded` is a small, runtime **verify-or-abstain** gate for RAG and LLM pipelines. It credits a claim only when the retrieved evidence actually supports it, with a citation, and **abstains** otherwise instead of guessing. It's model-agnostic and retriever-agnostic: a wrapper, not a model, not a post-hoc score.

```python
from isgrounded import is_grounded

is_grounded(
    claim="The system encrypts customer data at rest.",
    evidence="All records in our data stores are encrypted on disk using AES-256.",
)  # -> True

is_grounded(
    claim="The system encrypts customer data at rest.",
    evidence="Our team values security and reviews access quarterly.",
)  # -> False  (related topic, doesn't actually support the claim)
```

## Why

Most RAG apps demo beautifully and then can't ship, because under real queries they confidently make things up. For anything high-stakes, a confident-but-wrong answer is worse than "I can't verify that." `isgrounded` makes that call explicit and checkable:

- **A runtime decision, not a score.** Faithfulness scorers tell you *after the fact* how bad an answer was. `isgrounded` decides *before it ships*: ground it (with a citation) or abstain.
- **Grounding, not confidence.** It abstains based on whether a claim can be tied to a passage you can open and read, not on an uncalibrated confidence number.
- **Checkable citations.** When it credits a claim, you get the passage. When the retriever cited the wrong passage, **re-grounding** re-searches your corpus by the claim and either fixes the citation or abstains, so "cited" means *verified*, not *plausible*.
- **Abstention is a signal.** Every "unassessed" is a hole in your knowledge base, a content roadmap, not just a failure.

## Install

```bash
pip install isgrounded                 # core (bring your own LLM client)
pip install "isgrounded[openai]"       # OpenAI / Ollama / OpenRouter / any OpenAI-compatible endpoint
pip install "isgrounded[gemini]"       # Google Gemini
pip install "isgrounded[tfidf]"        # the bundled torch-free lexical retriever
```

Point it at an LLM via env:

```bash
export ISGROUNDED_LLM=openai           # or: ollama | openrouter | gemini
export ISGROUNDED_MODEL=gpt-4o-mini
export OPENAI_API_KEY=sk-...
```

## Use

**The gate:**
```python
from isgrounded import GroundedVerifier

v = GroundedVerifier()                 # LLM from env; single-judge by default
r = v.verify(claim="...", evidence="...")
r.grounded        # True / False
r.verdict         # "grounded" | "ungrounded"
r.reason          # one-sentence explanation
r.citations       # the supporting passage(s)
```

**With a retriever → bounded re-grounding** (rescue a mis-cited-but-real claim without rescuing a hallucination):
```python
from isgrounded import GroundedVerifier, TfidfRetriever

retriever = TfidfRetriever([("doc1#p3", "...passage text..."), ("doc1#p9", "...")])
v = GroundedVerifier(retriever=retriever)
r = v.verify(claim="...", evidence="...the wrong chunk...", reground=True)
r.verdict         # "recovered" if re-grounding found real support, else "ungrounded"
```
Any object with `.search(query, k) -> list[Citation]` works — wrap your LangChain / LlamaIndex / vector-DB retriever, or use the bundled `TfidfRetriever`.

**High-stakes mode (multi-judge panel):**
```python
GroundedVerifier(strict=True)          # majority vote of 3 seeds (two-then-tiebreak)
```

**CLI:**
```bash
isgrounded verify --claim "..." --evidence "..."
```

## How it works

1. **Gate** — a strict judge decides whether the evidence *directly supports the specific claim* (not a related topic, not a more general statement, not a plausible inference). Single-judge by default; `strict=True` runs a majority-of-3 panel.
2. **Re-grounding** — on "ungrounded", re-search the corpus by the **claim** (not the original query), drop already-cited passages, and re-verify with the **same strict gate**. A mis-citation (support is in the corpus, wrong chunk retrieved) gets rescued; a confabulation (support is nowhere) does not, because the bar is unchanged.
3. **Abstain** — if it still can't ground the claim, it returns `ungrounded` rather than guessing. Fail-closed: on a provider error or an unparseable judge response, it abstains.

## Validation (measured, honest)

Measured with **this package's own prompts**, across four judge models. Full tables + reproducible logs: **[benchmarks/RESULTS.md](benchmarks/RESULTS.md)**.

**The gate is only as sharp as the judge you give it.** On a capable hosted judge (Gemini 2.5 Flash):
- **Gate robustness** , VitaminC, 150 hard contrastive cases: **accuracy 0.90, F1 0.93**, 5 dangerous misses.
- **Re-grounding** , public EU AI Act text: recovered **5/5** mis-citations (to the correct article) and credited **0 of 4** confabulations. The safety property holds.

On small local open models (`llama3.1:8b`, `mistral:7b`, `gemma3:12b`) the gate is **too lenient**: it under-abstains (VitaminC accuracy 0.60–0.71, mostly missing insufficient-evidence cases) and credits **1–2 of 4** fabrications in re-grounding (one model laundered a fake statutory fine *through* re-grounding). They run, they just aren't reliable judges. **Use a capable judge for high-stakes work.**

**Honest limits:**
- It validates **faithfulness to your source**, not correctness of the world. A wrong corpus yields faithful-but-wrong answers.
- It makes **answers** trustworthy; it does not make autonomous high-stakes **decisions** safe. Keep a human in the loop.
- Small n (VitaminC 150, re-grounding 9), directional, not paper-grade. The re-grounding rescue rate uses a lexical (TF-IDF) retriever and is a **lower bound** vs. a semantic retriever; a semantic re-grounding run is a pending next step.

## License

Apache-2.0. Built by [Annexa](https://github.com/annexa).
