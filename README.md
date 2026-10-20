# isgrounded

**A runtime verify-or-abstain grounding gate for LLM/RAG pipelines , with bounded re-grounding (rescues a mis-citation without laundering a hallucination) and a harness to measure how reliable your judge actually is.**

`isgrounded` is a small, runtime **verify-or-abstain** gate for RAG and LLM pipelines: it credits a claim only when the evidence actually supports it, with a citation, and **abstains** otherwise instead of guessing. The gate itself is deliberately simple , a strict LLM judge. The real work is the two things built around it:

- **Bounded re-grounding.** When the cited passage is the wrong one, it re-searches your corpus *by the claim* and re-verifies at the **same strict bar** , so it rescues a *mis-citation* without ever laundering a *hallucination*. "Cited" ends up meaning **verified**, not plausible. (See [How it works](#how-it-works).)
- **A judge-measurement harness.** A grounding gate is only as good as the model and prompt behind it , so isgrounded ships the benchmarks to measure *your* judge and prompt, not just a badge that says "grounded." (See [Measure your judge](#measure-your-judge).)

Model-agnostic and retriever-agnostic: a wrapper, not a model, not a post-hoc score.

```python
from isgrounded import is_grounded

# Grounded , the evidence entails the claim (6000 > 5000), even though the number differs.
is_grounded(
    claim="Dragon Con had over 5000 guests.",
    evidence="Among the more than 6000 guests and performers at the 2009 convention were Patrick Stewart, William Shatner, and Leonard Nimoy.",
)  # -> True

# Not grounded , same topic, but the evidence doesn't establish the specific claim.
is_grounded(
    claim="Dragon Con had over 5000 guests.",
    evidence="Among the more than 512 guests and performers at the 2009 convention were Patrick Stewart, William Shatner, and Leonard Nimoy.",
)  # -> False  ("more than 512" doesn't support "over 5000")
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

**1. Input , a claim and its evidence.** Pass `is_grounded(claim, evidence)`: the claim an LLM produced, plus the evidence it's supposedly based on. No evidence in hand? Pass a retriever and it fetches the top passages *by the claim* first.

**2. The gate , a strict judge decides support.** Claim + evidence go to an LLM judge with one job: does the evidence **state or entail** the claim? Strict by design , it credits paraphrase and entailment, but rejects a different subject, a contradiction, a **missing specific element** (a number/date/entity the claim needs but the evidence omits), or something merely related or plausible. Returns a verdict + one-sentence reason as rigid JSON. Single judge by default; `strict=True` runs a 3-vote panel (majority, two-then-tiebreak).

**3. Fail-closed.** If the judge errors or returns anything unparseable → treated as **not grounded**. When in doubt, abstain , a false "grounded" is the dangerous error.

**4. If grounded → return it with the citation.** Verdict `grounded`, supporting passage attached.

**5. If not grounded → bounded re-grounding (the core mechanism).** The first evidence may have been the **wrong chunk**, so:
- re-search the corpus **by the claim** (not the original query);
- drop passages already cited, and drop near-zero-overlap passages (a retrieval floor) so it can't grab junk;
- re-verify the fresh top passages with the **same strict gate**.

Supported now → verdict `recovered`. Still not → abstain. **The safety property:** a *mis-citation* (support was in the corpus, wrong chunk cited first) gets rescued; a *confabulation* (support nowhere) does not , because the bar never dropped. This is what makes "cited" mean *verified*.

**6. Output , a decision you can act on.** `grounded` (True/False) · `verdict` (grounded | ungrounded | recovered) · `reason` · `citations`. Every "ungrounded" is a **signal** , a hole in your knowledge base, not just a failure.

### How it's built
- **Model-agnostic** , an `LLM` protocol; bring OpenAI, Ollama, any OpenAI-compatible endpoint, or Gemini via env. It's a wrapper, not a model.
- **Retriever-agnostic** , a `Retriever` protocol (`.search`); wrap your LangChain / LlamaIndex / vector DB, or use a bundled one (conveniences , BYO for production).
- **Fail-closed abstention** is the default stance throughout.
- **The prompt is the gate's behavioral core** , and, per the [benchmarks](#measure-your-judge), its exact wording is a precision/recall dial you tune per model.

## Measure your judge

A grounding gate is only as reliable as the judge and prompt behind it , so isgrounded ships the harness to **measure your own setup**, not just a number I'm asking you to trust. Point it at your model (same env as [Install](#install)) and run:

```bash
python benchmarks/vitaminc_bench.py    --label my-model   # gate robustness: abstain vs. credit
python benchmarks/regrounding_bench.py --label my-model   # re-grounding: rescue vs. false-rescue
```

**Small sets by design** , 150 VitaminC cases and 9 re-grounding cases. Read these as directional: they show the mechanism, not a leaderboard.

Here's what that harness found across four judge models (this package's own prompts, on both a lexical and a semantic retriever). Full tables + reproducible logs: **[benchmarks/RESULTS.md](benchmarks/RESULTS.md)**.

**The gate is only as sharp as the judge you give it , and the retriever doesn't change that.** On a capable hosted judge (Gemini 2.5 Flash):
- **Gate robustness** , VitaminC, 150 hard contrastive cases: **accuracy 0.90, F1 0.93**, 5 dangerous misses.
- **Re-grounding** , public EU AI Act text: it recovered every mis-citation it was given and credited **0 of 4** confabulations, on *both* retrievers. The safety property is the gate's, not the retriever's.

On the small local open models (`llama3.1:8b`, `mistral:7b`, `gemma3:12b`), the **default** prompt under-abstains , on VitaminC they miss insufficient-evidence cases (accuracy 0.60–0.71), and in re-grounding they credit **1–3 of 4** fabrications (one laundered a fake statutory fine through re-grounding). But that turned out to be **largely the prompt, not the model**: a principle-built *strict-abstention* prompt drove fabrications-credited to **0/4 on every model, including the 7B ones.** The shipped prompt was developed on Gemini and simply didn't transfer , the open models aren't "too weak to be safe," they needed a prompt written for them.

The real lesson is the catch: **the prompt is a precision/recall dial, and its right setting is model-dependent.** Under the strict prompt, Gemini and gemma3:12b stay safe *and* keep recovering real mis-citations (gemma improved on both axes) , but the 7–8B models over-correct and start rejecting *everything* (rescue collapses to 1/5 and 0/5). Gemini was **prompt-robust** (identical across prompts and retrievers); the small models are prompt-sensitive in *both* directions. (A related default-prompt finding: a better *retriever* made the most lenient judge worse , mistral 2/4 → 3/4 , because stronger retrieval hands a lenient judge more plausible wrong passages to rationalize.) Full tables: **[RESULTS.md](benchmarks/RESULTS.md)**.

**The takeaway isn't "use a hosted model."** It's **measure your judge and prompt** , if you run a smaller or open model, tune the prompt and check its abstention behavior. This harness shows you how.

**Honest limits:**
- It validates **faithfulness to your source**, not correctness of the world. A wrong corpus yields faithful-but-wrong answers.
- It makes **answers** trustworthy; it does not make autonomous high-stakes **decisions** safe. Keep a human in the loop.
- The shipped prompt was developed on Gemini and **under-transfers**: a strict variant recovers safety on open models (0/4 fabrications , RESULTS §2c), but can tip 7–8B models into over-abstention. Tune the prompt per judge and watch **both** failure modes (crediting fabrications *and* rejecting real support).
- Small n (VitaminC 150; re-grounding 9 cases, 5 mis-citation / 4 confabulation). Directional, not paper-grade , the figures illustrate the *mechanism* (judge quality dominates), not a leaderboard.

## Roadmap

Prompts don't transfer cleanly across models, so the direction is to make the *specification* portable and calibrate per judge, not to chase one universal prompt:

- **Per-model prompt defaults.** A strict-abstention variant already recovers safety (0/4 fabrications) on every model tested, but over-abstains on 7–8B models , so the next step isn't one stricter prompt for all, it's a small set of calibrated defaults (capable vs. small) plus the harness to pick the right one per judge.
- **Per-model prompt adaptation** , a thin adapter that injects model-specific scaffolding, or compiling the gate as a [DSPy](https://github.com/stanfordnlp/dspy) signature so the prompt is optimized per judge instead of hand-written once.
- **Cheap deterministic pre-pass** , auto-abstain on zero-signal evidence before calling the judge, to save cost on easy cases. (This only catches *no-overlap* cases; it cannot catch plausible-but-unsupported claims, since those overlap heavily , that judgment stays with the gate.)
- **Wider benchmarks** , larger VitaminC and re-grounding sets; the current numbers are directional.

## License

Apache-2.0. Built by [Emily Luskind](https://github.com/eluskind).
