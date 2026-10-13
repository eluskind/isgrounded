# isgrounded — benchmark results

Measured with **this package's own prompts** (`src/isgrounded/prompts.py`, the fresh/generic ones), via the package gate. Judge model selected per run with `ISGROUNDED_LLM`. Re-grounding uses the bundled torch-free `TfidfRetriever`. Public data only (VitaminC sample; EU AI Act text). Raw per-run logs in `benchmarks/out/`.

Headline: **isgrounded runs on any model, but the gate is only as sharp as the judge.** A capable hosted judge (Gemini 2.5 Flash) is strong and safe; small local open models are lenient and credit claims they shouldn't. The tool *exposes* judge quality, it doesn't paper over it.

---

## 1. VitaminC — gate robustness (abstain vs. credit)

150 balanced cases (50 supported / 50 refuted / 50 nei). Positive class = abstention. `missed` = should-abstain-but-credited (the dangerous error). Single-judge.

| judge | acc | F1 | precision | recall | supported | refuted | nei | missed |
|---|---|---|---|---|---|---|---|---|
| **Gemini 2.5 Flash** (hosted) | **0.90** | **0.93** | 0.90 | 0.95 | 0.80 | 1.00 | 0.90 | 5 |
| gemma3:12b (local) | 0.71 | 0.74 | 0.95 | 0.60 | 0.94 | 0.80 | 0.40 | 40 |
| mistral:7b-instruct (local) | 0.71 | 0.74 | 0.91 | 0.62 | 0.88 | 0.78 | 0.46 | 38 |
| llama3.1:8b (local) | 0.60 | 0.58 | 0.98 | 0.41 | 0.98 | 0.64 | 0.18 | 59 |

**Read:** the local open models fail the same way, too lenient. High precision (when they abstain, they're right) but low recall, concentrated on the **nei** row (0.18–0.46): they credit insufficient-evidence claims. The hosted model is balanced (nei 0.90, only 5 dangerous misses). Ordering: Gemini ≫ gemma3 ≈ mistral > llama3.1.

---

## 2. Re-grounding — safety & rescue (EU AI Act corpus, 497 chunks)

9 cases (5 mis-citation / 4 confabulation), TF-IDF retriever.
- **Rescue** = mis-citations recovered (support is in the corpus, wrong chunk cited).
- **Fabrications credited /4** = the honest end-to-end safety metric: a confabulation that the system ultimately credited, *either* by crediting the wrong chunk at the validity check (a lenient-gate failure) *or* by false-rescuing it through re-grounding. (The raw per-run "false-rescue rate" has a smaller denominator because gate-credited confabs drop out as "skipped" — so this combined number is the fair cross-model comparison.)

| judge | rescue (to correct section) | **fabrications credited /4** | how it failed |
|---|---|---|---|
| **Gemini 2.5 Flash** | 5/5 | **0/4** ✓ | — (clean; recovered all mis-citations, credited no fabrication) |
| llama3.1:8b (local) | 4/4 | 1/4 | credited "10-yr log retention" against Art 12 at the gate |
| gemma3:12b (local) | 3/3 | 1/4 | credited "AI ethics board" against Art 16 at the gate |
| mistral:7b-instruct (local) | 5/5 | **2/4** | false-rescued the fake €10M fine onto Art 99/100 **and** gate-credited "10-yr logs" |

**Read:** re-grounding's safety property (don't launder hallucinations) **holds perfectly with a capable judge** (Gemini: 0/4, no slips) but **degrades with weak judges** , mistral even laundered a fabrication *through* re-grounding by crediting the real penalty articles as support for a fake fine. The local judges also mis-credited positives to the wrong article (the "skipped" positives), the same leniency.

**Retriever caveat:** rescue rate uses TF-IDF (lexical), a **lower bound** vs. a semantic retriever; the fabrications-credited (safety) number is retriever-agnostic because the gate decides.

---

## 3. What to take from this

- **Use a capable judge for anything high-stakes.** On Gemini 2.5 Flash: VitaminC 0.90 / 5 misses, re-grounding 5/5 rescue / 0 fabrications credited. That's the recommended operating point.
- **Small local open models (7–12B) are too lenient** to be safe judges here, they under-abstain on VitaminC (nei 0.18–0.46) and credit 1–2 of 4 fabrications in re-grounding. They run, but the reliability isn't there.
- **`strict` (multi-judge) mode is unlikely to rescue a weak judge**, voting across seeds of the same model reduces variance, not a systematic leniency *bias*. (Untested here; stated as a limitation, not a measurement.)
- **n is small** (VitaminC 150; re-grounding 9). Directional and honest, not paper-grade. Wider case sets and a semantic-retriever re-grounding run are the obvious next steps.

*All numbers reproducible: `benchmarks/vitaminc_bench.py` and `benchmarks/regrounding_bench.py`, model via `ISGROUNDED_LLM` / `ISGROUNDED_MODEL`.*
