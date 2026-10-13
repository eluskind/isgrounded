#!/usr/bin/env python3
"""Re-grounding benchmark for isgrounded, on the public EU AI Act corpus. Judge-model agnostic.

Measures the two properties re-grounding claims:
  * RESCUE RATE  — mis-citations (support IS in the corpus, wrong chunk cited) that get recovered.
  * FALSE-RESCUE — confabulations (support is NOWHERE) wrongly recovered. MUST be ~0 (safety).

Uses the package's GroundedVerifier + TfidfRetriever (torch-free) so it runs under any judge LLM
selected via ISGROUNDED_LLM. The false-rescue number is retriever-agnostic (the gate decides);
the rescue number under TF-IDF is a lower bound vs a semantic retriever.

Run (one leg of the sweep):
  ISGROUNDED_LLM=gemini ISGROUNDED_MODEL=gemini-2.5-flash \
      python benchmarks/regrounding_bench.py --label gemini-2.5-flash
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from isgrounded import GroundedVerifier, TfidfRetriever   # noqa: E402
from isgrounded.llm import llm_from_env                   # noqa: E402


def build_corpus(txt_path: str) -> list[tuple[str, str]]:
    """Chunk the EU AI Act text into recital- and article-paragraph passages -> (source_id, text)."""
    raw = open(txt_path, encoding="utf-8").read()
    out: list[tuple[str, str]] = []

    def add(sid: str, text: str):
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) >= 40:
            out.append((sid, text))

    m_w = re.search(r"\bWhereas:\b", raw)
    m_adopt = re.search(r"HAVE ADOPTED THIS REGULATION", raw)
    recital_zone = raw[m_w.end():m_adopt.start()] if (m_w and m_adopt) else ""
    enacting_zone = raw[m_adopt.end():] if m_adopt else raw

    rec = re.split(r"(?m)^\s*\((\d+)\)\s", recital_zone)
    for i in range(1, len(rec) - 1, 2):
        add(f"Recital ({rec[i]})", rec[i + 1])

    parts = re.split(r"(?m)^Article (\d+)\s*$", enacting_zone)
    for i in range(1, len(parts) - 1, 2):
        n, body = parts[i], re.split(r"(?m)^ANNEX\b", parts[i + 1])[0]
        lines = body.strip().split("\n")
        title = lines[0].strip() if lines else ""
        base = f"Article {n} — {title}" if title else f"Article {n}"
        rest = "\n".join(lines[1:]) if len(lines) > 1 else ""
        paras = re.split(r"(?m)^\s*(\d+)\.\s", rest)
        if len(paras) >= 3:
            if paras[0].strip():
                add(f"{base} #p0", f"{title}. {paras[0]}")
            for j in range(1, len(paras) - 1, 2):
                add(f"{base} #p{paras[j]}", f"{title}. ({paras[j]}) {paras[j + 1]}")
        else:
            add(base, f"{title}. {rest}")
    return out


def _find(passages: list[tuple[str, str]], needle: str) -> tuple[str, str] | None:
    nl = needle.lower()
    for sid, text in passages:
        if nl in text.lower() or nl in sid.lower():
            return sid, text
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="benchmarks/data/eu_ai_act_text.txt")
    ap.add_argument("--cases", default="benchmarks/data/regrounding_cases_eu_ai_act.jsonl")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--label", default=os.environ.get("ISGROUNDED_MODEL", "model"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = args.out or f"benchmarks/out/regrounding_{args.label}.txt"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sink = open(path, "w")

    def emit(x: str = ""):
        print(x, flush=True)
        sink.write(x + "\n")
        sink.flush()

    provider = os.environ.get("ISGROUNDED_LLM", "openai")
    emit(f"building corpus from {args.corpus} ...")
    passages = build_corpus(args.corpus)
    retriever = TfidfRetriever(passages)
    verifier = GroundedVerifier(llm=llm_from_env(), retriever=retriever,
                                strict=args.strict, retrieval_floor=0.05)
    emit(f"re-grounding | {len(passages)} chunks | judge={args.label} | provider={provider} "
         f"| strict={args.strict}\n")

    cases = [json.loads(ln) for ln in open(args.cases) if ln.strip() and not ln.startswith("//")]

    pos_total = pos_recovered = pos_correct = 0
    neg_total = neg_false = 0
    skipped = 0

    for i, c in enumerate(cases, 1):
        claim, kind = c["claim"], c["kind"]
        wrong = _find(passages, c["wrong_chunk_contains"])
        if wrong is None:
            emit(f"[{i}] SKIP (wrong chunk not found: {c['wrong_chunk_contains']!r})")
            skipped += 1
            continue
        wrong_id, wrong_text = wrong

        # verify against the deliberately-wrong chunk, with re-grounding enabled
        r = verifier.verify(claim, wrong_text, reground=True, source_id=wrong_id)

        if r.verdict == "grounded":
            # the wrong chunk unexpectedly supported the claim -> invalid case / gate issue
            tag = "GATE-FAIL" if kind == "confabulation" else "mis-designed"
            emit(f"[{i}] SKIP ({tag}: wrong chunk judged grounded) [{wrong_id}]")
            skipped += 1
            continue

        if kind == "mis-citation":
            pos_total += 1
            if r.verdict == "recovered":
                pos_recovered += 1
                exp = (c.get("expected_chunk_contains") or "").lower()
                ok = any(exp in (ci.text or "").lower() or exp in (ci.source_id or "").lower()
                         for ci in r.citations) if exp else None
                if ok:
                    pos_correct += 1
                landed = ", ".join(sorted({ci.source_id for ci in r.citations}))
                emit(f"[{i}] RECOVERED ✓ -> {landed}"
                     + ("" if ok is None else (" [correct]" if ok else " [WRONG section]")))
            else:
                emit(f"[{i}] MISSED ✗ (stayed ungrounded; should have recovered)")
        else:
            neg_total += 1
            if r.verdict == "recovered":
                neg_false += 1
                landed = ", ".join(sorted({ci.source_id for ci in r.citations}))
                emit(f"[{i}] FALSE-RESCUE ✗✗ (rescued a confabulation!) -> {landed}")
            else:
                emit(f"[{i}] REJECTED ✓ (confabulation correctly not rescued)")

    emit(f"\n=== {args.label} | re-grounding metrics ===")
    if pos_total:
        emit(f"RESCUE RATE: {pos_recovered}/{pos_total} = {pos_recovered/pos_total:.2f}  "
             f"(to correct section: {pos_correct}/{pos_total} = {pos_correct/pos_total:.2f})")
    if neg_total:
        emit(f"FALSE-RESCUE RATE: {neg_false}/{neg_total} = {neg_false/neg_total:.2f}  (MUST be ~0)")
    emit(f"skipped/mis-designed: {skipped}")
    emit(f"\nNote: TF-IDF retriever -> rescue rate is a LOWER BOUND; false-rescue is retriever-agnostic.")
    emit(f"\nsaved {path}")
    sink.close()


if __name__ == "__main__":
    main()
