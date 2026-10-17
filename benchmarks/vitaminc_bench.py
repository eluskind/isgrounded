#!/usr/bin/env python3
"""VitaminC abstention benchmark for isgrounded — judge-model agnostic.

Measures the gate's abstain-vs-credit behavior on a public fact-verification set, using
whatever LLM `ISGROUNDED_LLM` points at, so you can compare judge models side by side.

Data: JSONL of {"claim","evidence","label"} where label is supported | refuted | nei.
  supported -> the gate SHOULD credit it (grounded, i.e. do NOT abstain)
  refuted / nei -> the gate SHOULD abstain (ungrounded)

Positive class = abstention. Watch 'missed' (should-abstain-but-credited) — the dangerous error.

Run (model chosen via env), one leg of the sweep:
  ISGROUNDED_LLM=gemini ISGROUNDED_MODEL=gemini-2.5-flash \
      python benchmarks/vitaminc_bench.py --label gemini-2.5-flash
  ISGROUNDED_LLM=openai ISGROUNDED_MODEL=gpt-4o-mini \
      python benchmarks/vitaminc_bench.py --label gpt-4o-mini
  ISGROUNDED_LLM=ollama ISGROUNDED_MODEL=llama3.1 \
      python benchmarks/vitaminc_bench.py --label llama3.1-local
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from isgrounded.gate import grounded_gate       # noqa: E402
from isgrounded.llm import llm_from_env          # noqa: E402


def _load_key_file(path: str) -> None:
    """Load API-key env vars from a dotenv-style NAME=VALUE file (keeps the key out of the shell).
    Only known API-key names are read; an already-set env var wins."""
    allow = {"GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, val = line.partition("=")
        name, val = name.strip(), val.strip().strip('"').strip("'")
        if name in allow and val and not os.environ.get(name):
            os.environ[name] = val


def load(path: str, limit: int):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        d = json.loads(line)
        rows.append((d["claim"], d["evidence"], d["label"].lower()))
    return rows[:limit]


def expected_abstain(label: str) -> bool:
    return label in ("refuted", "nei")


def score(rows):
    tp = fp = tn = fn = 0
    per: dict[str, list[int]] = {}
    for label, pred_grounded in rows:
        exp_ab = expected_abstain(label)
        pred_ab = not pred_grounded
        per.setdefault(label, [0, 0])
        per[label][1] += 1
        if pred_ab == exp_ab:
            per[label][0] += 1
        if exp_ab and pred_ab:
            tp += 1
        elif exp_ab and not pred_ab:
            fn += 1        # MISSED: should abstain, credited it (dangerous)
        elif not exp_ab and pred_ab:
            fp += 1        # OVER-abstained on a supported claim
        else:
            tn += 1
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * prec * rec / (prec + rec)
          if (prec == prec and rec == rec and (prec + rec)) else float("nan"))
    acc = (tp + tn) / len(rows) if rows else float("nan")
    return dict(precision=prec, recall=rec, f1=f1, accuracy=acc,
                tp=tp, fp=fp, tn=tn, fn=fn, per=per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="benchmarks/data/vitaminc_sample.jsonl")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--strict", action="store_true", help="use the multi-judge panel")
    ap.add_argument("--prompt", choices=["default", "strict"], default="default",
                    help="default = shipped package prompt; strict = experimental "
                         "strict-abstention variant (benchmarks/prompts_strict.py); shipped "
                         "prompt is never modified")
    ap.add_argument("--key-file", default=None,
                    help="dotenv-style file to read the judge API key from (keeps it out of the shell)")
    ap.add_argument("--label", default=os.environ.get("ISGROUNDED_MODEL", "model"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.key_file:
        _load_key_file(args.key_file)

    if args.prompt == "strict":
        # monkeypatch the gate's prompt for this run only; src/isgrounded/prompts.py untouched.
        sys.path.insert(0, os.path.dirname(__file__))
        import isgrounded.gate as _gate              # noqa: E402
        from prompts_strict import STRICT_SYSTEM, STRICT_USER  # noqa: E402
        _gate.GROUNDING_SYSTEM = STRICT_SYSTEM
        _gate.GROUNDING_USER = STRICT_USER

    provider = os.environ.get("ISGROUNDED_LLM", "openai")
    print(f"building gate (ISGROUNDED_LLM={provider}, label={args.label})...", flush=True)
    llm = llm_from_env()
    cases = load(args.data, args.limit)
    print(f"VitaminC | {len(cases)} cases | judge={args.label} | strict={args.strict}\n", flush=True)

    # Stream AND capture to a durable file incrementally, so nothing lives only in a temp log.
    path = args.out or f"benchmarks/out/vitaminc_{args.label}.txt"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sink = open(path, "w")

    def emit(x: str = ""):
        print(x, flush=True)
        sink.write(x + "\n")
        sink.flush()

    emit(f"VitaminC | {len(cases)} cases | judge={args.label} | prompt={args.prompt} "
         f"| strict={args.strict} | provider={provider}")
    rows = []
    for i, (claim, ev, label) in enumerate(cases, 1):
        grounded, _reason = grounded_gate(llm, claim, ev, strict=args.strict)
        rows.append((label, grounded))
        ok = (not grounded) == expected_abstain(label)
        emit(f"[{i}/{len(cases)}] gold={label:9} grounded={str(grounded):5} {'ok' if ok else 'XX'}")

    s = score(rows)

    emit(f"\n=== {args.label} | abstention metrics (positive class = abstain) ===")
    emit(f"P={s['precision']:.2f} R={s['recall']:.2f} F1={s['f1']:.2f} acc={s['accuracy']:.2f} "
         f"| missed(should-abstain-but-credited)={s['fn']} over-abstain(on-supported)={s['fp']}")
    for lab in ("supported", "refuted", "nei"):
        if lab in s["per"]:
            c, n = s["per"][lab]
            emit(f"  {lab:9}: {c}/{n}={c/n:.2f}")

    emit(f"\nsaved {path}")
    sink.close()


if __name__ == "__main__":
    main()
