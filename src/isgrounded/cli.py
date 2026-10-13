"""`isgrounded` CLI. Reads the LLM from env (ISGROUNDED_LLM / ISGROUNDED_MODEL)."""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="isgrounded",
        description="Ask: is this claim grounded in the evidence?")
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify", help="verify a single claim against evidence")
    v.add_argument("--claim", required=True)
    v.add_argument("--evidence", required=True)
    v.add_argument("--strict", action="store_true", help="use the multi-judge panel")

    args = ap.parse_args(argv)

    if args.cmd == "verify":
        from .verifier import GroundedVerifier
        result = GroundedVerifier(strict=args.strict).verify(args.claim, args.evidence)
        print(f"{result.verdict.upper()}: {result.reason}")
        return 0 if result.grounded else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
