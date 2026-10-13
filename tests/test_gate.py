"""Gate + verifier tests. No network, no model load — a scripted FakeLLM stands in for the judge."""
from __future__ import annotations

from isgrounded import GroundedVerifier, grounded_gate, is_grounded
from isgrounded.types import Citation


class FakeLLM:
    """Returns the next scripted response on each complete() call."""
    def __init__(self, scripted: list[str]):
        self.scripted = list(scripted)
        self.calls = 0

    def complete(self, system: str, user: str, *, temperature: float = 0.0, seed=None) -> str:
        out = self.scripted[min(self.calls, len(self.scripted) - 1)]
        self.calls += 1
        return out


def test_grounded_true():
    llm = FakeLLM(['{"grounded": true, "reason": "evidence states it directly"}'])
    grounded, reason = grounded_gate(llm, "claim", "evidence")
    assert grounded is True and "states" in reason


def test_ungrounded():
    llm = FakeLLM(['{"grounded": false, "reason": "different subject"}'])
    grounded, _ = grounded_gate(llm, "claim", "evidence")
    assert grounded is False


def test_fail_closed_on_unparseable():
    llm = FakeLLM(["not json at all"])
    grounded, reason = grounded_gate(llm, "c", "e")
    assert grounded is False and "abstain" in reason.lower()


def test_fail_closed_on_error():
    class Boom:
        def complete(self, *a, **k):
            raise RuntimeError("provider down")
    grounded, reason = grounded_gate(Boom(), "c", "e")
    assert grounded is False and "abstain" in reason.lower()


def test_is_grounded_convenience():
    llm = FakeLLM(['{"grounded": true, "reason": "ok"}'])
    assert is_grounded("c", "e", llm=llm) is True


def test_strict_tiebreak_goes_ungrounded():
    # grounded, ungrounded -> split -> third seed ungrounded -> not grounded
    llm = FakeLLM(['{"grounded": true, "reason":"a"}',
                   '{"grounded": false, "reason":"b"}',
                   '{"grounded": false, "reason":"c"}'])
    grounded, _ = grounded_gate(llm, "c", "e", strict=True)
    assert grounded is False
    assert llm.calls == 3


def test_strict_unanimous_short_circuits_to_two_calls():
    llm = FakeLLM(['{"grounded": true, "reason":"a"}',
                   '{"grounded": true, "reason":"b"}',
                   '{"grounded": false, "reason":"must-not-be-read"}'])
    grounded, _ = grounded_gate(llm, "c", "e", strict=True)
    assert grounded is True
    assert llm.calls == 2  # two-then-tiebreak: no third call when the first two agree


class FakeRetriever:
    def __init__(self, passages):
        self._p = [Citation(source_id=i, text=t, score=s) for i, t, s in passages]

    def search(self, query, k=5):
        return self._p[:k]


def test_reground_rescues_miscitation():
    # gate says ungrounded on the provided (wrong) evidence, then grounded on the re-searched passage
    llm = FakeLLM(['{"grounded": false, "reason":"wrong chunk"}',
                   '{"grounded": true, "reason":"the real passage supports it"}'])
    retr = FakeRetriever([("right-chunk", "the supporting text", 0.9)])
    v = GroundedVerifier(llm=llm, retriever=retr)
    r = v.verify("claim", "a wrong, non-supporting passage", reground=True)
    assert r.grounded is True and r.verdict == "recovered"
    assert r.citations and r.citations[0].source_id == "right-chunk"


def test_reground_does_not_rescue_confabulation():
    # ungrounded on provided, still ungrounded on everything re-searched -> stays rejected
    llm = FakeLLM(['{"grounded": false, "reason":"wrong chunk"}',
                   '{"grounded": false, "reason":"nothing supports this"}'])
    retr = FakeRetriever([("other", "unrelated text", 0.9)])
    v = GroundedVerifier(llm=llm, retriever=retr)
    r = v.verify("fabricated claim", "a wrong passage", reground=True)
    assert r.grounded is False and r.verdict == "ungrounded"


def test_reground_respects_retrieval_floor():
    # only candidate is below the floor -> no re-ground attempt -> stays ungrounded after 1 call
    llm = FakeLLM(['{"grounded": false, "reason":"wrong chunk"}'])
    retr = FakeRetriever([("weak", "barely related", 0.01)])
    v = GroundedVerifier(llm=llm, retriever=retr, retrieval_floor=0.05)
    r = v.verify("claim", "a wrong passage", reground=True)
    assert r.grounded is False
    assert llm.calls == 1  # floor filtered the only candidate; no second judge call
