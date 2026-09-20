"""Cascade tests: confidence-threshold routing partitions correctly."""

from mt_flash.cascade import FastSlowRouter
from mt_flash.config import o1_flash_tiny
from mt_flash.model import O1Flash
from mt_flash.schema import ChoiceQuestion


def _slow_handler():
    calls = []

    def handler(state: str, q) -> str:
        calls.append((q.id, state))
        return f"slow:{q.id}"
    return handler, calls


def test_all_handled_at_zero_threshold():
    m = O1Flash(o1_flash_tiny())
    handler, calls = _slow_handler()
    router = FastSlowRouter(m, threshold=0.0, slow=handler)
    q = ChoiceQuestion(id="r", options=("a", "b"))
    results = router.route("some state", [q])
    assert all(r.handled and not r.escalated for r in results)
    assert calls == []


def test_all_escalated_at_unity_threshold():
    m = O1Flash(o1_flash_tiny())
    handler, calls = _slow_handler()
    router = FastSlowRouter(m, threshold=1.0, slow=handler)
    q = ChoiceQuestion(id="r", options=("a", "b"))
    results = router.route("state text", [q])
    assert all(not r.handled and r.escalated for r in results)
    assert len(calls) == 1


def test_partition_matches_confidence():
    """Handled iff answer confidence >= threshold (checked per answer)."""
    m = O1Flash(o1_flash_tiny())
    threshold = 0.5
    handler, calls = _slow_handler()
    router = FastSlowRouter(m, threshold=threshold, slow=handler)
    qs = [ChoiceQuestion(id=f"q{i}", options=("a", "b", "c"))
          for i in range(8)]
    resp = m.decide("partition test state", qs)
    results = router.route("partition test state", qs)
    for r, q in zip(results, qs):
        conf = resp.answers[q.id].confidence
        assert r.handled == (conf >= threshold)
        assert r.escalated == (conf < threshold)


def test_threshold_validation():
    import pytest
    m = O1Flash(o1_flash_tiny())
    with pytest.raises(ValueError):
        FastSlowRouter(m, threshold=1.5, slow=lambda s, q: None)
    with pytest.raises(ValueError):
        FastSlowRouter(m, threshold=-0.1, slow=lambda s, q: None)
