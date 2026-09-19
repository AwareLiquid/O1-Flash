"""Parallel-head tests: question independence, schema-bounded outputs,
confidence = peakedness."""

import torch

from mt_flash.config import o1_flash_tiny
from mt_flash.model import O1Flash
from mt_flash.schema import (ChoiceQuestion, ProbabilityQuestion,
                             ScoreQuestion)


def _model():
    m = O1Flash(o1_flash_tiny())
    m.eval()
    return m


def test_questions_are_independent():
    """Jev's key property: question A's answer is identical whether B is
    present or not — no cross-question interaction exists."""
    m = _model()
    state = "The server is down since 03:12; tickets piling up."
    qa = ChoiceQuestion(id="route", options=("billing", "infra", "other"))
    qb = ProbabilityQuestion(id="urgent", prompt="needs escalation now")
    qc = ScoreQuestion(id="sev", levels=("low", "mid", "high"))

    with torch.no_grad():
        a_only = m.decide(state, [qa]).answers["route"].distribution
        a_with = m.decide(state, [qa, qb, qc]).answers["route"].distribution
    assert a_only == a_with


def test_choice_distribution_is_schema_bounded():
    m = _model()
    q = ChoiceQuestion(id="r", options=("a", "b", "c"))
    ans = m.decide("some state text", [q]).answers["r"]
    d = ans.distribution
    assert len(d) == 3
    assert abs(sum(d) - 1.0) < 1e-6
    assert all(0.0 <= p <= 1.0 for p in d)
    assert 0 <= ans.choice < 3
    assert 0.0 <= ans.confidence <= 1.0


def test_probability_is_in_unit_interval():
    m = _model()
    q = ProbabilityQuestion(id="p", prompt="will it rain")
    ans = m.decide("weather data", [q]).answers["p"]
    assert 0.0 <= ans.probability <= 1.0
    assert 0.0 <= ans.confidence <= 1.0


def test_score_levels_are_schema_bounded():
    m = _model()
    q = ScoreQuestion(id="s", levels=("low", "mid", "high"))
    ans = m.decide("review text", [q]).answers["s"]
    assert len(ans.distribution) == 3
    assert abs(sum(ans.distribution) - 1.0) < 1e-6
    assert 0 <= ans.level < 3


def test_confidence_is_peakedness():
    """Flat distribution -> low confidence; peaked -> high, monotonic."""
    from mt_flash.decision_heads import _peakedness
    flat = torch.full((1, 5), 0.2)
    peaked = torch.tensor([[0.95, 0.05, 0.0, 0.0, 0.0]])
    assert _peakedness(flat).item() < 0.05
    assert _peakedness(peaked).item() > 0.5


def test_schema_rejects_out_of_contract_questions():
    import pytest
    from mt_flash.schema import ChoiceQuestion
    with pytest.raises(ValueError):
        ChoiceQuestion(id="x", options=tuple(f"o{i}" for i in range(256)))
    with pytest.raises(ValueError):
        ChoiceQuestion(id="x", options=())
