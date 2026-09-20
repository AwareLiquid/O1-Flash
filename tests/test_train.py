"""Training pipeline tests: JOINT multi-question training learns all
questions (the hybrid-readout resolution of DESIGN §7)."""

from mt_flash.config import o1_flash_tiny
from mt_flash.model import O1Flash
from mt_flash.train import (default_questions, make_synthetic_state, train)


def test_synthetic_state_has_valid_labels():
    import random
    text, dept, sev, urgent = make_synthetic_state(random.Random(0))
    assert "tail:dept=" in text and "severity=" in text
    assert 0 <= dept < 4 and 0 <= sev < 4 and urgent in (0, 1)


def test_joint_training_learns_all_questions():
    """Joint calibration training: all three question types beat chance.

    Chance: dept 0.25 (4-way), sev 0.25 (4-way), urgent 0.5 (binary).
    Measured with hybrid readout: 0.73 / 0.81 / 1.0 at 600 steps.
    """
    model = O1Flash(o1_flash_tiny())
    m = train(model, steps=600, batch=32)
    assert m["dept_acc"] > 0.55
    assert m["sev_acc"] > 0.55
    assert m["urgent_acc"] > 0.7
    for k in ("dept", "sev", "urgent"):
        assert 0.0 <= m[f"{k}_ece"] <= 1.0


def test_default_questions_are_typed():
    from mt_flash.schema import (ChoiceQuestion, ProbabilityQuestion,
                                 ScoreQuestion)
    qs = default_questions()
    assert isinstance(qs[0], ChoiceQuestion)
    assert isinstance(qs[1], ScoreQuestion)
    assert isinstance(qs[2], ProbabilityQuestion)
