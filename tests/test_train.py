"""Training pipeline tests: single-question training learns; labels valid."""

from mt_flash.config import o1_flash_tiny
from mt_flash.model import O1Flash
from mt_flash.train import make_question, make_synthetic_state, train_single


def test_synthetic_state_has_valid_labels():
    import random
    text, dept, sev, urgent = make_synthetic_state(random.Random(0))
    assert "tail:dept=" in text and "severity=" in text
    assert 0 <= dept < 4 and 0 <= sev < 4 and urgent in (0, 1)


def test_single_question_training_learns():
    """The pipeline trains end-to-end on ONE question: acc well above chance."""
    model = O1Flash(o1_flash_tiny())
    m = train_single(model, make_question("dept"), steps=400, batch=32)
    # chance = 0.25 (4-way dept)
    assert m["dept_acc"] > 0.55
    assert 0.0 <= m["dept_ece"] <= 1.0


def test_make_question_types():
    from mt_flash.schema import (ChoiceQuestion, ProbabilityQuestion,
                                 ScoreQuestion)
    assert isinstance(make_question("dept"), ChoiceQuestion)
    assert isinstance(make_question("sev"), ScoreQuestion)
    assert isinstance(make_question("urgent"), ProbabilityQuestion)
