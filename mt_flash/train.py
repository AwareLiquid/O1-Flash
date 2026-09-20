"""Training reference: synthetic decision tasks + calibration loop.

Trains all typed questions JOINTLY on the shared encoder with the
calibration objective (Brier + decision CE) and ECE reporting. This
proves the PIPELINE — state encode once, parallel heads, calibrated
loss, ECE report — end to end, in the mode the parallel-readout design
is meant for.

The synthetic tasks are deliberately linear-separable: the point is the
pipeline, not decision skill.
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn

from .calibration import calibration_step, expected_calibration_error
from .model import O1Flash
from .schema import (ChoiceQuestion, ProbabilityQuestion, Question,
                     ScoreQuestion)

DEPT_OPTIONS = ("billing", "infra", "sales", "other")
SEV_LEVELS = ("low", "mid", "high", "critical")


def make_synthetic_state(rng: random.Random) -> tuple[str, int, int, int]:
    """One synthetic ticket: dept + severity + urgent flag -> labels.

    Returns (state_text, dept_idx, sev_idx, urgent 0/1). All three signals
    sit at the END of the text where a causal scan readout can see them.
    """
    dept = rng.randrange(4)
    sev = rng.randrange(4)
    urgent = int(rng.random() < 0.35) if sev < 2 else int(rng.random() < 0.8)
    noise = rng.choice(("user cannot login", "payment stuck", "latency spike",
                        "quota exceeded", "404 on dashboard",
                        "email bounce", "db connection refused"))
    return (f"note:{noise} tail:dept={DEPT_OPTIONS[dept]} "
            f"severity={SEV_LEVELS[sev]} urgent={'yes' if urgent else 'no'}",
            dept, sev, urgent)


def make_question(kind: str) -> Question:
    """One typed question for the synthetic state (dept/sev/urgent)."""
    if kind == "dept":
        return ChoiceQuestion(id="dept", options=DEPT_OPTIONS)
    if kind == "sev":
        return ScoreQuestion(id="sev", levels=SEV_LEVELS)
    if kind == "urgent":
        return ProbabilityQuestion(id="urgent", prompt="needs escalation now")
    raise ValueError(f"unknown kind: {kind}")


def default_questions() -> list[Question]:
    return [make_question(k) for k in ("dept", "sev", "urgent")]


_LABEL = {"dept": 1, "sev": 2, "urgent": 3}


def train(model: O1Flash, questions: list[Question] | None = None,
          steps: int = 600, batch: int = 32, lr: float = 3e-3,
          seed: int = 0, log_every: int | None = None) -> dict[str, float]:
    """Joint calibration training over all questions; returns metrics."""
    questions = questions or default_questions()
    rng = random.Random(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    for step in range(steps):
        rows = [make_synthetic_state(rng) for _ in range(batch)]
        ids = torch.nn.utils.rnn.pad_sequence(
            [model._ids_for(r[0]) for r in rows],
            batch_first=True, padding_value=256,
        )
        y = model(ids)
        pad_mask = ids != 256
        probs = model.heads.train_forward(y, questions, pad_mask=pad_mask)
        loss = sum(
            calibration_step(probs[q.id],
                             torch.tensor([r[_LABEL[q.id]] for r in rows]))
            for q in questions
        )
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log_every and (step + 1) % log_every == 0:
            print(f"step {step + 1:4d}  loss {loss.item():.4f}")

    # -- eval ----------------------------------------------------------
    model.eval()
    rows = [make_synthetic_state(random.Random(999 + i)) for i in range(256)]
    ids = torch.nn.utils.rnn.pad_sequence(
        [model._ids_for(r[0]) for r in rows],
        batch_first=True, padding_value=256,
    )
    with torch.no_grad():
        y = model(ids)
        pad_mask = ids != 256
        probs = model.heads.train_forward(y, questions, pad_mask=pad_mask)
    out: dict[str, float] = {}
    for q in questions:
        targets = torch.tensor([r[_LABEL[q.id]] for r in rows])
        p = probs[q.id]
        out[f"{q.id}_acc"] = (p.argmax(-1) == targets).float().mean().item()
        out[f"{q.id}_ece"] = expected_calibration_error(p, targets)
    return out
