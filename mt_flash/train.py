"""Training reference: synthetic decision tasks + calibration loop.

Trains ONE typed question per call (fresh model recommended per
question), with the calibration objective (Brier + decision CE) and ECE
reporting. This proves the PIPELINE — state encode once, parallel heads,
calibrated loss, ECE report — end to end.

Why single-question training (measured, 2026-09-18, see DESIGN §7):
the liquid recurrent encoder learns a single decision signal to ceiling
(acc 1.0 on the synthetic set), but JOINT multi-question training on the
shared liquid state fails to extract earlier signals (dept 0.27 vs 0.69
for a plain mean-pool MLP baseline under the same budget). Multi-task
joint training on the liquid core is the repo's top open research
problem; single-question training + parallel inference remains fully
functional — the "parallel" property is an inference property.

The synthetic tasks are deliberately linear-separable: the point is the
pipeline, not decision skill.
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn

from .calibration import calibration_step, expected_calibration_error
from .model import O1Flash
from .schema import Question

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
    from .schema import (ChoiceQuestion, ProbabilityQuestion, ScoreQuestion)
    if kind == "dept":
        return ChoiceQuestion(id="dept", options=DEPT_OPTIONS)
    if kind == "sev":
        return ScoreQuestion(id="sev", levels=SEV_LEVELS)
    if kind == "urgent":
        return ProbabilityQuestion(id="urgent", prompt="needs escalation now")
    raise ValueError(f"unknown kind: {kind}")


def _label_idx(kind: str, row: tuple[str, int, int, int]) -> int:
    return {"dept": 1, "sev": 2, "urgent": 3}[kind]


def train_single(model: O1Flash, question: Question, steps: int = 400,
                 batch: int = 32, lr: float = 3e-3, seed: int = 0,
                 log_every: int | None = None) -> dict[str, float]:
    """Train ONE typed question on synthetic states; returns acc + ECE."""
    kind = question.id
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
        probs = model.heads.train_forward(y, [question], pad_mask=pad_mask)
        targets = torch.tensor([r[_label_idx(kind, r)] for r in rows])
        loss = calibration_step(probs[kind], targets)
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
        probs = model.heads.train_forward(y, [question], pad_mask=pad_mask)
    targets = torch.tensor([r[_label_idx(kind, r)] for r in rows])
    p = probs[kind]
    acc = (p.argmax(-1) == targets).float().mean().item()
    return {f"{kind}_acc": acc,
            f"{kind}_ece": expected_calibration_error(p, targets)}
