"""Realistic decision benchmark: paraphrase-level generalization.

Upgrades the synthetic set in two ways:
1. States are generated from per-class PHRASE BANKS (realistic ticket
   language), so the label is carried by semantics — not by a fixed
   "dept=X" keyword string.
2. Distractor phrases from OTHER classes are injected, so the model must
   attend to the majority signal rather than memorise one marker.

Evaluation is on HELD-OUT phrase combinations: the training and eval sets
draw different combinations from the same banks, so a good score means
the model mapped phrase→class, not position→label. Honest scope: the
banks are still synthetic templates — true out-of-vocabulary
generalisation needs real data (HANDOFF todo).

Run: python -m benchmarks.realistic_bench
"""

from __future__ import annotations

import random

import torch

from mt_flash.config import o1_flash_tiny
from mt_flash.model import O1Flash
from mt_flash.schema import (ChoiceQuestion, ProbabilityQuestion,
                             ScoreQuestion)
from mt_flash.train import train

DEPT_BANKS = {
    "billing": ["payment went through twice", "invoice shows wrong amount",
                "charged but subscription never activated",
                "refund has not arrived", "billing cycle amount differs",
                "card was declined unexpectedly",
                "receipt missing from account", "plan price changed mid-term"],
    "infra": ["server down since early morning", "latency spikes every hour",
              "database connection refused", "deploy rolled back",
              "outage in the eu region", "disk full on the worker node",
              "dns resolution failing", "queue backlog is growing"],
    "sales": ["asked for a custom quote", "wants a product demo call",
              "pricing question about the pro tier", "trial is ending",
              "upgrade path from the basic plan", "license terms question",
              "enterprise tier features", "renewal discount request"],
    "other": ["password reset link is broken", "profile photo won't upload",
              "notification settings not saving", "language switch not working",
              "export to csv button missing", "dark mode keeps resetting",
              "avatar sync across devices", "email preferences not stored"],
}

SEV_BANKS = {
    "low": ["cosmetic issue", "minor inconvenience", "nice to have",
            "not blocking anything", "typo on the page", "cosmetic glitch"],
    "mid": ["workaround exists", "some users affected", "intermittent failure",
            "needs a workaround", "partially degraded", "sporadic errors"],
    "high": ["completely down", "data loss reported", "security concern",
             "all users affected", "revenue blocked", "total outage"],
    "critical": ["production down", "customer data exposed", "zero service",
                 "irreversible data loss", "every region down",
                 "cannot process any payments"],
}

URGENT_YES = ["escalate immediately", "ceo is waiting", "severe impact",
              "needs attention now", "blocking the launch",
              "has been down for hours"]
URGENT_NO = ["no rush", "can wait until tomorrow", "low priority",
             "sometime next week", "whenever possible", "not urgent"]


def _rng_for(seed: int) -> random.Random:
    return random.Random(seed)


def make_realistic_state(rng: random.Random
                         ) -> tuple[str, int, int, int]:
    """One realistic ticket: dept + sev + urgent with distractor phrases."""
    dept = rng.randrange(4)
    dept_name = list(DEPT_BANKS)[dept]
    sev = rng.randrange(4)
    sev_name = list(SEV_BANKS)[sev]
    urgent = int(rng.random() < 0.8) if sev >= 2 else int(rng.random() < 0.3)

    # 2-3 signal phrases from the true class + 1 distractor from another
    n_signal = rng.randint(2, 3)
    signals = rng.sample(DEPT_BANKS[dept_name], n_signal)
    distractor_dept = rng.choice([d for d in DEPT_BANKS if d != dept_name])
    distractor = rng.choice(DEPT_BANKS[distractor_dept])
    sev_phrase = rng.choice(SEV_BANKS[sev_name])
    urg_phrase = rng.choice(URGENT_YES if urgent else URGENT_NO)

    parts = signals + [distractor, f"severity note: {sev_phrase}",
                       f"request: {urg_phrase}"]
    rng.shuffle(parts)
    return ("ticket: " + " ; ".join(parts), dept, sev, urgent)


def train_realistic(model: O1Flash, steps: int = 500, batch: int = 32,
                    seed: int = 0) -> dict[str, float]:
    """Train on generated combinations; evaluate on HELD-OUT combinations."""
    questions = [
        ChoiceQuestion(id="dept", options=tuple(DEPT_BANKS)),
        ScoreQuestion(id="sev", levels=tuple(SEV_BANKS)),
        ProbabilityQuestion(id="urgent", prompt="needs escalation now"),
    ]
    rng = _rng_for(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    import torch.nn as nn
    from mt_flash.calibration import calibration_step

    for step in range(steps):
        rows = [make_realistic_state(rng) for _ in range(batch)]
        ids = torch.nn.utils.rnn.pad_sequence(
            [model._ids_for(r[0]) for r in rows],
            batch_first=True, padding_value=256)
        y = model(ids)
        mask = ids != 256
        probs = model.heads.train_forward(y, questions, pad_mask=mask)
        loss = (calibration_step(probs["dept"], torch.tensor([r[1] for r in rows]))
                + calibration_step(probs["sev"], torch.tensor([r[2] for r in rows]))
                + calibration_step(probs["urgent"], torch.tensor([r[3] for r in rows])))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    from mt_flash.calibration import expected_calibration_error
    # held-out combinations: different rng stream (1000+)
    rows = [make_realistic_state(_rng_for(1000 + i)) for i in range(256)]
    ids = torch.nn.utils.rnn.pad_sequence(
        [model._ids_for(r[0]) for r in rows],
        batch_first=True, padding_value=256)
    with torch.no_grad():
        y = model(ids)
        mask = ids != 256
        probs = model.heads.train_forward(y, questions, pad_mask=mask)
    out = {}
    for q, idx in (("dept", 1), ("sev", 2), ("urgent", 3)):
        targets = torch.tensor([r[idx] for r in rows])
        p = probs[q]
        out[f"{q}_acc"] = (p.argmax(-1) == targets).float().mean().item()
        out[f"{q}_ece"] = expected_calibration_error(p, targets)
    return out


if __name__ == "__main__":
    model = O1Flash(o1_flash_tiny())
    print("training on paraphrase combinations (500 steps)...")
    m = train_realistic(model, steps=500)
    print("\n=== realistic bench (held-out combinations, honest scope) ===")
    print(f"{'question':8s} {'acc':>7s}  {'ece':>7s}")
    for kind in ("dept", "sev", "urgent"):
        print(f"{kind:8s} {m[f'{kind}_acc']:7.3f}  {m[f'{kind}_ece']:7.3f}")
