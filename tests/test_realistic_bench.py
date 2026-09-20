"""Realistic benchmark test: held-out paraphrase generalization above
chance (the honest baseline; not a quality claim)."""

from benchmarks.realistic_bench import train_realistic
from mt_flash.config import o1_flash_tiny
from mt_flash.model import O1Flash


def test_realistic_bench_generalizes_above_chance():
    import torch
    torch.manual_seed(0)          # deterministic init (see variance note)
    model = O1Flash(o1_flash_tiny())
    m = train_realistic(model, steps=350, batch=32)
    # chance: dept 0.25 (4-way), sev 0.25 (4-way), urgent 0.5. The
    # assertion is "learns on paraphrases": >2σ above chance on the
    # held-out set (σ ≈ 0.027 at n=256), NOT a quality claim.
    assert m["dept_acc"] > 0.30
    assert m["sev_acc"] > 0.30
    assert m["urgent_acc"] > 0.52
    for k in ("dept", "sev", "urgent"):
        assert 0.0 <= m[f"{k}_ece"] <= 1.0

