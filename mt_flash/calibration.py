"""Calibrated-confidence training reference (RLCD-inspired).

Jev's RLCD (Reinforcement Learning for Calibrated Decisions) optimises for
probabilities that match outcomes: across many predictions, a 90% answer
should be right about 90% of the time. This module provides the open
reference objective for the same property:

- ``brier_loss``      proper scoring rule over the predicted distribution
- ``expected_calibration_error``   the standard ECE metric (with target bin
                      assignment by argmax — the "decision ECE" flavour)
- ``decision_ce_loss`` cross-entropy on the argmax target (the accuracy term)

A training step minimises brier + decision CE jointly. ECE is the report
metric, not the loss. No RL here: it is a reference recipe, honest about
that difference from TypeSafe's unpublished method.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def brier_loss(probs: torch.Tensor, target_onehot: torch.Tensor) -> torch.Tensor:
    """Mean squared error between distribution and one-hot outcome."""
    return torch.mean((probs - target_onehot) ** 2)


def decision_ce_loss(probs: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(probs, target_idx)


def expected_calibration_error(probs: torch.Tensor, target_idx: torch.Tensor,
                               n_bins: int = 10) -> float:
    """ECE on decision tasks: confidence = peakedness, correctness = argmax."""
    conf = 1.0 - torch.special.entr(probs).sum(dim=-1) / torch.log(
        torch.tensor(probs.shape[-1], dtype=probs.dtype, device=probs.device)
    )
    correct = (probs.argmax(dim=-1) == target_idx).float()
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean().item()
        avg_conf = conf[mask].mean().item()
        ece += (mask.sum().item() / probs.shape[0]) * abs(acc - avg_conf)
    return ece


def calibration_step(probs: torch.Tensor, target_idx: torch.Tensor,
                     brier_weight: float = 1.0) -> torch.Tensor:
    """Combined training objective; ECE is tracked separately, not trained on."""
    onehot = F.one_hot(target_idx, num_classes=probs.shape[-1]).float()
    return brier_weight * brier_loss(probs, onehot) \
        + decision_ce_loss(probs, target_idx)
