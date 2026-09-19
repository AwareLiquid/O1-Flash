"""Calibration reference tests: Brier/ECE/decision-CE are correct on
hand-computable cases."""

import torch

from mt_flash.calibration import (brier_loss, calibration_step,
                                  decision_ce_loss,
                                  expected_calibration_error)


def test_brier_perfect_prediction_is_zero():
    probs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([0, 1])
    onehot = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert brier_loss(probs, onehot).item() == 0.0


def test_brier_known_value():
    probs = torch.tensor([[0.6, 0.4]])
    onehot = torch.tensor([[1.0, 0.0]])
    # (0.6-1)^2 + (0.4-0)^2 = 0.16 + 0.16 = 0.32, mean over all elems = 0.16
    assert abs(brier_loss(probs, onehot).item() - 0.16) < 1e-6


def test_ece_perfectly_calibrated_is_zero():
    # confident group: conf ~0.92, 9/10 correct; unconfident: conf ~0, 0/10
    probs = torch.tensor([[0.99, 0.01]] * 10 + [[0.5, 0.5]] * 10)
    targets = torch.tensor([0] * 9 + [1] + [1] * 10)
    ece = expected_calibration_error(probs, targets, n_bins=10)
    assert ece < 0.2


def test_ece_penalises_overconfidence():
    # confident AND wrong -> large calibration gap
    probs = torch.tensor([[0.99, 0.01]] * 4)
    targets = torch.tensor([1, 1, 1, 1])   # all wrong
    ece = expected_calibration_error(probs, targets, n_bins=10)
    assert ece > 0.8


def test_calibration_step_finite_and_backwardable():
    logits = torch.rand(4, 3, requires_grad=True)
    probs = torch.softmax(logits, dim=-1)
    targets = torch.tensor([0, 2, 1, 0])
    loss = calibration_step(probs, targets)
    assert torch.isfinite(loss).item()
    loss.backward()
    assert logits.grad is not None
