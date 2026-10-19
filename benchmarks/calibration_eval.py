"""Calibration evaluation of the saved O1-Flash banking77 checkpoint.

Loads `checkpoints/o1flash_banking77_200k_sched.pt` and runs the official
banking77 test split through the model, then applies post-hoc temperature
scaling (fit on the calibration half, evaluated on the other half) — the same
protocol as `benchmarks/real_data_bench.py`, but on the CHECKPOINT instead of
a fresh training run.

Reports: accuracy, ECE before/after calibration, fitted temperature.

Run:
    python -m benchmarks.calibration_eval
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.real_data_bench import fit_temperature, load_banking77  # noqa: E402
from mt_flash.calibration import expected_calibration_error  # noqa: E402
from mt_flash.config import FlashConfig  # noqa: E402
from mt_flash.model import O1Flash  # noqa: E402
from mt_flash.schema import ChoiceQuestion  # noqa: E402

CKPT = Path(__file__).resolve().parent.parent / "checkpoints" / "o1flash_banking77_200k_sched.pt"


@torch.no_grad()
def model_probs(model, texts, q, device="cpu", bsz=64):
    outs = []
    for i in range(0, len(texts), bsz):
        chunk = texts[i:i + bsz]
        ids = torch.nn.utils.rnn.pad_sequence(
            [model._ids_for(t) for t in chunk], batch_first=True, padding_value=256).to(device)
        mask = ids != 256
        y = model(ids)
        outs.append(model.heads.train_forward(y, [q], pad_mask=mask)["intent"].cpu())
    return torch.cat(outs, 0)


def main():
    st = torch.load(CKPT, map_location="cpu", weights_only=False)
    cats = st["cats"]
    model = O1Flash(FlashConfig())
    model.load_state_dict(st["model"], strict=True)
    model.eval()
    print(f"checkpoint: {CKPT.name}  cats {len(cats)}  "
          f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    train, test, cats2 = load_banking77()
    assert cats == cats2
    q = ChoiceQuestion(id="intent", options=tuple(cats))
    texts = [r[0] for r in test]
    y = torch.tensor([r[1] for r in test])
    probs = model_probs(model, texts, q)
    acc = (probs.argmax(-1) == y).float().mean().item()

    half = len(y) // 2
    cal_p, cal_t = probs[:half], y[:half]
    ev_p, ev_t = probs[half:], y[half:]
    ece_before = expected_calibration_error(ev_p, ev_t)
    T = fit_temperature(cal_p, cal_t)
    ev_cal = F.softmax(torch.log(ev_p.clamp_min(1e-9)) / T, dim=-1)
    ece_after = expected_calibration_error(ev_cal, ev_t)
    print(f"\n=== banking77 checkpoint calibration (77-way, chance 1.3%) ===")
    print(f"acc {acc:.4f}  n_test {len(y)}")
    print(f"ECE  before {ece_before:.3f}  ->  after {ece_after:.3f}   (T = {T:.3f})")
    print(f"ECE reduction: {(1 - ece_after / max(ece_before, 1e-9))*100:.1f}%")


if __name__ == "__main__":
    main()
