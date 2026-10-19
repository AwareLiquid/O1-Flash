"""Architecture diagnostic: byte-level GRU baseline vs O1-Flash on banking77.

The O1-Flash reaches ~40.8% (200K steps) / 37.9% (20K) on banking77 with the
SAME byte-level input. This script trains a plain GRU encoder on identical
data / objective (Brier + decision CE) / budget, to isolate whether the
remaining gap is the ENCODER ARCHITECTURE (liquid recurrence) or the
byte-level input + data coverage.

Run:
    python -m benchmarks.baseline_gru_bank77 --steps 20000 --d 768
"""
from __future__ import annotations

import argparse
import math
import random

import torch
import torch.nn as nn

from benchmarks.real_data_bench import fit_temperature, load_banking77
from mt_flash.calibration import calibration_step, expected_calibration_error
from mt_flash.state_tokenizer import encode


class GRUBank(nn.Module):
    def __init__(self, d: int = 768, n_classes: int = 77, vocab: int = 260):
        super().__init__()
        self.emb = nn.Embedding(vocab, d, padding_idx=256)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.head = nn.Linear(d, n_classes)

    def forward(self, ids, mask):
        h = self.emb(ids)
        out, _ = self.gru(h)
        m = mask.unsqueeze(-1).float()
        pooled = (out * m).sum(1) / m.sum(1).clamp_min(1.0)
        return self.head(pooled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train, test, cats = load_banking77()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model = GRUBank(args.d, len(cats))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"=== GRU baseline (byte-level, d={args.d}, {n_params/1e6:.2f}M params) "
          f"vs O1-Flash refs: 20K 37.9% / 200K 40.8% ===", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    for step in range(args.steps):
        if step < args.warmup:
            cur = args.lr * (step + 1) / max(1, args.warmup)
        else:
            prog = (step - args.warmup) / max(1, args.steps - args.warmup)
            cur = args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))
        for g in opt.param_groups:
            g["lr"] = cur
        rows = rng.sample(train, args.batch)
        ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(encode(t, 4096)) for t, _ in rows],
            batch_first=True, padding_value=256)
        mask = ids != 256
        targets = torch.tensor([y for _, y in rows])
        logits = model(ids, mask)
        probs = torch.softmax(logits, dim=-1)
        loss = calibration_step(probs, targets)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 2000 == 0:
            print(f"  step {step+1} loss {loss.item():.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(encode(t, 4096)) for t, _ in test],
            batch_first=True, padding_value=256)
        mask = ids != 256
        logits = model(ids, mask)
        probs = torch.softmax(logits, -1)
    y = torch.tensor([r[1] for r in test])
    acc = (probs.argmax(-1) == y).float().mean().item()
    half = len(y) // 2
    ece_before = expected_calibration_error(probs[half:], y[half:])
    T = fit_temperature(probs[:half], y[:half])
    cal = torch.softmax(torch.log(probs[half:].clamp_min(1e-9)) / T, -1)
    ece_after = expected_calibration_error(cal, y[half:])
    print(f"\n=== GRU baseline result ({args.steps} steps) ===")
    print(f"params {n_params/1e6:.2f}M  acc {acc*100:.2f}%  "
          f"ECE {ece_before:.3f} -> {ece_after:.3f} (T={T:.2f})")
    print(f"O1-Flash refs: 20K 37.9% / 200K 40.8%")


if __name__ == "__main__":
    main()
