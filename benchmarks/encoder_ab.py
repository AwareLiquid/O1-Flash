"""Encoder A/B, controlled: liquid core vs GRU with the SAME readout + budget.

Both encoders feed the same mean-pool + linear 77-way head and train for the
same steps on the same banking77 batches (byte tokens, T<=64):

  A. LiquidStateEncoder (O1-Flash's liquid core, 1.18M)  + mean-pool head
  B. GRU encoder (d=448, ~1.35M)                          + mean-pool head

Reference points: full O1-Flash (typed heads) 200K steps = 40.8%;
GRU-baseline (d=512, mean-pool) 6000 steps = 79.06%.

Run:
    python -m benchmarks.encoder_ab --steps 2000
"""
from __future__ import annotations

import argparse
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.real_data_bench import fit_temperature, load_banking77
from mt_flash.calibration import calibration_step, expected_calibration_error
from mt_flash.config import FlashConfig
from mt_flash.liquid_core import LiquidStateEncoder
from mt_flash.state_tokenizer import encode

MAX_TOK = 64


class MeanHead(nn.Module):
    def __init__(self, d, n=77):
        super().__init__()
        self.fc = nn.Linear(d, n)

    def forward(self, y, mask):
        m = mask.unsqueeze(-1).float()
        pooled = (y * m).sum(1) / m.sum(1).clamp_min(1.0)
        return self.fc(pooled)


class LiquidModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = LiquidStateEncoder(FlashConfig())
        self.head = MeanHead(768)

    def forward(self, ids, mask):
        y, _ = self.enc(ids)
        return self.head(y, mask)


class GRUModel(nn.Module):
    def __init__(self, d=448):
        super().__init__()
        self.emb = nn.Embedding(260, d, padding_idx=256)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.head = MeanHead(d)

    def forward(self, ids, mask):
        y, _ = self.gru(self.emb(ids))
        return self.head(y, mask)


def batch_rows(rows):
    ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(encode(t, MAX_TOK), dtype=torch.long) for t, _ in rows],
        batch_first=True, padding_value=256)
    if ids.shape[1] < MAX_TOK:
        ids = F.pad(ids, (0, MAX_TOK - ids.shape[1]), value=256)
    return ids, ids != 256, torch.tensor([y for _, y in rows])


@torch.no_grad()
def evaluate(model, test):
    model.eval()
    ids, mask, y = batch_rows(test)
    probs = torch.softmax(model(ids, mask), -1)
    acc = (probs.argmax(-1) == y).float().mean().item()
    half = len(y) // 2
    ece_b = expected_calibration_error(probs[half:], y[half:])
    T = fit_temperature(probs[:half], y[:half])
    cal = torch.softmax(torch.log(probs[half:].clamp_min(1e-9)) / T, -1)
    ece_a = expected_calibration_error(cal, y[half:])
    return acc, ece_b, ece_a, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train, test, cats = load_banking77()
    results = {}
    for name, ctor in (("liquid", LiquidModel), ("gru448", GRUModel)):
        torch.manual_seed(args.seed)
        rng = random.Random(args.seed)
        model = ctor()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"=== {name} ({n_params/1e6:.2f}M params), {args.steps} steps ===",
              flush=True)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        model.train()
        t0 = time.time()
        for step in range(args.steps):
            if step < args.warmup:
                cur = args.lr * (step + 1) / max(1, args.warmup)
            else:
                prog = (step - args.warmup) / max(1, args.steps - args.warmup)
                cur = args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))
            for g in opt.param_groups:
                g["lr"] = cur
            rows = rng.sample(train, args.batch)
            ids, mask, y = batch_rows(rows)
            logits = model(ids, mask)
            loss = calibration_step(torch.softmax(logits, -1), y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if (step + 1) % 500 == 0:
                print(f"  step {step+1} loss {loss.item():.4f}", flush=True)
        acc, ece_b, ece_a, T = evaluate(model, test)
        results[name] = (acc, ece_b, ece_a, T, n_params, time.time() - t0)
        print(f"  [{name}] acc {acc*100:.2f}%  ECE {ece_b:.3f}->{ece_a:.3f} "
              f"(T={T:.2f})  {time.time()-t0:.0f}s", flush=True)

    print("\n=== encoder A/B (same mean-pool readout, same budget) ===")
    for k, (acc, eb, ea, T, n, sec) in results.items():
        print(f"{k:8s}: {acc*100:.2f}%  params {n/1e6:.2f}M  ({sec:.0f}s)")
    print("\nrefs: O1-Flash typed-heads 200K = 40.8%; GRU d512 mean-pool 6K = 79.06%")


if __name__ == "__main__":
    main()
