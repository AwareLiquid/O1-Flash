"""Hybrid A/B: does the liquid O(1) state add value ON TOP of a GRU?

Controlled protocol (2000 steps each, same data / head family):
  - gru      : GRU(d=256) over byte embeddings -> mean-pool -> linear 77
  - liquid   : frozen O1-Flash encoder features (768) -> mean-pool -> linear 77
  - hybrid   : concat[ frozen liquid 768 , GRU 256 ] -> linear 77

The liquid features come from the 200K checkpoint's encoder (frozen). If the
hybrid beats the GRU, the liquid state carries complementary information and
the hybrid architecture is worth building; if not, the GRU dominates and the
liquid core adds nothing to this decision task.

Run:
    python -m benchmarks.hybrid_ab --steps 2000
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
from mt_flash.model import O1Flash
from mt_flash.state_tokenizer import encode

MAX_TOK = 64
CKPT = "checkpoints/o1flash_banking77_200k_sched.pt"


def batch_ids(rows):
    ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(encode(t, MAX_TOK), dtype=torch.long) for t, _ in rows],
        batch_first=True, padding_value=256)
    if ids.shape[1] < MAX_TOK:
        ids = F.pad(ids, (0, MAX_TOK - ids.shape[1]), value=256)
    return ids


@torch.no_grad()
def liquid_features(enc, texts, bsz=32):
    """Mean-pooled frozen liquid features (N, 768) in fp16."""
    out = []
    for i in range(0, len(texts), bsz):
        rows = [(t, 0) for t in texts[i:i + bsz]]
        ids = batch_ids(rows)
        y, _ = enc(ids)
        m = (ids != 256).unsqueeze(-1).float()
        out.append(((y * m).sum(1) / m.sum(1).clamp_min(1.0)).half())
    return torch.cat(out, 0)


class GRUHead(nn.Module):
    def __init__(self, d=256, n=77):
        super().__init__()
        self.emb = nn.Embedding(260, d, padding_idx=256)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.fc = nn.Linear(d, n)

    def forward(self, ids, mask, extra=None):
        y, _ = self.gru(self.emb(ids))
        m = mask.unsqueeze(-1).float()
        pooled = (y * m).sum(1) / m.sum(1).clamp_min(1.0)
        if extra is not None:
            pooled = torch.cat([pooled, extra.float()], dim=-1)
        return self.fc(pooled)


class HybridHead(nn.Module):
    def __init__(self, d=256, n=77, extra=768):
        super().__init__()
        self.emb = nn.Embedding(260, d, padding_idx=256)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.fc = nn.Linear(d + extra, n)

    def forward(self, ids, mask, extra):
        y, _ = self.gru(self.emb(ids))
        m = mask.unsqueeze(-1).float()
        pooled = (y * m).sum(1) / m.sum(1).clamp_min(1.0)
        return self.fc(torch.cat([pooled, extra.float()], dim=-1))


def train_eval(kind, train, test, tr_liq, te_liq, steps, lr, seed):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    if kind == "gru":
        model = GRUHead(256)
    elif kind == "liquid":
        model = nn.Linear(768, 77)
    elif kind == "hybrid":
        model = HybridHead(256, 77, 768)
    else:
        raise ValueError(kind)
    print(f"=== {kind} ({sum(p.numel() for p in model.parameters())/1e6:.2f}M) ===",
          flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    t0 = time.time()
    for step in range(steps):
        warm, prog = 300, min(1.0, max(0.0, (step - 300) / max(1, steps - 300)))
        cur = lr * ((step + 1) / warm if step < warm else
                    0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))
        for g in opt.param_groups:
            g["lr"] = cur
        idx = [rng.randrange(len(train)) for _ in range(32)]
        rows = [train[i] for i in idx]
        ids = batch_ids(rows)
        mask = ids != 256
        y = torch.tensor([r[1] for r in rows])
        extra = tr_liq[idx]
        if kind == "liquid":
            logits = model(extra.float())
        elif kind == "hybrid":
            logits = model(ids, mask, extra)
        else:
            logits = model(ids, mask)
        loss = calibration_step(torch.softmax(logits, -1), y)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 500 == 0:
            print(f"  step {step+1} loss {loss.item():.4f}", flush=True)

    ids = batch_ids(test)
    mask = ids != 256
    y = torch.tensor([r[1] for r in test])
    with torch.no_grad():
        if kind == "liquid":
            logits = model(te_liq.float())
        elif kind == "hybrid":
            logits = model(ids, mask, te_liq)
        else:
            logits = model(ids, mask)
        probs = torch.softmax(logits, -1)
    acc = (probs.argmax(-1) == y).float().mean().item()
    half = len(y) // 2
    ece_b = expected_calibration_error(probs[half:], y[half:])
    T = fit_temperature(probs[:half], y[:half])
    cal = torch.softmax(torch.log(probs[half:].clamp_min(1e-9)) / T, -1)
    ece_a = expected_calibration_error(cal, y[half:])
    print(f"  [{kind}] acc {acc*100:.2f}%  ECE {ece_b:.3f}->{ece_a:.3f} (T={T:.2f}) "
          f"{time.time()-t0:.0f}s", flush=True)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    st = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = O1Flash(FlashConfig())
    model.load_state_dict(st["model"], strict=True)
    enc = model.encoder.eval()
    for p in enc.parameters():
        p.requires_grad_(False)

    train, test, cats = load_banking77()
    print("precomputing frozen liquid features...", flush=True)
    tr_liq = liquid_features(enc, [r[0] for r in train])
    te_liq = liquid_features(enc, [r[0] for r in test])
    print(f"features {tuple(tr_liq.shape)} / {tuple(te_liq.shape)}", flush=True)

    results = {}
    for kind in ("liquid", "gru", "hybrid"):
        results[kind] = train_eval(kind, train, test, tr_liq, te_liq,
                                   args.steps, args.lr, args.seed)
    print("\n=== hybrid A/B (2000 steps each) ===")
    for k, v in results.items():
        print(f"{k:8s}: {v*100:.2f}%")
    print(f"liquid adds {((results['hybrid']-results['gru'])*100):+.1f} pts over GRU")
    print("VERDICT:", "LIQUID ADDS VALUE" if results["hybrid"] > results["gru"] + 0.02
          else "GRU DOMINATES (liquid adds nothing)")


if __name__ == "__main__":
    main()
