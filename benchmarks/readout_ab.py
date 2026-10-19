"""Readout A/B on the frozen O1-Flash encoder (banking77).

The 200K checkpoint's typed heads reach 40.55%; a byte-level GRU with a
mean-pool + linear head reaches 79% with far less compute. This script freezes
the checkpoint's ENCODER and swaps the readout only:

  A. mean-pool over positions + linear 77-way head
  B. attention-pool (learned query) + linear 77-way head

If A/B jump well above 40.55%, the typed-head readout is the bottleneck; if
they stay near it, the encoder features are.

Run:
    python -m benchmarks.readout_ab
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.real_data_bench import fit_temperature, load_banking77
from mt_flash.calibration import expected_calibration_error
from mt_flash.config import FlashConfig
from mt_flash.model import O1Flash
from mt_flash.state_tokenizer import encode as byte_encode

CKPT = Path(__file__).resolve().parent.parent / "checkpoints" / "o1flash_banking77_200k_sched.pt"


MAX_TOK = 64

def encode_all(encoder, model, texts, bsz=32):
    feats, hids = [], []
    with torch.no_grad():
        for i in range(0, len(texts), bsz):
            chunk = texts[i:i + bsz]
            ids = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(byte_encode(t, MAX_TOK), dtype=torch.long) for t in chunk],
                batch_first=True, padding_value=256)
            if ids.shape[1] < MAX_TOK:
                ids = torch.nn.functional.pad(ids, (0, MAX_TOK - ids.shape[1]), value=256)
            y, _ = encoder(ids)
            feats.append(y.cpu().half())
            hids.append((ids != 256).cpu())
    return feats, hids


class MeanPoolHead(nn.Module):
    def __init__(self, d, n=77):
        super().__init__()
        self.fc = nn.Linear(d, n)

    def forward(self, y, mask):
        y = y.float()
        m = mask.unsqueeze(-1).float()
        pooled = (y * m).sum(1) / m.sum(1).clamp_min(1.0)
        return self.fc(pooled)


class AttnPoolHead(nn.Module):
    def __init__(self, d, n=77):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d) * 0.02)
        self.fc = nn.Linear(d, n)

    def forward(self, y, mask):
        y = y.float()
        scores = (y @ self.q) / (y.shape[-1] ** 0.5)
        scores = scores.masked_fill(~mask, -1e9)
        w = torch.softmax(scores, 1).unsqueeze(-1)
        pooled = (y * w).sum(1)
        return self.fc(pooled)


def evaluate(head, feats, hids, y):
    head.eval()
    with torch.no_grad():
        logits = head(feats, hids)
        probs = torch.softmax(logits, -1)
    acc = (probs.argmax(-1) == y).float().mean().item()
    half = len(y) // 2
    ece_b = expected_calibration_error(probs[half:], y[half:])
    T = fit_temperature(probs[:half], y[:half])
    cal = torch.softmax(torch.log(probs[half:].clamp_min(1e-9)) / T, -1)
    ece_a = expected_calibration_error(cal, y[half:])
    return acc, ece_b, ece_a, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--bsz", type=int, default=64)
    args = ap.parse_args()

    st = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = O1Flash(FlashConfig())
    model.load_state_dict(st["model"], strict=True)
    model.eval()
    enc = model.encoder
    for p in enc.parameters():
        p.requires_grad_(False)

    train, test, cats = load_banking77()
    tr_texts = [r[0] for r in train]
    tr_y = torch.tensor([r[1] for r in train])
    te_texts = [r[0] for r in test]
    te_y = torch.tensor([r[1] for r in test])

    print("encoding train/test with the frozen encoder...", flush=True)
    tr_f, tr_m = encode_all(enc, model, tr_texts)
    te_f, te_m = encode_all(enc, model, te_texts)
    tr_f, tr_m = torch.cat(tr_f, 0), torch.cat(tr_m, 0)
    te_f, te_m = torch.cat(te_f, 0), torch.cat(te_m, 0)
    d = tr_f.shape[-1]
    print(f"encoder output dim {d}  (typed-head ref: acc 40.55%)", flush=True)

    results = {}
    for name, head in (("mean_pool", MeanPoolHead(d)),
                       ("attn_pool", AttnPoolHead(d))):
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
        n_tr = len(tr_y)
        for step in range(args.steps):
            idx = torch.randint(0, n_tr, (args.bsz,))
            logits = head(tr_f[idx], tr_m[idx])
            loss = F.cross_entropy(logits, tr_y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if (step + 1) % 200 == 0:
                print(f"  [{name}] step {step+1} loss {loss.item():.4f}", flush=True)
        acc, ece_b, ece_a, T = evaluate(head, te_f, te_m, te_y)
        results[name] = (acc, ece_b, ece_a, T)
        print(f"  [{name}] acc {acc*100:.2f}%  ECE {ece_b:.3f} -> {ece_a:.3f} (T={T:.2f})",
              flush=True)

    print("\n=== readout A/B (frozen encoder) ===")
    print(f"typed heads (200K ckpt): 40.55%")
    for k, (acc, eb, ea, T) in results.items():
        print(f"{k:10s}: {acc*100:.2f}%  (ECE {eb:.3f} -> {ea:.3f}, T={T:.2f})")
    best = max(a for a, *_ in results.values())
    print(f"\nVERDICT: {'READOUT IS THE BOTTLENECK' if best > 0.50 else 'ENCODER FEATURES ARE THE BOTTLENECK'}"
          f" (best readout {best*100:.1f}% vs typed 40.6%)")


if __name__ == "__main__":
    main()
