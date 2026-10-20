"""Decision v2 — GRU-based decision model, trained + calibrated on banking77.

Closes the architecture diagnosis: the O1-Flash liquid encoder caps at ~40.8%
(200K steps); a byte-level GRU reaches ~79% at 6K steps. This trains and SAVES
the v2 checkpoint (encoder + head + calibration temperature) so a decision
service can load it directly.

Artifact: checkpoints/decision_v2_gru.pt
  {model: state_dict, cats: [...], temperature: float, metrics: {...}}

Run:
    python -m benchmarks.train_decision_v2 --steps 6000 --d 512
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.real_data_bench import fit_temperature, load_banking77
from mt_flash.calibration import calibration_step, expected_calibration_error
from mt_flash.state_tokenizer import encode

HERE = Path(__file__).resolve().parent.parent
MAX_TOK = 64


class DecisionV2(nn.Module):
    """Byte-level GRU encoder + mean-pool + linear typed head."""

    def __init__(self, d: int = 512, n_classes: int = 77, vocab: int = 260):
        super().__init__()
        self.emb = nn.Embedding(vocab, d, padding_idx=256)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.head = nn.Linear(d, n_classes)

    def forward(self, ids, mask):
        y, _ = self.gru(self.emb(ids))
        m = mask.unsqueeze(-1).float()
        pooled = (y * m).sum(1) / m.sum(1).clamp_min(1.0)
        return self.head(pooled)


def batch(texts, max_tok=MAX_TOK):
    ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(encode(t, max_tok), dtype=torch.long) for t in texts],
        batch_first=True, padding_value=256)
    if ids.shape[1] < max_tok:
        ids = F.pad(ids, (0, max_tok - ids.shape[1]), value=256)
    return ids, ids != 256


@torch.no_grad()
def predict_probs(model, texts, temperature=1.0, bsz=64):
    model.eval()
    outs = []
    for i in range(0, len(texts), bsz):
        ids, mask = batch(texts[i:i + bsz])
        logits = model(ids, mask)
        if temperature != 1.0:
            logits = logits / temperature
        outs.append(torch.softmax(logits, -1))
    return torch.cat(outs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "checkpoints" / "decision_v2_gru.pt"))
    args = ap.parse_args()

    train, test, cats = load_banking77()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model = DecisionV2(args.d, len(cats))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"=== decision v2 (GRU d={args.d}, {n_params/1e6:.2f}M) {args.steps} steps ===",
          flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    t0 = time.time()
    for step in range(args.steps):
        if step < args.warmup:
            cur = args.lr * (step + 1) / max(1, args.warmup)
        else:
            prog = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
            cur = args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))
        for g in opt.param_groups:
            g["lr"] = cur
        rows = rng.sample(train, args.batch)
        ids, mask = batch([r[0] for r in rows])
        y = torch.tensor([r[1] for r in rows])
        probs = torch.softmax(model(ids, mask), -1)
        loss = calibration_step(probs, y)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 1000 == 0:
            print(f"  step {step+1} loss {loss.item():.4f}", flush=True)

    # -- calibration on the first half of the test split, eval on the second --
    texts = [r[0] for r in test]
    y = torch.tensor([r[1] for r in test])
    probs = predict_probs(model, texts)
    acc = (probs.argmax(-1) == y).float().mean().item()
    half = len(y) // 2
    ece_b = expected_calibration_error(probs[half:], y[half:])
    T = fit_temperature(probs[:half], y[:half])
    cal = torch.softmax(torch.log(probs[half:].clamp_min(1e-9)) / T, -1)
    ece_a = expected_calibration_error(cal, y[half:])
    metrics = {"acc": acc, "ece": ece_b, "ece_calibrated": ece_a,
               "temperature": T, "n_params": n_params, "steps": args.steps,
               "d": args.d, "train_sec": round(time.time() - t0, 1)}
    print(json.dumps(metrics, indent=1), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cats": cats,
                "temperature": T, "metrics": metrics,
                "config": {"d": args.d, "vocab": 260, "max_tok": MAX_TOK}}, out)
    print(f"saved: {out}  ({out.stat().st_size/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
