"""决策语料训练 — O1-Flash heads on JevBench-shaped items (per-row option sets).

每题不同选项集：readout = 各选项文本嵌入 ⊗ 状态序列逐位置相似度的均值池化
（复用 heads 的 opt_proj/_mean_embed/scale，仅做 per-row 批量）。

    python benchmarks/decision_train.py --corpus corpus.jsonl --steps 60000 \
        --batch 32 --lr 1e-3 --out checkpoints/decision_v1.pt [--val-every 5000]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mt_flash.config import FlashConfig  # noqa: E402
from mt_flash.model import O1Flash  # noqa: E402


def option_text(row: dict, lab: str) -> str:
    crit = row.get("criteria") or {}
    desc = crit.get(lab) if isinstance(crit, dict) else None
    return f"{lab}: {desc}" if desc else lab


def core_text(row: dict) -> str:
    state = str(row.get("state", ""))
    instr = row.get("instructions", "") or ""
    return (state + "\n" + instr).strip()


@torch.no_grad()
def embed_options(heads, rows: list[dict], device) -> tuple[torch.Tensor, torch.Tensor]:
    k_max = max(len(r["labels"]) for r in rows)
    dim = heads.opt_proj.out_features
    opts = torch.zeros(len(rows), k_max, dim, device=device)
    mask = torch.zeros(len(rows), k_max, dtype=torch.bool, device=device)
    for b, r in enumerate(rows):
        for k, lab in enumerate(r["labels"]):
            opts[b, k] = heads.opt_proj(
                heads._mean_embed(option_text(r, lab), device))
            mask[b, k] = True
    return opts, mask


def decision_scores(heads, y: torch.Tensor, opts: torch.Tensor,
                    opt_mask: torch.Tensor,
                    state_mask: torch.Tensor) -> torch.Tensor:
    sim = torch.einsum("bkd,btd->bkt", opts, y) * heads.scale     # (B,K,T)
    sim = sim.masked_fill(~state_mask.unsqueeze(1), 0.0)
    denom = state_mask.sum(dim=1, keepdim=True).clamp(min=1)
    scores = sim.sum(dim=-1) / denom                              # (B,K)
    return scores.masked_fill(~opt_mask, float("-inf"))


def batch_tensors(model, rows: list[dict], device):
    ids = torch.nn.utils.rnn.pad_sequence(
        [model._ids_for(core_text(r)) for r in rows],
        batch_first=True, padding_value=256).to(device)
    mask = ids != 256
    targets = torch.tensor(
        [r["labels"].index(r["expected"]) for r in rows], device=device)
    return ids, mask, targets


@torch.no_grad()
def evaluate(model, rows: list[dict], device, batch: int = 64) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        ids, mask, targets = batch_tensors(model, chunk, device)
        y = model(ids)
        opts, omask = embed_options(model.heads, chunk, device)
        scores = decision_scores(model.heads, y, opts, omask, mask)
        correct += int((scores.argmax(dim=-1) == targets).sum())
    model.train()
    return correct / max(len(rows), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/decision_v1.pt")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--val-every", type=int, default=5000)
    ap.add_argument("--log-every", type=int, default=500)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.corpus, encoding="utf-8")
            if l.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_val = max(64, int(len(rows) * args.val_frac))
    val, train = rows[:n_val], rows[n_val:]
    print(f"corpus: {len(train)} train / {len(val)} val", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    model = O1Flash(FlashConfig()).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()

    t0 = time.time()
    for step in range(args.steps):
        if step < args.warmup:
            cur = args.lr * (step + 1) / max(args.warmup, 1)
        else:
            prog = (step - args.warmup) / max(1, args.steps - args.warmup)
            cur = args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))
        for g in opt.param_groups:
            g["lr"] = cur
        chunk = rng.sample(train, args.batch)
        ids, mask, targets = batch_tensors(model, chunk, device)
        y = model(ids)
        opts, omask = embed_options(model.heads, chunk, device)
        scores = decision_scores(model.heads, y, opts, omask, mask)
        loss = F.cross_entropy(scores, targets)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % args.log_every == 0:
            print(f"step {step+1}/{args.steps} loss {loss.item():.4f} "
                  f"lr {cur:.2e} {(step+1)/(time.time()-t0):.1f} step/s",
                  flush=True)
        if (step + 1) % args.val_every == 0 or step + 1 == args.steps:
            acc = evaluate(model, val, device)
            print(f"  [val] step {step+1} acc {acc:.4f}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model": model.state_dict(),
                "config": FlashConfig().__dict__ if hasattr(FlashConfig(), "__dict__") else {},
                "steps": args.steps, "corpus": args.corpus}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
