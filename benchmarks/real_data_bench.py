"""Real-data benchmark: banking77 intent routing (the Jev pitch scenario).

Trains the default-config O1-Flash (d256/4L, ~4.5M params) on real
customer-service queries (10,003 train rows, 77 intents) and reports
accuracy + ECE on the official test split (3,080 rows). 77-way chance
is 1.3% — every number is measured, reported honestly, and reproducible:

    python -m benchmarks.real_data_bench            # full run (~30 min CPU)
    python -m benchmarks.real_data_bench --steps 200 --n_train 2000  # smoke

Data: PolyAI banking77 (CC BY 4.0), fetched from the task-specific-datasets
repo on first run and cached under the system temp dir.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile

import torch
import torch.nn as nn

from mt_flash.calibration import calibration_step, expected_calibration_error
from mt_flash.config import FlashConfig
from mt_flash.model import O1Flash
from mt_flash.schema import ChoiceQuestion

_BASE = ("https://raw.githubusercontent.com/PolyAI-LDN/"
         "task-specific-datasets/master/banking_data/")


def _fetch(name: str, cache_dir: str) -> str:
    path = os.path.join(cache_dir, name)
    if not os.path.exists(path):
        import requests
        proxies = {"http": "http://127.0.0.1:9674",
                   "https": "http://127.0.0.1:9674"}
        r = requests.get(_BASE + name, proxies=proxies, timeout=120)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    return path


def load_banking77() -> tuple[list[tuple[str, int]], list[tuple[str, int]],
                              list[str]]:
    cache = os.path.join(tempfile.gettempdir(), "o1flash_banking77")
    os.makedirs(cache, exist_ok=True)
    cats = json.load(open(_fetch("categories.json", cache), encoding="utf-8"))
    idx = {c: i for i, c in enumerate(cats)}
    train, test = [], []
    for fname, out in (("train.csv", train), ("test.csv", test)):
        with open(_fetch(fname, cache), encoding="utf-8") as f:
            for row in csv.DictReader(f):
                out.append((row["text"], idx[row["category"]]))
    return train, test, cats


def train_banking77(steps: int = 1500, batch: int = 32, n_train: int | None = None,
                    seed: int = 0, log_every: int | None = 300
                    ) -> dict[str, float]:
    train, test, cats = load_banking77()
    if n_train:
        train = train[:n_train]
    q = ChoiceQuestion(id="intent", options=tuple(cats))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = O1Flash(FlashConfig()).to(device)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    import random
    rng = random.Random(seed)

    for step in range(steps):
        rows = rng.sample(train, batch)
        ids = torch.nn.utils.rnn.pad_sequence(
            [model._ids_for(r[0]) for r in rows],
            batch_first=True, padding_value=256).to(device)
        y = model(ids)
        mask = ids != 256
        probs = model.heads.train_forward(y, [q], pad_mask=mask)
        targets = torch.tensor([r[1] for r in rows], device=device)
        loss = calibration_step(probs["intent"], targets)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log_every and (step + 1) % log_every == 0:
            print(f"step {step + 1:5d}  loss {loss.item():.4f}")

    # -- official test split --------------------------------------------
    model.eval()
    test_ids = torch.nn.utils.rnn.pad_sequence(
        [model._ids_for(r[0]) for r in test],
        batch_first=True, padding_value=256)
    test_t = torch.tensor([r[1] for r in test])
    all_probs = []
    for i in range(0, len(test), 64):
        ids = test_ids[i:i + 64].to(device)
        mask = ids != 256
        with torch.no_grad():
            y = model(ids)
            p = model.heads.train_forward(y, [q], pad_mask=mask)["intent"]
        all_probs.append(p.cpu())
    probs = torch.cat(all_probs, dim=0)
    acc = (probs.argmax(-1) == test_t).float().mean().item()
    ece = expected_calibration_error(probs, test_t)
    return {"acc": acc, "ece": ece, "n_params": sum(p.numel() for p in model.parameters()),
            "n_test": len(test), "n_classes": len(cats)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--n_train", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    print("banking77 intent routing — real-data training (default config)")
    m = train_banking77(steps=args.steps, n_train=args.n_train, batch=args.batch)
    print(f"\n=== banking77 (77-way, chance 1.3%) ===")
    print(f"params {m['n_params']/1e6:.2f}M  n_test {m['n_test']}  "
          f"acc {m['acc']:.4f}  ece {m['ece']:.3f}")
