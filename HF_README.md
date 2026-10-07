---
library_name: pytorch
license: mit
tags:
  - liquid-neural-network
  - intent-classification
  - edge
  - calibrated
---

# O1-Flash

**A liquid-core intent router — 4.53M parameters answering 77-way
customer-service routing as a calibrated multiple-choice decision.**

Part of the AwareLiquid O-Series. The recurrent core carries per-channel
learnable time constants (τ, geometrically spaced), so one small model holds
both phoneme-scale and utterance-scale state with a fixed-size hidden state.

## Honest status

Measured on **banking77** (10,003 train / 3,080 test rows, 77 intents,
byte-level tokenizer, no external vocab), default config d256/4L/4.53M params,
batch 64:

| steps | test acc | ECE | ECE (temperature-scaled) |
|---|---|---|---|
| 1,500 | 23.28% | — | — |
| 20,000 (lr 1e-3) | 30.23% | 0.485 | 0.161 |
| 20,000 (scheduled 3e-3) | 33.28% | 0.469 | 0.161 |
| **200,000 (scheduled)** | **40.75%** | 0.456 | **0.127** |

- Chance = 1.3%. The default-config model is far from a production router;
  40.75% at 4.53M params is the honest ceiling measured so far.
- **Calibration works**: temperature scaling (LBFGS on a held-out half) cuts
  ECE by ~3.6× on the final model (T=4.41).
- **Training-stability finding (2026-10-04)**: at lr 3e-3, training diverges
  under GPU contention (learns, then collapses to chance — reproduced on two
  machines). Fix: `--lr 1e-3` default plus linear-warmup + cosine schedule.
  The 200k run above ran fully stable under the schedule.
- The temperature parameter (T=4.41) is saved with the evaluation artifact,
  not baked into the checkpoint: apply it to the softmax only when calibrated
  probabilities are needed.

## Files

- `o1flash_banking77_200k_sched.pt` — PyTorch checkpoint (18.2 MB,
  4.53M params, trained 200k steps with the lr schedule). Contains
  `model` state dict + `config`.

## Usage

```python
import torch
from benchmarks.real_data_bench import O1Flash, FlashConfig  # github.com/AwareLiquid/O1-Flash

ck = torch.load("o1flash_banking77_200k_sched.pt", map_location="cpu",
                weights_only=True)
model = O1Flash(FlashConfig(**ck["config"]))
model.load_state_dict(ck["model"])
```

## Limits

- English customer-service text only; byte-level tokenizer.
- 40.75% top-1 on 77 intents — not production. The point of this checkpoint
  is reproducibility of the measured numbers and the stability/calibration
  findings, not deployment.
- The same architecture/discussion as the rest of the O-Series: see
  [EverestAn/MT-LNN](https://huggingface.co/EverestAn/MT-LNN) for the family
  card and the awareliquid.ai benchmark page.
