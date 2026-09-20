"""Decision benchmark: joint calibration training on the synthetic
decision set; report accuracy + ECE per question type.

Honest scope: synthetic keyword-signal tasks (linear-separable), so the
numbers here prove the pipeline trains end-to-end in joint mode — they
are NOT decision-quality claims. Run:

    python -m benchmarks.decision_bench
"""

from mt_flash.config import o1_flash_tiny
from mt_flash.model import O1Flash
from mt_flash.train import train

if __name__ == "__main__":
    model = O1Flash(o1_flash_tiny())
    print("training tiny O1-Flash (joint 3 questions, 600 steps)...")
    m = train(model, steps=600, batch=32, log_every=200)
    print("\n=== decision bench (synthetic, honest scope) ===")
    print(f"{'question':8s} {'acc':>7s}  {'ece':>7s}")
    for kind in ("dept", "sev", "urgent"):
        print(f"{kind:8s} {m[f'{kind}_acc']:7.3f}  {m[f'{kind}_ece']:7.3f}")
