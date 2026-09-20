"""Decision benchmark: train one fresh tiny model per question type and
report accuracy + ECE on the synthetic decision set.

Honest scope: synthetic keyword-signal tasks (linear-separable), so the
numbers here prove the pipeline trains — they are NOT decision-quality
claims. Single-question training per model (see DESIGN §7 for why joint
multi-question training on the liquid core is an open problem). Run:

    python -m benchmarks.decision_bench
"""

from mt_flash.config import o1_flash_tiny
from mt_flash.model import O1Flash
from mt_flash.train import make_question, train_single

if __name__ == "__main__":
    print("=== decision bench (synthetic, honest scope) ===")
    print(f"{'question':8s} {'acc':>7s}  {'ece':>7s}")
    for kind in ("dept", "sev", "urgent"):
        model = O1Flash(o1_flash_tiny())
        print(f"training {kind} (400 steps, fresh model)...")
        m = train_single(model, make_question(kind), steps=400, log_every=None)
        print(f"{kind:8s} {m[f'{kind}_acc']:7.3f}  {m[f'{kind}_ece']:7.3f}")
