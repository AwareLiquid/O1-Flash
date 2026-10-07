# O1-Flash — 端侧 System One 决策模型

**Typed decisions, not text.** O1-Flash is the O-series (attention-free ARR)
line of AwareLiquid, re-cast as a System One model in the spirit of
TypeSafe's Jev (Sept 2026): state in, calibrated parallel probabilities out.
No text generation exists anywhere in the output path.

```
state text/JSON ──> liquid recurrent core (O(1) carried state)
                        │  h_final (B, D)  ── encoded once
                        ├─ Choice      ──> distribution over options  + confidence
                        ├─ Score       ──> distribution over levels   + confidence
                        └─ Probability ──> p ∈ [0,1]                  + confidence
                    (all read in parallel, no cross-question interaction)
```

## Why it exists

The M1 repo's O-series measured three edge properties no transformer can
match: **O(1) inference state** (0.381 MB flat, 8063× smaller than a KV
cache at 1M tokens), **5 MB ONNX wake-word export**, and **CPU-only
inference** on irregular sensor streams. Jev (TypeSafe, 2026-09-15) showed
the product shape for such an engine: give up text generation, return typed
calibrated decisions in one parallel pass — 40–400× cheaper than routing
every small decision through an LLM.

O1-Flash combines the two: the Jev parallel-readout paradigm on the O(1)
liquid state, sized for the edge instead of the cloud.

## What is proven / what is not (honest)

**Structural, proven by tests in this repo:**

- **O(1) state**: carried state size depends only on architecture, never on
  input length (tested to 4096 tokens; the M1 repo sweeps to 1M).
- **Question independence**: answer A is bit-identical whether B is in the
  request — no cross-question attention exists (`test_decision_heads.py`).
- **Schema-bounded outputs**: every answer is a probability vector over
  caller-defined options/levels; an out-of-schema value cannot be produced.
- **Calibrated-confidence training reference**: Brier + decision-CE
  objective with ECE reporting (`mt_flash/calibration.py`).

**Not proven (no trained checkpoint yet):**

- The decision head ships as trained weights for the JevBench experiment
  (`docs/JEVBENCH_RESULTS.md`) but ships **no product checkpoint** in this
  repo. The encoder is the O-series liquid core ported from M1
  (`mt_lnn/mt_lnn_v2.py` MTLNNLayerV2 math). This is the architecture +
  training reference, same discipline as M2's DPO/GRPO reference.
- No comparison against Jev's published workflow accuracy (67.8%).
- Calibration claims are reference-level: the recipe exists and is tested
  on synthetic data, but RLCD itself is TypeSafe's unpublished method.

## Benchmarks — official JevBench harness (2026-10)

Measured through **Benchmark Heaven's open JevBench harness**
(`github.com/fstandhartinger/jevbench`): the model serves its own
probability distribution over the exact label set on a native
`/v1/systemone` endpoint; scoring, schema validation and latency come from
the official scorer. Public tiers: 48 easy + 72 original decisions.

| model | raw acc | ECE | p50 | Speed axis* |
|---|---|---|---|---|
| M2-2B SFT (byte-level 2B) | 35.8% | 0.34–0.50 | 0.71–0.92 s | ~56 |
| **O1-Flash decision head** (4.5M, single forward) | 27.1% | 0.41–0.43 | **16–21 ms** | **~100 (ceiling)** |

\* official formula `100 − 20·log10(p50/0.1s)`, capped at 100.

**Honest reading:** speed and cost are solved — 16–21 ms is ~8× faster
than the board's leading systems (0.15–0.25 s) and ~40× faster than the
2B path — but accuracy is the open problem (the board's leaders reach
53–66 chance-corrected Intelligence; our checkpoints sit far below, so
this is signal, not a podium). Pipeline, per-item raw responses and
official summaries: [`docs/JEVBENCH_RESULTS.md`](docs/JEVBENCH_RESULTS.md)
+ [`results/jevbench/`](results/jevbench/).

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests/ -q          # 22 tests (joint-training test ~3 min)

# smoke: one state, three typed questions, one parallel pass
python -m mt_flash.smoke

# decision bench: joint calibration training on the synthetic set
python -m benchmarks.decision_bench
```

```python
from mt_flash.model import O1Flash
from mt_flash.schema import ChoiceQuestion, ProbabilityQuestion, ScoreQuestion

m = O1Flash()
resp = m.decide(
    "Server down since 03:12; 400 tickets in the infra queue.",
    [
        ChoiceQuestion(id="route", options=("billing", "infra", "other")),
        ScoreQuestion(id="sev", levels=("low", "mid", "high")),
        ProbabilityQuestion(id="escalate", prompt="needs escalation now"),
    ],
)
print(resp.to_dict())
```

## Design

The Jev-parallel mapping and every architectural decision are documented in
[DESIGN.md](DESIGN.md). Lineage: the liquid core is the standalone port of
M1's `mt_lnn/mt_lnn_v2.py` (MTLNNLayerV2) and `mt_lnn/arr.py`
(MTRecurrentMixer, attention-free). Sequential scan here (O(T) compute,
O(1) state); the Blelloch parallel-scan path is a documented optimisation.

## Repository boundaries

- [everest-an/M1](https://github.com/everest-an/M1) — the M1 model core
  (private). The O-series ARR and liquid math live there.
- [AwareLiquid/M2](https://github.com/AwareLiquid/M2) — experimental
  architecture (DPO/GRPO references).
- **AwareLiquid/O1-Flash (this repo)** — the standalone edge decision model.
- [AwareLiquid/AwareLiquid-World](https://github.com/AwareLiquid/AwareLiquid-World) —
  JEPA world-model line.

## License

MIT — see [LICENSE](LICENSE).
