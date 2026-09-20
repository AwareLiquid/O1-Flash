"""Configuration for O1-Flash."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class FlashConfig:
    """Hyperparameters for the liquid recurrent decision encoder.

    Defaults are the O-series (attention-free ARR) profile: a multi-scale
    liquid recurrence whose carried state size does not depend on input
    length (O(1) memory), with typed parallel decision heads on top.
    """

    # -- recurrent core -------------------------------------------------
    d_model: int = 256
    n_layers: int = 4
    n_protofilaments: int = 13          # biological prior (microtubule count)
    d_proto: int = 32                   # state dim per protofilament
    n_time_scales: int = 5              # tau ladder rungs
    proj_rank: int = 64                 # factorization rank of in/out projections
    tau_min: float = 0.1
    tau_max: float = 100.0
    dt: float = 1.0
    tau_init: Tuple[float, ...] = (0.5, 2.0, 8.0, 32.0, 90.0)
    dropout: float = 0.0
    # Selective (input-dependent) decay: content decides persistence.
    # At init dt == 1 reproduces the static path exactly (strict generalisation).
    selective_decay: bool = False
    # Bidirectional readout: the decision heads attend over a sequence whose
    # every position sees the whole input (forward scan + reversed scan,
    # shared weights — BiRNN style). The O(1) carried state remains the
    # forward scan only, so streaming replay is unaffected.
    bidirectional: bool = True
    # Hybrid readout: concatenate the RAW byte embeddings alongside the
    # liquid outputs. The liquid scan smears positions over time, which
    # hurts exact content matching (measured: joint multi-task fails,
    # DESIGN §7); raw-embedding access restores direct token-level
    # matching — the "compressed state + exact recall" hybrid the M1 repo
    # already validated in its HOLA-style family.
    hybrid_readout: bool = True

    # -- input encoding ------------------------------------------------
    vocab_size: int = 260               # byte-level tokenizer: 256 bytes + specials
    max_seq_len: int = 4096             # input budget; state stays O(1) beyond it

    # -- decision heads ------------------------------------------------
    max_options: int = 255              # per Choice question (Jev's 2^8-1 cap)
    max_questions: int = 64             # per request
    max_score_levels: int = 10

    # -- training ------------------------------------------------------
    brier_weight: float = 1.0
    ece_n_bins: int = 10


def o1_flash_default() -> FlashConfig:
    return FlashConfig()


def o1_flash_tiny() -> FlashConfig:
    """Small config for tests and CPU smoke runs."""
    return FlashConfig(
        d_model=64, n_layers=2, d_proto=16, proj_rank=32,
        n_protofilaments=13, n_time_scales=5, max_seq_len=512,
    )
