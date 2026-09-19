"""Liquid recurrent core: multi-timescale gated linear recurrence with O(1) state.

Mirrors the O-series (attention-free ARR) math from the M1 repo
(``mt_lnn/mt_lnn_v2.py`` MTLNNLayerV2), re-implemented standalone:

    x (B, T, D)
      -> factorized in-proj            D -> r -> P*d
      -> per-proto mixing W_mix        (P, d, d), shared across scales
      -> per-(proto, scale) diagonal sigmoid modulation A_ps
      -> decay scan:  h_ps,t = decay_ps * h_ps,t-1 + (1-decay_ps) * A_ps,t
      -> kappa-gated softmax blend across the tau ladder
      -> per-dim SiLU output gate
      -> factorized out-proj           P*d -> r -> D

The carried state is h_last (B, P, S, d): its size depends only on the
architecture, never on T. That is the O(1) memory property the whole
"Flash" pitch rests on.

The scan here is a plain sequential loop (O(T) compute, O(1) state). The
Blelloch parallel-scan / chunkwise (SSD-style) paths from M1 are noted in
DESIGN.md as a later optimisation; the math is identical.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FlashConfig


class MultiScaleLiquidBlock(nn.Module):
    """One liquid recurrent block (attention-free token mixer)."""

    def __init__(self, cfg: FlashConfig):
        super().__init__()
        P, d, S, r = (cfg.n_protofilaments, cfg.d_proto,
                      cfg.n_time_scales, cfg.proj_rank)
        self.P, self.d, self.S = P, d, S
        self.dt = cfg.dt
        dpt = P * d

        # Factorized projections (bias-free; residual carries identity)
        self.in_a = nn.Linear(cfg.d_model, r, bias=False)
        self.in_b = nn.Linear(r, dpt, bias=False)
        self.out_a = nn.Linear(dpt, r, bias=False)
        self.out_b = nn.Linear(r, cfg.d_model, bias=False)

        # Shared per-proto mixing + per-(proto, scale) diagonal modulation
        self.W_mix = nn.Parameter(torch.empty(P, d, d))
        nn.init.normal_(self.W_mix, std=0.02)
        self.scale_gain = nn.Parameter(torch.ones(P, S, d))
        self.scale_bias = nn.Parameter(torch.zeros(P, S, d))

        # Tau ladder: softplus-parameterised, geometric extension if short
        taus = list(cfg.tau_init)[:S]
        while len(taus) < S:
            taus.append(min(taus[-1] * 4.0, cfg.tau_max * 0.9))
        log_tau = torch.empty(P, S)
        for s, t in enumerate(taus):
            log_tau[:, s] = math.log(math.expm1(max(t - cfg.tau_min, 1e-6)))
        self.log_tau = nn.Parameter(log_tau)
        self.tau_min = cfg.tau_min

        # Selective decay (opt-in): per-token dt from the input, init dt==1
        self.selective_decay = cfg.selective_decay
        # softplus^-1(1) = ln(e-1): b_dt init makes dt == 1 exactly at init
        self.b_dt = nn.Parameter(torch.tensor(math.log(math.e - 1.0)))
        self.w_dt = nn.Linear(cfg.d_model, 1, bias=False)
        # Zero-init: dt == 1 EXACTLY at init (strict no-op), selectivity
        # emerges only through gradient — no init noise in the decay path.
        nn.init.zeros_(self.w_dt.weight)

        # Scale blend (kappa gate): content-dependent weighting of the ladder
        self.kappa = nn.Linear(cfg.d_model, S, bias=False)
        nn.init.zeros_(self.kappa.weight)

        # SiLU output gate (Mamba-style)
        self.out_gate = nn.Linear(dpt, dpt, bias=False)
        nn.init.ones_(self.out_gate.weight)

        self.dropout = nn.Dropout(cfg.dropout)

    def _tau(self) -> torch.Tensor:
        return F.softplus(self.log_tau) + self.tau_min  # (P, S)

    def _dt_t(self, u: torch.Tensor) -> torch.Tensor:
        """Per-token dt: softplus(W_dt u + b_dt), init == 1 exactly."""
        return F.softplus(self.w_dt(u) + self.b_dt)  # (B, T, 1)

    def forward(self, x: torch.Tensor,
                h_prev: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, D) -> (y: (B, T, D), h_last: (B, P, S, d))."""
        B, T, _ = x.shape
        P, S, d = self.P, self.S, self.d

        u = self.in_b(self.in_a(x))                       # (B, T, P*d)
        u = u.view(B, T, P, d)

        a = torch.einsum("btpd,pde->btpe", u, self.W_mix)  # per-proto mix
        A = torch.sigmoid(self.scale_gain * a.unsqueeze(3)
                          + self.scale_bias)               # (B,T,P,S,d)

        tau = self._tau()                                  # (P, S)
        if self.selective_decay:
            dt_t = self._dt_t(x)                           # (B, T, 1)
            decay = torch.exp(-dt_t.unsqueeze(-1).unsqueeze(-1)
                              / tau.view(1, 1, P, S, 1))  # (B,T,P,S,1)
        else:
            decay = torch.exp(-self.dt / tau.view(1, 1, P, S, 1)
                              ).expand(B, T, P, S, 1)     # (B,T,P,S,1)

        # Sequential scan (O(T) compute, O(1) state)
        h = torch.zeros(B, P, S, d, device=x.device, dtype=x.dtype)
        if h_prev is not None:
            h = h + h_prev
        for t in range(T):
            h = decay[:, t] * h + (1.0 - decay[:, t]) * A[:, t]

        # Kappa-gated blend across scales: softmax over ladder weighted by
        # content gate, per (proto, dim)
        kappa = torch.softmax(self.kappa(x), dim=-1)        # (B, T, S)
        h_blend = torch.einsum("bts,bpsd->btpd", kappa, h)  # (B, T, P, d)

        g = torch.sigmoid(self.out_gate(h_blend.reshape(B, T, P * d)))
        y_raw = g * h_blend.reshape(B, T, P * d)
        y = self.out_b(self.out_a(y_raw))
        y = self.dropout(y)
        return y, h  # h = h_last (B, P, S, d)


class LiquidStateEncoder(nn.Module):
    """Byte-level token embedding + N liquid blocks -> final O(1) state."""

    def __init__(self, cfg: FlashConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            MultiScaleLiquidBlock(cfg) for _ in range(cfg.n_layers)
        )
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, ids: torch.Tensor,
                h_prev: list[torch.Tensor] | None = None
                ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """ids: (B, T) -> (state: (B, D), h_last: list of (B,P,S,d))."""
        x = self.embedding(ids)
        h_last: list[torch.Tensor] = []
        for i, blk in enumerate(self.blocks):
            x, h = blk(x, h_prev=h_prev[i] if h_prev is not None else None)
            h_last.append(h)
        state = self.norm(x[:, -1])  # final carried representation
        return state, h_last

    def reset_stream(self, h_last: list[torch.Tensor] | None) -> None:
        """Streaming hook kept explicit: callers pass h_prev=None to reset."""
        del h_last  # stateless by design; kept for API symmetry
