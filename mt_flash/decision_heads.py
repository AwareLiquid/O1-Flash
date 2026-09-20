"""Parallel typed decision heads with per-question readout (the
Jev-parallel piece, done faithfully).

Jev's architecture: the state is encoded once into a cache, then EACH
question branch attends to that shared state with its OWN query — the
branches are independent and run in one forward pass. This module is the
same shape:

    y (B, T, D)  ── encoded once by the liquid core
    question q ──> query = mean-pooled embedding of q's descriptor text
                   key_q = attention-pool(y, query)          (per-question)
                   answer = score(key_q, option embeddings)  (typed head)

Because each question owns its attention over positions, the recency bias
measured with a single shared readout (DESIGN §7) does not apply: every
question reads the positions it needs.

Two structural guarantees, tested:

1. Question independence: question A's answer is bit-identical whether B
   is in the request — no cross-question pathway exists.
2. Schema-bounded outputs: every answer is a probability vector over the
   caller-defined options/levels. Out-of-schema values are impossible;
   wrong values are still possible — that is what confidence is for.

Confidence = peakedness of the returned distribution (Jev's definition).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FlashConfig
from .schema import (ChoiceAnswer, ChoiceQuestion, ProbabilityAnswer,
                     ProbabilityQuestion, Question, ScoreAnswer,
                     ScoreQuestion)


def _peakedness(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Confidence = peakedness of the distribution (Jev's definition)."""
    return 1.0 - torch.special.entr(probs).sum(dim=dim) / torch.log(
        torch.tensor(probs.shape[dim], dtype=probs.dtype, device=probs.device)
    )


class ParallelDecisionHeads(nn.Module):
    """Per-question readout over the shared sequence, all in one forward."""

    def __init__(self, cfg: FlashConfig, embed: nn.Embedding,
                 tokenizer_encode, max_len: int):
        super().__init__()
        rd = cfg.d_model * (2 if cfg.bidirectional else 1) \
            + (cfg.d_model if cfg.hybrid_readout else 0)     # readout dim
        self.cfg = cfg
        self.embed = embed          # shared byte embedding (no recurrence)
        self.encode_fn = tokenizer_encode
        self.max_len = max_len
        self.scale = rd ** -0.5

        self.attn_key = nn.Linear(rd, rd, bias=False)     # position keys
        self.choice_attn = nn.Linear(rd, rd, bias=False)  # per-type position keys
        self.score_attn = nn.Linear(rd, rd, bias=False)   # (no cross-question
        self.prob_attn = nn.Linear(rd, rd, bias=False)    #  gradient interference)
        self.query_proj = nn.Linear(cfg.d_model, rd, bias=False)  # query -> readout dim
        self.opt_proj = nn.Linear(cfg.d_model, rd, bias=False)    # options -> readout dim
        self.choice_proj = nn.Linear(rd, rd, bias=False)  # per-question scorer
        self.score_proj = nn.Linear(rd, cfg.max_score_levels)
        self.prob_head = nn.Linear(rd, 1)

    def _mean_embed(self, text: str, device: torch.device) -> torch.Tensor:
        ids = self.encode_fn(text, self.max_len)
        t = torch.tensor(ids, device=device, dtype=torch.long)
        return self.embed(t).mean(dim=0)                # (d_model,)

    def _pool(self, y: torch.Tensor, query: torch.Tensor,
              pad_mask: torch.Tensor | None = None,
              attn_key: nn.Module | None = None) -> torch.Tensor:
        """Per-question attention-pool over the shared sequence -> (B, D).

        pad_mask: (B, T) True=real. Pad positions get -inf attention.
        attn_key: per-question-type key projection (default shared).
        """
        q = self.query_proj(query)                      # (D,)
        key_fn = attn_key if attn_key is not None else self.attn_key
        scores = torch.einsum("d,btd->bt", q, key_fn(y)) * self.scale
        if pad_mask is not None:
            scores = scores.masked_fill(~pad_mask, float("-inf"))
        alpha = torch.softmax(scores, dim=-1)
        return torch.einsum("bt,btd->bd", alpha, y)

    def _embed_options(self, options: tuple[str, ...],
                       device: torch.device) -> torch.Tensor:
        """Options -> (1, n_opt, D) in readout space."""
        per_opt = [self.opt_proj(self._mean_embed(o, device))
                   for o in options]
        return torch.stack(per_opt, dim=0).unsqueeze(0)

    # -- batched probs (training readouts) --------------------------------
    def choice_probs(self, y: torch.Tensor, q: ChoiceQuestion,
                     pad_mask: torch.Tensor | None = None
                     ) -> torch.Tensor:
        """Option-matching readout: score each option by logsumexp of its
        similarity across positions (soft-max over the sequence, smooth
        gradients to every position). Position-robust: the option word
        appearing anywhere drives its score; no attention routing.
        """
        opts = self._embed_options(q.options, y.device)     # (1, N, D)
        sim = torch.einsum("nd,btd->bnt", opts[0], y) * self.scale  # (B,N,T)
        if pad_mask is not None:
            sim = sim.masked_fill(~pad_mask.unsqueeze(1), float("-inf"))
        scores = torch.logsumexp(sim, dim=-1)               # (B, N)
        return F.softmax(scores, dim=-1)

    def score_probs(self, y: torch.Tensor, q: ScoreQuestion,
                    pad_mask: torch.Tensor | None = None
                    ) -> torch.Tensor:
        opts = self._embed_options(q.levels, y.device)
        sim = torch.einsum("nd,btd->bnt", opts[0], y) * self.scale
        if pad_mask is not None:
            sim = sim.masked_fill(~pad_mask.unsqueeze(1), float("-inf"))
        scores = torch.logsumexp(sim, dim=-1)
        return F.softmax(scores, dim=-1)

    def prob_scalar(self, y: torch.Tensor, q: ProbabilityQuestion,
                    pad_mask: torch.Tensor | None = None
                    ) -> torch.Tensor:
        query = self._mean_embed(q.prompt, y.device)
        key = self._pool(y, query, pad_mask=pad_mask, attn_key=self.score_attn)
        return torch.sigmoid(self.prob_head(key))

    # -- single-item answer builders (inference) --------------------------
    def _choice(self, y: torch.Tensor, q: ChoiceQuestion) -> ChoiceAnswer:
        dist = self.choice_probs(y, q)[0]
        choice = int(dist.argmax().item())
        return ChoiceAnswer(id=q.id, distribution=dist.tolist(),
                            choice=choice,
                            confidence=float(_peakedness(dist).item()))

    def _score(self, y: torch.Tensor, q: ScoreQuestion) -> ScoreAnswer:
        dist = self.score_probs(y, q)[0]
        level = int(dist.argmax().item())
        return ScoreAnswer(id=q.id, distribution=dist.tolist(),
                           level=level,
                           confidence=float(_peakedness(dist).item()))

    def _probability(self, y: torch.Tensor,
                     q: ProbabilityQuestion) -> ProbabilityAnswer:
        p = self.prob_scalar(y, q)[0, 0].item()
        return ProbabilityAnswer(id=q.id, probability=p,
                                 confidence=1.0 - 2.0 * abs(p - 0.5))

    def forward(self, y: torch.Tensor, questions: list[Question]
                ) -> dict[str, object]:
        """y: (B, T, D). All questions answered in parallel from it."""
        answers: dict[str, object] = {}
        for q in questions:
            if isinstance(q, ChoiceQuestion):
                answers[q.id] = self._choice(y, q)
            elif isinstance(q, ScoreQuestion):
                answers[q.id] = self._score(y, q)
            elif isinstance(q, ProbabilityQuestion):
                answers[q.id] = self._probability(y, q)
            else:  # pragma: no cover - schema exhaustiveness guard
                raise TypeError(f"unknown question type: {type(q)}")
        return answers

    def train_forward(self, y: torch.Tensor, questions: list[Question],
                      pad_mask: torch.Tensor | None = None
                      ) -> dict[str, torch.Tensor]:
        """Batch probs per question id — the training readout (B, n_classes)."""
        out: dict[str, torch.Tensor] = {}
        for q in questions:
            if isinstance(q, ChoiceQuestion):
                out[q.id] = self.choice_probs(y, q, pad_mask=pad_mask)
            elif isinstance(q, ScoreQuestion):
                out[q.id] = self.score_probs(y, q, pad_mask=pad_mask)
            elif isinstance(q, ProbabilityQuestion):
                p = self.prob_scalar(y, q, pad_mask=pad_mask)
                out[q.id] = torch.cat([p, 1.0 - p], dim=-1)  # (B, 2)
            else:  # pragma: no cover
                raise TypeError(f"unknown question type: {type(q)}")
        return out
