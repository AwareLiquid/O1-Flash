"""Parallel typed decision heads (the Jev-parallel piece).

One shared state is encoded ONCE by the liquid core; every question branch
then reads it independently and in parallel:

    score_q = state · W_q · option_embedding        (bilinear)

There is NO cross-question attention and NO token generation anywhere in
this module, so two structural guarantees hold by construction:

1. Questions are independent: the answer to question A is bit-identical
   whether or not question B is in the same request (tested).
2. Outputs are schema-bounded: every answer is a probability vector over
   the caller-defined options/levels — a value outside the schema cannot
   be produced (a wrong value still can; that is what confidence is for).

Confidence follows Jev's definition: peakedness of the returned
distribution. A flat distribution = low confidence regardless of which
option won.
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
    """Typed readouts over the shared liquid state, all in one forward."""

    def __init__(self, cfg: FlashConfig, embed: nn.Embedding,
                 tokenizer_encode, max_len: int):
        super().__init__()
        D = cfg.d_model
        self.cfg = cfg
        self.embed = embed          # shared frozen byte embedding (no recurrence)
        self.encode_fn = tokenizer_encode
        self.max_len = max_len

        self.choice_proj = nn.Linear(D, D, bias=False)      # per-question scorer
        self.score_proj = nn.Linear(D, cfg.max_score_levels)
        self.prob_head = nn.Linear(D, 1)

    def _embed_options(self, options: tuple[str, ...],
                       device: torch.device) -> torch.Tensor:
        """Options -> (B, n_opt, D) via the shared byte embedding (mean-pooled)."""
        per_opt = []
        for opt in options:
            ids = self.encode_fn(opt, self.max_len)
            t = torch.tensor(ids, device=device, dtype=torch.long)
            emb = self.embed(t).mean(dim=0)                # (D,)
            per_opt.append(emb)
        return torch.stack(per_opt, dim=0).unsqueeze(0)    # (1, n_opt, D)

    def _choice(self, state: torch.Tensor, q: ChoiceQuestion) -> ChoiceAnswer:
        opts = self._embed_options(q.options, state.device)       # (1,N,D)
        key = self.choice_proj(state)                             # (B,D)
        logits = torch.einsum("bd,bnd->bn", key, opts)            # (B,N)
        dist = F.softmax(logits, dim=-1)[0]
        choice = int(dist.argmax().item())
        return ChoiceAnswer(id=q.id, distribution=dist.tolist(),
                            choice=choice,
                            confidence=float(_peakedness(dist).item()))

    def _score(self, state: torch.Tensor, q: ScoreQuestion) -> ScoreAnswer:
        n = len(q.levels)
        logits = self.score_proj(state)[0, :n]                    # (n,)
        dist = F.softmax(logits, dim=-1)
        level = int(dist.argmax().item())
        return ScoreAnswer(id=q.id, distribution=dist.tolist(),
                           level=level,
                           confidence=float(_peakedness(dist).item()))

    def _probability(self, state: torch.Tensor,
                     q: ProbabilityQuestion) -> ProbabilityAnswer:
        p = torch.sigmoid(self.prob_head(state))[0, 0].item()
        return ProbabilityAnswer(id=q.id, probability=p,
                                 confidence=1.0 - 2.0 * abs(p - 0.5))

    def forward(self, state: torch.Tensor, questions: list[Question]
                ) -> dict[str, object]:
        """state: (B, D). All questions answered in parallel from it."""
        answers: dict[str, object] = {}
        for q in questions:
            if isinstance(q, ChoiceQuestion):
                answers[q.id] = self._choice(state, q)
            elif isinstance(q, ScoreQuestion):
                answers[q.id] = self._score(state, q)
            elif isinstance(q, ProbabilityQuestion):
                answers[q.id] = self._probability(state, q)
            else:  # pragma: no cover - schema exhaustiveness guard
                raise TypeError(f"unknown question type: {type(q)}")
        return answers
