"""O1Flash — top-level model: liquid O(1) state encoder + parallel typed heads."""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import FlashConfig
from .decision_heads import ParallelDecisionHeads
from .liquid_core import LiquidStateEncoder
from .schema import DecisionResponse, Question
from .state_tokenizer import encode


class O1Flash(nn.Module):
    """State in, typed parallel decisions out. No text generation anywhere."""

    def __init__(self, cfg: FlashConfig | None = None):
        super().__init__()
        self.cfg = cfg or FlashConfig()
        self.encoder = LiquidStateEncoder(self.cfg)
        self.heads = ParallelDecisionHeads(
            self.cfg, self.encoder.embedding,
            tokenizer_encode=lambda text, max_len: encode(text, max_len),
            max_len=self.cfg.max_seq_len,
        )

    def _ids_for(self, text: str) -> torch.Tensor:
        """Encode one text to token ids (helper for batched training)."""
        return torch.tensor(encode(text, self.cfg.max_seq_len),
                            dtype=torch.long)

    def encode_state(self, text: str, h_prev: list[torch.Tensor] | None = None
                     ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Encode the shared sequence once; h_last is the O(1) carried state."""
        ids = self._ids_for(text).unsqueeze(0)
        return self.encoder(ids, h_prev=h_prev)

    @torch.no_grad()
    def decide(self, state_text: str, questions: list[Question],
               h_prev: list[torch.Tensor] | None = None
               ) -> DecisionResponse:
        """Run the typed questions in parallel over the shared state."""
        self.eval()
        y, _ = self.encoder(self._ids_for(state_text).unsqueeze(0),
                            h_prev=h_prev)
        answers = self.heads(y, questions)
        return DecisionResponse(answers=answers)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Train path: return the sequence output (heads pool per question)."""
        y, _ = self.encoder(ids)
        return y
