"""ONNX export for fixed-schema deployment.

Edge deployment shape: a trained O1Flash with a FIXED question schema
(the three typed questions are baked in at export time). The option
embeddings become constant tensors, the tokenizer drops out entirely —
the exported graph is ids in, typed probability vectors out, runnable on
any ONNX runtime (CPU-first, no Python).

Mirrors the M1 repo's 5 MB ONNX wake-word export pattern: freeze the
schema, trace the forward, ship the bytes.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .model import O1Flash
from .schema import (ChoiceQuestion, ProbabilityQuestion, Question,
                     ScoreQuestion)


class FixedSchemaModel(nn.Module):
    """O1Flash wrapped with a baked question schema for ONNX export.

    forward(ids) -> (dept_probs, sev_probs, urgent_probs)
    All question descriptors are pre-encoded at export time; the runtime
    input is raw byte token ids only.
    """

    def __init__(self, model: O1Flash, questions: list[Question]):
        super().__init__()
        self.model = model
        self.questions = questions
        self._prep_questions()

    def _prep_questions(self) -> None:
        """Precompute the option embeddings once (they freeze at export)."""
        heads = self.model.heads
        device = next(heads.parameters()).device
        for q in self.questions:
            if isinstance(q, ChoiceQuestion):
                heads.register_buffer(
                    f"_opt_{q.id}", heads._embed_options(q.options, device),
                    persistent=False)
            elif isinstance(q, ScoreQuestion):
                heads.register_buffer(
                    f"_opt_{q.id}", heads._embed_options(q.levels, device),
                    persistent=False)

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        y, _ = self.model.encoder(ids)
        pad_mask = ids != 256
        heads = self.model.heads
        out: list[torch.Tensor] = []
        for q in self.questions:
            if isinstance(q, ChoiceQuestion):
                opts = getattr(heads, f"_opt_{q.id}")
                sim = torch.einsum("nd,btd->bnt", opts[0], y) * heads.scale
                sim = sim.masked_fill(~pad_mask.unsqueeze(1), 0.0)
                denom = pad_mask.sum(1, keepdim=True).clamp(min=1)
                out.append(torch.softmax(sim.sum(-1) / denom, dim=-1))
            elif isinstance(q, ScoreQuestion):
                opts = getattr(heads, f"_opt_{q.id}")
                sim = torch.einsum("nd,btd->bnt", opts[0], y) * heads.scale
                sim = sim.masked_fill(~pad_mask.unsqueeze(1), 0.0)
                denom = pad_mask.sum(1, keepdim=True).clamp(min=1)
                out.append(torch.softmax(sim.sum(-1) / denom, dim=-1))
            elif isinstance(q, ProbabilityQuestion):
                scores = heads.prob_pos(y).squeeze(-1)      # (B, T)
                scores = scores.masked_fill(~pad_mask, 0.0)
                denom = pad_mask.sum(1, keepdim=True).clamp(min=1)
                logit = scores.sum(-1, keepdim=True) / denom
                p = torch.sigmoid(logit)
                out.append(torch.cat([p, 1.0 - p], dim=-1))
        return tuple(out)


def export_onnx(model: O1Flash, questions: list[Question], path: str,
                seq_len: int = 128, opset: int = 14) -> None:
    """Export the fixed-schema model to ONNX at a fixed sequence length.

    Batch axis is dynamic; the sequence length is fixed at export (the
    scan loop unrolls to it). Callers pad/truncate inputs to seq_len.
    """
    wrapped = FixedSchemaModel(model, questions).eval()
    dummy = torch.zeros(1, seq_len, dtype=torch.long)
    torch.onnx.export(
        wrapped, dummy, path,
        input_names=["ids"],
        output_names=[q.id for q in questions],
        dynamic_axes={"ids": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )
