"""Byte-level state tokenizer: any text/JSON in, token ids out.

Byte-level (like ByT5 / GPT-2 BPE's byte fallback) so the encoder is
language-agnostic — English, Chinese, Arabic, raw JSON all map to the
same 256-byte alphabet. Special token 256 = padding, 257 = bos, 258 = eos.
"""

from __future__ import annotations

PAD_ID, BOS_ID, EOS_ID = 256, 257, 258


def encode(text: str, max_len: int) -> list[int]:
    """Encode text to byte ids, truncated to max_len (no padding here)."""
    ids = [BOS_ID] + list(text.encode("utf-8"))[: max_len - 2] + [EOS_ID]
    return ids[:max_len]


def decode(ids: list[int] | bytes) -> str:
    if isinstance(ids, bytes):
        return ids.decode("utf-8", errors="replace")
    payload = bytes(b for b in ids if b < 256)
    return payload.decode("utf-8", errors="replace")
