"""Opt-in speedup: give Breeze's depth decoder a KV cache.

mlx-audio 0.5.1 decodes the 16 codec codebooks by re-running the depth decoder
over the whole growing prefix at every step, with no cache:

    tokens = [0, first_codebook]
    for _ in range(num_codebooks - 1):
        logits = depth_decoder.next_logits(mx.array(tokens)[None, :], hidden)
        tokens.append(sample(logits))

That is 2 + 3 + ... + 16 = 135 transformer positions per audio frame where 16
would do -- and the depth decoder is ~91% of the generation loop.

This module swaps in an incremental version: the first step runs positions 0
and 1 (backbone state + codebook 0), each later step feeds only the new token
and reuses the cache. Positions, RoPE offsets, the per-codebook embedding
offset and per-step output head follow the original layout. The source bench
found 4-bit bit-identical, but 8-bit can diverge through bf16 rounding. This is
not a bit-exact 8-bit transform. Ported from Local/breeze-tts-2/scripts/fast_depth.py.

Usage:
    from audio.breeze_fast_depth import enable_fast_depth
    enable_fast_depth()          # before/after load(); patches the class
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
from mlx_audio.lm.models.cache import KVCache
from mlx_audio.tts.models.breeze_tts.breeze_tts import Model

_ORIGINAL = Model._depth_tokens
_PATCHED = False


def _depth_tokens_cached(
    self,
    first_codebook: int,
    conditional_hidden: mx.array,
    *,
    unconditional_hidden: Optional[mx.array],
    cfg_scale: float,
    temperature: float,
    top_p: float,
    top_k: int,
) -> list[int]:
    depth = self.depth_decoder.model
    heads = self.depth_decoder.codebooks_head.weight
    vocab = depth.vocab_size
    layers = depth.layers

    def project(hidden: mx.array) -> mx.array:
        if depth.backbone_hidden_state_projector is not None:
            return depth.backbone_hidden_state_projector(hidden)
        return hidden

    def forward(embeds: mx.array, cache: list[KVCache]) -> mx.array:
        hidden = depth.inputs_embeds_projector(embeds)
        # Single-token steps need no mask; the two-position priming step is causal.
        mask = "causal" if hidden.shape[1] > 1 else None
        for layer, layer_cache in zip(layers, cache):
            hidden = layer(hidden, mask, layer_cache)
        return depth.norm(hidden)[:, -1, :]

    # Position 0 carries the backbone hidden state, position 1 carries codebook
    # 0 with embedding offset 0 -- the same layout the uncached path builds.
    batch = conditional_hidden.shape[0]
    first = mx.full((batch, 1), first_codebook, dtype=mx.int32)
    prime = mx.concatenate(
        [project(conditional_hidden)[:, None, :], depth.embed_tokens(first)], axis=1
    )

    cond_cache = [KVCache() for _ in layers]
    cond_hidden = forward(prime, cond_cache)

    uncond_cache = None
    uncond_out = None
    if unconditional_hidden is not None:
        uncond_prime = mx.concatenate(
            [project(unconditional_hidden)[:, None, :], depth.embed_tokens(first)],
            axis=1,
        )
        uncond_cache = [KVCache() for _ in layers]
        uncond_out = forward(uncond_prime, uncond_cache)

    codes = [first_codebook]
    last = self.num_codebooks - 2
    for step in range(self.num_codebooks - 1):
        logits = cond_hidden @ heads[step]
        if uncond_out is not None:
            unconditional = uncond_out @ heads[step]
            logits = unconditional + cfg_scale * (logits - unconditional)
        logits = self._mask_reserved_codec_logits(logits)
        token = self._sample(
            logits, temperature=temperature, top_p=top_p, top_k=top_k
        )
        codes.append(token)
        if step == last:
            break
        # Token for codebook `step + 1` sits at position `step + 1`, which the
        # original addresses through the embedding offset `(step + 1) * vocab`.
        nxt = depth.embed_tokens(
            mx.full((batch, 1), token + (step + 1) * vocab, dtype=mx.int32)
        )
        cond_hidden = forward(nxt, cond_cache)
        if uncond_cache is not None:
            uncond_out = forward(nxt, uncond_cache)

    return codes


def enable_fast_depth() -> bool:
    """Patch in the cached depth decoder. Returns True if it changed anything."""
    global _PATCHED
    if _PATCHED:
        return False
    Model._depth_tokens = _depth_tokens_cached
    _PATCHED = True
    return True


def disable_fast_depth() -> bool:
    """Restore mlx-audio's original uncached depth decoder."""
    global _PATCHED
    if not _PATCHED:
        return False
    Model._depth_tokens = _ORIGINAL
    _PATCHED = False
    return True


def is_enabled() -> bool:
    return _PATCHED
