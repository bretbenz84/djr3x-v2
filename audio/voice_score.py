"""
audio/voice_score.py — voice-similarity score mapping between embedder backends.

Every speaker-ID threshold in config (accept 0.50, confident 0.70, floors, the
intro/signature bars, ~20 keys in all) was field-tuned on RESEMBLYZER cosine
scores. ECAPA-TDNN produces cosines on a different scale: genuine matches land
roughly 0.3-0.75 (vs Resemblyzer's 0.45-0.93) and impostors 0.0-0.2 (vs 0.3-0.5).
Rather than re-tuning twenty thresholds blind, ECAPA scores are shifted onto the
Resemblyzer-calibrated scale by a constant offset:

    mapped = raw + VOICE_SCORE_OFFSET_ECAPA (0.25), clamped to [-1, 0.99]

A constant offset preserves ORDER and — critically — GAPS: every margin threshold
(SPEAKER_ID_KNOWN_MARGIN 0.07, thin-challenger relief, etc.) works unchanged
because offsets cancel in score differences. ECAPA's much wider genuine/impostor
separation is what retires the ambiguity bugs; the offset just lets the existing
decision logic read it.

This module is import-light on purpose (config only): it's shared by
audio/speaker_id.py, memory/people.py, memory/voice_signatures.py, and
intelligence/interaction.py without creating import cycles.
"""

import logging

import config

_log = logging.getLogger(__name__)

# Set by audio.speaker_id when the encoder actually loads (it may fall back to
# resemblyzer if the ECAPA model is missing). Until then, assume the configured
# backend so scores read consistently even in odd import orders.
_active_backend: str = str(getattr(config, "VOICE_EMBEDDER", "ecapa") or "ecapa").lower()


def set_active_backend(backend: str) -> None:
    global _active_backend
    _active_backend = str(backend or "").lower()
    _log.info("[voice_score] active embedder backend: %s", _active_backend)


def active_backend() -> str:
    return _active_backend


def map_similarity(raw: float) -> float:
    """Map a raw cosine similarity onto the Resemblyzer-calibrated threshold scale."""
    if _active_backend != "ecapa":
        return float(raw)
    offset = float(getattr(config, "VOICE_SCORE_OFFSET_ECAPA", 0.25) or 0.0)
    return max(-1.0, min(0.99, float(raw) + offset))


def embedding_dim() -> int:
    """Embedding dimension of the active backend (192 ECAPA / 256 Resemblyzer).
    Lets matchers and print-counters distinguish native rows from stale
    other-backend enrollments during migration."""
    return 192 if _active_backend == "ecapa" else 256
