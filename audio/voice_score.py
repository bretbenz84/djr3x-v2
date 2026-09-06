"""Model-specific voice storage and score policy.

CAM++ uses raw cosine and its own acceptance/margin settings. ECAPA retains
its historical offset for rollback compatibility. Equal embedding dimensions
never imply compatible models: CAM++ storage is explicitly namespaced.
This module stays import-light to avoid audio/memory import cycles.
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
    """Embedding dimension (192 CAM++/ECAPA, 256 Resemblyzer).
    Use biometric_type as well to distinguish native rows from stale
    other-backend enrollments during migration."""
    return 192 if _active_backend in {"ecapa", "campplus"} else 256


def biometric_type() -> str:
    # CAM++ and ECAPA both output 192 floats: dimension alone is NOT identity.
    return "voice_campplus_zh_en_v1" if _active_backend == "campplus" else "voice"


def signature_table() -> str:
    return "voice_signatures_campplus" if _active_backend == "campplus" else "voice_signatures"


def match_threshold() -> float:
    key = "CAMPPLUS_MATCH_THRESHOLD" if _active_backend == "campplus" else "SPEAKER_ID_SIMILARITY_THRESHOLD"
    return float(getattr(config, key, .50))
