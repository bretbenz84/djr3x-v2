"""
memory/semantic.py — embedding-cosine relevance backend for unified retrieval.

The keyword overlap in memory.text_match is brittle (it misses synonyms and meaning:
an "ocean" topic never lifts a "sailing" interest). This backend scores topic relevance
by EMBEDDING cosine instead, plugging into memory.retrieval's relevance seam.

OPT-IN (config.MEMORY_SEMANTIC_RECALL_ENABLED, default off): it needs a local embed model
(`ollama pull nomic-embed-text`) and adds a per-turn embedding call. It is built to be
SAFE to leave on: every failure path (model not pulled, Ollama down, malformed response)
falls back to keyword overlap, and a circuit breaker stops hammering a dead endpoint — so
turning it on can never make recall worse than keyword, only better when it's healthy.

Caching: the live topic is embedded ONCE per turn (memoized by the topic-token key) and
candidate texts are stable strings cached in-process, so after warm-up a turn costs ~one
embed call. (A persistent embedding column is a future optimization.)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

_log = logging.getLogger(__name__)

# In-process caches.
_cand_cache: dict[str, np.ndarray] = {}   # SUCCESSES only — see _embed_candidate
_topic_cache: tuple[str, Optional[np.ndarray]] = ("", None)

# Circuit breaker so a dead endpoint doesn't cost a timeout per candidate per turn.
_fail_count = 0
_disabled_until = 0.0
_FAIL_THRESHOLD = 3
_COOLDOWN_SECS = 60.0
_warned = False
# Last embed exception, surfaced in the breaker warning. Without it the warning
# could only GUESS at the cause ("is the model pulled?"), and on 2026-08-20 that
# guess was wrong — the model was pulled and answering direct requests fine, so the
# investigation started in the wrong place.
_last_error: str = ""


def _cfg(name: str, default):
    try:
        import config
        return getattr(config, name, default)
    except Exception:
        return default


def _healthy() -> bool:
    return time.monotonic() >= _disabled_until


def _note_failure() -> None:
    global _fail_count, _disabled_until, _warned
    _fail_count += 1
    if _fail_count >= _FAIL_THRESHOLD:
        _disabled_until = time.monotonic() + _COOLDOWN_SECS
        _fail_count = 0
        if not _warned:
            # Report the ACTUAL exception, not a guess. The old line asked whether
            # the model was pulled; on the robot Mac 2026-08-20 it demonstrably was
            # (`nomic-embed-text:latest`, answering direct curl requests fine), so
            # the hint sent the investigation the wrong way. `exc` comes from the
            # caller — _embed already has it and only logged it at DEBUG.
            _log.warning(
                "[semantic] embedding endpoint unavailable — falling back to keyword "
                "relevance for %.0fs (model=%s, last error: %s)",
                _COOLDOWN_SECS,
                _cfg("MEMORY_SEMANTIC_EMBED_MODEL", "nomic-embed-text"),
                _last_error or "unknown",
            )
            _warned = True


def _note_success() -> None:
    """Clear the failure run — and, if the breaker had tripped, say so.

    _warned used to latch for the life of the process, so the log recorded the
    FIRST degradation and nothing after it: no recovery, and no re-trip. The
    2026-08-20 run therefore cannot answer whether semantic recall was live or
    degraded across its 78 conversational turns — one WARNING at 20:09:08 and 24
    minutes of silence. Both edges are logged now, matching the TTS breaker's
    "ElevenLabs recovered" contract."""
    global _fail_count, _warned
    _fail_count = 0
    if _warned:
        _warned = False
        _log.info("[semantic] embedding endpoint recovered — semantic relevance live")


def _embed(text: str) -> Optional[np.ndarray]:
    """Return an L2-normalized embedding for `text`, or None on any failure."""
    text = (text or "").strip()
    if not text or not _healthy():
        return None
    try:
        import requests
        base = str(_cfg("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        model = str(_cfg("MEMORY_SEMANTIC_EMBED_MODEL", "nomic-embed-text"))
        timeout = float(_cfg("MEMORY_SEMANTIC_EMBED_TIMEOUT_SECS", 2.0))
        resp = requests.post(
            f"{base}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=max(0.2, timeout),
        )
        resp.raise_for_status()
        vec = np.asarray(resp.json().get("embedding") or [], dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if vec.size == 0 or norm <= 1e-10:
            _note_failure()
            return None
        _note_success()
        return vec / norm
    except Exception as exc:
        global _last_error
        _last_error = f"{type(exc).__name__}: {exc}"
        _note_failure()
        _log.debug("[semantic] embed failed: %s", exc)
        return None


def _embed_candidate(text: str) -> Optional[np.ndarray]:
    if text in _cand_cache:
        return _cand_cache[text]
    vec = _embed(text)
    # Cache SUCCESSES only. A None memoized during an outage is never retried —
    # _cand_cache is cleared only on overflow (1024 entries) — so a transient
    # endpoint blip permanently demoted those exact candidate texts to keyword
    # matching for the rest of the run.
    if vec is None:
        return None
    cap = int(_cfg("MEMORY_SEMANTIC_CACHE_SIZE", 1024))
    if len(_cand_cache) >= max(16, cap):
        _cand_cache.clear()
    _cand_cache[text] = vec
    return vec


def _topic_vector(topic_tokens) -> Optional[np.ndarray]:
    """Embed the live topic once per turn, memoized by its token key."""
    global _topic_cache
    key = " ".join(sorted(str(t) for t in topic_tokens)) if topic_tokens else ""
    if not key:
        return None
    if _topic_cache[0] == key:
        return _topic_cache[1]
    vec = _embed(key)
    if vec is None:
        # Same negative-caching trap as _embed_candidate, and worse here: the topic
        # key holds until the conversation's topic tokens change, so one failure
        # could mute semantic recall for a whole subject.
        return None
    _topic_cache = (key, vec)
    return vec


def relevance(topic_tokens, text: str, cap: int) -> float:
    """retrieval relevance backend: scaled embedding cosine in [0, cap]. Falls back to
    keyword overlap whenever embeddings are unavailable, so it's never worse than keyword."""
    tvec = _topic_vector(topic_tokens)
    cvec = _embed_candidate(text) if tvec is not None else None
    if tvec is None or cvec is None or tvec.shape != cvec.shape:
        from memory import text_match
        return float(min(text_match.overlap_count(text, topic_tokens), cap))
    cos = float(np.dot(tvec, cvec))
    floor = float(_cfg("MEMORY_SEMANTIC_FLOOR", 0.55))
    scaled = max(0.0, (cos - floor) / max(1e-6, 1.0 - floor))
    return scaled * float(cap)


def reset_cache() -> None:
    """Test/diagnostic hook: clear caches + circuit breaker."""
    global _cand_cache, _topic_cache, _fail_count, _disabled_until, _warned
    _cand_cache = {}
    _topic_cache = ("", None)
    _fail_count = 0
    _disabled_until = 0.0
    _warned = False
