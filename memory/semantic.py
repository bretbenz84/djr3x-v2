"""
memory/semantic.py — embedding-cosine relevance backend for unified retrieval.

The keyword overlap in memory.text_match is brittle (it misses synonyms and meaning:
an "ocean" topic never lifts a "sailing" interest). This backend scores topic relevance
by EMBEDDING cosine instead, plugging into memory.retrieval's relevance seam.

It needs a local embed model (`ollama pull nomic-embed-text`) and adds a per-turn
embedding call ON THE REPLY PATH — retrieval runs inside the prompt build, so every
millisecond this module spends is a millisecond before Rex answers. It is built to be
SAFE to leave on: every failure path (model not pulled, Ollama down, malformed
response) falls back to keyword overlap, and a circuit breaker keeps a slow or dead
endpoint from taxing the reply — so turning it on can never make recall worse than
keyword, only better when it's healthy.

BREAKER CONTRACT (rewritten 2026-09-01 after the robot logs from 08-21 through 09-01):
the old breaker needed THREE failures to trip, each a full 2.0 s timeout, and then
re-armed inline after 60 s — so on the robot Mac, where the endpoint never answered
in time during a live session, every turn that came more than a minute after the
last trip stalled the reply by 6 s (llm_first_sentence 6.7-7.3 s vs 0.8-1.2 s on the
other turns; 4 of 12 turns in the 2026-09-01 23:00 run, 6 of 11 on 08-29). Now:

  * ONE inline failure opens the breaker. An inline timeout is already a full
    reply-path stall; there is nothing to learn from two more.
  * While open, inline calls return None immediately (keyword relevance).
  * Recovery is a BACKGROUND probe, never an inline retry. When the cooldown
    expires the next inline call launches a probe thread and still returns None;
    the breaker closes only when the probe round-trips comfortably inside the
    inline budget. A probe that fails or is merely slow re-opens with a doubled
    cooldown (capped), so a flapping endpoint costs the reply at most one short
    timeout per open, and those opens get rarer.
  * warmup() (main.py, background thread at boot) pins the embed model in Ollama
    with keep_alive, measures a cold and a warm round trip, and opens the breaker
    up front if the warm trip is slower than the inline budget — so the first turn
    of a session never pays for discovering the endpoint is slow.

Caching: the live topic is embedded ONCE per turn (memoized by the topic-token key) and
candidate texts are stable strings cached in-process, so after warm-up a turn costs ~one
embed call. (A persistent embedding column is a future optimization.)
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np

_log = logging.getLogger(__name__)

# In-process caches.
_cand_cache: dict[str, np.ndarray] = {}   # SUCCESSES only — see _embed_candidate
_topic_cache: tuple[str, Optional[np.ndarray]] = ("", None)

# Circuit breaker so a slow or dead endpoint doesn't cost the reply path.
_FAIL_THRESHOLD = 1        # one inline failure opens the breaker (see module doc)
_COOLDOWN_SECS = 60.0      # first cooldown; doubles per consecutive failed probe
_COOLDOWN_MAX_SECS = 600.0
_PROBE_TEXT = "rex embedding health probe"

_state_lock = threading.Lock()
_fail_count = 0
_disabled_until = 0.0      # while open: no probe before this instant
_open = False              # True = tripped; inline calls stay off until a probe passes
_cooldown_secs = _COOLDOWN_SECS
_probe_in_flight = False
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


def _inline_timeout() -> float:
    return max(0.2, float(_cfg("MEMORY_SEMANTIC_EMBED_TIMEOUT_SECS", 1.0)))


def _base_cooldown() -> float:
    return max(1.0, float(_cfg("MEMORY_SEMANTIC_BREAKER_COOLDOWN_SECS", _COOLDOWN_SECS)))


def _max_cooldown() -> float:
    return max(_base_cooldown(),
               float(_cfg("MEMORY_SEMANTIC_BREAKER_COOLDOWN_MAX_SECS", _COOLDOWN_MAX_SECS)))


def is_open() -> bool:
    """True while the breaker is tripped (inline embedding off, keyword relevance)."""
    return _open


def _healthy() -> bool:
    """Inline gate. Closed breaker → True. Open → False, and once the cooldown has
    expired, kick off ONE background recovery probe (the inline caller never waits)."""
    if not _open:
        return True
    if time.monotonic() >= _disabled_until:
        _launch_recovery_probe()
    return False


def _open_breaker(reason: str, *, warn: bool = True) -> None:
    """Trip (or re-trip) the breaker and schedule the next probe after the current
    cooldown, which then doubles for the next consecutive failure."""
    global _open, _disabled_until, _cooldown_secs, _fail_count, _warned
    with _state_lock:
        _fail_count = 0
        cooldown = _cooldown_secs
        _disabled_until = time.monotonic() + cooldown
        _cooldown_secs = min(_cooldown_secs * 2.0, _max_cooldown())
        _open = True
        first_warning = not _warned
        _warned = True
    if warn and first_warning:
        # Report the ACTUAL exception, not a guess (see _last_error).
        _log.warning(
            "[semantic] embedding endpoint unavailable — falling back to keyword "
            "relevance; background probe in %.0fs (model=%s, %s: %s)",
            cooldown,
            _cfg("MEMORY_SEMANTIC_EMBED_MODEL", "nomic-embed-text"),
            reason,
            _last_error or "unknown",
        )
    elif warn:
        _log.info(
            "[semantic] embedding endpoint still degraded — next probe in %.0fs (%s: %s)",
            cooldown, reason, _last_error or "unknown",
        )


def _note_failure() -> None:
    global _fail_count
    with _state_lock:
        _fail_count += 1
        tripped = _fail_count >= _FAIL_THRESHOLD
    if tripped:
        _open_breaker("inline embed failed")


def _note_success() -> None:
    """Clear the failure run — and, if the breaker had tripped, close it and say so.

    Both edges are logged (matching the TTS breaker's "ElevenLabs recovered"
    contract): the 2026-08-20 run logged one WARNING and then 24 minutes of
    silence, so it could not answer whether recall was live or degraded."""
    global _fail_count, _warned, _open, _cooldown_secs, _disabled_until
    with _state_lock:
        _fail_count = 0
        was_open = _open
        _open = False
        _disabled_until = 0.0
        _cooldown_secs = _base_cooldown()
        recovered = _warned
        _warned = False
    if recovered or was_open:
        _log.info("[semantic] embedding endpoint recovered — semantic relevance live")


def _request_embedding(text: str, *, timeout: Optional[float] = None) -> np.ndarray:
    """One raw embedding round trip. Raises on ANY failure (transport, HTTP status,
    malformed/empty vector). keep_alive rides on every request so Ollama keeps the
    model resident between turns instead of reloading it under load."""
    import requests
    base = str(_cfg("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
    model = str(_cfg("MEMORY_SEMANTIC_EMBED_MODEL", "nomic-embed-text"))
    payload = {"model": model, "prompt": text}
    keep_alive = _cfg("MEMORY_SEMANTIC_EMBED_KEEP_ALIVE", -1)
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive
    # Lean Brain phase 0: every inline embedding round trip is counted against
    # the turn, so a cold candidate cache shows up as N calls, not as mystery ms.
    from utils import turn_trace as _tt
    _tt.count("embed")
    resp = requests.post(
        f"{base}/api/embeddings",
        json=payload,
        timeout=max(0.2, float(timeout if timeout is not None else _inline_timeout())),
    )
    resp.raise_for_status()
    vec = np.asarray(resp.json().get("embedding") or [], dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if vec.size == 0 or norm <= 1e-10:
        raise ValueError("empty embedding vector")
    return vec / norm


def _embed(text: str) -> Optional[np.ndarray]:
    """Return an L2-normalized embedding for `text`, or None on any failure.
    Never blocks while the breaker is open."""
    global _last_error
    text = (text or "").strip()
    if not text or not _healthy():
        return None
    try:
        vec = _request_embedding(text)
    except Exception as exc:
        _last_error = f"{type(exc).__name__}: {exc}"
        _note_failure()
        _log.debug("[semantic] embed failed: %s", exc)
        return None
    _note_success()
    return vec


# ── Background recovery ──────────────────────────────────────────────────────

def _probe_budget() -> float:
    """A probe must round-trip within this to close the breaker: comfortably under
    the inline budget, so a barely-alive endpoint doesn't re-enable inline calls
    that will only time out again (each of those is a reply-path stall)."""
    return max(0.05, float(_cfg("MEMORY_SEMANTIC_PROBE_MAX_SECS", 0.4)))


def _warmup_timeout() -> float:
    """Budget for a request that may have to LOAD the model. Field 2026-09-02
    00:27:43: under the real live stack the load took 16.66 s (0.3 s on an idle
    machine, and no synthetic load reproduced it). Any request that may trigger a
    load must get this budget, or it becomes the abort loop the old breaker was —
    Ollama kills an in-progress load the moment the client hangs up."""
    return max(5.0, float(_cfg("MEMORY_SEMANTIC_WARMUP_TIMEOUT_SECS", 60.0)))


def _recovery_probe() -> bool:
    """One health probe. Closes the breaker on a fast WARM round trip; re-opens it
    with a longer cooldown on failure or a slow reply. Runs on a background thread.

    Two requests, deliberately: the first gets the LOAD budget (the model may have
    been evicted, and a probe on the inline budget would abort the load exactly the
    way the old breaker did — off the reply path now, but the feature would stay
    dead forever); the second is the one that is timed, because the inline calls it
    would re-enable are warm calls."""
    global _last_error, _probe_in_flight
    try:
        try:
            _request_embedding(_PROBE_TEXT, timeout=_warmup_timeout())
        except Exception as exc:
            _last_error = f"{type(exc).__name__}: {exc}"
            _open_breaker("probe failed")
            return False
        started = time.monotonic()
        try:
            _request_embedding(_PROBE_TEXT + " again", timeout=_inline_timeout())
        except Exception as exc:
            _last_error = f"{type(exc).__name__}: {exc}"
            _open_breaker("probe failed")
            return False
        took = time.monotonic() - started
        budget = _probe_budget()
        if took > budget:
            _last_error = f"probe took {took:.2f}s (budget {budget:.2f}s)"
            _open_breaker("probe too slow")
            return False
        _note_success()
        return True
    finally:
        with _state_lock:
            _probe_in_flight = False


def _launch_recovery_probe() -> bool:
    """Start one background probe if none is in flight. Returns True when launched."""
    global _probe_in_flight
    with _state_lock:
        if _probe_in_flight:
            return False
        _probe_in_flight = True
    threading.Thread(target=_recovery_probe, daemon=True,
                     name="semantic-embed-probe").start()
    return True


def warmup() -> bool:
    """Boot-time warm-up (background thread from main.py): pin the embed model in
    Ollama, measure a cold and a warm round trip, and decide UP FRONT whether inline
    embedding is affordable on this machine. A missing/slow endpoint opens the
    breaker here, so the session's first turn never pays for discovering it."""
    global _last_error
    if not bool(_cfg("MEMORY_SEMANTIC_RECALL_ENABLED", False)):
        return False
    model = str(_cfg("MEMORY_SEMANTIC_EMBED_MODEL", "nomic-embed-text"))
    warm_timeout = _warmup_timeout()
    try:
        started = time.monotonic()
        _request_embedding(_PROBE_TEXT, timeout=warm_timeout)
        cold = time.monotonic() - started
        started = time.monotonic()
        _request_embedding(_PROBE_TEXT + " again", timeout=warm_timeout)
        warm = time.monotonic() - started
    except Exception as exc:
        _last_error = f"{type(exc).__name__}: {exc}"
        _open_breaker("warm-up failed")
        return False
    budget = _probe_budget()
    _log.info(
        "[semantic] embed model %s warm — cold %.2fs, warm %.3fs (keep_alive=%r, "
        "inline timeout %.1fs, probe budget %.2fs)",
        model, cold, warm, _cfg("MEMORY_SEMANTIC_EMBED_KEEP_ALIVE", -1),
        _inline_timeout(), budget,
    )
    if warm > budget:
        _last_error = f"warm round trip {warm:.2f}s exceeds probe budget {budget:.2f}s"
        _open_breaker("warm-up too slow")
        return False
    _note_success()
    return True


# ── Relevance backend ────────────────────────────────────────────────────────

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
    global _open, _cooldown_secs, _probe_in_flight, _last_error
    with _state_lock:
        _cand_cache = {}
        _topic_cache = ("", None)
        _fail_count = 0
        _disabled_until = 0.0
        _warned = False
        _open = False
        _cooldown_secs = _base_cooldown()
        _probe_in_flight = False
        _last_error = ""
