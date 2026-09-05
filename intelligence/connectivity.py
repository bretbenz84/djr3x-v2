"""
intelligence/connectivity.py — offline mode: detect internet loss, degrade, recover.

Owner spec (2026-08-01): "If the mac loses internet, I'd like the program to
degrade rather than stop functioning." Field motivation: an outage at 22:47 that
night wedged ONE turn for 125 s (55 s of OpenAI retries + 62 s of ElevenLabs
timeouts) while AEC suppression held the mic shut — Rex went mute for two minutes
instead of saying his link was down.

Design:
  * `is_online()` is a CACHED state read — never a network call — so hot paths
    can consult it for free.
  * Detection is failure-driven: callers report API failures via `note_failure()`,
    which triggers ONE rate-limited active probe (TCP connect to public anycast
    endpoints, ~1s). No polling at all while healthy.
  * While OFFLINE, a background monitor re-probes every OFFLINE_RECHECK_SECS so
    recovery is noticed within ~20 s.
  * Transitions fire registered callbacks (main.py registers the in-character
    announcer: "my connection to the galactic internet is down…").

What offline mode changes elsewhere (each seam checks `is_offline()`):
  * lean_brain: replies/directives route to the LOCAL model (Ollama, config
    OFFLINE_LLM_MODEL) instead of OpenAI — degraded but alive;
  * audio/tts: skips the ElevenLabs attempt and speaks in the local voice;
  * llm extract_* / vision GPT-4o / web search / news+weather fetch: fast-skip
    instead of each paying a timeout.

Fail-safe: with OFFLINE_MODE_ENABLED False every function is inert and reports
online, so nothing changes.
"""

from __future__ import annotations

import logging
import random
import socket
import threading
import time
from typing import Callable, Optional

import config

_log = logging.getLogger(__name__)

# Public anycast endpoints — reaching ANY one of them means the internet is up.
# (Cloudflare DNS over TLS port, Google DNS. Plain TCP connect, nothing sent.)
_PROBE_ENDPOINTS = (("1.1.1.1", 443), ("8.8.8.8", 53))

_lock = threading.Lock()
_online = True                      # cached state — the only thing hot paths read
_last_probe_at = 0.0                # rate-limits failure-driven probes
_monitor_thread: Optional[threading.Thread] = None
_monitor_stop = threading.Event()
_listeners: list = []               # fn(online: bool) transition callbacks
_offline_since = 0.0


def _enabled() -> bool:
    return bool(getattr(config, "OFFLINE_MODE_ENABLED", True))


def is_online() -> bool:
    """Cached connectivity state. Free to call from hot paths."""
    if not _enabled():
        return True
    return _online


def is_offline() -> bool:
    return not is_online()


def offline_secs() -> float:
    """How long we've been offline (0.0 when online)."""
    with _lock:
        if _online or not _offline_since:
            return 0.0
        return max(0.0, time.monotonic() - _offline_since)


def add_listener(fn: Callable[[bool], None]) -> None:
    """Register a transition callback fn(online). Called OFF the probing thread's
    lock; exceptions are swallowed."""
    _listeners.append(fn)


def _probe(timeout_secs: Optional[float] = None) -> bool:
    """One active check: can we open a TCP connection to any public endpoint?"""
    t = float(timeout_secs or getattr(config, "OFFLINE_PROBE_TIMEOUT_SECS", 1.2))
    for host, port in _PROBE_ENDPOINTS:
        try:
            with socket.create_connection((host, port), timeout=t):
                return True
        except OSError:
            continue
    return False


def _set_state(online: bool) -> None:
    global _online, _offline_since
    fire = False
    with _lock:
        if online != _online:
            _online = online
            _offline_since = 0.0 if online else time.monotonic()
            fire = True
    if not fire:
        return
    if online:
        _log.warning("[connectivity] internet RESTORED — leaving offline mode")
    else:
        _log.warning("[connectivity] internet LOST — entering offline mode "
                     "(local models, external calls blocked)")
        _prewarm_offline_brain()
    for fn in list(_listeners):
        try:
            fn(online)
        except Exception as exc:
            _log.debug("[connectivity] listener failed: %s", exc)


def note_failure(source: str = "") -> bool:
    """Report an external-API failure. Runs ONE rate-limited probe; flips to
    OFFLINE when the probe also fails. Returns the (possibly updated) online
    state so callers can immediately pick the local path this same turn."""
    global _last_probe_at
    if not _enabled():
        return True
    now = time.monotonic()
    with _lock:
        already_offline = not _online
        recent = (now - _last_probe_at) < float(
            getattr(config, "OFFLINE_PROBE_MIN_INTERVAL_SECS", 5.0))
        if not recent:
            _last_probe_at = now
    if already_offline:
        return False
    if recent:
        return _online
    up = _probe()
    if not up:
        _log.info("[connectivity] failure from %s confirmed by probe", source or "?")
        _set_state(False)
        _ensure_monitor()
    return up


def _monitor_loop() -> None:
    """While offline, re-probe periodically; exit once back online."""
    interval = float(getattr(config, "OFFLINE_RECHECK_SECS", 20.0))
    while not _monitor_stop.wait(interval):
        if _online:
            return
        if _probe():
            _set_state(True)
            return


def _ensure_monitor() -> None:
    global _monitor_thread
    if _monitor_thread is not None and _monitor_thread.is_alive():
        return
    _monitor_stop.clear()
    _monitor_thread = threading.Thread(
        target=_monitor_loop, name="connectivity-monitor", daemon=True
    )
    _monitor_thread.start()


def _prewarm_offline_brain() -> None:
    """Load the offline reply model in the background the moment we go offline,
    so the FIRST degraded reply doesn't pay the ~8s model-load (measured) — the
    announcement line covers the warmup instead."""
    def _warm() -> None:
        try:
            from intelligence import local_llm
            for _ in local_llm.stream_chat(
                [{"role": "user", "content": "hi"}], max_tokens=1, timeout_secs=60.0
            ):
                break
            _log.info("[connectivity] offline brain prewarmed")
        except Exception as exc:
            _log.debug("[connectivity] offline-brain prewarm failed: %s", exc)
    threading.Thread(target=_warm, name="offline-brain-prewarm", daemon=True).start()


def check_now() -> bool:
    """Force an immediate probe and update state. Used at startup."""
    if not _enabled():
        return True
    up = _probe()
    _set_state(up)
    if not up:
        _ensure_monitor()
    return up


def stop() -> None:
    _monitor_stop.set()


class OfflineError(RuntimeError):
    """Raised instead of a hosted API call while offline — so callers fail in
    microseconds (their existing except-blocks handle it) instead of paying a
    30s timeout x retries per background call."""


def _count_client(client, label: str):
    """Count every request the client makes (utils.turn_trace), per turn and in
    the process totals, keyed "hosted.<label>" for chat completions and
    "hosted.<label>.responses" for the Responses API. Lean Brain phase 0: the
    per-turn call inventory has to include EVERY hosted request, and every
    module's client passes through guard_client, so this is the one chokepoint.
    Applied regardless of the offline-mode flag. Skips any attribute the client
    (or a test double) does not have."""
    from utils import turn_trace as _tt
    for path, suffix in ((("chat", "completions"), ""), (("responses",), ".responses")):
        try:
            target = client
            for part in path:
                target = getattr(target, part)
            real = target.create
        except Exception:
            continue
        kind = f"hosted.{label}{suffix}"

        def _counted(*args, _real=real, _kind=kind, **kwargs):
            _tt.count(_kind)
            return _real(*args, **kwargs)

        try:
            target.create = _counted
        except Exception as exc:
            _log.debug("[connectivity] count wrapper for %s skipped: %s", kind, exc)
    return client


def guard_client(client, label: str = "openai"):
    """Wrap an OpenAI client's chat.completions.create so that:
      * while OFFLINE it raises OfflineError immediately (no timeout burned);
      * any transport failure reports note_failure(label) — automatic outage
        detection from every call site, no per-caller wiring;
      * every request is counted for the turn telemetry (_count_client — this
        part is applied even when offline mode is disabled).
    Returns the same client (instance-attribute shadowing). Inert when the
    feature is disabled or the client shape is unexpected."""
    try:
        client = _count_client(client, label)
    except Exception as exc:
        _log.debug("[connectivity] call counting for %s skipped: %s", label, exc)
    if not _enabled():
        return client
    try:
        real = client.chat.completions.create

        def _guarded(*args, **kwargs):
            if is_offline():
                raise OfflineError(f"offline mode: {label} call skipped")
            try:
                return real(*args, **kwargs)
            except Exception:
                # Cheap + rate-limited; a non-network API error probes once,
                # confirms the link is fine, and changes nothing.
                note_failure(label)
                raise

        client.chat.completions.create = _guarded
    except Exception as exc:
        _log.debug("[connectivity] guard_client(%s) skipped: %s", label, exc)
    return client


# ── In-character lines ──────────────────────────────────────────────────────────

_OFFLINE_LINES = (
    "Uh oh — my connection to the galactic internet just dropped. I'm running on "
    "local circuits now, so fair warning: I'm going to be a little stupider until "
    "it's back.",
    "Well, that's not great — the galactic internet is down. I've switched to my "
    "backup brain. It's… cozier in here. Dumber, but cozier.",
    "Heads up: I just lost the galactic internet. No weather, no news, no outside "
    "galaxy — just me, my local circuits, and my winning personality.",
)

_ONLINE_LINES = (
    "Galactic internet restored! Full brainpower back online — I almost missed "
    "being smart.",
    "We're back — the galactic internet found me again. Resuming normal levels of "
    "brilliance.",
    "Connection restored. The galaxy and I are on speaking terms again.",
)

_NO_INTERNET_REPLIES = (
    "No can do — my galactic internet link is still down, so the outside galaxy "
    "is a mystery to me right now.",
    "That needs the galactic internet, and mine's out. Local circuits only — ask "
    "me something I'd know off the top of my dome.",
    "Can't reach that from here — the galactic internet is down. I'm flying on "
    "local instruments.",
)


def offline_announcement() -> str:
    return random.choice(_OFFLINE_LINES)


def online_announcement() -> str:
    return random.choice(_ONLINE_LINES)


def no_internet_reply() -> str:
    """The in-character refusal for an internet-requiring ask while offline."""
    return random.choice(_NO_INTERNET_REPLIES)
