"""
intelligence/callback_engine.py — banked-callback humor.

The brain of the callback-humor feature (design: docs/callback_humor_design.md).
Rex banks durable, light, SELF-volunteered "fun facts" about a person during
conversation (the BANKER), keeps a background read on whether the live topic
connects to one (the RELEVANCE STASH), and — when every tone gate clears —
claims the turn's single callback slot so the reply weaves one premise back in
(the REACTIVE TRIGGER), or hands a premise to the consciousness lull step (the
LULL PICK). Storage is memory/callbacks.py (people.db person_callback_material).

Hard rules this module enforces:

  • Sensitivity wall — a deterministic protected-category regex bank (health,
    grief, body/appearance, orientation, finances, religion/politics, family
    conflict, addiction/legal/immigration) classifies at CAPTURE time and the
    model can only move material toward 'excluded', never toward 'safe'.
    Secondhand/third-party material never enters the pool (the banker only
    sees the speaker's own turns; "tell me about" briefings are consumed
    upstream and never reach it).
  • Subordinate to every sincerity gate — the trigger checks, in order: the
    shared pacing ledger, the sober-room window, comedy mode + social frame,
    live-turn sensitivity + boundary regexes, the empathy cache, tone-repair
    cooldown, unacknowledged emotional events, topic-thread weight/stance,
    crowd discretion, per-person consent boundaries, tier, and the person's
    own callback-style preference. Any failure → silence. Errors → silence.
  • One callback per reply — a successful claim is visible to llm.py's
    callback-hook chain (turn_claim_active), which then skips its other hooks.
  • Spend-at-speak — a premise is only marked used when its words actually
    appear in what Rex SAID (settle_turn), or in the lull path's on_spoke.

Latency: nothing here runs on the time-to-first-speech path. The banker and
relevance judge run on the post-response background thread (local qwen2.5:1.5b
by default — labelled-line output, never JSON, validated, fail-closed). The
reactive trigger is DB reads + regex + one probability roll.
"""

from __future__ import annotations

import logging
import os
import random
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config

_log = logging.getLogger(__name__)

# ── Sensitivity wall ──────────────────────────────────────────────────────────
# Deterministic, testable, model-can't-override. Scanned over premise + topic +
# the source quote. Overbroad by design: losing a borderline premise costs a
# joke; missing a protected one costs trust. Vocabulary aligned with
# empathy._LOCAL_*_PAT, comedy_modes._SENSITIVE_PAT, and
# emotional_events._HEAVY_NEGATIVE_CATEGORIES.
_PROTECTED_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("health", re.compile(
        r"\b(?:cancer|chemo\w*|therap\w*|medicat\w*|meds|diagnos\w*|hospital|"
        r"hospice|surgery|illness|sick|disease|disorder|disab\w*|adhd|autis\w*|"
        r"depress\w*|anxiety|anxious|panic|insomnia|migraine|allerg\w*|injur\w*|"
        r"chronic|pain)\b", re.I)),
    ("grief", re.compile(
        r"\b(?:died|dead|death|passed away|funeral|grief|grieving|memorial|"
        r"widow\w*|loss of)\b", re.I)),
    ("body", re.compile(
        r"\b(?:weight|overweight|skinny|fat|thin|diet\w*|body|bald\w*|acne|"
        r"wrinkl\w*|aging|appearance|attractive|ugly|my height|too short|"
        r"too tall)\b", re.I)),
    ("orientation_romance", re.compile(
        r"\b(?:gay|lesbian|bisexual|queer|trans\w*|nonbinary|non-binary|"
        r"sexuality|orientation|coming out|came out|closet\w*|dating|tinder|"
        r"hinge|crush|girlfriend|boyfriend|sex|sexual|virgin\w*)\b", re.I)),
    ("finances", re.compile(
        r"\b(?:salary|income|debt|broke|bankrupt\w*|loan|mortgage|rent|evict\w*|"
        r"paycheck|fired|laid off|layoff|unemploy\w*|welfare|food stamps)\b", re.I)),
    ("religion_politics", re.compile(
        r"\b(?:christian\w*|catholic|muslim|islam\w*|jewish|judaism|hindu\w*|"
        r"buddhis\w*|atheis\w*|agnostic|church|mosque|synagogue|temple|"
        r"republican|democrat\w*|conservative|liberal|maga|leftist|"
        r"right.wing|left.wing|voted? for)\b", re.I)),
    ("family_conflict", re.compile(
        r"\b(?:divorce\w*|breakup|broke up|custody|estrange\w*|cheat\w*|"
        r"affair|abusive|abuse)\b", re.I)),
    ("addiction_legal", re.compile(
        r"\b(?:addict\w*|sober\w*|sobriety|recovery|relapse|rehab|alcoholi\w*|"
        r"drunk|drugs?|overdose|arrest\w*|jail|prison|felony|lawsuit|sued|"
        r"probation|immigration|visa|deport\w*|undocumented)\b", re.I)),
]

# Empathy modes that mean "the room is not in joke territory" — the suppress
# sets used at social_frame._roast_level plus the kind/crisis variants.
_CARING_MODES = {
    "listen", "support", "validate", "ground", "brief", "kind_default",
    "child_kind", "course_correct", "crisis", "gentle_probe",
    "acknowledge_then_yield",
}
_BLOCKED_PURPOSES = {"closure", "repair", "identity", "answer_ack", "boundary"}

# Categories whose material is identity-level enough to be 'safe' even when the
# person volunteered it flatly; everything else needs a playful/engaged stance
# at capture or it lands in 'guarded' (never joked, kept for audit).
_SAFE_WHEN_FLAT_CATEGORIES = {"passion", "hobby", "project"}

_WORD_RE = re.compile(r"[a-z0-9']+")
_FIELD_RE_CACHE: dict[str, re.Pattern[str]] = {}

# Stopwordy tokens that must not count as premise↔speech overlap evidence.
_OVERLAP_STOPWORDS = {
    "that", "this", "with", "have", "they", "them", "their", "about", "into",
    "really", "very", "like", "likes", "loves", "their", "your", "from",
    "does", "doing", "been", "what", "when", "where", "thing", "things",
    "stuff", "going", "want", "wants", "good", "great", "know", "knows",
}

_BANKER_SYSTEM = (
    "You label ONE user turn for a droid's long-term memory. Reply with ONLY "
    "the requested labelled lines, nothing else — no preamble, no commentary."
)

_RELEVANCE_SYSTEM = (
    "You judge whether a conversation connects to a stored fact. Reply with "
    "ONLY the requested labelled lines, nothing else."
)


@dataclass
class TurnClaim:
    """One reactive callback claim: which premise owns this reply's slot."""
    person_id: int
    premise_id: int
    premise: str
    topic_slug: str
    claimed_at: float = field(default_factory=time.monotonic)


# ── Module state (all guarded by _lock) ───────────────────────────────────────

_lock = threading.Lock()
_session_token: str = uuid.uuid4().hex[:12]
_used_premise_ids: set[int] = set()         # fired this session (cleared at session end)
_fired_count: int = 0                       # session volume across BOTH paths
_last_fired_at: float = 0.0                 # monotonic
_last_fired_transcript_len: int = 0
_last_attempt_transcript_len: int = -10     # failed-settle soft backoff
_last_heavy_at: float = 0.0                 # sober-room anchor (monotonic)
_active_claim: Optional[TurnClaim] = None
_relevance_stash: Optional[dict] = None     # {person_id, premise_id, score, transcript_len, ts}


# ── Gating helpers ────────────────────────────────────────────────────────────

def _under_test_runner() -> bool:
    """Inert under unittest/pytest unless explicitly opted in — the banker and
    relevance judge fire from inside the turn path's background thread and
    would otherwise hit Ollama/OpenAI from unit tests (the arc/rex_pov idiom)."""
    if os.environ.get("DJR3X_CALLBACK_TEST_OPT_IN"):
        return False
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    argv0 = (sys.argv[0] if sys.argv else "").lower()
    return "unittest" in argv0 or "pytest" in argv0 or "py.test" in argv0


def _bank_enabled() -> bool:
    return bool(getattr(config, "CALLBACK_BANK_ENABLED", False))


def _humor_enabled() -> bool:
    return bool(getattr(config, "CALLBACK_HUMOR_ENABLED", False))


def session_token() -> str:
    with _lock:
        return _session_token


def clear_session() -> None:
    """Session teardown: new session token, per-session no-repeat set and the
    volume ledger reset. The sober-room anchor is deliberately KEPT — a heavy
    moment shortly before a session ends shouldn't be joked past by a restart
    of the conversation. (llm.py's never-cleared 'session' sets are a known
    bug-shape; this engine clears for real via interaction._end_session.)"""
    global _session_token, _fired_count, _last_fired_at
    global _last_fired_transcript_len, _last_attempt_transcript_len
    global _active_claim, _relevance_stash
    with _lock:
        _session_token = uuid.uuid4().hex[:12]
        _used_premise_ids.clear()
        _fired_count = 0
        _last_fired_at = 0.0
        _last_fired_transcript_len = 0
        _last_attempt_transcript_len = -10
        _active_claim = None
        _relevance_stash = None


def reset_state_for_tests() -> None:
    """Test helper: full module-state reset, including the sober-room anchor
    that clear_session() deliberately preserves."""
    global _last_heavy_at
    clear_session()
    with _lock:
        _last_heavy_at = 0.0


def note_heavy_moment() -> None:
    """Arm the sober-room window. Called from interaction's deterministic
    sensitive-topic prepass so grief-flow turns (which never reach the claim
    seam) still suppress later callbacks."""
    global _last_heavy_at
    with _lock:
        _last_heavy_at = time.monotonic()


def _heavy_recently() -> bool:
    window = float(getattr(config, "CALLBACK_SUPPRESS_AFTER_HEAVY_SECS", 1800.0))
    with _lock:
        last = _last_heavy_at
    return last > 0.0 and (time.monotonic() - last) < window


def unacked_emotional_event_pending(person_id: Optional[int]) -> bool:
    """A surfaceable, not-yet-acknowledged emotional event exists for this
    person — sincerity claims the turn's callback budget before any humor.
    Shared with llm._build_person_context (single implementation of the check)."""
    if not isinstance(person_id, int):
        return False
    try:
        from memory import emotional_events as _emo
        import world_state
        ws_now = world_state.snapshot()
        crowd = int((ws_now.get("crowd") or {}).get("count", 1) or 1)
        suppress_in_crowd = bool(getattr(config, "EMPATHY_DISCRETION_IN_CROWD", True))
        return any(
            not ev.get("last_acknowledged_at")
            and _emo.can_surface_event(ev)
            and not (suppress_in_crowd and crowd > 1 and _emo.is_heavy_event(ev))
            for ev in _emo.get_active_events(person_id, limit=3)
        )
    except Exception:
        return False


def _empathy_clear(person_id: Optional[int]) -> bool:
    """No fresh empathy signal that the person is in a caring/sensitive state.
    A stale cache entry (past TTL) is no-signal → clear; the live-turn regex
    prepass is checked separately by the caller."""
    try:
        from intelligence import empathy
        entry = empathy.peek(person_id)
        if not entry:
            return True
        ttl = float(getattr(config, "EMPATHY_CACHE_TTL_SECS", 300.0))
        if (time.time() - float(entry.get("ts") or 0.0)) > ttl:
            return True
        mode = str(((entry.get("mode") or {}).get("mode")) or "").lower()
        if mode in _CARING_MODES:
            return False
        result = entry.get("result") or {}
        if str(result.get("affect") or "").lower() in {"sad", "withdrawn", "angry", "anxious"}:
            return False
        if str(result.get("topic_sensitivity") or "none").lower() != "none":
            return False
        return True
    except Exception:
        return False  # fail closed: unknown empathy state → no joke


def _restraint_preferred(person_id: int) -> bool:
    """The person has told Rex to ease off callbacks ('stop bringing that up')
    — captured by friendship_patterns as a callback_style preference fact."""
    try:
        from memory import facts as _facts
        prefs = _facts.get_facts_by_category(person_id, "preference")
        text = " ".join(str(p.get("value") or "").lower() for p in prefs)
        return "prefers callback restraint" in text
    except Exception:
        return False


def _tier_eligible(person_id: int) -> bool:
    try:
        from memory import people as _people
        person = _people.get_person(person_id)
        tier = str((person or {}).get("friendship_tier") or "stranger")
        eligible = tuple(getattr(
            config, "CALLBACK_ELIGIBLE_TIERS",
            ("acquaintance", "friend", "close_friend", "best_friend"),
        ))
        return tier in eligible
    except Exception:
        return False


def _crowd_ok() -> bool:
    try:
        import world_state
        ws = world_state.snapshot()
        crowd = int((ws.get("crowd") or {}).get("count", 1) or 1)
        return crowd <= int(getattr(config, "CALLBACK_MAX_CROWD", 2))
    except Exception:
        return False


def _boundary_blocked(person_id: int, topic_slug: str) -> bool:
    try:
        from memory import boundaries as _boundaries
        topic = (topic_slug or "").replace("_", " ")
        return _boundaries.is_blocked(person_id, "roast", topic) or _boundaries.is_blocked(
            person_id, "mention", topic
        )
    except Exception:
        return True  # fail closed


def _transcript_len() -> int:
    try:
        from memory import conversations
        return len(conversations.get_session_transcript())
    except Exception:
        return 0


def _ledger_allows(now_len: int) -> bool:
    cap = int(getattr(config, "CALLBACK_MAX_PER_SESSION", 2))
    min_gap = int(getattr(config, "CALLBACK_MIN_GAP_EXCHANGES", 8))
    cooldown = float(getattr(config, "CALLBACK_COOLDOWN_SECS", 240.0))
    now = time.monotonic()
    with _lock:
        if _fired_count >= cap:
            return False
        if _last_fired_at and (now - _last_fired_at) < cooldown:
            return False
        if (now_len - _last_fired_transcript_len) < min_gap and _fired_count > 0:
            return False
        # Soft backoff after a claim the model didn't voice: wait two exchanges
        # before offering the slot again so an ignored hook doesn't hammer.
        if (now_len - _last_attempt_transcript_len) < 2:
            return False
    return True


def _record_fire(premise_id: int) -> None:
    global _fired_count, _last_fired_at, _last_fired_transcript_len
    now_len = _transcript_len()
    with _lock:
        _used_premise_ids.add(int(premise_id))
        _fired_count += 1
        _last_fired_at = time.monotonic()
        _last_fired_transcript_len = now_len


# ── Local/cloud generation plumbing ───────────────────────────────────────────

def _generate(prompt: str, *, system: str, max_tokens: int, timeout: float) -> str:
    """Dispatch a short labelled-line generation to the configured banker
    backend. Raises on failure so callers fall back cheaply (the arc's
    dual-backend seam)."""
    backend = str(getattr(config, "CALLBACK_BANK_BACKEND", "local")).lower()
    if backend == "openai":
        from intelligence import llm
        return llm.summarize_conversation_arc(
            prompt, system=system, max_tokens=max_tokens, timeout_secs=timeout
        ).strip()
    from intelligence import local_llm
    return local_llm.generate(
        prompt, system=system, temperature=0.0,
        max_tokens=max_tokens, timeout_secs=timeout,
    ).strip()


def _llm_allowed() -> bool:
    if _under_test_runner():
        return False
    backend = str(getattr(config, "CALLBACK_BANK_BACKEND", "local")).lower()
    if backend == "openai":
        return True
    try:
        from intelligence import local_llm
        return bool(local_llm.enabled())
    except Exception:
        return False


def _field(text: str, label: str) -> str:
    pat = _FIELD_RE_CACHE.get(label)
    if pat is None:
        pat = re.compile(rf"(?mi)^\s*{re.escape(label)}\s*:\s*(.+)$")
        _FIELD_RE_CACHE[label] = pat
    m = pat.search(text or "")
    return m.group(1).strip() if m else ""


def _content_words(text: str, *, min_len: int = 4) -> set[str]:
    return {
        w for w in _WORD_RE.findall((text or "").lower())
        if len(w) >= min_len and w not in _OVERLAP_STOPWORDS
    }


def protected_category_hit(text: str) -> Optional[str]:
    """Name of the first protected category the text trips, or None. Public so
    tests can pin the wall directly."""
    for name, pat in _PROTECTED_PATTERNS:
        if pat.search(text or ""):
            return name
    return None


# ── The banker ────────────────────────────────────────────────────────────────

_HEURISTIC_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    # (pattern, category, premise template) — explicit self-declarations only.
    (re.compile(r"\bi(?:'m| am) (?:really |very |super )?(?:into|obsessed with) ([a-z0-9' \-]{3,40})", re.I),
     "passion", "is into {}"),
    (re.compile(r"\bi love ([a-z0-9' \-]{3,40})", re.I), "passion", "loves {}"),
    (re.compile(r"\bi collect ([a-z0-9' \-]{3,40})", re.I), "hobby", "collects {}"),
    (re.compile(r"\bmy hobby is ([a-z0-9' \-]{3,40})", re.I), "hobby", "their hobby is {}"),
    (re.compile(r"\bi(?:'m| am) (?:building|making|printing|restoring|writing) ([a-z0-9' \-]{3,40})", re.I),
     "project", "is building {}"),
]


def _heuristic_candidate(text: str) -> Optional[dict]:
    for pat, category, template in _HEURISTIC_PATTERNS:
        m = pat.search(text or "")
        if m:
            captured = re.sub(r"\s+", " ", m.group(1)).strip(" '-")
            if len(captured.split()) > 6:
                continue
            return {
                "premise": template.format(captured),
                "topic": captured,
                "category": category,
            }
    return None


def _llm_candidate(text: str) -> Optional[dict]:
    prompt = (
        f'User said: "{text}"\n\n'
        "Did the user reveal a durable, light, personal fact about THEMSELVES — "
        "a passion, hobby, project, quirky habit, a strong opinion about "
        "something trivial, or a self-description?\n"
        "NOT bankable: small talk, moods, plans, questions, requests, anything "
        "about another person, anything about health, body, money, romance, "
        "religion, politics, or grief.\n"
        "Output EXACTLY these four lines, nothing else:\n"
        "Found: yes | no\n"
        "Premise: <one short third-person line, or ->\n"
        "Topic: <one-to-three word label, or ->\n"
        "Category: passion | hobby | project | quirk | opinion | self_description"
    )
    try:
        raw = _generate(prompt, system=_BANKER_SYSTEM, max_tokens=90, timeout=3.0)
    except Exception as exc:
        _log.debug("[callback_engine] banker backend unavailable: %s", exc)
        return None
    if _field(raw, "Found").lower()[:3] != "yes":
        return None
    premise = _field(raw, "Premise").strip(" -")
    topic = _field(raw, "Topic").strip(" -")
    category_raw = _field(raw, "Category").lower()
    category = next(
        (c for c in ("passion", "hobby", "project", "quirk", "opinion", "self_description")
         if c in category_raw),
        "quirk",
    )
    if not premise or not topic:
        return None
    # Anti-hallucination: the premise must be grounded in the actual utterance.
    if not (_content_words(premise) & _content_words(text)):
        return None
    if re.search(r"(?mi)^[~\s>*\-]*(user|rex)\s*:", premise):  # transcript echo
        return None
    return {"premise": premise, "topic": topic, "category": category}


def bank_from_turn(person_id: Optional[int], text: str) -> Optional[int]:
    """Extract and store at most one callback candidate from one user turn.
    Runs on the post-response background thread; caller has already applied
    suppress_memory_learning and the forgotten-terms transcript filter.

    Inert under the test runner (like the arc / rex_pov / rex_db idioms): the
    flags default ON, so without this gate any existing test that drives
    _post_response would write callback rows into the REAL people.db."""
    if _under_test_runner():
        return None
    if not _bank_enabled() or not isinstance(person_id, int):
        return None
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if len(cleaned.split()) < 4:
        return None

    candidate = None
    if _llm_allowed():
        candidate = _llm_candidate(cleaned)
    if candidate is None:
        candidate = _heuristic_candidate(cleaned)
    if candidate is None:
        return None

    # Sensitivity: deterministic wall over everything we'd store; the model
    # never gets a vote on 'excluded'.
    scan = " ".join((candidate["premise"], candidate["topic"], cleaned))
    hit = protected_category_hit(scan)
    playful = False
    try:
        from intelligence import topic_thread
        snap = topic_thread.snapshot() or {}
        playful = str(snap.get("user_stance") or "") in {"playful", "engaged"}
    except Exception:
        playful = False

    from memory import callbacks as callbacks_db
    if hit is not None:
        sensitivity = callbacks_db.SENSITIVITY_EXCLUDED
    elif candidate["category"] in _SAFE_WHEN_FLAT_CATEGORIES or playful:
        sensitivity = callbacks_db.SENSITIVITY_SAFE
    else:
        sensitivity = callbacks_db.SENSITIVITY_GUARDED

    row_id = callbacks_db.bank(
        person_id,
        candidate["premise"],
        category=candidate["category"],
        topic=candidate["topic"],
        sensitivity=sensitivity,
        source_quote=cleaned,
        volunteered_playfully=playful,
        session_id=session_token(),
    )
    if row_id is not None:
        _log.info(
            "[callback_engine] banked premise for person %s: %r (topic=%s, %s%s)",
            person_id, candidate["premise"], candidate["topic"], sensitivity,
            f", wall={hit}" if hit else "",
        )
    return row_id


# ── The relevance stash ───────────────────────────────────────────────────────

def refresh_relevance(person_id: Optional[int]) -> None:
    """Re-judge which (if any) banked premise connects to the live topic.
    Post-turn, background; the reactive trigger reads the stash instantly next
    turn. One-turn lag is deliberate — a callback that lands one exchange into
    a topic reads as wit, not parroting.

    Inert under the test runner — without this, any existing test that drives
    _post_response would open (and create) the REAL people.db from the
    background thread, since the flags default ON."""
    global _relevance_stash
    if _under_test_runner():
        return
    if not _humor_enabled() or not isinstance(person_id, int):
        return
    try:
        from memory import callbacks as callbacks_db
        pool = [
            row for row in callbacks_db.active_pool(person_id)
            if callbacks_db.off_cooldown(row)
            and int(row.get("id") or 0) not in _used_ids_snapshot()
        ]
        if not pool:
            with _lock:
                _relevance_stash = None
            return
        pool = pool[:8]  # keep the judgment prompt small

        from memory import conversations
        transcript = conversations.get_session_transcript()
        recent_user = [
            str(e.get("text") or "") for e in transcript[-6:]
            if str(e.get("speaker") or "").lower() != "rex"
        ][-2:]
        context_bits = list(recent_user)
        try:
            from intelligence import topic_thread
            snap = topic_thread.snapshot() or {}
            if snap.get("label"):
                context_bits.insert(0, f"topic: {snap['label']}")
        except Exception:
            pass
        context = " | ".join(b for b in context_bits if b.strip())
        if not context:
            with _lock:
                _relevance_stash = None
            return

        best_id, score = _judge_relevance(context, pool)
        with _lock:
            if best_id is None:
                _relevance_stash = None
            else:
                _relevance_stash = {
                    "person_id": person_id,
                    "premise_id": best_id,
                    "score": score,
                    "transcript_len": len(transcript),
                    "ts": time.monotonic(),
                }
    except Exception as exc:
        _log.debug("[callback_engine] relevance refresh failed: %s", exc)
        with _lock:
            _relevance_stash = None


def _used_ids_snapshot() -> set[int]:
    with _lock:
        return set(_used_premise_ids)


def _judge_relevance(context: str, pool: list[dict]) -> tuple[Optional[int], float]:
    # Deterministic pass first: a premise whose topic/content words literally
    # appear in the live context is a strong match, no model needed.
    ctx_words = _content_words(context)
    for row in pool:
        premise_words = _content_words(
            f"{(row.get('topic_slug') or '').replace('_', ' ')} {row.get('premise') or ''}"
        )
        if premise_words & ctx_words:
            return int(row["id"]), 1.0

    if not _llm_allowed():
        return None, 0.0
    numbered = "\n".join(
        f"{i + 1}. {row.get('premise')}" for i, row in enumerate(pool)
    )
    prompt = (
        f"Current conversation: {context}\n\n"
        f"Stored facts about this person:\n{numbered}\n\n"
        "Which ONE stored fact (if any) naturally connects to the current "
        "conversation? Only a real topical connection counts.\n"
        "Output EXACTLY these two lines, nothing else:\n"
        "Match: <number, or none>\n"
        "Strength: strong | weak | none"
    )
    try:
        raw = _generate(prompt, system=_RELEVANCE_SYSTEM, max_tokens=24, timeout=2.0)
    except Exception as exc:
        _log.debug("[callback_engine] relevance backend unavailable: %s", exc)
        return None, 0.0
    match = _field(raw, "Match").lower()
    strength = _field(raw, "Strength").lower()
    m = re.search(r"\d+", match)
    if not m or "none" in match:
        return None, 0.0
    idx = int(m.group(0)) - 1
    if not (0 <= idx < len(pool)):
        return None, 0.0
    score = 1.0 if "strong" in strength else (0.4 if "weak" in strength else 0.0)
    if score <= 0.0:
        return None, 0.0
    return int(pool[idx]["id"]), score


# ── The reactive trigger ──────────────────────────────────────────────────────

def turn_claim_active(person_id: Optional[int]) -> bool:
    """Whether THIS turn's callback slot is already claimed for this person —
    consulted by llm._build_person_context so its hook chain stands down.
    A claim lives for one reply; if settle never ran (exception mid-turn), a
    stale claim self-expires rather than muting the chain indefinitely."""
    with _lock:
        claim = _active_claim
    if claim is None or claim.person_id != person_id:
        return False
    return (time.monotonic() - claim.claimed_at) < 60.0


def maybe_claim_reactive(
    person_id: Optional[int],
    text: str,
    *,
    frame,
    comedy_mode,
    turn_plan=None,
) -> Optional[TurnClaim]:
    """Run every tone/pacing gate; on full clearance claim the reply's callback
    slot for the stashed relevant premise. Deterministic only (DB reads, regex,
    one roll) — this sits on the reply path. Returns the claim or None."""
    global _active_claim
    if not _humor_enabled() or not isinstance(person_id, int):
        return None
    now_len = _transcript_len()
    if not _ledger_allows(now_len):
        return None
    if _heavy_recently():
        return None

    # Comedy/frame stance for THIS turn (inherits sensitive→straight, purpose
    # overrides, and roast_level's whole checklist).
    if comedy_mode is not None and not bool(getattr(comedy_mode, "allow_callback", False)):
        return None
    roast = str(getattr(frame, "allow_roast", "none") or "none").lower()
    allowed_roasts = {"normal"}
    if bool(getattr(config, "CALLBACK_ALLOW_LIGHT_ROAST_FRAME", False)):
        allowed_roasts.add("light")
    if roast not in allowed_roasts:
        return None
    if str(getattr(frame, "purpose", "") or "").lower() in _BLOCKED_PURPOSES:
        return None
    if turn_plan is not None and str(getattr(turn_plan, "purpose", "") or "") == "boundary":
        return None

    # Live-turn safety (the async empathy classifier lags one turn; these are
    # the deterministic same-turn reads).
    try:
        from intelligence import empathy as _empathy
        if _empathy.classify_local_sensitivity(text or "") is not None:
            note_heavy_moment()
            return None
    except Exception:
        return None
    try:
        from intelligence import social_frame as _social_frame
        if _social_frame._looks_like_boundary(text or ""):
            return None
    except Exception:
        return None

    if not _empathy_clear(person_id):
        return None
    try:
        from intelligence import repair_moves as _repair
        if _repair.recent_tone_repair(
            float(getattr(config, "TONE_REPAIR_NO_ROAST_SECS", 180.0))
        ):
            return None
    except Exception:
        return None
    if unacked_emotional_event_pending(person_id):
        return None

    # Room reads: heavy thread, avoidant/terse stance, or a flat arc → no bit.
    try:
        from intelligence import topic_thread as _topic_thread
        snap = _topic_thread.snapshot() or {}
        if str(snap.get("emotional_weight") or "") == "heavy":
            return None
        if str(snap.get("user_stance") or "") in {"avoidant", "terse"}:
            return None
        if _topic_thread.arc_reads_flat():
            return None
    except Exception:
        return None

    if not _crowd_ok():
        return None
    if not _tier_eligible(person_id):
        return None
    if _restraint_preferred(person_id):
        return None

    # The stashed relevance verdict, validated fresh and against live consent.
    with _lock:
        stash = dict(_relevance_stash) if _relevance_stash else None
    if not stash or stash.get("person_id") != person_id:
        return None
    max_stale = int(getattr(config, "CALLBACK_RELEVANCE_MAX_STALE_EXCHANGES", 4))
    if (now_len - int(stash.get("transcript_len") or 0)) > max_stale:
        return None
    if float(stash.get("score") or 0.0) < float(
        getattr(config, "CALLBACK_RELEVANCE_MIN_SCORE", 0.5)
    ):
        return None

    try:
        from memory import callbacks as callbacks_db
        row = next(
            (
                r for r in callbacks_db.active_pool(person_id)
                if int(r.get("id") or 0) == int(stash["premise_id"])
            ),
            None,
        )
        if row is None or not callbacks_db.off_cooldown(row):
            return None
    except Exception:
        return None
    premise_id = int(row["id"])
    with _lock:
        if premise_id in _used_premise_ids:
            return None
    if _boundary_blocked(person_id, str(row.get("topic_slug") or "")):
        return None

    if random.random() >= float(getattr(config, "CALLBACK_FIRE_PROBABILITY", 0.6)):
        return None

    claim = TurnClaim(
        person_id=person_id,
        premise_id=premise_id,
        premise=str(row.get("premise") or ""),
        topic_slug=str(row.get("topic_slug") or ""),
    )
    global _last_attempt_transcript_len
    with _lock:
        _active_claim = claim
        _last_attempt_transcript_len = now_len
    _log.info(
        "[callback_engine] claimed reply callback for person %s: %r",
        person_id, claim.premise,
    )
    return claim


def _stem_match(word: str, spoken: set[str]) -> bool:
    """Loose word match: exact, or one contains the other at >=5 chars — so
    'photograph' in the spoken line matches the banked 'astrophotography'."""
    for s in spoken:
        if word == s:
            return True
        if (len(word) >= 5 and word in s) or (len(s) >= 5 and s in word):
            return True
    return False


def settle_turn(spoken_text: str) -> None:
    """Spend-at-speak: after the reply's final text is known, mark the claimed
    premise used ONLY if its words actually made it into what Rex said; an
    ignored hook releases the claim (the soft backoff in the ledger stops an
    immediate retry). Always clears the claim.

    Echo test: the topic word alone does NOT count as voiced — the claim only
    existed because the live topic already connected to the premise, so an
    ordinary on-topic reply (the directive says to skip the bit when the
    moment turned) naturally repeats the topic word without the joke. Spending
    requires a premise-content word beyond the topic, or two matches total;
    a premise with no words beyond its topic falls back to any match (else it
    could never be spent and would retry forever)."""
    global _active_claim
    with _lock:
        claim = _active_claim
        _active_claim = None
    if claim is None:
        return
    spoken_words = _content_words(spoken_text or "")
    topic_words = _content_words(claim.topic_slug.replace("_", " "))
    premise_words = _content_words(claim.premise) | topic_words
    matched = {w for w in premise_words if _stem_match(w, spoken_words)}
    non_topic_premise = premise_words - topic_words
    if non_topic_premise:
        voiced = bool(matched & non_topic_premise) or len(matched) >= 2
    else:
        voiced = bool(matched)
    if voiced:
        try:
            from memory import callbacks as callbacks_db
            callbacks_db.mark_used(claim.premise_id)
        except Exception as exc:
            _log.debug("[callback_engine] mark_used failed: %s", exc)
        _record_fire(claim.premise_id)
        _log.info(
            "[callback_engine] callback FIRED for person %s (premise %s)",
            claim.person_id, claim.premise_id,
        )
    else:
        _log.info("[callback_engine] claim not voiced — released without spend")


def build_callback_directive(claim: TurnClaim) -> str:
    """The per-turn prompt instruction for a claimed premise. Rendered into the
    comedy directive (comedy_modes.with_banked_premise) so the governor layers
    see one coherent comedy stance, not a competing instruction."""
    return (
        f"Callback material — they once told you: \"{claim.premise}\". "
        "If (and only if) it fits the current beat, land ONE short callback "
        "connecting that to what's happening right now — affectionate roast, "
        "dry, specific, never mean. They volunteered this about themselves, so "
        "it's fair game. Don't announce it as a memory, don't explain the "
        "reference, and if the moment has turned sincere, skip it entirely."
    )


# ── The lull pick ─────────────────────────────────────────────────────────────

def lull_gates_clear(person_id: Optional[int]) -> bool:
    """Every person/room gate that applies when there's no live utterance —
    the consciousness lull step calls this before submitting a candidate, and
    again inside speak_fn right before composing."""
    if not _humor_enabled() or not bool(getattr(config, "CALLBACK_LULL_ENABLED", True)):
        return False
    if not isinstance(person_id, int):
        return False
    if not _ledger_allows(_transcript_len()):
        return False
    if _heavy_recently():
        return False
    if not _empathy_clear(person_id):
        return False
    if unacked_emotional_event_pending(person_id):
        return False
    try:
        from intelligence import repair_moves as _repair
        if _repair.recent_tone_repair(
            float(getattr(config, "TONE_REPAIR_NO_ROAST_SECS", 180.0))
        ):
            return False
    except Exception:
        return False
    if not _crowd_ok():
        return False
    if not _tier_eligible(person_id):
        return False
    if _restraint_preferred(person_id):
        return False
    return True


def pick_lull_premise(person_id: int) -> Optional[dict]:
    """Best premise for a quiet-moment callback: fresh (use-decay), boundary-
    clear, with a boost for material banked THIS session ('earlier tonight
    you said…' lands better in a lull than a week-old fact)."""
    try:
        from memory import callbacks as callbacks_db
        used = _used_ids_snapshot()
        same_session_w = float(getattr(config, "CALLBACK_LULL_W_SAME_SESSION", 0.3))
        current = session_token()
        best, best_score = None, 0.0
        for row in callbacks_db.active_pool(person_id):
            if int(row.get("id") or 0) in used:
                continue
            if not callbacks_db.off_cooldown(row):
                continue
            if _boundary_blocked(person_id, str(row.get("topic_slug") or "")):
                continue
            score = callbacks_db.freshness_factor(row)
            if str(row.get("session_id") or "") == current:
                score *= 1.0 + same_session_w
            if int(row.get("volunteered_playfully") or 0):
                score *= 1.15
            if score > best_score:
                best, best_score = row, score
        return best
    except Exception as exc:
        _log.debug("[callback_engine] lull pick failed: %s", exc)
        return None


def spend_lull_premise(premise_row: dict) -> None:
    """on_spoke hook for the lull path — the line actually played."""
    try:
        from memory import callbacks as callbacks_db
        callbacks_db.mark_used(int(premise_row["id"]))
    except Exception as exc:
        _log.debug("[callback_engine] lull mark_used failed: %s", exc)
    _record_fire(int(premise_row.get("id") or 0))


def build_lull_prompt(person_name: str, premise_row: dict) -> str:
    """Compose prompt for the lull line. The proactive compose path carries no
    person dossier, so the safety rules ride in the prompt itself (the
    do_people_roast idiom)."""
    premise = str(premise_row.get("premise") or "")
    return (
        f"The conversation with {person_name} has gone quiet for a moment. "
        f"Earlier, they told you this about themselves: \"{premise}\". "
        "Break the silence with ONE short, dry callback line that playfully "
        "resurfaces that fact — affectionate roast, classic Rex, two sentences "
        "max. It must feel like the thought just drifted in, not like a recap. "
        "Hard rules: never joke about body, age, identity, health, money, "
        "religion, romance, grief, or anything they didn't volunteer; no "
        "questions; don't announce that you're remembering; if you can't make "
        "it land kindly, say something shorter and gentler instead."
    )
