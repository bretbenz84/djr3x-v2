"""Opportunistic impressions — the ones nobody asked for.

The explicit flow (features/impersonation.py) is a REQUEST: "impersonate Jimmy
Carter" → stall line → thinking loop while the clone renders → the bit. This
module is the UNPROMPTED version, and it deliberately sounds different:

- Somebody mentions a famous person Rex has a voice for ("I'm going to Jimmy
  Carter's hometown"), or says something mock-worthy in a voice Rex has captured
  of THEM. That claim is made per reply turn, next to the banked-callback claim.
- The script + clone start rendering in the background IMMEDIATELY, and Rex's
  ordinary ElevenLabs reply covers the wait — no stall line, no thinking chirp
  (owner note 2026-08-18: the processing loop is fine when you asked for an
  impression and wrong when you didn't).
- When the reply is over, the take is rendered, and the proactive gates say the
  floor is free, Rex bridges in his own voice ("Oh — hang on. That reminds
  me...") and plays the take, then takes his bow. If that moment never comes
  inside the wait budget — the person kept talking, the room went heavy, the
  render was slow — the bit is dropped silently. An impression that arrives late
  is worse than one that never happens.

Two triggers, one player:

  famous mention  — deterministic roster scan of the utterance against
                    assets/voices/famous (full name, alias slug, or a title +
                    surname; a bare surname is NOT enough — "Ford", "Bush",
                    "Johnson", "Carter" are all people you know).
  self-mock       — the speaker has a captured voice ref and just said something
                    that reads as absurd/boastful/whiny; ONE small LLM call both
                    judges and writes the ≤18-word playback line (or says NONE).
                    Gated on the social frame allowing a roast, a probability
                    dial, and a long cooldown, so it stays a rare surprise and a
                    bounded cost.

Every fire records an episode with `detail.trigger` = what programmatically
caused it, so "why did you just do Carter?" has a true answer on file.
"""

from __future__ import annotations

import logging
import random
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import config
from audio import local_tts

logger = logging.getLogger(__name__)


# ── State ────────────────────────────────────────────────────────────────────

@dataclass
class _Prep:
    kind: str                                   # "famous" | "self"
    ref: local_tts.VoiceRef
    subject_name: str
    person_id: Optional[int]
    utterance: str
    trigger: str
    claimed_at: float = field(default_factory=time.monotonic)
    max_wait_secs: float = 60.0
    script: Optional[str] = None
    speech_text: str = ""
    take: Optional[local_tts.Take] = None
    prepared: threading.Event = field(default_factory=threading.Event)
    reply_done: threading.Event = field(default_factory=threading.Event)
    cancelled: bool = False
    failed: bool = False
    voice_key: str = ""

    def deadline(self) -> float:
        return self.claimed_at + float(self.max_wait_secs)


_lock = threading.Lock()
_pending: Optional[_Prep] = None
_last_fire_at: float = 0.0
_voice_last_fire: dict[str, float] = {}
_session_fires: int = 0
_recent_self_mock_scripts: list[str] = []


def reset_state() -> None:
    """Tests only."""
    global _pending, _last_fire_at, _session_fires
    with _lock:
        if _pending is not None:
            _pending.cancelled = True
        _pending = None
        _last_fire_at = 0.0
        _session_fires = 0
        _voice_last_fire.clear()
        _recent_self_mock_scripts.clear()


# ── Enablement / roster ──────────────────────────────────────────────────────

def enabled() -> bool:
    if not bool(getattr(config, "IMPERSONATION_ORGANIC_ENABLED", True)):
        return False
    try:
        from features import impersonation
        return impersonation.is_enabled()
    except Exception:
        return False


def _famous_dir() -> Path:
    from features import impersonation
    return impersonation._voices_dir() / "famous"


_TITLE_RE = r"(?:president|senator|governor|secretary|mr\.?|mister|dr\.?|doctor|chancellor|prime minister)"

_roster_cache: Optional[tuple[float, list[tuple[str, re.Pattern]]]] = None


def _roster() -> list[tuple[str, re.Pattern]]:
    """(slug, pattern) per famous clip, longest names first. Cached for a minute —
    the folder changes about never, but tests swap it."""
    global _roster_cache
    now = time.monotonic()
    if _roster_cache is not None and now - _roster_cache[0] < 60.0:
        return _roster_cache[1]
    entries: list[tuple[str, re.Pattern]] = []
    try:
        base = _famous_dir()
        stems = sorted({p.stem for p in base.glob("*.wav")}) if base.exists() else []
    except Exception:
        stems = []
    for stem in stems:
        tokens = [t for t in stem.split("-") if t]
        if not tokens:
            continue
        if len(tokens) == 1:
            if len(tokens[0]) < 3:
                continue
            # Alias slugs (fdr, jfk, ike) and mononyms: whole word only.
            pat = re.compile(rf"\b{re.escape(tokens[0])}\b", re.IGNORECASE)
        else:
            full = r"\W+".join(re.escape(t) for t in tokens)
            surname = re.escape(tokens[-1])
            pat = re.compile(
                rf"\b(?:{full}|{_TITLE_RE}\s+{surname})\b", re.IGNORECASE
            )
        entries.append((stem, pat))
    entries.sort(key=lambda e: -len(e[0]))
    _roster_cache = (now, entries)
    return entries


def invalidate_roster() -> None:
    global _roster_cache
    _roster_cache = None


def detect_famous_mention(text: str) -> Optional[tuple[str, local_tts.VoiceRef]]:
    """A famous voice named in ordinary conversation → (display name, ref)."""
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return None
    for slug, pat in _roster():
        m = pat.search(cleaned)
        if not m:
            continue
        from features import impersonation
        ref = impersonation.find_famous_ref(slug.replace("-", " "))
        if ref is None:
            continue
        name = _ALIAS_DISPLAY.get(slug) or " ".join(t.capitalize() for t in slug.split("-"))
        return name, ref
    return None


_ALIAS_DISPLAY = {"fdr": "FDR", "jfk": "JFK", "lbj": "LBJ", "ike": "Ike"}


def _voice_key(ref: local_tts.VoiceRef) -> str:
    """Cooldown key that sees through alias symlinks (fdr → franklin-roosevelt)."""
    try:
        stem = Path(str(getattr(ref, "wav_path", "") or "")).resolve().stem
        if stem:
            return f"famous:{stem}"
    except Exception:
        pass
    return getattr(ref, "label", "") or ""


_EXPLICIT_RE = re.compile(
    r"\b(impersonat|impression of|imitat|do (?:a |an )?(?:voice|impression)|talk like|sound like)",
    re.IGNORECASE,
)


# ── Claim (called per reply turn, from interaction._stream_llm_response) ─────

def maybe_claim(text: str, person_id: Optional[int], *, frame=None) -> Optional[str]:
    """Look for an opportunity in this turn. Starts background prep and returns
    a directive line for the reply model (or None). Never blocks on the LLM or
    the synthesizer.
    """
    global _pending
    if not enabled():
        return None
    cleaned = " ".join((text or "").split())
    if not cleaned or _EXPLICIT_RE.search(cleaned):
        return None
    try:
        from intelligence import connectivity
        if connectivity.is_offline():
            return None
    except Exception:
        pass
    now = time.monotonic()
    with _lock:
        if _pending is not None and not _pending.cancelled:
            # One bit in flight at a time; a fresh trigger does not queue behind it.
            return None
        if _session_fires >= int(getattr(config, "IMPERSONATION_ORGANIC_MAX_PER_SESSION", 4)):
            return None
        min_gap = float(getattr(config, "IMPERSONATION_ORGANIC_MIN_GAP_SECS", 600.0))
        if _last_fire_at and now - _last_fire_at < min_gap:
            return None

    roast = str(getattr(frame, "allow_roast", "normal") or "normal")

    hit = detect_famous_mention(cleaned)
    if hit is not None and roast != "none":
        name, ref = hit
        key = _voice_key(ref) or name
        voice_gap = float(getattr(config, "IMPERSONATION_ORGANIC_VOICE_MIN_GAP_SECS", 3600.0))
        last = _voice_last_fire.get(key, 0.0)
        if last and now - last < voice_gap:
            return None
        prep = _Prep(
            kind="famous", ref=ref, subject_name=name, person_id=None,
            utterance=cleaned, trigger=f"mention:{key}",
            max_wait_secs=float(getattr(config, "IMPERSONATION_ORGANIC_MAX_WAIT_SECS", 60.0)),
            voice_key=key,
        )
        _launch(prep)
        logger.info("[organic_impersonation] famous mention claimed: %s (%s)", name, key)
        return (
            f"A real voice impression of {name} is already being prepared and will play "
            f"in {name}'s cloned voice shortly after this reply. Reply to what they said "
            f"as yourself. Do NOT do a {name} impression, quote {name}, announce an "
            f"impression, or call any impersonation tool in this reply — it is handled."
        )

    if _self_mock_eligible(cleaned, person_id, roast):
        from features import impersonation
        ref = impersonation.person_ref(int(person_id))  # type: ignore[arg-type]
        if ref is None:
            return None
        name = _person_name(person_id) or "you"
        key = getattr(ref, "label", "") or f"person:{person_id}"
        prep = _Prep(
            kind="self", ref=ref, subject_name=name, person_id=person_id,
            utterance=cleaned, trigger="self_mock:judged",
            max_wait_secs=float(getattr(config, "IMPERSONATION_SELF_MOCK_MAX_WAIT_SECS", 35.0)),
            voice_key=key,
        )
        _launch(prep)
        logger.info("[organic_impersonation] self-mock considered for %s", name)
        # No directive: the reply stays exactly what it would have been. The
        # judge may say NONE, and a reply that braced for a bit that never came
        # would give the game away.
        return None
    return None


def _self_mock_eligible(text: str, person_id: Optional[int], roast: str) -> bool:
    if not bool(getattr(config, "IMPERSONATION_SELF_MOCK_ENABLED", True)):
        return False
    if person_id is None or roast not in {"normal", "sharp"}:
        return False
    if len(text.split()) < int(getattr(config, "IMPERSONATION_SELF_MOCK_MIN_UTTERANCE_WORDS", 4)):
        return False
    gap = float(getattr(config, "IMPERSONATION_SELF_MOCK_MIN_GAP_SECS", 900.0))
    last = _voice_last_fire.get(f"person:{person_id}", 0.0)
    if last and time.monotonic() - last < gap:
        return False
    prob = float(getattr(config, "IMPERSONATION_SELF_MOCK_CONSIDER_PROB", 0.5))
    if random.random() >= prob:
        return False
    try:
        from features import impersonation
        return impersonation.person_ref(int(person_id)) is not None
    except Exception:
        return False


def _person_name(person_id: Optional[int]) -> str:
    if person_id is None:
        return ""
    try:
        from memory import people as people_db
        row = people_db.get_person(int(person_id))
        return str((row or {}).get("name") or "").strip()
    except Exception:
        return ""


def _launch(prep: _Prep) -> None:
    global _pending
    with _lock:
        _pending = prep
    threading.Thread(target=_prepare, args=(prep,), daemon=True,
                     name="organic-impersonation-prep").start()
    threading.Thread(target=_player, args=(prep,), daemon=True,
                     name="organic-impersonation-play").start()


# ── Background prep: script → take ───────────────────────────────────────────

def _prepare(prep: _Prep) -> None:
    try:
        if prep.kind == "famous":
            script = _famous_script(prep)
        else:
            script = _self_mock_script(prep)
        if prep.cancelled:
            return
        if not script:
            prep.failed = True
            logger.info("[organic_impersonation] no script (%s) — dropped", prep.trigger)
            return
        prep.script = script
        speech_text = script
        try:
            from audio import tts as tts_module
            speech_text = tts_module.spoken_form(script) or script
        except Exception:
            pass
        prep.speech_text = speech_text
        # --local-tts: the reply itself needs the engine and generation is
        # serialized, so the take waits for the reply to finish. Otherwise start
        # rendering now — the ElevenLabs reply is the cover.
        if bool(getattr(config, "LOCAL_TTS_MODE", False)):
            prep.reply_done.wait(timeout=max(1.0, prep.deadline() - time.monotonic()))
            if prep.cancelled:
                return
        prep.take = local_tts.start_take(speech_text, prep.ref)
    except Exception as exc:
        prep.failed = True
        logger.warning("[organic_impersonation] prep failed: %s", exc)
    finally:
        prep.prepared.set()


def _famous_script(prep: _Prep) -> Optional[str]:
    from features import impersonation
    max_words = int(getattr(config, "IMPERSONATION_ORGANIC_SCRIPT_MAX_WORDS", 30))
    return impersonation.build_parody_script(
        prep.subject_name, None,
        voice_key=(getattr(prep.ref, "label", "") or prep.voice_key),
        context=prep.utterance, max_words=max_words,
    )


def _self_mock_prompt(prep: _Prep, do_not: list[str], avoid: list[str]) -> str:
    name = prep.subject_name or "the person"
    max_words = int(getattr(config, "IMPERSONATION_SELF_MOCK_MAX_WORDS", 18))
    parts = [
        f"You are DJ-R3X, a witty Star Wars droid. {name} just said to you: "
        f"\"{prep.utterance}\"",
        f"You have {name}'s cloned voice and could play back a short mocking "
        f"impression of them in their OWN voice — the way a friend repeats your "
        f"words back in a whiny voice. Decide first whether this line is genuinely "
        f"MOCK-WORTHY: an absurd claim, an over-the-top boast, a spectacularly bad "
        f"idea, a dramatic complaint, a flimsy excuse, an obvious contradiction, "
        f"whining, or confidently wrong. Ordinary chatter, questions, requests, "
        f"instructions to you, real problems, and anything sad, vulnerable, or "
        f"about health, grief, money, family, work stress, or relationships is NOT "
        f"mock-worthy.",
        f"If it is NOT mock-worthy, output exactly: NONE",
        f"If it IS: output ONE line, at most {max_words} words, in {name}'s "
        f"first-person voice, exaggerating what they just said — affectionate, PG, "
        f"punching at the silliness, never at the person. No quotation marks, no "
        f"stage directions, no tags, no preamble. Just the line, or NONE.",
    ]
    if do_not:
        parts.append(
            "Hard boundaries — if the line touches any of these, output NONE:\n- "
            + "\n- ".join(do_not[:12])
        )
    if avoid:
        parts.append(
            "You have played this trick before. Do not reuse a joke or shape from:\n- "
            + "\n- ".join(avoid[:5])
        )
    return "\n\n".join(parts)


def _self_mock_script(prep: _Prep) -> Optional[str]:
    from features import impersonation
    do_not: list[str] = []
    if prep.person_id is not None:
        try:
            _material, do_not = impersonation._gather_material(int(prep.person_id))
        except Exception:
            do_not = []
    prompt = _self_mock_prompt(prep, do_not, list(_recent_self_mock_scripts))
    try:
        from intelligence import llm
        resp = llm._client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=80,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning("[organic_impersonation] self-mock judge failed: %s", exc)
        return None
    text = llm.clean_response_text(raw) if raw else ""
    text = (text or "").strip().strip('"').strip()
    if not text or text.upper().startswith("NONE"):
        return None
    max_words = int(getattr(config, "IMPERSONATION_SELF_MOCK_MAX_WORDS", 18))
    words = text.split()
    if len(words) > max_words + 6:
        text = " ".join(words[:max_words]).rstrip(",;:") + "."
    _recent_self_mock_scripts.append(text)
    del _recent_self_mock_scripts[:-6]
    return text


# ── Player: wait for the moment, then perform ────────────────────────────────

def note_reply_done() -> None:
    """The reply turn that made the claim has finished speaking (or was skipped).
    Called from the main turn handler after every reply, claim or no claim."""
    with _lock:
        prep = _pending
    if prep is not None:
        prep.reply_done.set()


def cancel(reason: str) -> None:
    """Drop the pending bit (explicit impersonation request, shutdown, ...)."""
    global _pending
    with _lock:
        prep = _pending
        _pending = None
    if prep is None or prep.cancelled:
        return
    prep.cancelled = True
    prep.reply_done.set()
    logger.info("[organic_impersonation] cancelled (%s): %s", reason, prep.trigger)
    _release(prep)


def has_pending() -> bool:
    with _lock:
        return _pending is not None and not _pending.cancelled


def _release(prep: _Prep) -> None:
    take = prep.take
    if take is None:
        return
    try:
        local_tts.pop_take(prep.speech_text, prep.ref)
        take.close()
    except Exception as exc:
        logger.debug("[organic_impersonation] take release failed: %s", exc)


def _floor_is_free() -> bool:
    try:
        from audio import speech_queue
        if not speech_queue.is_drained():
            return False
    except Exception:
        return False
    try:
        from intelligence import speech_engine
        return bool(speech_engine.can_proactive_speak(reactive=True))
    except Exception:
        return False


def _finish(prep: _Prep, outcome: str) -> None:
    global _pending
    with _lock:
        if _pending is prep:
            _pending = None
    if outcome != "spoken":
        _release(prep)
    logger.info("[organic_impersonation] %s: %s", outcome, prep.trigger)


def _player(prep: _Prep) -> None:
    poll = 0.25
    try:
        # 1. Script + take launched (or given up).
        while not prep.prepared.is_set():
            if prep.cancelled or time.monotonic() > prep.deadline():
                return _finish(prep, "cancelled" if prep.cancelled else "expired_preparing")
            prep.prepared.wait(timeout=poll)
        if prep.cancelled or prep.failed or prep.take is None:
            return _finish(prep, "cancelled" if prep.cancelled else "no_take")
        # 2. The reply that carried the claim has been spoken.
        while not prep.reply_done.is_set():
            if prep.cancelled or time.monotonic() > prep.deadline():
                return _finish(prep, "cancelled" if prep.cancelled else "expired_reply")
            prep.reply_done.wait(timeout=poll)
        # 3. The clone is rendered. NO thinking loop here — silence between turns
        #    is just Rex not talking; the chirp would announce a bit was coming.
        while not prep.take.first_ready.is_set():
            if prep.cancelled or time.monotonic() > prep.deadline():
                return _finish(prep, "cancelled" if prep.cancelled else "expired_rendering")
            prep.take.first_ready.wait(timeout=poll)
        if prep.take.failed or prep.take.is_closed:
            return _finish(prep, "take_failed")
        # 4. The floor is free: nothing playing, nobody mid-sentence, no game/DJ/
        #    heavy-moment window. Wait for it, but not forever.
        while not _floor_is_free():
            if prep.cancelled or time.monotonic() > prep.deadline():
                return _finish(prep, "cancelled" if prep.cancelled else "expired_waiting_floor")
            time.sleep(poll)
        if prep.cancelled or prep.take.is_closed:
            return _finish(prep, "cancelled")
        _perform(prep)
        _finish(prep, "spoken")
    except Exception as exc:
        logger.warning("[organic_impersonation] player failed: %s", exc)
        _finish(prep, "error")


def _bridge_line(prep: _Prep) -> str:
    from features import impersonation
    if prep.kind == "self":
        lines = getattr(config, "IMPERSONATION_SELF_MOCK_BRIDGE_LINES", [])
        fallback = "Hang on. Let me play that back for you."
        return impersonation._pick_cycling(
            lines, fallback, "organic_self_mock_bridge.json",
            "IMPERSONATION_SELF_MOCK_BRIDGE_STATE_PATH",
        )
    lines = getattr(config, "IMPERSONATION_ORGANIC_BRIDGE_LINES", [])
    fallback = "Oh — hang on. That reminds me. {name}, everybody:"
    picked = impersonation._pick_cycling(
        lines, fallback, "organic_bridge.json", "IMPERSONATION_ORGANIC_BRIDGE_STATE_PATH",
    )
    try:
        return picked.format(name=prep.subject_name)
    except Exception:
        return picked.replace("{name}", prep.subject_name)


def _outro_line(prep: _Prep) -> Optional[str]:
    from features import impersonation
    if prep.kind == "self":
        if not bool(getattr(config, "IMPERSONATION_SELF_MOCK_OUTRO_ENABLED", True)):
            return None
        return impersonation._pick_cycling(
            getattr(config, "IMPERSONATION_SELF_MOCK_OUTRO_LINES", []),
            "That's you. That's what you sound like.",
            "organic_self_mock_outro.json", "IMPERSONATION_SELF_MOCK_OUTRO_STATE_PATH",
        )
    return impersonation.outro_line()


def _perform(prep: _Prep) -> None:
    from audio import speech_queue

    def _say(line: str, emotion: str, *, voice_ref=None, log_text: bool) -> None:
        try:
            done = speech_queue.enqueue(
                line, emotion, priority=1, tag="impersonation",
                voice_ref=voice_ref, log_text=log_text,
            )
            done.wait(timeout=45.0)
        except Exception as exc:
            logger.debug("[organic_impersonation] enqueue failed: %s", exc)

    global _last_fire_at, _session_fires
    now = time.monotonic()
    with _lock:
        _last_fire_at = now
        _session_fires += 1
        _voice_last_fire[prep.voice_key] = now
        if prep.kind == "self" and prep.person_id is not None:
            _voice_last_fire[f"person:{prep.person_id}"] = now

    try:
        from intelligence import decision_ledger
        decision_ledger.record(
            "impression",
            (f"{prep.subject_name} said something mock-worthy (\"{prep.utterance[:80]}\") "
             f"and I have their voice on file, so I played it back at them"
             if prep.kind == "self" else
             f"someone mentioned {prep.subject_name} (\"{prep.utterance[:80]}\") and I "
             f"have that voice on file, so I slipped in an impression"),
            said=prep.script or "", detail={"trigger": prep.trigger},
        )
    except Exception:
        pass
    _say(_bridge_line(prep), "amused" if prep.kind == "self" else "excited", log_text=True)
    if prep.cancelled or prep.take is None or prep.take.is_closed:
        return
    script = prep.script or ""
    try:
        from utils import conv_log
        conv_log.log_rex(script)
    except Exception:
        pass
    try:
        _say(prep.speech_text, "excited", voice_ref=prep.ref, log_text=False)
    finally:
        _release(prep)
    outro = _outro_line(prep)
    if outro:
        _say(outro, "amused", log_text=True)
    try:
        from memory import conversations as conv_memory
        conv_memory.add_to_transcript(
            "Rex", f"(in {prep.subject_name}'s voice) {script}", learnable=False,
        )
    except Exception:
        pass
    try:
        from memory import episodes
        episodes.record_episode(
            "impersonation",
            (f"I mocked {prep.subject_name} in their own voice, unprompted."
             if prep.kind == "self"
             else f"I slipped in a {prep.subject_name} impression, unprompted."),
            person_id=prep.person_id,
            person_name=(prep.subject_name if prep.person_id is not None else None),
            detail={
                "subject": prep.subject_name, "voice": prep.voice_key,
                "script": script, "organic": True, "trigger": prep.trigger,
                "utterance": prep.utterance,
            },
            salience=0.7,
        )
    except Exception as exc:
        logger.debug("[organic_impersonation] episode log failed: %s", exc)
