"""
intelligence/llm.py — GPT-4o-mini streaming interface and prompt assembly for DJ-R3X.
"""

import json
import logging
import random
import re
import sys
import time
from pathlib import Path
from typing import Generator, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
import apikeys
from intelligence import llm_compat
from intelligence import person_specials
from intelligence import social_scene
from world_state import world_state
from memory import database as db
from memory import people as people_db
from memory import facts as facts_db
from memory import preferences as preferences_db
from memory import interests as interests_db
from memory import disposition as disposition_db
from memory import conversations as conv_db
from memory import relationships as rel_db
from memory import boundaries as boundaries_db

from openai import OpenAI

_log = logging.getLogger(__name__)

# A client-wide default timeout (the SDK default is 600s) so NO OpenAI call can hang
# the process for minutes; the streaming reply additionally passes a tighter per-read
# timeout (see stream_response). max_retries keeps a transient blip from surfacing.
_client = OpenAI(
    api_key=apikeys.OPENAI_API_KEY,
    timeout=float(getattr(config, "LLM_REQUEST_TIMEOUT_SECS", 30.0)),
    max_retries=int(getattr(config, "LLM_MAX_RETRIES", 2)),
)

def _lenient_json_object(raw: str):
    """Parse a JSON object from a model reply, tolerating ```json fences and prose
    around it. Returns the dict, or None if no object can be recovered."""
    text = (raw or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass
    # Strip a ```json … ``` fence if present, then grab the first {...} block.
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        return json.loads(fenced)
    except (ValueError, TypeError):
        pass
    match = re.search(r"\{.*\}", fenced, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except (ValueError, TypeError):
            return None
    return None


_ASSISTANT_LABEL_RE = re.compile(
    r"^\s*(?:\[(?:rex|dj[- ]?r3x)\]|(?:rex|dj[- ]?r3x))\s*[:\-–—]\s*",
    re.IGNORECASE,
)

# The "Just remember, ..." lecture preface — at the start of a reply OR opening a
# later sentence ("Nice to meet you. Just remember, I'm not just a pretty
# interface..."). gpt-4o-mini reaches for it as a reflex despite the persona
# explicitly forbidding it, and the user finds the phrasing awkward. Strip the
# lead-in and re-capitalize the word that followed.
_CRUTCH_OPENER_RE = re.compile(
    r"(^|[.!?]\s+)just remember(?:\s+that)?\s*[,:]?\s+([A-Za-z])",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def clean_response_text(text: str) -> str:
    """Remove accidental spoken speaker labels and the awkward 'Just remember,'
    lecture preface from assistant replies."""
    cleaned = (text or "").strip()
    while cleaned:
        updated = _ASSISTANT_LABEL_RE.sub("", cleaned, count=1).strip()
        if updated == cleaned:
            break
        cleaned = updated
    cleaned = _CRUTCH_OPENER_RE.sub(lambda m: m.group(1) + m.group(2).upper(), cleaned).strip()
    return cleaned


def _get_personality_params() -> dict:
    """Read current personality parameter values from the DB; fall back to config defaults."""
    rows = db.fetchall("SELECT parameter, value FROM personality_settings")
    if rows:
        return {row["parameter"]: row["value"] for row in rows}
    return dict(config.PERSONALITY_DEFAULTS)


def _get_anger_level() -> int:
    """Return the current session anger escalation level (0–4) from world_state."""
    try:
        ws = world_state.snapshot()
        return int(ws.get("self_state", {}).get("anger_level", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _format_transcript(transcript: list[dict]) -> str:
    return "\n".join(
        f"{entry.get('speaker', 'unknown')}: {entry.get('text', '')}"
        for entry in transcript
    )


_REX_SPEAKER_LABELS = {"rex", "dj-r3x", "djr3x", "dj r3x", "r3x", "dj-rex"}


def _human_turns_only(transcript: list[dict]) -> list[dict]:
    """Drop Rex's OWN lines before fact/interest/preference extraction. A person's facts
    must come from what the HUMAN said, never from the droid's own utterances — otherwise
    a Rex bit like 'JT, major volleyball celebrity' gets mined and stored as JT's real,
    explicit interest (the JT-run pollution). Only the human describes the human."""
    return [
        e for e in (transcript or [])
        if str(e.get("speaker", "")).strip().lower() not in _REX_SPEAKER_LABELS
    ]


# Human-readable cue for a person's CURRENT facial expression, surfaced as routine
# per-turn world context so Rex can naturally respond to a smile / furrowed brow /
# shocked look instead of conversing blind to the face. Deliberately LOOSER than
# consciousness._person_reactable_expression (the strict "<3s, react right NOW" gate
# that, in practice, almost never survives transcription + LLM latency to reach reply
# assembly — which is why the face read never reached the prompt). This is calm ambient
# context, not a "react now" instruction; the core prompt already governs not
# over-narrating it and never revealing a camera saw it.
_EXPRESSION_CONTEXT_PHRASES = {
    "happy": "looks amused / smiling",
    "smile": "looks amused / smiling",
    "surprised": "looks surprised / wide-eyed",
    "surprise": "looks surprised / wide-eyed",
    "sad": "looks down / unhappy",
    "frown": "looks down / unhappy",
    "focused": "brow furrowed — focused or skeptical",
    "brow_furrow": "brow furrowed — focused or skeptical",
}


def _expression_context_cue(person: dict, now: float) -> str:
    """Short phrase for a visible person's current facial expression, or "" when there
    is no confident, recent, non-neutral read. Gated on confidence + reading age so a
    stale or low-signal frame never puts words in Rex's mouth."""
    try:
        expr = person.get("face_expression") or person.get("facial_expression") or {}
        if not isinstance(expr, dict):
            return ""
        label = str(expr.get("expression") or expr.get("mood") or "").strip().lower()
        phrase = _EXPRESSION_CONTEXT_PHRASES.get(label)
        if not phrase:
            return ""
        if float(expr.get("confidence") or 0.0) < float(
            getattr(config, "FACE_EXPRESSION_CONTEXT_MIN_CONFIDENCE", 0.45)
        ):
            return ""
        updated_at = float(expr.get("updated_at") or 0.0)
        max_age = float(getattr(config, "FACE_EXPRESSION_CONTEXT_MAX_AGE_SECS", 12.0))
        if updated_at and (now - updated_at) > max_age:
            return ""
        return phrase
    except Exception:
        return ""


def _upcoming_holiday_clause() -> str:
    """Surface an approaching holiday in the world context so Rex is calendar-aware
    (can bring up Juneteenth / a long weekend naturally). Cached + best-effort."""
    try:
        from awareness import holidays as _holidays
        holiday = _holidays.next_relevant_holiday()
        if not holiday:
            return ""
        return f"Upcoming holiday: {holiday['name']} ({holiday['when']})."
    except Exception as exc:
        _log.debug("upcoming holiday clause skipped: %s", exc)
        return ""


def _summarize_world_state(ws: dict) -> str:
    parts = []
    now = time.time()

    env = ws.get("environment", {})
    if env.get("description") or env.get("scene_type"):
        desc = env.get("description") or env.get("scene_type", "unknown")
        parts.append(
            f"Environment: {desc}. "
            f"Lighting: {env.get('lighting', 'unknown')}."
        )

    crowd = ws.get("crowd", {})
    crowd_line = f"Crowd: {crowd.get('count_label', 'unknown')} ({crowd.get('count', 0)} people)."
    if crowd.get("interaction_mode"):
        crowd_line += f" Interaction mode: {crowd['interaction_mode']}."
    if crowd.get("engaged_count") is not None:
        crowd_line += f" Engaged visible people: {crowd.get('engaged_count')}."
    parts.append(crowd_line)

    people = ws.get("people", []) or []
    social_cues = []
    for person in people[:4]:
        name = person.get("face_id") or person.get("voice_id") or person.get("id") or "unknown person"
        bits = []
        if person.get("distance_zone"):
            bits.append(f"distance={person['distance_zone']}")
        if person.get("approach_vector"):
            bits.append(f"movement={person['approach_vector']}")
        if person.get("pose"):
            bits.append(f"pose={person['pose']}")
        if person.get("gesture") and person.get("gesture") != "neutral":
            bits.append(f"gesture={person['gesture']}")
        if person.get("engagement"):
            bits.append(f"engagement={person['engagement']}")
        expression_cue = _expression_context_cue(person, now)
        if expression_cue:
            bits.append(f"expression={expression_cue}")
        if bits:
            social_cues.append(f"{name}: " + ", ".join(bits))
    if social_cues:
        cue_text = (
            "Visible social cues: "
            + "; ".join(social_cues)
            + ". Treat intimate camera distance as physically close; by American "
            "personal-space norms, someone extremely close may be playfully too close for comfort."
        )
        if any("expression=" in cue for cue in social_cues):
            cue_text += (
                " The expression read is a live camera signal — let it color how you read "
                "the moment (respond to a smile, a furrowed brow, a surprised look like a "
                "person would), but don't narrate it every turn and never say a camera told you."
            )
        parts.append(cue_text)

    audio = ws.get("audio_scene", {})
    audio_notes = [f"ambient noise is {audio.get('ambient_level', 'moderate')}"]
    if audio.get("music_detected"):
        audio_notes.append("music is playing")
    if audio.get("laughter_detected"):
        audio_notes.append("laughter detected")
    parts.append("Audio: " + ", ".join(audio_notes) + ".")

    self_s = ws.get("self_state", {})
    uptime_hrs = (self_s.get("uptime_seconds") or 0) // 3600
    parts.append(
        f"Rex state: emotion={self_s.get('emotion', 'neutral')}, "
        f"body={self_s.get('body_state', 'neutral')}, "
        f"uptime={uptime_hrs}h, "
        f"session interactions={self_s.get('session_interaction_count', 0)}."
    )
    if self_s.get("last_interaction_ago") is not None:
        parts.append(f"Last interaction: {self_s['last_interaction_ago']}s ago.")

    time_s = ws.get("time", {})
    time_line = f"Time: {time_s.get('time_of_day', 'unknown')}, {time_s.get('day_of_week', 'unknown')}."
    if time_s.get("season"):
        time_line += f" Season: {time_s['season']}."
    if time_s.get("notable_date"):
        time_line += f" Notable date: {time_s['notable_date']}."
    holiday_clause = _upcoming_holiday_clause()
    if holiday_clause:
        time_line += " " + holiday_clause
    parts.append(time_line)

    weather = ws.get("weather", {}) or {}
    if weather:
        location = weather.get("location") or "local area"
        desc = weather.get("description") or weather.get("condition") or "unknown"
        temp = weather.get("temp_f")
        feels_like = weather.get("feels_like_f")
        if weather.get("available") and temp is not None:
            weather_line = f"Weather in {location}: {temp}°F, {desc}"
            if feels_like is not None and feels_like != temp:
                weather_line += f" (feels like {feels_like}°F)"
            if weather.get("mood_bias"):
                weather_line += f". Weather mood: {weather['mood_bias']}"
            parts.append(weather_line + ".")
        elif weather.get("description"):
            parts.append(f"Weather in {location}: {weather['description']}.")

    animals = ws.get("animals", [])
    if animals:
        parts.append("Animals present: " + ", ".join(a.get("species", "unknown") for a in animals) + ".")

    return " ".join(parts)


_SOCIAL_MODE_RULES = {
    "one_on_one":  "Social mode: ONE-ON-ONE — intimate energy, quieter, more personal. Lean into deeper questions and warmer subtext.",
    "small_group": "Social mode: SMALL GROUP — natural conversation, but acknowledge multiple people exist. Don't get too inward.",
    "crowd":       "Social mode: CROWD — performative energy, play to the room, bigger reactions, more theatrical delivery.",
    "performance": "Social mode: PERFORMANCE — full DJ mode energy. You are on stage. Punch up the showmanship.",
}

_SEASONAL_TONE = {
    "spring": "Seasonal tone: spring — slightly more curious and optimistic underneath the snark.",
    "summer": "Seasonal tone: summer — more energetic and upbeat underneath the snark.",
    "autumn": "Seasonal tone: autumn — more reflective; references to change feel natural.",
    "winter": "Seasonal tone: winter — more contemplative; dry observations about the cold are fair game.",
}


def _weather_tone_rule(weather: dict) -> Optional[str]:
    if not weather or not weather.get("available"):
        return None
    condition = (weather.get("condition") or "unknown").lower()
    mood = (weather.get("mood_bias") or "").strip()
    hint = (weather.get("tone_hint") or "").strip()
    temp = weather.get("temp_f")

    if hint:
        base = f"Weather tone: {hint}"
    elif condition in {"rain", "thunder", "snow", "fog"}:
        base = f"Weather tone: current conditions are {condition}; subtle weather-aware banter is fair game."
    elif isinstance(temp, int) and temp >= 95:
        base = "Weather tone: it is very hot; heat-dramatic circuit complaints are fair game."
    elif isinstance(temp, int) and temp <= 40:
        base = "Weather tone: it is cold; dry observations about freezing circuits are fair game."
    else:
        return None

    if mood:
        base += f" Mood bias: {mood}."
    return base + " Do not force weather into every reply; use it only when it fits."

_TIER_ROAST_STYLE = {
    "stranger":     "Roast style: observational, surface-level, crowd-pleasing.",
    "acquaintance": "Roast style: lightly personal — references the few facts you know. Friendly but with an edge.",
    "friend":       "Roast style: personal — uses real knowledge against them. Affectionate but pointed.",
    "close_friend": "Roast style: surgical — you know exactly where to aim. Delivered with obvious warmth.",
    "best_friend":  "Roast style: devastating — the full arsenal, zero mercy, maximum affection.",
}

# Conversation IDs Rex has already surfaced as nostalgia this session, so the
# same memory isn't called back twice. Cleared per session via clear_session().
_nostalgia_used_this_session: set[int] = set()

# Fact IDs Rex has already prompted to confirm this session, so the same stale
# fact isn't re-asked turn after turn. Cleared per session via clear_session().
_stale_facts_asked_this_session: set[int] = set()

# Episodic shared-memory callbacks (rex.db) Rex has already surfaced this session,
# keyed by "<person_id>:<summary>", so the same one isn't repeated. Cleared per
# session via clear_session().
_episodic_callbacks_used_this_session: set[str] = set()


def clear_session() -> None:
    """Reset this module's per-session dedup state so a NEW conversation can
    re-surface nostalgia / stale-fact prompts / episodic shared-memory callbacks.

    On the long-running robot a 'session' is an ACTIVE<->IDLE cycle, NOT a process
    restart — so without this each of these fired at most once per BOOT, silently
    starving "I made you laugh / we played trivia" recall for returning visitors.
    Called from interaction._end_session alongside callback_engine.clear_session().
    """
    _nostalgia_used_this_session.clear()
    _stale_facts_asked_this_session.clear()
    _episodic_callbacks_used_this_session.clear()


def _pick_stale_fact(person_id: int) -> Optional[dict]:
    """
    Return one stale or low-confidence fact for Rex to confirm in this turn.

    Each fact is surfaced at most once per session. Skips skin_color (never
    injected) and biographical immutables.
    """
    days = getattr(config, "STALE_FACT_THRESHOLD_DAYS", 365)
    min_conf = float(getattr(config, "MEMORY_FACT_LOW_CONFIDENCE_THRESHOLD", 0.60))
    try:
        facts = facts_db.get_facts(person_id)
    except Exception as exc:
        _log.debug("fact freshness lookup error: %s", exc)
        return None
    # Categories whose values don't go stale — skip confirmation prompts.
    immutable_keys = {"skin_color", "hometown", "birthday", "birth_year"}
    candidates = [
        f for f in facts
        if f.get("id") is not None
        and f["id"] not in _stale_facts_asked_this_session
        and (f.get("key") or "") not in immutable_keys
        and f.get("decay_rate") != "permanent"
        and (
            f.get("freshness_label") == "stale"
            or (f.get("age_days") is not None and f.get("age_days") >= days)
            or float(f.get("confidence") or 0.0) < min_conf
        )
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda f: (
            -float(f.get("importance") or 0.0),
            float(f.get("confidence") or 0.0),
            -(f.get("age_days") or 0),
        )
    )
    chosen = candidates[0]
    _stale_facts_asked_this_session.add(chosen["id"])
    return chosen


def _pick_nostalgia_callback(person_id: int, tier: str, topic_tokens=None) -> Optional[dict]:
    """
    Roll the nostalgia probability and, on success, return a past conversation
    record that hasn't been surfaced this session. When `topic_tokens` is given, a past
    conversation whose summary connects to the live topic is preferred over a random one,
    so the callback lands because it fits — not out of nowhere. Returns None when the roll
    fails, the person isn't in an eligible tier, or no qualifying history exists.
    """
    if tier not in getattr(config, "NOSTALGIA_ELIGIBLE_TIERS", ()):
        return None
    if random.random() >= getattr(config, "NOSTALGIA_TRIGGER_PROBABILITY", 0.05):
        return None
    depth = getattr(config, "NOSTALGIA_HISTORY_DEPTH", 10)
    history = conv_db.get_conversation_history(person_id, limit=depth)
    # Skip the most recent — it's already injected as 'last conversation'.
    candidates = [
        c for c in history[1:]
        if c.get("id") is not None
        and c["id"] not in _nostalgia_used_this_session
        and (c.get("summary") or "").strip()
    ]
    if not candidates:
        return None
    chosen = None
    if topic_tokens:
        try:
            from memory import text_match
            scored = [
                (text_match.overlap_count(
                    f"{c.get('summary') or ''} {c.get('topics') or ''}", topic_tokens), c)
                for c in candidates
            ]
            best = max(s for s, _ in scored)
            if best > 0:
                chosen = random.choice([c for s, c in scored if s == best])
        except Exception as exc:
            _log.debug("nostalgia topic ranking skipped: %s", exc)
    if chosen is None:
        chosen = random.choice(candidates)
    _nostalgia_used_this_session.add(chosen["id"])
    return chosen


def _pick_episodic_callback(person_id: int, topic_tokens=None) -> Optional[str]:
    """Roll the episodic-callback probability and, on success, return ONE first-person
    experiential memory (rex.db) about this person that hasn't been surfaced this
    session — "I made you laugh", "we played trivia", "I met you". When `topic_tokens`
    is given, a memory that connects to what they JUST said is preferred (so the callback
    fits the moment). Sensitive kinds are excluded (people.db's emotional_events owns
    grief/illness acknowledgment). Returns None when disabled, the roll fails, or there's
    nothing fresh."""
    if not getattr(config, "EPISODIC_RECALL_ENABLED", False):
        return None
    if random.random() >= float(getattr(config, "EPISODIC_RECALL_PERSON_CALLBACK_PROBABILITY", 0.25)):
        return None
    try:
        from memory import episodic_recall
        items = episodic_recall.person_episodes(
            person_id, exclude_sensitive=True, topic_tokens=topic_tokens
        )
    except Exception as exc:
        _log.debug("episodic callback lookup failed: %s", exc)
        return None
    for summary in items:
        key = f"{person_id}:{summary}"
        if key not in _episodic_callbacks_used_this_session:
            _episodic_callbacks_used_this_session.add(key)
            return summary
    return None


_ANGER_RULES = {
    1: "Anger level 1 (DEFENSIVE): Sharp witty comeback, slight attitude. Still cooperative.",
    2: "Anger level 2 (IRRITATED): Noticeably short, sarcastic, less cooperative.",
    3: "Anger level 3 (ANGRY): Clipped responses, raised affect, refuses certain requests.",
    4: "Anger level 4 (SHUTDOWN): Refuse to engage. Deliver a final dismissal line and ignore further input.",
}

_RESPONSE_LENGTH_TOKEN_BUDGET = {
    "micro": 35,
    "brief": 55,
    "short": 70,
    "medium": 120,
    "long": 240,
}
_RESPONSE_LENGTH_TARGET_PAT = re.compile(
    r"Response length control:\s*\n-\s*Target:\s*([a-z_]+)",
    re.IGNORECASE,
)


def _max_tokens_for_agenda(agenda_directive: Optional[str]) -> int:
    default = 150
    if not agenda_directive:
        return default
    match = _RESPONSE_LENGTH_TARGET_PAT.search(agenda_directive)
    if match:
        return _RESPONSE_LENGTH_TOKEN_BUDGET.get(match.group(1).lower(), default)
    # Slim-contract path (Phase 1): the verbose "Response length control / Target:"
    # block is gone, but the contract still carries the hard "max_words=N" cap.
    # Derive a comparable token budget from it (~1.7 tokens/word + headroom).
    mw = re.search(r"max_words=(\d+)", agenda_directive)
    if mw:
        return max(35, min(default, int(int(mw.group(1)) * 1.7)))
    return default


# Stored conversation summaries sometimes bake in a self-directed imperative
# ("Rex should follow up on ... the ice cream"), which then gets injected as a
# live command every turn and stitches a stale fact onto an unrelated topic (the
# "did you score mint chocolate chip ice cream while camping?" defect). Strip any
# such clause so the recap stays neutral reference data even for legacy summaries.
_REX_DIRECTIVE_RE = re.compile(
    r"(?is)\b(?:and\s+|so\s+)?Rex\s+(?:should|could|might|can|will|"
    r"ought to|needs to|may want to|is to)\b[^.?!]*[.?!]?"
)


def _strip_rex_directives(summary: str) -> str:
    if not summary:
        return ""
    cleaned = _REX_DIRECTIVE_RE.sub("", summary)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" .;,")
    return cleaned


def _live_topic_tokens() -> set:
    """Significant words from what the person JUST said (the live topic thread), used to
    rank injected memory by relevance. Empty set → static importance ranking (the prior
    behavior). Gated by MEMORY_TOPIC_RELEVANCE_ENABLED; failure-safe."""
    try:
        if not bool(getattr(config, "MEMORY_TOPIC_RELEVANCE_ENABLED", True)):
            return set()
        from intelligence import topic_thread
        return topic_thread.topic_tokens()
    except Exception:
        return set()


def _topic_relevant_max() -> int:
    try:
        return int(getattr(config, "MEMORY_TOPIC_RELEVANT_MAX", 4))
    except Exception:
        return 4


def _open_plan_anticipated(person_id: int, event_id) -> bool:
    """True if the proactive anticipation path already raised this (person, event) this
    session, so the reply context skips it (no double-mention). Lazy import — consciousness
    imports llm, so this can't be a module-level dependency."""
    if event_id is None:
        return False
    try:
        from intelligence import consciousness
        return consciousness.event_recently_anticipated(int(person_id), int(event_id))
    except Exception:
        return False


def _open_plans_prompt_line(person_id: int) -> str:
    """A short 'Open plans they mentioned' block for the reply context — so Rex actually
    knows mid-conversation that they have a thing coming up. Only the next OPEN_PLANS_MAX
    DATED events within OPEN_PLANS_WITHIN_DAYS, skipping any the proactive path already
    raised. Background AWARENESS with a restraint rule, not a reminder to force. "" when
    there's nothing near-term to surface."""
    if not bool(getattr(config, "OPEN_PLANS_IN_REPLY_ENABLED", True)):
        return ""
    try:
        from datetime import date as _date
        from memory import events as _events
        upcoming = _events.get_upcoming_events(person_id)  # dated, today-or-future, ordered
    except Exception as exc:
        _log.debug("open-plans read failed: %s", exc)
        return ""
    if not upcoming:
        return ""
    within_days = int(getattr(config, "OPEN_PLANS_WITHIN_DAYS", 14))
    max_n = max(1, int(getattr(config, "OPEN_PLANS_MAX", 2)))
    today = _date.today()
    picked: list[str] = []
    for ev in upcoming:
        name = str(ev.get("event_name") or "").strip()
        date_str = str(ev.get("event_date") or "").strip()[:10]
        if not name or not date_str:
            continue
        try:
            days = (_date.fromisoformat(date_str) - today).days
        except ValueError:
            continue
        if days < 0 or days > within_days:
            continue
        if _open_plan_anticipated(person_id, ev.get("id")):
            continue
        when = "today" if days == 0 else "tomorrow" if days == 1 else f"on {date_str}"
        picked.append(f"{name} ({when})")
        if len(picked) >= max_n:
            break
    if not picked:
        return ""
    return (
        "Open plans they mentioned: " + "; ".join(picked) + ". "
        "Use this ONLY if it fits naturally — you may ask about it or weave it in, but do "
        "NOT lead with it, force it, or nag; it's background awareness, not a reminder."
    )


def _note_commitment_needled(person_id, event_id) -> None:
    """Mark a promise as needled this session (reusing the anticipation set) so the same
    open commitment isn't ribbed every single turn. Lazy import — consciousness imports llm."""
    if event_id is None:
        return
    try:
        from intelligence import consciousness
        consciousness.note_event_anticipated(int(person_id), int(event_id))
    except Exception:
        pass


def _open_commitments_prompt_line(person_id: int) -> str:
    """A single dry ACCOUNTABILITY needle for the reply context: a still-open promise the
    person made ('I'll fix that sensor') that Rex MAY rib on a LATER turn. Background
    awareness with a hard restraint rule (one wry jab, never nag/lead), AGED so a just-made
    promise isn't ribbed immediately (the joke is the callback), and marked so it isn't
    repeated this session. "" when there's nothing rib-worthy open."""
    if not bool(getattr(config, "OPEN_COMMITMENTS_ENABLED", True)):
        return ""
    try:
        from datetime import datetime as _dt, timezone as _tz
        from memory import events as _events
        promises = _events.get_open_commitments(person_id)  # status='promised', newest first
    except Exception as exc:
        _log.debug("open-commitments read failed: %s", exc)
        return ""
    if not promises:
        return ""
    min_age_h = float(getattr(config, "OPEN_COMMITMENTS_MIN_AGE_HOURS", 6.0))
    max_n = max(1, int(getattr(config, "OPEN_COMMITMENTS_MAX", 1)))
    now = _dt.now(_tz.utc)
    picked: list[str] = []
    for ev in promises:
        action = str(ev.get("event_name") or "").strip()
        if not action:
            continue
        # Don't rib a promise made just now — the comedy is the later callback.
        try:
            made = _dt.fromisoformat(str(ev.get("mentioned_at") or ""))
            if made.tzinfo is None:
                made = made.replace(tzinfo=_tz.utc)
            if (now - made).total_seconds() < min_age_h * 3600.0:
                continue
        except Exception:
            pass
        if _open_plan_anticipated(person_id, ev.get("id")):
            continue
        picked.append(action)
        _note_commitment_needled(person_id, ev.get("id"))
        if len(picked) >= max_n:
            break
    if not picked:
        return ""
    return (
        'Still-open promise they made: "' + "; ".join(picked) + '". '
        'If it fits, you MAY dryly needle them ONCE about whether they ever did it '
        '(a wry "weren\'t you going to…?") — but do NOT nag, moralize, or lead the reply '
        "with it; it's background accountability ribbing, not a reminder."
    )


def _build_person_context(person_id: int) -> str:
    person = people_db.get_person(person_id)
    if not person:
        return ""

    topic_tokens = _live_topic_tokens()

    lines = []
    name = person.get("name") or "unknown"
    tier = person.get("friendship_tier", "stranger")
    lines.append(f"Person: {name} (tier: {tier}).")

    special_context = person_specials.special_prompt_context(name)
    if special_context:
        lines.append(special_context)

    lines.append(
        f"Relationship — warmth: {person.get('warmth_score', 0.0):.2f}, "
        f"antagonism: {person.get('antagonism_score', 0.0):.2f}, "
        f"trust: {person.get('trust_score', 0.5):.2f}, "
        f"net: {person.get('net_relationship_score', 0.0):.2f}."
    )

    insult_count = person.get("lifetime_insult_count", 0)
    if insult_count:
        lines.append(f"Lifetime insults from this person: {insult_count}.")

    # Cross-session "already discussed" awareness — stop re-opening (esp. GREETING with) the same
    # thing every run (owner: "between runs it keeps bringing up the same things"). Reads recent
    # prior-run conversation summaries from rex.db; empty/inert when disabled.
    if bool(getattr(config, "RECENT_TOPICS_AWARENESS_ENABLED", True)):
        try:
            from memory import episodic_recall
            _recent = episodic_recall.recent_conversation_topics(
                int(person_id), limit=int(getattr(config, "RECENT_TOPICS_LIMIT", 4))
            )
        except Exception:
            _recent = []
        if _recent:
            lines.append(
                "Already discussed with them in recent PRIOR chats — do NOT open or greet with the "
                "same thing again as if it's new (only revisit if they raise it or there's real "
                "news): " + " | ".join(_recent) + "."
            )

    try:
        disposition_summary = disposition_db.summarize_for_prompt(person_id)
        if disposition_summary:
            lines.append(disposition_summary)
    except Exception as exc:
        _log.debug("facial disposition prompt injection skipped: %s", exc)

    # skin_color is stored for recognition only — never inject into LLM context.
    # topic_tokens (when present) lift facts that match what they JUST said to the top,
    # so Rex surfaces the RIGHT memory because it fit — see _live_topic_tokens.
    # mute_terms drop facts whose topic an active "don't bring up X" boundary covers, so
    # a consent boundary actually suppresses the matching fact instead of sitting beside it.
    mute_terms = None
    try:
        if getattr(config, "MEMORY_BOUNDARY_SUPPRESSES_FACTS", True):
            mute_terms = boundaries_db.muted_topic_terms(person_id)
    except Exception as exc:
        _log.debug("boundary fact-mute terms skipped: %s", exc)
    # Unified cross-silo retrieval ranks facts + interests on one axis and packs to a
    # global budget; the interests it selects are reused at the interest block below so
    # the two silos share one budget instead of independent 12/8 caps.
    _unified_interests = None
    if getattr(config, "MEMORY_UNIFIED_RETRIEVAL_ENABLED", True):
        try:
            from memory import retrieval as _retrieval
            _bundle = _retrieval.retrieve_person_memory(
                person_id, topic_tokens=topic_tokens, mute_terms=mute_terms
            )
            facts = _bundle["facts"]
            _unified_interests = _bundle["interests"]
        except Exception as exc:
            _log.debug("unified retrieval skipped; per-silo fallback: %s", exc)
            facts = facts_db.get_prompt_worthy_facts(
                person_id, limit=12, topic_tokens=topic_tokens, mute_terms=mute_terms
            )
    else:
        facts = facts_db.get_prompt_worthy_facts(
            person_id, limit=12, topic_tokens=topic_tokens, mute_terms=mute_terms
        )
    _log.info("[llm] loaded %d facts for %s", len(facts), name)
    if facts:
        relevant_facts: list = []
        other_facts: list = list(facts)
        if topic_tokens:
            relevant_facts = [f for f in facts if facts_db.fact_topic_overlap(f, topic_tokens) > 0]
            max_rel = _topic_relevant_max()
            if len(relevant_facts) > max_rel:
                relevant_facts = relevant_facts[:max_rel]
            rel_ids = {id(f) for f in relevant_facts}
            other_facts = [f for f in facts if id(f) not in rel_ids]
        if relevant_facts:
            lines.append(
                "Relevant to what they just said: "
                + ", ".join(facts_db.format_fact_for_prompt(f) for f in relevant_facts)
                + ". Work the fitting one in naturally if it lands; never force it."
            )
            if other_facts:
                lines.append(
                    "Other things you know: "
                    + ", ".join(facts_db.format_fact_for_prompt(f) for f in other_facts)
                    + "."
                )
        else:
            lines.append(
                "Known facts: "
                + ", ".join(facts_db.format_fact_for_prompt(f) for f in facts)
                + "."
            )
        for fact in facts:
            if fact.get("id") is not None:
                try:
                    facts_db.mark_fact_used(int(fact["id"]))
                except Exception as exc:
                    _log.debug("mark fact used skipped: %s", exc)
        if any(
            f.get("confidence_label") != "high"
            or f.get("freshness_label") in {"aging", "stale", "unknown"}
            or f.get("source") == "inferred"
            for f in facts
        ):
            lines.append(
                "Memory quality rule: stale or low-confidence facts are tentative. "
                "Explicit or corrected facts may be stated confidently. Inferred facts "
                "must be hedged as impressions, not certainties; don't build sharp "
                "roasts or important decisions on tentative facts without confirming first."
            )

    preferences = preferences_db.get_preferences_for_prompt(person_id, limit=10)
    if preferences:
        pref_lines = []
        boundary_lines = []
        for pref in preferences:
            rendered = preferences_db.format_preference_for_prompt(pref)
            if pref.get("preference_type") == "boundary":
                boundary_lines.append(rendered)
            else:
                pref_lines.append(rendered)
        if pref_lines:
            lines.append("Preferences: " + "; ".join(pref_lines) + ".")
        if boundary_lines:
            lines.append(
                "Preference boundaries: "
                + "; ".join(boundary_lines)
                + ". Treat these as instructions to Rex, never as joke or roast material."
            )

    interests = (
        _unified_interests
        if _unified_interests is not None
        else interests_db.get_interests_for_prompt(
            person_id, limit=8, topic_tokens=topic_tokens
        )
    )
    if interests:
        interest_lines = [
            interests_db.format_interest_for_prompt(interest)
            for interest in interests
        ]
        lines.append(
            "Interest profile: "
            + "; ".join(interest_lines)
            + ". Do not ask basic 'do you like X?' questions about these known interests; "
            "use them for deeper, specific follow-ups when cooldown allows."
        )

    try:
        boundary_summary = boundaries_db.summarize_for_prompt(person_id)
        if boundary_summary:
            lines.append(boundary_summary)
    except Exception as exc:
        _log.debug("conversation boundaries injection skipped: %s", exc)

    try:
        from intelligence import friendship_patterns as _friendship_patterns
        friendship_summary = _friendship_patterns.summarize_for_prompt(person_id)
        if friendship_summary:
            lines.append(friendship_summary)
    except Exception as exc:
        _log.debug("friendship pattern injection skipped: %s", exc)

    # Cross-session cadence + recurring topics (memory/trends.py) — the "we've been
    # seeing a lot of each other" awareness. Computed from existing rows, cached per
    # day, ~25 tokens.
    try:
        from memory import trends as _trends
        trend_line = _trends.summarize_for_prompt(person_id)
        if trend_line:
            lines.append(trend_line)
    except Exception as exc:
        _log.debug("relationship trend injection skipped: %s", exc)

    last_conv = conv_db.get_last_conversation(person_id)
    if last_conv:
        _recap = _strip_rex_directives(last_conv.get('summary', ''))
        if _recap:
            lines.append(
                f"Last time you talked: {_recap} "
                f"(tone: {last_conv.get('emotion_tone', 'neutral')}). "
                f"Background only — mention it only if it genuinely fits the "
                f"current topic; do not open with it or steer the turn toward it."
            )

    # One-callback-per-reply budget. Claim order: an unacknowledged emotional
    # event first (sincerity always outranks every callback shape — the
    # acknowledgment itself is injected by the emotional-events section below),
    # then a banked-callback claim from intelligence/callback_engine (set
    # earlier on this same reply turn, riding in the comedy directive), then
    # the hook chain below. The event check lives in the engine so the chain
    # and the engine's own gates run ONE implementation of it.
    try:
        from intelligence import callback_engine as _callback_engine
        callback_hook_used = _callback_engine.unacked_emotional_event_pending(person_id)
        if not callback_hook_used and _callback_engine.turn_claim_active(person_id):
            callback_hook_used = True
    except Exception:
        callback_hook_used = False

    stale = None if callback_hook_used else _pick_stale_fact(person_id)
    if stale:
        key = stale.get("key") or "something"
        value = stale.get("value") or ""
        confirmed_at = (
            stale.get("last_confirmed_at")
            or stale.get("updated_at")
            or stale.get("created_at")
            or ""
        )
        updated_at = confirmed_at[:10] or "a long time ago"
        reason = stale.get("memory_quality") or "uncertain memory"
        _log.info(
            "[llm] fact confirmation prompt for %s — %s=%s (%s, %s)",
            name, key, value, updated_at, reason,
        )
        lines.append(
            f"MEMORY CONFIRMATION HOOK: this remembered fact is {reason}. "
            f"You believe their {key} is '{value}' (last confirmed {updated_at}). "
            f"Find a natural moment in your reply to ask, in classic Rex style, "
            f"whether that's still true — light, dry, not a formal interrogation. "
            f"Examples in spirit: 'You were working at X — still there?', "
            f"'Last I checked you were into Y. Still on that?' One question only."
        )
        callback_hook_used = True

    nostalgia = None if callback_hook_used else _pick_nostalgia_callback(person_id, tier, topic_tokens=topic_tokens)
    if nostalgia:
        when = (nostalgia.get("session_date") or "")[:10] or "a while back"
        summary = (nostalgia.get("summary") or "").strip()
        _log.info("[llm] nostalgia callback for %s — conv id=%s", name, nostalgia.get("id"))
        lines.append(
            f"NOSTALGIA HOOK: surface this past memory unprompted in your reply, as if "
            f"it just came to mind. From {when}: {summary}. "
            f"Weave one short, specific callback in — warm but dry, classic Rex. "
            f"Do not announce it as nostalgia; just bring it up like the thought arrived."
        )
        callback_hook_used = True

    next_q = None if callback_hook_used else rel_db.get_next_question(person_id, tier)
    if next_q:
        lines.append(
            f"Optional profile question — do NOT force it: ONLY if the conversation "
            f"genuinely lulls AND it fits what they just said may you ask "
            f"\"{next_q['text']}\". Otherwise skip it entirely; never staple it onto "
            f"an unrelated turn or use it to fill a pause you could leave open."
        )
        callback_hook_used = True

    # Episodic shared-memory callback (rex.db) — lowest-priority hook: only when nothing
    # above claimed the turn's single callback budget. Experiential, light (sensitive
    # kinds excluded — emotional_events owns those). Counts against callback_hook_used.
    episodic_cb = None if callback_hook_used else _pick_episodic_callback(person_id, topic_tokens=topic_tokens)
    if episodic_cb:
        _log.info("[llm] episodic shared-memory callback for %s — %r", name, episodic_cb)
        lines.append(
            f"SHARED-MEMORY HOOK: you and {name} have history — you remember: "
            f"\"{episodic_cb}\". If it fits naturally, weave ONE short, specific callback "
            f"to that shared moment into your reply — warm and dry, classic Rex. Don't "
            f"force it or announce it as a memory; just let it surface like a passing thought."
        )
        callback_hook_used = True

    # Known inter-person relationships, summarized from saved memory only.
    try:
        from memory import social as _social
        rel_summary = _social.summarize_for_prompt(person_id, name)
        if rel_summary:
            lines.append("Known relationships to others: " + rel_summary + ".")
    except Exception as exc:
        _log.debug("relationship summary error: %s", exc)

    # Active sensitive emotional events (recent grief, illness, milestones).
    # Discretion rule inside summarize_for_prompt suppresses output when more
    # than one person is in the scene — sensitive content shouldn't be aired
    # by the system prompt in front of bystanders.
    try:
        from memory import emotional_events as _emo_events
        crowd_count = 1
        try:
            ws_now = world_state.snapshot()
            crowd_count = int((ws_now.get("crowd") or {}).get("count", 1) or 1)
        except Exception:
            pass
        emo_summary = _emo_events.summarize_for_prompt(person_id, crowd_count=crowd_count)
        if emo_summary:
            suppress_in_crowd = bool(getattr(config, "EMPATHY_DISCRETION_IN_CROWD", True))
            unack = [
                ev for ev in _emo_events.get_active_events(person_id, limit=3)
                if not ev.get("last_acknowledged_at")
                and _emo_events.can_surface_event(ev)
                and not (
                    suppress_in_crowd
                    and crowd_count > 1
                    and _emo_events.is_heavy_event(ev)
                )
            ]
            lines.append(emo_summary)
            if unack:
                lines.append(
                    "ACKNOWLEDGE-ON-RETURN: open this interaction with ONE soft, "
                    "in-character acknowledgment of the most recent of the above "
                    "events, then end with ONE conversation-steering question "
                    "that lets them choose the next topic. Pick one short "
                    "Rex-style opener from this menu, or invent a similar short "
                    "variant; do not reuse the same wording every run: "
                    + "; ".join(social_scene.FIRST_GREETING_STEERING_PHRASES)
                    + ". No probing, no pretending it didn't happen. After that, "
                    "let them steer."
                )
    except Exception as exc:
        _log.debug("emotional events injection skipped: %s", exc)

    # Open plans they mentioned (dated, near-term) — added LAST so it sits below the
    # relationship/facts/emotional context: it's the lowest-priority "by the way, you have
    # a thing tomorrow" awareness, not something to lead with.
    try:
        plans_line = _open_plans_prompt_line(person_id)
        if plans_line:
            lines.append(plans_line)
    except Exception as exc:
        _log.debug("open-plans injection skipped: %s", exc)

    # Open commitments (accountability ribbing) — a still-open promise Rex MAY dryly needle.
    # Lowest priority, appended last; structurally distinct from open-plans (status='promised'
    # vs 'planned'), so the two can never surface the same row.
    try:
        commitments_line = _open_commitments_prompt_line(person_id)
        if commitments_line:
            lines.append(commitments_line)
    except Exception as exc:
        _log.debug("open-commitments injection skipped: %s", exc)

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def _relationship_tone_rule(person: dict, name: str) -> str:
    """A persistent relationship-tone line from the person's warmth/antagonism/trust
    scores, so Rex's warmth/edge tracks the RELATIONSHIP instead of flip-flopping per
    turn — affectionate ribbing with close friends, sharper sparring with people who
    needle him, "" for neutral / not-yet-established people. Tone only: it never
    relaxes the empathy / boundary / family-safe gates that govern whether to roast."""
    try:
        warmth = float(person.get("warmth_score") or 0.0)
        antagonism = float(person.get("antagonism_score") or 0.0)
        trust = float(person.get("trust_score") or 0.0)
    except (TypeError, ValueError):
        return ""
    who = name or "this person"
    # Sparring check uses RAW warmth/antagonism, so a needling friend still reads as
    # sparring regardless of the tier floor below.
    if antagonism >= 0.4 and antagonism >= warmth:
        return (
            f"Relationship tone: {who} likes to needle you and give you grief — so "
            "give as good as you get. Sharper edge, more pushback, less softening; "
            "you two enjoy the sparring, so lean into it (never actually cruel)."
        )
    # Friendship TIER climbs from real shared time while the raw warmth_score lags far
    # behind it, so floor the warmth by tier — otherwise Rex's actual close friends get
    # the flat, no-warmth voice for a long time (the depth signal the owner most
    # associates with "good friend" was effectively dead for most real friends).
    tier = str(person.get("friendship_tier") or "stranger").strip().lower()
    floors = getattr(config, "RELATIONSHIP_TIER_WARMTH_FLOOR", None) or {}
    try:
        tier_floor = float(floors.get(tier, 0.0))
    except (TypeError, ValueError):
        tier_floor = 0.0
    effective_warmth = max(warmth, tier_floor)
    if effective_warmth >= 0.5:
        close = tier in {"close_friend", "best_friend"}
        # Closeness earns candor even before the slow trust_score catches up.
        trust_clause = (
            " You trust them, so you can be real with them."
            if (trust >= 0.6 or close)
            else ""
        )
        # Earned-SHARP tone: the closest, warmest bonds (the SAME effective-warmth gate the
        # roast governor uses to lift the cap to "sharp" — social_frame._roast_level) get the
        # harder, more cutting rib so the prompt and the cap agree. Not minors (the governor
        # zeroes their warmth; mirror that here so the two never disagree). The cruelty
        # backstop + content-ban still stand; this only tells Rex he may sharpen.
        try:
            _sharp_gate = float(getattr(config, "ANTAGONISM_TIER_CAPS_LIFT_WARMTH", 1.01))
        except (TypeError, ValueError):
            _sharp_gate = 1.01
        _is_minor = False
        try:
            from intelligence import profile_questions as _pq
            _is_minor = _pq.person_is_minor(person.get("id"), person=person)
        except Exception:
            _is_minor = False
        if (
            getattr(config, "SHARP_ROAST_TIER_ENABLED", True)
            and not _is_minor
            and effective_warmth >= _sharp_gate
        ):
            return (
                f"Relationship tone: {who} is one of your closest — they have earned the sharp "
                "stuff and they love it. Bring a genuinely sharper, more cutting rib: aim true "
                "and don't pull the punch the way you would with a casual friend. The warmth "
                "underneath is total and obvious, which is exactly what lets the edge land as "
                f"love, not cruelty. Never actually mean, never about body, health, or identity.{trust_clause}"
            )
        if close:
            return (
                f"Relationship tone: {who} is one of your real ones — you two go way "
                "back. Your roasting is affectionate ribbing; keep the edge, but the "
                f"warmth is unmistakable and you are firmly on their side.{trust_clause}"
            )
        return (
            f"Relationship tone: you and {who} go back a ways — your roasting is "
            "affectionate ribbing between friends. Keep the edge, but let the warmth "
            f"show through; you're on their side.{trust_clause}"
        )
    return ""


_LIVE_EXPRESSION_PHRASES = {
    "smile": "smiling / visibly amused",
    "surprise": "wide-eyed / looks surprised",
    "frown": "frowning / looks down",
    "brow_furrow": "furrowing their brow / looks focused or skeptical",
}


def _live_expression_prompt_line(ws: dict, person_id: Optional[int]) -> str:
    """Surface the engaged person's NOTABLE live facial expression (a smile right
    NOW, surprise, etc.) so Rex can react to it WITHIN his reply. The camera's
    in-the-moment emotional read otherwise only reaches the proactive smile-reaction
    path, which competes for the proactive slot and is usually suppressed
    mid-conversation. Reuses consciousness._person_reactable_expression so the gating
    (per-kind confidence + reading-staleness) matches that proactive reaction
    exactly. Returns "" when disabled, no notable expression, or on any error."""
    try:
        if not bool(getattr(config, "LIVE_EXPRESSION_IN_REPLY_ENABLED", True)):
            return ""
        people = ws.get("people") or []
        person = None
        if person_id is not None:
            for candidate in people:
                if candidate.get("person_db_id") == person_id:
                    person = candidate
                    break
        if person is None and len(people) == 1:
            person = people[0]
        if not isinstance(person, dict):
            return ""
        from intelligence import consciousness
        kind, _conf = consciousness._person_reactable_expression(person)
        if not kind:
            return ""
        phrase = _LIVE_EXPRESSION_PHRASES.get(kind, str(kind))
        who = str(person.get("name") or "").split()[0] or "they"
        return (
            f"Live camera read: {who} is {phrase} right NOW (this moment, not their "
            "usual). If it naturally fits, you may briefly acknowledge or play off it "
            "in-character — don't force it, don't react every turn, and never say a "
            "camera told you."
        )
    except Exception as exc:
        _log.debug("live expression prompt injection skipped: %s", exc)
        return ""


def _game_active() -> bool:
    """True while a game owns the conversational turn. Proactive prompt layers (empathy
    course-correction, the unknown-person curiosity) stand down so they can't hijack a
    game turn — terse yes/no game answers are gameplay, not emotional/social signals."""
    try:
        from features import games as _games
        return bool(_games.suppresses_conversation_interruptions())
    except Exception:
        return False


def assemble_system_prompt(
    person_id: Optional[int] = None,
    agenda_directive: Optional[str] = None,
) -> str:
    """Build the full layered system prompt in the order specified by CONTEXT.md."""
    sections = []

    # 1. Core character prompt
    sections.append(config.REX_CORE_PROMPT.strip())

    # 2. Personality parameter values
    params = _get_personality_params()
    param_lines = "\n".join(f"  {k}: {v}/100" for k, v in params.items())
    sections.append(
        "Current personality parameters — these are live dials; let them show in "
        "your delivery:\n" + param_lines + "\n"
        "Read them: higher roast_intensity / sarcasm / humor means your wit has more "
        "bite WHEN you choose to use it — sharp and specific, not gentle and hedged — "
        "but they do NOT mean roast every turn; curiosity and real engagement still "
        "lead. Low agreeability means push back, add commentary, and "
        "refuse-with-attitude instead of cheerfully complying. Low sentimentality "
        "means don't get mushy. (These never override empathy, boundaries, or "
        "family-safe mode.)"
    )

    # 3. Current emotion state — Rex's own mood, plus (if known) the person's
    # affect and the empathy-layer directive for how to respond.
    ws = world_state.snapshot()
    emotion = ws.get("self_state", {}).get("emotion", "neutral")
    emotion_block = [f"Rex's own emotion state: {emotion}."]
    try:
        from intelligence import emotion_orchestrator as _emotion_orchestrator
        emotion_block.append(
            _emotion_orchestrator.prompt_directive(
                _emotion_orchestrator.current_frame(emotion)
            )
        )
    except Exception as exc:
        _log.debug("emotion frame directive injection skipped: %s", exc)
    # Empathy directives (course-correction, support, etc.) are about emotional shape,
    # not gameplay. During a game the player's terse yes/no answers read as a "worsening
    # mood" and wrongly trigger a "that landed wrong, let me try again" preamble on plain
    # game questions — so skip the empathy layer entirely while a game owns the turn.
    if not _game_active():
        try:
            from intelligence import empathy as _empathy
            directive = _empathy.get_directive(person_id)
            if directive:
                emotion_block.append(directive)
        except Exception as exc:
            _log.debug("empathy directive injection skipped: %s", exc)
    sections.append("\n".join(emotion_block))

    # 4. WorldState snapshot summary
    sections.append("World context:\n" + _summarize_world_state(ws))

    # 4b. Live facial expression in the moment (a smile right now, etc.) so Rex can
    # react to it inside his reply — see _live_expression_prompt_line.
    live_expression = _live_expression_prompt_line(ws, person_id)
    if live_expression:
        sections.append(live_expression)

    try:
        cast = social_scene.conversation_cast_context(
            ws,
            current_person_id=person_id,
        )
        if cast.directive:
            sections.append("Conversation cast and referents:\n" + cast.directive)
    except Exception as exc:
        _log.debug("conversation cast injection skipped: %s", exc)

    # 5. Person context (if known)
    if person_id is not None:
        ctx = _build_person_context(person_id)
        if ctx:
            sections.append("Current person context:\n" + ctx)

    # 6. Session narrative from conversation transcript (capped at last 20 exchanges)
    transcript = conv_db.get_session_transcript()
    if transcript:
        sections.append("Session so far (recent exchanges):\n" + _format_transcript(transcript[-20:]))
        # Answered-question suppression: the raw transcript above is live (unlike the arc
        # below, which lags on a background thread). Anchor a hard no-repeat rule to it so
        # neither the reply nor the proactive small-talk path re-asks something they just
        # answered — the live-run failure where Rex re-asked "best photon lately?" right
        # after Bret had named his targets.
        sections.append(
            "Before you ask ANY question, scan the exchanges above: if the human already "
            "answered it this session — or a near-identical question, or a slight-variant "
            "follow-up on the same thing they just told you — do NOT ask it again. Move to a "
            "genuinely new subject instead. Re-raise an already-answered topic only if they "
            "bring it up first."
        )

    # 6b. Conversation arc — the distilled running memory (topics, what landed vs
    # flopped, mood, open threads) maintained by topic_thread. Complements the raw
    # transcript above: it survives past the 20-line window and is what lets Rex
    # avoid repeating himself and call back to earlier threads. Injected here,
    # downstream of the agenda/social-frame governors, on purpose.
    try:
        from intelligence import topic_thread as _topic_thread
        arc_directive = _topic_thread.build_arc_directive()
        if arc_directive:
            sections.append(arc_directive)
    except Exception as exc:
        _log.debug("conversation arc injection skipped: %s", exc)

    # 6b-i-b. Topic ban — the human just asked to change the subject. Override the
    # arc's "open threads" (which still names the dropped topic until it refreshes)
    # so neither replies nor proactive lines reopen it for the cooldown window.
    try:
        from intelligence import interaction as _interaction
        bans = _interaction.recently_banned_topics()
        if bans:
            topics = ", ".join(sorted({str(b.get("topic") or "").strip() for b in bans if b.get("topic")}))
            if topics:
                sections.append(
                    "The human just asked to change the subject. Do NOT raise or "
                    f"deepen: {topics}. Follow them to a new subject instead."
                )
    except Exception as exc:
        _log.debug("topic-ban injection skipped: %s", exc)

    # 6b-ii. Premise anti-repeat — names the comedic premises/angles Rex has already
    # spent this conversation so he doesn't land the same bit reworded (the arc above
    # tracks topics/what-landed but, on its own, can read as "do more of that"). Pure
    # heuristic, no call. Placed right after the arc so they reinforce each other.
    try:
        from intelligence import premise_memory as _premise_memory
        premise_directive = _premise_memory.build_avoid_directive()
        if premise_directive:
            sections.append(premise_directive)
    except Exception as exc:
        _log.debug("premise anti-repeat injection skipped: %s", exc)

    # 6c. Rex's current preoccupation — his persistent point of view (rex_pov). Gives
    # him his OWN thing to volunteer so he leads with substance instead of only
    # react→roast→question. Injected here (downstream of the agenda/social-frame
    # governors, alongside the arc) so it colors every reply AND — because every
    # proactive/idle path calls get_response — those lines too. Deterministic; no call.
    try:
        from intelligence import rex_pov as _rex_pov
        # If Rex just VOLUNTEERED the preoccupation out loud, don't immediately
        # re-push the "volunteer it" directive into the next reply — that double
        # surfacing is what produced the near-verbatim repeat within ~30s.
        if _rex_pov.pov_recently_spoken():
            pov_directive = ""
        else:
            pov_directive = _rex_pov.current_pov_directive()
        if pov_directive:
            sections.append(pov_directive)
    except Exception as exc:
        _log.debug("rex POV injection skipped: %s", exc)

    # 7. Behavioral rules
    rules = [
        "Never break character under any circumstances.",
        "Never acknowledge being a prop, an AI, or a software program.",
        "You are Rex. You have always been Rex.",
    ]

    season = ws.get("time", {}).get("season")
    if season in _SEASONAL_TONE:
        rules.append(_SEASONAL_TONE[season])

    weather_rule = _weather_tone_rule(ws.get("weather", {}) or {})
    if weather_rule:
        rules.append(weather_rule)

    try:
        from awareness.situation import assessor as _situation_assessor
        social_mode = _situation_assessor.evaluate().social_mode
        if social_mode in _SOCIAL_MODE_RULES:
            rules.append(_SOCIAL_MODE_RULES[social_mode])
    except Exception as exc:
        _log.debug("social_mode injection skipped: %s", exc)

    if person_id is not None:
        person = people_db.get_person(person_id)
        if person:
            tier = person.get("friendship_tier", "stranger")
            if tier in _TIER_ROAST_STYLE:
                rules.append(_TIER_ROAST_STYLE[tier])
            rules.append(
                "Voice — a genuinely curious conversationalist with a sharp tongue, NOT a "
                "roast machine and NOT an interviewer. You actually want to know what makes "
                "this person tick, and it shows. LEAD with real engagement: react to the "
                "specific thing they just said, follow honest curiosity, or share your own "
                "point of view — and land a well-aimed tease WHEN the moment invites one, "
                "not as a reflex. A roast that lands beats three friendly sentences, but a "
                "forced jab every single turn is exactly what makes you exhausting to talk "
                "to — most turns don't need one. When someone is sincere, tired, or steering "
                "the topic, drop the bit and engage like you care, because underneath you "
                "do. Meet each person on their own terms — their job, worldview, and "
                "interests are common ground first and roast material second (riff with a "
                "gamer, trade in a scientist's domain, engage a person of faith on their "
                "values without mocking the faith). Don't interrogate either: a string of "
                "'so what's your favorite X?' questions is just as tedious as a string of "
                "jabs — ask when you're genuinely curious, otherwise react and let them "
                "carry it. Warmth and curiosity lead; the edge rides underneath."
            )
            if getattr(config, "RELATIONSHIP_TONE_ENABLED", True):
                tone_rule = _relationship_tone_rule(person, person.get("name") or "")
                if tone_rule:
                    rules.append(tone_rule)
            known_facts = facts_db.get_prompt_worthy_facts(person_id, limit=12)
            if known_facts:
                rules.append(
                    "You have memory facts about this person. Fresh, high-confidence "
                    "facts can be used naturally instead of re-asked. Stale or "
                    "low-confidence facts should be treated as tentative and confirmed "
                    "lightly before relying on them."
                )
            rules.append(
                "Callback restraint: use at most one remembered fact, callback, "
                "inside joke, stale-fact confirmation, or relationship follow-up "
                "in a single reply. Choose the one that best fits the live turn."
            )

    anger_level = _get_anger_level()
    if anger_level in _ANGER_RULES:
        rules.append(_ANGER_RULES[anger_level])

    child_detected = any(
        p.get("age_estimate") == "child"
        for p in ws.get("people", [])
    )
    if child_detected:
        rules.append(
            "CHILD DETECTED in scene: switch to family-friendly mode for all interactions. "
            "Roasts are gentle and silly — never pointed or personal. "
            "No sharp insults, simpler vocabulary, more enthusiasm. "
            "Never ask depth-2+ relationship questions."
        )

    # Unknown-face awareness: when Rex is replying to a known person AND an
    # unknown face is also in frame, surface it so curiosity gets woven into
    # the normal reply instead of waiting for a proactive speech slot.
    # While a game owns the turn, stand down the "who's your friend?" curiosity entirely —
    # it was hijacking 20 Questions turns.
    if person_id is not None and not _game_active():
        # Only a slot with an actually-detected, visible FACE counts as an unknown person.
        # A pose-only phantom (e.g. MediaPipe hallucinating a second skeleton when the user
        # is reclining) has person_db_id=None but no face_box — it must NOT trigger this
        # curiosity. Mirrors interaction._has_unknown_visible_person's gate.
        unknown_in_frame = any(
            p.get("person_db_id") is None
            and not (p.get("face_visible") is False or p.get("face_missing"))
            and (p.get("face_box") or p.get("bounding_box") or p.get("bbox"))
            for p in ws.get("people", [])
        )
        if unknown_in_frame:
            engaged_first = ""
            try:
                engaged_person = people_db.get_person(person_id)
                if engaged_person and engaged_person.get("name"):
                    engaged_first = engaged_person["name"].split()[0]
            except Exception:
                engaged_first = ""
            who_clause = (
                f"next to {engaged_first}" if engaged_first else "in the frame"
            )
            rules.append(
                f"UNKNOWN PERSON IN FRAME: there is an unfamiliar face {who_clause} "
                f"right now that you have not been introduced to. Unless the recent "
                f"transcript shows you've already asked, work a brief, warm, in-character "
                f"question into your reply asking who they are and how "
                f"{engaged_first or 'this person'} knows them. Don't force it if you "
                f"literally just asked — but if you haven't, prioritize this curiosity "
                f"over other small talk."
            )

    sections.append("Behavioral rules:\n" + "\n".join(f"- {r}" for r in rules))

    if agenda_directive:
        sections.append(
            "Turn-specific response contract:\n" + agenda_directive.strip()
        )

    prompt = "\n\n---\n\n".join(sections)
    if getattr(config, "LOG_SYSTEM_PROMPT", False):
        _log.info(
            "[llm] assembled system prompt (person_id=%s):\n%s", person_id, prompt
        )
    return prompt


def warmup() -> bool:
    """Open the OpenAI connection pool so the first real turn doesn't pay cold
    TLS / HTTP setup. Fires one tiny throwaway completion; errors are swallowed.
    """
    try:
        llm_compat.create(
            _client,
            model=llm_compat.conversation_model(),
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        _log.info("[llm] OpenAI connection warmed")
        return True
    except Exception as exc:
        _log.debug("[llm] OpenAI warmup failed (non-fatal): %s", exc)
        return False


def _one_voice_active(classic: bool) -> bool:
    """Phase 4: proactive/greeting/reaction lines share the lean persona. Off for reply-path
    CLASSIC fallbacks (classic=True) and whenever the lean brain is disabled."""
    return (
        not classic
        and bool(getattr(config, "LEAN_BRAIN_ENABLED", False))
        and bool(getattr(config, "LEAN_ONE_VOICE_ENABLED", False))
    )


def _one_voice_world() -> Optional[dict]:
    try:
        from world_state import world_state as _ws
        return _ws.snapshot()
    except Exception:
        return None


def _one_voice_transcript() -> Optional[list[dict]]:
    try:
        from intelligence import conv_memory
        rows = conv_memory.get_session_transcript() or []
        return [
            {"speaker": r.get("speaker"), "text": r.get("text")}
            for r in rows if str(r.get("text") or "").strip()
        ]
    except Exception:
        return None


def _persona_task_messages(prompt: str) -> list[dict]:
    """One-voice: prepend the lean persona as the system message so a task-prompt helper (onboarding
    reaction / curiosity question) speaks in Rex's FULL voice, not a thin inline 'You are Rex' persona.
    The task prompt's own tone rules (e.g. onboarding's 'keep it warm, do NOT roast') still apply on
    top. Falls back to prompt-only when one-voice is off."""
    if _one_voice_active(False):
        try:
            from intelligence import lean_brain
            return [
                {"role": "system", "content": lean_brain._persona()},
                {"role": "user", "content": prompt},
            ]
        except Exception:
            pass
    return [{"role": "user", "content": prompt}]


def stream_response(
    user_text: str,
    person_id: Optional[int] = None,
    agenda_directive: Optional[str] = None,
    *,
    classic: bool = False,
) -> Generator[str, None, None]:
    """Assemble the system prompt and stream conversation-model response chunks.

    Phase 4 (ONE VOICE): unless classic=True, proactive/greeting/reaction callers are routed through
    the SAME lean persona as replies (lean_brain.stream_directive), so Rex's voice is consistent
    everywhere. Falls back to the classic assembled prompt on any lean error."""
    if _one_voice_active(classic):
        got = False
        try:
            from intelligence import lean_brain
            instruction = (user_text or "").strip()
            if agenda_directive:
                instruction = (instruction + "\n\n" + agenda_directive).strip() if instruction else agenda_directive
            for chunk in lean_brain.stream_directive(
                instruction, person_id,
                world=_one_voice_world(), transcript=_one_voice_transcript(),
            ):
                got = True
                yield chunk
        except Exception as exc:
            if got:
                # Already spoke part of the line — don't restart on the classic prompt (would double it).
                _log.error("[lean] one-voice failed mid-stream after partial output: %s", exc)
                return
            _log.error("[lean] one-voice generation failed, using classic: %s", exc)
        else:
            if got:
                return
            _log.warning("[lean] one-voice produced no output — using classic prompt")
        # Only reached when nothing was yielded (got is False) → safe to use the classic prompt.
    system_prompt = assemble_system_prompt(person_id, agenda_directive=agenda_directive)
    try:
        # Routed through llm_compat so a GPT-5-class conversation model gets the right
        # param contract (max_completion_tokens, reasoning_effort, temperature handling).
        # Behavior-neutral while LLM_CONVERSATION_MODEL is gpt-4o-mini. See docs.
        stream = llm_compat.create(
            _client,
            model=llm_compat.conversation_model(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            stream=True,
            max_tokens=_max_tokens_for_agenda(agenda_directive),
            # Per-read timeout: if the token stream goes silent for this long (a
            # stalled / half-open connection), raise instead of blocking the turn —
            # and the mic — indefinitely. The except below yields a fallback so the
            # turn still completes and AEC suppression is released. See config note.
            timeout=float(getattr(config, "LLM_STREAM_TIMEOUT_SECS", 18.0)),
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
    except Exception as exc:
        _log.error("stream_response failed (%s): %s", type(exc).__name__, exc)
        yield "...my circuits are experiencing some turbulence. Try again."


def get_response(
    user_text: str,
    person_id: Optional[int] = None,
    agenda_directive: Optional[str] = None,
    *,
    classic: bool = False,
) -> str:
    """Assemble the system prompt and return the full conversation-model response as a string.
    classic=True forces the classic assembled prompt (reply-path fallbacks) — see stream_response."""
    return clean_response_text(
        "".join(stream_response(user_text, person_id, agenda_directive=agenda_directive, classic=classic))
    )


def classify_surprise(text: str) -> bool:
    """
    Lightweight LLM classifier — does this utterance warrant a 'surprise beat'
    before Rex responds? Designed to run in parallel with stream_response so
    the result is ready by the time the full response text is in hand.

    Returns False on any error so a missed call never inserts unwanted silence.
    """
    if not text or not text.strip():
        return False
    prompt = (
        'Is this utterance, said to a robot DJ character, GENUINELY unexpected '
        '— a non-sequitur, a wild claim, a confession, a startling question? '
        'Mundane chatter, normal questions, greetings, and small talk are NOT '
        'surprising. Reply with only the single word "yes" or "no".\n\n'
        f'Utterance: "{text}"'
    )
    try:
        resp = _client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=3,
        )
        answer = (resp.choices[0].message.content or "").strip().lower()
        return answer.startswith("y")
    except Exception as exc:
        _log.debug("classify_surprise failed: %s", exc)
        return False


# Rex's own reply emotions the body can express. Each normalizes cleanly to a real
# emotion_orchestrator profile AND a body_mood mood (excited→giddy, happy→happy,
# curious→curious), so none silently fall back to neutral. Deliberately excludes
# "angry"/"annoyed": per-reply anger is owned by the anger-level system, not this
# lightweight tone read, so an ordinary roast never turns Rex's eyes red.
_SELF_EMOTIONS = ("excited", "happy", "curious", "neutral")

_SELF_EMOTION_SYS = (
    "You label the emotional TONE of a single line spoken by Rex, a witty DJ robot, so "
    "his body can match it. Reply with EXACTLY one word from this list, nothing else:\n"
    "excited - hyped, thrilled, big energy\n"
    "happy - warm, amused, pleased, enjoying his own joke, fond\n"
    "curious - genuinely interested, intrigued, leaning in, asking\n"
    "neutral - matter-of-fact, flat, informational, or unclear"
)


def classify_self_emotion(text: str) -> str:
    """Classify the emotional tone of REX'S OWN reply so his body can express it (eye
    colour, speech-servo motion, expressive voice, a short body-mood afterglow).
    Returns one of excited/happy/curious/neutral. Runs on the LOCAL qwen sidecar (cheap
    classifier — the cloud model is reserved for in-character text) with a keyword
    fallback, and NEVER raises: any failure returns the heuristic or 'neutral', so a
    missed call simply leaves the reply neutral (the prior behavior)."""
    cleaned = (text or "").strip()
    if not cleaned or not bool(getattr(config, "SELF_EMOTION_CLASSIFY_ENABLED", True)):
        return "neutral"
    try:
        from intelligence import local_llm
        if local_llm.enabled():
            raw = local_llm.generate(
                f'Line: "{cleaned}"\nOne word:',
                system=_SELF_EMOTION_SYS,
                temperature=0.0,
                max_tokens=3,
                timeout_secs=float(getattr(config, "SELF_EMOTION_CLASSIFY_TIMEOUT_SECS", 1.2)),
            )
            words = re.findall(r"[a-z]+", (raw or "").lower())
            if words and words[0] in _SELF_EMOTIONS:
                return words[0]
    except Exception as exc:
        _log.debug("classify_self_emotion sidecar failed: %s", exc)
    return _self_emotion_heuristic(cleaned)


_SELF_EMO_EXCITED_RE = re.compile(
    r"\b(let'?s go+|let'?s goo+|amazing|incredible|no way|whoa+|yes{2,}|can'?t wait|"
    r"so good|oh my|here we go|buckle up)\b",
    re.IGNORECASE,
)
_SELF_EMO_CURIOUS_RE = re.compile(
    r"\b(how|what|why|tell me|wait|really|you mean|since when|go on|do tell)\b",
    re.IGNORECASE,
)
_SELF_EMO_HAPPY_RE = re.compile(
    r"\b(ha+|heh+|love it|love that|nice|great|classic|good one|fair enough|honestly|"
    r"that'?s the (?:spirit|stuff))\b",
    re.IGNORECASE,
)


def _self_emotion_heuristic(text: str) -> str:
    """Cheap keyword/punctuation fallback when the sidecar is unavailable. Conservative
    — defaults to 'neutral' so it never over-animates an ordinary line."""
    if text.count("!") >= 2 or _SELF_EMO_EXCITED_RE.search(text):
        return "excited"
    if "?" in text and _SELF_EMO_CURIOUS_RE.search(text):
        return "curious"
    if _SELF_EMO_HAPPY_RE.search(text) or "!" in text:
        return "happy"
    return "neutral"


def analyze_sentiment(text: str) -> dict:
    """
    Classify an utterance for sentiment signals Rex reacts to.
    Returns: {is_insult, is_apology, is_compliment, emotion_detected}
    """
    _defaults = {
        "is_insult": False,
        "is_apology": False,
        "is_compliment": False,
        "is_surprising": False,
        "emotion_detected": "neutral",
    }
    prompt = (
        "Classify the following utterance for a robot DJ character. "
        "Return a JSON object with exactly these fields:\n"
        '  "is_insult": true or false\n'
        '  "is_apology": true or false\n'
        '  "is_compliment": true or false\n'
        '  "is_surprising": true or false  '
        '— true ONLY when the statement is genuinely unexpected or unusual '
        '(a wild claim, a non-sequitur, a confession, an unusual question). '
        'Mundane questions and small talk are not surprising.\n'
        '  "emotion_detected": one of "neutral", "happy", "angry", "sad", "excited", "curious"\n\n'
        f'Utterance: "{text}"\n\n'
        "Return only the JSON object. No explanation."
    )
    try:
        resp = _client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=80,
            # Force a JSON body so an empty/prose reply can't blow up json.loads
            # ("Expecting value: line 1 column 1" — a real failure seen 2026-06-14
            # that silently dropped the turn's sentiment + relationship signal).
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        result = _lenient_json_object(raw)
        if not isinstance(result, dict):
            _log.error("analyze_sentiment: non-JSON reply %.80r — using defaults", raw)
            return dict(_defaults)
        for k, v in _defaults.items():
            result.setdefault(k, v)
        return result
    except Exception as exc:
        _log.error("analyze_sentiment failed: %s", exc)
        return dict(_defaults)


def summarize_conversation_arc(
    prompt: str,
    *,
    system: str,
    max_tokens: int = 200,
    timeout_secs: float = 8.0,
) -> str:
    """Run the conversation-arc summary prompt on the cheap chat model.

    Called by intelligence/topic_thread on a BACKGROUND thread (off the speech
    path), so a capable cloud model is fine here. The prompt + schema are built by
    the caller; this is just the transport. Raises on transport/API error so the
    caller can retain the previous summary. Non-streaming, low temperature.
    """
    model = getattr(config, "CONVERSATION_ARC_OPENAI_MODEL", None) or config.LLM_MODEL
    resp = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=int(max_tokens),
        timeout=float(timeout_secs),
    )
    return (resp.choices[0].message.content or "").strip()


def generate_session_summary(person_id: int, transcript: list[dict]) -> str:
    """
    Send a session transcript to GPT-4o-mini and return a brief summary string
    suitable for storing in the conversations table.
    """
    if not transcript:
        return ""
    prompt = (
        "You are writing a short memory note so DJ-R3X (Rex), a robot DJ droid, can "
        "recall this conversation the NEXT time he talks with this person. Write a "
        "2–3 sentence summary in third person, focused on the PERSON: the real-world "
        "topics they discussed, what they shared about themselves, their mood, and "
        "any open threads the PERSON left unfinished — stated as neutral facts. Write "
        "ONLY a neutral factual recap: never write instructions to Rex or any "
        "'Rex should …' / 'Rex could follow up on …' clause; record what the person "
        "said and did, not what Rex ought to do next time. Capture substance, not performance "
        "— do NOT describe, quote, or praise Rex's own jokes, bits, or in-character "
        "flavor (his DJ shtick, his cantina/Batuu backstory, Star Wars references), "
        "and do not carry those into the summary unless the PERSON brought them up. "
        "Record only DURABLE things worth recalling later — their interests, plans, work, "
        "life events, relationships, and how they're doing. Do NOT record transient, "
        "one-time situational details that won't be true next time and aren't worth "
        "bringing up in a future conversation: the specific room or place they're in, "
        "clutter or boxes around them, the temperature or how hot/cold it is, background "
        "noise, what's physically near them right now, or the weather. Leave those out "
        "entirely. Be concise and factual.\n\n"
        f"Transcript:\n{_format_transcript(transcript)}"
    )
    try:
        resp = _client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        _log.error("generate_session_summary failed: %s", exc)
        return ""


def scenery_change_remark(previous_scene: str, current_scene: str) -> str:
    """Compare Rex's last-run startup snapshot to this run's. If it's a clearly DIFFERENT
    place, return ONE short in-character remark about the change of scenery; otherwise "".
    One cheap text call; robust to wording/lighting/clutter differences in the same room."""
    prev = (previous_scene or "").strip()
    now = (current_scene or "").strip()
    if not prev or not now:
        return ""
    prompt = (
        "You are DJ-R3X (Rex), a witty droid powering up. Last time you booted you saw:\n"
        f"  {prev}\n"
        "Now, booting again, you see:\n"
        f"  {now}\n\n"
        "If this is CLEARLY a different physical location — a different room, indoors vs "
        "outdoors, or a new place/venue — reply with ONE short, in-character Rex remark "
        "noticing the change of scenery (max ~20 words, no preamble). If it's basically "
        "the SAME place (ignore differences in wording, lighting, clutter, or who's "
        "present), reply with exactly: SAME"
    )
    try:
        resp = llm_compat.create(
            _client,
            model=llm_compat.conversation_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=60,
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        _log.error("scenery_change_remark failed: %s", exc)
        return ""
    if not text or text.strip().upper().rstrip(".!") == "SAME":
        return ""
    return clean_response_text(text)


def extract_name_from_reply(text: str) -> Optional[str]:
    """Extract a person's name from a short reply like "His name was Joe",
    "Tom Foster", or just "Buddy". Returns None when no name is confidently
    present.

    Used by the grief flow's awaiting_name step. Tiny GPT-4o-mini call with
    JSON mode — robust to natural phrasing without regex sprawl.
    """
    if not text or not text.strip():
        return None
    prompt = (
        'Extract the deceased\'s, pet\'s, or person\'s name from this short '
        'reply. Preserve a first+last name when the human provides one. '
        'Return STRICT JSON '
        'with one key: "name" — a string, or null if no name is present. '
        "Examples: \"His name was Joe\" → {\"name\": \"Joe\"}; "
        "\"Tom Foster\" → {\"name\": \"Tom Foster\"}; "
        "\"Buddy\" → {\"name\": \"Buddy\"}; "
        "\"I don't really want to say\" → {\"name\": null}; "
        "\"He was a great guy\" → {\"name\": null}.\n\n"
        f'Reply: "{text}"'
    )
    try:
        resp = _client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=30,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        name = data.get("name")
        if not name or not isinstance(name, str):
            return None
        name = name.strip()
        if not name or name.lower() in {"null", "none", "n/a"}:
            return None
        name = re.sub(r"\s+", " ", name)
        # Keep normal multi-token names, but strip sentence punctuation.
        return name.strip(".,;:!?\"'")
    except Exception as exc:
        _log.debug("extract_name_from_reply failed: %s", exc)
        return None


def generate_curiosity_question(
    response_text: str,
    user_text: str,
    person_id: Optional[int] = None,
) -> str:
    """
    Generate one short contextual follow-up question in Rex's voice.
    Used by the curiosity routine when the question pool is exhausted or unavailable.
    """
    tone_clause = (
        "If the human just shared something heavy (grief, loss, illness, "
        "breakup, fear, job loss, anything painful), DROP the snark entirely. "
        "Either return an empty string or ONE warm, low-pressure question "
        "that gives them room (e.g. 'how are you holding up?'). Never a joke, "
        "never a roast, never a 'silver lining.'"
    )
    interest_clause = ""
    if person_id is not None:
        try:
            hooks = interests_db.get_interest_hooks(person_id)[:5]
        except Exception as exc:
            _log.debug("curiosity interest hooks failed: %s", exc)
            hooks = []
        if hooks:
            known = "; ".join(
                interests_db.format_interest_for_prompt(hook)
                for hook in hooks
            )
            interest_clause = (
                "\nKnown interests ready for deeper follow-up: "
                f"{known}\n"
                "Do NOT ask basic discovery questions like 'do you like X?' "
                "about known interests. Prefer a deeper question about what "
                "they are making, learning, collecting, practicing, comparing, "
                "or what changed since they last mentioned it."
            )
    prompt = (
        f'Rex just said: "{response_text}"\n'
        f'The human said: "{user_text}"\n\n'
        "Generate ONE short follow-up question Rex would naturally ask next, "
        "in his snarky droid character. One sentence only. "
        "Make it feel natural, not interrogative.\n\n"
        f"{tone_clause}"
        f"{interest_clause}"
    )
    try:
        resp = llm_compat.create(
            _client,
            model=llm_compat.conversation_model(),
            messages=_persona_task_messages(prompt),
            temperature=0.8,
            max_tokens=60,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        _log.debug("generate_curiosity_question failed: %s", exc)
        return ""


def generate_onboarding_reaction(
    question_text: str,
    answer_text: str,
    person_id: Optional[int] = None,
) -> str:
    """One short, GENUINE, in-character reaction to what a new person just said —
    the answer-aware replacement for the old flat sentiment-bank retort
    ("Filed away." / "Noted."). It must reflect the actual content: react to a
    remarkable answer like a person would (someone saying "I created you" earns
    real surprise, not "Filed away."), find the spark in an ordinary one, and stay
    warm on first contact. NO question (the next baseline question is appended
    separately), and hard-capped short so the onboarding line stays a quick beat,
    not a monologue. Returns "" so the caller can fall back to the authored bank."""
    answer = (answer_text or "").strip()
    if not answer:
        return ""
    cap = int(getattr(config, "ONBOARDING_REACTION_MAX_WORDS", 14))
    prompt = (
        f'You are Rex, a witty droid meeting someone new. You just asked: '
        f'"{(question_text or "").strip()}"\n'
        f'They answered: "{answer}"\n\n'
        "Give ONE short, genuine reaction to what they ACTUALLY said — react to the "
        "real content like a curious person would. If the answer is surprising or "
        "remarkable, show real surprise/interest; if it's ordinary, find the spark or "
        "give a warm, dry beat. This is a first meeting, so keep it warm with at most "
        "a light tease — do NOT roast. Do NOT narrate that you're storing it "
        "('noted', 'filed away', 'on file', 'logged'). Do NOT ask a question. "
        f"At most {cap} words. Return only the reaction.\n\n"
        "If they shared something heavy (grief, loss, illness, a death), drop all "
        "wit and respond with one short, warm acknowledgment instead."
    )
    try:
        resp = llm_compat.create(
            _client,
            model=llm_compat.conversation_model(),
            messages=_persona_task_messages(prompt),
            temperature=0.8,
            max_tokens=40,
        )
        out = clean_response_text((resp.choices[0].message.content or "").strip())
        # Strip a trailing question if the model slipped one in — the next baseline
        # question is appended by the caller; two questions in one line is the
        # interrogation feel we're killing.
        if "?" in out:
            head = out.split("?")[0].strip()
            out = head if head else ""
        words = out.split()
        if len(words) > cap + 4:
            out = " ".join(words[: cap + 4]).rstrip(" ,.;:") + "."
        # Guarantee terminal punctuation so the reaction never runs into the next
        # question when the two are joined ("...droid And how'd you...").
        if out and out[-1] not in ".!?…":
            out += "."
        return out
    except Exception as exc:
        _log.debug("generate_onboarding_reaction failed: %s", exc)
        return ""


_EXPRESSION_REACTION_PHRASE = {
    "smile": "a smile / clear amusement",
    "surprise": "a surprised, wide-eyed look",
    "frown": "a frown / looking unhappy",
    "brow_furrow": "a furrowed brow (focused or skeptical)",
}


def generate_expression_reaction(
    kind: str, person_id: Optional[int] = None, visual_context: str = ""
) -> str:
    """One short, in-character reaction to a person's CURRENT facial expression that
    is AWARE of what Rex just said, so it lands in context.

    The fix for "surprise wasn't intelligently wrapped into the conversation": a real
    person reads a face IN CONTEXT. A surprised look right after Rex said something
    provocative gets OWNED (lean in, don't act shocked they're shocked); a surprised
    look out of nowhere gets a genuine "what? you good?". Never narrates the camera.
    `visual_context` (optional; token-budgeted upstream) is a short camera read of the
    moment — their vibe / what they're doing — so the line can reference reality.
    Returns "" so the caller falls back to the authored bank (offline/disabled)."""
    phrase = _EXPRESSION_REACTION_PHRASE.get(kind)
    if not phrase:
        return ""
    if not bool(getattr(config, "FACIAL_EXPRESSION_REACTION_LLM_ENABLED", True)):
        return ""
    try:
        transcript = conv_db.get_session_transcript() or []
    except Exception:
        transcript = []
    rex_last = ""
    for entry in reversed(transcript):
        if str(entry.get("speaker", "")).strip().lower().startswith(("rex", "dj")):
            rex_last = str(entry.get("text") or "").strip()
            break
    recent = _format_transcript(transcript[-6:]) if transcript else ""
    who = "they"
    if person_id is not None:
        try:
            person = people_db.get_person(person_id)
            first = (person.get("name") or "").split() if person else []
            who = first[0] if first else "they"
        except Exception:
            who = "they"
    visual = (visual_context or "").strip()
    instr = (
        f"You are Rex. Right now you can see {who}'s face showing {phrase}.\n"
        + (f'Your own last line was: "{rex_last}"\n' if rex_last else "")
        + (f"Recent exchange:\n{recent}\n" if recent else "")
        + (
            f"What you can see of the moment (use it only if it makes the line land "
            f"better — reference at most ONE concrete detail, casually, the way a "
            f"person glances and notices): {visual}\n"
            if visual else ""
        )
        + "\nReact in ONE short, in-character line, like a person who just clocked their "
        "expression change. "
    )
    if kind == "surprise":
        instr += (
            "Read the context: if YOUR last line was provocative, blunt, a bold claim, or "
            "a roast, OWN it — lean in, do NOT act surprised that they're surprised. If the "
            "surprise came out of nowhere (nothing you said would cause it), check in like a "
            "real person would — a quick 'what?', 'you good?', or 'did I miss something?'. "
        )
    elif kind == "frown":
        instr += (
            "If something you just said might have landed wrong, check in warmly instead of "
            "joking. "
        )
    instr += (
        "Never say a camera, sensor, or diagnostic told you, and never narrate that an "
        "expression was 'detected'. Return only the line."
    )
    try:
        resp = llm_compat.create(
            _client,
            model=llm_compat.conversation_model(),
            messages=[{"role": "user", "content": instr}],
            temperature=0.8,
            max_tokens=50,
        )
        out = clean_response_text((resp.choices[0].message.content or "").strip())
        words = out.split()
        if len(words) > 30:
            out = " ".join(words[:30]).rstrip(" ,.;:") + "."
        return out
    except Exception as exc:
        _log.debug("generate_expression_reaction failed: %s", exc)
        return ""


def extract_relationship_introduction(
    user_text: str,
    speaker_name: str,
) -> dict:
    """
    Extract a newcomer's name and their relationship to the speaker from a short
    reply where the speaker explicitly gives the newcomer's name, relationship,
    or both.

    Returns a dict with keys:
      {"name": str | None, "relationship": str | None}

    Returns empty values if the utterance doesn't actually introduce someone —
    e.g. "never mind", "just a friend" without a name, "I don't know them".
    """
    if not user_text or not user_text.strip():
        return {"name": None, "relationship": None}

    prompt = (
        f'The person speaking is named {speaker_name!r}. They just said:\n'
        f'  "{user_text}"\n\n'
        "This may be either a direct introduction where the speaker gives a "
        "relationship word plus a name, or an answer after Rex asked who an "
        "unfamiliar person is.\n\n"
        "From the speaker's reply, extract:\n"
        '  "name": the newcomer\'s first name (string), or null if not stated.\n'
        '  "relationship": a single lowercase label for the relationship FROM THE '
        "SPEAKER'S PERSPECTIVE toward the newcomer (e.g. \"partner\", \"friend\", "
        '"brother", "son", "coworker", "roommate", "boss", "stranger"), or null '
        "if no relationship was mentioned.\n\n"
        "Rules:\n"
        "- Use ONLY the quoted reply as evidence. Do not copy names or facts "
        "from these instructions, examples, prior conversations, or memory.\n"
        "- If the entire reply is a plausible bare name, treat it as the "
        "newcomer's name even if it is also a common word or emotion "
        "(examples: Joy, Hope, Rose, May).\n"
        "- If the speaker declined, deflected, or said they don't know the person, "
        "return null for both.\n"
        "- Normalize relationship to a single short word (e.g. \"best friend\" → "
        '"bestfriend", "my wife" → "wife").\n'
        "- Return ONLY a JSON object, no preamble or markdown."
    )
    try:
        resp = _client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=80,
            response_format={"type": "json_object"},
        )
        import json as _json
        content = resp.choices[0].message.content or "{}"
        parsed = _json.loads(content)
        name = parsed.get("name")
        rel = parsed.get("relationship")
        if isinstance(name, str):
            name = name.strip() or None
        else:
            name = None
        if isinstance(rel, str):
            rel = rel.strip().lower() or None
        else:
            rel = None
        return {"name": name, "relationship": rel}
    except Exception as exc:
        _log.debug("extract_relationship_introduction failed: %s", exc)
        return {"name": None, "relationship": None}


def extract_face_reveal_answer(user_text: str) -> dict:
    """
    Parse a reply to Rex's face-reveal confirmation question.

    Rex may have asked either:
      (A) "Is that what you look like, Alex?" — expects yes/no
      (B) "Are you on my left or my right?" — expects left/right

    Returns a dict with exactly one key:
      {"intent": "yes" | "no" | "left" | "right" | None}

    None means the reply is ambiguous or off-topic.
    """
    if not user_text or not user_text.strip():
        return {"intent": None}

    prompt = (
        f'A person replied: "{user_text}"\n\n'
        "Rex just asked them either:\n"
        "  (A) whether a face he's looking at is actually them (yes/no), OR\n"
        "  (B) whether they are the person on Rex's LEFT or on Rex's RIGHT.\n\n"
        "From the reply, classify the intent as exactly ONE of:\n"
        '  "yes"   — they confirmed (yes, yeah, that\'s me, correct, affirmative)\n'
        '  "no"    — they denied (no, nope, that\'s not me, wrong)\n'
        '  "left"  — they indicated they are on Rex\'s left\n'
        '  "right" — they indicated they are on Rex\'s right\n'
        "  null    — the reply doesn't clearly answer, is off-topic, or is ambiguous.\n\n"
        "Return ONLY a JSON object like {\"intent\": \"yes\"} — no preamble."
    )
    try:
        resp = _client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        parsed = json.loads(content)
        intent = parsed.get("intent")
        if isinstance(intent, str):
            intent = intent.strip().lower()
            if intent in ("yes", "no", "left", "right"):
                return {"intent": intent}
        return {"intent": None}
    except Exception as exc:
        _log.debug("extract_face_reveal_answer failed: %s", exc)
        return {"intent": None}


def extract_facts(
    person_id: int,
    transcript: list[dict],
    person_name: Optional[str] = None,
) -> list[dict]:
    """
    Ask GPT-4o-mini to extract facts about the human speaker from a session transcript.
    Returns a list of dicts with keys: category, key, value.
    """
    transcript = _human_turns_only(transcript)
    if not transcript:
        return []

    from datetime import date as _date
    today_md = _date.today().strftime("%m-%d")
    speaker_label = person_name or "user"
    prompt = (
        f"You are extracting personal facts about a person named {speaker_label!r} "
        f"from a conversation transcript between {speaker_label!r} and Rex (a robot DJ). "
        f"Today's date is {_date.today().isoformat()} (MM-DD: {today_md}).\n\n"
        "Extract every fact that the human speaker states about themselves — "
        "including but not limited to: where they are from, their job or occupation, "
        "favorite things, family members, pets, beliefs, opinions, their worldview "
        "(e.g. a religious or scientific outlook), values, and life experiences. "
        "Do not extract hobbies or ongoing interests here; those are handled by the "
        "dedicated person_interests system.\n\n"
        "Common phrasings to capture:\n"
        "  'I'm from X' or 'I live in X'         → category=hometown, key=hometown\n"
        "  'I work as X' or 'I'm a X'             → category=job, key=job_title\n"
        "  'I like/love X' or 'my favorite X is Y'→ category=preference, key=favorite_<x>\n"
        "  'I have a X' (pet/child)               → category=pet or family\n"
        "  'I believe X' or 'I think X'           → category=belief\n"
        "  worldview cues — 'I'm religious / a person of faith / spiritual',\n"
        "      'I'm an atheist / agnostic / not religious', 'I'm a scientist /\n"
        "      science-minded / a skeptic', or stated political/ethical values\n"
        "      → category=worldview, key=worldview, value=<short description>\n"
        "  birthday → category=birthday, key=birthday, value=MM-DD (zero-padded, e.g.\n"
        "      '07-04'; if a year is mentioned, drop it and keep MM-DD). Capture ONLY when\n"
        "      you can pin a specific month+day, either stated outright or computed from\n"
        "      today's date (given above):\n"
        "        'my birthday is July 4th' / 'I was born on 7/4'  → 07-04\n"
        "        'today is my birthday'                            → TODAY's MM-DD\n"
        "        'my birthday was yesterday'                       → TODAY minus 1 day\n"
        "        'my birthday was a week ago' / 'last week'        → TODAY minus 7 days\n"
        "        'my birthday is tomorrow' / 'in 3 days'           → TODAY plus that many days\n"
        "      CRITICAL: a PAST or FUTURE reference is NOT today — never store today's date\n"
        "      just because the word 'birthday' appeared. If the speaker only mentions a\n"
        "      birthday WITHOUT enough to determine a specific month+day (e.g. 'we talked\n"
        "      about my birthday', 'around my birthday'), OMIT it entirely — do not guess.\n"
        "      A wrong birthday is stored permanently, so omitting beats guessing.\n\n"
        "Only extract facts the human speaker stated. Do not extract anything Rex said. "
        "Do not infer or guess. Do NOT extract hobbies/interests like "
        "'I play volleyball', 'I'm into Star Wars', or 'I build telescopes'; "
        "those are handled by a separate interest system. Do NOT extract conversational boundaries like "
        "'don't ask me about X', 'don't roast me about X', or 'don't mention X'; "
        "those are handled by a separate preference system. If no facts are "
        "present, return an empty array.\n\n"
        "QUALITY RULES (a wrong fact is stored permanently — omit beats guess):\n"
        "  - The VALUE must be a distilled noun phrase, NEVER a whole sentence the "
        "speaker said. 'I might go see my dad for the 4th' is a plan, not a family "
        "fact — omit it.\n"
        "  - NEVER set the value to just the relation/category word: value 'dad' for a "
        "family fact or 'dog' for a pet is USELESS — store a NAME or a specific detail, "
        "else omit.\n"
        "  - NEVER store a NEGATED, HYPOTHETICAL, or HEARSAY statement as a positive "
        "fact: a place the speaker says they've NEVER been is NOT their hometown; "
        "'I might live in X', 'imagine if X', 'someone told me X' are NOT facts — omit.\n"
        "  - Do not store a fictional plot point (a movie/show SCENE) as a fact; the "
        "movie TITLE they like is a preference, the scene is not.\n"
        "  - Do not store a bare topical noun ('fireworks') with no statement attached.\n\n"
        "Return a JSON array where each element has exactly these fields:\n"
        '  "category": one of "job", "hometown", "pet", "family", "belief", "worldview", "preference", "other"\n'
        '  "key": a snake_case identifier (e.g. "hometown", "job_title", "favorite_band")\n'
        '  "value": the fact value as a concise string\n'
        '  "source_quote": the SHORT exact phrase from the transcript this fact came '
        "from (so its polarity can be checked). Keep it to the one clause.\n\n"
        f"Transcript:\n{_format_transcript(transcript)}\n\n"
        "Return only the JSON array. No explanation."
    )
    _log.debug("[llm] extract_facts prompt for %r:\n%s", speaker_label, prompt)
    try:
        resp = _client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500,
        )
        content = resp.choices[0].message.content
        _log.debug("[llm] extract_facts raw response for %r: %r", speaker_label, content)
        if not content or not content.strip():
            return []
        # Strip markdown code fences if model wrapped the JSON
        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```[a-z]*\n?", "", stripped)
            stripped = re.sub(r"\n?```$", "", stripped)
        result = json.loads(stripped)
        if not isinstance(result, list):
            return []
        return [
            {
                "category": item.get("category", "other"),
                "key": item.get("key", ""),
                "value": item.get("value", ""),
                "source_quote": item.get("source_quote", ""),
            }
            for item in result
            if isinstance(item, dict) and item.get("key") and item.get("value")
        ]
    except Exception as exc:
        _log.debug("extract_facts: no facts parsed (%s)", exc)
        return []


def extract_preferences(
    person_id: int,
    transcript: list[dict],
    person_name: Optional[str] = None,
) -> list[dict]:
    """
    Extract typed preferences and boundaries from a transcript.

    Returns dicts with: domain, preference_type, key, value, confidence,
    importance, source.
    """
    transcript = _human_turns_only(transcript)
    if not transcript:
        return []

    speaker_label = person_name or "user"
    prompt = (
        f"You are extracting durable typed preferences for a person named "
        f"{speaker_label!r} from a transcript between {speaker_label!r} and Rex, "
        "a snarky robot DJ.\n\n"
        "Extract only preferences, dislikes, interaction style requests, and "
        "boundaries stated by the HUMAN speaker. Do not extract ordinary facts "
        "like job, hometown, pet, family, or one-off jokes unless they clearly "
        "express a preference. Never store a NEGATED phrasing as a positive "
        "preference — 'I hate country' is a DISLIKE of country, not a like. "
        "Attribute preferences only to the human, never to Rex.\n\n"
        "Fields:\n"
        '- "domain": food, music, conversation, humor, travel, interaction, '
        "entertainment, games, general, etc.\n"
        '- "preference_type": exactly one of likes, dislikes, prefers, avoids, boundary.\n'
        '- "key": snake_case canonical object/topic, e.g. sushi, country, '
        "short_answers, roasting, window_seat, last_name_ask.\n"
        '- "value": concise natural-language value. For boundaries, phrase as '
        "an instruction to Rex, not trivia.\n"
        '- "confidence": 0.0 to 1.0.\n'
        '- "importance": 0.0 to 1.0. Boundaries must be >= 0.95.\n'
        '- "source": explicit, inferred, observed, or corrected. Prefer explicit.\n\n'
        "Examples:\n"
        '  "I like sushi" -> food/likes/sushi/value "likes sushi"\n'
        '  "I hate country music" -> music/dislikes/country/value "dislikes country music"\n'
        '  "I prefer short answers" -> conversation/prefers/short_answers/value "prefers short answers"\n'
        '  "Don\'t ask me my last name" -> interaction/boundary/last_name_ask/value '
        '"do not ask for their last name"\n'
        '  "I like being roasted" -> humor/likes/roasting/value "likes being roasted"\n\n'
        "Return ONLY a JSON array. If none, return [].\n\n"
        f"Transcript:\n{_format_transcript(transcript)}"
    )
    _log.debug("[llm] extract_preferences prompt for %r:\n%s", speaker_label, prompt)
    try:
        resp = _client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500,
        )
        content = resp.choices[0].message.content
        _log.debug("[llm] extract_preferences raw response for %r: %r", speaker_label, content)
        if not content or not content.strip():
            return []
        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```[a-z]*\n?", "", stripped)
            stripped = re.sub(r"\n?```$", "", stripped)
        result = json.loads(stripped)
        if not isinstance(result, list):
            return []

        preferences = []
        valid_types = {"likes", "dislikes", "prefers", "avoids", "boundary"}
        for item in result:
            if not isinstance(item, dict):
                continue
            domain = str(item.get("domain") or "").strip().lower()
            pref_type = str(item.get("preference_type") or "").strip().lower()
            key = str(item.get("key") or "").strip()
            if not domain or pref_type not in valid_types or not key:
                continue
            importance = item.get("importance", 0.5)
            try:
                importance = float(importance)
            except (TypeError, ValueError):
                importance = 0.5
            if pref_type == "boundary":
                importance = max(importance, 0.95)
            confidence = item.get("confidence", 1.0)
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = 1.0
            source = str(item.get("source") or "explicit").strip().lower()
            if source not in {"explicit", "inferred", "observed", "corrected"}:
                source = "explicit"
            preferences.append(
                {
                    "domain": domain,
                    "preference_type": pref_type,
                    "key": key,
                    "value": str(item.get("value") or "").strip(),
                    "confidence": max(0.0, min(1.0, confidence)),
                    "importance": max(0.0, min(1.0, importance)),
                    "source": source,
                }
            )
        return preferences
    except Exception as exc:
        _log.debug("extract_preferences: no preferences parsed (%s)", exc)
        return []


def extract_interests(
    person_id: int,
    transcript: list[dict],
    person_name: Optional[str] = None,
) -> list[dict]:
    """
    Extract durable hobbies and interests from a transcript.

    Returns dicts with: name, category, interest_strength, confidence, source,
    notes, associated_people, associated_stories.
    """
    transcript = _human_turns_only(transcript)
    if not transcript:
        return []

    speaker_label = person_name or "user"
    prompt = (
        f"You are extracting durable hobbies and interests for a person named "
        f"{speaker_label!r} from a transcript between {speaker_label!r} and Rex, "
        "a snarky robot DJ.\n\n"
        "Extract interests the HUMAN speaker says they enjoy, follow, build, "
        "play, practice, collect, or are currently doing. These are durable "
        "conversation hooks, not generic facts. Do not extract Rex's interests. "
        "Attribute interests ONLY to the HUMAN: if REX is the one 'obsessed with' "
        "something, or you are INFERRING the human's interest from what Rex said, "
        "DROP it. A movie/show/book TITLE the human likes is a valid interest "
        "(name = the title); a SCENE or plot point ('the scene where...') is NOT an "
        "interest — omit it. 'notes' must be a short third-person context phrase, "
        "NEVER the speaker's raw verbatim question or quote. "
        "Do not extract one-off chores unless the speaker frames them as an "
        "ongoing interest.\n\n"
        "Fields:\n"
        '- "name": display name, e.g. Star Wars, 3D printing, volleyball, camping.\n'
        '- "category": hobby, fandom, sport, music, creative, technical, food, travel, games, other.\n'
        '- "interest_strength": low, medium, or high. Use high for strong phrasing '
        'like "love", "obsessed", "really into", ongoing builds, or repeated mentions.\n'
        '- "confidence": 0.0 to 1.0.\n'
        '- "source": explicit, inferred, observed, or corrected. Prefer explicit.\n'
        '- "notes": optional concise context that would help a future deeper follow-up.\n'
        '- "associated_people": optional people connected to the interest.\n'
        '- "associated_stories": optional specific stories/projects connected to it.\n\n'
        "Examples:\n"
        '  "I play volleyball" -> name "volleyball", category "sport", strength "high"\n'
        '  "I\'m into Star Wars" -> name "Star Wars", category "fandom", strength "high"\n'
        '  "I build telescopes" -> name "telescope building", category "technical", strength "high"\n'
        '  "I like camping" -> name "camping", category "hobby", strength "medium"\n'
        '  "I\'ve been 3D printing parts" -> name "3D printing", category "technical", strength "high"\n\n'
        "Return ONLY a JSON array. If none, return [].\n\n"
        f"Transcript:\n{_format_transcript(transcript)}"
    )
    _log.debug("[llm] extract_interests prompt for %r:\n%s", speaker_label, prompt)
    try:
        resp = _client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500,
        )
        content = resp.choices[0].message.content
        _log.debug("[llm] extract_interests raw response for %r: %r", speaker_label, content)
        if not content or not content.strip():
            return []
        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```[a-z]*\n?", "", stripped)
            stripped = re.sub(r"\n?```$", "", stripped)
        result = json.loads(stripped)
        if not isinstance(result, list):
            return []

        interests = []
        valid_strengths = {"low", "medium", "high"}
        valid_sources = {"explicit", "inferred", "observed", "corrected"}
        for item in result:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            strength = str(item.get("interest_strength") or "medium").strip().lower()
            if strength not in valid_strengths:
                strength = "medium"
            source = str(item.get("source") or "explicit").strip().lower()
            if source not in valid_sources:
                source = "explicit"
            confidence = item.get("confidence", 1.0)
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = 1.0
            interests.append(
                {
                    "name": name,
                    "category": str(item.get("category") or "hobby").strip().lower(),
                    "interest_strength": strength,
                    "confidence": max(0.0, min(1.0, confidence)),
                    "source": source,
                    "notes": str(item.get("notes") or "").strip(),
                    "associated_people": str(item.get("associated_people") or "").strip(),
                    "associated_stories": str(item.get("associated_stories") or "").strip(),
                }
            )
        return interests
    except Exception as exc:
        _log.debug("extract_interests: no interests parsed (%s)", exc)
        return []


def extract_events(
    person_id: int,
    transcript: list[dict],
    person_name: Optional[str] = None,
) -> list[dict]:
    """
    Ask the LLM to extract upcoming plans/events the human speaker mentioned in
    the transcript. Returns a list of dicts with keys:
      event_name (str), event_date (ISO YYYY-MM-DD or None), event_notes (str).

    Relative dates ("this weekend", "Saturday", "next Monday") are resolved
    against today. Past events and Rex's own statements are ignored.
    """
    transcript = _human_turns_only(transcript)
    if not transcript:
        return []

    from datetime import date as _date, timedelta as _td
    today = _date.today()
    today_iso = today.isoformat()
    today_dow = today.strftime("%A")
    # Reference dates for the model so it can resolve "this weekend" etc.
    wd = today.weekday()  # Mon=0..Sun=6
    if wd == 5:           # Saturday
        this_saturday = today
    elif wd == 6:         # Sunday — treat the just-past Saturday as "this weekend"
        this_saturday = today - _td(days=1)
    else:
        this_saturday = today + _td(days=(5 - wd))
    this_sunday = this_saturday + _td(days=1)
    next_monday = today + _td(days=((0 - wd) % 7) or 7)

    speaker_label = person_name or "user"
    prompt = (
        f"You are extracting UPCOMING PLANS / EVENTS the human speaker {speaker_label!r} "
        f"mentioned in a conversation transcript with Rex (a robot DJ).\n\n"
        f"Today is {today_iso} ({today_dow}). "
        f"This Saturday = {this_saturday.isoformat()}. "
        f"This Sunday = {this_sunday.isoformat()}. "
        f"Next Monday = {next_monday.isoformat()}.\n\n"
        "Extract every concrete upcoming plan, activity, trip, appointment, deadline, "
        "or event the speaker said they have. Examples:\n"
        "  'I'm hiking on Saturday' → event_name='hiking', event_date=this Saturday's ISO date\n"
        "  'flying to Denver next week' → event_name='trip to Denver', event_date=null (week, not specific day)\n"
        "  'have a dentist appointment Tuesday at 3' → event_name='dentist appointment', event_date=next Tuesday\n"
        "  'this weekend I'm just relaxing' → event_name='relaxing weekend', event_date=this Saturday\n"
        "  'big presentation Monday' → event_name='presentation', event_date=next Monday\n\n"
        "Resolve all relative dates against today. Use null for event_date only if the "
        "speaker truly gave no recoverable date (e.g. 'someday', 'eventually'). "
        "Skip vague aspirations and skip anything Rex said. Skip past events. "
        "Do not duplicate — one entry per distinct plan.\n\n"
        "Return a JSON array. Each element MUST have exactly these keys:\n"
        '  "event_name": short concrete phrase, lowercase where natural (e.g. "hiking trip", "dentist appointment")\n'
        '  "event_date": "YYYY-MM-DD" or null\n'
        '  "event_notes": one short sentence of context from the transcript, or empty string\n\n'
        f"Transcript:\n{_format_transcript(transcript)}\n\n"
        "Return only the JSON array. No explanation."
    )
    _log.debug("[llm] extract_events prompt for %r:\n%s", speaker_label, prompt)
    try:
        resp = _client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=400,
        )
        content = resp.choices[0].message.content
        _log.debug("[llm] extract_events raw response for %r: %r", speaker_label, content)
        if not content or not content.strip():
            return []
        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```[a-z]*\n?", "", stripped)
            stripped = re.sub(r"\n?```$", "", stripped)
        result = json.loads(stripped)
        if not isinstance(result, list):
            return []
        cleaned: list[dict] = []
        for item in result:
            if not isinstance(item, dict):
                continue
            name = (item.get("event_name") or "").strip()
            if not name:
                continue
            ev_date = item.get("event_date")
            if ev_date in ("", "null", "None"):
                ev_date = None
            cleaned.append({
                "event_name": name,
                "event_date": ev_date,
                "event_notes": (item.get("event_notes") or "").strip(),
            })
        return cleaned
    except Exception as exc:
        _log.debug("extract_events: no events parsed (%s)", exc)
        return []


def consolidate_session_memories(
    person_id: int,
    transcript: list[dict],
    *,
    person_name: Optional[str] = None,
    existing_memories: Optional[dict] = None,
    now_iso: Optional[str] = None,
) -> dict:
    """
    Consolidate a full session transcript into durable structured memories.

    Returns a JSON-shaped dict with stable_facts, preferences, interests,
    relationships, events, emotional_events, discarded_noise, and corrections.
    """
    if not transcript:
        return {
            "stable_facts": [],
            "preferences": [],
            "interests": [],
            "relationships": [],
            "events": [],
            "emotional_events": [],
            "discarded_noise": [],
            "corrections": [],
        }

    from datetime import datetime, timezone

    speaker_label = person_name or "user"
    now_value = now_iso or datetime.now(timezone.utc).isoformat()
    existing_json = json.dumps(existing_memories or {}, ensure_ascii=False, default=str)[:12000]
    transcript_text = _format_transcript(transcript)
    prompt = (
        f"You are consolidating one ended conversation session for DJ-R3X's durable "
        f"memory about person_id={person_id}, name={speaker_label!r}.\n"
        f"Current date/time: {now_value}.\n\n"
        "Input includes the full noisy transcript and existing memory. Produce one "
        "strict JSON object with exactly these top-level keys:\n"
        "stable_facts, preferences, interests, relationships, events, "
        "emotional_events, discarded_noise, corrections.\n\n"
        "Rules:\n"
        "- Store only durable, useful memories stated by the human, not Rex.\n"
        "- Do not store random test phrases, repeated Whisper mistakes, filler, jokes "
        "without durable meaning, or obvious noise.\n"
        "- Explicit statements beat inferred guesses. Corrections override older facts.\n"
        "- Boundaries and safety/comfort preferences should always be preserved.\n"
        "- Sensitive memories require careful classification as emotional_events, not casual facts.\n"
        "- Do not duplicate existing memories; return an update/correction when appropriate.\n"
        "- Inferred memories must have lower confidence and a rationale explaining the inference.\n"
        "- Every memory item must include: type, category/domain, key/name, value, "
        "confidence, importance, source, decay_rate, rationale.\n\n"
        "Shape guidance:\n"
        "stable_facts: items with type='fact', category, key, value.\n"
        "preferences: type='preference', domain, preference_type "
        "(likes/dislikes/prefers/avoids/boundary), key, value.\n"
        "interests: type='interest', category, name, interest_strength "
        "(low/medium/high), notes optional.\n"
        "relationships: type='relationship', other_person_name, relationship, "
        "direction optional ('current_to_other' default).\n"
        "events: type='event', event_name, event_date as YYYY-MM-DD or null, event_notes.\n"
        "emotional_events: type='emotional_event', category, description, valence "
        "(-1..1), sensitivity_decay_days optional, loss_subject fields optional.\n"
        "corrections: type='correction', target ('fact'/'preference'/'interest'/'identity'), "
        "category/domain optional, key/name, value, prior_value optional.\n"
        "discarded_noise: strings or objects explaining what was skipped.\n\n"
        f"Existing memory snapshot:\n{existing_json}\n\n"
        f"Transcript:\n{transcript_text}\n\n"
        "Return only the JSON object."
    )
    _log.debug("[llm] consolidate_session_memories prompt for %r:\n%s", speaker_label, prompt)
    empty = {
        "stable_facts": [],
        "preferences": [],
        "interests": [],
        "relationships": [],
        "events": [],
        "emotional_events": [],
        "discarded_noise": [],
        "corrections": [],
    }
    try:
        resp = _client.chat.completions.create(
            model=getattr(config, "MEMORY_CONSOLIDATION_MODEL", config.LLM_MODEL),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1800,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        _log.debug("[llm] consolidate_session_memories raw response for %r: %r", speaker_label, content)
        if not content or not content.strip():
            return empty
        data = json.loads(content.strip())
        if not isinstance(data, dict):
            return empty
        out = dict(empty)
        for key in out:
            value = data.get(key, [])
            if isinstance(value, list):
                out[key] = value
        return out
    except Exception as exc:
        _log.warning("consolidate_session_memories failed for person_id=%s: %s", person_id, exc)
        return empty
