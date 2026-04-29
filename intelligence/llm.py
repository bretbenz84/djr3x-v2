"""
intelligence/llm.py — GPT-4o-mini streaming interface and prompt assembly for DJ-R3X.
"""

import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Generator, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
import apikeys
from intelligence import social_scene
from world_state import world_state
from memory import database as db
from memory import people as people_db
from memory import facts as facts_db
from memory import conversations as conv_db
from memory import relationships as rel_db
from memory import boundaries as boundaries_db

from openai import OpenAI

_log = logging.getLogger(__name__)

_client = OpenAI(api_key=apikeys.OPENAI_API_KEY)

_ASSISTANT_LABEL_RE = re.compile(
    r"^\s*(?:\[(?:rex|dj[- ]?r3x)\]|(?:rex|dj[- ]?r3x))\s*[:\-–—]\s*",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def clean_response_text(text: str) -> str:
    """Remove accidental spoken speaker labels from assistant replies."""
    cleaned = (text or "").strip()
    while cleaned:
        updated = _ASSISTANT_LABEL_RE.sub("", cleaned, count=1).strip()
        if updated == cleaned:
            break
        cleaned = updated
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


def _summarize_world_state(ws: dict) -> str:
    parts = []

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
        if bits:
            social_cues.append(f"{name}: " + ", ".join(bits))
    if social_cues:
        parts.append(
            "Visible social cues: "
            + "; ".join(social_cues)
            + ". Treat intimate camera distance as physically close; by American "
            "personal-space norms, someone extremely close may be playfully too close for comfort."
        )

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
    parts.append(time_line)

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

_TIER_ROAST_STYLE = {
    "stranger":     "Roast style: observational, surface-level, crowd-pleasing.",
    "acquaintance": "Roast style: lightly personal — references the few facts you know. Friendly but with an edge.",
    "friend":       "Roast style: personal — uses real knowledge against them. Affectionate but pointed.",
    "close_friend": "Roast style: surgical — you know exactly where to aim. Delivered with obvious warmth.",
    "best_friend":  "Roast style: devastating — the full arsenal, zero mercy, maximum affection.",
}

# Conversation IDs Rex has already surfaced as nostalgia this session, so the
# same memory isn't called back twice. Cleared on process restart.
_nostalgia_used_this_session: set[int] = set()

# Fact IDs Rex has already prompted to confirm this session, so the same stale
# fact isn't re-asked turn after turn. Cleared on process restart.
_stale_facts_asked_this_session: set[int] = set()


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
        and (
            (f.get("age_days") is not None and f.get("age_days") >= days)
            or float(f.get("confidence") or 0.0) < min_conf
        )
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda f: (
            float(f.get("confidence") or 0.0),
            -(f.get("age_days") or 0),
        )
    )
    chosen = candidates[0]
    _stale_facts_asked_this_session.add(chosen["id"])
    return chosen


def _pick_nostalgia_callback(person_id: int, tier: str) -> Optional[dict]:
    """
    Roll the nostalgia probability and, on success, return a past conversation
    record that hasn't been surfaced this session. Returns None when the roll
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
    chosen = random.choice(candidates)
    _nostalgia_used_this_session.add(chosen["id"])
    return chosen


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
    if not match:
        return default
    return _RESPONSE_LENGTH_TOKEN_BUDGET.get(match.group(1).lower(), default)


def _build_person_context(person_id: int) -> str:
    person = people_db.get_person(person_id)
    if not person:
        return ""

    lines = []
    name = person.get("name") or "unknown"
    tier = person.get("friendship_tier", "stranger")
    lines.append(f"Person: {name} (tier: {tier}).")

    lines.append(
        f"Relationship — warmth: {person.get('warmth_score', 0.0):.2f}, "
        f"antagonism: {person.get('antagonism_score', 0.0):.2f}, "
        f"trust: {person.get('trust_score', 0.5):.2f}, "
        f"net: {person.get('net_relationship_score', 0.0):.2f}."
    )

    insult_count = person.get("lifetime_insult_count", 0)
    if insult_count:
        lines.append(f"Lifetime insults from this person: {insult_count}.")

    # skin_color is stored for recognition only — never inject into LLM context
    facts = facts_db.get_prompt_facts(person_id, limit=12)
    _log.info("[llm] loaded %d facts for %s", len(facts), name)
    if facts:
        fact_strs = [facts_db.format_fact_for_prompt(f) for f in facts]
        lines.append("Known facts: " + ", ".join(fact_strs) + ".")
        if any(
            f.get("confidence_label") != "high"
            or f.get("freshness_label") in {"aging", "stale", "unknown"}
            for f in facts
        ):
            lines.append(
                "Memory quality rule: stale or low-confidence facts are tentative. "
                "Phrase them as memories, not certainties; don't build sharp roasts "
                "or important decisions on them without confirming first."
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

    last_conv = conv_db.get_last_conversation(person_id)
    if last_conv:
        lines.append(
            f"Last conversation: {last_conv.get('summary', '')} "
            f"(tone: {last_conv.get('emotion_tone', 'neutral')})."
        )

    callback_hook_used = False
    try:
        from memory import emotional_events as _emo_events_for_hooks
        ws_now = world_state.snapshot()
        crowd_count_for_hooks = int((ws_now.get("crowd") or {}).get("count", 1) or 1)
        suppress_in_crowd = bool(getattr(config, "EMPATHY_DISCRETION_IN_CROWD", True))
        callback_hook_used = any(
            not ev.get("last_acknowledged_at")
            and _emo_events_for_hooks.can_surface_event(ev)
            and not (
                suppress_in_crowd
                and crowd_count_for_hooks > 1
                and _emo_events_for_hooks.is_heavy_event(ev)
            )
            for ev in _emo_events_for_hooks.get_active_events(person_id, limit=3)
        )
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

    nostalgia = None if callback_hook_used else _pick_nostalgia_callback(person_id, tier)
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
            f"Next unanswered question to weave in naturally: "
            f"\"{next_q['text']}\" (depth {next_q['depth']})."
        )
        callback_hook_used = True

    # Known inter-person relationships (e.g. "Bret is partner of JT").
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

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

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
    sections.append("Current personality parameters:\n" + param_lines)

    # 3. Current emotion state — Rex's own mood, plus (if known) the person's
    # affect and the empathy-layer directive for how to respond.
    ws = world_state.snapshot()
    emotion = ws.get("self_state", {}).get("emotion", "neutral")
    emotion_block = [f"Rex's own emotion state: {emotion}."]
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

    # 7. Behavioral rules
    rules = [
        "Never break character under any circumstances.",
        "Never acknowledge being a prop, an AI, or a software program.",
        "You are Rex. You have always been Rex.",
    ]

    season = ws.get("time", {}).get("season")
    if season in _SEASONAL_TONE:
        rules.append(_SEASONAL_TONE[season])

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
            known_facts = facts_db.get_prompt_facts(person_id, limit=12)
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
    if person_id is not None:
        unknown_in_frame = any(
            p.get("person_db_id") is None for p in ws.get("people", [])
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

    return "\n\n---\n\n".join(sections)


def stream_response(
    user_text: str,
    person_id: Optional[int] = None,
    agenda_directive: Optional[str] = None,
) -> Generator[str, None, None]:
    """Assemble the system prompt and stream GPT-4o-mini response chunks."""
    system_prompt = assemble_system_prompt(person_id, agenda_directive=agenda_directive)
    try:
        stream = _client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            stream=True,
            max_tokens=_max_tokens_for_agenda(agenda_directive),
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
    except Exception as exc:
        _log.error("stream_response failed: %s", exc)
        yield "...my circuits are experiencing some turbulence. Try again."


def get_response(
    user_text: str,
    person_id: Optional[int] = None,
    agenda_directive: Optional[str] = None,
) -> str:
    """Assemble the system prompt and return the full GPT-4o-mini response as a string."""
    return clean_response_text(
        "".join(stream_response(user_text, person_id, agenda_directive=agenda_directive))
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
        )
        result = json.loads(resp.choices[0].message.content.strip())
        for k, v in _defaults.items():
            result.setdefault(k, v)
        return result
    except Exception as exc:
        _log.error("analyze_sentiment failed: %s", exc)
        return dict(_defaults)


def generate_session_summary(person_id: int, transcript: list[dict]) -> str:
    """
    Send a session transcript to GPT-4o-mini and return a brief summary string
    suitable for storing in the conversations table.
    """
    if not transcript:
        return ""
    prompt = (
        "You are summarizing a conversation between DJ-R3X (Rex), a robot DJ droid, "
        "and a person he met. Write a 2–3 sentence summary capturing: "
        "main topics discussed, emotional tone, anything notable or memorable. "
        "Write in third person. Be concise.\n\n"
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


def generate_curiosity_question(response_text: str, user_text: str) -> str:
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
    prompt = (
        f'Rex just said: "{response_text}"\n'
        f'The human said: "{user_text}"\n\n'
        "Generate ONE short follow-up question Rex would naturally ask next, "
        "in his snarky droid character. One sentence only. "
        "Make it feel natural, not interrogative.\n\n"
        f"{tone_clause}"
    )
    try:
        resp = _client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=60,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        _log.debug("generate_curiosity_question failed: %s", exc)
        return ""


def extract_relationship_introduction(
    user_text: str,
    speaker_name: str,
) -> dict:
    """
    Extract a newcomer's name and their relationship to the speaker from a short
    reply like "Oh this is my partner JT" or "that's my brother Mike".

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
        "This may be either a direct introduction (\"this is my partner JT\") "
        "or an answer after Rex asked who an unfamiliar person is.\n\n"
        "From the speaker's reply, extract:\n"
        '  "name": the newcomer\'s first name (string), or null if not stated.\n'
        '  "relationship": a single lowercase label for the relationship FROM THE '
        "SPEAKER'S PERSPECTIVE toward the newcomer (e.g. \"partner\", \"friend\", "
        '"brother", "son", "coworker", "roommate", "boss", "stranger"), or null '
        "if no relationship was mentioned.\n\n"
        "Rules:\n"
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
      (A) "Is that what you look like, JT?" — expects yes/no
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
        "hobbies, interests, favorite things, family members, pets, beliefs, opinions, "
        "life experiences, and personal preferences.\n\n"
        "Common phrasings to capture:\n"
        "  'I'm from X' or 'I live in X'         → category=hometown, key=hometown\n"
        "  'I work as X' or 'I'm a X'             → category=job, key=job_title\n"
        "  'I like/love X' or 'my favorite X is Y'→ category=preference, key=favorite_<x>\n"
        "  'I have a X' (pet/child)               → category=pet or family\n"
        "  'I'm into X' or 'I do X for fun'       → category=hobby\n"
        "  'I believe X' or 'I think X'           → category=belief\n"
        "  'my birthday is X' / 'I was born on X' / 'today is my birthday'\n"
        "      → category=birthday, key=birthday, value=MM-DD (zero-padded, e.g. '07-04')\n"
        "      If the year is mentioned use MM-DD only — drop the year.\n"
        "      If the speaker says 'today is my birthday', use today's MM-DD.\n\n"
        "Only extract facts the human speaker stated. Do not extract anything Rex said. "
        "Do not infer or guess. Do NOT extract conversational boundaries like "
        "'don't ask me about X', 'don't roast me about X', or 'don't mention X'; "
        "those are handled by a separate preference system. If no facts are "
        "present, return an empty array.\n\n"
        "Return a JSON array where each element has exactly these fields:\n"
        '  "category": one of "job", "hometown", "hobby", "interest", "pet", "family", "belief", "preference", "other"\n'
        '  "key": a snake_case identifier (e.g. "hometown", "job_title", "favorite_band")\n'
        '  "value": the fact value as a concise string\n\n'
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
            }
            for item in result
            if isinstance(item, dict) and item.get("key") and item.get("value")
        ]
    except Exception as exc:
        _log.debug("extract_facts: no facts parsed (%s)", exc)
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
