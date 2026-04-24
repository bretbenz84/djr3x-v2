"""
intelligence/llm.py — GPT-4o-mini streaming interface and prompt assembly for DJ-R3X.
"""

import json
import logging
import re
import sys
from pathlib import Path
from typing import Generator, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
import apikeys
from world_state import world_state
from memory import database as db
from memory import people as people_db
from memory import facts as facts_db
from memory import conversations as conv_db
from memory import relationships as rel_db

from openai import OpenAI

_log = logging.getLogger(__name__)

_client = OpenAI(api_key=apikeys.OPENAI_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

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
    parts.append(f"Crowd: {crowd.get('count_label', 'unknown')} ({crowd.get('count', 0)} people).")

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
    if time_s.get("notable_date"):
        time_line += f" Notable date: {time_s['notable_date']}."
    parts.append(time_line)

    animals = ws.get("animals", [])
    if animals:
        parts.append("Animals present: " + ", ".join(a.get("species", "unknown") for a in animals) + ".")

    return " ".join(parts)


_TIER_ROAST_STYLE = {
    "stranger":     "Roast style: observational, surface-level, crowd-pleasing.",
    "acquaintance": "Roast style: lightly personal — references the few facts you know. Friendly but with an edge.",
    "friend":       "Roast style: personal — uses real knowledge against them. Affectionate but pointed.",
    "close_friend": "Roast style: surgical — you know exactly where to aim. Delivered with obvious warmth.",
    "best_friend":  "Roast style: devastating — the full arsenal, zero mercy, maximum affection.",
}

_ANGER_RULES = {
    1: "Anger level 1 (DEFENSIVE): Sharp witty comeback, slight attitude. Still cooperative.",
    2: "Anger level 2 (IRRITATED): Noticeably short, sarcastic, less cooperative.",
    3: "Anger level 3 (ANGRY): Clipped responses, raised affect, refuses certain requests.",
    4: "Anger level 4 (SHUTDOWN): Refuse to engage. Deliver a final dismissal line and ignore further input.",
}


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
    facts = [f for f in facts_db.get_facts(person_id) if f.get("key") != "skin_color"]
    _log.info("[llm] loaded %d facts for %s", len(facts), name)
    if facts:
        fact_strs = [f"{f['key']}: {f['value']}" for f in facts[:12]]
        lines.append("Known facts: " + ", ".join(fact_strs) + ".")

    last_conv = conv_db.get_last_conversation(person_id)
    if last_conv:
        lines.append(
            f"Last conversation: {last_conv.get('summary', '')} "
            f"(tone: {last_conv.get('emotion_tone', 'neutral')})."
        )

    next_q = rel_db.get_next_question(person_id, tier)
    if next_q:
        lines.append(
            f"Next unanswered question to weave in naturally: "
            f"\"{next_q['text']}\" (depth {next_q['depth']})."
        )

    # Known inter-person relationships (e.g. "Bret is partner of JT").
    try:
        from memory import social as _social
        rel_summary = _social.summarize_for_prompt(person_id, name)
        if rel_summary:
            lines.append("Known relationships to others: " + rel_summary + ".")
    except Exception as exc:
        _log.debug("relationship summary error: %s", exc)

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def assemble_system_prompt(person_id: Optional[int] = None) -> str:
    """Build the full layered system prompt in the order specified by CONTEXT.md."""
    sections = []

    # 1. Core character prompt
    sections.append(config.REX_CORE_PROMPT.strip())

    # 2. Personality parameter values
    params = _get_personality_params()
    param_lines = "\n".join(f"  {k}: {v}/100" for k, v in params.items())
    sections.append("Current personality parameters:\n" + param_lines)

    # 3. Current emotion state
    ws = world_state.snapshot()
    emotion = ws.get("self_state", {}).get("emotion", "neutral")
    sections.append(f"Current emotion state: {emotion}.")

    # 4. WorldState snapshot summary
    sections.append("World context:\n" + _summarize_world_state(ws))

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

    if person_id is not None:
        person = people_db.get_person(person_id)
        if person:
            tier = person.get("friendship_tier", "stranger")
            if tier in _TIER_ROAST_STYLE:
                rules.append(_TIER_ROAST_STYLE[tier])
            known_facts = [
                f for f in facts_db.get_facts(person_id)
                if f.get("key") != "skin_color"
            ]
            if known_facts:
                rules.append(
                    "You already know the following facts about this person — do NOT ask about "
                    "things already listed in their facts or mentioned in the conversation summary. "
                    "Use what you know naturally in conversation instead of re-asking it."
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

    sections.append("Behavioral rules:\n" + "\n".join(f"- {r}" for r in rules))

    return "\n\n---\n\n".join(sections)


def stream_response(
    user_text: str,
    person_id: Optional[int] = None,
) -> Generator[str, None, None]:
    """Assemble the system prompt and stream GPT-4o-mini response chunks."""
    system_prompt = assemble_system_prompt(person_id)
    try:
        stream = _client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            stream=True,
            max_tokens=150,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
    except Exception as exc:
        _log.error("stream_response failed: %s", exc)
        yield "...my circuits are experiencing some turbulence. Try again."


def get_response(user_text: str, person_id: Optional[int] = None) -> str:
    """Assemble the system prompt and return the full GPT-4o-mini response as a string."""
    return "".join(stream_response(user_text, person_id))


def analyze_sentiment(text: str) -> dict:
    """
    Classify an utterance for sentiment signals Rex reacts to.
    Returns: {is_insult, is_apology, is_compliment, emotion_detected}
    """
    _defaults = {
        "is_insult": False,
        "is_apology": False,
        "is_compliment": False,
        "emotion_detected": "neutral",
    }
    prompt = (
        "Classify the following utterance for a robot DJ character. "
        "Return a JSON object with exactly these fields:\n"
        '  "is_insult": true or false\n'
        '  "is_apology": true or false\n'
        '  "is_compliment": true or false\n'
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


def generate_curiosity_question(response_text: str, user_text: str) -> str:
    """
    Generate one short contextual follow-up question in Rex's voice.
    Used by the curiosity routine when the question pool is exhausted or unavailable.
    """
    prompt = (
        f'Rex just said: "{response_text}"\n'
        f'The human said: "{user_text}"\n\n'
        "Generate ONE short follow-up question Rex would naturally ask next, "
        "in his snarky droid character. One sentence only. "
        "Make it feel natural, not interrogative."
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
        "Rex had just asked them who a new unfamiliar person in the room is, and "
        "what that person's relationship to them is.\n\n"
        "From the speaker's reply, extract:\n"
        '  "name": the newcomer\'s first name (string), or null if not stated.\n'
        '  "relationship": a single lowercase label for the relationship FROM THE '
        "SPEAKER'S PERSPECTIVE toward the newcomer (e.g. \"partner\", \"friend\", "
        '"brother", "son", "coworker", "roommate", "boss", "stranger"), or null '
        "if no relationship was mentioned.\n\n"
        "Rules:\n"
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

    speaker_label = person_name or "user"
    prompt = (
        f"You are extracting personal facts about a person named {speaker_label!r} "
        f"from a conversation transcript between {speaker_label!r} and Rex (a robot DJ).\n\n"
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
        "  'I believe X' or 'I think X'           → category=belief\n\n"
        "Only extract facts the human speaker stated. Do not extract anything Rex said. "
        "Do not infer or guess. If no facts are present, return an empty array.\n\n"
        "Return a JSON array where each element has exactly these fields:\n"
        '  "category": one of "job", "hometown", "hobby", "pet", "family", "belief", "preference", "other"\n'
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
