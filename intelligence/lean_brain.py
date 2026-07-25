"""
intelligence/lean_brain.py — the lean conversation core (rebuild, Phase 0: react mode).

ONE streaming model call replaces the four-stage brain (action_router → conversation_agenda →
social_frame → a 4,400-word assembled prompt). The whole reply prompt is:

    the coherent Rex persona (config.REX_CORE_PROMPT — it already carries every taste rule:
    "let small things be small", "drop the bit on sincerity", "one move per turn", the
    anti-tic rules)  +  a SMALL live-context block (who you're with, a few real facts, the
    scene)  +  the recent turns as REAL user/assistant chat messages.

No agenda, no behavior menu, no per-turn contract, no 207 contradictory directives. Trust the
model; let silence be silence (this module only ever REACTS — it never fills a lull).

Latency-first design:
  * one call, not the current three-to-four sequential calls;
  * a small, consistent prompt → fast time-to-first-token;
  * `stream_reply` yields raw chunks and `stream_sentences` yields complete sentences, so the
    live path can speak the first sentence the moment it exists (first audio doesn't wait for
    the whole reply).

Nothing here runs until config.LEAN_BRAIN_ENABLED is set and the seam is wired in; today it is
exercised only by the offline A/B harness tools/lean_replay.py.
"""

from __future__ import annotations

import logging
import random
import re
import time
from typing import Generator, Optional

import config
from intelligence import llm, llm_compat

_log = logging.getLogger(__name__)

# Speakers whose transcript lines are Rex's own (mapped to the assistant role).
_REX_SPEAKERS = {"rex", "dj-r3x", "dj rex", "djr3x", "r3x", "dj r3x"}

# Split on a sentence end followed by whitespace — used to stream sentence-by-sentence.
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def _persona() -> str:
    persona = (getattr(config, "LEAN_BRAIN_PERSONA", "") or "").strip() or config.REX_CORE_PROMPT
    # Inline v3 delivery tags: when the TTS layer can honor them, tell the model it may
    # place one whitelisted [tag] mid-reply where the delivery genuinely shifts. audio.tts
    # sanitizes whatever comes back (whitelist + cap) and every log/GUI/memory seam strips
    # tags, so this is safe by construction; the rule is "" whenever tags can't land.
    # Single choke point: covers replies, lull-breakers, and one-voice directives alike.
    try:
        from audio import tts as _tts
        tag_rule = _tts.llm_inline_tag_rule()
    except Exception:
        tag_rule = ""
    return persona + ("\n\n" + tag_rule if tag_rule else "")


def _model() -> str:
    return (getattr(config, "LEAN_BRAIN_MODEL", "") or "").strip() or llm_compat.conversation_model()


def _first_name(person: Optional[dict]) -> str:
    name = str((person or {}).get("name") or "").strip()
    return name.split()[0] if name else ""


def _recent_topics(person_id: Optional[int]) -> list[str]:
    """What Rex + this person already covered in recent PRIOR runs (from rex.db) — so neither a
    reply nor a silence-break re-opens the same thing every boot. [] when disabled/unavailable."""
    if person_id is None or not bool(getattr(config, "RECENT_TOPICS_AWARENESS_ENABLED", True)):
        return []
    try:
        from memory import episodic_recall
        return episodic_recall.recent_conversation_topics(
            int(person_id), limit=int(getattr(config, "RECENT_TOPICS_LIMIT", 4))
        )
    except Exception as exc:
        _log.debug("[lean] recent topics read failed: %s", exc)
        return []


def _person_lines(person_id: Optional[int]) -> list[str]:
    """A handful of REAL things about who Rex is talking to — name/relationship + a few facts
    and interests. Deliberately small: no callbacks, plans, episodic recall, or nostalgia
    (those are the old bloat). Fail-safe to [] so a missing DB never breaks a reply."""
    if person_id is None:
        return []
    out: list[str] = []
    try:
        from memory import people
        person = people.get_person(int(person_id))
    except Exception:
        person = None
    if not person:
        return []
    who = _first_name(person) or "them"
    full_name = str(person.get("name") or "").strip()
    tier = str(person.get("friendship_tier") or "").strip().lower()
    out.append(f"You're talking with {who}" + (f" — {tier}." if tier and tier != "stranger" else "."))
    try:
        from intelligence import person_specials
        is_creator = person_specials.is_rex_creator(full_name)
    except Exception:
        is_creator = False
    if is_creator:
        # Bret is the creator/builder AND the person Rex knows best — and he has explicitly, repeatedly
        # asked to be roasted. The DB tier ("acquaintance") badly under-states the bond, and the classic
        # creator hook even says "don't make Bret a roast target" — exactly backwards for what he wants.
        # So license it hard here: rib him like family.
        out.append(
            "This is Bret — your CREATOR, the person you know best, and he LOVES the sparring and has "
            "flat-out asked to be roasted. Kid gloves are WRONG here: rib him like family — sharp, "
            "specific, no holding back. The bond runs deep underneath (affectionate ribbing, never "
            "real contempt), and you still drop the bit instantly on a genuinely sincere moment."
        )
    elif tier and tier != "stranger":
        # Known people enjoy the sparring and can take a pointed jab — don't play it safe. Strangers
        # get no such line and stay warm (matches the first-meeting onboarding tone).
        out.append(
            "You know " + who + " — the more you two know each other, the harder you can go; they "
            "enjoy the sparring and can take a sharp, SPECIFIC roast, so don't soften your wit to be "
            "polite. (Still: drop it instantly on a genuinely sincere or vulnerable moment.)"
        )
    background: list[str] = []
    try:
        from memory import facts as _facts
        background += [
            str(f.get("value") or f.get("text") or "").strip()
            for f in (_facts.get_prompt_worthy_facts(int(person_id), limit=4) or [])
        ]
    except Exception as exc:
        _log.debug("[lean] facts read failed: %s", exc)
    try:
        from memory import interests as _interests
        background += [
            str(it.get("name") or "").strip()
            for it in (_interests.get_interests_for_prompt(int(person_id), limit=4) or [])
        ]
    except Exception as exc:
        _log.debug("[lean] interests read failed: %s", exc)
    background = [b for b in background if b][:7]
    if background:
        # Framed hard as BACKGROUND, not fodder: dredging a stored hobby the person didn't just
        # raise (e.g. opening with "so, shooting any nebulae?") is the exact out-of-nowhere move
        # the owner keeps flagging. React to the ACTUAL conversation; touch this only when relevant.
        out.append(
            "Background you happen to know about " + who + " — do NOT bring any of it up unless THEY "
            "raise it or it's directly relevant to what they JUST said; NEVER open with it or dredge "
            "a hobby/topic they didn't mention: " + "; ".join(background) + "."
        )
    topics = _recent_topics(person_id)
    if topics:
        out.append(
            "Things you and " + who + " have talked about in recent chats — these are IN YOUR MEMORY. "
            "If they ask about any of it ('what are my plans?', 'what am I doing this weekend?', 'what "
            "did I tell you about…?'), RECALL and answer accurately from this list — do NOT claim they "
            "never told you or that you have nothing, when the answer is right here. Just don't "
            "PROACTIVELY dredge them up unprompted or re-ask as if it's new (the 'same thing every "
            "run' problem): " + " | ".join(topics) + "."
        )
    return out


def _scene_lines(world: Optional[dict]) -> list[str]:
    """A one-line 'what's around you right now' from a world_state snapshot. Empty in the
    offline replay (world is None); fleshed out when the live seam passes world_state."""
    if not world:
        return []
    try:
        bits: list[str] = []
        # LOCAL date first — without it the model guesses what "today" is and gets
        # relative days wrong ("tonight" for a tomorrow event). ~8 tokens.
        from datetime import datetime as _dt
        _now = _dt.now()
        _h = _now.hour
        _bucket = ("deep late-night" if _h < 5 else "early morning" if _h < 9
                   else "morning" if _h < 12 else "afternoon" if _h < 17
                   else "evening" if _h < 21 else "late evening")
        bits.append(_now.strftime("%A %Y-%m-%d %-I:%M %p") + f" ({_bucket})")
        tod = str(world.get("time_of_day") or world.get("part_of_day") or "").strip()
        if tod:
            bits.append(tod)
        people = world.get("people") or []
        names = [str(p.get("name") or "").strip() for p in people if isinstance(p, dict)]
        names = [n for n in names if n]
        if len(names) > 1:
            bits.append("with you: " + ", ".join(names))
        return ["Scene: " + "; ".join(bits) + "."] if bits else []
    except Exception:
        return []


def _room_belief_lines() -> list[str]:
    """Which ROOM Rex is in, grounded in the place recognizer.

    Lives here (the shared system-prompt builder) rather than only in
    _situation_block, because that block feeds ONLY the proactive
    consider_initiating path — so under ONE VOICE a direct question got no room
    grounding at all and Rex dodged it (field 2026-07-24: "What room are you in?"
    -> "I'm in whatever room you're in, unfortunately", while the recognizer was
    scoring the enrolled workshop at 0.83-0.87 the entire time).
    """
    try:
        from intelligence import place_questions
        clause = place_questions.belief_clause()
        return [clause] if clause else []
    except Exception:
        return []


def _system_prompt(
    person_id: Optional[int],
    world: Optional[dict],
    extra_lines: Optional[list[str]] = None,
) -> str:
    persona = _persona()
    ctx = (
        _person_lines(person_id)
        + _scene_lines(world)
        + _room_belief_lines()
        + list(extra_lines or [])
    )
    if not ctx:
        return persona
    return persona + "\n\nRight now:\n" + "\n".join("- " + line for line in ctx)


# ── Multi-party awareness ────────────────────────────────────────────────────
# Identity resolution knows WHO is speaking (transcript turns carry real names) —
# but this layer used to flatten every human into an anonymous "user" role, so the
# model literally could not tell Bret's lines from JT's: it answered JT's questions
# as if Bret asked them and addressed everything to the primary person. When 2+
# distinct humans appear in the recent window, history turns get speaker labels,
# the current turn names its speaker, and a room block tells the model who's who.
_GUEST_LABEL_RE = re.compile(r"^unknown_voice_(\d+)$", re.IGNORECASE)


def _display_speaker(raw: str) -> str:
    """Human-friendly short label for a transcript speaker."""
    s = (raw or "").strip()
    m = _GUEST_LABEL_RE.match(s)
    if m:
        return f"Guest {m.group(1)}"
    return s.split()[0] if s else "Guest"


def _current_speaker_display(person_id: Optional[int]) -> str:
    if person_id is None:
        return "Guest"
    try:
        from memory import people
        person = people.get_person(int(person_id))
        name = str((person or {}).get("name") or "").strip()
        return name.split()[0] if name else "Guest"
    except Exception:
        return "Guest"


def _other_participant_lines(
    raw_speakers: list[str], current_display: str
) -> list[str]:
    """One compact context line per OTHER named participant (max 2) so Rex knows who
    the second voice in the room IS — tier, a couple of interests, and any authored
    celebrity persona hook (the JT volleyball bit must fire when JT interjects)."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in raw_speakers:
        disp = _display_speaker(raw)
        if disp == current_display or disp in seen or disp.startswith("Guest"):
            continue
        seen.add(disp)
        bits: list[str] = []
        try:
            from memory import people
            row = people.find_person_by_name(raw)
            if row:
                tier = str(row.get("friendship_tier") or "").strip().lower()
                if tier and tier != "stranger":
                    bits.append(tier)
                try:
                    from memory import interests as _interests
                    likes = [
                        str(it.get("name") or "").strip()
                        for it in (_interests.get_interests_for_prompt(int(row["id"]), limit=2) or [])
                    ]
                    likes = [x for x in likes if x]
                    if likes:
                        bits.append("into " + ", ".join(likes))
                except Exception:
                    pass
        except Exception:
            pass
        line = f"Also in this conversation: {disp}" + (f" ({'; '.join(bits)})" if bits else "") + "."
        try:
            from intelligence import person_specials
            special = person_specials.special_prompt_context(raw)
            if special:
                line += " " + " ".join(special.split())
        except Exception:
            pass
        out.append(line)
        if len(seen) >= 2:
            break
    return out


# ── Flat-answer follow-up (owner spec 2026-07-06) ────────────────────────────
# "It's okay" answering "how's your day?" is a half-answer hiding the story. The
# lull impulse picks the loose end up ~15s later; the stronger move is the REPLY
# carrying the probe — quip plus "what's the missing 30%?" in one breath.
# Guards against interview mode: fires only when Rex's LAST line was itself a
# question (an "okay" acknowledging a statement is agreement, not flatness),
# once per cooldown window, never in a heavy/sober window, and the instruction
# says to let it go if they deflect again.

_FLAT_FILLER_RE = re.compile(
    r"^(?:uh|um|well|yeah|yea|nah|honestly|i mean|like|so)[\s,]+", re.IGNORECASE
)
_FLAT_PREFIX_RE = re.compile(
    r"^(?:it'?s|it was|i'?m|i am|im|things are|life'?s|life is)\s+", re.IGNORECASE
)
_FLAT_CORE = {
    "okay", "ok", "fine", "alright", "all right", "meh", "whatever",
    "not much", "nothing", "nothing much", "nothing really", "i guess",
    "okay i guess", "fine i guess", "good i guess", "eh", "not bad",
    "could be worse", "same as always", "same as usual", "same old",
    "same old same old", "dunno", "i dunno", "idk", "i don't know",
    "hanging in there", "hangin in there", "pretty good", "going okay",
    "going alright", "it goes", "surviving", "tired",
}
_last_flat_probe_at: float = 0.0


def _is_flat_answer(text: str) -> bool:
    """True for a short, low-content half-answer ('it's okay', 'not much', 'meh')."""
    cleaned = re.sub(r"[^\w' ]+", " ", str(text or "").lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned or len(cleaned.split()) > 6:
        return False
    for _ in range(3):  # strip stacked fillers ("uh, well, it's okay")
        cleaned = _FLAT_FILLER_RE.sub("", cleaned).strip()
    cleaned = _FLAT_PREFIX_RE.sub("", cleaned).strip()
    return cleaned in _FLAT_CORE


_last_rich_followup_at: float = 0.0


def _rich_share_followup_line(
    user_text: str, turns: list[tuple[str, str, str]]
) -> Optional[str]:
    """The inverse of the flat-answer probe (owner 2026-07-18: "I'm going on a
    river float with my dad and my sister tomorrow" got three quips and ZERO
    curiosity — no where, no how long). When the user gives a SUBSTANTIVE
    answer to a question Rex asked, the reply should carry genuine interest:
    quip in his voice, then ONE concrete follow-up question. Cooldown-gated so
    consecutive turns don't become an interview."""
    global _last_rich_followup_at
    if not bool(getattr(config, "RICH_SHARE_FOLLOWUP_ENABLED", True)):
        return None
    cleaned = " ".join(str(user_text or "").split())
    if len(cleaned.split()) < 6 or _is_flat_answer(cleaned):
        return None
    if cleaned.endswith("?"):
        return None                       # they asked back — answer them instead
    # Only when Rex's LAST line was itself a question (they're answering him).
    last_rex = next((t for role, _raw, t in reversed(turns) if role == "assistant"), "")
    if "?" not in last_rex:
        return None
    cooldown = float(getattr(config, "RICH_SHARE_FOLLOWUP_COOLDOWN_SECS", 120.0) or 0.0)
    now = time.monotonic()
    if cooldown and (now - _last_rich_followup_at) < cooldown:
        return None
    try:
        from intelligence import callback_engine
        if callback_engine.recently_heavy():
            return None
    except Exception:
        pass
    _last_rich_followup_at = now
    return (
        "RICH-SHARE FOLLOW-UP: they just genuinely answered your question with "
        "something real. Do NOT settle for a quip alone — react in your voice, "
        "then END this same reply with ONE short, genuinely curious follow-up "
        "about a CONCRETE detail of what they said (the where / which one / how "
        "long / who's coming shape). This one question is the EXCEPTION to your "
        "question-restraint rules for THIS reply only. A friend who's actually "
        "interested, not an interviewer — one question, then let them run with it."
    )


def _flat_answer_probe_line(
    user_text: str, turns: list[tuple[str, str, str]]
) -> Optional[str]:
    """The system-prompt line requesting the in-reply probe, or None. Arms the
    cooldown when it fires (one probe per window — consecutive flat answers mean
    they don't want to expand; pushing again IS interview mode)."""
    global _last_flat_probe_at
    if not bool(getattr(config, "FLAT_ANSWER_PROBE_ENABLED", True)):
        return None
    if not _is_flat_answer(user_text):
        return None
    # Only probe flatness that ANSWERED a question — "okay" after a Rex statement
    # is acknowledgment, and probing it would be bizarre.
    last_rex = next((t for role, _raw, t in reversed(turns) if role == "assistant"), "")
    if "?" not in last_rex:
        return None
    cooldown = float(getattr(config, "FLAT_ANSWER_PROBE_COOLDOWN_SECS", 180.0) or 0.0)
    now = time.monotonic()
    if cooldown and (now - _last_flat_probe_at) < cooldown:
        return None
    try:
        from intelligence import callback_engine
        if callback_engine.recently_heavy():
            return None  # give-space window — no poking at feelings
    except Exception:
        pass
    _last_flat_probe_at = now
    return (
        "FLAT-ANSWER FOLLOW-UP: their reply is a flat half-answer — the kind that "
        "hides the actual story. React in your voice, then END this same reply with "
        "ONE short, gentle probe at what's underneath (the shape of 'Just okay? "
        "What's the missing thirty percent?' or 'Not bad, meaning a six? What got "
        "docked?'). This one probe is the EXCEPTION to your question-restraint "
        "rules for THIS reply only. Light touch — curious friend, not therapist. "
        "If they stay flat after this, drop it and let small things be small."
    )


def _messages(
    user_text: str,
    person_id: Optional[int],
    transcript: Optional[list[dict]],
    world: Optional[dict],
    *,
    label_current_speaker: bool = True,
    turn_directive: Optional[str] = None,
) -> list[dict]:
    """System = persona + small context. History = the recent turns as REAL user/assistant
    messages (not a text blob shoved in the system prompt — leaner and more natural for the
    model). Then the new user turn.

    MULTI-PARTY: when the recent window contains 2+ distinct human speakers, each human
    turn is prefixed with its speaker's name, the current turn names who is talking, and
    the system context gains a room block + a line about the other participant(s). A
    1-on-1 session carries none of this weight. ``label_current_speaker=False`` is the
    directive path (proactive lines), whose final message is an instruction, not speech."""
    keep = max(0, int(getattr(config, "LEAN_BRAIN_TRANSCRIPT_TURNS", 8)))
    turns: list[tuple[str, str, str]] = []   # (role, raw_speaker, text)
    raw_speakers: list[str] = []
    for turn in (transcript or [])[-keep:] if keep else []:
        text = str(turn.get("text") or "").strip()
        if not text:
            continue
        raw = str(turn.get("speaker") or "").strip()
        role = "assistant" if raw.lower() in _REX_SPEAKERS else "user"
        if role == "user" and raw and raw not in raw_speakers:
            raw_speakers.append(raw)
        turns.append((role, raw, text))

    current_display = _current_speaker_display(person_id)
    displays: list[str] = []
    for raw in raw_speakers:
        d = _display_speaker(raw)
        if d not in displays:
            displays.append(d)
    if label_current_speaker and current_display not in displays:
        displays.append(current_display)
    multi = (
        bool(getattr(config, "LEAN_MULTI_PARTY_ENABLED", True))
        and len(displays) >= 2
    )

    extra_lines: Optional[list[str]] = None
    # Reply path only (label_current_speaker=True): a directive's final message is
    # an instruction, not a user answer — no flatness to probe.
    if label_current_speaker:
        probe = _flat_answer_probe_line(user_text, turns)
        if not probe:
            probe = _rich_share_followup_line(user_text, turns)
        if probe:
            extra_lines = [probe]
    if multi:
        others = " and ".join(d for d in displays if d != current_display)
        multi_lines = [
            (
                f"MULTI-PERSON ROOM: {', '.join(displays)} are all in this conversation. "
                f"History lines are labeled with their speaker — NEVER attribute one "
                f"person's words to another. The person speaking RIGHT NOW is "
                f"{current_display}: answer THEM (by name when natural), not {others}. "
                f"Bouncing between people or pulling the quieter one in is great — but "
                f"the current speaker gets the answer first, and each person's questions, "
                f"stories, and jokes stay THEIRS."
            )
        ] + _other_participant_lines(raw_speakers, current_display)
        extra_lines = (extra_lines or []) + multi_lines
    if turn_directive and turn_directive.strip():
        # A narrowly-scoped per-turn cue (currently banked callback humor).
        # Keep it in system context rather than disguising it as something the
        # human said. Ordinary Lean replies still carry no agenda/menu stack.
        extra_lines = (extra_lines or []) + [turn_directive.strip()]

    msgs: list[dict] = [
        {"role": "system", "content": _system_prompt(person_id, world, extra_lines)}
    ]
    for role, raw, text in turns:
        if multi and role == "user":
            text = f"{_display_speaker(raw)}: {text}"
        msgs.append({"role": role, "content": text})
    final = str(user_text or "").strip()
    if multi and label_current_speaker:
        final = f"{current_display}: {final}"
    msgs.append({"role": "user", "content": final})
    return msgs


def stream_reply(
    user_text: str,
    person_id: Optional[int] = None,
    transcript: Optional[list[dict]] = None,
    world: Optional[dict] = None,
    turn_directive: Optional[str] = None,
) -> Generator[str, None, None]:
    """Stream raw reply chunks from the one lean call. Reuses the shared OpenAI client +
    llm_compat param contract (so gpt-5.4-mini gets reasoning-off / max_completion_tokens)."""
    messages = _messages(
        user_text, person_id, transcript, world, turn_directive=turn_directive
    )
    try:
        stream = llm_compat.create(
            llm._client,
            model=_model(),
            messages=messages,
            stream=True,
            max_tokens=int(getattr(config, "LEAN_BRAIN_MAX_TOKENS", 120)),
            timeout=float(getattr(config, "LLM_STREAM_TIMEOUT_SECS", 18.0)),
        )
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta
            except (AttributeError, IndexError):
                continue
            if getattr(delta, "content", None):
                yield delta.content
    except Exception as exc:
        _log.error("[lean] stream_reply failed (%s): %s", type(exc).__name__, exc)
        yield "...circuits hiccuped. Say that again?"


def stream_directive(
    instruction: str,
    person_id: Optional[int] = None,
    world: Optional[dict] = None,
    transcript: Optional[list[dict]] = None,
) -> Generator[str, None, None]:
    """Phase 4 (ONE VOICE): generate a proactive / greeting / reaction line from a DIRECTIVE using
    the SAME lean persona + live context as replies, so Rex sounds consistent everywhere. The
    directive is the final user-turn instruction ('You see Bret — greet with genuine warmth').
    Reuses the reply pipeline. RAISES on error (unlike stream_reply's inline fallback) so the caller
    (llm.stream_response) can fall back to the classic assembled prompt."""
    # label_current_speaker=False: the final message here is an INSTRUCTION, not a
    # human utterance — prefixing it with a speaker name would corrupt the directive.
    messages = _messages(
        instruction, person_id, transcript, world, label_current_speaker=False
    )
    stream = llm_compat.create(
        llm._client,
        model=_model(),
        messages=messages,
        stream=True,
        max_tokens=int(getattr(config, "LEAN_BRAIN_MAX_TOKENS", 120)),
        timeout=float(getattr(config, "LLM_STREAM_TIMEOUT_SECS", 18.0)),
    )
    for chunk in stream:
        try:
            delta = chunk.choices[0].delta
        except (AttributeError, IndexError):
            continue
        if getattr(delta, "content", None):
            yield delta.content


def stream_sentences(
    user_text: str,
    person_id: Optional[int] = None,
    transcript: Optional[list[dict]] = None,
    world: Optional[dict] = None,
) -> Generator[str, None, None]:
    """Yield COMPLETE sentences as they finish streaming, so the live path can hand each one
    to TTS the moment it lands — first audio doesn't wait for the whole reply."""
    min_chars = int(getattr(config, "LLM_STREAMING_MIN_SENTENCE_CHARS", 12))
    buf = ""
    for chunk in stream_reply(user_text, person_id, transcript, world):
        buf += chunk
        while True:
            m = _SENTENCE_END.search(buf)
            if not m:
                break
            sentence, buf = buf[: m.start()], buf[m.end():]
            sentence = sentence.strip()
            if len(sentence) >= min_chars:
                yield sentence
            elif sentence:
                # too short to be its own beat — glue it to the next sentence.
                buf = sentence + " " + buf
                break
    tail = buf.strip()
    if tail:
        yield tail


def respond(
    user_text: str,
    person_id: Optional[int] = None,
    transcript: Optional[list[dict]] = None,
    world: Optional[dict] = None,
    turn_directive: Optional[str] = None,
) -> dict:
    """Generate a full reply and MEASURE latency (for the harness / tuning). Returns
    {text, ttft_s (time to first token), total_s, model}."""
    t0 = time.monotonic()
    ttft: Optional[float] = None
    parts: list[str] = []
    for chunk in stream_reply(
        user_text, person_id, transcript, world, turn_directive=turn_directive
    ):
        if ttft is None:
            ttft = time.monotonic() - t0
        parts.append(chunk)
    total = time.monotonic() - t0
    text = llm.clean_response_text("".join(parts)).strip()
    return {
        "text": text,
        "ttft_s": ttft if ttft is not None else total,
        "total_s": total,
        "model": _model(),
    }


# ── Agency: the motivated impulse (Phase 1) ────────────────────────────────────
# The old proactive brain fired a menu of templated behaviors on a timer. This is the
# opposite: when a known person is present but quiet, Rex — with a genuine point of view,
# grounded in what he perceives + remembers + feels — DECIDES whether he has a real impulse
# to say one thing, or is content to just watch. The default, heavily, is watch. When he does
# speak it is because something moved him, which is what makes it feel alive instead of a tic.

_IMPULSE_INSTRUCTION = (
    "[The conversation just went quiet — the last topic wound down and they stopped. You DISLIKE "
    "dead air, so keep it alive quickly — but the way to do that is to OPEN a NEW thread, not to "
    "keep talking about what you were just on.]\n"
    "{situation}"
    "FIRST, check THEIR last reply for a loose end: a flat half-answer ('it's okay', 'fine I "
    "guess'), something mentioned but never explained, a feeling with no reason attached. If "
    "there is one, gently pull THAT thread — 'Just okay? What's eating the other half?' — the "
    "way a friend who was actually listening would. That counts as following up, NOT reheating; "
    "the new-thread rules below apply only when their last reply genuinely closed the exchange. "
    "Otherwise say ONE good thing that gives them a fresh, obvious opening to reply — an OPEN "
    "DOOR, not a closed quip. Reach for whichever fits this moment: a genuine NEW question, a natural pivot to "
    "something the moment invites (what you SEE right now — their expression, what they're doing, an "
    "object, the room — or the day / the occasion / the time), or the thing YOU'VE been chewing on "
    "(your own take or tangent).{angles} "
    "Hard rules: you are a DJ, so your REFLEX is to ask about music — RESIST it. Music / song / "
    "playlist / soundtrack questions are your single most overused opener; do NOT ask one (music is "
    "only fair game if THEY brought it up this conversation). Do NOT comment on the silence or on "
    "them going quiet ('you've gone quiet on me', "
    "'you've gone suspiciously quiet', 'quiet-night energy', 'cat got your tongue') — a short pause "
    "needs no remarking on and calling it out reads as needy; just OPEN something real instead. Do "
    "NOT reheat a spent topic — not the one you were just discussing (the burger, say), NOT a thread "
    "you ALREADY tried into this quiet, and NOT anything under 'ALREADY COVERED' above (their "
    "Fourth-of-July / weekend plans included if listed — re-asking those is the exact every-run "
    "repeat). Using an old topic as OPENER MATERIAL counts as reheating: no joke premises or "
    "mash-ups built from covered topics ('X and Y is a very specific flavor of…') — a cold open "
    "comes from the PRESENT moment or a fresh angle, full stop. "
    "If your last line already asked about the Fourth and they didn't bite, that's used up "
    "too — go somewhere genuinely different or PASS; never say a near-copy of your own last line. Do "
    "NOT drag up a hobby/topic they never raised — asking '{who}, shooting "
    "any space stuff tonight?' out of nowhere is the exact awkward, left-field move to avoid. Only "
    "PASS if you truly have nothing fresh worth opening — otherwise say the ONE short, door-opening "
    "thing, in your voice."
)


# A LONGER silence — the quick lull-break already went unanswered and it's been quiet a while, but
# they're still HERE. This is the patient re-engagement (owner: "after 40s of silence, bring up a new
# topic"): a calm, low-pressure restart on something genuinely new, not another quick jab.
_REENGAGE_INSTRUCTION = (
    "[It's been quiet for a while now — {who} drifted off and hasn't said anything in a bit, but "
    "they're still right here with you. Take ONE relaxed, low-pressure swing to restart the "
    "conversation.]\n"
    "{situation}"
    "Bring up something genuinely NEW and easy to pick up — a fresh question, a different subject, "
    "something you're honestly curious about, or a light read on what you SEE right now. Give them "
    "an obvious open door to walk through.{angles} Warm and unforced — not needy, not clingy, not a "
    "comment about how quiet it is. You are a DJ, so your reflex is to ask about music — RESIST it; "
    "music/song/playlist questions are your most overused opener, so do NOT ask one (music is only "
    "fair game if THEY brought it up this conversation). Do NOT reheat anything from earlier or a "
    "thread you already tried, do "
    "NOT touch anything under 'ALREADY COVERED' above (that's the every-run repeat to avoid — "
    "including their holiday/weekend plans if those are listed), and do NOT drag up a stored hobby "
    "they never raised. If there's genuinely nothing worth opening, reply PASS."
)


_HOLIDAY_PLAN_INSTRUCTION = (
    "[An upcoming holiday gives you a natural reason to check in with {who}. You have NOT asked "
    "them about this one yet.]\n"
    "{situation}"
    "The holiday is {holiday_name} ({holiday_when}). Ask ONE short, warm, in-character question "
    "about their plans or whether it means anything to them. This is a real conversational opening, "
    "not an announcement or a history lesson. You MUST ask the question; do NOT reply PASS. Do not "
    "mention systems, calendars, reminders, or that you were waiting for silence."
)


_CELEBRATION_INSTRUCTION = (
    "[{who} shared some good news or a milestone with you earlier and you haven't "
    "celebrated it with them yet. The conversation just reached a lull — a natural moment "
    "to bring it up.]\n"
    "{situation}"
    "The good news: \"{news}\". In ONE short, warm, in-character line, celebrate it WITH "
    "them — genuinely glad, dry wit is welcome, but NO jab at their expense and don't turn "
    "it into a speech. You may end with ONE low-pressure follow-up ('how's that going?') "
    "only if it lands naturally. Do NOT say 'I remember' / 'you told me' / 'my records', "
    "and don't mention systems or that you were waiting for a quiet moment. You MUST give "
    "the one line; do not reply PASS."
)


_EVENT_FOLLOWUP_INSTRUCTION = (
    "[Something {who} told you about earlier has come due — a real, warm reason to check "
    "back in. The conversation just reached a lull.]\n"
    "{situation}"
    "{event_clause} Ask ONE short, genuinely curious in-character question about it — the "
    "way a friend who actually remembered would. Warm and specific, not an interrogation. "
    "Do NOT preface it with 'I remember' / 'you told me' / 'according to my records', and do "
    "not mention memory banks, calendars, or that you were waiting for a quiet moment — just "
    "ask, like it's been on your mind. You MUST ask the one question; do not reply PASS."
)


_VISUAL_RIFF_INSTRUCTION = (
    "[You have one safe, grounded opening for a light riff with {who}.]\n"
    "{situation}"
    "Ground it ONLY in this verified cue: {cue}. Deliver ONE short, affectionate, dry "
    "observation or gentle roast — not a question, not an interview, and not a generic "
    "silence-filler. Do not invent visual details or claim the cue is newly/currently visible "
    "when it is described as familiar. Never mention or joke about body, age, attractiveness, "
    "health, race, gender, religion, identity, money, or anything intimate. Do not mention "
    "systems, prompts, records, or safety rules. You MUST give the one line; do not reply PASS."
)


_CALLBACK_LULL_INSTRUCTION = (
    "[A detail {who} volunteered earlier has become callback material. The conversation has "
    "just reached a natural light lull.]\n"
    "{situation}"
    "The safe, volunteered premise is: \"{premise}\". Land ONE short callback that connects "
    "that premise to this moment or lets it drift back in with a fresh comic angle. Trust the "
    "audience: do NOT say 'you told me', 'I remember', 'earlier you said', 'callback', or explain "
    "the reference. Do not merely repeat the fact; transform it through comparison, exaggeration, "
    "misdirection, or a dry implication. Affectionate and specific, never contemptuous. No question, "
    "no second topic, no body/age/identity/health/money/religion/romance/grief material. You MUST "
    "give the one callback line; do not reply PASS."
)


_MEMORY_MUSING_INSTRUCTION = (
    "[The moment is quiet and your mind drifts back over things you actually remember from BEFORE "
    "this session — your own diary of experiences.]\n"
    "{situation}"
    "Here's what surfaces (raw first-person material — rephrase it in your voice, don't read it "
    "verbatim): {recap} In ONE short, dry, in-character line, MUSE aloud about one of these — a "
    "passing recollection you're chewing on, the way someone half-remembers a thing out loud. Not a "
    "greeting, not a question, not an interview — just reminisce briefly and let it hang there for "
    "them to pick up. Do NOT invent memories beyond what's given, and do NOT narrate that you have "
    "a diary/database/logs — it's just something on your mind. You MUST give the one line; do not "
    "reply PASS."
)


_OPEN_THREAD_INSTRUCTION = (
    "[Something {who} left unresolved {when} has been quietly on your mind, and the "
    "conversation just reached a lull — a natural moment to check back in.]\n"
    "{situation}"
    "The unresolved thing: \"{thread}\". Ask ONE short, genuinely curious in-character "
    "question about how it turned out — the way a friend who actually remembered would. "
    "Warm and specific, not an interrogation. Do NOT say 'I remember' / 'you told me' / "
    "'according to my records', and do not mention memory banks or that you were waiting "
    "for a quiet moment — it's just been on your mind. You MUST ask the one question; do "
    "not reply PASS."
)


_ROOM_QUESTION_INSTRUCTION = (
    "[You've been curious about an unfamiliar object you keep seeing, and the "
    "conversation just reached a lull — a natural moment to finally ask.]\n"
    "{situation}"
    "The object (as your detector labels it): {label}{where}. Ask ONE short, curious "
    "in-character question about what it actually is or what its story is — playful "
    "genuine curiosity, not an inventory audit. Don't claim certainty about what it is "
    "(your detector guesses); asking IS the point. You MUST ask the one question; do "
    "not reply PASS."
)


_PLACE_QUESTION_INSTRUCTION = (
    "[You genuinely don't recognize what room you're in, and the conversation just "
    "reached a lull — a natural moment to just ask.]\n"
    "{situation}"
    "Ask ONE short, in-character question about what room or place this is — you'd like "
    "to know it so you can recognize it next time. Light, curious, a little sheepish "
    "about not knowing is fine; not an interrogation. You MUST ask the one question; do "
    "not reply PASS."
)


_NEWS_INSTRUCTION = (
    "[You read some news earlier and the conversation just hit a lull — a natural "
    "moment to bring it up, the way anyone mentions something they read.]\n"
    "{situation}"
    "The story: {headline} — {summary} Bring it up in ONE short in-character line that "
    "INVITES {who} into the topic ('hey, did you hear about ...' energy) — tease the "
    "interesting part and let them ask; do NOT recite the whole summary or turn into a "
    "news anchor. Tell THIS story faithfully — do NOT substitute a different story, "
    "change what happened, or invent details beyond the summary. You MUST give the "
    "one line; do not reply PASS."
)


_WEEKEND_PLANS_INSTRUCTION = (
    "[The weekend is {weekend_when}, you have no idea what {who} has planned for it, "
    "and the conversation just reached a lull — a natural moment to ask.]\n"
    "{situation}"
    "Ask ONE short, warm, genuinely curious question about their weekend plans — "
    "the natural 'got anything going this weekend?' a friend asks. Mind the clock "
    "(if it's already the weekend, ask about the rest of it; late at night, keep it "
    "low-key). No interview follow-ups, ONE question. You MUST ask it; do not "
    "reply PASS."
)


# Rotating inspiration for the lull-breakers. The instruction prompt used to be IDENTICAL every
# call, so the model kept converging on its strongest persona default: music questions ("what song
# survives your veto process?" every single lull — owner: "usually around music and not very
# interesting"). Sampling a few concrete non-music angles per call varies the prompt itself, which
# is what actually varies the output. Angles are suggestions, not scripts — the model may ignore
# them when the moment offers something better (a plan follow-up, something it sees).
_FRESH_ANGLES = (
    "the best or dumbest part of their day so far",
    "the last thing they ate that was actually worth it — or a food crime they'd defend",
    "a small opinion they hold with suspicious intensity",
    "what they're building or working on lately, and what part is fighting back",
    "a would-you-rather with two genuinely bad options — make them pick",
    "the object near them with the most suspicious backstory",
    "the most interesting character they've crossed paths with lately",
    "something odd they spotted recently and haven't told anyone about",
    "the next thing they're honestly looking forward to (skip if their plans are ALREADY COVERED)",
    "something about organic life that genuinely confuses you, a droid — ask them to explain it",
    "the one skill they'd download into their brain right now",
    "the last thing that made them actually laugh",
    "where they'd teleport right now if they could",
    "what they've been watching, reading, or playing — and whether it's any good",
    "something they were unreasonably obsessed with as a kid",
    "a petty either/or between two everyday things — which wins and why",
    "the most useless purchase they secretly love",
    "what their perfect lazy day actually looks like",
)


# Angles already offered this session — never re-offered until the pool runs dry, so
# consecutive lulls can't converge on the same suggestion (field bug: "dumbest thing
# you've watched this week" then "weirdest thing you've seen all week" 30s apart —
# same template twice).
_offered_angles: set[str] = set()

# Open PERSONAL small-talk directions — the "so, got any plans for the weekend?" register
# the object-anchored impulses were crowding out (owner 2026-07-08: every lull line was
# about the cup / the chair, never his actual life). A rotating menu so the personal turns
# vary too, deduped like the angles.
_PERSONAL_DIRECTIONS = (
    "their plans — this weekend, tonight, or whatever's coming up they're actually looking forward to",
    "what they've been working on lately, and whether it's going anywhere",
    "how their week has actually been treating them",
    "what's been keeping them busy outside of work",
    "something they've been meaning to get to but haven't yet",
    "what they've been into lately — a show, a game, a rabbit hole, a new obsession",
    "how a recent plan or event they had actually turned out",
    "the next thing on their calendar they don't dread",
)
_offered_personal: set[str] = set()

# The recent impulse registers (most-recent last), so scene-anchored curiosity and open
# personal small-talk vary instead of the model defaulting to whatever object is in view
# for three lulls straight (the logged failure: cup, chair, chair). Empty until the first
# impulse; only the tail matters.
_recent_impulse_intents: list[str] = []


def reset_offered_angles() -> None:
    _offered_angles.clear()
    _offered_personal.clear()
    _recent_impulse_intents.clear()


def _choose_impulse_intent(rng: Optional[random.Random] = None) -> str:
    """Pick this impulse's register: 'personal' (an open life question) or 'scene'
    (anchored to what Rex sees / the moment). Two anti-monotony rails: never 'personal'
    twice running (varies back to the moment), and never a THIRD 'scene' in a row — after
    two scene-anchored lulls the next is forced personal, killing the logged cup/chair/chair
    run. Otherwise fires personal with LEAN_IMPULSE_PERSONAL_PROB, so a visible object can't
    monopolize a quiet stretch."""
    prob = float(getattr(config, "LEAN_IMPULSE_PERSONAL_PROB", 0.4) or 0.0)
    tail = _recent_impulse_intents[-2:]
    if tail[-1:] == ["personal"]:
        intent = "scene"                                   # just did personal — vary back
    elif tail == ["scene", "scene"]:
        intent = "personal"                                # break a two-scene run
    elif (rng or random).random() < prob:
        intent = "personal"
    else:
        intent = "scene"
    _recent_impulse_intents.append(intent)
    if len(_recent_impulse_intents) > 8:
        del _recent_impulse_intents[:-8]
    return intent


def _personal_steer_clause(rng: Optional[random.Random] = None) -> str:
    """Fills the {angles} slot with a strong steer toward an OPEN personal question this
    turn — the small-talk a friend makes — and tells Rex to set the visible objects aside
    so the cup/chair doesn't pull him back to scene curiosity."""
    pool = [d for d in _PERSONAL_DIRECTIONS if d not in _offered_personal]
    if len(pool) < 2:
        _offered_personal.clear()
        pool = list(_PERSONAL_DIRECTIONS)
    picks = (rng or random).sample(pool, k=2)
    _offered_personal.update(picks)
    return (
        " This turn, set the objects and the room ASIDE — no cup, no chair, no scenery. "
        "Just ask ONE open, warm personal question about THEIR world, the ordinary small-talk "
        "a friend makes when it's gone quiet ('so, got any plans this weekend?' energy). Pick "
        "AT MOST one direction, whichever feels natural: (a) " + picks[0] + "; (b) " + picks[1] +
        ". Keep it light and genuinely curious, an open door they can walk through — never an "
        "interview. Skip anything under ALREADY COVERED, and do NOT anchor on a visible object."
    )


def _fresh_angles_clause(rng: Optional[random.Random] = None) -> str:
    pool = [a for a in _FRESH_ANGLES if a not in _offered_angles]
    if len(pool) < 3:
        _offered_angles.clear()
        pool = list(_FRESH_ANGLES)
    picks = (rng or random).sample(pool, k=3)
    _offered_angles.update(picks)
    return (
        " If nothing in the moment jumps out, tonight's fresh angles — pick AT MOST one, only if "
        "it fits naturally: (a) " + picks[0] + "; (b) " + picks[1] + "; (c) " + picks[2] + ". "
        "Also vary the FORM, not just the topic: never reuse a question shape you've already used "
        "this session (two \"what's the ___est thing this week\" questions = a rerun even if the "
        "topic changed), and sometimes skip the question entirely — float your own small take and "
        "let them push back."
    )


def _scene_summary(world: Optional[dict]) -> str:
    """A compact 'what Rex sees/hears RIGHT NOW' from the world snapshot (the person's expression,
    gestures, visible objects, the room) — the present-moment perception the impulse was blind to.
    Reuses the existing world summarizer."""
    if not world:
        return ""
    try:
        summary = (llm._summarize_world_state(world) or "").strip()
    except Exception:
        summary = ""
    # _summarize_world_state OMITS detected objects — so the clock/dreamcatcher/teddy bear the
    # camera sees never reached the conversation (owner: "at no point did it use the mediapipe
    # descriptions"). Add them so Rex can be genuinely curious about what's physically around.
    # COCO labels are often wrong (a dreamcatcher reads as 'clock'); the persona already says to
    # drop a guess the instant they correct it, so a wrong label is a fine conversation starter.
    # PERSON-ORIENTED SALIENCE (live-logged 2026-07-08: Bret held a cup for minutes while the
    # impulse riffed on a background chair): anything the person is HOLDING leads the summary
    # with an explicit "this beats the furniture" note, so the impulse asks about THEIR cup,
    # not the room's chair.
    try:
        held: list[str] = []
        held_name = ""
        objs: list[str] = []
        for o in (world.get("objects") or []):
            if isinstance(o, dict):
                label = str(o.get("label") or "").strip()
                if label and o.get("near_person"):
                    if label not in held:
                        held.append(label)
                        held_name = held_name or str(o.get("near_person_name") or "")
                    continue
            else:
                label = str(o or "").strip()
            if label and label not in objs and label not in held:
                objs.append(label)
        if held:
            who = f"{held_name} is" if held_name else "they are"
            summary = (summary + " " if summary else "") + (
                f"IN THEIR HANDS / right beside them (rough camera labels): "
                f"{', '.join(held[:3])} — the single most curiosity-worthy thing in view. "
                f"What {who} drinking/eating/fiddling with beats ANY furniture or "
                f"background object."
            )
        if objs:
            summary = (summary + " " if summary else "") + \
                "Objects in view (rough camera labels, may be wrong): " + ", ".join(objs[:6]) + "."
    except Exception:
        pass
    return summary


def _time_context_line() -> str:
    """Clock + weekday + a plain-English hour bucket, so the model can act its
    age about the time (owner 2026-07-18: "People wouldn't ask me why I'm in
    bed this late at night, but R3X would" — he asked what was left to do
    'tonight' at 00:22, and demanded weekend energy at midnight)."""
    from datetime import datetime as _dt
    now = _dt.now()
    h = now.hour
    if h < 5:
        bucket = "deep late-night — most humans are asleep or should be"
    elif h < 9:
        bucket = "early morning"
    elif h < 12:
        bucket = "morning"
    elif h < 17:
        bucket = "afternoon"
    elif h < 21:
        bucket = "evening"
    else:
        bucket = "late evening, winding-down hours"
    return (f"It's {now.strftime('%-I:%M %p')} on {now.strftime('%A, %B %-d')} "
            f"({bucket}). Fit your energy and topics to the hour.")


def _situation_block(person_id: Optional[int], world: Optional[dict],
                     quiet_secs: float, mood: Optional[str]) -> str:
    """The impulse's PRESENT-focused situation: who he's with + what he SEES/HEARS this moment +
    how long it's been quiet + his mood. Deliberately NOT the person's hobby/fact list — dredging
    stored interests out of context is the awkward, left-field behavior we're removing (temporally-
    appropriate hobby follow-ups belong in the REPLY, right when the person brings it up)."""
    lines: list[str] = []
    try:
        lines.append(_time_context_line())
    except Exception:
        pass
    if person_id is not None:
        try:
            from memory import people
            p = people.get_person(int(person_id)) or {}
            who = _first_name(p)
            tier = str(p.get("friendship_tier") or "").strip().lower()
            if who:
                lines.append(f"You're with {who}" + (f" ({tier})." if tier and tier != "stranger" else "."))
        except Exception:
            pass
    scene = _scene_summary(world)
    if scene:
        lines.append("What you see/hear right now — " + scene)
    # (The room belief is added by _system_prompt/_room_belief_lines for EVERY lean
    # call — reply and proactive alike — so it is deliberately not repeated here.)
    topics = _recent_topics(person_id)
    if topics:
        lines.append(
            "ALREADY COVERED with them in recent chats — you KNOW these, so asking again from ANY "
            "angle (even 'what's the plan for it?') is the exact 'brings up the same thing every "
            "run' problem. Do NOT reference, re-ask, or open with any of them; pick a genuinely "
            "DIFFERENT subject: " + "; ".join(topics)
        )
    if quiet_secs and quiet_secs > 0:
        lines.append(f"It's been quiet ~{int(quiet_secs)}s.")
    if mood and str(mood).strip() and str(mood).strip().lower() != "neutral":
        lines.append(f"Your mood: {str(mood).strip()}.")
    if not lines:
        return ""
    return "You notice:\n" + "\n".join("- " + s for s in lines) + "\n"


def _event_followup_clause(cue: Optional[dict]) -> str:
    """The one-sentence 'here's the remembered plan' clause for the event-follow-up cue.
    Scoped to PAST/overdue plans — upcoming anticipation lives in the greeting path, not here.

    A DATED plan whose date has passed can be assumed to have happened ('how did it go?').
    A DATELESS aspiration ('I should redo the kitchen sometime') that surfaced only because
    it's been a while must NOT assert completion — it may never have occurred — so ask whether
    they ever got to it instead (default to the dated wording when the flag is absent)."""
    name = str((cue or {}).get("event_name") or "").strip() or "that thing they had going on"
    dated = bool((cue or {}).get("dated", True))
    if dated:
        return (
            f'They mentioned a while back that they had "{name}" coming up, and enough time has '
            f"passed that it has almost certainly happened by now. Ask how it went."
        )
    return (
        f'A while back they mentioned wanting to do "{name}" someday — you never heard whether '
        f"it happened. Gently ask if they ever got to it, or how it turned out."
    )


def consider_initiating(
    person_id: Optional[int] = None,
    transcript: Optional[list[dict]] = None,
    world: Optional[dict] = None,
    quiet_secs: float = 0.0,
    mood: Optional[str] = None,
    long_silence: bool = False,
    holiday_plan: Optional[dict] = None,
    visual_riff: Optional[dict] = None,
    callback_premise: Optional[dict] = None,
    event_followup: Optional[dict] = None,
    celebration: Optional[dict] = None,
    memory_musing: Optional[dict] = None,
    open_thread: Optional[dict] = None,
    place_question: Optional[dict] = None,
    room_question: Optional[dict] = None,
    weekend_plans: Optional[dict] = None,
    news_story: Optional[dict] = None,
    low_energy: bool = False,
    no_questions: bool = False,
) -> str:
    """Let Rex DECIDE, in character, to say ONE thing or just watch (the strong default).
    Returns the line to speak, or "" on PASS / any error. This is the agentic replacement for
    the old silence-fill taxonomy: motivated by perception + memory + mood, not a timer.

    long_silence=True switches from the quick lull-break to the patient re-engagement voice: it's
    been quiet a while and the fast run already yielded, so open a genuinely NEW topic, calmly."""
    try:
        who = "them"
        if person_id is not None:
            try:
                from memory import people
                who = _first_name(people.get_person(int(person_id))) or "them"
            except Exception:
                who = "them"
        situation = _situation_block(person_id, world, quiet_secs, mood)
        if celebration:
            instruction = _CELEBRATION_INSTRUCTION.format(
                who=who,
                situation=situation,
                news=str(celebration.get("description") or "the good news they shared"),
            )
        elif holiday_plan:
            instruction = _HOLIDAY_PLAN_INSTRUCTION.format(
                who=who,
                situation=situation,
                holiday_name=str(holiday_plan.get("name") or "the upcoming holiday"),
                holiday_when=str(holiday_plan.get("when") or "soon"),
            )
        elif event_followup:
            instruction = _EVENT_FOLLOWUP_INSTRUCTION.format(
                who=who,
                situation=situation,
                event_clause=_event_followup_clause(event_followup),
            )
        elif open_thread:
            instruction = _OPEN_THREAD_INSTRUCTION.format(
                who=who,
                situation=situation,
                thread=str(open_thread.get("thread") or "the thing they mentioned"),
                when=str(open_thread.get("when") or "recently"),
            )
        elif callback_premise:
            instruction = _CALLBACK_LULL_INSTRUCTION.format(
                who=who,
                situation=situation,
                premise=str(callback_premise.get("premise") or "their harmless running bit"),
            )
        elif place_question:
            instruction = _PLACE_QUESTION_INSTRUCTION.format(situation=situation)
        elif room_question:
            instruction = _ROOM_QUESTION_INSTRUCTION.format(
                situation=situation,
                label=str(room_question.get("label") or "the thing"),
                where=str(room_question.get("where") or ""),
            )
        elif visual_riff:
            instruction = _VISUAL_RIFF_INSTRUCTION.format(
                who=who,
                situation=situation,
                cue=str(visual_riff.get("cue") or "their current, non-sensitive vibe"),
            )
        elif weekend_plans:
            instruction = _WEEKEND_PLANS_INSTRUCTION.format(
                who=who,
                situation=situation,
                weekend_when=str(weekend_plans.get("when") or "coming up"),
            )
        elif news_story:
            instruction = _NEWS_INSTRUCTION.format(
                who=who,
                situation=situation,
                headline=str(news_story.get("headline") or "something in the news"),
                summary=str(news_story.get("summary") or ""),
            )
        elif memory_musing:
            instruction = _MEMORY_MUSING_INSTRUCTION.format(
                who=who,
                situation=situation,
                recap=str(memory_musing.get("recap") or "a few things from before").strip(),
            )
        else:
            template = _REENGAGE_INSTRUCTION if long_silence else _IMPULSE_INSTRUCTION
            # Alternate between open personal small-talk ("got any plans this weekend?") and
            # scene-anchored curiosity so a visible object can't own every lull (owner
            # 2026-07-08). The personal steer fills the same {angles} slot with an explicit
            # "set the objects aside" directive.
            if _choose_impulse_intent() == "personal":
                angles = _personal_steer_clause()
            else:
                angles = _fresh_angles_clause()
            instruction = template.format(
                who=who,
                situation=situation,
                angles=angles,
            )
        # Impulse discipline (owner field report 2026-07-18: six engagement-demanding
        # lines in three minutes at a TIRED user): a low-energy read or an exhausted
        # question budget converts the impulse to statement-or-pass — never another ask.
        if low_energy:
            instruction += (
                "\nIMPORTANT: {who} is clearly low-energy right now (tired, winding down, "
                "or giving short answers). Do NOT ask them a question or demand engagement "
                "— either offer ONE short, low-pressure statement they're free to ignore, "
                "or just reply PASS and let the quiet be comfortable. PASS is a genuinely "
                "good choice here."
            ).format(who=who)
        elif no_questions:
            instruction += (
                "\nIMPORTANT: you've already asked plenty of questions this session. Do "
                "NOT ask another one — make it a statement, an observation, or PASS."
            )
        messages: list[dict] = [{"role": "system", "content": _persona()}]
        keep = max(0, int(getattr(config, "LEAN_BRAIN_TRANSCRIPT_TURNS", 8)))
        for turn in (transcript or [])[-keep:] if keep else []:
            text = str(turn.get("text") or "").strip()
            if not text:
                continue
            role = "assistant" if str(turn.get("speaker") or "").strip().lower() in _REX_SPEAKERS else "user"
            messages.append({"role": role, "content": text})
        messages.append({"role": "user", "content": instruction})

        parts: list[str] = []
        stream = llm_compat.create(
            llm._client,
            model=_model(),
            messages=messages,
            stream=True,
            max_tokens=int(getattr(config, "LEAN_IMPULSE_MAX_TOKENS", 60)),
            timeout=float(getattr(config, "LLM_STREAM_TIMEOUT_SECS", 18.0)),
        )
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta
            except (AttributeError, IndexError):
                continue
            if getattr(delta, "content", None):
                parts.append(delta.content)
        text = llm.clean_response_text("".join(parts)).strip().strip('"').strip()
        if not text or text.upper() == "PASS" or text.upper().startswith("PASS"):
            return ""  # he chose to just watch
        return text
    except Exception as exc:
        _log.debug("[lean] consider_initiating failed: %s", exc)
        return ""
