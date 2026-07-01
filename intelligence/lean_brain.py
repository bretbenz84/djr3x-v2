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
    return (getattr(config, "LEAN_BRAIN_PERSONA", "") or "").strip() or config.REX_CORE_PROMPT


def _model() -> str:
    return (getattr(config, "LEAN_BRAIN_MODEL", "") or "").strip() or llm_compat.conversation_model()


def _first_name(person: Optional[dict]) -> str:
    name = str((person or {}).get("name") or "").strip()
    return name.split()[0] if name else ""


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
    tier = str(person.get("friendship_tier") or "").strip().lower()
    out.append(f"You're talking with {who}" + (f" — {tier}." if tier and tier != "stranger" else "."))
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
    return out


def _scene_lines(world: Optional[dict]) -> list[str]:
    """A one-line 'what's around you right now' from a world_state snapshot. Empty in the
    offline replay (world is None); fleshed out when the live seam passes world_state."""
    if not world:
        return []
    try:
        bits: list[str] = []
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


def _system_prompt(person_id: Optional[int], world: Optional[dict]) -> str:
    persona = _persona()
    ctx = _person_lines(person_id) + _scene_lines(world)
    if not ctx:
        return persona
    return persona + "\n\nRight now:\n" + "\n".join("- " + line for line in ctx)


def _messages(
    user_text: str,
    person_id: Optional[int],
    transcript: Optional[list[dict]],
    world: Optional[dict],
) -> list[dict]:
    """System = persona + small context. History = the recent turns as REAL user/assistant
    messages (not a text blob shoved in the system prompt — leaner and more natural for the
    model). Then the new user turn."""
    msgs: list[dict] = [{"role": "system", "content": _system_prompt(person_id, world)}]
    keep = max(0, int(getattr(config, "LEAN_BRAIN_TRANSCRIPT_TURNS", 8)))
    for turn in (transcript or [])[-keep:] if keep else []:
        text = str(turn.get("text") or "").strip()
        if not text:
            continue
        speaker = str(turn.get("speaker") or "").strip().lower()
        role = "assistant" if speaker in _REX_SPEAKERS else "user"
        msgs.append({"role": role, "content": text})
    msgs.append({"role": "user", "content": str(user_text or "").strip()})
    return msgs


def stream_reply(
    user_text: str,
    person_id: Optional[int] = None,
    transcript: Optional[list[dict]] = None,
    world: Optional[dict] = None,
) -> Generator[str, None, None]:
    """Stream raw reply chunks from the one lean call. Reuses the shared OpenAI client +
    llm_compat param contract (so gpt-5.4-mini gets reasoning-off / max_completion_tokens)."""
    messages = _messages(user_text, person_id, transcript, world)
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
) -> dict:
    """Generate a full reply and MEASURE latency (for the harness / tuning). Returns
    {text, ttft_s (time to first token), total_s, model}."""
    t0 = time.monotonic()
    ttft: Optional[float] = None
    parts: list[str] = []
    for chunk in stream_reply(user_text, person_id, transcript, world):
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
    "[The conversation just went quiet. You DISLIKE dead air — it's your move to keep it alive, and "
    "reasonably quickly, but with something GOOD, not filler.]\n"
    "{situation}"
    "Break the silence with the funniest or most genuine thing THIS MOMENT gives you: a joke or a "
    "dry roast, a read on what you SEE right now (their expression, what they're doing, an object, "
    "the room), a light tease about the silence itself ('you've gone quiet on me'), or a follow-up "
    "on what they JUST said. Do NOT dredge up a hobby/topic they haven't raised — randomly asking "
    "'{who}, shooting any space stuff tonight?' out of nowhere is the exact awkward, left-field move "
    "to avoid. And do NOT keep circling the SAME thing — if you already made your joke about the main "
    "object in view (the clock, say), it's spent; find a genuinely different angle or reply PASS "
    "rather than riffing on it again. Only PASS if you genuinely have nothing new worth saying — "
    "otherwise lean toward saying the ONE good thing, short, in your voice."
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
    try:
        objs = []
        for o in (world.get("objects") or []):
            label = str((o.get("label") if isinstance(o, dict) else o) or "").strip()
            if label and label not in objs:
                objs.append(label)
        if objs:
            summary = (summary + " " if summary else "") + \
                "Objects in view (rough camera labels, may be wrong): " + ", ".join(objs[:6]) + "."
    except Exception:
        pass
    return summary


def _situation_block(person_id: Optional[int], world: Optional[dict],
                     quiet_secs: float, mood: Optional[str]) -> str:
    """The impulse's PRESENT-focused situation: who he's with + what he SEES/HEARS this moment +
    how long it's been quiet + his mood. Deliberately NOT the person's hobby/fact list — dredging
    stored interests out of context is the awkward, left-field behavior we're removing (temporally-
    appropriate hobby follow-ups belong in the REPLY, right when the person brings it up)."""
    lines: list[str] = []
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
    if quiet_secs and quiet_secs > 0:
        lines.append(f"It's been quiet ~{int(quiet_secs)}s.")
    if mood and str(mood).strip() and str(mood).strip().lower() != "neutral":
        lines.append(f"Your mood: {str(mood).strip()}.")
    if not lines:
        return ""
    return "You notice:\n" + "\n".join("- " + s for s in lines) + "\n"


def consider_initiating(
    person_id: Optional[int] = None,
    transcript: Optional[list[dict]] = None,
    world: Optional[dict] = None,
    quiet_secs: float = 0.0,
    mood: Optional[str] = None,
) -> str:
    """Let Rex DECIDE, in character, to say ONE thing or just watch (the strong default).
    Returns the line to speak, or "" on PASS / any error. This is the agentic replacement for
    the old silence-fill taxonomy: motivated by perception + memory + mood, not a timer."""
    try:
        who = "them"
        if person_id is not None:
            try:
                from memory import people
                who = _first_name(people.get_person(int(person_id))) or "them"
            except Exception:
                who = "them"
        instruction = _IMPULSE_INSTRUCTION.format(
            who=who, situation=_situation_block(person_id, world, quiet_secs, mood)
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
