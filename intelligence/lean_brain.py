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
    return (getattr(config, "LEAN_BRAIN_PERSONA", "") or "").strip() or config.REX_CORE_PROMPT


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
    messages = _messages(instruction, person_id, transcript, world)
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
    "[The conversation just went quiet — the last topic wound down and they stopped. You DISLIKE "
    "dead air, so keep it alive quickly — but the way to do that is to OPEN a NEW thread, not to "
    "keep talking about what you were just on.]\n"
    "{situation}"
    "Say ONE good thing that gives them a fresh, obvious opening to reply — an OPEN DOOR, not a "
    "closed quip. Reach for whichever fits this moment: a genuine NEW question, a natural pivot to "
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
    "repeat). If your last line already asked about the Fourth and they didn't bite, that's used up "
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


def reset_offered_angles() -> None:
    _offered_angles.clear()


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


def consider_initiating(
    person_id: Optional[int] = None,
    transcript: Optional[list[dict]] = None,
    world: Optional[dict] = None,
    quiet_secs: float = 0.0,
    mood: Optional[str] = None,
    long_silence: bool = False,
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
        template = _REENGAGE_INSTRUCTION if long_silence else _IMPULSE_INSTRUCTION
        instruction = template.format(
            who=who,
            situation=_situation_block(person_id, world, quiet_secs, mood),
            angles=_fresh_angles_clause(),
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
