"""
features/games.py — Interactive game management for DJ-R3X.

Manages one active game at a time. All games are interruptible via stop_game().

Supported games:
    "i_spy"            — Rex picks an object from the camera frame; player guesses
    "20_questions"     — Player thinks of something; Rex guesses it via yes/no questions
    "trivia"           — Rex runs a short scored trivia round
    "jeopardy"         — A verbal Jeopardy-style board with real clue data
    "word_association" — Rapid back-and-forth word chain

Public API:
    can_play(game_name)                  → (bool, str | None)  # repeat-limit gate
    start_game(game_name, person_id=None) → str    # opening line for Rex to speak
    start_trivia(person_id=None)         → str    # convenience wrapper for trivia
    handle_input(text, person_id=None, audio_array=None) → str
                                            # Rex's response to player input
    stop_game(person_id=None)            → str    # graceful closing line
    is_active()                          → bool
    current_game()                       → str | None
"""

import json
import copy
import logging
import random
import re
import sys
import threading
import time
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rapidfuzz import fuzz

import config

from features import twentyq_kb
from vision.image_utils import encode_jpeg_base64

_log = logging.getLogger(__name__)

# ── Module state ──────────────────────────────────────────────────────────────

_lock = threading.Lock()
_active_game: Optional[str] = None     # normalized game key
_game_state: dict = {}                  # game-specific state

# Play history for repeat-limit tracking: game_key → list of monotonic timestamps
_game_play_log: dict[str, list[float]] = {}

_GAME_DISPLAY_NAMES: dict[str, str] = {
    "i_spy":            "I Spy",
    "20_questions":     "20 Questions",
    "trivia":           "Trivia",
    "jeopardy":         "Jeopardy",
    "word_association": "Word Association",
}

# ── Game name aliases ─────────────────────────────────────────────────────────

_GAME_ALIASES: dict[str, str] = {
    "i spy":            "i_spy",
    "eye spy":          "i_spy",
    "ispy":             "i_spy",
    "i_spy":            "i_spy",
    "spy":              "i_spy",
    "20 questions":     "20_questions",
    "twenty questions": "20_questions",
    "twenty questions game": "20_questions",
    "20 questions game": "20_questions",
    "20questions":      "20_questions",
    "20_questions":     "20_questions",
    "trivia":           "trivia",
    "jeopardy":         "jeopardy",
    "jeopardy!":        "jeopardy",
    "word association": "word_association",
    "word_association": "word_association",
    "word assoc":       "word_association",
    "association":      "word_association",
}


def _normalize_game(name: str) -> Optional[str]:
    clean = name.strip().lower()
    if clean.startswith("jeopardy"):
        return "jeopardy"
    if clean != "trivia" and (clean.endswith(" trivia") or clean.endswith(" trivia game")):
        return "trivia"
    return _GAME_ALIASES.get(clean)


def available_game_names() -> list[str]:
    """Return display names for games that have registered handlers."""
    return [
        _GAME_DISPLAY_NAMES[key]
        for key in _GAME_HANDLERS
        if key in _GAME_DISPLAY_NAMES
    ]


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _get_client():
    try:
        import apikeys
        from openai import OpenAI
        from intelligence import connectivity as _connectivity
        return _connectivity.guard_client(OpenAI(api_key=apikeys.OPENAI_API_KEY), "games")
    except ImportError as exc:
        raise ImportError(f"games requires apikeys and openai: {exc}") from exc


def _quick_call(prompt: str, temperature: float = 0.7, max_tokens: int = 100) -> str:
    """Lightweight GPT-4o-mini call for game logic decisions (not Rex's voice)."""
    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        _log.error("[games] _quick_call failed: %s", exc)
        return ""


def _smart_call(prompt: str, *, temperature: float = 0.3, max_tokens: int = 700,
                reasoning_effort: str = "low") -> str:
    """Higher-reasoning call for decisions where a stronger model clearly pays off — namely
    20 Questions' question selection and final guess (deciding the right yes/no question and
    making the leap to the answer). Routes through the conversation model (gpt-5.4-mini) via
    llm_compat so GPT-5-family params (max_completion_tokens, reasoning_effort, temperature
    gating) are handled in one place. `max_tokens` is intentionally generous: reasoning models
    spend tokens thinking before they emit, so a tight cap can starve the visible answer.
    Falls back to the cheap GPT-4o-mini path on any error."""
    try:
        from intelligence import llm_compat
        client = _get_client()
        resp = llm_compat.create(
            client,
            model=config.LLM_CONVERSATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            # Reasoning models reject a non-default temperature once reasoning is engaged, so
            # we omit it here; `temperature` only flavors the GPT-4o-mini fallback below.
            temperature=None,
            reasoning_effort=reasoning_effort,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text or _quick_call(prompt, temperature=temperature, max_tokens=120)
    except Exception as exc:
        _log.warning("[games] _smart_call fell back to _quick_call: %s", exc)
        return _quick_call(prompt, temperature=temperature, max_tokens=120)


def _rex_respond(game_context: str, person_id: Optional[int] = None) -> str:
    """Generate a Rex in-character game response using the full LLM pipeline."""
    try:
        from intelligence import llm
        return llm.get_response(game_context, person_id)
    except Exception as exc:
        _log.error("[games] _rex_respond failed: %s", exc)
        return "...my circuits are experiencing some turbulence. Stand by."


def _parse_json(text: str):
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    if stripped.startswith("```"):
        nl = stripped.find("\n")
        if nl != -1:
            stripped = stripped[nl + 1:]
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    for oc, cc in [("{", "}"), ("[", "]")]:
        s = text.strip().find(oc)
        e = text.strip().rfind(cc)
        if s != -1 and e > s:
            try:
                return json.loads(text.strip()[s: e + 1])
            except json.JSONDecodeError:
                pass
    return None


def _get_agreeability() -> int:
    """Read agreeability from the DB; fall back to config default."""
    try:
        from memory import database as db
        rows = db.fetchall(
            "SELECT value FROM personality_settings WHERE parameter = 'agreeability'"
        )
        if rows:
            return int(rows[0]["value"])
    except Exception:
        pass
    return config.PERSONALITY_DEFAULTS.get("agreeability", 35)


def _body_beat(name: str, **context) -> None:
    """Trigger a mapped embodied reaction without making game logic wait."""
    try:
        from intelligence import performance_plan
        from sequences import animations
        beat = (
            performance_plan.body_beat_for_event(name, **context)
            or performance_plan.canonical_body_beat(name)
            or name
        )
        animations.play_body_beat(beat)
    except Exception as exc:
        _log.debug("[games] body beat %s skipped: %s", name, exc)


# ── I Spy game ────────────────────────────────────────────────────────────────

_ISPY_MAX_GUESSES = 5
_ISPY_SCAN_VIEWS = ("left", "center", "right")


def _ispy_scan_room() -> list[tuple[str, object]]:
    """Physically look around the room before picking a target — the bit of
    showmanship the physical droid is supposed to do (owner call 2026-07-07),
    which also widens the object pool beyond whatever the head happened to be
    pointing at. Sweeps left → center → right with a directed-gaze hold (so the
    face-tracking loop doesn't fight the sweep), capturing a frame at each pose,
    then recenters. Returns [(view, frame), ...]. Without servos (or with the
    scan disabled) it degrades to one frame from the current gaze."""
    try:
        from vision import camera
    except ImportError:
        _log.warning("[games] Camera unavailable for I Spy")
        return []

    scan_possible = bool(getattr(config, "ISPY_SCAN_ENABLED", True))
    if scan_possible:
        try:
            from hardware import servos
            scan_possible = servos.connected()
        except Exception:
            scan_possible = False
    if not scan_possible:
        frame = camera.get_frame()
        return [("center", frame)] if frame is not None else []

    views: list[tuple[str, object]] = []
    settle = float(getattr(config, "ISPY_SCAN_SETTLE_SECS", 0.35))
    try:
        from intelligence import consciousness
        from sequences import animations
    except Exception as exc:
        _log.debug("[games] I Spy scan primitives unavailable: %s", exc)
        frame = camera.get_frame()
        return [("center", frame)] if frame is not None else []

    for view in _ISPY_SCAN_VIEWS:
        try:
            consciousness.hold_directed_gaze(view, secs=6.0)
            animations.directed_look_pose(view)
            frame = camera.capture_current_gaze(settle_secs=settle)
            if frame is not None:
                views.append((view, frame))
        except Exception as exc:
            _log.debug("[games] I Spy scan pose %r failed: %s", view, exc)
    try:
        consciousness.clear_directed_gaze_hold()
        animations.directed_look_pose("center")
    except Exception:
        pass
    if not views:
        frame = camera.get_frame()
        return [("center", frame)] if frame is not None else []
    return views


def _ispy_pick_target(views: list[tuple[str, object]]) -> Optional[dict]:
    """Ask GPT-4o to pick one I Spy object across the captured views.
    Returns {"object": "red chair", "clue": "red", "view": "left"} or None."""
    if not views:
        return None

    content: list[dict] = []
    detail = config.VISION_DETAIL.get("active_conversation", "auto")
    labeled_views: list[str] = []
    for view, frame in views:
        b64 = encode_jpeg_base64(frame, quality=85)
        if b64 is None:
            continue
        labeled_views.append(view)
        content.append({"type": "text", "text": f"View looking {view}:"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": detail},
        })
    if not labeled_views:
        return None

    view_names = ", ".join(f'"{v}"' for v in labeled_views)
    content.append({
        "type": "text",
        "text": (
            "These are views of the same room from a robot looking around. Pick ONE "
            "specific object visible in ONE of the views that would work well for the "
            "game 'I Spy'. Choose something clearly visible and guessable. Not a person.\n"
            "Return a JSON object with exactly three keys:\n"
            '  "object": the full name of the object (e.g. "red chair", "blue mug"),\n'
            '  "clue": a single descriptive word for "I spy something ___" '
            '(e.g. "red", "shiny", "round"),\n'
            f'  "view": which view it is in — one of {view_names}.\n'
            "Return ONLY the JSON object — no preamble, no markdown."
        ),
    })

    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=config.VISION_MODEL,
            messages=[{"role": "user", "content": content}],
            max_tokens=150,
        )
        data = _parse_json(resp.choices[0].message.content.strip())
        if isinstance(data, dict) and data.get("object") and data.get("clue"):
            view = str(data.get("view") or "").strip().lower()
            data["view"] = view if view in labeled_views else labeled_views[0]
            return data
    except Exception as exc:
        _log.error("[games] I Spy vision call failed: %s", exc)
    return None


def _ispy_get_target() -> Optional[dict]:
    """Look around the room, then pick an I Spy object from what was seen."""
    return _ispy_pick_target(_ispy_scan_room())


def _ispy_announce_scan() -> None:
    """Speak a quick canned 'casing the room' line WITHOUT blocking, so the head
    sweep and the vision call run under it instead of as dead air."""
    lines = list(getattr(config, "ISPY_SCAN_LINES", [])) or [
        "Hold on — casing the room for a worthy target.",
    ]
    try:
        from audio import speech_queue
        speech_queue.enqueue(random.choice(lines), "curious", priority=1, tag="ispy:scan")
    except Exception as exc:
        _log.debug("[games] I Spy scan announce failed: %s", exc)


def _ispy_glance_at_target() -> None:
    """Look toward the view the chosen object was seen in — the reveal beat."""
    view = str(_game_state.get("target_view") or "")
    if view not in _ISPY_SCAN_VIEWS:
        return
    try:
        from sequences import animations
        animations.directed_look_pose(view)
    except Exception as exc:
        _log.debug("[games] I Spy reveal glance failed: %s", exc)


def _ispy_start(person_id: Optional[int]) -> str:
    _ispy_announce_scan()
    target = _ispy_get_target()
    if target is None:
        return _rex_respond(
            "[GAME: I Spy] Rex tried to start I Spy but the camera isn't cooperating. "
            "Apologize in character — something about his photoreceptors — and suggest "
            "playing 20 Questions or Word Association instead.",
            person_id,
        )

    _game_state.update({
        "target_object": target["object"],
        "clue": target["clue"],
        "target_view": target.get("view", "center"),
        "guess_count": 0,
    })
    _body_beat("dramatic_visor_peek")

    return _rex_respond(
        f"[GAME: I Spy — START] Rex just looked around the room and picked a secret "
        f"object. Give Rex's opening line for I Spy. "
        f"Say \"I spy with my little eye, something that is {target['clue']}\" "
        f"and add a brief Rex-style flourish. Players have {_ISPY_MAX_GUESSES} guesses. "
        f"Do not reveal the object or where it is.",
        person_id,
    )


def _ispy_handle(text: str, person_id: Optional[int]) -> tuple[str, bool]:
    target = _game_state.get("target_object", "")
    clue = _game_state.get("clue", "")
    guess_count = _game_state.get("guess_count", 0) + 1
    _game_state["guess_count"] = guess_count

    target_lower = target.lower()
    user_lower = text.strip().lower()

    is_correct = (
        fuzz.ratio(user_lower, target_lower) >= 70
        or fuzz.partial_ratio(user_lower, target_lower) >= 70
        or target_lower in user_lower
    )

    if is_correct:
        # Look right at the object as he concedes it — the reveal beat.
        _ispy_glance_at_target()
        _body_beat("tiny_victory_dance")
        _game_state.clear()
        return (
            _rex_respond(
                f"[GAME: I Spy — CORRECT] Player correctly guessed \"{target}\" "
                f"on guess #{guess_count}. Rex celebrates briefly — punchy, in character, "
                f"maybe slightly annoyed they got it.",
                person_id,
            ),
            True,
        )

    if guess_count >= _ISPY_MAX_GUESSES:
        _ispy_glance_at_target()
        _body_beat("suspicious_glance")
        _game_state.clear()
        return (
            _rex_respond(
                f"[GAME: I Spy — GAME OVER] Player used all {_ISPY_MAX_GUESSES} guesses. "
                f"The object was \"{target}\". Rex reveals it, lightly roasts the player "
                f"for not getting it.",
                person_id,
            ),
            True,
        )

    give_hint = guess_count >= 2
    _body_beat("suspicious_glance")
    return (
        _rex_respond(
            f"[GAME: I Spy — WRONG GUESS #{guess_count}/{_ISPY_MAX_GUESSES}] "
            f"The secret object is \"{target}\" (clue: \"{clue}\"). "
            f"Player guessed: \"{text.strip()}\". Wrong. "
            + (
                "Give a subtle additional hint without revealing the object directly."
                if give_hint else
                "Tell them wrong — snappy."
            )
            + " Rex stays in character.",
            person_id,
        ),
        False,
    )


def _ispy_stop(person_id: Optional[int]) -> str:
    target = _game_state.get("target_object", "something")
    _ispy_glance_at_target()
    _game_state.clear()
    return _rex_respond(
        f"[GAME: I Spy — STOPPED] Game ended early. The object was \"{target}\". "
        f"Rex delivers a brief in-character close.",
        person_id,
    )


# ── 20 Questions game ─────────────────────────────────────────────────────────

_20Q_MAX_QUESTIONS = 20      # hard cap; Rex must commit to a guess by here
_20Q_MAX_GUESSES = 3         # how many final guesses Rex gets before conceding
_20Q_SPINE_TURNS = 12        # spine (dataset + authored tier-2 branches) leads for up to N questions
_20Q_EARLY_GUESS_FLOOR = 7   # from this many questions on, check between spine questions whether
                             # the shortlist has collapsed and strike early instead of grinding on
_20Q_GUESS_SHORTLIST_MAX = 2  # a mid-game guess needs the shortlist down to this many candidates

# Answer keyword sets for parsing the player's yes/no replies (LLM fallback for the rest).
_20Q_YES = {
    "yes", "yeah", "yep", "yup", "yah", "ya", "sure", "correct", "right", "true",
    "definitely", "absolutely", "affirmative", "mhm", "mmhm", "uh huh", "uhuh",
    "of course", "exactly", "indeed", "totally", "for sure", "you got it",
    "that's right", "thats right", "you're right", "youre right", "got it", "bingo",
    "yes it is", "yep that's it", "you nailed it",
}
_20Q_NO = {
    "no", "nope", "nah", "naw", "negative", "false", "wrong", "incorrect", "nuh uh",
    "not really", "i don't think so", "i dont think so", "definitely not", "no way",
    "not at all", "afraid not", "nope sorry", "not even close", "not quite",
}
_20Q_MAYBE = {
    "sometimes", "maybe", "kind of", "kinda", "sort of", "sorta", "partly", "partially",
    "occasionally", "depends", "it depends", "in a way", "more or less", "somewhat",
}
_20Q_UNKNOWN = {
    "i don't know", "i dont know", "dont know", "don't know", "not sure", "no idea",
    "unsure", "dunno", "hard to say", "can't say", "cant say", "who knows",
}
# Whisper near-homophones of "no" that show up in far-field game answers (the mic hears the
# clipped "no" as "now"/"know"). Corrected to "no" ONLY inside the 20Q yes/no classifier —
# never in general conversation, where the user may legitimately say "now".
_20Q_NO_MISHEARS = {"now", "know", "know.", "gnaw", "knnow", "noh"}


def _norm_q(q: str) -> str:
    """Normalize a question for de-dup / spine matching."""
    q = re.sub(r"\s+", " ", (q or "").strip().lower())
    if q and not q.endswith("?"):
        q += "?"
    return q


def _20q_classify_answer(text: str) -> str:
    """Map the player's reply to one of: yes / no / sometimes / unknown.
    Keyword-first (deterministic, test-friendly); LLM fallback for anything ambiguous."""
    t = re.sub(r"\s+", " ", (text or "").strip().lower()).strip(" .!?")
    if not t:
        return "unknown"
    # Correct the common far-field mishear of "no" before anything else (game-scoped).
    if t in _20Q_NO_MISHEARS:
        return "no"
    # Whole-phrase membership first (so "no idea" -> unknown, not "no").
    for label, vocab in (("unknown", _20Q_UNKNOWN), ("yes", _20Q_YES),
                         ("no", _20Q_NO), ("sometimes", _20Q_MAYBE)):
        if t in vocab:
            return label
    # Multi-word cue anywhere in the reply (check unknown/no before the rest).
    for label, vocab in (("unknown", _20Q_UNKNOWN), ("no", _20Q_NO),
                         ("sometimes", _20Q_MAYBE), ("yes", _20Q_YES)):
        if any(phrase in t for phrase in vocab if " " in phrase):
            return label
    # Leading token.
    first = t.split()[0]
    if first in {"yes", "yeah", "yep", "yup", "ya", "yah", "correct", "right", "true",
                 "sure", "definitely", "absolutely", "exactly", "bingo", "totally"}:
        return "yes"
    if first in {"no", "nope", "nah", "naw", "negative", "false", "wrong", "incorrect"}:
        return "no"
    if first in {"sometimes", "maybe", "kinda", "sorta", "occasionally", "somewhat"}:
        return "sometimes"
    # Ambiguous — ask the model.
    raw = _quick_call(
        f'In a yes/no game, the player answered: "{text.strip()}". '
        f"Classify their answer as ONLY one word: yes, no, sometimes, or unknown.",
        temperature=0, max_tokens=4,
    ).strip().lower()
    # Match whole words, not substrings — "no" is a substring of "unknown", so a substring
    # check would silently turn every "unknown" verdict into "no".
    raw_tokens = set(re.findall(r"[a-z]+", raw))
    for label in ("yes", "no", "sometimes", "unknown"):
        if label in raw_tokens:
            return label
    return "unknown"


def _20q_question_is_redundant(question: str, asked: list) -> bool:
    """True when a proposed question is a repeat or a blatant rephrase of one already asked.
    Conservative on purpose (near-duplicate only, token_set_ratio >= 95): the live 2026-07-07
    game burned Q17 on 'is it primarily used for carrying items?' one turn after 'is it mainly
    a container for carrying or storing things?'. A legit NARROWING ('is it a stringed
    instrument?' after 'is it a musical instrument?') must NOT be blocked, so the threshold
    stays high and the prompt carries the semantic no-re-ask rule."""
    nq = _norm_q(question)
    if nq in set(asked):
        return True
    return any(fuzz.token_set_ratio(nq, a) >= 95 for a in asked)


def _20q_fallback_question(asked: list) -> str:
    """A deterministic, decent splitter when the LLM's question was unusable: first an
    applicable unasked spine/tier-2 question, then a generic bank."""
    entry = twentyq_kb.next_spine_question(
        _game_state.get("concept_answers", {}), set(asked))
    if entry is not None:
        return entry["question"]
    for q in (
        "Is it something you'd find in most homes?",
        "Is it used every day?",
        "Is it bigger than a microwave?",
        "Is it mainly for entertainment?",
        "Would most people own one?",
    ):
        if not _20q_question_is_redundant(q, asked):
            return q
    return "Is it something you could buy in a regular store?"


def _20q_guess_gate_ok(q_count: int, remaining: int, candidates: list) -> bool:
    """Deterministic commit gate: Rex only spends a guess when the evidence supports it.
    The live 2026-07-07 game burned a guess on 'wallet' at Q14 off a broad shortlist —
    a guess now needs the shortlist collapsed to the front-runners (the model's own
    confidence signal), a late-game near-collapse, or no road left."""
    if remaining <= 2:
        return True
    if len(candidates) <= _20Q_GUESS_SHORTLIST_MAX and q_count >= 5:
        return True
    if q_count >= 12 and len(candidates) <= 3:
        return True
    return False


def _20q_decide(qa_log: list, asked: list, q_count: int,
                candidates: list, guesses: list) -> dict:
    """Candidate-tracking turn engine — plays like the classic 20Q toy. Each turn it keeps a
    running SHORTLIST of the answers still consistent with EVERY clue, then either asks the
    yes/no question that best SPLITS that shortlist or commits to the front-runner. Tracking
    candidates (instead of improvising one question at a time) is what stops the tunnel-vision
    that handed away easy wins. A deterministic gate (_20q_guess_gate_ok) vets any guess and a
    near-duplicate check vets any question, so a wobbly model turn degrades to a proven
    splitter instead of a wasted move. Returns:
        {"candidates": [...], "action": "ask"|"guess", "question": str, "subject": str}
    """
    hard = [e for e in qa_log if e.get("a") in ("yes", "no")]
    soft = [e for e in qa_log if e.get("a") not in ("yes", "no")]
    facts = "\n".join(
        f"  - {e.get('q','')} -> {str(e.get('a','')).upper()}" for e in hard) or "  (none yet)"
    hints = "\n".join(
        f"  - {e.get('q','')} -> {e.get('a','')}" for e in soft) or "  (none)"
    asked_list = "\n".join(f"  - {q}" for q in asked) or "  (none)"
    menu = twentyq_kb.spine_menu(_game_state.get("concept_answers", {}), set(asked), limit=4)
    menu_line = (
        "Proven unasked splitters you may use if they fit better than your own question: "
        + "; ".join(menu) if menu else ""
    )
    remaining = _20Q_MAX_QUESTIONS - q_count
    prior = ", ".join(candidates) if candidates else "(none yet — build it now)"
    avoid = ", ".join(guesses) if guesses else "none"
    raw = _smart_call(
        "You are an expert 20 Questions guesser that plays like the classic 20Q toy: you keep a "
        "running SHORTLIST of candidate answers and narrow it down with every clue.\n\n"
        f"ESTABLISHED FACTS (hard yes/no answers — every candidate MUST satisfy all of them):\n"
        f"{facts}\n\n"
        f"Soft hints (maybe/sometimes/unknown — suggestive, not binding):\n{hints}\n\n"
        f"Questions already asked — NEVER repeat one, NEVER rephrase one, and NEVER ask "
        f"anything whose answer the facts above already determine:\n{asked_list}\n\n"
        f"Your shortlist from last turn (reconsider it, don't just trust it): {prior}\n"
        f"Already guessed wrong — never propose these again: {avoid}\n"
        f"You have asked {q_count} questions; {remaining} remain.\n\n"
        "Do BOTH steps:\n"
        "1. SHORTLIST — REBUILD it FRESH from ALL the facts (do not just trim last turn's "
        "list). Rules:\n"
        "   - Weigh ALL clues together; never fixate on the latest answer.\n"
        "   - Strongly prefer COMMON, everyday answers. Keep the list diverse across CATEGORIES "
        "you haven't ruled out (until size is known, include both hand-held things like a guitar "
        "or book AND large ones), but NEVER drift into rare, exotic, regional, or hyper-specific "
        "variants — if 'pizza' fits, do not list 'tostada' or 'huarache'.\n"
        "   - If your last guesses were wrong, your whole CATEGORY may be wrong — seriously "
        "consider a different KIND of thing rather than narrowing deeper into the same one.\n"
        "   - List 3-6 candidates while still exploring; once you are genuinely confident, "
        "list ONLY the 1-2 real contenders — a short list is your signal to strike.\n"
        "2. MOVE —\n"
        "   - GUESS only when your shortlist is down to 1-2 real contenders, or 2 or fewer "
        "questions remain. Guess the COMMON general name ('bicycle', not 'commuter bicycle'; "
        "'pizza', not 'tostada').\n"
        "   - Otherwise ASK one yes/no question whose answer best SPLITS your shortlist (about "
        "half your candidates would say yes, half no). Split the BROADEST open dimension first — "
        "purpose or location beats material or brand. Once a category is already confirmed by "
        "the facts, your question must DISCRIMINATE among your shortlisted candidates — a "
        "concrete feature some have and others lack (moving parts, uses refills, has a screen, "
        "worn on the feet) — NOT another 'is it used for X?' subcategory probe; fishing through "
        "subcategories one at a time is how you lose. "
        f"{menu_line}\n"
        'Return ONLY JSON: {"candidates":["c1","c2",...],"action":"ask" or "guess",'
        '"question":"<yes/no question if asking>","subject":"<candidate to guess if guessing>"}',
        temperature=0.3, max_tokens=900, reasoning_effort="medium",
    )
    data = _parse_json(raw)
    new_candidates = list(candidates)
    if isinstance(data, dict):
        cand = data.get("candidates")
        if isinstance(cand, list):
            cleaned = [str(c).strip() for c in cand if str(c).strip()]
            if cleaned:
                new_candidates = cleaned[:6]
        action = str(data.get("action", "")).lower()
        if action == "guess" and data.get("subject"):
            if _20q_guess_gate_ok(q_count, remaining, new_candidates):
                return {"candidates": new_candidates, "action": "guess",
                        "subject": str(data["subject"]).strip()}
            # Premature stab — convert to a question so the evidence catches up first.
            question = str(data.get("question") or "").strip()
            if not question or _20q_question_is_redundant(question, asked):
                question = _20q_fallback_question(asked)
            _log.info("[20q] guess gate held fire (q=%d, shortlist=%d) — asking instead",
                      q_count, len(new_candidates))
            return {"candidates": new_candidates, "action": "ask", "question": question}
        if action == "ask" and data.get("question"):
            question = str(data["question"]).strip()
            if not _20q_question_is_redundant(question, asked):
                return {"candidates": new_candidates, "action": "ask", "question": question}
            _log.info("[20q] rejected near-duplicate question %r — using fallback", question)
            return {"candidates": new_candidates, "action": "ask",
                    "question": _20q_fallback_question(asked)}
    # Fallback: out of road -> guess the front-runner; otherwise a proven splitter.
    if remaining <= 1:
        return {"candidates": new_candidates, "action": "guess",
                "subject": new_candidates[0] if new_candidates else ""}
    return {"candidates": new_candidates, "action": "ask",
            "question": _20q_fallback_question(asked)}


def _20q_best_guess(qa_log: list, candidates: list, guesses: list) -> str:
    """Rex's single best guess. Prefer the tracked shortlist's front-runner (skipping anything
    already guessed wrong); fall back to deriving one from the full Q&A."""
    rejected = {g.strip().lower() for g in guesses}
    for c in candidates or []:
        if c.strip() and c.strip().lower() not in rejected:
            return c.strip()
    transcript = "\n".join(
        f"  Q: {e.get('q','')}  ->  A: {e.get('a','')}" for e in qa_log) or "  (none)"
    raw = _smart_call(
        "You are playing 20 Questions as the guesser. Based on the answers, name your single "
        "best guess — the most likely COMMON, everyday thing consistent with EVERY answer "
        "(its general name, e.g. 'bicycle' or 'pizza', not a rare sub-type or regional variant)."
        "\n\n"
        f"Questions and answers:\n{transcript}\n\n"
        f"Already guessed wrong (do not repeat): {', '.join(guesses) or 'none'}\n"
        "Return ONLY the name of your guess — no punctuation, no explanation.",
        temperature=0.5, max_tokens=400, reasoning_effort="medium",
    ).strip().strip(".!?\"'")
    return raw.split("\n")[0].strip() if raw else ""


def _20q_make_guess(person_id: Optional[int], suggested: str = "",
                    forced: bool = False) -> tuple[str, bool]:
    """Commit to a guess: name it, ground it against the real-subject vocabulary, and wait
    for the player to confirm. Mutates state into the 'guessing' phase. Never terminal here —
    the player's next reply decides win/lose."""
    guesses = _game_state.get("guesses", [])
    subject = (suggested or "").strip()
    if not subject:
        subject = _20q_best_guess(_game_state.get("qa_log", []),
                                  _game_state.get("candidates", []), guesses)
    final = twentyq_kb.snap_guess(subject) or subject or "a protocol droid"
    guesses.append(final)
    _game_state["guesses"] = guesses
    _game_state["pending_guess"] = final
    _game_state["phase"] = "guessing"
    _body_beat("thinking_tilt")
    q_count = _game_state.get("question_count", 0)
    pressure = " He's out of questions, so this is his final answer." if forced else ""
    return (
        _rex_respond(
            f"[GAME: 20 Questions — REX GUESSES] After {q_count} questions, Rex is ready to "
            f"guess what the player is thinking of. Rex guesses it is: \"{final}\". Rex phrases "
            f"it as a confident guess (\"Is it ___?\") with swagger and asks the player to "
            f"confirm whether he nailed it.{pressure} One or two sentences.",
            person_id,
        ),
        False,
    )


def _20q_ask_next(person_id: Optional[int]) -> tuple[str, bool]:
    """Ask Rex's next question (dataset spine first, then LLM), or pivot to a guess.
    Mutates state. Returns (response, done)."""
    q_count = _game_state.get("question_count", 0)
    asked = _game_state.get("asked", [])
    concept_answers = _game_state.get("concept_answers", {})

    if q_count >= _20Q_MAX_QUESTIONS:
        return _20q_make_guess(person_id, forced=True)

    # Opening/mid-game: the proven discriminator spine (dataset + authored tier-2 branches)
    # leads for strong, non-redundant narrowing.
    entry = None
    if q_count < _20Q_SPINE_TURNS:
        entry = twentyq_kb.next_spine_question(concept_answers, set(asked))

    # Confident early exit: once enough evidence is in, check between spine questions whether
    # the shortlist has already collapsed — an obvious answer (phone, dog, pizza) should be
    # guessed at Q8, not ground through the rest of the spine. Only worth a model call right
    # after a YES (a yes collapses the space; a string of no's never does), and the guess
    # still has to pass the deterministic gate inside _20q_decide, so a broad shortlist
    # keeps asking.
    qa_log_so_far = _game_state.get("qa_log", [])
    last_answer_was_yes = bool(qa_log_so_far) and qa_log_so_far[-1].get("a") == "yes"
    if entry is not None and q_count >= _20Q_EARLY_GUESS_FLOOR and last_answer_was_yes:
        decision = _20q_decide(
            _game_state.get("qa_log", []), asked, q_count,
            _game_state.get("candidates", []), _game_state.get("guesses", []))
        _game_state["candidates"] = decision.get(
            "candidates", _game_state.get("candidates", []))
        if decision.get("action") == "guess":
            return _20q_make_guess(person_id, suggested=decision.get("subject", ""))
        # Not confident yet — ignore the engine's question and stay on the vetted spine.

    if entry is not None:
        question = entry["question"]
        _game_state["last_question"] = question
        _game_state["last_concept"] = entry.get("concept")
        asked.append(_norm_q(question))
        _game_state["question_count"] = q_count + 1
        return (
            _rex_respond(
                f"[GAME: 20 Questions — Q#{q_count + 1}/{_20Q_MAX_QUESTIONS}] Rex is trying to "
                f"guess what the player is thinking of. Rex asks them this yes/no question, in "
                f"his voice — keep it a clear yes/no question, brief: \"{question}\"",
                person_id,
            ),
            False,
        )

    # Mid/late game: the candidate-tracking engine narrows the shortlist or commits to a guess.
    decision = _20q_decide(
        _game_state.get("qa_log", []), asked, q_count,
        _game_state.get("candidates", []), _game_state.get("guesses", []))
    _game_state["candidates"] = decision.get("candidates", _game_state.get("candidates", []))
    if decision.get("action") == "guess":
        return _20q_make_guess(person_id, suggested=decision.get("subject", ""))

    question = decision.get("question") or "Is it something you could hold in one hand?"
    _game_state["last_question"] = question
    _game_state["last_concept"] = None
    asked.append(_norm_q(question))
    _game_state["question_count"] = q_count + 1
    return (
        _rex_respond(
            f"[GAME: 20 Questions — Q#{q_count + 1}/{_20Q_MAX_QUESTIONS}] Rex is narrowing in on "
            f"what the player is thinking of. Rex asks them this yes/no question in his voice — "
            f"keep it a clear yes/no question, brief: \"{question}\"",
            person_id,
        ),
        False,
    )


def _20q_start(person_id: Optional[int]) -> str:
    """Player thinks of something; Rex will guess it. Roles are reversed from classic 20Q."""
    _game_state.update({
        "phase": "ready",
        "qa_log": [],
        "asked": [],
        "concept_answers": {},
        "question_count": 0,
        "guesses": [],
        "candidates": [],
        "last_question": "",
        "last_concept": None,
    })
    _body_beat("thinking_tilt")
    return _rex_respond(
        f"[GAME: 20 Questions — START] Roles are reversed: the PLAYER secretly picks something "
        f"and REX guesses it. Give Rex's opening line in his cocky voice, but it MUST make all "
        f"THREE of these unmistakable to the player:\n"
        f"  1) Think of any one person, place, or thing.\n"
        f"  2) Keep it SECRET — do NOT say it out loud.\n"
        f"  3) Tell Rex when you're ready (e.g. say \"ready\" or \"I've got it\") and the "
        f"questions begin.\n"
        f"He's confident he'll crack it in {_20Q_MAX_QUESTIONS} yes/no questions. Keep it punchy "
        f"(2-3 short sentences) but leave NONE of the three instructions out.",
        person_id,
    )


def _20q_handle(text: str, person_id: Optional[int]) -> tuple[str, bool]:
    phase = _game_state.get("phase", "ready")

    # The player has thought of something; their first reply kicks off the questioning.
    if phase == "ready":
        _game_state["phase"] = "asking"
        return _20q_ask_next(person_id)

    # Rex has made a guess and is awaiting the player's verdict.
    if phase == "guessing":
        verdict = _20q_classify_answer(text)
        guess = _game_state.get("pending_guess", "it")
        q_count = _game_state.get("question_count", 0)
        if verdict == "yes":
            _body_beat("tiny_victory_dance")
            _game_state["result"] = "win"
            _game_state["final_guess"] = guess
            return (
                _rex_respond(
                    f"[GAME: 20 Questions — REX WINS] Rex correctly guessed \"{guess}\" in "
                    f"{q_count} questions. Rex gloats — insufferably proud of his deductive "
                    f"superiority, fully in character.",
                    person_id,
                ),
                True,
            )
        # Not a clean yes → the guess was wrong.
        guesses = _game_state.get("guesses", [])
        if len(guesses) >= _20Q_MAX_GUESSES or q_count >= _20Q_MAX_QUESTIONS:
            _body_beat("suspicious_glance")
            _game_state["result"] = "lose"
            return (
                _rex_respond(
                    f"[GAME: 20 Questions — REX LOSES] Rex guessed \"{guess}\" and was wrong, and "
                    f"he's out of guesses. Rex concedes — grudging and dramatic, blames the player "
                    f"for thinking of something absurd, and asks what it actually was. In character.",
                    person_id,
                ),
                True,
            )
        # Wrong, but Rex still has road left: log the rejected guess (as both a fact and an
        # asked question, so the engine can neither re-guess it nor re-fish it) and keep
        # narrowing.
        _body_beat("suspicious_glance")
        _game_state["phase"] = "asking"
        _game_state.get("qa_log", []).append({"q": f"is it {guess}?", "a": "no"})
        _game_state.setdefault("asked", []).append(_norm_q(f"is it {guess}?"))
        return _20q_ask_next(person_id)

    # phase == "asking": the player's input is the answer to Rex's last question.
    answer = _20q_classify_answer(text)
    last_q = _game_state.get("last_question", "")
    last_concept = _game_state.get("last_concept")
    qa_log = _game_state.get("qa_log", [])
    qa_log.append({"q": last_q, "a": answer})
    _game_state["qa_log"] = qa_log
    if last_concept and answer in ("yes", "no"):
        _game_state.setdefault("concept_answers", {})[last_concept] = (answer == "yes")
    return _20q_ask_next(person_id)


def _20q_stop(person_id: Optional[int]) -> str:
    q_count = _game_state.get("question_count", 0)
    return _rex_respond(
        f"[GAME: 20 Questions — STOPPED] The player ended the guessing game early after Rex asked "
        f"{q_count} questions — before he could crack it. Rex is annoyed to be denied the win and "
        f"insists he was about to get it. Brief, in character.",
        person_id,
    )


# ── Trivia game ───────────────────────────────────────────────────────────────

_TRIVIA_CORRECT_LINES = [
    "Correct. Apparently the organic processor still boots.",
    "Correct. I am recording that as suspicious competence.",
    "Correct. Tiny parade, very tiny budget.",
    "Correct. I will notify my programming that hope was briefly justified.",
    "Correct. The scoreboard and I are both handling it professionally.",
    "Correct. Disturbing, but technically legal.",
]

_TRIVIA_WRONG_LINES = [
    "Nope. A brave answer, if bravery means ignoring facts.",
    "Incorrect. The answer wandered off and you chased a chair.",
    "No. The trivia board remains unimpressed.",
    "Incorrect. Strong confidence, poor landing. I relate.",
    "No. My sensors detected certainty, not accuracy.",
    "Incorrect. The facts filed a complaint.",
]


def _trivia_round_length() -> int:
    return max(1, int(getattr(config, "TRIVIA_ROUND_LENGTH", 5) or 5))


def _trivia_theme_from_game_name(name: str) -> Optional[str]:
    clean = " ".join(name.lower().strip().split())
    for suffix in (" trivia game", " trivia"):
        if clean.endswith(suffix):
            theme = clean[: -len(suffix)].strip()
            if theme:
                return theme
    return None


def _trivia_resolve_preset_category(game_name: str) -> Optional[str]:
    theme = _trivia_theme_from_game_name(game_name)
    if not theme:
        return None
    try:
        from features import trivia as trivia_bank
        return trivia_bank.resolve_category(theme, trivia_bank.get_categories())
    except Exception as exc:
        _log.debug("[games] trivia preset category resolution failed: %s", exc)
        return None


def _trivia_format_categories(categories: list[str]) -> str:
    if not categories:
        return "none"
    if len(categories) == 1:
        return categories[0]
    return ", ".join(categories[:-1]) + f", or {categories[-1]}"


def _trivia_setup_line(categories: list[str]) -> str:
    return (
        f"Trivia systems online. Choose a category and difficulty. "
        f"Categories are: {_trivia_format_categories(categories)}. "
        "Difficulties are easy, medium, hard, or mixed. "
        "Say something like 'Science medium' or 'surprise me.'"
    )


def _trivia_question_line(*, prefix: str = "") -> str:
    question = _game_state.get("question") or {}
    q_num = int(_game_state.get("question_number", 1) or 1)
    total = int(_game_state.get("total_questions", _trivia_round_length()) or 1)
    category = _game_state.get("category", "Trivia")
    difficulty_label = _game_state.get("difficulty_label", "mixed")
    return (
        f"{prefix}{category}, {difficulty_label}. "
        f"Question {q_num} of {total}: {question.get('question', 'Question missing. Blame the card catalog.')}"
    )


def _trivia_prepare_question() -> bool:
    try:
        from features import trivia as trivia_bank
        question = trivia_bank.get_question(
            str(_game_state.get("category") or ""),
            _game_state.get("difficulty"),
        )
    except Exception as exc:
        _log.error("[games] trivia question load failed: %s", exc)
        question = None
    if not question:
        return False
    _game_state["question"] = question
    _game_state["phase"] = "awaiting_answer"
    return True


def _trivia_begin_round(
    category: str,
    difficulty: Optional[int],
    *,
    person_id: Optional[int],
) -> tuple[str, bool]:
    try:
        from features import trivia as trivia_bank
        difficulty_label = trivia_bank.difficulty_label(difficulty)
    except Exception:
        difficulty_label = "mixed"

    _game_state.update({
        "phase": "awaiting_answer",
        "category": category,
        "difficulty": difficulty,
        "difficulty_label": difficulty_label,
        "score": 0,
        "question_number": 1,
        "total_questions": _trivia_round_length(),
        "history": [],
    })

    if not _trivia_prepare_question():
        return (
            _rex_respond(
                f"[GAME: Trivia — NO QUESTION] Rex tried to load a {difficulty_label} "
                f"question from category \"{category}\" but came up empty. Apologize in character.",
                person_id,
            ),
            True,
        )

    _body_beat("thinking_tilt")
    return (_trivia_question_line(prefix="Locked in. "), False)


def _trivia_is_pass(text: str) -> bool:
    clean = " ".join(text.lower().strip().split())
    return clean in {"pass", "skip", "i don't know", "i dont know", "no idea", "not sure"}


def _trivia_final_line(prefix: str) -> str:
    score = int(_game_state.get("score", 0) or 0)
    total = int(_game_state.get("total_questions", _trivia_round_length()) or 1)
    if score == total:
        verdict = "Perfect score. Annoying, but statistically elegant."
    elif score >= max(1, round(total * 0.7)):
        verdict = "Respectable. I will pretend not to be mildly impressed."
    elif score > 0:
        verdict = "Not catastrophic. A low bar, but you cleared it."
    else:
        verdict = "Zero correct. The scoreboard is now filing a grievance."
    return f"{prefix} Final score: {score} out of {total}. {verdict}"


def _trivia_start(person_id: Optional[int], preset_category: Optional[str] = None) -> str:
    try:
        from features import trivia as trivia_bank
    except Exception as exc:
        _log.error("[games] trivia import failed: %s", exc)
        return _rex_respond(
            "[GAME: Trivia — START FAILED] Trivia question loading failed. "
            "Rex apologizes in character and suggests another game.",
            person_id,
        )

    categories = trivia_bank.get_categories()
    if not categories:
        return _rex_respond(
            "[GAME: Trivia — NO QUESTIONS] No trivia categories are available. "
            "Rex apologizes in character and suggests another game.",
            person_id,
        )

    if preset_category and preset_category in categories:
        return _trivia_begin_round(preset_category, None, person_id=person_id)[0]

    _game_state.update({
        "phase": "setup",
        "categories": categories,
    })

    return _trivia_setup_line(categories)


def _trivia_handle(text: str, person_id: Optional[int]) -> tuple[str, bool]:
    phase = _game_state.get("phase", "setup")
    if phase == "setup":
        try:
            from features import trivia as trivia_bank
            categories = list(_game_state.get("categories") or trivia_bank.get_categories())
            category = trivia_bank.resolve_category(text, categories)
            difficulty = trivia_bank.parse_difficulty(text)
        except Exception as exc:
            _log.error("[games] trivia setup parse failed: %s", exc)
            categories = list(_game_state.get("categories") or [])
            category = None
            difficulty = None

        if not category:
            return (
                f"Pick a category first, carbon-based contestant. "
                f"Categories are: {_trivia_format_categories(categories)}. "
                "You can add easy, medium, hard, or mixed.",
                False,
            )

        return _trivia_begin_round(category, difficulty, person_id=person_id)

    question = _game_state.get("question")
    category = _game_state.get("category", "Trivia")
    if not isinstance(question, dict):
        _game_state.clear()
        return (
            _rex_respond(
                "[GAME: Trivia — STATE ERROR] The trivia round lost its question state. "
                "Rex acknowledges the glitch in character.",
                person_id,
            ),
            True,
        )

    passed = _trivia_is_pass(text)
    try:
        from features import trivia as trivia_bank
        is_correct = False if passed else trivia_bank.check_answer(question, text)
    except Exception as exc:
        _log.error("[games] trivia answer check failed: %s", exc)
        is_correct = False

    answer = question.get("answer", "unknown")
    q_num = int(_game_state.get("question_number", 1) or 1)
    total = int(_game_state.get("total_questions", _trivia_round_length()) or 1)
    score = int(_game_state.get("score", 0) or 0)

    if is_correct:
        score += 1
        _game_state["score"] = score
        _body_beat("tiny_victory_dance")
    else:
        _body_beat("suspicious_glance")

    history = list(_game_state.get("history") or [])
    history.append({
        "question": question.get("question", ""),
        "answer": answer,
        "user_answer": text.strip(),
        "correct": bool(is_correct),
    })
    _game_state["history"] = history

    if is_correct:
        feedback = f"{random.choice(_TRIVIA_CORRECT_LINES)} Score: {score} out of {q_num}. "
    elif passed:
        feedback = f"No answer. Correct answer was {answer}. Score: {score} out of {q_num}. "
    else:
        feedback = (
            f"{random.choice(_TRIVIA_WRONG_LINES)} Correct answer was {answer}. "
            f"Score: {score} out of {q_num}. "
        )

    if q_num >= total:
        return (_trivia_final_line(feedback), True)

    _game_state["question_number"] = q_num + 1
    if not _trivia_prepare_question():
        return (
            _trivia_final_line(
                f"{feedback}I ran out of {category} questions before the round ended. "
            ),
            True,
        )

    return (
        _trivia_question_line(prefix=feedback),
        False,
    )


def _trivia_stop(person_id: Optional[int]) -> str:
    question = _game_state.get("question", {})
    answer = question.get("answer", "unknown") if isinstance(question, dict) else "unknown"
    score = int(_game_state.get("score", 0) or 0)
    total = int(_game_state.get("total_questions", _trivia_round_length()) or 1)
    q_num = int(_game_state.get("question_number", 1) or 1)
    answered = len(_game_state.get("history") or [])
    _game_state.clear()
    suffix = f" Current answer was {answer}." if answer != "unknown" else ""
    return (
        f"Trivia stopped on question {q_num} of {total}. "
        f"Score: {score} out of {answered}.{suffix} "
        "A merciful pause for the neurons."
    )


# ── Jeopardy-style verbal game ────────────────────────────────────────────────

_JEOPARDY_CLIPS = {
    "intro": "jeopardy-intro.mp3",
    "board": "jeopardy-board-sms.mp3",
    "daily_double": "jeopardy-daily-double.mp3",
    "right": "jeopardy-rightanswer.mp3",
    "wrong": "jeopardy-incorrect-answer.mp3",
    "timesup": "jeopardy-timesup.mp3",
    "theme": "jeopardy-theme.mp3",
    "final_theme": "jeopardy-final-jeopardy-thinking-music.mp3",
    "outro": "jeopardy-outro-no-talking.mp3",
}


def _jeopardy_clip_path(key: str) -> Optional[str]:
    filename = _JEOPARDY_CLIPS.get(key)
    if not filename:
        return None
    path = Path(getattr(config, "JEOPARDY_AUDIO_DIR", "assets/audio/jeopardy")) / filename
    if not path.exists():
        _log.debug("[jeopardy] audio clip missing: %s", path)
        return None
    return str(path)


def _jeopardy_queue_clip(key: str, *, priority: int = 1) -> None:
    path = _jeopardy_clip_path(key)
    if not path:
        return
    try:
        from audio import speech_queue
        speech_queue.enqueue_audio_file(
            path,
            priority=priority,
            tag=f"jeopardy:{key}",
        )
    except Exception as exc:
        _log.debug("[jeopardy] could not queue clip %s: %s", key, exc)


def _jeopardy_person_name(person_id: Optional[int]) -> Optional[str]:
    if person_id is None:
        return None
    try:
        from memory import people as people_memory
        row = people_memory.get_person(person_id)
        if row and row.get("name"):
            return str(row["name"])
    except Exception as exc:
        _log.debug("[jeopardy] person lookup failed: %s", exc)
    return None


_JEOPARDY_NICKNAME_CANDIDATES = {
    "jen": ["Jennifer"],
    "jenn": ["Jennifer"],
    "dan": ["Daniel"],
    "danny": ["Daniel"],
    "will": ["William", "Will"],
    "bill": ["William", "Bill"],
    "bret": ["Bret", "Brett"],
    "brett": ["Brett", "Bret"],
}


def _jeopardy_player_display_name(name: str) -> str:
    cleaned = " ".join((name or "Player").split()) or "Player"
    return cleaned.split()[0]


def _jeopardy_find_or_create_player(name: str) -> tuple[Optional[int], str]:
    try:
        from memory import people as people_memory
    except Exception as exc:
        _log.debug("[jeopardy] people memory unavailable: %s", exc)
        return None, _jeopardy_player_display_name(name)

    candidates = [name]
    candidates.extend(_JEOPARDY_NICKNAME_CANDIDATES.get((name or "").strip().lower(), []))
    for candidate in candidates:
        try:
            existing = people_memory.find_person_by_name(candidate)
        except Exception:
            existing = None
        if existing:
            stored_name = str(existing.get("name") or candidate)
            return int(existing["id"]), _jeopardy_player_display_name(stored_name)

    try:
        pid, _created = people_memory.find_or_create_person(name)
        return (int(pid) if pid is not None else None), _jeopardy_player_display_name(name)
    except Exception as exc:
        _log.debug("[jeopardy] player row create failed for %r: %s", name, exc)
        return None, _jeopardy_player_display_name(name)


def _jeopardy_prepare_players(names: list[str]) -> tuple[list[dict], list[int]]:
    players: list[dict] = []
    needs_voice: list[int] = []
    try:
        from memory import people as people_memory
    except Exception:
        people_memory = None

    for raw_name in names:
        person_id, display_name = _jeopardy_find_or_create_player(raw_name)
        player = {"name": display_name, "score": 0}
        if person_id is not None:
            player["person_id"] = person_id
            try:
                if people_memory is not None and not people_memory.has_voice_biometric(person_id):
                    needs_voice.append(len(players))
            except Exception:
                pass
        players.append(player)
    return players, needs_voice


def _jeopardy_voice_check_prompt(player: dict, *, prefix: str = "") -> str:
    name = player.get("name") or "player"
    return (
        f"{prefix}I need a cleaner voice print for {name} before the board starts. "
        f"{name}, say: \"My name is {name}, and I'm playing Jeopardy.\" "
        f"Or say \"skip {name}\" to play without it."
    )


def _jeopardy_confident_other_speaker(person_id: Optional[int]) -> Optional[dict]:
    """The registered player who spoke, when it is demonstrably NOT the player
    whose turn it is. None when the speaker is unresolved, is the current
    player, or is not on the roster.

    Used to decide whose money is on the line, and ONLY in the losing
    direction — see _jeopardy_handle_answer.
    """
    if person_id is None:
        return None
    players = _game_state.get("players") or []
    if len(players) < 2:
        return None
    idx = int(_game_state.get("current_player_idx", 0)) % len(players)
    for i, player in enumerate(players):
        pid = (player or {}).get("person_id")
        if pid is None:
            continue
        try:
            if int(pid) != int(person_id):
                continue
        except (TypeError, ValueError):
            continue
        return None if i == idx else dict(player)
    return None


def _jeopardy_current_player() -> dict:
    players = _game_state.get("players") or [{"name": "Player", "score": 0}]
    idx = int(_game_state.get("current_player_idx", 0)) % len(players)
    return players[idx]


def _jeopardy_advance_player() -> dict:
    players = _game_state.get("players") or [{"name": "Player", "score": 0}]
    idx = (int(_game_state.get("current_player_idx", 0)) + 1) % len(players)
    _game_state["current_player_idx"] = idx
    return players[idx]


def _jeopardy_cancel_timeout() -> None:
    timer = _game_state.pop("answer_timer", None)
    _game_state.pop("answer_timer_token", None)
    _game_state.pop("answer_timer_deadline", None)
    _game_state.pop("awaiting_prompt_delivery", None)
    if timer is not None:
        try:
            timer.cancel()
        except Exception:
            pass


def _jeopardy_maybe_offer_round_jump() -> str:
    """Once per round, mention the jump — at half a board OR after N clues.

    Owner note 2026-08-25: a full round is 30 clues and the table quit
    mid-round-one — they never knew fresh categories were a sentence away.
    The remaining-based trigger alone could not reach them: a real table spends
    1-2 minutes per square, so 15-remaining is 20+ minutes away. Field
    2026-08-26: 11 squares in 17 minutes, offer never fired, and the owner had
    to say "new board" himself.
    """
    if _game_state.get("jump_offered"):
        return ""
    threshold = int(getattr(config, "JEOPARDY_ROUND_JUMP_OFFER_REMAINING", 15))
    after_clues = int(getattr(config, "JEOPARDY_ROUND_JUMP_OFFER_AFTER_CLUES", 6))
    if threshold <= 0 and after_clues <= 0:
        return ""
    remaining = int((_game_state.get("board") or {}).get("remaining", 0) or 0)
    if remaining <= 0:
        return ""
    played = max(0, int(_game_state.get("board_size", 0) or 0) - remaining)
    by_remaining = threshold > 0 and remaining <= threshold
    by_played = after_clues > 0 and played >= after_clues
    if not (by_remaining or by_played):
        return ""
    _game_state["jump_offered"] = True
    current_round = int(_game_state.get("jeopardy_round", 1) or 1)
    destination = "Double Jeopardy" if current_round < 2 else "Final Jeopardy"
    return f"Say 'next round' any time and I'll deal {destination}. "


def _jeopardy_score_announcement(player: dict) -> str:
    """The score line after a scoring event.

    Owner call 2026-08-25: reading EVERY player's total after EVERY answer made
    single responses run 20+ seconds. Normally just the answerer's new total;
    the full scoreboard every JEOPARDY_SCOREBOARD_EVERY-th scoring event (and
    always on round transitions, at the finish, and on "what's the score?").
    """
    events = int(_game_state.get("score_events", 0) or 0)
    _game_state["score_events"] = events + 1
    every = int(getattr(config, "JEOPARDY_SCOREBOARD_EVERY", 4))
    try:
        from features import jeopardy as jeopardy_bank
    except Exception:
        return ""
    if every > 0 and (events + 1) % every == 0:
        return f"Scores: {jeopardy_bank.format_scores(_game_state.get('players') or [])}. "
    total = jeopardy_bank.format_score(int(player.get("score", 0) or 0))
    return f"That puts {player['name']} at {total}. "


def _jeopardy_finish_line(prefix: str = "") -> str:
    try:
        from features import jeopardy as jeopardy_bank
        scores = jeopardy_bank.format_scores(_game_state.get("players") or [])
    except Exception:
        scores = "scores unavailable, naturally"
    _jeopardy_cancel_timeout()
    _jeopardy_queue_clip("outro")
    return (
        f"{prefix}That's the board. Final scores: {scores}. "
        "Jeopardy systems powering down before someone asks me to host Wheel of Fortune."
    )


def _jeopardy_correct_response_text(clue: dict) -> str:
    answer = str((clue or {}).get("answer") or "unknown")
    try:
        from features import jeopardy as jeopardy_bank
        response = jeopardy_bank.format_correct_response(
            answer,
            clue=str((clue or {}).get("clue") or ""),
            category=str((clue or {}).get("category") or ""),
        )
    except Exception:
        response = f"What is {answer.strip(' .!?') or 'unknown'}?"
    return f'Correct response was: "{response}"'


# The judge answers two questions in ONE call — "is this right?" and "was this
# even an answer?" — so the ignore gate costs no extra round-trip. The verdict
# is stashed here for _jeopardy_grade to read back; tests that patch
# _jeopardy_llm_judge leave it empty, which reads as "no opinion".
_LAST_JUDGE_VERDICT: dict = {"key": None, "verdict": ""}


def _jeopardy_llm_verdict(user_text: str, expected_answer: str, clue: dict) -> str:
    """"correct" | "wrong" | "not_an_answer" | "" (no opinion).

    Player answers arrive via SPEECH: a RIGHT answer can reach the matcher
    phonetically mangled ("day cart" for Descartes, "shack" for Shaq) or phrased
    in a way lexical fuzzy matching can't score. This gives the borderline miss
    ONE strict look before the value is deducted. It can only rescue a wrong
    verdict — the deterministic matcher's accepts are never re-litigated — and
    any error fails safe to "wrong".

    "not_an_answer" is the third road (owner report 2026-08-26): the room talks
    over a live clue, and every one of those turns used to be a deduction — PJ
    calling the dog ("Come here, Toby") cost Bret $400.
    """
    _LAST_JUDGE_VERDICT["key"] = None
    _LAST_JUDGE_VERDICT["verdict"] = ""
    if not bool(getattr(config, "JEOPARDY_LLM_JUDGE_ENABLED", True)):
        return ""
    guess = (user_text or "").strip()
    max_chars = int(getattr(config, "JEOPARDY_LLM_JUDGE_MAX_ANSWER_CHARS", 120))
    if not guess or len(guess) > max_chars:
        return ""
    try:
        raw = _quick_call(
            "You are a strict Jeopardy judge. The player's answer was transcribed "
            "from SPEECH, so a correct answer may arrive misspelled or phonetically "
            "mangled (e.g. 'day cart' for Descartes).\n"
            f"Clue: \"{(clue or {}).get('clue', '')}\" "
            f"(category: {(clue or {}).get('category', '')})\n"
            f"Correct answer: \"{expected_answer}\"\n"
            f"Player said: \"{guess}\"\n"
            "Reply with ONLY one word:\n"
            "yes — the response identifies the SAME answer (same person, place or "
            "thing), allowing phonetic/transcription mangling, filler words and "
            "question phrasing.\n"
            "no — they attempted an answer and it is a different answer, a broader "
            "category, or missing a required part of a multi-part answer.\n"
            "none — they were not answering the clue at all: talking to someone "
            "else in the room or to a pet, complaining about the game, or "
            "carrying on a side conversation.",
            temperature=0,
            max_tokens=3,
        ).strip().lower()
    except Exception as exc:
        _log.debug("[jeopardy] LLM judge failed: %s", exc)
        return ""
    if raw.startswith("yes"):
        verdict = "correct"
        _log.info(
            "[jeopardy] LLM judge rescued answer %r for expected %r",
            guess, expected_answer,
        )
    elif raw.startswith("none"):
        verdict = "not_an_answer"
        _log.info("[jeopardy] LLM judge: %r was not an answer attempt", guess)
    else:
        verdict = "wrong"
    _LAST_JUDGE_VERDICT["key"] = (guess, str(expected_answer or ""))
    _LAST_JUDGE_VERDICT["verdict"] = verdict
    return verdict


def _jeopardy_llm_judge(user_text: str, expected_answer: str, clue: dict) -> bool:
    """Strict rescue judge: True only when the LLM says the answer is correct."""
    return _jeopardy_llm_verdict(user_text, expected_answer, clue) == "correct"


def _jeopardy_categories_reminder() -> str:
    # The fatigue curve below (not a blanket mute) is what keeps the reminder
    # from being tiresome — it runs in BOTH modes. The old GUI mute is now
    # opt-in only (JEOPARDY_READ_CATEGORIES_WITH_GUI=False), for a table that is
    # actually looking at the JeopardyPanel: as a default it silently killed the
    # read-out for players sitting around the ROBOT, and the 2026-08-26 game
    # (a manual `main.py --gui --jeopardy`) spoke none at all.
    if bool(getattr(config, "GUI_ENABLED", False)) and not bool(
        getattr(config, "JEOPARDY_READ_CATEGORIES_WITH_GUI", False)
    ):
        # Never suppress SILENTLY. The 2026-08-26 run spoke no reminder at all
        # and the log could not say which branch was responsible — this mute,
        # the fatigue curve, or an empty board — which is exactly the kind of
        # question a postmortem should not have to guess at.
        if not _game_state.get("categories_reminder_muted_logged"):
            _game_state["categories_reminder_muted_logged"] = True
            _log.info(
                "[jeopardy] spoken category reminder muted for this round "
                "(GUI_ENABLED and JEOPARDY_READ_CATEGORIES_WITH_GUI is False)"
            )
        return ""
    # Voice-only fatigue curve (owner call 2026-08-25: great early game, tiresome
    # once everyone knows the board): the first FULL_READS scoring turns repeat
    # the list every time, then only every EVERY-th turn. The counter resets each
    # round (a fresh board is announced in full anyway), and an explicit "what
    # are the categories?" bypasses this entirely via _jeopardy_board_text.
    reads = int(_game_state.get("categories_reminder_reads", 0) or 0)
    _game_state["categories_reminder_reads"] = reads + 1
    full_reads = int(getattr(config, "JEOPARDY_CATEGORIES_REMINDER_FULL_READS", 4))
    every = int(getattr(config, "JEOPARDY_CATEGORIES_REMINDER_EVERY", 3))
    if reads >= full_reads:
        if every <= 0:
            return ""
        if (reads - full_reads + 1) % every != 0:
            return ""
    board = _game_state.get("board") or {}
    try:
        from features import jeopardy as jeopardy_bank
        categories = jeopardy_bank.format_categories(
            board,
            remaining_only=True,
            separator=". ",
        )
    except Exception:
        categories = ""
    if not categories:
        _log.info("[jeopardy] category reminder empty — no categories left to read")
        return ""
    return f"Remaining categories: {categories}. "


def _jeopardy_board_text() -> str:
    """The live board spoken aloud, or "" if nothing is left.

    Unlike _jeopardy_categories_reminder this ignores the GUI mute: the mute
    exists so the board is not re-read EVERY turn, but an explicit "what are
    the categories?" always deserves an answer.
    """
    board = _game_state.get("board") or {}
    try:
        from features import jeopardy as jeopardy_bank
        return jeopardy_bank.format_board_readout(board)
    except Exception as exc:
        _log.debug("[jeopardy] board readout failed: %s", exc)
        return ""


def _jeopardy_speak_category(name: Optional[str]) -> str:
    try:
        from features import jeopardy as jeopardy_bank
        return jeopardy_bank.speak_category(name or "")
    except Exception:
        return str(name or "")


def _jeopardy_answer_board_question(text: str) -> Optional[str]:
    """Deterministic answer to a mid-game board/score/turn question, or None.

    Owner ask 2026-08-25: "what's left in pop culture?", "is the 400 still
    there in history?", "what's the score?" — answerable mid-game without
    consuming a square or grading the question as a wrong answer. Value
    availability runs FIRST: it is the only question shape that carries a
    dollar value, and everything downstream treats a value as a pick.
    """
    try:
        from features import jeopardy as jeopardy_bank
    except Exception:
        return None
    board = _game_state.get("board") or {}
    players = _game_state.get("players") or []

    avail = jeopardy_bank.value_availability_query(text, board)
    if avail is not None:
        value = int(avail["value"])
        category = avail["category"]
        if category is not None:
            name = jeopardy_bank.speak_category(category.get("name") or "")
            if value in (category.get("clues") or {}):
                return f"Yes — {name} for ${value} is still on the board. "
            return f"No — {name} for ${value} is gone. "
        open_in = avail["open_in"]
        if open_in:
            names = ", ".join(jeopardy_bank.speak_category(n) for n in open_in)
            return f"The ${value} squares still live: {names}. "
        return f"Every ${value} square is gone. "

    catq = jeopardy_bank.category_board_query(text, board)
    if catq is not None:
        category, fragment = catq
        if category is None:
            readout = _jeopardy_board_text()
            if readout:
                return (
                    f"No category sounds like '{fragment}'. "
                    f"Still on the board: {readout}. "
                )
            return "The board is picked clean. "
        name = jeopardy_bank.speak_category(category.get("name") or "")
        values = sorted(int(v) for v in (category.get("clues") or {}).keys())
        if values:
            values_text = ", ".join(f"${v}" for v in values)
            return f"{name} still has {values_text}. "
        return f"{name} is cleaned out. "

    if jeopardy_bank.is_score_request(text):
        return f"Scores: {jeopardy_bank.format_scores(players)}. "

    if jeopardy_bank.is_turn_request(text):
        player = _jeopardy_current_player()
        if _game_state.get("phase") == "awaiting_answer":
            return f"{player['name']} is on the clock for this clue. "
        return f"It's {player['name']}'s pick. "

    return None


def _jeopardy_board_question_llm(text: str, person_id: Optional[int]) -> Optional[str]:
    """LLM fallback for a board question no deterministic lane recognized.

    Selecting phase ONLY (a live clue keeps strict deterministic grading), and
    only for value-free, question-shaped turns — a mangled pick keeps the
    canned retry error, which lists what is actually available.
    """
    if not bool(getattr(config, "JEOPARDY_BOARD_QA_LLM_FALLBACK_ENABLED", True)):
        return None
    try:
        from features import jeopardy as jeopardy_bank
    except Exception:
        return None
    if jeopardy_bank.mentions_value(text):
        return None
    # The STRICT gate, not looks_like_question: this lane hands the text to a
    # free-form persona generation with no pattern-matching backstop, so a bare
    # auxiliary opener with no question mark (the classic clipped-ASR fragment)
    # must not reach it.
    if not jeopardy_bank.looks_like_board_question(text):
        return None
    board_text = _jeopardy_board_text()
    players = _game_state.get("players") or []
    player = _jeopardy_current_player()
    try:
        scores = jeopardy_bank.format_scores(players)
    except Exception:
        scores = "unavailable"
    round_no = int(_game_state.get("jeopardy_round", 1) or 1)
    context = (
        f'[GAME: Jeopardy — BOARD QUESTION] Mid-game, a player asked: "{text}". '
        "Answer from THIS data only — never invent squares, values, or scores. "
        f"Remaining board: {board_text or 'nothing — the board is empty'}. "
        f"Scores: {scores}. Round {round_no}. It is {player['name']}'s turn to pick. "
        f"Reply in one or two short sentences, then tell {player['name']} to pick "
        "a category and dollar value."
    )
    response = _rex_respond(context, person_id)
    return response or None


def _jeopardy_table_talk_aside(text: str, person_id: Optional[int]) -> Optional[str]:
    """One in-character line for a side remark that is not a pick or a question.

    Field 2026-08-26 20:25:06: "Hey, take her points away. She cheated." was
    answered "Pick a dollar value too, before my game-show circuits start
    smoking", which reads as Rex not listening. Shares the board-QA kill switch
    — both are the same "let the LLM handle what the parser can't" lane.
    """
    if not bool(getattr(config, "JEOPARDY_BOARD_QA_LLM_FALLBACK_ENABLED", True)):
        return None
    try:
        from features import jeopardy as jeopardy_bank
        named_category = bool(jeopardy_bank.selection_category_hint(
            text, _game_state.get("board") or {}))
        if not jeopardy_bank.is_table_chatter(text, named_category):
            return None
        player = _jeopardy_current_player()
        aside = _rex_respond(
            f'[GAME: Jeopardy — TABLE TALK] Mid-game a player said: "{text}". '
            "That is table talk, not a pick and not a question about the board. "
            "React in ONE short line, in character, then tell "
            f"{player['name']} to pick a category and dollar value. "
            "Never change anyone's score.",
            person_id,
        )
        return aside or None
    except Exception as exc:
        _log.debug("[jeopardy] table-talk aside failed: %s", exc)
        return None


def _jeopardy_repeat_clue_reply(clue: dict, player: dict, prefix: str = "") -> str:
    """Re-read the live clue and restart the answer window, scoring nothing."""
    _game_state["phase"] = "awaiting_answer"
    _game_state["awaiting_prompt_delivery"] = True
    if bool(getattr(config, "JEOPARDY_PLAY_THINKING_THEME", False)):
        _game_state["pending_after_response_clip"] = "theme"
    return (
        f"{prefix}{player['name']}, "
        f"{_jeopardy_speak_category(clue.get('category'))} for ${clue.get('value')}. "
        f"Clue: {clue.get('clue')}."
    )


def _jeopardy_offer_rebound() -> Optional[dict]:
    """Give the same clue to the next player who has not tried it yet."""
    players = _game_state.get("players") or []
    if len(players) <= 1:
        return None

    current_idx = int(_game_state.get("current_player_idx", 0)) % len(players)
    attempted = set(int(i) for i in (_game_state.get("current_clue_attempts") or []))
    attempted.add(current_idx)
    _game_state["current_clue_attempts"] = sorted(attempted)

    # One second chance around the table, then the answer is revealed and the
    # board moves on. The same clue read to four people in a row, each with its
    # own 12 s clock and thinking theme, is what made the 2026-08-26 game feel
    # stuck ("every time it's my turn, it's from a category I would have never
    # chosen").
    max_rebounds = max(0, int(getattr(config, "JEOPARDY_MAX_REBOUNDS", 1)))
    if len(attempted) > max_rebounds:
        return None

    for offset in range(1, len(players)):
        next_idx = (current_idx + offset) % len(players)
        if next_idx in attempted:
            continue
        _game_state["current_player_idx"] = next_idx
        _game_state["phase"] = "awaiting_answer"
        _game_state["awaiting_prompt_delivery"] = True
        if bool(getattr(config, "JEOPARDY_PLAY_THINKING_THEME", False)):
            _game_state["pending_after_response_clip"] = "theme"
        return players[next_idx]

    return None


def _jeopardy_rebound_prompt(prefix: str, next_player: dict, clue: dict) -> str:
    return (
        f"{prefix}{next_player['name']}'s turn. "
        f"{_jeopardy_speak_category(clue.get('category'))} for ${clue.get('value')}. "
        f"Clue: {clue.get('clue')}."
    )


def _jeopardy_finish_missed_clue(
    prefix: str,
    correct_response: str,
    *,
    done: bool,
    players: list[dict],
    score_line: bool = True,
    score_player: Optional[dict] = None,
) -> tuple[str, bool]:
    _game_state.pop("current_clue", None)
    _game_state.pop("current_clue_attempts", None)
    _game_state.pop("ignored_turns", None)
    _game_state["phase"] = "selecting"

    if done:
        return _jeopardy_complete_round_or_finish(
            f"{prefix}{correct_response}. ",
            advance_player=True,
        )

    next_player = _jeopardy_advance_player()
    scores = ""
    if score_line:
        if score_player is not None:
            scores = _jeopardy_score_announcement(score_player)
        else:
            try:
                from features import jeopardy as jeopardy_bank
                scores = f"Scores: {jeopardy_bank.format_scores(players)} "
            except Exception:
                scores = "Scores unavailable. "
    categories = _jeopardy_categories_reminder()
    return (
        f"{prefix}{correct_response}. {scores}{categories}"
        f"{_jeopardy_maybe_offer_round_jump()}"
        f"{next_player['name']}, choose the next square.",
        False,
    )


def _jeopardy_schedule_post_timeout_rebound(
    done_event: threading.Event,
    rebound_at: float = 0.0,
) -> None:
    """Start the rebound timer/theme after a timeout prompt finishes speaking."""
    def _wait_then_start() -> None:
        try:
            if not done_event.wait(timeout=45.0):
                return
            with _lock:
                # A grace answer superseded this announcement (its queued line was
                # dropped, or it was graded mid-play): the graded response owns
                # the flow now, and arming the next answer timer from here would
                # start it while that response is still being spoken.
                current = _game_state.get("timeout_rebound")
                if rebound_at and (
                    current is None or float(current.get("at") or 0.0) != rebound_at
                ):
                    return
            on_response_spoken()
            after_audio = consume_pending_audio_after_response()
            if not after_audio:
                return
            from audio import speech_queue
            speech_queue.enqueue_audio_file(after_audio, priority=1, tag="game:after_audio")
        except Exception as exc:
            _log.debug("[jeopardy] post-timeout rebound scheduling failed: %s", exc)

    threading.Thread(
        target=_wait_then_start,
        daemon=True,
        name="jeopardy-timeout-rebound",
    ).start()


def _jeopardy_board_low_value(board: dict) -> int:
    values = [
        int(value)
        for category in board.get("categories") or []
        for value in (category.get("clues") or {}).keys()
    ]
    return min(values) if values else 0


def _jeopardy_load_round(
    round_no: int,
    players: list[dict],
    *,
    current_player_idx: Optional[int] = None,
) -> Optional[str]:
    try:
        from features import jeopardy as jeopardy_bank
        board = jeopardy_bank.build_board(round_no=round_no)
    except Exception as exc:
        _log.error("[jeopardy] round %s board build failed: %s", round_no, exc)
        board = None

    if not board:
        return None

    update = {
        "phase": "selecting",
        "players": players,
        "board": board,
        "board_values": _jeopardy_board_values(board),
        # How big the board STARTED, so the jump offer can trigger on clues
        # played (wall-clock progress) as well as squares remaining.
        "board_size": int(board.get("remaining", 0) or 0),
        "last_category": None,
        "jeopardy_round": round_no,
        # Fresh board, fresh memories: the fatigue curves and the once-per-round
        # jump offer start over.
        "categories_reminder_reads": 0,
        "score_events": 0,
        "jump_offered": False,
    }
    if current_player_idx is not None:
        update["current_player_idx"] = current_player_idx
    _game_state.update(update)
    _jeopardy_queue_clip("board")

    categories_text = jeopardy_bank.format_categories(board, separator=". ")
    player = _jeopardy_current_player()
    if round_no == 2:
        low_value = _jeopardy_board_low_value(board)
        start_note = f" Values start at ${low_value}." if low_value else ""
        return (
            f"Double Jeopardy is loaded.{start_note} "
            f"Categories are: {categories_text}. "
            f"{player['name']}, pick a category and dollar value."
        )
    return (
        f"Categories are: {categories_text}. "
        f"{player['name']}, pick a category and dollar value."
    )


def _jeopardy_board_values(board: dict) -> list[int]:
    values = {
        int(value)
        for category in board.get("categories") or []
        for value in (category.get("clues") or {}).keys()
    }
    return sorted(values)


def _jeopardy_complete_round_or_finish(
    prefix: str,
    *,
    advance_player: bool = False,
) -> tuple[str, bool]:
    current_round = int(_game_state.get("jeopardy_round", 1) or 1)
    if current_round < 2:
        if advance_player:
            _jeopardy_advance_player()
        players = _game_state.get("players") or [{"name": "Player", "score": 0}]
        current_idx = int(_game_state.get("current_player_idx", 0)) % len(players)
        try:
            from features import jeopardy as jeopardy_bank
            scores = jeopardy_bank.format_scores(players)
        except Exception:
            scores = "scores unavailable"
        next_round = _jeopardy_load_round(2, players, current_player_idx=current_idx)
        if next_round:
            return (
                f"{prefix}That's round one. Scores: {scores}. {next_round}",
                False,
            )

    # Double Jeopardy is done — to Final (which degrades to the finish line
    # when disabled, when no final clues exist, or when nobody has money).
    return _jeopardy_begin_final(f"{prefix}That's the board. ")


def _jeopardy_round_jump(kind: str) -> tuple[str, bool]:
    """Voice-requested round change: "next round" / "final jeopardy"."""
    players = _game_state.get("players") or [{"name": "Player", "score": 0}]
    current_round = int(_game_state.get("jeopardy_round", 1) or 1)
    if kind == "next" and current_round < 2:
        try:
            from features import jeopardy as jeopardy_bank
            scores = jeopardy_bank.format_scores(players)
        except Exception:
            scores = "scores unavailable"
        current_idx = int(_game_state.get("current_player_idx", 0)) % len(players)
        next_round = _jeopardy_load_round(2, players, current_player_idx=current_idx)
        if next_round:
            return (f"Fresh board coming up. Scores so far: {scores}. {next_round}", False)
        return ("I couldn't deal a new board — the one you have will have to do.", False)
    return _jeopardy_begin_final("Very well — to Final Jeopardy. ")


def _jeopardy_begin_final(prefix: str = "") -> tuple[str, bool]:
    """Start Final Jeopardy: category announced, wagers collected one by one
    (lowest score first, show style), then the clue + think music, then answers,
    then the reveal. Degrades to the plain finish line when disabled, when no
    final clues load, or when every player is at zero or below."""
    final_clue = None
    if bool(getattr(config, "JEOPARDY_FINAL_ENABLED", True)):
        try:
            from features import jeopardy as jeopardy_bank
            final_clue = jeopardy_bank.pick_final_clue()
        except Exception as exc:
            _log.debug("[jeopardy] final clue pick failed: %s", exc)
    players = _game_state.get("players") or []
    if (
        not final_clue
        or not players
        or all(int(p.get("score", 0) or 0) <= 0 for p in players)
    ):
        return _jeopardy_finish_line(prefix), True

    _jeopardy_cancel_timeout()
    order = sorted(
        range(len(players)), key=lambda i: int(players[i].get("score", 0) or 0)
    )
    _game_state.update({
        "phase": "final_wager",
        "final": {"clue": final_clue, "wagers": {}, "answers": {}, "order": order},
        "final_queue": list(order),
    })
    _game_state.pop("current_clue", None)
    _game_state.pop("current_clue_attempts", None)
    _jeopardy_queue_clip("board")
    category = _jeopardy_speak_category(final_clue.get("category"))
    return _jeopardy_next_final_wager_prompt(
        f"{prefix}This is Final Jeopardy. The category: {category}. "
    )


def _jeopardy_next_final_wager_prompt(prefix: str) -> tuple[str, bool]:
    """Prompt the next positive-score player for a wager; players at zero or
    below ride along at $0 (still get to answer — pride is on the line)."""
    players = _game_state.get("players") or []
    final = _game_state.get("final") or {}
    queue = _game_state.get("final_queue") or []
    bits: list[str] = [prefix]
    try:
        from features import jeopardy as jeopardy_bank
        fmt = jeopardy_bank.format_score
    except Exception:
        fmt = lambda s: f"${s}"    # noqa: E731
    while queue:
        idx = int(queue[0])
        player = players[idx]
        score = int(player.get("score", 0) or 0)
        if score <= 0:
            final.setdefault("wagers", {})[idx] = 0
            bits.append(
                f"{player['name']}, you're at {fmt(score)} — you ride along "
                "for pride. "
            )
            queue.pop(0)
            continue
        bits.append(
            f"{player['name']}, you have {fmt(score)} — what's your wager?"
        )
        return ("".join(bits), False)
    return _jeopardy_read_final_clue("".join(bits))


def _jeopardy_handle_final_wager(text: str, person_id: Optional[int]) -> tuple[str, bool]:
    try:
        from features import jeopardy as jeopardy_bank
    except Exception:
        return ("My final-round circuits glitched. Say your wager again.", False)
    players = _game_state.get("players") or []
    final = _game_state.get("final") or {}
    queue = _game_state.get("final_queue") or []
    if not queue or not players:
        return _jeopardy_read_final_clue("")

    idx = int(queue[0])
    player = players[idx]
    score = int(player.get("score", 0) or 0)

    meta = _jeopardy_answer_board_question(text)
    if meta is not None:
        return (
            f"{meta}{player['name']}, what's your wager — zero to ${score}?",
            False,
        )

    wager = jeopardy_bank.parse_wager(text, min_wager=0, max_wager=score)
    if wager is None:
        return (
            f"A number, {player['name']} — anything from zero to ${score}.",
            False,
        )
    if wager < 0 or wager > score:
        return (
            f"${wager} is outside the rails. Zero to ${score}, "
            f"{player['name']} — what's your wager?",
            False,
        )
    final.setdefault("wagers", {})[idx] = int(wager)
    queue.pop(0)
    return _jeopardy_next_final_wager_prompt(
        f"${wager} locked for {player['name']}. "
    )


def _jeopardy_read_final_clue(prefix: str) -> tuple[str, bool]:
    final = _game_state.get("final") or {}
    order = list(final.get("order") or [])
    players = _game_state.get("players") or []
    clue = final.get("clue") or {}
    _game_state["phase"] = "final_answer"
    _game_state["final_queue"] = list(order)
    _game_state["pending_after_response_clip"] = "final_theme"
    first = players[int(order[0])]["name"] if order and players else "Player"
    return (
        f"{prefix}Wagers are locked. Here is your clue: {clue.get('clue')}. "
        f"Think it over while the music plays — {first}, your answer first.",
        False,
    )


def _jeopardy_handle_final_answer(text: str, person_id: Optional[int]) -> tuple[str, bool]:
    try:
        from features import jeopardy as jeopardy_bank
    except Exception:
        jeopardy_bank = None
    players = _game_state.get("players") or []
    final = _game_state.get("final") or {}
    queue = _game_state.get("final_queue") or []
    clue = final.get("clue") or {}
    if not queue or not players or jeopardy_bank is None:
        return _jeopardy_reveal_final()

    if jeopardy_bank.is_clue_repeat_request(text):
        current = players[int(queue[0])]["name"]
        return (
            f"Once more. The category: "
            f"{_jeopardy_speak_category(clue.get('category'))}. "
            f"Clue: {clue.get('clue')}. {current}, your answer?",
            False,
        )

    idx = int(queue[0])
    correct, passed = _jeopardy_grade(text, clue)
    final.setdefault("answers", {})[idx] = {
        "said": "no answer" if (passed and not correct) else text,
        "correct": bool(correct),
    }
    queue.pop(0)
    if queue:
        nxt = players[int(queue[0])]["name"]
        return (f"Locked in. {nxt}, your answer?", False)
    return _jeopardy_reveal_final()


def _jeopardy_reveal_final() -> tuple[str, bool]:
    """Score the wagers, reveal the response, crown a winner, end the game."""
    players = _game_state.get("players") or []
    final = _game_state.get("final") or {}
    clue = final.get("clue") or {}
    correct_response = _jeopardy_correct_response_text({
        "answer": clue.get("answer", "unknown"),
        "clue": clue.get("clue", ""),
        "category": clue.get("category", ""),
    })
    bits = [f"Time to settle up. The correct response was: {correct_response}. "]
    for idx in final.get("order") or range(len(players)):
        idx = int(idx)
        if idx >= len(players):
            continue
        player = players[idx]
        entry = (final.get("answers") or {}).get(idx)
        wager = int((final.get("wagers") or {}).get(idx, 0) or 0)
        if entry is None:
            continue
        if entry.get("correct"):
            player["score"] = int(player.get("score", 0) or 0) + wager
            bits.append(
                f"{player['name']} had it — plus ${wager}. " if wager
                else f"{player['name']} had it right, for the honor alone. "
            )
        else:
            player["score"] = int(player.get("score", 0) or 0) - wager
            bits.append(
                f"{player['name']} — no, minus ${wager}. " if wager
                else f"{player['name']} — no, but nothing lost. "
            )
    try:
        from features import jeopardy as jeopardy_bank
        scores = jeopardy_bank.format_scores(players)
    except Exception:
        scores = "unavailable"
    top = max((int(p.get("score", 0) or 0) for p in players), default=0)
    winners = [p["name"] for p in players if int(p.get("score", 0) or 0) == top]
    if len(winners) == 1:
        crown = f"{winners[0]} takes the game."
    else:
        crown = f"A tie between {' and '.join(winners)}. The rematch writes itself."
    _jeopardy_queue_clip("outro")
    return (f"{''.join(bits)}Final scores: {scores}. {crown}", True)


def _jeopardy_answer_in_flight() -> bool:
    """True when a player is speaking right now, or a just-finished utterance is
    still being processed — an answer is in the pipe and the timer must not
    steal the turn out from under it (field 2026-08-25: "Floral" was spoken as
    the time's-up beeper fired, the rebound had already advanced the turn, and
    the $1000 went to the wrong player)."""
    try:
        from awareness.situation import assessor
        return bool(assessor.is_user_speaking() or assessor.is_interaction_busy())
    except Exception:
        return False


def _jeopardy_timeout_fired(token: str) -> None:
    line = ""
    queue_timesup = True
    schedule_rebound = False
    rebound_at = 0.0
    with _lock:
        if _active_game != "jeopardy":
            return
        if _game_state.get("phase") != "awaiting_answer":
            return
        if _game_state.get("answer_timer_token") != token:
            return
        clue = dict(_game_state.get("current_clue") or {})
        if not clue:
            return
        # A deferral is a courtesy to ONE in-flight answer, not an open-ended
        # hold. `deadline` is the absolute ceiling set when the clock was armed;
        # past it the clue times out even if the room is still talking. Field
        # 2026-08-26 20:20:42-20:21:13: thirteen back-to-back deferrals, 31 s
        # past a 12 s clock, because a five-person room with a barking dog kept
        # is_user_speaking()/is_interaction_busy() true forever — the table's
        # verdict was "it should have timed out".
        deadline = float(_game_state.get("answer_timer_deadline") or 0.0)
        if _jeopardy_answer_in_flight() and (deadline <= 0.0 or time.monotonic() < deadline):
            # Give the in-flight utterance a beat to land: it will cancel this
            # timer when it grades. Re-arm with the SAME token so a stale defer
            # can never outlive a legitimate re-arm.
            grace = float(getattr(config, "JEOPARDY_TIMEOUT_SPEECH_GRACE_SECS", 2.5))
            if deadline > 0.0:
                grace = min(grace, max(0.25, deadline - time.monotonic()))
            timer = threading.Timer(grace, _jeopardy_timeout_fired, args=(token,))
            timer.daemon = True
            _game_state["answer_timer"] = timer
            timer.start()
            _log.info(
                "[jeopardy] answer timeout deferred %.1fs — player speech in flight "
                "(%.1fs left before the hard ceiling)",
                grace, max(0.0, deadline - time.monotonic()) if deadline > 0.0 else -1.0,
            )
            return
        if _jeopardy_answer_in_flight():
            _log.info(
                "[jeopardy] answer timeout FIRING through in-flight speech — hit the "
                "%.1fs deferral ceiling",
                float(getattr(config, "JEOPARDY_TIMEOUT_MAX_DEFER_SECS", 10.0)),
            )
        _game_state.pop("answer_timer_deadline", None)
        _game_state.pop("answer_timer", None)
        _game_state.pop("answer_timer_token", None)
        correct_response = _jeopardy_correct_response_text(clue)
        timed_out_idx = int(_game_state.get("current_player_idx", 0))
        # A Daily Double belongs to its picker alone — no rebound (show rules).
        next_player = None if clue.get("daily_double") else _jeopardy_offer_rebound()
        if next_player:
            line = _jeopardy_rebound_prompt("Time's up. ", next_player, clue)
            schedule_rebound = True
            # Until the rebound announcement finishes, an incoming answer belongs
            # to the player whose time just ran out — they were mid-thought at
            # the beeper, not the player who has not even heard the re-read yet.
            rebound_at = time.monotonic()
            _game_state["timeout_rebound"] = {
                "from_idx": timed_out_idx,
                "at": rebound_at,
            }
        else:
            done = int((_game_state.get("board") or {}).get("remaining", 0) or 0) <= 0
            line, game_done = _jeopardy_finish_missed_clue(
                "Time's up. ",
                correct_response,
                done=done,
                players=_game_state.get("players") or [{"name": "Player", "score": 0}],
                score_line=False,
            )
            if game_done:
                _clear_game()

    if not line:
        return
    try:
        from audio import speech_queue
        from intelligence import llm
        if queue_timesup:
            _body_beat("dramatic_visor_peek")
            _jeopardy_queue_clip("timesup")
        done_event = speech_queue.enqueue(
            llm.clean_response_text(line),
            priority=1,
            tag="jeopardy:timeout",
        )
        if schedule_rebound:
            _jeopardy_schedule_post_timeout_rebound(done_event, rebound_at=rebound_at)
    except Exception as exc:
        _log.debug("[jeopardy] timeout speech failed: %s", exc)


def _jeopardy_arm_timeout() -> None:
    if _active_game != "jeopardy":
        return
    if _game_state.get("phase") != "awaiting_answer":
        return
    if not _game_state.pop("awaiting_prompt_delivery", False):
        return
    # The rebound announcement is fully out — from here an answer is the rebound
    # player's, not a late grace answer from whoever timed out.
    _game_state.pop("timeout_rebound", None)
    timeout = float(getattr(config, "JEOPARDY_ANSWER_TIMEOUT_SECS", 14.0))
    if timeout <= 0:
        return
    token = f"{time.monotonic():.6f}:{random.random():.6f}"
    timer = threading.Timer(timeout, _jeopardy_timeout_fired, args=(token,))
    timer.daemon = True
    _game_state["answer_timer_token"] = token
    _game_state["answer_timer"] = timer
    # Absolute ceiling for the whole clock INCLUDING speech-in-flight deferrals.
    _game_state["answer_timer_deadline"] = time.monotonic() + timeout + max(
        0.0, float(getattr(config, "JEOPARDY_TIMEOUT_MAX_DEFER_SECS", 10.0))
    )
    timer.start()


def _jeopardy_start(person_id: Optional[int]) -> str:
    speaker_name = _jeopardy_person_name(person_id)
    _jeopardy_queue_clip("intro")
    _body_beat("proud_dj_pose")
    _game_state.update({
        "phase": "awaiting_players",
        "speaker_name": speaker_name,
    })
    return _rex_respond(
        "[GAME: Jeopardy — PLAYER SETUP] Rex is starting a verbal Jeopardy-style "
        "game. Ask who is playing. Tell them they can say one to four names "
        "in one reply. Make it feel like a game-show intro, but keep it brief.",
        person_id,
    )


def _jeopardy_begin_board_for_players(players: list[dict], person_id: Optional[int]) -> tuple[str, bool]:
    round_line = _jeopardy_load_round(1, players, current_player_idx=0)
    if not round_line:
        _game_state.clear()
        return (
            _rex_respond(
                "[GAME: Jeopardy — NO BOARD] Rex tried to start Jeopardy but no "
                "playable round-one clue board was available. Apologize in "
                "character and suggest Trivia instead.",
                person_id,
            ),
            True,
        )

    player_text = ", ".join(str(p.get("name") or "Player") for p in players)
    quip = random.choice([
        "Try not to make the scoreboard file a complaint.",
        "May your answers be less questionable than my wiring.",
        "Brains armed, dignity optional.",
    ])
    return (
        f"Contestants logged: {player_text}. {quip} "
        f"{round_line}",
        False,
    )


def _jeopardy_begin_board(names: list[str], person_id: Optional[int]) -> tuple[str, bool]:
    players, needs_voice = _jeopardy_prepare_players(names)
    if needs_voice:
        _game_state.update({
            "phase": "voice_enroll",
            "players": players,
            "voice_enroll_queue": needs_voice,
        })
        return (_jeopardy_voice_check_prompt(players[needs_voice[0]]), False)
    return _jeopardy_begin_board_for_players(players, person_id)


def _jeopardy_handle_player_setup(text: str, person_id: Optional[int]) -> tuple[str, bool]:
    try:
        from features import jeopardy as jeopardy_bank
        max_players = int(getattr(config, "JEOPARDY_MAX_PLAYERS", 3))
        names = jeopardy_bank.parse_player_names(
            text,
            speaker_name=_game_state.get("speaker_name"),
            limit=max_players,
        )
    except Exception as exc:
        _log.debug("[jeopardy] player parse failed: %s", exc)
        names = []

    if not names:
        return (
            "I need actual player names, not mysterious cantina fog. Say something like 'Bret, Joy, Daniel, and Jen'.",
            False,
        )

    return _jeopardy_begin_board(names, person_id)


def _jeopardy_handle_selection(text: str, person_id: Optional[int]) -> tuple[str, bool]:
    try:
        from features import jeopardy as jeopardy_bank
        board = _game_state.get("board") or {}
        # Board QUESTIONS come before pick parsing: "is the 400 still there in
        # history?" carries a value, and the value-wins pick rule would consume
        # the square instead of answering (owner ask 2026-08-25).
        meta = _jeopardy_answer_board_question(text)
        if meta is not None:
            player = _jeopardy_current_player()
            return (
                f"{meta}{player['name']}, pick a category and dollar value.",
                False,
            )
        # The categories are announced once when the round loads. A voice-only
        # player who missed them had no way to get them back — the old path fell
        # through to "pick a dollar value too", which reads as being ignored.
        if jeopardy_bank.is_board_request(text):
            readout = _jeopardy_board_text()
            player = _jeopardy_current_player()
            if not readout:
                return ("The board is picked clean. Nothing left to read back.", False)
            return (
                f"Still on the board: {readout}. "
                f"{player['name']}, pick a category and dollar value.",
                False,
            )
        # "Next round" / "final jeopardy" — the table votes for a fresh board
        # or the endgame instead of grinding all thirty clues.
        jump = jeopardy_bank.round_jump_request(text)
        if jump is not None:
            return _jeopardy_round_jump(jump)
        pending_category = _game_state.pop("pending_category", None)
        clue, error = jeopardy_bank.parse_selection(
            text,
            board,
            last_category=pending_category or _game_state.get("last_category"),
        )
        if not clue:
            # Remember the category a FAILED pick named, so the bare value that
            # usually follows ("Pop culture for 300" → "no $300 square" → "400")
            # completes THAT category — not the last one played (field
            # 2026-08-25 18:50: the bare "400" picked BIBLICAL PEOPLE instead
            # of the Pop Culture the table had just asked for).
            hint = jeopardy_bank.selection_category_hint(text, board)
            if hint:
                _game_state["pending_category"] = hint
            elif pending_category:
                # The retry named nothing either (another bare value) — the
                # remembered category stays live for the next attempt.
                _game_state["pending_category"] = pending_category
    except Exception as exc:
        _log.error("[jeopardy] selection parse failed: %s", exc)
        clue, error = None, "My board parser fell into a reactor shaft. Try the category and value again."

    if not clue:
        # A question the deterministic lanes did not recognize gets one LLM look
        # with the real board in context, instead of "pick a dollar value too".
        fallback = _jeopardy_board_question_llm(text, person_id)
        if fallback is not None:
            return (fallback, False)
        aside = _jeopardy_table_talk_aside(text, person_id)
        if aside is not None:
            return (aside, False)
        return (error, False)

    player = _jeopardy_current_player()
    daily = bool(clue.get("daily_double"))
    if daily and bool(getattr(config, "JEOPARDY_DD_WAGER_ENABLED", True)):
        return _jeopardy_begin_daily_double(clue, player)

    effective_value = int(clue.get("value", 0) or 0)
    if daily:
        # Legacy no-wager mode (JEOPARDY_DD_WAGER_ENABLED=False): flat double.
        effective_value *= 2
        _jeopardy_queue_clip("daily_double")
        _body_beat("dramatic_visor_peek")
    else:
        _body_beat("thinking_tilt")

    clue["effective_value"] = effective_value
    _game_state.update({
        "phase": "awaiting_answer",
        "current_clue": clue,
        "current_clue_attempts": [],
        "last_category": clue.get("category"),
        "awaiting_prompt_delivery": True,
    })
    if bool(getattr(config, "JEOPARDY_PLAY_THINKING_THEME", False)):
        _game_state["pending_after_response_clip"] = "theme"

    daily_line = "Daily Double. Automatic double. " if daily else ""
    return (
        f"{daily_line}{player['name']}, "
        f"{jeopardy_bank.speak_category(clue.get('category') or '')} for ${clue.get('value')}. "
        f"Clue: {clue.get('clue')}.",
        False,
    )


def _jeopardy_max_wager(score: int) -> int:
    """Show rules: wager up to your score, floored at the round's top value."""
    current_round = int(_game_state.get("jeopardy_round", 1) or 1)
    return max(int(score), 1000 * max(1, current_round))


def _jeopardy_begin_daily_double(clue: dict, player: dict) -> tuple[str, bool]:
    """A Daily Double square: sting, then ask for the wager before the clue."""
    _jeopardy_queue_clip("daily_double")
    _body_beat("dramatic_visor_peek")
    _game_state.update({
        "phase": "awaiting_wager",
        "current_clue": clue,
        "current_clue_attempts": [],
        "last_category": clue.get("category"),
    })
    score = int(player.get("score", 0) or 0)
    min_wager = int(getattr(config, "JEOPARDY_DD_MIN_WAGER", 5))
    max_wager = _jeopardy_max_wager(score)
    try:
        from features import jeopardy as jeopardy_bank
        score_text = jeopardy_bank.format_score(score)
    except Exception:
        score_text = f"${score}"
    return (
        f"Daily Double! {player['name']}, you're at {score_text}. "
        f"Wager anything from ${min_wager} to ${max_wager}. What's your wager?",
        False,
    )


def _jeopardy_handle_wager(text: str, person_id: Optional[int]) -> tuple[str, bool]:
    """Consume the Daily Double wager, then read the clue for it."""
    try:
        from features import jeopardy as jeopardy_bank
    except Exception:
        jeopardy_bank = None
    clue = dict(_game_state.get("current_clue") or {})
    if not clue or jeopardy_bank is None:
        _game_state["phase"] = "selecting"
        return ("I lost the Daily Double in a reactor shaft. Pick another square.", False)

    player = _jeopardy_current_player()
    score = int(player.get("score", 0) or 0)
    min_wager = int(getattr(config, "JEOPARDY_DD_MIN_WAGER", 5))
    max_wager = _jeopardy_max_wager(score)

    # "What's the score?" is a fair thing to check before wagering.
    meta = _jeopardy_answer_board_question(text)
    if meta is not None:
        return (
            f"{meta}{player['name']}, what's your wager — ${min_wager} to ${max_wager}?",
            False,
        )

    wager = jeopardy_bank.parse_wager(text, min_wager=min_wager, max_wager=max_wager)
    if wager is None:
        return (
            f"Give me a number, {player['name']} — anything from "
            f"${min_wager} to ${max_wager}.",
            False,
        )
    if wager < min_wager or wager > max_wager:
        return (
            f"${wager} is outside the rails. ${min_wager} to ${max_wager} — "
            "what's your wager?",
            False,
        )

    clue["effective_value"] = int(wager)
    _game_state.update({
        "current_clue": clue,
        "phase": "awaiting_answer",
        "awaiting_prompt_delivery": True,
    })
    if bool(getattr(config, "JEOPARDY_PLAY_THINKING_THEME", False)):
        _game_state["pending_after_response_clip"] = "theme"
    _body_beat("thinking_tilt")
    return (
        f"${wager} on the line. "
        f"{_jeopardy_speak_category(clue.get('category'))}. Clue: {clue.get('clue')}.",
        False,
    )


def _jeopardy_grade(text: str, clue: dict) -> tuple[bool, bool]:
    """(correct, passed) — the full grading ladder used everywhere an answer is
    scored. A hedge is a LEAD-IN when a real answer follows it ("no idea, maybe
    Lincoln"): the residual can only PROMOTE to correct, never demote a shrug
    into a deduction (routing audit 2026-08-13). The strict LLM judge gets one
    look at a borderline miss, and is deliberately NOT re-run on the residual —
    a hedged answer the lexical matcher cannot score fails safe to a pass."""
    try:
        from features import jeopardy as jeopardy_bank
    except Exception:
        return (False, False)
    answer = clue.get("answer", "unknown")
    passed = bool(jeopardy_bank.is_pass_or_timeout(text))
    correct = bool(jeopardy_bank.is_correct(text, answer))
    if passed and not correct:
        residual = jeopardy_bank.strip_pass_hedge(text)
        if residual and jeopardy_bank.is_correct(residual, answer):
            correct = True
    if not correct and not passed:
        correct = _jeopardy_llm_judge(text, answer, clue)
    return correct, passed


def _jeopardy_last_judge_said_not_an_answer(text: str, expected_answer: str) -> bool:
    """True when the judge call _jeopardy_grade just made ruled "not an answer"."""
    key = ((text or "").strip(), str(expected_answer or ""))
    return (
        _LAST_JUDGE_VERDICT.get("key") == key
        and _LAST_JUDGE_VERDICT.get("verdict") == "not_an_answer"
    )


def _jeopardy_ignore_non_answer(text: str, clue: dict) -> Optional[str]:
    """Reason to score NOTHING for this utterance, or None to grade it.

    Deterministic lanes only — the LLM's "none" verdict is read back after the
    grading ladder has already paid for its one call.
    """
    if not bool(getattr(config, "JEOPARDY_IGNORE_NON_ANSWERS", True)):
        return None
    try:
        from features import jeopardy as jeopardy_bank
    except Exception:
        return None
    if jeopardy_bank.is_bare_question_stem(text):
        # A truncated answer, not a guess: the player got "What is" out and the
        # endpointer closed on their thinking pause. Say nothing and let the
        # rest of the sentence arrive as the next segment (field 2026-08-26
        # 20:13:45 — a bare "What is?" was graded as a miss and the real answer
        # then landed on the rebound player).
        return "bare question stem"
    if jeopardy_bank.is_too_long_for_an_answer(text):
        # Past the length any Jeopardy response reaches. The rescue judge
        # already refused to rule on these; the miss simply stood (field
        # 2026-08-26 20:21:13 — a 230-character complaint about the game cost
        # T'Joy $100).
        return "too long to be an answer"
    names = [str((p or {}).get("name") or "") for p in (_game_state.get("players") or [])]
    reason = jeopardy_bank.is_addressed_elsewhere(text, names)
    # Guarded by is_correct: a clue's answer really can be a player's name, and
    # a right answer must never be swallowed by a chatter pattern.
    if reason and not jeopardy_bank.is_correct(text, str(clue.get("answer") or "")):
        return reason
    return None


def _jeopardy_handle_answer(text: str, person_id: Optional[int]) -> tuple[str, bool]:
    try:
        from features import jeopardy as jeopardy_bank
    except Exception as exc:
        _log.error("[jeopardy] answer helper import failed: %s", exc)
        jeopardy_bank = None

    clue = dict(_game_state.get("current_clue") or {})
    if not clue:
        _jeopardy_cancel_timeout()
        _game_state["phase"] = "selecting"
        return ("I lost the clue state. Pick another square before I blame a power converter.", False)

    # Not every noise in the room is a response. Checked BEFORE the clock is
    # cancelled so an ignored turn leaves the live timer running untouched —
    # the square stays open, nobody is charged, and Rex says nothing rather
    # than talking over a side conversation.
    ignore_reason = _jeopardy_ignore_non_answer(text, clue)
    if ignore_reason is not None:
        _log.info("[jeopardy] ignoring %r — %s (clock still running)", text, ignore_reason)
        return ("", False)

    _jeopardy_cancel_timeout()

    players = _game_state.get("players") or [{"name": "Player", "score": 0}]
    # TIMEOUT-REBOUND GRACE: this answer landed after the time's-up beeper but
    # before Rex finished announcing whose turn it is next. That is the timed-out
    # player getting their answer out a beat late — not the rebound player, who
    # has not even heard the clue re-read (field 2026-08-25: "What is Nike" and
    # "Floral" both scored for the WRONG player this way). Grade it for the
    # player whose time ran out, and drop the now-moot rebound announcement.
    grace = _game_state.get("timeout_rebound")
    if grace is not None and _game_state.get("awaiting_prompt_delivery"):
        _game_state.pop("timeout_rebound", None)
        from_idx = int(grace.get("from_idx", 0)) % len(players)
        _game_state["current_player_idx"] = from_idx
        try:
            from audio import speech_queue
            dropped = speech_queue.drop_by_tag("jeopardy:timeout")
        except Exception:
            dropped = 0
        _log.info(
            "[jeopardy] grace answer during rebound announce — grading for %s "
            "(dropped %d queued timeout line[s])",
            players[from_idx].get("name"), dropped,
        )
    idx = int(_game_state.get("current_player_idx", 0)) % len(players)
    player = players[idx]
    answer = clue.get("answer", "unknown")
    value = int(clue.get("effective_value", clue.get("value", 0)) or 0)
    correct_response = _jeopardy_correct_response_text(clue)

    # "Say that again" is a request, not a guess. These shapes are never a valid
    # "What is X?" response, so they are safe to intercept before scoring.
    if jeopardy_bank is not None and jeopardy_bank.is_clue_repeat_request(text):
        return (_jeopardy_repeat_clue_reply(clue, player, prefix="Once more. "), False)

    correct, passed = _jeopardy_grade(text, clue)

    # The judge's third verdict, from the call _jeopardy_grade just made: the
    # player was not answering at all. Re-arm the window and stay silent — the
    # deterministic lanes above cannot spot "Come here, Toby" said to a dog.
    if (
        not correct
        and not passed
        and bool(getattr(config, "JEOPARDY_IGNORE_NON_ANSWERS", True))
        and _jeopardy_last_judge_said_not_an_answer(text, answer)
    ):
        # Unlike the deterministic lanes above (which leave the LIVE timer
        # running), this branch re-arms a fresh clock — so a room that keeps
        # talking could walk the deadline forward forever, which is the exact
        # complaint the table voiced. Cap the streak and settle the square.
        ignored = int(_game_state.get("ignored_turns", 0) or 0) + 1
        _game_state["ignored_turns"] = ignored
        cap = int(getattr(config, "JEOPARDY_IGNORE_STREAK_CAP", 4))
        if cap <= 0 or ignored < cap:
            _log.info(
                "[jeopardy] ignoring %r — judge says it was not an answer (%d/%d)",
                text, ignored, cap,
            )
            _game_state["phase"] = "awaiting_answer"
            # Re-arming the window would otherwise re-open the timeout-rebound
            # grace (field 2026-08-25) — nobody timed out here, so close it.
            _game_state.pop("timeout_rebound", None)
            _game_state["awaiting_prompt_delivery"] = True
            return ("", False)
        _log.info(
            "[jeopardy] %d non-answers in a row on this clue — settling it as a "
            "no-answer instead of holding the square open", ignored,
        )
        _game_state.pop("ignored_turns", None)
        passed = True    # falls into the no-answer branch below

    # "What are the categories?" / "what's left in pop culture?" / "what's the
    # score?" asked a beat late — questions, not wrong answers. Checked only
    # AFTER is_correct and the judge both said no, so they can never swallow a
    # legitimate response like "What is the board of directors?". Each answers
    # the question, then re-reads the live clue and restarts the window.
    if not correct and not passed and jeopardy_bank is not None:
        meta = _jeopardy_answer_board_question(text)
        if meta is not None:
            return (
                _jeopardy_repeat_clue_reply(
                    clue, player, prefix=f"{meta}Back to your square. "
                ),
                False,
            )
        if jeopardy_bank.is_board_request(text):
            readout = _jeopardy_board_text()
            prefix = f"Still on the board: {readout}. Now, back to your square. " if readout else ""
            return (_jeopardy_repeat_clue_reply(clue, player, prefix=prefix), False)

    done = int((_game_state.get("board") or {}).get("remaining", 0) or 0) <= 0
    # `and not correct`: a hedged RIGHT answer is right. Pass used to win this
    # branch outright, which is how "I don't know, Paris?" lost a correct Paris even
    # though is_correct had already said True (routing audit 2026-08-13).
    if passed and not correct:
        _body_beat("suspicious_glance")
        _jeopardy_queue_clip("timesup")
        # A Daily Double belongs to its picker alone — no rebound (show rules).
        next_player = None if clue.get("daily_double") else _jeopardy_offer_rebound()
        if next_player:
            return (_jeopardy_rebound_prompt("No answer. ", next_player, clue), False)
        return _jeopardy_finish_missed_clue(
            "No answer. ",
            correct_response,
            done=done,
            players=players,
            score_line=False,
        )

    if correct:
        _body_beat("tiny_victory_dance")
        player["score"] = int(player.get("score", 0)) + value
        _game_state.pop("current_clue", None)
        _game_state.pop("current_clue_attempts", None)
        _game_state.pop("ignored_turns", None)
        _jeopardy_queue_clip("right")
        flourish = random.choice([
            "Correct. The organics survive another clue.",
            "Correct. I am marking this as suspiciously competent.",
            "Correct. The scoreboard briefly respects you.",
        ])
        if done:
            return _jeopardy_complete_round_or_finish(f"{flourish} ")
        _game_state["phase"] = "selecting"
        scores_line = _jeopardy_score_announcement(player)
        categories = _jeopardy_categories_reminder()
        return (
            f"{flourish} ${value} to {player['name']}. {scores_line}"
            f"{categories}{_jeopardy_maybe_offer_round_jump()}"
            f"{player['name']}, pick the next category and value.",
            False,
        )

    # WHOSE money is on the line. A wrong answer only COSTS the current player
    # when it could plausibly be theirs — the speaker is unresolved (the common
    # case) or resolves to them. A confident OTHER contestant shouting a guess
    # is the room helping out, and this table plays that way: a right answer
    # from a helper still counts, a wrong one is not billed to whoever's turn it
    # happens to be (field 2026-08-26: PJ calling the dog took $400 off Bret).
    # Credits are deliberately unchanged — asymmetric on purpose.
    _body_beat("offended_recoil")
    helper = (
        _jeopardy_confident_other_speaker(person_id)
        if bool(getattr(config, "JEOPARDY_ONLY_CHARGE_THE_ANSWERER", True))
        else None
    )
    if helper is not None:
        _log.info(
            "[jeopardy] wrong answer came from %s, not %s — no deduction",
            helper.get("name"), player.get("name"),
        )
        _jeopardy_queue_clip("wrong")
        heckle = random.choice([
            f"{helper['name']}, that's not your square, and it wasn't right either.",
            f"Wrong, {helper['name']} — and it's not even your turn. No charge.",
            f"Rejected, {helper['name']}. {player['name']} keeps the money.",
        ])
        # The square stays with its owner: nobody was charged, so nothing was
        # spent. Rebounding here would let a heckler take the current player's
        # square for free AND burn their single JEOPARDY_MAX_REBOUNDS chance.
        # _jeopardy_repeat_clue_reply re-arms phase + awaiting_prompt_delivery,
        # so on_response_spoken() restarts the clock through the normal path.
        return (
            _jeopardy_repeat_clue_reply(clue, player, prefix=f"{heckle} "),
            False,
        )

    player["score"] = int(player.get("score", 0)) - value
    _jeopardy_queue_clip("wrong")
    roast = random.choice([
        "A bold miss.",
        "The board accepts your sacrifice.",
        "That answer landed somewhere near Alderaan.",
    ])
    # A Daily Double belongs to its picker alone — no rebound (show rules).
    next_player = None if clue.get("daily_double") else _jeopardy_offer_rebound()
    if next_player:
        return (
            _jeopardy_rebound_prompt(
                f"{roast} ${value} off {player['name']}. ",
                next_player,
                clue,
            ),
            False,
        )

    return _jeopardy_finish_missed_clue(
        # Name the deduction here too. This is the arm a Daily Double ALWAYS
        # takes (no rebound), and the wager was otherwise never spoken aloud —
        # field 2026-08-26 20:26:19: a $200 DD loss announced only the new total.
        f"{roast} ${value} off {player['name']}. ",
        correct_response,
        done=done,
        players=players,
        score_player=player,
    )


def _jeopardy_handle_voice_enroll(
    text: str,
    person_id: Optional[int],
    audio_array=None,
) -> tuple[str, bool]:
    players = _game_state.get("players") or []
    queue = list(_game_state.get("voice_enroll_queue") or [])
    if not players or not queue:
        return _jeopardy_begin_board_for_players(players or [{"name": "Player", "score": 0}], person_id)

    player_idx = int(queue.pop(0))
    player = players[player_idx]
    name = str(player.get("name") or "player")
    normalized = " ".join((text or "").lower().split())
    skipped = (
        normalized in {"skip", "skip voice", "start anyway", "begin anyway", "play anyway"}
        or normalized == f"skip {name.lower()}"
    )

    prefix = ""
    if skipped:
        prefix = f"Skipping {name}'s voice print. "
    else:
        pid = player.get("person_id")
        if pid is None or audio_array is None:
            queue.insert(0, player_idx)
            _game_state["voice_enroll_queue"] = queue
            return (_jeopardy_voice_check_prompt(player, prefix="I could not store that one. "), False)
        try:
            from audio import speaker_id
            ok = speaker_id.enroll_voice(int(pid), audio_array)
        except Exception as exc:
            _log.debug("[jeopardy] voice enrollment failed for %s: %s", name, exc)
            ok = False
        if not ok:
            queue.insert(0, player_idx)
            _game_state["voice_enroll_queue"] = queue
            return (_jeopardy_voice_check_prompt(player, prefix="That voice print was too fuzzy. "), False)
        player["voice_enrolled"] = True
        prefix = f"Voice print stored for {name}. "

    _game_state["voice_enroll_queue"] = queue
    if queue:
        return (_jeopardy_voice_check_prompt(players[int(queue[0])], prefix=prefix), False)
    return _jeopardy_begin_board_for_players(players, person_id)


def _jeopardy_handle(text: str, person_id: Optional[int], audio_array=None) -> tuple[str, bool]:
    phase = _game_state.get("phase")
    if phase == "awaiting_wager":
        return _jeopardy_handle_wager(text, person_id)
    if phase == "final_wager":
        return _jeopardy_handle_final_wager(text, person_id)
    if phase == "final_answer":
        return _jeopardy_handle_final_answer(text, person_id)
    if phase == "awaiting_players":
        return _jeopardy_handle_player_setup(text, person_id)
    if phase == "voice_enroll":
        return _jeopardy_handle_voice_enroll(text, person_id, audio_array)
    if phase == "selecting":
        return _jeopardy_handle_selection(text, person_id)
    if phase == "awaiting_answer":
        return _jeopardy_handle_answer(text, person_id)
    _game_state.clear()
    return ("Jeopardy state went sideways. Game over before the lawyers arrive.", True)


def _jeopardy_stop(person_id: Optional[int]) -> str:
    try:
        from features import jeopardy as jeopardy_bank
        scores = jeopardy_bank.format_scores(_game_state.get("players") or [])
    except Exception:
        scores = "scores unavailable"
    _jeopardy_cancel_timeout()
    _jeopardy_queue_clip("outro")
    _game_state.clear()
    return (
        f"Jeopardy stopped. Final scores: {scores}. "
        "A merciful ending for everyone with a central nervous system."
    )


# ── Word Association game ─────────────────────────────────────────────────────

_WORD_ASSOC_STARTERS = [
    "cantina", "hyperspace", "droid", "parsec", "galaxy",
    "credits", "blaster", "Batuu", "starship", "protocol",
    "asteroid", "binary", "reactor", "wookiee", "hangar",
]


def _wordassoc_start(person_id: Optional[int]) -> str:
    first_word = random.choice(_WORD_ASSOC_STARTERS)
    _game_state.update({
        "last_word": first_word,
        "chain": [first_word],
        "turn_count": 0,
    })
    _body_beat("thinking_tilt")

    return _rex_respond(
        f"[GAME: Word Association — START] Rex is starting Word Association. "
        f"Explain the rules briefly: Rex says a word, player responds with an "
        f"associated word, back and forth — Rex calls any breaks in logic. "
        f"Rex's opening word is \"{first_word}\". Deliver it with flair.",
        person_id,
    )


def _wordassoc_handle(text: str, person_id: Optional[int]) -> tuple[str, bool]:
    last_word = _game_state.get("last_word", "")
    chain: list = _game_state.get("chain", [])
    turn_count = _game_state.get("turn_count", 0)

    # Extract just the first word from the player's input
    player_word = text.strip().split()[0].strip(".,!?\"'").lower() if text.strip() else ""

    if not player_word:
        return (
            _rex_respond(
                "[GAME: Word Association] Player didn't give a word. "
                "Rex prompts them to say one — brief, impatient.",
                person_id,
            ),
            False,
        )

    # Validate the association
    valid_raw = _quick_call(
        f'Word Association game. Previous word: "{last_word}". '
        f'Player responded: "{player_word}". '
        f"Is this a reasonable word association? Answer ONLY: yes or no.",
        temperature=0,
        max_tokens=5,
    ).strip().lower()

    is_valid = "yes" in valid_raw

    turn_count += 1
    _game_state["turn_count"] = turn_count

    if not is_valid:
        chain_str = " → ".join(chain + [player_word])
        _body_beat("suspicious_glance")
        _game_state.clear()
        return (
            _rex_respond(
                f"[GAME: Word Association — BREAK CALLED] Rex calls a break in logic. "
                f"Previous word was \"{last_word}\", player said \"{player_word}\" — "
                f"not a valid association. Game over. Chain length: {len(chain)} words. "
                f"Rex is characteristically smug. Full chain: {chain_str}",
                person_id,
            ),
            True,
        )

    # Valid — Rex generates the next word
    next_word_raw = _quick_call(
        f'Word Association. Chain so far: {" → ".join(chain + [player_word])}. '
        f"Give ONE word that naturally follows \"{player_word}\". "
        f"Favor space, sci-fi, or playful words when possible. "
        f"Return ONLY the single word, no punctuation.",
        temperature=0.8,
        max_tokens=10,
    ).strip().split()[0].strip(".,!?\"'").lower()

    next_word = next_word_raw if next_word_raw else "systems"

    chain.append(player_word)
    chain.append(next_word)
    _game_state["last_word"] = next_word
    _game_state["chain"] = chain
    _game_state["turn_count"] = turn_count

    return (
        _rex_respond(
            f"[GAME: Word Association — VALID TURN #{turn_count}] "
            f"Player said \"{player_word}\" following \"{last_word}\" — valid. "
            f"Rex's next word is \"{next_word}\". "
            f"Rex delivers it: brief, punchy, in character.",
            person_id,
        ),
        False,
    )


def _wordassoc_stop(person_id: Optional[int]) -> str:
    chain = _game_state.get("chain", [])
    turn_count = _game_state.get("turn_count", 0)
    _game_state.clear()
    return _rex_respond(
        f"[GAME: Word Association — STOPPED] Game ended after {turn_count} turns, "
        f"{len(chain)} words in the chain. Rex delivers a brief in-character close.",
        person_id,
    )


# ── Game dispatch table ───────────────────────────────────────────────────────

_GAME_HANDLERS: dict[str, dict] = {
    "i_spy": {
        "start":  _ispy_start,
        "handle": _ispy_handle,
        "stop":   _ispy_stop,
    },
    "20_questions": {
        "start":  _20q_start,
        "handle": _20q_handle,
        "stop":   _20q_stop,
    },
    "trivia": {
        "start":  _trivia_start,
        "handle": _trivia_handle,
        "stop":   _trivia_stop,
    },
    "jeopardy": {
        "start":  _jeopardy_start,
        "handle": _jeopardy_handle,
        "stop":   _jeopardy_stop,
    },
    "word_association": {
        "start":  _wordassoc_start,
        "handle": _wordassoc_handle,
        "stop":   _wordassoc_stop,
    },
}


def _clear_game() -> None:
    global _active_game, _game_state
    # A still-armed answer timer must not outlive the game it belongs to — the
    # token checks make a stray fire harmless, but only until someone edits them.
    timer = _game_state.pop("answer_timer", None)
    if timer is not None:
        try:
            timer.cancel()
        except Exception:
            pass
    _active_game = None
    _game_state = {}


def _extract_game_outcome(state: dict) -> str:
    """Best-effort one-phrase outcome from a game's state, for the episodic memory.
    Reliable for trivia (score survives to natural end); other games clear their state
    internally, so this returns "" and the memory is just "I played X". Defensive."""
    try:
        if not isinstance(state, dict) or not state:
            return ""
        if "score" in state and ("total_questions" in state or "history" in state):
            score = int(state.get("score", 0) or 0)
            total = int(
                state.get("total_questions")
                or len(state.get("history") or [])
                or 0
            )
            if total > 0:
                return f"scored {score} out of {total}"
        # 20 Questions (Rex is the guesser): result survives to the natural end.
        result = state.get("result")
        if result == "win":
            guess = state.get("final_guess", "it")
            return f"guessed it — {guess} — in {state.get('question_count', 0)} questions"
        if result == "lose":
            return "couldn't guess it"
    except Exception:
        pass
    return ""


# ── Session game ledger (diary context) ──────────────────────────────────────
# Every game played this session, snapshotted at its end. The shutdown diary
# extractor (main._episodic_shutdown_summary → llm.generate_diary_entry) reads
# this so a game session is remembered as "I hosted Jeopardy with X and Y — PJ
# won", instead of being mined for fake life-event threads out of game chatter
# (field 2026-08-26: "take her points away, she cheated" became the stored open
# thread "whether T'Joy's points were actually taken away", asked cold the next
# day). Winner comes from the REAL scoreboard here, never from the transcript.
_session_games: list[dict] = []


def _game_session_entry(game: str, state: dict, *, finished: bool) -> dict:
    """Pure snapshot of one game for the session ledger. Caller holds _lock."""
    display = _GAME_DISPLAY_NAMES.get(game, game.replace("_", " ").title())
    entry: dict = {
        "game": game, "display": display, "finished": bool(finished),
        "players": [], "scores": {}, "winner": None,
        "outcome": _extract_game_outcome(state),
    }
    for player in (state or {}).get("players") or []:
        try:
            name = str(player.get("name") or "").strip()
        except AttributeError:
            continue
        if not name:
            continue
        entry["players"].append(name)
        try:
            entry["scores"][name] = int(player.get("score", 0) or 0)
        except (TypeError, ValueError):
            pass
    # A winner needs a FINISHED game and a strict high score — a tie is not a
    # winner, and a game cut short mid-board crowned nobody.
    if finished and entry["scores"]:
        top = max(entry["scores"].values())
        leaders = [n for n, s in entry["scores"].items() if s == top]
        if len(leaders) == 1:
            entry["winner"] = leaders[0]
    return entry


def _record_session_game(game: Optional[str], state: dict, *, finished: bool) -> None:
    """Append one ended game to the session ledger. Caller holds _lock."""
    if not game:
        return
    try:
        _session_games.append(_game_session_entry(game, state or {}, finished=finished))
    except Exception as exc:
        _log.debug("[games] session ledger record failed: %s", exc)


def session_games_played() -> list[dict]:
    """Games played this session, oldest first — plus a live snapshot of a game
    still running (shutdown never stops the active game, so a mid-game
    power-off must still reach the diary)."""
    with _lock:
        out = [dict(e) for e in _session_games]
        if _active_game:
            try:
                out.append(_game_session_entry(_active_game, _game_state, finished=False))
            except Exception:
                pass
    return out


def _episodic_game_played(game: Optional[str], person_id, outcome: str = "") -> None:
    """Log "I played Trivia with Bret — scored 4 out of 5" to Rex's episodic memory.
    Gated + failure-safe (no-ops under the test runner / when episodic memory is off)."""
    if not game:
        return
    try:
        from memory import episodes
        display = _GAME_DISPLAY_NAMES.get(game, game.replace("_", " ").title())
        name = _jeopardy_person_name(person_id)  # generic people.db name lookup
        episodes.record_game_played(
            display, outcome,
            person_id=person_id if isinstance(person_id, int) else None,
            person_name=name,
            detail={"game": game} if game else None,
        )
    except Exception as exc:
        _log.debug("[games] episodic game_played failed: %s", exc)


# ── Public API ────────────────────────────────────────────────────────────────

def can_play(game_name: str) -> tuple[bool, Optional[str]]:
    """
    Check whether Rex is willing to play game_name right now.

    Returns (True, None) if the game is within its repeat limit.
    Returns (False, refusal_line) if Rex has played it too many times recently.

    The effective repeat limit scales with the agreeability personality parameter:
      - Low agreeability → Rex tires of the same game faster (lower effective limit)
      - High agreeability → Rex plays more willingly (higher effective limit)

    Unknown game names pass through as (True, None) — start_game() handles them.
    """
    normalized = _normalize_game(game_name)
    if normalized is None:
        return True, None

    now = time.monotonic()
    window = config.GAME_REPEAT_WINDOW_SECS

    # Prune timestamps outside the rolling window
    history = _game_play_log.get(normalized, [])
    history = [t for t in history if now - t < window]
    _game_play_log[normalized] = history

    # Scale the repeat limit by agreeability around the 0–100 scale's neutral
    # midpoint (50): agreeability=50 → limit unchanged, below tires faster, above
    # plays more. Rex's default agreeability is deliberately low (35), so he tires
    # a touch faster than a neutral personality — matching his needling persona.
    agreeability = _get_agreeability()
    multiplier = agreeability / 50.0
    effective_limit = max(1, round(config.GAME_REPEAT_LIMIT * multiplier))

    if len(history) < effective_limit:
        return True, None

    # Rex refuses — generate an in-character line
    display = _GAME_DISPLAY_NAMES.get(normalized, normalized.replace("_", " ").title())
    window_mins = window // 60
    stubbornness = (
        "Rex is particularly stubborn and unenthusiastic about it."
        if agreeability < 30 else
        "Rex is mildly reluctant but polite about it."
        if agreeability > 75 else
        ""
    )
    refusal = _rex_respond(
        f"[GAME: Repeat Limit] Rex has played \"{display}\" {len(history)} time(s) "
        f"in the last {window_mins} minutes (limit: {effective_limit}). "
        f"Rex refuses to play it again right now — he's had enough of that game for a while. "
        f"Express this in Rex's voice: dry, a little dramatic, with attitude. "
        f"Suggest the other games ({', '.join(n for n in _GAME_DISPLAY_NAMES.values() if n != display)}) "
        f"as alternatives. {stubbornness}",
    )
    return False, refusal


def start_game(game_name: str, person_id: Optional[int] = None) -> str:
    """
    Initialize the named game and return Rex's opening line.
    Stops any currently active game first.
    Accepts natural variations: "i spy", "twenty questions", "word association", etc.
    """
    global _active_game, _game_state

    # Repeat-limit gate — check before normalizing unknown names
    ok, refusal = can_play(game_name)
    if not ok:
        return refusal

    normalized = _normalize_game(game_name)
    trivia_preset_category = (
        _trivia_resolve_preset_category(game_name)
        if normalized == "trivia"
        else None
    )
    if normalized is None:
        known_names = available_game_names()
        if len(known_names) > 1:
            known = ", ".join(known_names[:-1]) + f", and {known_names[-1]}"
        else:
            known = known_names[0] if known_names else "no games, somehow"
        return _rex_respond(
            f"[GAME: Unknown] Player asked to play \"{game_name}\" — Rex doesn't know that game. "
            f"Rex lists the games he does know ({known}) in character.",
            person_id,
        )

    with _lock:
        _active_game = normalized
        _game_state = {}

    # Record this play for future can_play() checks
    _game_play_log.setdefault(normalized, []).append(time.monotonic())

    _log.info("[games] Starting game: %s", normalized)
    if normalized == "trivia" and trivia_preset_category:
        return _trivia_start(person_id, preset_category=trivia_preset_category)
    return _GAME_HANDLERS[normalized]["start"](person_id)


def start_trivia(person_id: Optional[int] = None) -> str:
    """Convenience wrapper so trivia can be launched by a dedicated command."""
    return start_game("trivia", person_id)


def handle_input(text: str, person_id: Optional[int] = None, audio_array=None) -> str:
    """
    Process player input for the current game and return Rex's response.
    Automatically clears game state when a game ends naturally.
    Returns an idle Rex line if no game is active.
    """
    global _active_game

    with _lock:
        game = _active_game

    if not game:
        return _rex_respond(
            "[No game active] Player said something but no game is running. "
            "Rex notes there is no game in progress — brief.",
            person_id,
        )

    if game == "jeopardy":
        response, done = _jeopardy_handle(text, person_id, audio_array)
    else:
        response, done = _GAME_HANDLERS[game]["handle"](text, person_id)

    if done:
        with _lock:
            outcome = _extract_game_outcome(_game_state)
            _record_session_game(game, _game_state, finished=True)
            _clear_game()
        # "I played Trivia with Bret — scored 4 out of 5" → rex.db.
        _episodic_game_played(game, person_id, outcome)
        _log.info("[games] Game %s ended naturally", game)

    return response


def stop_game(person_id: Optional[int] = None) -> str:
    """End the current game gracefully and return Rex's closing line."""
    global _active_game

    with _lock:
        game = _active_game

    if not game:
        return _rex_respond(
            "[GAME: Stop] No game is currently running. Rex notes this — brief.",
            person_id,
        )

    _log.info("[games] Stopping game: %s", game)
    with _lock:
        outcome = _extract_game_outcome(_game_state)  # snapshot before the stop handler clears it
        _record_session_game(game, _game_state, finished=False)
    response = _GAME_HANDLERS[game]["stop"](person_id)

    with _lock:
        _clear_game()

    # "I played Trivia with Bret — scored 3 out of 5" → rex.db (user-stopped mid-round).
    _episodic_game_played(game, person_id, outcome)

    return response


# ── Stop-confirmation guard (owner ask 2026-08-25) ───────────────────────────
# "Stop playing" mid-game gets an are-you-sure before the board evaporates —
# an ASR mishear or one grumpy player should not end everyone's game on the
# spot. The pending ask lives in _game_state (dies with the game), is one-shot,
# and expires after GAME_STOP_CONFIRM_WINDOW_SECS.

_STOP_CONFIRM_QUESTION = (
    "But we're having so much fun, are you sure you want to end the game?"
)


def _stop_confirm_verdict(text: str) -> str:
    """'yes' / 'no' / 'other' for the reply to the are-you-sure ask.

    Anything that is neither a clear yes nor a clear no — including a player
    just carrying on with the game — is 'other': the ask is dropped and the
    utterance is handled as a normal game turn.
    """
    plain = " ".join(
        re.sub(r"[^a-z0-9\s]", " ", (text or "").lower().replace("'", "")).split()
    )
    if not plain:
        return "other"
    leadin = r"(?:um+ |uh+ |well |ok(?:ay)? |oh |hmm+ )*"
    if re.match(
        rf"^{leadin}"
        r"(?:yes|yeah|yep|yup|sure|absolutely|definitely|positive|affirmative|"
        r"correct|of course|i ?am sure|im sure|we ?are sure|were sure|"
        r"end it|stop it|kill it|do it|please do|go ahead)\b",
        plain,
    ):
        return "yes"
    if re.match(
        rf"^{leadin}"
        r"(?:no|nope|nah|never ?mind|just kidding|kidding|keep playing|"
        r"keep going|lets keep|continue|carry on|dont|do not|not yet|"
        r"not really|we ?are good|were good|im good|i ?am good|stay)\b",
        plain,
    ):
        return "no"
    return "other"


def _stop_confirm_resume_line() -> str:
    """Rex's line when the table votes to keep playing."""
    if _active_game == "jeopardy":
        phase = _game_state.get("phase")
        clue = _game_state.get("current_clue")
        players = _game_state.get("players") or []
        try:
            player = _jeopardy_current_player()
        except Exception:
            player = {"name": "Player"}
        if phase == "awaiting_answer" and clue:
            return _jeopardy_repeat_clue_reply(
                dict(clue), player, prefix="That's the spirit — back to it. "
            )
        if phase == "selecting":
            return (
                f"That's the spirit. {player['name']}, "
                "pick a category and dollar value."
            )
        if phase == "awaiting_wager":
            score = int(player.get("score", 0) or 0)
            min_wager = int(getattr(config, "JEOPARDY_DD_MIN_WAGER", 5))
            return (
                f"That's the spirit. {player['name']}, what's your wager — "
                f"${min_wager} to ${_jeopardy_max_wager(score)}?"
            )
        if phase in ("final_wager", "final_answer") and players:
            queue = _game_state.get("final_queue") or []
            if queue:
                name = players[int(queue[0]) % len(players)]["name"]
                ask = "what's your wager?" if phase == "final_wager" else "your answer?"
                return f"That's the spirit. {name}, {ask}"
    return "That's the spirit. Back to the game — where were we?"


def request_stop_confirmation() -> Optional[str]:
    """Arm the are-you-sure guard for the active game; returns the question.

    None when no game is running (nothing to guard). For Jeopardy a live
    answer clock is frozen — "Time's up" firing over the are-you-sure
    exchange would steal the very turn it paused.
    """
    with _lock:
        if _active_game is None:
            return None
        _game_state["stop_confirm_at"] = time.monotonic()
        if _active_game == "jeopardy":
            _jeopardy_cancel_timeout()
    _log.info("[games] stop attempt — asking for confirmation")
    return _STOP_CONFIRM_QUESTION


def resolve_stop_confirmation(
    text: str,
    person_id: Optional[int] = None,
    *,
    stop_shaped: bool = False,
) -> Optional[tuple[str, Optional[str]]]:
    """Resolve the reply to a pending are-you-sure ask.

    Returns None when nothing (fresh) is pending. Otherwise:
      ("stop",   closing_line) — affirmative (or the stop was demanded again):
                                 the game is stopped here.
      ("resume", resume_line)  — a clear no: back to the game.
      ("pass",   None)         — neither: the ask is dropped and the caller
                                 should handle the utterance as a normal turn.
    The pending ask is one-shot: consumed whatever the outcome.
    """
    with _lock:
        game = _active_game
        asked_at = float(_game_state.get("stop_confirm_at") or 0.0)
        _game_state.pop("stop_confirm_at", None)
    if game is None or asked_at <= 0.0:
        return None
    window = float(getattr(config, "GAME_STOP_CONFIRM_WINDOW_SECS", 45.0))
    if (time.monotonic() - asked_at) > window:
        return None    # the moment passed; this turn is a normal move
    verdict = _stop_confirm_verdict(text)
    if stop_shaped or verdict == "yes":
        _log.info("[games] stop confirmed — ending %s", game)
        return ("stop", stop_game(person_id))
    if verdict == "no":
        _log.info("[games] stop declined — resuming %s", game)
        return ("resume", _stop_confirm_resume_line())
    return ("pass", None)


def stop_game_fast(person_id: Optional[int] = None) -> str:
    """End the current game without an LLM-generated closing line."""
    global _active_game

    with _lock:
        game = _active_game

    if not game:
        return "No game is running."

    if game == "jeopardy":
        try:
            _jeopardy_cancel_timeout()
        except Exception:
            pass

    display_name = _GAME_DISPLAY_NAMES.get(game, game.replace("_", " ").title())
    _log.info("[games] Fast stopping game: %s", game)
    with _lock:
        _record_session_game(game, _game_state, finished=False)
        _clear_game()
    return f"{display_name} stopped."


def consume_pending_audio_after_response() -> Optional[str]:
    """Return an audio file that should play after Rex's just-spoken game line."""
    with _lock:
        if _active_game != "jeopardy":
            return None
        clip_key = _game_state.pop("pending_after_response_clip", None)
    if not clip_key:
        return None
    return _jeopardy_clip_path(str(clip_key))


def on_response_spoken() -> None:
    """Notify the active game that Rex's spoken response has finished."""
    with _lock:
        if _active_game == "jeopardy":
            _jeopardy_arm_timeout()


def is_active() -> bool:
    """Return True if a game is currently running."""
    with _lock:
        return _active_game is not None


def active_roster_person_ids() -> frozenset:
    """person_ids of the people registered as players in the active game.

    A game roster is a standing declaration that these people are in the room
    and are EXPECTED to speak — most of them from off camera. Identity
    resolution uses it to stop the one visible face from absorbing everyone
    else's turns (field 2026-08-26: PJ was the only recognized face on camera
    and was credited with Bret's, Jeremy's and T'Joy's answers for the whole
    first round).
    """
    with _lock:
        players = list(_game_state.get("players") or []) if _active_game else []
    ids = set()
    for player in players:
        try:
            pid = player.get("person_id")
        except AttributeError:
            continue
        if pid is not None:
            try:
                ids.add(int(pid))
            except (TypeError, ValueError):
                continue
    return frozenset(ids)


def active_game_current_player_id() -> "Optional[int]":
    """person_id of the player whose turn it is in a running multi-player game.

    A declared turn order is a real prior: picks and answers come from the
    current player far more often than from anyone else. Used ONLY to break an
    acoustic near-tie between two registered players — never to override a
    voice match that has a clear margin.
    """
    with _lock:
        if not _active_game:
            return None
        players = list(_game_state.get("players") or [])
        phase = _game_state.get("phase")
        idx = int(_game_state.get("current_player_idx", 0) or 0)
    if len(players) < 2 or phase in ("final_wager", "final_answer"):
        # Final Jeopardy runs off final_queue, not current_player_idx — a stale
        # index here would name the wrong player.
        return None
    try:
        pid = (players[idx % len(players)] or {}).get("person_id")
        return int(pid) if pid is not None else None
    except (AttributeError, TypeError, ValueError):
        return None


def jeopardy_answer_window_open() -> bool:
    """True while a Jeopardy clue is live and Rex is waiting on a response.

    Deliberately cheap (no board copy — see snapshot()): the endpointing loop
    calls this once per turn to decide how long a thinking pause may run before
    the segment is closed.
    """
    with _lock:
        return _active_game == "jeopardy" and _game_state.get("phase") in (
            "awaiting_answer", "final_answer",
        )


def suppresses_conversation_interruptions() -> bool:
    """Return True while game flow should own the next spoken turn."""
    return is_active()


def current_game() -> Optional[str]:
    """Return the normalized name of the current game, or None if no game is active."""
    with _lock:
        return _active_game


def snapshot() -> dict:
    """Return a GUI-safe copy of the active game state."""
    with _lock:
        game = _active_game
        if game != "jeopardy":
            return {"active_game": game}
        state = {
            key: copy.deepcopy(value)
            for key, value in _game_state.items()
            if key not in {"answer_timer"}
        }

    board = state.get("board") or {}
    categories = []
    for category in board.get("categories") or []:
        clues = category.get("clues") or {}
        categories.append({
            "name": category.get("name") or "Category",
            "remaining_values": sorted(int(v) for v in clues.keys()),
        })
    values = list(state.get("board_values") or [])
    if not values:
        values = sorted({v for cat in categories for v in cat["remaining_values"]})

    current_clue = state.get("current_clue") or None
    return {
        "active_game": "jeopardy",
        "phase": state.get("phase"),
        "players": state.get("players") or [],
        "current_player_idx": int(state.get("current_player_idx", 0) or 0),
        "round": int(state.get("jeopardy_round", 1) or 1),
        "categories": categories,
        "values": values,
        "remaining": int(board.get("remaining", 0) or 0),
        "current_clue": current_clue,
        "last_category": state.get("last_category"),
        "voice_enroll_queue": list(state.get("voice_enroll_queue") or []),
    }
