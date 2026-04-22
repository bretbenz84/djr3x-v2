"""
features/games.py — Interactive game management for DJ-R3X.

Manages one active game at a time. All games are interruptible via stop_game().

Supported games:
    "i_spy"            — Rex picks an object from the camera frame; player guesses
    "20_questions"     — Rex thinks of something; player asks yes/no questions
    "trivia"           — Rex asks one trivia question and judges the answer
    "word_association" — Rapid back-and-forth word chain

Public API:
    can_play(game_name)                  → (bool, str | None)  # repeat-limit gate
    start_game(game_name, person_id=None) → str    # opening line for Rex to speak
    start_trivia(person_id=None)         → str    # convenience wrapper for trivia
    handle_input(text, person_id=None)   → str    # Rex's response to player input
    stop_game(person_id=None)            → str    # graceful closing line
    is_active()                          → bool
    current_game()                       → str | None
"""

import json
import logging
import random
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
    "word_association": "Word Association",
}

# ── Game name aliases ─────────────────────────────────────────────────────────

_GAME_ALIASES: dict[str, str] = {
    "i spy":            "i_spy",
    "ispy":             "i_spy",
    "i_spy":            "i_spy",
    "spy":              "i_spy",
    "20 questions":     "20_questions",
    "twenty questions": "20_questions",
    "20questions":      "20_questions",
    "20_questions":     "20_questions",
    "trivia":           "trivia",
    "word association": "word_association",
    "word_association": "word_association",
    "word assoc":       "word_association",
    "association":      "word_association",
}


def _normalize_game(name: str) -> Optional[str]:
    return _GAME_ALIASES.get(name.strip().lower())


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _get_client():
    try:
        import apikeys
        from openai import OpenAI
        return OpenAI(api_key=apikeys.OPENAI_API_KEY)
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
    return config.PERSONALITY_DEFAULTS.get("agreeability", 60)


# ── I Spy game ────────────────────────────────────────────────────────────────

_ISPY_MAX_GUESSES = 5


def _ispy_get_target() -> Optional[dict]:
    """
    Grab a camera frame and ask GPT-4o to pick an I Spy object.
    Returns {"object": "red chair", "clue": "red"} or None on failure.
    """
    try:
        from vision import camera
    except ImportError:
        _log.warning("[games] Camera unavailable for I Spy")
        return None

    frame = camera.get_frame()
    if frame is None:
        return None

    b64 = encode_jpeg_base64(frame, quality=85)
    if b64 is None:
        return None

    detail = config.VISION_DETAIL.get("active_conversation", "auto")
    prompt = (
        "Pick ONE specific object visible in this image that would work well for "
        "the game 'I Spy'. Choose something clearly visible. Not a person.\n"
        "Return a JSON object with exactly two keys:\n"
        '  "object": the full name of the object (e.g. "red chair", "blue mug"),\n'
        '  "clue": a single descriptive word for "I spy something ___" '
        '(e.g. "red", "shiny", "round").\n'
        "Return ONLY the JSON object — no preamble, no markdown."
    )

    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=config.VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": detail},
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=150,
        )
        data = _parse_json(resp.choices[0].message.content.strip())
        if isinstance(data, dict) and data.get("object") and data.get("clue"):
            return data
    except Exception as exc:
        _log.error("[games] I Spy vision call failed: %s", exc)

    return None


def _ispy_start(person_id: Optional[int]) -> str:
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
        "guess_count": 0,
    })

    return _rex_respond(
        f"[GAME: I Spy — START] Give Rex's opening line for I Spy. "
        f"Say \"I spy with my little eye, something that is {target['clue']}\" "
        f"and add a brief Rex-style flourish. Players have {_ISPY_MAX_GUESSES} guesses. "
        f"Do not reveal the object.",
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
    _game_state.clear()
    return _rex_respond(
        f"[GAME: I Spy — STOPPED] Game ended early. The object was \"{target}\". "
        f"Rex delivers a brief in-character close.",
        person_id,
    )


# ── 20 Questions game ─────────────────────────────────────────────────────────

_20Q_MAX_QUESTIONS = 20


def _20q_pick_secret() -> dict:
    raw = _quick_call(
        "Pick something interesting for a game of 20 Questions — a person, place, or thing. "
        "Choose something reasonably well-known, not too obscure. "
        "Bias toward Star Wars, science, pop culture, or space themes. "
        "Return a JSON object with exactly two keys:\n"
        '  "secret": the name of the person, place, or thing,\n'
        '  "category": one of "person", "place", or "thing".\n'
        "Return ONLY the JSON object — no preamble, no markdown.",
        temperature=0.9,
        max_tokens=80,
    )
    data = _parse_json(raw)
    if isinstance(data, dict) and data.get("secret") and data.get("category"):
        return data
    return {"secret": "a lightsaber", "category": "thing"}


def _20q_start(person_id: Optional[int]) -> str:
    secret_data = _20q_pick_secret()
    _game_state.update({
        "secret": secret_data["secret"],
        "category": secret_data["category"],
        "question_count": 0,
        "questions_log": [],
    })

    return _rex_respond(
        f"[GAME: 20 Questions — START] Rex is thinking of a {secret_data['category']}. "
        f"Give Rex's opening line: he's got something in mind and the player gets "
        f"{_20Q_MAX_QUESTIONS} yes/no questions to figure it out. Rex adds his usual flair.",
        person_id,
    )


def _20q_handle(text: str, person_id: Optional[int]) -> tuple[str, bool]:
    secret = _game_state.get("secret", "")
    category = _game_state.get("category", "thing")
    q_count = _game_state.get("question_count", 0)
    q_log = _game_state.get("questions_log", [])

    text_lower = text.strip().lower()
    questions_left = _20Q_MAX_QUESTIONS - q_count

    # Detect final guess: explicit phrasing or out of questions
    is_final_guess = (
        questions_left <= 0
        or any(phrase in text_lower for phrase in [
            "is it", "is the answer", "i think it", "my answer", "i guess", "final answer",
        ])
    )

    if is_final_guess:
        correct = (
            fuzz.ratio(text_lower, secret.lower()) >= 70
            or secret.lower() in text_lower
        )
        _game_state.clear()
        if correct:
            return (
                _rex_respond(
                    f"[GAME: 20 Questions — CORRECT GUESS] Player correctly identified "
                    f"\"{secret}\" after {q_count} questions. Rex delivers a grudging "
                    f"congratulations — mildly annoyed, fully in character.",
                    person_id,
                ),
                True,
            )
        outcome = "ran out of questions" if questions_left <= 0 else "wrong final guess"
        return (
            _rex_respond(
                f"[GAME: 20 Questions — WRONG/DONE] Player {outcome}. "
                f"The answer was \"{secret}\" (a {category}). "
                f"Rex reveals it — smug and amused in equal measure.",
                person_id,
            ),
            True,
        )

    # Answer the yes/no question
    q_count += 1
    _game_state["question_count"] = q_count

    answer = _quick_call(
        f'I am thinking of "{secret}" (a {category}) in a game of 20 Questions.\n'
        f'The player asked: "{text.strip()}"\n'
        f"Answer with ONLY one word: yes, no, or sometimes.",
        temperature=0,
        max_tokens=5,
    ).strip().lower()
    if answer not in ("yes", "no", "sometimes"):
        answer = "no"

    q_log.append({"q": text.strip(), "a": answer})
    _game_state["questions_log"] = q_log

    questions_left = _20Q_MAX_QUESTIONS - q_count
    running_low = questions_left <= 5

    return (
        _rex_respond(
            f"[GAME: 20 Questions — Q#{q_count}/{_20Q_MAX_QUESTIONS}] "
            f"Rex is thinking of \"{secret}\" (a {category}). "
            f'Player asked: "{text.strip()}". The truthful answer is: "{answer}". '
            f"Rex delivers this answer in character — brief and punchy. "
            f"{questions_left} questions remaining."
            + (" Mention they're running low on questions." if running_low else ""),
            person_id,
        ),
        False,
    )


def _20q_stop(person_id: Optional[int]) -> str:
    secret = _game_state.get("secret", "something")
    q_count = _game_state.get("question_count", 0)
    _game_state.clear()
    return _rex_respond(
        f"[GAME: 20 Questions — STOPPED] Game cut short after {q_count} questions. "
        f"The answer was \"{secret}\". Rex delivers a brief in-character close.",
        person_id,
    )


# ── Trivia game ───────────────────────────────────────────────────────────────

def _trivia_start(person_id: Optional[int]) -> str:
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

    category = "General Knowledge" if "General Knowledge" in categories else categories[0]
    question = trivia_bank.get_question(category)
    if not question:
        return _rex_respond(
            f"[GAME: Trivia — NO QUESTION] Rex tried to load a trivia question from "
            f"category \"{category}\" but came up empty. Apologize in character.",
            person_id,
        )

    _game_state.update({
        "category": category,
        "question": question,
    })

    return _rex_respond(
        f"[GAME: Trivia — START] Rex is starting a trivia round in category "
        f"\"{category}\". After a brief Rex-style intro, ask this question exactly: "
        f"\"{question['question']}\"",
        person_id,
    )


def _trivia_handle(text: str, person_id: Optional[int]) -> tuple[str, bool]:
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

    try:
        from features import trivia as trivia_bank
        is_correct = trivia_bank.check_answer(question, text)
    except Exception as exc:
        _log.error("[games] trivia answer check failed: %s", exc)
        is_correct = False

    answer = question.get("answer", "unknown")
    _game_state.clear()

    if is_correct:
        return (
            _rex_respond(
                f"[GAME: Trivia — CORRECT] Category: \"{category}\". "
                f"Question: \"{question.get('question', '')}\". "
                f"Player answered: \"{text.strip()}\". Correct answer: \"{answer}\". "
                "Rex celebrates briefly in character.",
                person_id,
            ),
            True,
        )

    return (
        _rex_respond(
            f"[GAME: Trivia — WRONG] Category: \"{category}\". "
            f"Question: \"{question.get('question', '')}\". "
            f"Player answered: \"{text.strip()}\". Correct answer: \"{answer}\". "
            "Rex reveals the answer and lightly roasts the miss in character.",
            person_id,
        ),
        True,
    )


def _trivia_stop(person_id: Optional[int]) -> str:
    question = _game_state.get("question", {})
    answer = question.get("answer", "unknown") if isinstance(question, dict) else "unknown"
    _game_state.clear()
    return _rex_respond(
        f"[GAME: Trivia — STOPPED] Trivia ended early. The answer was \"{answer}\". "
        "Rex delivers a brief in-character close.",
        person_id,
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
    "word_association": {
        "start":  _wordassoc_start,
        "handle": _wordassoc_handle,
        "stop":   _wordassoc_stop,
    },
}


def _clear_game() -> None:
    global _active_game, _game_state
    _active_game = None
    _game_state = {}


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

    # Scale limit by agreeability: agreeability=60 → limit unchanged
    agreeability = _get_agreeability()
    multiplier = agreeability / 60.0
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
    if normalized is None:
        known = "I Spy, 20 Questions, and Word Association"
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
    return _GAME_HANDLERS[normalized]["start"](person_id)


def start_trivia(person_id: Optional[int] = None) -> str:
    """Convenience wrapper so trivia can be launched by a dedicated command."""
    return start_game("trivia", person_id)


def handle_input(text: str, person_id: Optional[int] = None) -> str:
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

    response, done = _GAME_HANDLERS[game]["handle"](text, person_id)

    if done:
        with _lock:
            _clear_game()
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
    response = _GAME_HANDLERS[game]["stop"](person_id)

    with _lock:
        _clear_game()

    return response


def is_active() -> bool:
    """Return True if a game is currently running."""
    with _lock:
        return _active_game is not None


def current_game() -> Optional[str]:
    """Return the normalized name of the current game, or None if no game is active."""
    with _lock:
        return _active_game
