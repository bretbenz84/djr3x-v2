"""
features/games.py — Interactive game management for DJ-R3X.

Manages one active game at a time. All games are interruptible via stop_game().

Supported games:
    "i_spy"            — Rex picks an object from the camera frame; player guesses
    "20_questions"     — Rex thinks of something; player asks yes/no questions
    "trivia"           — Rex runs a short scored trivia round
    "jeopardy"         — A verbal Jeopardy-style board with real clue data
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

_TRIVIA_CORRECT_LINES = [
    "Correct. Apparently the organic processor still boots.",
    "Correct. I am recording that as suspicious competence.",
    "Correct. Tiny parade, very tiny budget.",
]

_TRIVIA_WRONG_LINES = [
    "Nope. A brave answer, if bravery means ignoring facts.",
    "Incorrect. The answer wandered off and you chased a chair.",
    "No. The trivia board remains unimpressed.",
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

    try:
        from features import trivia as trivia_bank
        is_correct = False if _trivia_is_pass(text) else trivia_bank.check_answer(question, text)
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
    elif _trivia_is_pass(text):
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
    _game_state.pop("awaiting_prompt_delivery", None)
    if timer is not None:
        try:
            timer.cancel()
        except Exception:
            pass


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


def _jeopardy_categories_reminder() -> str:
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
        return ""
    return f"Remaining categories: {categories}. "


def _jeopardy_offer_rebound() -> Optional[dict]:
    """Give the same clue to the next player who has not tried it yet."""
    players = _game_state.get("players") or []
    if len(players) <= 1:
        return None

    current_idx = int(_game_state.get("current_player_idx", 0)) % len(players)
    attempted = set(int(i) for i in (_game_state.get("current_clue_attempts") or []))
    attempted.add(current_idx)
    _game_state["current_clue_attempts"] = sorted(attempted)

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
        f"{clue.get('category')} for ${clue.get('value')}. "
        f"Clue: {clue.get('clue')}."
    )


def _jeopardy_finish_missed_clue(
    prefix: str,
    correct_response: str,
    *,
    done: bool,
    players: list[dict],
    score_line: bool = True,
) -> tuple[str, bool]:
    _game_state.pop("current_clue", None)
    _game_state.pop("current_clue_attempts", None)
    _game_state["phase"] = "selecting"

    if done:
        return _jeopardy_complete_round_or_finish(
            f"{prefix}{correct_response}. ",
            advance_player=True,
        )

    next_player = _jeopardy_advance_player()
    scores = ""
    if score_line:
        try:
            from features import jeopardy as jeopardy_bank
            scores = f"Scores: {jeopardy_bank.format_scores(players)} "
        except Exception:
            scores = "Scores unavailable. "
    categories = _jeopardy_categories_reminder()
    return (
        f"{prefix}{correct_response}. {scores}{categories}"
        f"{next_player['name']}, choose the next square.",
        False,
    )


def _jeopardy_schedule_post_timeout_rebound(done_event: threading.Event) -> None:
    """Start the rebound timer/theme after a timeout prompt finishes speaking."""
    def _wait_then_start() -> None:
        try:
            if not done_event.wait(timeout=45.0):
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
        "last_category": None,
        "jeopardy_round": round_no,
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

    return _jeopardy_finish_line(prefix), True


def _jeopardy_timeout_fired(token: str) -> None:
    line = ""
    queue_timesup = True
    schedule_rebound = False
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
        _game_state.pop("answer_timer", None)
        _game_state.pop("answer_timer_token", None)
        correct_response = _jeopardy_correct_response_text(clue)
        next_player = _jeopardy_offer_rebound()
        if next_player:
            line = _jeopardy_rebound_prompt("Time's up. ", next_player, clue)
            schedule_rebound = True
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
            _jeopardy_queue_clip("timesup")
        done_event = speech_queue.enqueue(
            llm.clean_response_text(line),
            priority=1,
            tag="jeopardy:timeout",
        )
        if schedule_rebound:
            _jeopardy_schedule_post_timeout_rebound(done_event)
    except Exception as exc:
        _log.debug("[jeopardy] timeout speech failed: %s", exc)


def _jeopardy_arm_timeout() -> None:
    if _active_game != "jeopardy":
        return
    if _game_state.get("phase") != "awaiting_answer":
        return
    if not _game_state.pop("awaiting_prompt_delivery", False):
        return
    timeout = float(getattr(config, "JEOPARDY_ANSWER_TIMEOUT_SECS", 14.0))
    if timeout <= 0:
        return
    token = f"{time.monotonic():.6f}:{random.random():.6f}"
    timer = threading.Timer(timeout, _jeopardy_timeout_fired, args=(token,))
    timer.daemon = True
    _game_state["answer_timer_token"] = token
    _game_state["answer_timer"] = timer
    timer.start()


def _jeopardy_start(person_id: Optional[int]) -> str:
    speaker_name = _jeopardy_person_name(person_id)
    _jeopardy_queue_clip("intro")
    _game_state.update({
        "phase": "awaiting_players",
        "speaker_name": speaker_name,
    })
    return _rex_respond(
        "[GAME: Jeopardy — PLAYER SETUP] Rex is starting a verbal Jeopardy-style "
        "game. Ask who is playing. Tell them they can say one, two, or three names "
        "in one reply. Make it feel like a game-show intro, but keep it brief.",
        person_id,
    )


def _jeopardy_begin_board(names: list[str], person_id: Optional[int]) -> tuple[str, bool]:
    players = [{"name": name, "score": 0} for name in names]
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

    player_text = ", ".join(name for name in names)
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
            "I need actual player names, not mysterious cantina fog. Say something like 'Bret and Joy'.",
            False,
        )

    return _jeopardy_begin_board(names, person_id)


def _jeopardy_handle_selection(text: str, person_id: Optional[int]) -> tuple[str, bool]:
    try:
        from features import jeopardy as jeopardy_bank
        board = _game_state.get("board") or {}
        clue, error = jeopardy_bank.parse_selection(
            text,
            board,
            last_category=_game_state.get("last_category"),
        )
    except Exception as exc:
        _log.error("[jeopardy] selection parse failed: %s", exc)
        clue, error = None, "My board parser fell into a reactor shaft. Try the category and value again."

    if not clue:
        return (error, False)

    player = _jeopardy_current_player()
    effective_value = int(clue.get("value", 0) or 0)
    daily = bool(clue.get("daily_double"))
    if daily:
        effective_value *= 2
        _jeopardy_queue_clip("daily_double")

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

    daily_line = (
        "Daily Double. Automatic double, because my wagering subsystem was built during lunch. "
        if daily else ""
    )
    return (
        f"{daily_line}{player['name']}, {clue.get('category')} for ${clue.get('value')}. "
        f"Clue: {clue.get('clue')}.",
        False,
    )


def _jeopardy_handle_answer(text: str, person_id: Optional[int]) -> tuple[str, bool]:
    try:
        from features import jeopardy as jeopardy_bank
    except Exception as exc:
        _log.error("[jeopardy] answer helper import failed: %s", exc)
        jeopardy_bank = None

    _jeopardy_cancel_timeout()
    clue = dict(_game_state.get("current_clue") or {})
    if not clue:
        _game_state["phase"] = "selecting"
        return ("I lost the clue state. Pick another square before I blame a power converter.", False)

    players = _game_state.get("players") or [{"name": "Player", "score": 0}]
    idx = int(_game_state.get("current_player_idx", 0)) % len(players)
    player = players[idx]
    answer = clue.get("answer", "unknown")
    value = int(clue.get("effective_value", clue.get("value", 0)) or 0)
    correct_response = _jeopardy_correct_response_text(clue)

    passed = bool(jeopardy_bank and jeopardy_bank.is_pass_or_timeout(text))
    correct = bool(jeopardy_bank and jeopardy_bank.is_correct(text, answer))

    done = int((_game_state.get("board") or {}).get("remaining", 0) or 0) <= 0
    if passed:
        _jeopardy_queue_clip("timesup")
        next_player = _jeopardy_offer_rebound()
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
        player["score"] = int(player.get("score", 0)) + value
        _game_state.pop("current_clue", None)
        _game_state.pop("current_clue_attempts", None)
        _jeopardy_queue_clip("right")
        flourish = random.choice([
            "Correct. The organics survive another clue.",
            "Correct. I am marking this as suspiciously competent.",
            "Correct. The scoreboard briefly respects you.",
        ])
        if done:
            return _jeopardy_complete_round_or_finish(f"{flourish} ")
        _game_state["phase"] = "selecting"
        scores = jeopardy_bank.format_scores(players) if jeopardy_bank else "scores unavailable"
        categories = _jeopardy_categories_reminder()
        return (
            f"{flourish} ${value} to {player['name']}. Scores: {scores}. "
            f"{categories}"
            f"{player['name']}, pick the next category and value.",
            False,
        )

    player["score"] = int(player.get("score", 0)) - value
    _jeopardy_queue_clip("wrong")
    roast = random.choice([
        "A bold miss.",
        "The board accepts your sacrifice.",
        "That answer landed somewhere near Alderaan.",
    ])
    next_player = _jeopardy_offer_rebound()
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
        f"{roast} ",
        correct_response,
        done=done,
        players=players,
    )


def _jeopardy_handle(text: str, person_id: Optional[int]) -> tuple[str, bool]:
    phase = _game_state.get("phase")
    if phase == "awaiting_players":
        return _jeopardy_handle_player_setup(text, person_id)
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


def current_game() -> Optional[str]:
    """Return the normalized name of the current game, or None if no game is active."""
    with _lock:
        return _active_game
