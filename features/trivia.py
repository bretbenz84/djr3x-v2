"""
features/trivia.py — Trivia question bank loader and category management.

Loads JSON files from assets/trivia/ (one file per category).
If the directory is empty, generates a starter set via GPT-4o-mini and saves them.

Each question format: {"question": "...", "answer": "...", "alternatives": [...], "difficulty": 1-3}

Public API:
    get_categories()                        → list[str]
    get_question(category, difficulty=None) → dict | None
    check_answer(question, user_answer)     → bool
    reset_session()
"""

import json
import logging
import random
import sys
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rapidfuzz import fuzz

import config

_log = logging.getLogger(__name__)

# ── Module state ──────────────────────────────────────────────────────────────

_bank: dict[str, list[dict]] = {}       # category → list of question dicts
_bank_loaded: bool = False
_asked: dict[str, set[int]] = {}        # category → set of already-asked indices

_STARTER_CATEGORIES = [
    "General Knowledge",
    "Science",
    "Star Wars",
    "Space & Astronomy",
    "History",
    "Pop Culture",
    "Music",
    "Animals",
    "Sports",
]

_QUESTIONS_PER_CATEGORY = 20


# ── Internal helpers ──────────────────────────────────────────────────────────

def _trivia_dir() -> Path:
    return Path(config.TRIVIA_DIR)


def _category_to_stem(category: str) -> str:
    return category.lower().replace("&", "and").replace(" ", "_")


def _stem_to_category(stem: str) -> str:
    name = stem.replace("_", " ").title()
    return name.replace(" And ", " & ")


def _parse_json(text: str):
    """Parse JSON tolerating markdown code fences."""
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
    for open_c, close_c in [("[", "]"), ("{", "}")]:
        s = text.strip().find(open_c)
        e = text.strip().rfind(close_c)
        if s != -1 and e > s:
            try:
                return json.loads(text.strip()[s: e + 1])
            except json.JSONDecodeError:
                pass
    return None


def _get_client():
    try:
        import apikeys
        from openai import OpenAI
        return OpenAI(api_key=apikeys.OPENAI_API_KEY)
    except ImportError as exc:
        raise ImportError(f"trivia generation requires apikeys and openai: {exc}") from exc


def _generate_starter_set() -> None:
    trivia_dir = _trivia_dir()
    trivia_dir.mkdir(parents=True, exist_ok=True)

    try:
        client = _get_client()
    except ImportError as exc:
        _log.error("[trivia] Cannot generate starter set: %s", exc)
        return

    for category in _STARTER_CATEGORIES:
        path = trivia_dir / (_category_to_stem(category) + ".json")
        if path.exists():
            continue

        _log.info("[trivia] Generating %d questions for: %s", _QUESTIONS_PER_CATEGORY, category)
        prompt = (
            f'Generate {_QUESTIONS_PER_CATEGORY} trivia questions for the category: "{category}".\n'
            "Return a JSON array. Each element must have exactly these keys:\n"
            '  "question": the question string,\n'
            '  "answer": the correct answer as a concise string,\n'
            '  "alternatives": list of 2–4 alternative acceptable answers '
            "(abbreviations, alternate phrasings, partial answers that should be accepted),\n"
            '  "difficulty": integer 1, 2, or 3 (1=easy, 2=medium, 3=hard).\n'
            "Include ~40% easy, ~40% medium, ~20% hard.\n"
            "Return ONLY the JSON array — no preamble, no markdown fences."
        )

        try:
            resp = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            data = _parse_json(resp.choices[0].message.content.strip())
            if not isinstance(data, list) or not data:
                _log.error("[trivia] Bad response for category: %s", category)
                continue
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            _log.info("[trivia] Saved %d questions → %s", len(data), path.name)
        except Exception as exc:
            _log.error("[trivia] Generation failed for %s: %s", category, exc)


def _load_bank() -> None:
    global _bank, _bank_loaded
    trivia_dir = _trivia_dir()
    trivia_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(trivia_dir.glob("*.json"))
    if not json_files:
        _log.info("[trivia] No files found — generating starter set")
        _generate_starter_set()
        json_files = sorted(trivia_dir.glob("*.json"))

    loaded: dict[str, list[dict]] = {}
    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                questions = json.load(f)
            if not isinstance(questions, list):
                _log.warning("[trivia] %s: expected list, skipping", path.name)
                continue
            category = _stem_to_category(path.stem)
            loaded[category] = questions
            _log.info("[trivia] Loaded %d questions from %s (%s)", len(questions), path.name, category)
        except Exception as exc:
            _log.error("[trivia] Failed to load %s: %s", path.name, exc)

    _bank = loaded
    _bank_loaded = True


def _ensure_loaded() -> None:
    if not _bank_loaded:
        _load_bank()


# ── Public API ────────────────────────────────────────────────────────────────

def get_categories() -> list[str]:
    """Return a sorted list of available category names."""
    _ensure_loaded()
    return sorted(_bank.keys())


def get_question(category: str, difficulty: Optional[int] = None) -> Optional[dict]:
    """
    Return a random unasked question from category, filtered by difficulty if given.
    Tracks which questions have been asked this session; resets when all are exhausted.
    Returns None if the category does not exist.
    """
    _ensure_loaded()

    questions = _bank.get(category)
    if not questions:
        _log.warning("[trivia] Unknown category: %r", category)
        return None

    asked = _asked.setdefault(category, set())

    def _candidates(diff_filter):
        return [
            i for i, q in enumerate(questions)
            if i not in asked
            and (diff_filter is None or q.get("difficulty") == diff_filter)
        ]

    candidates = _candidates(difficulty)

    if not candidates:
        # Relax difficulty filter first, then reset all
        candidates = [i for i in range(len(questions)) if i not in asked]

    if not candidates:
        asked.clear()
        candidates = _candidates(difficulty) or list(range(len(questions)))

    idx = random.choice(candidates)
    asked.add(idx)
    return questions[idx]


def check_answer(question: dict, user_answer: str) -> bool:
    """
    Return True if user_answer fuzzy-matches the correct answer or any alternative.
    Uses config.TRIVIA_FUZZY_THRESHOLD (0–1 scale) as the acceptance threshold.
    """
    threshold = getattr(config, "TRIVIA_FUZZY_THRESHOLD", 0.75) * 100
    user = user_answer.strip().lower()
    candidates = [question.get("answer", "")] + list(question.get("alternatives", []))

    for candidate in candidates:
        normalized = candidate.strip().lower()
        if not normalized:
            continue
        if fuzz.ratio(user, normalized) >= threshold:
            return True
        if fuzz.partial_ratio(user, normalized) >= threshold:
            return True
    return False


def reset_session() -> None:
    """Clear all asked-question tracking for the current session."""
    global _asked
    _asked = {}
    _log.info("[trivia] Session reset — asked question tracking cleared")
