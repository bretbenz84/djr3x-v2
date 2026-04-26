"""
features/jeopardy.py - Jeopardy-style clue bank and parsing helpers.

The conversational state lives in features.games so the existing game dispatcher
can keep owning one active game at a time. This module stays deterministic:
load real clues, build a playable board, parse player/board choices, and judge
answers without making LLM calls.
"""

from __future__ import annotations

import csv
import logging
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz

import config

_log = logging.getLogger(__name__)

_BOARD_CACHE: Optional[list[dict]] = None

_VALUE_WORDS: list[tuple[str, int]] = [
    ("two thousand", 2000),
    ("sixteen hundred", 1600),
    ("one thousand six hundred", 1600),
    ("twelve hundred", 1200),
    ("one thousand two hundred", 1200),
    ("one thousand", 1000),
    ("a thousand", 1000),
    ("eight hundred", 800),
    ("six hundred", 600),
    ("five hundred", 500),
    ("four hundred", 400),
    ("three hundred", 300),
    ("two hundred", 200),
    ("one hundred", 100),
]

_PLAYER_FILLER_RE = re.compile(
    r"\b("
    r"the|players?|are|is|be|playing|player|contestants?|we|have|got|"
    r"with|please|just|meant|mean|it'?s|it\s+is|that'?s|that\s+is"
    r")\b",
    re.IGNORECASE,
)

_QUESTION_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"what|who|where|when|why|how"
    r")\s+(?:"
    r"is|are|was|were|am|be|being|been|do|does|did|would|could|should|"
    r"can|might|may"
    r")\s+",
    re.IGNORECASE,
)


def _clues_path() -> Path:
    return Path(getattr(config, "JEOPARDY_CLUES_FILE", "assets/jeopardy/clues.tsv"))


def _plain(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", (text or "").lower()).split())


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(str(value or "").replace("$", "").replace(",", "").strip())
    except Exception:
        return default


def _clean_category(category: str) -> str:
    cleaned = " ".join((category or "").strip().split())
    return cleaned or "Potpourri"


def _clean_cell(text: str) -> str:
    return " ".join((text or "").replace('\\"', '"').split())


def _valid_clue(row: dict) -> bool:
    clue = (row.get("answer") or "").strip()
    correct = (row.get("question") or "").strip()
    category = (row.get("category") or "").strip()
    if not clue or not correct or not category:
        return False
    if len(clue) < 8 or len(correct) < 2:
        return False
    visual_markers = (
        "(audio", "(video", "(image", "(photo", "(shown", "(seen",
        "seen here", "shown here", "pictured here",
    )
    combined = f"{row.get('comments') or ''} {clue}".lower()
    return not any(marker in combined for marker in visual_markers)


def load_boards() -> list[dict]:
    """Load playable boards grouped by air date and round."""
    global _BOARD_CACHE
    if _BOARD_CACHE is not None:
        return _BOARD_CACHE

    path = _clues_path()
    if not path.exists():
        _log.warning("[jeopardy] clue file missing: %s", path)
        _BOARD_CACHE = []
        return _BOARD_CACHE

    grouped: dict[tuple[str, int], dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                round_no = _to_int(row.get("round"))
                if round_no not in (1, 2):
                    continue
                if not _valid_clue(row):
                    continue
                value = _to_int(row.get("clue_value"))
                if value <= 0:
                    continue
                air_date = (row.get("air_date") or "unknown").strip()
                category = _clean_category(row.get("category") or "")
                grouped[(air_date, round_no)][category].append({
                    "category": category,
                    "value": value,
                    "clue": _clean_cell(row.get("answer") or ""),
                    "answer": _clean_cell(row.get("question") or ""),
                    "daily_double": _to_int(row.get("daily_double_value")) > 0,
                    "air_date": air_date,
                    "round": round_no,
                })
    except Exception as exc:
        _log.error("[jeopardy] failed to load clues from %s: %s", path, exc)
        _BOARD_CACHE = []
        return _BOARD_CACHE

    boards: list[dict] = []
    for (air_date, round_no), categories in grouped.items():
        playable_categories: list[dict] = []
        for name, clues in categories.items():
            by_value: dict[int, dict] = {}
            for clue in clues:
                by_value.setdefault(int(clue["value"]), clue)
            values = sorted(by_value)
            if len(values) < 5:
                continue
            selected = [dict(by_value[v]) for v in values[:5]]
            playable_categories.append({"name": name, "clues": selected})
        if len(playable_categories) >= 6:
            boards.append({
                "air_date": air_date,
                "round": round_no,
                "categories": playable_categories,
            })

    _BOARD_CACHE = boards
    _log.info("[jeopardy] loaded %d playable boards from %s", len(boards), path)
    return _BOARD_CACHE


def build_board() -> Optional[dict]:
    """Return a fresh six-category board, or None if no board can be built."""
    boards = load_boards()
    if not boards:
        return None

    source = random.choice(boards)
    selected_categories = random.sample(source["categories"], 6)
    categories: list[dict] = []
    daily_candidates: list[tuple[int, int]] = []

    for cat_idx, category in enumerate(selected_categories):
        clues: dict[int, dict] = {}
        for clue in category["clues"]:
            copy = dict(clue)
            value = int(copy["value"])
            clues[value] = copy
            if copy.get("daily_double"):
                daily_candidates.append((cat_idx, value))
        categories.append({"name": category["name"], "clues": clues})

    if not daily_candidates:
        higher_values: list[tuple[int, int]] = []
        for cat_idx, category in enumerate(categories):
            values = sorted(category["clues"])
            higher_values.extend((cat_idx, value) for value in values[len(values) // 2:])
        if higher_values:
            cat_idx, value = random.choice(higher_values)
            categories[cat_idx]["clues"][value]["daily_double"] = True

    return {
        "air_date": source["air_date"],
        "round": source["round"],
        "categories": categories,
        "remaining": sum(len(c["clues"]) for c in categories),
    }


def format_board(board: dict) -> str:
    bits: list[str] = []
    for category in board.get("categories") or []:
        values = ", ".join(str(v) for v in sorted((category.get("clues") or {}).keys()))
        bits.append(f"{category.get('name')} for {values}")
    return "; ".join(bits)


def format_scores(players: list[dict]) -> str:
    if not players:
        return "no players"
    return ", ".join(f"{p['name']}: ${int(p.get('score', 0))}" for p in players)


def _display_name(fragment: str) -> str:
    fragment = re.sub(r"[^A-Za-z0-9'\-\s.]", " ", fragment or "")
    words = [w for w in fragment.split() if w]
    cleaned: list[str] = []
    for word in words:
        if word.lower() in {
            "my", "friend", "partner", "dad", "father", "mom", "mother",
            "coworker", "boss", "supervisor", "aunt", "uncle",
        }:
            continue
        if word.isupper() and len(word) <= 4:
            cleaned.append(word)
        elif len(word) <= 3 and word.lower() == word and word.isalpha():
            cleaned.append(word.upper() if len(word) == 2 else word.title())
        else:
            cleaned.append(word[:1].upper() + word[1:])
    return " ".join(cleaned).strip()


def parse_player_names(text: str, speaker_name: Optional[str] = None, limit: int = 3) -> list[str]:
    """Parse one to three player names from a spoken roster."""
    raw = (text or "").strip()
    if not raw:
        return []

    speaker = (speaker_name or "").strip()
    normalized = raw
    normalized = re.sub(r"\bmyself\b", "me", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bI\s+am\b", "me", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bI'?m\b", "me", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\band\s+I\b", "and me", normalized, flags=re.IGNORECASE)
    normalized = normalized.replace("&", " and ")

    parts = [
        p.strip(" .!?")
        for p in re.split(r"\s*(?:,|\band\b|/|\+)\s*", normalized, flags=re.IGNORECASE)
        if p and p.strip(" .!?")
    ]
    if not parts:
        parts = [normalized]

    names: list[str] = []
    seen: set[str] = set()
    for part in parts:
        plain = _plain(part)
        if plain in {"me", "i"}:
            name = speaker or "You"
        else:
            reduced = _PLAYER_FILLER_RE.sub(" ", part)
            name = _display_name(reduced)
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        names.append(name)
        if len(names) >= limit:
            break
    return names


def _extract_value(text: str, valid_values: list[int]) -> Optional[int]:
    plain = _plain(text)
    for match in re.finditer(r"\b(\d{2,4})\b", plain):
        value = int(match.group(1))
        if value in valid_values:
            return value
    for phrase, value in _VALUE_WORDS:
        if value in valid_values and re.search(rf"\b{re.escape(phrase)}\b", plain):
            return value
    return None


def _selection_query(text: str) -> str:
    query = _plain(text)
    query = re.sub(r"\b\d{2,4}\b", " ", query)
    for phrase, _value in _VALUE_WORDS:
        query = re.sub(rf"\b{re.escape(phrase)}\b", " ", query)
    query = re.sub(
        r"\b("
        r"i|ll|i'll|will|take|choose|pick|select|category|for|dollars?|please|"
        r"give|me|lets|let|s|same|again"
        r")\b",
        " ",
        query,
    )
    return " ".join(query.split())


def parse_selection(text: str, board: dict, last_category: Optional[str] = None) -> tuple[Optional[dict], str]:
    """Parse a board selection and return (clue, error_message)."""
    categories = board.get("categories") or []
    valid_values = sorted({
        int(value)
        for category in categories
        for value in (category.get("clues") or {}).keys()
    })
    value = _extract_value(text, valid_values)
    if value is None:
        return None, "Pick a dollar value too, before my game-show circuits start smoking."

    query = _selection_query(text)
    if not query and last_category:
        query = _plain(last_category)
    if "same category" in _plain(text) and last_category:
        query = _plain(last_category)

    best_idx = None
    best_score = 0
    for idx, category in enumerate(categories):
        clues = category.get("clues") or {}
        if value not in clues:
            continue
        name = category.get("name") or ""
        score = max(
            fuzz.ratio(query, _plain(name)),
            fuzz.partial_ratio(query, _plain(name)),
            fuzz.token_set_ratio(query, _plain(name)),
        )
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx is None or best_score < int(getattr(config, "JEOPARDY_SELECTION_FUZZY_THRESHOLD", 0.58) * 100):
        available = [
            category.get("name", "that mystery category")
            for category in categories
            if value in (category.get("clues") or {})
        ]
        if available:
            return None, f"I found ${value}, but not that category. Try one of these: {', '.join(available[:6])}."
        return None, f"${value} is already gone. The board is not a vending machine, sadly."

    category = categories[best_idx]
    clue = category["clues"].pop(value)
    board["remaining"] = max(0, int(board.get("remaining", 1)) - 1)
    clue["category"] = category.get("name")
    clue["value"] = value
    return clue, ""


def answer_candidates(answer: str) -> list[str]:
    raw = (answer or "").strip()
    if not raw:
        return []

    candidates = {raw}
    without_parens = re.sub(r"\([^)]*\)", " ", raw).strip()
    if without_parens:
        candidates.add(without_parens)
    for inner in re.findall(r"\(([^)]*)\)", raw):
        inner = inner.strip()
        if inner:
            candidates.add(inner)
            if without_parens:
                candidates.add(f"{inner} {without_parens}".strip())
    for part in re.split(r"\s*(?:/|;|\bor\b)\s*", raw, flags=re.IGNORECASE):
        part = part.strip()
        if part:
            candidates.add(part)
    return [c for c in candidates if c]


def normalize_answer(text: str) -> str:
    text = _QUESTION_PREFIX_RE.sub("", text or "").strip()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def is_correct(user_answer: str, expected_answer: str) -> bool:
    threshold = int(getattr(config, "JEOPARDY_FUZZY_THRESHOLD", 0.78) * 100)
    user = normalize_answer(user_answer)
    if not user:
        return False
    for candidate in answer_candidates(expected_answer):
        expected = normalize_answer(candidate)
        if not expected:
            continue
        if user == expected:
            return True
        if fuzz.ratio(user, expected) >= threshold:
            return True
        if len(expected) >= 5 and fuzz.partial_ratio(user, expected) >= threshold + 5:
            return True
        if fuzz.token_set_ratio(user, expected) >= threshold:
            return True
    return False


def is_pass_or_timeout(text: str) -> bool:
    plain = _plain(text)
    return plain in {
        "pass", "i pass", "skip", "times up", "time up", "time s up",
        "i don t know", "i dont know", "no idea", "not sure",
    }
