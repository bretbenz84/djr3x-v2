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

_ARTICLE_PREFIX_RE = re.compile(r"^\s*(?:a|an|the|this|that|these|those)\b", re.IGNORECASE)

_PERSON_CONTEXT_RE = re.compile(
    r"\b("
    r"who|author|wrote|writer|novelist|poet|actor|actress|singer|composer|"
    r"president|king|queen|emperor|inventor|artist|person|man|woman|he|she|"
    r"him|her|his|hers|born|died"
    r")\b",
    re.IGNORECASE,
)

_PLACE_CONTEXT_RE = re.compile(
    r"\b("
    r"where|country|city|state|capital|nation|island|continent|province|"
    r"territory|county|region|located|home\s+to"
    r")\b",
    re.IGNORECASE,
)

_THING_CONTEXT_RE = re.compile(
    r"\b("
    r"school|college|university|company|corporation|brand|team|movie|film|"
    r"book|novel|play|song|album|magazine|newspaper|vehicle|ship"
    r")\b",
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


def _board_low_value(board: dict) -> int:
    values = [
        int(clue.get("value", 0) or 0)
        for category in board.get("categories") or []
        for clue in category.get("clues") or []
    ]
    return min(values) if values else 0


def build_board(round_no: Optional[int] = None) -> Optional[dict]:
    """Return a fresh six-category board, or None if no board can be built."""
    boards = load_boards()
    if round_no in (1, 2):
        boards = [board for board in boards if int(board.get("round", 0) or 0) == round_no]
        if round_no == 2:
            double_boards = [board for board in boards if _board_low_value(board) >= 400]
            if double_boards:
                boards = double_boards
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


def format_categories(
    board: dict,
    *,
    remaining_only: bool = False,
    separator: str = ", ",
) -> str:
    return separator.join(
        str(category.get("name") or "Potpourri")
        for category in board.get("categories") or []
        if not remaining_only or (category.get("clues") or {})
    )


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


def _mentioned_any_value(text: str) -> Optional[int]:
    plain = _plain(text)
    match = re.search(r"\b(\d{2,4})\b", plain)
    if match:
        return int(match.group(1))
    for phrase, value in _VALUE_WORDS:
        if re.search(rf"\b{re.escape(phrase)}\b", plain):
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
        mentioned = _mentioned_any_value(text)
        if mentioned is not None and valid_values:
            available_values = ", ".join(f"${v}" for v in valid_values)
            return None, f"I heard ${mentioned}, but that square is not available. Try one of these values: {available_values}."
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


def _meaningful_tokens(text: str) -> list[str]:
    return [
        token
        for token in normalize_answer(text).split()
        if len(token) > 1 and token not in {"to", "of", "in", "on", "for"}
    ]


def _requires_all_parts(raw_answer: str) -> bool:
    return bool(re.search(r"\s(?:&|and)\s", raw_answer or "", re.IGNORECASE))


def _is_reasonable_partial(user: str, expected: str, raw_answer: str) -> bool:
    """Accept natural shorthand like "license" for "driver's license".

    Avoid accepting one piece of a genuinely two-part answer, such as
    "license" for "license & registration".
    """
    if _requires_all_parts(raw_answer):
        return False

    user_tokens = set(_meaningful_tokens(user))
    expected_tokens = set(_meaningful_tokens(expected))
    if not user_tokens or not expected_tokens:
        return False

    # The user supplied a specific core noun from a short modifier+noun answer.
    if user_tokens < expected_tokens and len(user_tokens) == 1 and len(expected_tokens) <= 2:
        token = next(iter(user_tokens))
        return len(token) >= 5

    # The user included all expected words plus harmless extras.
    return expected_tokens.issubset(user_tokens)


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
        if (
            not _requires_all_parts(candidate)
            and len(expected) >= 5
            and fuzz.partial_ratio(user, expected) >= threshold + 5
        ):
            return True
        if _is_reasonable_partial(user, expected, candidate):
            return True
    return False


def _answer_for_display(answer: str) -> str:
    subject = _clean_cell(answer).strip()
    subject = subject.strip(" .!?")
    subject = re.sub(r"\s*&\s*", " and ", subject)
    return subject or "unknown"


def _looks_plural_answer(answer: str) -> bool:
    plain = _plain(answer)
    if not plain:
        return False
    if re.search(r"\s(?:&|and)\s", answer or "", re.IGNORECASE):
        return True
    if plain.startswith(("these ", "those ")):
        return True
    words = plain.split()
    if not words:
        return False
    last = words[-1]
    return last.endswith("s") and not last.endswith(("ss", "us", "is"))


def _looks_like_person_answer(answer: str) -> bool:
    cleaned = _answer_for_display(answer)
    if not cleaned or cleaned[:1].islower():
        return False
    stripped = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
    capitalized = re.findall(r"\b[A-Z][a-zA-Z'.-]+\b", stripped)
    if len(capitalized) >= 2:
        return True
    return bool(re.search(r"\b(?:Jr|Sr|II|III|IV)\b", stripped))


def _response_prefix(answer: str, clue: str = "", category: str = "") -> str:
    context = f"{category or ''} {clue or ''}"
    plural = _looks_plural_answer(answer)
    if _THING_CONTEXT_RE.search(context):
        return "What are" if plural else "What is"
    if _PLACE_CONTEXT_RE.search(context):
        return "Where are" if plural else "Where is"
    if _PERSON_CONTEXT_RE.search(context) or _looks_like_person_answer(answer):
        return "Who are" if plural else "Who is"
    return "What are" if plural else "What is"


def _indefinite_article_for(subject: str) -> str:
    first = re.sub(r"[^A-Za-z0-9]", "", subject or "")
    if not first:
        return "a"
    return "an" if first[:1].lower() in "aeiou" else "a"


def _needs_indefinite_article(subject: str, prefix: str, clue: str = "") -> bool:
    if prefix != "What is":
        return False
    if not subject or not subject[:1].islower():
        return False
    if _ARTICLE_PREFIX_RE.match(subject):
        return False
    if _looks_plural_answer(subject) or "/" in subject:
        return False
    return bool(
        re.search(
            r"\b("
            r"one\s+of\s+these|one\s+of\s+those|this\s+item|this\s+object|"
            r"this\s+thing|this\s+document|this\s+type|kind\s+of"
            r")\b",
            clue or "",
            re.IGNORECASE,
        )
    )


def format_correct_response(answer: str, clue: str = "", category: str = "") -> str:
    """Format a revealed answer as a Jeopardy-style response question."""
    subject = _answer_for_display(answer)
    prefix = _response_prefix(subject, clue=clue, category=category)
    if _needs_indefinite_article(subject, prefix, clue=clue):
        subject = f"{_indefinite_article_for(subject)} {subject}"
    return f"{prefix} {subject}?"


def is_pass_or_timeout(text: str) -> bool:
    plain = _plain(text)
    if re.search(r"\b(?:i\s+)?(?:don\s+t|dont|do\s+not)\s+know\b", plain):
        return True
    if "not sure" in plain:
        return True
    return plain in {
        "pass", "i pass", "skip", "times up", "time up", "time s up",
        "i don t know", "i dont know", "no idea",
    }
