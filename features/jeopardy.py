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

_NON_PLAYER_NAME_WORDS = {
    "i",
    "me",
    "my",
    "myself",
    "you",
    "we",
    "us",
}

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
    r"book|novel|play|song|album|magazine|newspaper|vehicle|ship|"
    r"invention|device|tool|machine|gadget|appliance|equipment|instrument|"
    r"object|item|thing|product|technology|material|substance|chemical|"
    r"food|drink|dish|type|kind"
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


# The J! archive abbreviates category names ("COMBINED STATE ABBREV.") and the
# TTS reads the raw token as a word ("abreev" — field 2026-08-25: the players
# couldn't tell what the category was for several rounds). Expanded for SPEECH
# only; the board/GUI and the selection matcher keep the raw dataset name.
# Keys are matched against the lowercased token with any trailing "." removed —
# only tokens that are unambiguous abbreviations belong here (no "lit", no
# "pres": those collide with real words the dataset also uses).
_CATEGORY_SPEECH_EXPANSIONS = {
    "abbrev": "abbreviations",
    "abbrevs": "abbreviations",
    "anniv": "anniversary",
    "categ": "categories",
    "dept": "department",
    "depts": "departments",
    "geog": "geography",
    "govt": "government",
    "intl": "international",
    "misc": "miscellaneous",
    "natl": "national",
    "vocab": "vocabulary",
}


def speak_category(name: str) -> str:
    """Speech-friendly form of a category name; the raw name stays everywhere else.

    Expands known dataset abbreviations and drops a trailing period so the TTS
    does not read "ABBREV.." as a word plus two stops.
    """
    tokens = (name or "").split()
    out: list[str] = []
    for token in tokens:
        bare = token.rstrip(".").lower()
        expansion = _CATEGORY_SPEECH_EXPANSIONS.get(bare)
        if expansion is None:
            out.append(token)
        elif token.isupper():
            out.append(expansion.upper())
        elif token[:1].isupper():
            out.append(expansion.capitalize())
        else:
            out.append(expansion)
    return " ".join(out).rstrip(".") or (name or "")


def format_categories(
    board: dict,
    *,
    remaining_only: bool = False,
    separator: str = ", ",
) -> str:
    return separator.join(
        speak_category(str(category.get("name") or "Potpourri"))
        for category in board.get("categories") or []
        if not remaining_only or (category.get("clues") or {})
    )


def format_board_readout(board: dict) -> str:
    """Speak the squares that are still live: categories plus their values.

    Collapses to one shared value list while the board is still even (most of a
    round), and only itemizes per category once squares disappear unevenly.
    """
    entries: list[tuple[str, list[int]]] = []
    for category in board.get("categories") or []:
        values = sorted(int(value) for value in (category.get("clues") or {}).keys())
        if values:
            entries.append((str(category.get("name") or "Potpourri"), values))
    if not entries:
        return ""

    if len({tuple(values) for _name, values in entries}) == 1:
        names = ". ".join(speak_category(name) for name, _values in entries)
        values_text = ", ".join(f"${value}" for value in entries[0][1])
        return f"{names}. Each one still has {values_text}"
    return ". ".join(
        f"{speak_category(name)} for {', '.join(f'${value}' for value in values)}"
        for name, values in entries
    )


def format_scores(players: list[dict]) -> str:
    if not players:
        return "no players"
    return ", ".join(
        f"{p['name']}: {_format_score_value(int(p.get('score', 0)))}"
        for p in players
    )


def _format_score_value(score: int) -> str:
    amount = abs(int(score))
    if score < 0:
        return f"negative ${amount}"
    return f"${amount}"


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


def parse_player_names(text: str, speaker_name: Optional[str] = None, limit: int = 4) -> list[str]:
    """Parse one to four player names from a spoken roster."""
    raw = (text or "").strip()
    if not raw:
        return []

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
        if plain in _NON_PLAYER_NAME_WORDS:
            continue
        reduced = _PLAYER_FILLER_RE.sub(" ", part)
        name = _display_name(reduced)
        if not name:
            continue
        if _plain(name) in _NON_PLAYER_NAME_WORDS:
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


# Spoken lead-ins a live player wraps an answer in ("um, I think it's Paris?").
# Stripped iteratively so stacked fillers all come off. The contraction forms
# ("what's X") matter most: _QUESTION_PREFIX_RE only matches the two-word forms.
_SPOKEN_LEADIN_RES = [
    re.compile(r"^\s*(?:um+|uh+|er+|hmm+|well|okay|ok|oh|so|hey)\b[,\s]*", re.IGNORECASE),
    re.compile(
        r"^\s*(?:i\s+(?:think|believe|guess|mean|know)|maybe|probably|possibly)\b[,\s]*(?:it'?s|it\s+is|that'?s)?[,\s]*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:the\s+answer\s+is|that\s+would\s+be|it\s+would\s+be|is\s+it|it'?s|"
        r"it\s+is|that'?s|that\s+is|how\s+about|what\s+about)\b[,\s]*",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*(?:what|who|where|when|how)'?s\b[,\s]*", re.IGNORECASE),
]

_CARDINAL_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

_ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11,
    "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
    "twentieth": 20,
}

# Regnal-number roman numerals (Henry VIII). The single letters i/v/x/l/c are
# deliberately excluded — too ambiguous with ordinary words and initials.
_ROMAN_TOKENS = {
    "ii": 2, "iii": 3, "iv": 4, "vi": 6, "vii": 7, "viii": 8, "ix": 9,
    "xi": 11, "xii": 12, "xiii": 13, "xiv": 14, "xv": 15, "xvi": 16,
    "xvii": 17, "xviii": 18, "xix": 19, "xx": 20,
}

_DIGIT_ORDINAL_RE = re.compile(r"^(\d+)(?:st|nd|rd|th)$")


def _canon_token(token: str, next_token: Optional[str] = None) -> tuple[str, bool]:
    """Canonicalize one token toward digits ("eighth"/"viii"/"8th" -> "8").
    Returns (canonical, consumed_next) — consumed_next when a tens word absorbed
    the following unit word ("forty two" -> "42")."""
    if token in _ROMAN_TOKENS:
        return str(_ROMAN_TOKENS[token]), False
    if token in _ORDINAL_WORDS:
        return str(_ORDINAL_WORDS[token]), False
    match = _DIGIT_ORDINAL_RE.match(token)
    if match:
        return match.group(1), False
    value = _CARDINAL_WORDS.get(token)
    if value is not None:
        if (
            20 <= value <= 90
            and value % 10 == 0
            and next_token is not None
            and 1 <= _CARDINAL_WORDS.get(next_token, 99) <= 9
        ):
            return str(value + _CARDINAL_WORDS[next_token]), True
        return str(value), False
    return token, False


def _canonicalize_number_tokens(text: str) -> str:
    tokens = text.split()
    out: list[str] = []
    i = 0
    while i < len(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        canon, consumed = _canon_token(tokens[i], nxt)
        out.append(canon)
        i += 2 if consumed else 1
    return " ".join(out)


def _finish_normalize(text: str) -> str:
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return _canonicalize_number_tokens(" ".join(text.split()))


def normalize_answer(text: str) -> str:
    original = (text or "").strip()
    stripped = original
    for _ in range(3):
        before = stripped
        for pattern in _SPOKEN_LEADIN_RES:
            stripped = pattern.sub("", stripped)
        stripped = _QUESTION_PREFIX_RE.sub("", stripped).strip()
        if stripped == before:
            break
    result = _finish_normalize(stripped)
    if result:
        return result
    # The whole answer WAS a "filler" word (an answer of literally "Maybe") —
    # stripping it to nothing loses the turn, so fall back to the unstripped form.
    return _finish_normalize(original)


def _spoken_number_string(plain: str) -> Optional[str]:
    """Digits for an all-number-words utterance, or None.

    Handles the shapes players actually speak: years as paired groups
    ("fourteen ninety two" -> 1492, "nineteen oh five" -> 1905) and
    multiplier forms ("two thousand" -> 2000, "eight hundred fifty" -> 850).
    Tens+unit pairs are already merged by normalize_answer's canonicalizer.
    """
    tokens = (plain or "").split()
    if not tokens:
        return None
    parts: list[str] = []
    for token in tokens:
        if token == "and":
            continue
        if token in ("oh", "o"):
            parts.append("0")
            continue
        if token.isdigit():
            parts.append(token)
            continue
        if token in ("hundred", "thousand"):
            if not parts or not parts[-1].isdigit():
                return None
            parts[-1] = str(int(parts[-1]) * (100 if token == "hundred" else 1000))
            continue
        return None
    if not parts:
        return None
    out: list[str] = []
    for part in parts:
        if out:
            prev = int(out[-1])
            cur = int(part)
            # Additive remainder after a round multiplier ("eight hundred" + "fifty").
            if prev >= 100 and prev % 100 == 0 and cur < 100:
                out[-1] = str(prev + cur)
                continue
            # Year-style pairing pads the "oh five" group ("nineteen" + "0" + "5").
            if part == "0" or (out[-1] == "0" and cur < 10):
                if out[-1] == "0":
                    out.pop()
                    out.append(f"0{cur}")
                    continue
        out.append(part)
    return "".join(out)


_SOUNDEX_MAP = {
    "b": "1", "f": "1", "p": "1", "v": "1",
    "c": "2", "g": "2", "j": "2", "k": "2", "q": "2", "s": "2", "x": "2", "z": "2",
    "d": "3", "t": "3", "l": "4", "m": "5", "n": "5", "r": "6",
}


def _soundex_code(word: str, *, truncate: bool = True) -> str:
    letters = re.sub(r"[^a-z]", "", (word or "").lower())
    if not letters:
        return ""
    codes: list[str] = []
    prev = _SOUNDEX_MAP.get(letters[0], "")
    for ch in letters[1:]:
        code = _SOUNDEX_MAP.get(ch, "")
        if code and code != prev:
            codes.append(code)
        if ch not in "hw":       # classic rule: h/w do not separate same-coded letters
            prev = code
    full = letters[0].upper() + "".join(codes)
    return (full + "000")[:4] if truncate else full


def _soundex(word: str) -> str:
    return _soundex_code(word, truncate=True)


def _phonetic_match(user: str, expected: str) -> bool:
    """Whisper renders unfamiliar proper nouns phonetically ("day cart" for
    Descartes, "shack" for Shaq). Accept when the two answers SOUND the same:
    full-length (untruncated) soundex over the whole letters-only string —
    equal, or a prefix one code apart (a dropped weak syllable) — with a
    length-ratio guard and a loose lexical co-signal so "license" can't
    phonetic-match "license & registration" via a shared prefix
    (garbled-but-right pairs measure ratio ~55-85; unrelated answers ~0-15)."""
    user_letters = re.sub(r"[^a-z]", "", user)
    expected_letters = re.sub(r"[^a-z]", "", expected)
    if len(user_letters) < 3 or len(expected_letters) < 4:
        return False
    shorter, longer = sorted((len(user_letters), len(expected_letters)))
    if shorter / longer < 0.6:
        return False
    user_code = _soundex_code(user_letters, truncate=False)
    expected_code = _soundex_code(expected_letters, truncate=False)
    if user_code != expected_code:
        short_code, long_code = sorted((user_code, expected_code), key=len)
        if not (long_code.startswith(short_code) and len(long_code) - len(short_code) <= 1):
            return False
    return fuzz.ratio(user, expected) >= 50


def _surname_match(user: str, expected: str) -> bool:
    """Real Jeopardy accepts a person's surname alone ("Poe" for Edgar Allan
    Poe). When the expected answer is multi-token and the user gave one token,
    accept an exact or phonetic match on the LAST expected token."""
    user_tokens = normalize_answer(user).split()
    expected_tokens = normalize_answer(expected).split()
    if len(user_tokens) != 1 or len(expected_tokens) < 2:
        return False
    guess = user_tokens[0]
    surname = expected_tokens[-1]
    if len(guess) < 3 or len(surname) < 3:
        return False
    if guess == surname:
        return True
    return _soundex(guess) == _soundex(surname) and fuzz.ratio(guess, surname) >= 50


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
        # Spoken numbers/years: "fourteen ninety two" for 1492. Only when the
        # expected answer actually carries digits.
        if any(ch.isdigit() for ch in expected):
            spoken = _spoken_number_string(user)
            expected_digits = re.sub(r"[^0-9]", "", expected)
            if spoken and expected_digits and spoken == expected_digits:
                return True
        if fuzz.ratio(user, expected) >= threshold:
            return True
        # Guarded substring match: a user answer this short ("ed", "an") matches
        # inside almost any longer expected string, so require some substance.
        if (
            not _requires_all_parts(candidate)
            and len(expected) >= 5
            and len(user) >= 4
            and fuzz.partial_ratio(user, expected) >= threshold + 5
        ):
            return True
        if _is_reasonable_partial(user, expected, candidate):
            return True
        if _phonetic_match(user, expected):
            return True
        if _surname_match(user_answer, candidate):
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


def _spoken(text: str) -> str:
    """Lowercase words with contractions closed up ("what's" -> "whats")."""
    lowered = (text or "").lower().replace("’", "'")
    lowered = re.sub(r"[^a-z0-9\s']", " ", lowered)
    return " ".join(lowered.replace("'", "").split())


_BOARD_NOUN = r"(?:categor(?:y|ies)|board|squares?|options|choices)"

_BOARD_REQUEST_RES = [
    # "what are the categories", "whats on the board", "what categories are left"
    re.compile(rf"\bwhat(?:s|\s+(?:is|are|was|were))?\b.{{0,40}}\b{_BOARD_NOUN}\b"),
    # "repeat the categories", "read me the board", "remind me of the categories"
    re.compile(
        rf"\b(?:repeat|reread|read|list|name|say|give\s+me|tell\s+me|remind\s+me)\b"
        rf".{{0,40}}\b{_BOARD_NOUN}\b"
    ),
    # "the categories again", "board one more time"
    re.compile(rf"\b{_BOARD_NOUN}\b.{{0,30}}\b(?:again|one\s+more\s+time)\b"),
    # "whats left", "what is still available"
    re.compile(r"\bwhat(?:s|\s+(?:is|was))\s+(?:still\s+)?(?:left|available|open|remaining)\b"),
]

# Meta shapes only — never a plausible "What is X?" response, so these are safe
# to intercept while a clue is live and an answer is on the line.
_CLUE_REPEAT_RES = [
    re.compile(r"^(?:hey\s+rex\s+)?(?:can|could|would|will)\s+you\s+(?:please\s+)?(?:repeat|say|read)\b"),
    re.compile(r"^(?:please\s+)?(?:repeat|reread|read)\s+(?:the\s+|that\s+|it\s+)?(?:clue|question|that|it)?\s*(?:again)?$"),
    re.compile(r"\b(?:say|read|repeat)\s+(?:that|it|the\s+(?:clue|question))\s+again\b"),
    re.compile(r"^(?:repeat|come\s+again|one\s+more\s+time|again\s+please|say\s+again)$"),
    re.compile(r"\bwhat\s+was\s+the\s+(?:clue|question)\b"),
    re.compile(r"\b(?:sorry\s+)?what\s+was\s+that\b"),
    re.compile(r"\b(?:didnt|did\s+not)\s+(?:hear|catch)\b"),
]


def is_board_request(text: str) -> bool:
    """True when the player is asking to hear the board again, not picking.

    A dollar value means they are picking a square, so it always wins — the
    "not that category" error already lists what is available at that value.
    """
    if _mentioned_any_value(text) is not None:
        return False
    spoken = _spoken(text)
    if not spoken:
        return False
    return any(pattern.search(spoken) for pattern in _BOARD_REQUEST_RES) or is_clue_repeat_request(text)


def is_clue_repeat_request(text: str) -> bool:
    """True for "say that again" shapes — a request to re-hear, not an answer."""
    spoken = _spoken(text)
    if not spoken:
        return False
    return any(pattern.search(spoken) for pattern in _CLUE_REPEAT_RES)


def is_pass_or_timeout(text: str) -> bool:
    plain = _plain(text)
    if re.search(r"\b(?:i\s+)?(?:don\s+t|dont|do\s+not)\s+know\b", plain):
        return True
    if "not sure" in plain or "no idea" in plain or "no clue" in plain:
        return True
    if re.search(r"\bi\s+(?:give\s+up|got\s+nothing|have\s+nothing)\b", plain):
        return True
    return plain in {
        "pass", "i pass", "skip", "skip it", "times up", "time up", "time s up",
        "i don t know", "i dont know", "dunno", "i dunno", "beats me",
        "no guess", "nothing", "next",
    }


# A giving-up phrase is a LEAD-IN when real words follow it. The trailing group
# eats the filler that separates the disclaimer from the guess ("no idea, maybe
# Lincoln", "I don't know... is it Shakespeare?") so the residual is the answer
# alone and normalize_answer's own lead-in stripping can finish the job.
_PASS_HEDGE_LEADIN_RE = re.compile(
    r"^(?:"
    r"(?:i\s+)?(?:don\s+t|dont|do\s+not)\s+know|"
    r"(?:i\s+(?:have|got)\s+)?no\s+(?:idea|clue|guess)|"
    r"(?:i\s+m\s+|im\s+)?not\s+(?:really\s+|totally\s+|entirely\s+)?sure|"
    r"i\s+(?:give\s+up|got\s+nothing|have\s+nothing)|"
    r"(?:i\s+)?dunno|beats\s+me|pass|skip(?:\s+it)?"
    r")\b"
    r"(?:\s+(?:but|though|although|however|maybe|perhaps|possibly|probably|"
    r"is\s+it|it\s+s|it\s+is|that\s+s|could\s+be|something\s+like|"
    r"i\s+(?:think|guess|believe|d\s+say|ll\s+say|ll\s+guess)|"
    r"my\s+guess\s+is|uh+|um+|er+|hmm+|well|ok(?:ay)?|so))*"
)


def strip_pass_hedge(text: str) -> str:
    """What is LEFT after a giving-up lead-in, or "" when the turn is only a pass.

    "I don't know, Paris?" is a guess with a disclaimer bolted to the front, but
    is_pass_or_timeout says True because the hedge pattern matches ANYWHERE in the
    turn — so games._jeopardy_handle recorded "No answer" with the board's own
    answer sitting in the utterance (routing audit 2026-08-13).

    The residual is used ONE WAY ONLY: it can promote a swallowed right answer to
    correct, never demote a pass into a wrong answer with a deduction. That is what
    keeps "I don't know what that is" (residual "what that is") from turning a
    shrug into a lost $800.
    """
    plain = _plain(text)
    match = _PASS_HEDGE_LEADIN_RE.match(plain)
    if match is None:
        return ""
    return plain[match.end():].strip()
