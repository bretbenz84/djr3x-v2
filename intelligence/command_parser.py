"""
intelligence/command_parser.py — Local command resolution pipeline.

Resolution order:
  1. Exact match
  2. Prefix match (including personality parameter patterns)
  3. Fuzzy match  (rapidfuzz / fuzzywuzzy, threshold from config)
  4. Semantic exclusion (veto nonsensical fuzzy matches)
  5. LLM fallback → returns None
"""

import re
from collections import namedtuple

import config

try:
    from rapidfuzz import fuzz as _fuzz
except ImportError:
    from fuzzywuzzy import fuzz as _fuzz


CommandMatch = namedtuple("CommandMatch", ["command_key", "match_type", "args"])


def _similarity(a: str, b: str) -> float:
    return _fuzz.ratio(a, b) / 100.0


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _plain(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9'\s]", " ", text.lower()).split())


_GENERIC_VISUAL_TARGET_WORDS = {
    "a", "an", "the", "this", "that", "these", "those", "my", "your",
    "thing", "things", "stuff", "one", "here", "there",
}


def _has_specific_visual_target(text: str) -> bool:
    words = [w for w in _plain(text).split() if w not in _GENERIC_VISUAL_TARGET_WORDS]
    return any(len(w) > 2 for w in words)


def _parse_directed_look(normalized: str, original: str) -> dict | None:
    """
    Parse physical gaze commands: "look left", "look at this", "look down here".

    "look around" remains the normal scene-description command; this helper is
    for cases where the user is directing Rex's head/camera toward a target.
    """
    clean = _plain(normalized)
    if not clean.startswith("look "):
        return None
    if clean in {"look around", "look alive"}:
        return None

    direction = None
    if re.search(r"\b(?:the\s+)?other\s+way\b|\bopposite\s+way\b", clean):
        direction = "other_way"

    for word in ("left", "right", "up", "down"):
        if direction is None and re.search(rf"\b(?:your\s+)?{word}\b", clean):
            direction = word
            break

    if direction is None and re.search(
        r"\b(?:center|centre|front|forward|ahead|straight ahead)\b", clean
    ):
        direction = "center"

    target_hint = ""
    m_at = re.match(r"look\s+at\s+(.+)$", clean)
    if m_at:
        target_hint = m_at.group(1).strip()

    pointing_phrase = bool(re.search(
        r"\blook\s+(?:at\s+)?(?:this|that|here|there)\b|\blook\s+over\s+there\b",
        clean,
    ))
    broad_look_at = clean.startswith("look at ") and bool(target_hint)

    if direction is None and (pointing_phrase or broad_look_at):
        direction = "current"

    if direction is None:
        return None

    return {
        "direction": direction,
        "target_hint": target_hint,
        "search_target": _has_specific_visual_target(target_hint),
        "utterance": original.strip(),
    }


def _parse_visual_opinion(normalized: str, original: str) -> dict | None:
    clean = _plain(normalized)
    patterns = [
        r"^what\s+do\s+you\s+think\s+(?:of|about)\s+(?:this|that|my|the)\s+(.+)$",
        r"^what's\s+your\s+opinion\s+(?:of|on)\s+(?:this|that|my|the)\s+(.+)$",
        r"^check\s+out\s+(?:this|that|my|the)\s+(.+)$",
        r"^take\s+a\s+look\s+at\s+(?:this|that|my|the)\s+(.+)$",
    ]
    for pattern in patterns:
        m = re.match(pattern, clean)
        if not m:
            continue
        target_hint = m.group(1).strip()
        return {
            "direction": "current",
            "target_hint": target_hint,
            "search_target": _has_specific_visual_target(target_hint),
            "utterance": original.strip(),
        }
    return None


def _parse_play_options(normalized: str) -> dict | None:
    clean = _plain(normalized)
    if clean in {
        "what can you play",
        "what do you play",
        "what are you able to play",
    }:
        return {}
    return None


def _parse_themed_trivia(normalized: str, original: str) -> dict | None:
    clean = _plain(normalized)
    patterns = [
        r"^(?:play|start|run)\s+(.+?)\s+trivia(?:\s+game)?$",
        r"^(?:let's|lets)\s+(?:play|do)\s+(.+?)\s+trivia(?:\s+game)?$",
    ]
    for pattern in patterns:
        m = re.match(pattern, clean)
        if not m:
            continue
        theme = m.group(1).strip()
        if theme:
            return {"game": f"{theme} trivia"}
    return None


def _parse_wave(normalized: str, original: str) -> dict | None:
    clean = _plain(normalized)
    if not clean.startswith(("wave", "please wave", "can you wave", "can you please wave")):
        return None

    m = re.match(
        r"^(?:can you\s+)?(?:please\s+)?wave(?:\s+(?:to|at)\s+(.+))?$",
        clean,
    )
    if not m:
        return None

    target = (m.group(1) or "").strip()
    if target in {"", "me", "us", "them"}:
        target = target or "them"

    # Preserve original casing when possible for names like "JT".
    original_m = re.match(
        r"^(?:can you\s+)?(?:please\s+)?wave(?:\s+(?:to|at)\s+(.+))?$",
        original.strip(),
        re.IGNORECASE,
    )
    if original_m and original_m.group(1):
        target = original_m.group(1).strip()

    return {"target": target}


# ─── Exact-match commands ─────────────────────────────────────────────────────

EXACT_COMMANDS: dict[str, str] = {
    # Time & date
    "what time is it":                "time_query",
    "what's the time":                "time_query",
    "what is the time":               "time_query",
    "what day is it":                 "date_query",
    "what's today's date":            "date_query",
    "what is today's date":           "date_query",
    "what's the date":                "date_query",
    "what date is it":                "date_query",
    # System state
    "go to sleep":                    "sleep",
    "sleep":                          "sleep",
    "wake up":                        "wake_up",
    "be quiet":                       "quiet_mode",
    "quiet mode":                     "quiet_mode",
    "go quiet":                       "quiet_mode",
    "shut down":                      "shutdown",
    "shutdown":                       "shutdown",
    "power off":                      "shutdown",
    "turn off":                       "shutdown",
    # Memory
    "forget me":                      "forget_me",
    "delete me from your memory":     "forget_me",
    "erase me":                       "forget_me",
    "what's my name":                 "whats_my_name",
    "what is my name":                "whats_my_name",
    "do you know my name":            "whats_my_name",
    "forget everyone":                "forget_everyone",
    "forget everybody":               "forget_everyone",
    "wipe your memory":               "forget_everyone",
    "delete everyone":                "forget_everyone",
    # DJ controls
    "stop":                           "dj_stop",
    "stop music":                     "dj_stop",
    "stop the music":                 "dj_stop",
    "stop playing music":             "dj_stop",
    "stop the song":                  "dj_stop",
    "pause music":                    "dj_stop",
    "pause the music":                "dj_stop",
    "turn off the music":             "dj_stop",
    "turn the music off":             "dj_stop",
    "kill the music":                 "dj_stop",
    "cut the music":                  "dj_stop",
    "skip":                           "dj_skip",
    "skip this":                      "dj_skip",
    "skip this song":                 "dj_skip",
    "next song":                      "dj_skip",
    "next track":                     "dj_skip",
    "turn it up":                     "volume_up",
    "turn the music up":              "volume_up",
    "turn up the music":              "volume_up",
    "volume up":                      "volume_up",
    "louder":                         "volume_up",
    "turn it down":                   "volume_down",
    "turn the music down":            "volume_down",
    "turn down the music":            "volume_down",
    "volume down":                    "volume_down",
    "quieter":                        "volume_down",
    "lower the volume":               "volume_down",
    # Games
    "start trivia":                   "start_trivia",
    "play trivia":                    "start_trivia",
    "let's do trivia":                "start_trivia",
    "start i spy":                    "start_i_spy",
    "play i spy":                     "start_i_spy",
    "let's do i spy":                 "start_i_spy",
    "start eye spy":                  "start_i_spy",
    "play eye spy":                   "start_i_spy",
    "let's do eye spy":               "start_i_spy",
    "start 20 questions":             "start_20_questions",
    "play 20 questions":              "start_20_questions",
    "let's do 20 questions":          "start_20_questions",
    "start twenty questions":         "start_20_questions",
    "play twenty questions":          "start_20_questions",
    "let's do twenty questions":      "start_20_questions",
    "start jeopardy":                 "start_jeopardy",
    "play jeopardy":                  "start_jeopardy",
    "let's do jeopardy":              "start_jeopardy",
    "start word association":         "start_word_association",
    "play word association":          "start_word_association",
    "let's do word association":      "start_word_association",
    "stop the game":                  "stop_game",
    "end the game":                   "stop_game",
    "quit the game":                  "stop_game",
    "stop playing":                   "stop_game",
    # Vision
    "what do you see":                "vision_describe",
    "look around":                    "vision_describe",
    "describe what you see":          "vision_describe",
    "what's in front of you":         "vision_describe",
    "who am i":                       "vision_who_am_i",
    "do you know who i am":           "vision_who_am_i",
    "can you see me":                 "vision_who_am_i",
    # Status
    "how long have you been running": "status_uptime",
    "what's your uptime":             "status_uptime",
    "uptime":                         "status_uptime",
    "how long have you been on":      "status_uptime",
    "how long have you been awake":   "status_uptime",
    "how long have you been alive":   "status_uptime",
}


# ─── Prefix commands ──────────────────────────────────────────────────────────
# (prefix_string, command_key, arg_field_name) — sorted longest-first so the
# most specific prefix wins when multiple entries share a common root.

PREFIX_COMMANDS: list[tuple[str, str, str]] = sorted(
    [
        ("call me ",          "rename_me",    "name"),
        ("rename me to ",     "rename_me",    "name"),
        ("rename me ",        "rename_me",    "name"),
        ("play something ",   "dj_play_vibe", "vibe"),
        ("play me something ", "dj_play_vibe", "vibe"),
        ("let's play ",       "start_game",   "game"),
        ("lets play ",        "start_game",   "game"),
        ("i want to play ",   "start_game",   "game"),
        ("can we play ",      "start_game",   "game"),
        ("could we play ",    "start_game",   "game"),
        ("play a game of ",   "start_game",   "game"),
        ("start a game of ",  "start_game",   "game"),
    ],
    key=lambda t: len(t[0]),
    reverse=True,
)


# ─── Personality parameter patterns ──────────────────────────────────────────

_PARAMS = list(config.PERSONALITY_DEFAULTS.keys())
_LEVELS = list(config.PERSONALITY_NAMED_LEVELS.keys())

# Build alternation that matches each param with either underscore or space
# so "roast_intensity" matches the spoken form "roast intensity" too.
# re.escape does not escape underscores in Python 3.7+, so we handle them first.
def _param_alt(p: str) -> str:
    return "[_ ]".join(re.escape(part) for part in p.split("_"))

_RE_SET_PARAM = re.compile(
    r"(?:set|turn)\s+(" + "|".join(_param_alt(p) for p in _PARAMS) + r")"
    r"\s+(?:to|down\s+to|up\s+to)\s+"
    r"(\d+(?:\s*percent)?|" + "|".join(re.escape(l) for l in _LEVELS) + r")",
    re.IGNORECASE,
)

_RE_QUERY_PARAM = re.compile(
    r"what(?:'s| is) your (" + "|".join(_param_alt(p) for p in _PARAMS) + r")(?:\s+level)?",
    re.IGNORECASE,
)


def _canonical_param(raw: str) -> str:
    """Normalize spoken param name ('roast intensity') to stored key ('roast_intensity')."""
    return raw.strip().lower().replace(" ", "_")


def _resolve_level(raw: str) -> int:
    """Convert a level string ('90 percent', 'maximum', '47') to int 0–100."""
    raw = raw.strip().lower()
    if raw in config.PERSONALITY_NAMED_LEVELS:
        return config.PERSONALITY_NAMED_LEVELS[raw]
    digits = re.sub(r"[^\d]", "", raw)
    return max(0, min(100, int(digits))) if digits else 50


# ─── Fuzzy candidate pool ─────────────────────────────────────────────────────
# Maps candidate string → (command_key, arg_field_name | None).
# arg_field_name is set for prefix-command representatives so that the fuzzy
# step can attempt arg extraction when the rep is the best match.

_FUZZY_POOL: dict[str, tuple[str, str | None]] = {
    phrase: (key, None) for phrase, key in EXACT_COMMANDS.items()
}

for _prefix, _key, _arg in PREFIX_COMMANDS:
    _rep = _prefix.rstrip()
    if _rep not in _FUZZY_POOL:
        _FUZZY_POOL[_rep] = (_key, _arg)

for _param in _PARAMS:
    _FUZZY_POOL[f"set {_param} to"] = ("set_personality", None)
    _FUZZY_POOL[f"what's your {_param} level"] = ("query_personality", None)


# ─── Semantic exclusions ──────────────────────────────────────────────────────
# (input_fragment, blocked_command_key) — if the normalized input contains
# input_fragment AND the fuzzy winner is blocked_key, the match is vetoed.

SEMANTIC_EXCLUSIONS: list[tuple[str, str]] = [
    ("my name is",  "whats_my_name"),  # asserting a name ≠ querying Rex's memory of it
    ("your name",   "whats_my_name"),  # asking Rex's name ≠ asking what Rex calls the user
    ("your name",   "rename_me"),      # asking Rex's name ≠ rename command
    ("wake me",     "wake_up"),        # "wake me up" (song/joke) ≠ system wake command
    ("stop me",     "dj_stop"),        # figurative "stop me" ≠ DJ stop command
]


# ─── Public interface ─────────────────────────────────────────────────────────

def parse(text: str) -> CommandMatch | None:
    """
    Resolve *text* to a CommandMatch, or None to signal LLM fallback.

    Returns:
        CommandMatch(command_key, match_type, args)  — on any local match
        None                                          — LLM should handle this
    """
    normalized = _normalize(text)
    original = text.strip()

    # 1. Exact match
    if normalized in EXACT_COMMANDS:
        return CommandMatch(EXACT_COMMANDS[normalized], "exact", {})

    directed_look = _parse_directed_look(normalized, original)
    if directed_look is not None:
        return CommandMatch("directed_look", "pattern", directed_look)

    visual_opinion = _parse_visual_opinion(normalized, original)
    if visual_opinion is not None:
        return CommandMatch("directed_look", "pattern", visual_opinion)

    play_options = _parse_play_options(normalized)
    if play_options is not None:
        return CommandMatch("query_play_options", "pattern", play_options)

    themed_trivia = _parse_themed_trivia(normalized, original)
    if themed_trivia is not None:
        return CommandMatch("start_game", "pattern", themed_trivia)

    wave = _parse_wave(normalized, original)
    if wave is not None:
        return CommandMatch("wave_to", "pattern", wave)

    # 2a. Prefix match (variable-arg commands)
    for prefix, key, arg_name in PREFIX_COMMANDS:
        if normalized.startswith(prefix):
            # Match against original text (case-insensitive) to preserve
            # proper capitalization on extracted args (e.g. names from Whisper).
            pm = re.match(re.escape(prefix), original, re.IGNORECASE)
            arg_val = (original[pm.end():] if pm else normalized[len(prefix):]).strip()
            return CommandMatch(key, "prefix", {arg_name: arg_val})

    # 2b. Personality set: "set humor to 90 percent" / "turn darkness to maximum"
    m = _RE_SET_PARAM.search(normalized)
    if m:
        return CommandMatch(
            "set_personality",
            "prefix",
            {"param": _canonical_param(m.group(1)), "value": _resolve_level(m.group(2))},
        )

    # 2c. Personality query: "what's your sarcasm level"
    m = _RE_QUERY_PARAM.search(normalized)
    if m:
        return CommandMatch("query_personality", "prefix", {"param": _canonical_param(m.group(1))})

    # 3. Fuzzy match against full candidate pool
    best_score = 0.0
    best_candidate = ""
    for candidate in _FUZZY_POOL:
        score = _similarity(normalized, candidate)
        if score > best_score:
            best_score = score
            best_candidate = candidate

    if best_score >= config.COMMAND_FUZZY_THRESHOLD:
        best_key, best_arg = _FUZZY_POOL[best_candidate]

        # 4. Semantic exclusion veto
        for fragment, blocked_key in SEMANTIC_EXCLUSIONS:
            if fragment in normalized and best_key == blocked_key:
                return None

        # For prefix-representative matches, attempt trailing-arg extraction
        args: dict = {}
        if best_arg is not None:
            for prefix, key, arg_name in PREFIX_COMMANDS:
                if key == best_key and arg_name == best_arg:
                    rep = prefix.rstrip()
                    if normalized.startswith(rep):
                        tail = normalized[len(rep):].lstrip()
                        if tail:
                            args = {arg_name: tail}
                    break

        return CommandMatch(best_key, "fuzzy", args)

    # 5. LLM fallback
    return None
