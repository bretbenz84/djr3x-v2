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
    "stop the music":                 "dj_stop",
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
