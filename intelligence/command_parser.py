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


_DIRECT_COMMAND_LEADERS = (
    "hey dj r3x ",
    "hey djr3x ",
    "hey r3x ",
    "hey rex ",
    "okay rex ",
    "ok rex ",
    "dj r3x ",
    "djr3x ",
    "r3x ",
    "rex ",
    "please ",
    "okay ",
    "ok ",
    "alright ",
)


def _strip_direct_command_leaders(clean: str) -> str:
    """Remove short polite/vocative leaders from a direct command."""
    value = clean.strip()
    changed = True
    while changed:
        changed = False
        for prefix in _DIRECT_COMMAND_LEADERS:
            if value.startswith(prefix):
                value = value[len(prefix):].strip()
                changed = True
                break
    return value


# ── Verbal pause ("one sec, be right back") → QUIET mode ─────────────────────
# Owner 2026-07-18: a natural-language way to hit the GUI pause button. Maps to
# quiet_mode (State.QUIET), which any ONNX wake word already resumes. Matching
# is clause-aware: "Hey Rex, one sec, be right back" pauses on the "be right
# back" clause. The AMBIGUOUS shapes ("give me a sec", "one sec") only count
# when the utterance is Rex-ADDRESSED — unaddressed they stay a soft impulse
# deferral (interaction's engagement snooze), not a hard pause.
_PAUSE_CORES = {
    "pause",
    "be right back",
    "ill be right back",
    "i'll be right back",
    "i will be right back",
    "brb",
    "hold that thought",
}
_PAUSE_SHUTUP_RE = re.compile(
    r"^(?:shut up|quiet|hush|zip it|pipe down) for a (?:sec|second|minute|min|moment|bit|while)$"
)
_PAUSE_ADDRESSED_ONLY_RE = re.compile(
    r"^(?:(?:give me|gimme) (?:a|one|a few|two|\d+) (?:sec|secs|second|seconds|minute|minutes|min|mins|moment|moments)"
    r"|one sec|one second|just a sec|just a second|hold on|hang on)$"
)
_REX_ADDRESSED_RE = re.compile(r"^(?:hey |okay |ok |alright )?(?:dj ?)?r[e3]x\b")


def parse_pause_command(text: str) -> bool:
    """True when *text* is a natural-language pause request ("rex, pause",
    "pause please", "one sec, be right back", "shut up for a sec")."""
    normalized = _normalize(text)
    if not normalized or len(normalized) > 80:   # commands are short; narration isn't
        return False
    addressed = bool(_REX_ADDRESSED_RE.match(_plain(normalized)))
    for clause in re.split(r"[,.;!?]| and | so ", normalized):
        clean = _strip_direct_command_leaders(_plain(clause))
        for suffix in (" please", " now", " for a sec", " for a second", " for a minute", " for a bit"):
            if clean.endswith(suffix) and clean[: -len(suffix)].strip() in _PAUSE_CORES:
                clean = clean[: -len(suffix)].strip()
                break
        if not clean:
            continue
        if clean in _PAUSE_CORES:
            return True
        if _PAUSE_SHUTUP_RE.match(clean):
            return True
        if addressed and _PAUSE_ADDRESSED_ONLY_RE.match(clean):
            return True
    return False


_SLEEP_COMMAND_CORES = {
    "go to sleep",
    "sleep",
}
_SLEEP_COMMAND_PREFIXES = (
    "please ",
    "rex ",
    "hey rex ",
    # Matched to the shutdown prefix list 2026-08-13 — "Okay Rex, go to sleep."
    # reduced to nothing and was refused by every lane.
    "okay ",
    "ok ",
    "alright ",
    "hey ",
)
_SLEEP_COMMAND_SUFFIXES = (
    " please",
)


# A trailing address or particle carries no meaning for a mode command, but it
# stopped the clause from reducing to a bare core, so the command died silently.
# Field 2026-08-13: "Go to sleep, Rex.", "Go to sleep buddy." and "Time for bed,
# go to sleep now." were each refused by EVERY lane — and because action_router's
# system.sleep evidence gate re-uses these same matchers, even a correct LLM tool
# call was vetoed. This is surface normalization only: the negation,
# object-scoping ("shut down the music") and hypothetical guards are untouched,
# because a trailing vocative is stripped only AFTER those have seen the clause.
_MODE_COMMAND_TRAILER_RE = re.compile(
    r"\s+(?:rex|r3x|dj\s+rex|buddy|bud|pal|dude|man|boy|friend|"
    r"now|already|then|please|ok|okay|alright)$",
    re.IGNORECASE,
)


def _strip_mode_command_trailers(value: str) -> str:
    """Peel trailing vocatives/particles ("..., Rex", "... buddy", "... now")."""
    previous = None
    while previous != value:
        previous = value
        value = _MODE_COMMAND_TRAILER_RE.sub("", value).strip()
    return value


def _reduce_sleep_clause(clean: str) -> str:
    """Peel polite prefixes, suffixes and trailing vocatives from one clause."""
    value = clean.strip()
    changed = True
    while changed:
        changed = False
        for prefix in _SLEEP_COMMAND_PREFIXES:
            if value.startswith(prefix):
                value = value[len(prefix):].strip()
                changed = True
        for suffix in _SLEEP_COMMAND_SUFFIXES:
            if value.endswith(suffix):
                value = value[: -len(suffix)].strip()
                changed = True
        stripped = _strip_mode_command_trailers(value)
        if stripped != value:
            value = stripped
            changed = True
    return value


def is_standalone_sleep_command(text: str) -> bool:
    """True only for short, direct sleep commands, not embedded narration.

    Clause-aware since 2026-08-13, mirroring is_standalone_shutdown_command —
    "Time for bed, go to sleep now." never reduced under the whole-string match,
    so the sleep never happened. The same negation/hypothetical guard shutdown
    uses runs per clause, so "don't go to sleep" and "why would I sleep" are
    still refused, and narration ("the baby wouldn't go to sleep") never reduces
    to a bare core.
    """
    if not text or not text.strip():
        return False
    for clause in _SHUTDOWN_CLAUSE_SPLIT_RE.split(text.lower()):
        clean = _plain(clause)
        if not clean:
            continue
        if _SHUTDOWN_NEGATION_GUARD_RE.search(clean):
            continue
        if _reduce_sleep_clause(clean) in _SLEEP_COMMAND_CORES:
            return True
    return False


# Full-process shutdown is a heavier action than sleep (it exits main.py and
# hands control back to the always-on supervisor), so guard it the same way:
# only a short, direct phrase triggers it — never narration like "I had to shut
# down my old server yesterday".
_SHUTDOWN_COMMAND_CORES = {
    "shut down",
    "shutdown",
    "shut down rex",
    "shutdown rex",
    "power down",
    "power down rex",
    "power off",
    "power off rex",
    "turn off",
    "turn yourself off",
}
_SHUTDOWN_COMMAND_PREFIXES = (
    "please ",
    "rex ",
    "hey rex ",
    "okay ",
    "ok ",
)
_SHUTDOWN_COMMAND_SUFFIXES = (
    " please",
    " now",
)
# Split an utterance into candidate clauses so an embedded "shut down" is found
# even when it trails frustration ("shut up, shut down") or another clause
# ("stop talking and shut down").
_SHUTDOWN_CLAUSE_SPLIT_RE = re.compile(
    r"[,.;:!?]|\b(?:and|then|but|so|or)\b", re.IGNORECASE
)
# A clause that is negated, hypothetical, or interrogative is NOT a command — a
# destructive kill-switch must never fire on "don't shut down" / "why would I
# shut down" / "should I shut down" / "can you shut down the music".
_SHUTDOWN_NEGATION_GUARD_RE = re.compile(
    r"\b(?:don'?t|do not|never|can'?t|cannot|won'?t|will not|"
    r"why|how|when|whether|would|could|should|if|whenever|"
    r"can you|could you|would you|do you)\b",
    re.IGNORECASE,
)
# Per-clause leaders peeled in ADDITION to the polite prefixes, so a comma-less
# "shut up shut down" or "please just shut down" reduces to the bare core.
_SHUTDOWN_CLAUSE_LEADERS = (
    "shut up ",
    "stop talking ",
    "no wait ",
    "wait ",
    "stop ",
    "fine ",
    "alright ",
    "well ",
    "hey ",
    "oh ",
    "just ",
)


def _reduce_shutdown_clause(clean: str) -> str:
    """Peel polite prefixes/leaders, suffixes and trailing vocatives from one clause."""
    value = clean.strip()
    changed = True
    while changed:
        changed = False
        for prefix in _SHUTDOWN_COMMAND_PREFIXES + _SHUTDOWN_CLAUSE_LEADERS:
            if value.startswith(prefix):
                value = value[len(prefix):].strip()
                changed = True
        for suffix in _SHUTDOWN_COMMAND_SUFFIXES:
            if value.endswith(suffix):
                value = value[: -len(suffix)].strip()
                changed = True
        # "Shut down, Rex." / "power off buddy" — an address is not an object, but
        # it kept the clause from reducing to a bare core (field 2026-08-13). Runs
        # after the negation/object guards have already seen the clause.
        stripped = _strip_mode_command_trailers(value)
        if stripped != value:
            value = stripped
            changed = True
    return value


# Whisper near-homophones of "shut down" seen in the field ("Cut down.",
# 2026-07-30, wake model at 0.945). Accepted ONLY as confirmation of an
# acoustic shut_down wake hit — the wake model already heard the phrase; the
# transcript just spells it wrong. Never used for typed text or ordinary
# spoken turns, and "look down" (the phrase the confirm gate exists to
# reject) is deliberately NOT here.
_SHUTDOWN_WAKE_HOMOPHONE_CORES = {
    "cut down",
    "shot down",
    "shut town",
    "shet down",
    "shut it down",
}
# NOT included: "sit down" (could be aimed at a pet — powering off on a
# marginal wake hit there is worse than one missed shutdown) and "look down"
# (the very phrase this confirm gate exists to protect).


def is_shutdown_wake_confirmation(text: str) -> bool:
    """True when *text* confirms an acoustic shut_down wake hit.

    Same clause/negation discipline as is_standalone_shutdown_command, but the
    reduced clause may also be a known Whisper near-homophone of "shut down".
    Only call this when the shut_down wake model already fired.
    """
    if is_standalone_shutdown_command(text):
        return True
    if not text or not text.strip():
        return False
    for clause in _SHUTDOWN_CLAUSE_SPLIT_RE.split(text.lower()):
        clean = _plain(clause)
        if not clean:
            continue
        if _SHUTDOWN_NEGATION_GUARD_RE.search(clean):
            continue
        if _reduce_shutdown_clause(clean) in _SHUTDOWN_WAKE_HOMOPHONE_CORES:
            return True
    return False


# Polite request leaders accepted ONLY on the request paths ("Can you shut
# down, please?"). The deterministic standalone classifiers keep rejecting
# these (the negation guard's "can you" protects "can you shut down the
# music"); this just verifies the utterance really contains the direct phrase.
# Desire-form directives added 2026-08-03: "I will talk to you later, and I
# would like you to shut down." — the clause's own "would" tripped the negation
# guard, the leader regex didn't cover "I would like you to", and Rex answered
# "Powering down." as a FAREWELL QUIP without powering down. "I want/need you
# to <core>" is as direct as an imperative; only the surface is polite.
_POLITE_REQUEST_LEADER_RE = re.compile(
    r"^(?:can|could|would|will)\s+you\s+(?:please\s+)?"
    r"|^i\s+(?:would\s+like|'?d\s+like|want|need)\s+(?:for\s+)?you\s+to\s+"
    r"|^i'?d\s+like\s+you\s+to\s+",
    re.IGNORECASE,
)


def is_shutdown_request(text: str) -> bool:
    """is_standalone_shutdown_command, plus polite direct requests ('can you
    shut down, please?'). Object-scoped ('can you shut down the music'),
    negated, and hypothetical clauses are still rejected."""
    if is_standalone_shutdown_command(text):
        return True
    if not text or not text.strip():
        return False
    for clause in _SHUTDOWN_CLAUSE_SPLIT_RE.split(text.lower()):
        clean = _plain(clause)
        if not clean:
            continue
        stripped = _POLITE_REQUEST_LEADER_RE.sub("", clean)
        if stripped == clean:
            continue  # no request leader — the standalone check already ruled
        if _SHUTDOWN_NEGATION_GUARD_RE.search(stripped):
            continue
        if _reduce_shutdown_clause(stripped) in _SHUTDOWN_COMMAND_CORES:
            return True
    return False


def is_sleep_request(text: str) -> bool:
    """is_standalone_sleep_command, plus polite direct requests ('can you go
    to sleep, please?')."""
    if is_standalone_sleep_command(text):
        return True
    if not text or not text.strip():
        return False
    for clause in _SHUTDOWN_CLAUSE_SPLIT_RE.split(text.lower()):
        clean = _plain(clause)
        if not clean:
            continue
        stripped = _POLITE_REQUEST_LEADER_RE.sub("", clean)
        if stripped == clean:
            continue
        if is_standalone_sleep_command(stripped):
            return True
    return False


def is_standalone_shutdown_command(text: str) -> bool:
    """True when any clause of the utterance is a direct full-shutdown command.

    Clause-aware so "shut up, shut down" still triggers, but a clause that is
    object-scoped ("shut down the music"), narrated ("I had to shut down my
    server"), negated ("don't shut down"), or interrogative ("why would I shut
    down") never reduces to a bare core and so is rejected — preserving the
    destructive-action safety of the original whole-string match.
    """
    if not text or not text.strip():
        return False
    for clause in _SHUTDOWN_CLAUSE_SPLIT_RE.split(text.lower()):
        clean = _plain(clause)
        if not clean:
            continue
        if _SHUTDOWN_NEGATION_GUARD_RE.search(clean):
            continue
        if _reduce_shutdown_clause(clean) in _SHUTDOWN_COMMAND_CORES:
            return True
    return False


_GENERIC_VISUAL_TARGET_WORDS = {
    "a", "an", "the", "this", "that", "these", "those", "my", "your",
    "thing", "things", "stuff", "one", "here", "there",
}
_BARE_LOOK_DIRECTIONS = {
    "left": "left",
    "right": "right",
    "up": "up",
    "down": "down",
    "center": "center",
    "centre": "center",
}
# Pitch words that modify a gaze. "lower"/"upper" ride along because "look to
# your lower left" is the same physical request as "look down and to your left".
# "below"/"beneath" are deliberately EXCLUDED: they usually relate a target to
# another object ("the dog below the table"), not Rex's own gaze axis.
_PITCH_LOOK_WORDS = r"(?:down|downward|downwards|lower|up|upward|upwards|upper)"
_PITCH_LOOK_AXIS = {
    "down": "down", "downward": "down", "downwards": "down", "lower": "down",
    "up": "up", "upward": "up", "upwards": "up", "upper": "up",
}
# The embedded form captures a BOUNDED clause -- at most two direction tokens
# joined by "and"/"then"/","/"-"/"to"/"into" -- instead of the single word right
# after "look", so an embedded compound keeps both axes. The bound is the point:
# an unbounded tail would read "Look up and I'll be right back" as up_right.
_EMBEDDED_LOOK_DIRECTION_RE = re.compile(
    r"(?:^|[.!?]\s+)(?:please\s+)?look\s+(?P<clause>"
    r"(?:to\s+|into\s+)?(?:your\s+|the\s+)?"
    r"(?:left|right|up|down|lower|upper)"
    r"(?:\s*(?:,|-|and|then)?\s*(?:to\s+|into\s+)?(?:your\s+|the\s+)?"
    r"(?:left|right|up|down|lower|upper))?"
    r")\b",
    re.IGNORECASE,
)


def _look_axis_direction(clean: str) -> str | None:
    """Pick the gaze direction from a cleaned "look ..." clause.

    Yaw and pitch are DIFFERENT servo channels (neck vs headlift+headtilt), so a
    diagonal is physically expressible and has to survive the parse as one value:
    "down_left", "up_right". The old code walked ("left", "right", "up", "down")
    and broke on the first hit -- tuple order, not word order -- so yaw always won
    and the pitch of every compound phrasing was dropped on the floor. Field
    2026-08-13 21:01-21:03: the owner's dog was down and to Rex's left and he said
    so five ways ("Look down and to your left.", "Look down into your left. You'll
    see him."); every one parsed to a bare "left", so Rex swung his head left at
    standing height, saw nothing, and answered "What am I looking for?".
    """
    if re.search(r"\b(?:the\s+)?other\s+way\b|\bopposite\s+way\b", clean):
        return "other_way"

    yaw = None
    for word in ("left", "right"):
        if re.search(rf"\b(?:your\s+)?{word}\b", clean):
            yaw = word
            break

    pitch = None
    m_pitch = re.search(rf"\b(?:your\s+)?(?P<pitch>{_PITCH_LOOK_WORDS})\b", clean)
    if m_pitch:
        pitch = _PITCH_LOOK_AXIS.get(m_pitch.group("pitch").lower())

    if yaw and pitch:
        return f"{pitch}_{yaw}"
    if yaw:
        return _BARE_LOOK_DIRECTIONS.get(yaw, yaw)
    if pitch:
        return _BARE_LOOK_DIRECTIONS.get(pitch, pitch)
    if re.search(r"\b(?:center|centre|front|forward|ahead|straight ahead)\b", clean):
        return "center"
    return None
_LOOK_DIRECTION_META_RE = re.compile(
    r"^look\s+(?:to\s+)?(?:your\s+)?(?:left|right|up|down)\s+"
    r"(?:is|was|means|meant|sounds|refers|phrase|word)\b",
    re.IGNORECASE,
)


def _has_specific_visual_target(text: str) -> bool:
    words = [w for w in _plain(text).split() if w not in _GENERIC_VISUAL_TARGET_WORDS]
    return any(len(w) > 2 for w in words)


def _parse_directed_look(normalized: str, original: str) -> dict | None:
    """
    Parse physical gaze commands: "look left", "look at this", "look down here".

    "look around" remains the normal scene-description command; this helper is
    for cases where the user is directing Rex's head/camera toward a target.
    """
    clean = _strip_direct_command_leaders(_plain(normalized))
    # A bare directional word ("down", "up", "left", "right", "center") is NOT a
    # gaze command on its own: short STT fragments routinely clip a longer phrase
    # down to a single direction (e.g. "shut down" → "down"), and firing a silent
    # head-turn for that reads as Rex ignoring the user. Require an explicit
    # "look" — as a prefix ("look down") or embedded ("... look down") — instead.
    if not clean.startswith("look "):
        embedded = _EMBEDDED_LOOK_DIRECTION_RE.search(original or "")
        if embedded is None:
            return None
        direction = _look_axis_direction(_plain(embedded.group("clause")))
        if direction is None:
            return None
        return {
            "direction": direction,
            "target_hint": "",
            "search_target": False,
            "utterance": original.strip(),
        }
    if _LOOK_DIRECTION_META_RE.search(clean):
        return None
    if clean in {"look around", "look alive"}:
        return None

    direction = _look_axis_direction(clean)

    target_hint = ""
    m_at = re.match(r"look\s+at\s+(.+)$", clean)
    if m_at:
        target_hint = m_at.group(1).strip()
    m_for = re.match(r"look\s+for\s+(.+)$", clean)
    if m_for:
        target_hint = m_for.group(1).strip()

    pointing_phrase = bool(re.search(
        r"\blook\s+(?:at\s+)?(?:this|that|here|there)\b|\blook\s+over\s+there\b",
        clean,
    ))
    broad_look_at = clean.startswith("look at ") and bool(target_hint)
    search_phrase = clean.startswith("look for ") and bool(target_hint)

    if direction is None and (pointing_phrase or broad_look_at or search_phrase):
        direction = "current"

    if direction is None:
        return None

    return {
        "direction": direction,
        "target_hint": target_hint,
        "search_target": search_phrase or _has_specific_visual_target(target_hint),
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


def _parse_forget_specific(original: str) -> dict | None:
    try:
        from memory import forgetting
    except Exception:
        return None
    target = forgetting.extract_specific_forget_target(original)
    if not target:
        return None
    return {"target": target}


def _parse_memory_review(normalized: str, original: str) -> dict | None:
    clean = _plain(normalized)
    m = re.match(
        r"^(?:what\s+do\s+you\s+remember|what\s+do\s+you\s+know)"
        r"\s+about\s+(.+?)(?:\s+including\s+sensitive\s+memories)?$",
        clean,
    )
    if not m:
        return None
    target = m.group(1).strip()
    include_sensitive = bool(re.search(r"\bincluding\s+sensitive\s+memories\b", clean))
    original_m = re.match(
        r"^(?:what\s+do\s+you\s+remember|what\s+do\s+you\s+know)"
        r"\s+about\s+(.+?)(?:\s+including\s+sensitive\s+memories)?[?.!]*$",
        original.strip(),
        re.IGNORECASE,
    )
    if original_m:
        target = original_m.group(1).strip()
    if not _looks_like_memory_review_target(target):
        return None
    return {
        "target": target,
        "self_ref": target.lower() in {"me", "myself", "i"},
        "include_sensitive": include_sensitive,
    }


_MEMORY_REVIEW_TARGET_RE = re.compile(
    r"\b("
    r"me|myself|i|my|mine|my\s+|our\s+|person|people|friend|partner|"
    r"wife|husband|mom|mother|dad|father|brother|sister|kid|child|son|"
    r"daughter|bret|daniel|jeff|joy|jt"
    r")\b",
    re.IGNORECASE,
)


def _looks_like_memory_review_target(target: str) -> bool:
    clean = _plain(target)
    if not clean:
        return False
    if _MEMORY_REVIEW_TARGET_RE.search(clean):
        return True
    return bool(re.match(r"^[A-Z][A-Za-z'_-]{1,30}$", (target or "").strip()))


def _parse_memory_forget_fact(normalized: str, original: str) -> dict | None:
    clean = _plain(normalized)
    m = re.match(r"^(?:forget|delete|remove|erase)\s+that\s+(.+)$", clean)
    if not m:
        return None
    statement = m.group(1).strip()
    original_m = re.match(
        r"^(?:forget|delete|remove|erase)\s+that\s+(.+?)[?.!]*$",
        original.strip(),
        re.IGNORECASE,
    )
    if original_m:
        statement = original_m.group(1).strip()
    return {"statement": statement}


def _parse_memory_boundary(normalized: str, original: str) -> dict | None:
    clean = _plain(normalized)
    if clean in {
        "don't remember that",
        "do not remember that",
        "dont remember that",
        "don't store that",
        "do not store that",
        "dont store that",
        "don't save that",
        "do not save that",
        "dont save that",
        "forget that",
        "forget i said that",
        "forget i just said that",
        "forget what i said",
        "forget what i just said",
        "forget that thing i said",
        "forgot i said that",
        "forgot i say that",
    }:
        return {"scope": "recent"}
    return None


# "That's wrong, X" states outright that Rex got something wrong — the lead-in IS
# the evidence. "Actually / no / nope" is only a DISCOURSE MARKER: people open
# ordinary elaborations with it constantly, so on that branch the REMAINDER has to
# carry its own correction evidence or the turn belongs in conversation.
_CORRECTION_LEAD_INS = (
    (r"^(?:that's|that is|you got that|you have that)\s+wrong\s*,?\s+(.+)$", True),
    (r"^(?:actually|no|nope)\s*,?\s+(.+)$", False),
)

# A stored fact anchored to a NAMED person — the only third-person shape
# _execute_memory_correct_fact_command can actually resolve and write.
_CORRECTION_NAMED_FACT_RE = re.compile(
    r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}"
    r"(?:'s\s+\S|\s+(?:is|are|was|were|has|have|goes|lives|likes|loves|hates|"
    r"dislikes|prefers|avoids|works|wants|plays|collects)\b)",
)

# "his/her/their (first|last|…) name is X" — an identity correction with no name
# in subject position. "my name is X" stays out: first person is handled by the
# contextual-reply guard below (a sensitive reply must not become a memory write).
_CORRECTION_NAME_ATTR_RE = re.compile(
    r"^(?:his|her|their)\s+(?:first\s+|last\s+|full\s+|real\s+|middle\s+)?name\s+is\s+\S",
    re.IGNORECASE,
)


def _correction_carries_fact_evidence(correction: str) -> bool:
    """True when a bare 'actually/no/nope' lead-in is followed by something that
    is genuinely a memory correction rather than the speaker elaborating.

    Field 2026-08-13 11:30: Rex asked "what's the most annoying part of calibrating
    those sensors?" and the answer — "Actually, the most annoying part is figuring
    out where to mount them on your body." — matched the bare marker, so Rex replied
    "Corrected. I now have Bret Benziger as the most annoying part: ..." and wrote
    `the_most_annoying_part` into his facts. Answering his own question is not a
    correction; the marker alone proves nothing.
    """
    if _plain(correction).startswith("call me "):
        return True                              # explicit rename
    if _CORRECTION_NAME_ATTR_RE.match(correction.strip()):
        return True
    return bool(_CORRECTION_NAMED_FACT_RE.match(correction.strip()))


def _parse_memory_correct_fact(normalized: str, original: str) -> dict | None:
    clean = _plain(normalized)
    for pattern, lead_in_is_evidence in _CORRECTION_LEAD_INS:
        m = re.match(pattern, clean)
        if not m:
            continue
        correction = m.group(1).strip()
        original_m = re.match(pattern, original.strip(), re.IGNORECASE)
        if original_m:
            correction = original_m.group(1).strip(" .!?")
        if correction and _plain(correction) not in {
            "that is wrong",
            "that's wrong",
            "thats wrong",
            "wrong",
            "incorrect",
        }:
            correction_plain = _plain(correction)
            if (
                not correction_plain.startswith("call me ")
                and re.match(r"^(?:i|i'm|im|me|my|we|we're|were|our)\b", correction_plain)
            ):
                return None
            if not lead_in_is_evidence and not _correction_carries_fact_evidence(correction):
                return None
            return {"correction": correction}
    return None


def _parse_memory_remember_fact(normalized: str, original: str) -> dict | None:
    clean = _plain(normalized)
    m = re.match(r"^remember\s+that\s+(.+)$", clean)
    if not m:
        return None
    statement = m.group(1).strip()
    if _looks_like_remembered_anecdote(statement):
        return None
    original_m = re.match(
        r"^remember\s+that\s+(.+?)[?.!]*$",
        original.strip(),
        re.IGNORECASE,
    )
    if original_m:
        statement = original_m.group(1).strip()
    return {"statement": statement}


def _looks_like_remembered_anecdote(statement: str) -> bool:
    """Reject reminiscence prompts like "remember that time..." as memory writes."""
    clean = _plain(statement)
    return bool(re.match(
        r"^(?:one\s+)?(?:time|day|night|morning|afternoon|evening|week|"
        r"weekend|summer|winter|year)\b",
        clean,
    ))


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
    "resume":                         "wake_up",
    "resume talking":                 "wake_up",
    "talk again":                     "wake_up",
    "speak again":                    "wake_up",
    "stop being quiet":               "wake_up",
    "exit quiet mode":                "wake_up",
    "be quiet":                       "quiet_mode",
    "quiet mode":                     "quiet_mode",
    "go quiet":                       "quiet_mode",
    "shut down":                      "shutdown",
    "shutdown":                       "shutdown",
    "shut down rex":                  "shutdown",
    "shutdown rex":                   "shutdown",
    "power off":                      "shutdown",
    "power off rex":                  "shutdown",
    "power down":                     "shutdown",
    "power down rex":                 "shutdown",
    "turn off":                       "shutdown",
    "turn yourself off":              "shutdown",
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
    "muzzle":                         "dj_stop",
    "muzzle it":                      "dj_stop",
    "muzzle the music":               "dj_stop",
    "muzzle the jukebox":             "dj_stop",
    "muzzle the song":                "dj_stop",
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
    # Room climate (BMP280/BME280 on the drive base — env block in telemetry)
    "how hot is it in here":          "status_climate",
    "how hot is it":                  "status_climate",
    "how cold is it in here":         "status_climate",
    "how cold is it":                 "status_climate",
    "how warm is it in here":         "status_climate",
    "how warm is it":                 "status_climate",
    "what's the temperature inside":  "status_climate",
    "what is the temperature inside": "status_climate",
    "what's the temperature in here": "status_climate",
    "what is the temperature in here": "status_climate",
    "what's the temperature":         "status_climate",
    "what is the temperature":        "status_climate",
    "what's the temp in here":        "status_climate",
    "temperature":                    "status_climate",
    "how humid is it in here":        "status_climate",
    "how humid is it":                "status_climate",
    "what's the humidity":            "status_climate",
    "what is the humidity":           "status_climate",
    "humidity":                       "status_climate",
    "what's the air pressure":        "status_climate",
    "what is the air pressure":       "status_climate",
    "what's the barometric pressure": "status_climate",
    "air pressure":                   "status_climate",
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


_RENAME_ARG_BLOCK_RE = re.compile(
    r"^(?:when|if|because|after|before|while|until|at|on|in|for|about|"
    r"later|maybe|both|back|again|someone|somebody|everyone|everybody|"
    r"me|you|your)\b",
    re.IGNORECASE,
)
_DEFERRED_OR_META_ARG_RE = re.compile(
    r"\b(?:later|sometime|some\s+time|tomorrow|tonight|this\s+weekend|"
    r"next\s+week|next\s+time|after|before|when|if|because)\b|"
    r"^(?:is|was|means|meant|sounds|phrase|word)\b",
    re.IGNORECASE,
)
_GAME_ARG_BLOCK_RE = re.compile(
    r"^(?:with|outside|inside|around|by\s+ear|it\s+by\s+ear)\b",
    re.IGNORECASE,
)


def _valid_prefix_arg(command_key: str, arg_value: str) -> bool:
    clean = _plain(arg_value)
    if not clean:
        return False
    if command_key == "rename_me":
        return not _RENAME_ARG_BLOCK_RE.search(clean)
    if command_key in {"start_game", "dj_play_vibe"}:
        if _DEFERRED_OR_META_ARG_RE.search(clean):
            return False
        if command_key == "start_game" and _GAME_ARG_BLOCK_RE.search(clean):
            return False
    return True


# Game-start detection that tolerates a leading conversational clause ("I'm good, but let's
# play 20 questions") and bare game names ("20 questions") — phrasings that used to miss the
# start_game prefix and fall through to the canned "here are my games" list. Reuses the
# play-verb prefixes + _valid_prefix_arg, so deferred ("...trivia later"), narrated ("we
# played trivia"), and idiom ("play it by ear") uses stay blocked exactly as before.
_GAME_START_PREFIXES: tuple[str, ...] = tuple(
    prefix for prefix, key, _arg in PREFIX_COMMANDS if key == "start_game"
)
_KNOWN_GAME_NAMES = frozenset({
    "20 questions", "twenty questions", "trivia", "jeopardy",
    "i spy", "eye spy", "word association",
})
_GAME_CLAUSE_SPLIT_RE = re.compile(r"\s*(?:,|;|\bbut\b|\bso\b|\band then\b)\s+", re.IGNORECASE)


def _parse_game_start(normalized: str) -> dict | None:
    """Resolve a 'start this game' request to {'game': name}, even behind a leading clause or
    as a bare game name. Returns None for narrated/deferred/idiom uses (still vetoed by
    _valid_prefix_arg, exactly like the plain prefix path)."""
    for clause in _GAME_CLAUSE_SPLIT_RE.split(normalized):
        clause = clause.strip()
        if not clause:
            continue
        for prefix in _GAME_START_PREFIXES:
            if clause.startswith(prefix):
                arg = clause[len(prefix):].strip()
                if arg and _valid_prefix_arg("start_game", arg):
                    return {"game": arg}
        if _plain(clause) in _KNOWN_GAME_NAMES:
            return {"game": clause}
    return None


# ─── Personality parameter patterns ──────────────────────────────────────────

_PARAMS = list(config.PERSONALITY_DEFAULTS.keys())
_LEVELS = list(config.PERSONALITY_NAMED_LEVELS.keys())

# Build alternation that matches each param with either underscore or space
# so "roast_intensity" matches the spoken form "roast intensity" too.
# re.escape does not escape underscores in Python 3.7+, so we handle them first.
def _param_alt(p: str) -> str:
    return "[_ ]".join(re.escape(part) for part in p.split("_"))

_DIRECT_REQUEST_PREFIX = (
    r"(?:(?:please|kindly)\s+|"
    r"(?:(?:can|could|would)\s+you\s+(?:please\s+)?)?)"
)
_DIRECT_REQUEST_SUFFIX = r"(?:\s+please)?\s*$"

_RE_SET_PARAM = re.compile(
    r"^" + _DIRECT_REQUEST_PREFIX +
    r"(?:set|turn)\s+(" + "|".join(_param_alt(p) for p in _PARAMS) + r")"
    r"\s+(?:to|down\s+to|up\s+to)\s+"
    r"(\d+(?:\s*percent)?|" + "|".join(re.escape(l) for l in _LEVELS) + r")"
    + _DIRECT_REQUEST_SUFFIX,
    re.IGNORECASE,
)

_RE_QUERY_PARAM = re.compile(
    r"^" + _DIRECT_REQUEST_PREFIX +
    r"(?:tell\s+me\s+)?what(?:'s| is)\s+your\s+"
    r"(" + "|".join(_param_alt(p) for p in _PARAMS) + r")(?:\s+level)?"
    + _DIRECT_REQUEST_SUFFIX,
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

    if is_standalone_sleep_command(original):
        return CommandMatch("sleep", "exact", {})

    if is_standalone_shutdown_command(original):
        return CommandMatch("shutdown", "exact", {})

    # 1. Exact match — punctuation-tolerant: ASR finalizes commands with a
    # trailing period ("Forget me."), and _normalize keeps it, so the exact
    # lookup missed and the command degraded to fuzzy — which is execution-
    # disabled (field 2026-08-02 14:07: bare 'Forget me.' was refused).
    if normalized in EXACT_COMMANDS:
        return CommandMatch(EXACT_COMMANDS[normalized], "exact", {})
    plain = _plain(original)
    if plain in EXACT_COMMANDS:
        return CommandMatch(EXACT_COMMANDS[plain], "exact", {})

    # Verbal pause → quiet mode ("rex, pause" / "one sec, be right back").
    # After the exact table so "pause music" still routes to dj_stop.
    if parse_pause_command(original):
        return CommandMatch("quiet_mode", "pattern", {"flavor": "brb"})

    memory_boundary = _parse_memory_boundary(normalized, original)
    if memory_boundary is not None:
        return CommandMatch("memory_boundary", "pattern", memory_boundary)

    memory_review = _parse_memory_review(normalized, original)
    if memory_review is not None:
        return CommandMatch("memory_review", "pattern", memory_review)

    memory_remember = _parse_memory_remember_fact(normalized, original)
    if memory_remember is not None:
        return CommandMatch("memory_remember_fact", "pattern", memory_remember)

    memory_forget = _parse_memory_forget_fact(normalized, original)
    if memory_forget is not None:
        return CommandMatch("memory_forget_fact", "pattern", memory_forget)

    memory_correct = _parse_memory_correct_fact(normalized, original)
    if memory_correct is not None:
        return CommandMatch("memory_correct_fact", "pattern", memory_correct)

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

    game_start = _parse_game_start(normalized)
    if game_start is not None:
        return CommandMatch("start_game", "pattern", game_start)

    wave = _parse_wave(normalized, original)
    if wave is not None:
        return CommandMatch("wave_to", "pattern", wave)

    forget_specific = _parse_forget_specific(original)
    if forget_specific is not None:
        return CommandMatch("forget_specific", "pattern", forget_specific)

    # 2a. Prefix match (variable-arg commands)
    for prefix, key, arg_name in PREFIX_COMMANDS:
        if normalized.startswith(prefix):
            # Match against original text (case-insensitive) to preserve
            # proper capitalization on extracted args (e.g. names from Whisper).
            pm = re.match(re.escape(prefix), original, re.IGNORECASE)
            arg_val = (original[pm.end():] if pm else normalized[len(prefix):]).strip()
            if not _valid_prefix_arg(key, arg_val):
                return None
            return CommandMatch(key, "prefix", {arg_name: arg_val})

    # 2b. Personality set: "set humor to 90 percent" / "turn darkness to maximum"
    plain = _plain(normalized)
    m = _RE_SET_PARAM.match(plain)
    if m:
        return CommandMatch(
            "set_personality",
            "prefix",
            {"param": _canonical_param(m.group(1)), "value": _resolve_level(m.group(2))},
        )

    # 2c. Personality query: "what's your sarcasm level"
    m = _RE_QUERY_PARAM.match(plain)
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
        if best_key == "sleep" and not is_standalone_sleep_command(original):
            return None

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
                        if tail and _valid_prefix_arg(best_key, tail):
                            args = {arg_name: tail}
                    break
            if not args:
                return None

        return CommandMatch(best_key, "fuzzy", args)

    # 5. LLM fallback
    return None
