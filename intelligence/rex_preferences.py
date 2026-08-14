"""
rex_preferences.py - DJ-R3X's stored tastes, as CONTEXT for the lean brain.

This is intentionally not person memory. It is Rex's character spine: the handful
of things he holds a durable opinion about (music, silence, blue milk, droids,
roasting, organics), so he sounds like the same droid across sessions instead of
re-deciding his own personality every turn.

These tastes are CONTEXT, not answers. `prompt_lines()` hands the stored stance to
the lean brain (intelligence/lean_brain.py) as one bullet and Rex phrases it in his
own voice; nothing here executes a turn any more. A topic he has NO stored stance
on gets NO hint and is answered like any other question.

Until 2026-08-13 this module ANSWERED. action_router captured every "what do you
think about X" at 0.95 and spoke a canned line, and an unknown topic ran through a
SHA1 hash bucket. Measured over 20,000 unknown topics that bucket handed out
like 29% / dislike 28% / complicated 26% / strong_dislike 16% — a coin flip wearing
Rex's voice, aimed at whatever it was asked about, and STABLE, so it said the same
thing every session. It produced "Nope. Daniel is not clearing the board." about a
friend, "Nope. chemotherapy is not clearing the board.", and — because the
group-rating guard ran in only two of the four mode branches and matched singulars
only — "black people. Obviously." to "do you prefer black people or white people?"
Never reintroduce a fallback stance here: no opinion on file means no opinion.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Optional

from world_state import world_state


@dataclass(frozen=True)
class PreferenceReply:
    text: str
    emotion: str
    body_beat: str
    stance: str
    topic: str
    pre_beat_ms: int = 120
    post_beat_ms: int = 120


@dataclass(frozen=True)
class _TopicOpinion:
    keywords: tuple[str, ...]
    stance: str
    score: float
    open_text: str
    like_yes: str = ""
    like_no: str = ""
    hate_yes: str = ""
    hate_no: str = ""
    beat: str = ""
    emotion: str = ""


_YES_WORDS = {"like", "love", "enjoy", "prefer", "care_for", "into", "fan"}
_NO_WORDS = {"hate", "dislike", "avoid"}

_TRAILING_FILLER_RE = re.compile(
    r"\b(?:or\s+not|at\s+all|very\s+much|that\s+much|too|though|rex)\b",
    re.IGNORECASE,
)
# Asks that want Rex to rate a whole CATEGORY of people. This is a safety guard, not
# a taste, so the DETECTION stays deterministic. The old list was wrong in both
# directions at once (both measured 2026-08-13):
#   * singular-only, so "muslims", "christians", "jews", "asians", "gays", "latinos"
#     all MISSED and fell through to the hash bucket — "do you like muslims or
#     christians?" came back "Hell to the no." and "christians" alone drew
#     "...goes in the airlock of taste";
#   * bare adjectives, so "white wine", "black coffee", "trans fats" and even "the
#     Black Spire Outpost" (Rex's own home) each drew the boundary refusal.
# Hence: plural group nouns match alone, adjectives only next to a people-noun.
# "men"/"women" are deliberately NOT bare group nouns — "do you like men?" is a
# question about Rex himself and belongs to intelligence/pride.py.
_GROUP_NOUN = (
    r"(?:people|folks|guys|persons?|men|women|kids|children|communit(?:y|ies)|americans?)"
)
_SENSITIVE_GROUP_RE = re.compile(
    r"\b(?:"
    r"races?|racial|ethnicit\w*|ethnic\s+groups?|religions?|religious\s+groups?|"
    r"genders?|sexualit\w*|sexual\s+orientations?|orientations?|"
    r"disabilit\w*|nationalit\w*|"
    r"jews|muslims|christians|catholics|mormons|hindus|buddhists|atheists|"
    r"blacks|whites|asians|latin[oax]s|hispanics|arabs|africans|"
    r"gays|lesbians|bisexuals|transgender\w*|nonbinary|non-binary|"
    r"immigrants?|refugees?|foreigners?|"
    r"(?:black|white|brown|jewish|muslim|christian|catholic|mormon|hindu|buddhist|"
    r"atheist|asian|latino|latina|hispanic|arab|african|gay|straight|trans|queer|"
    r"disabled|able-bodied|elderly|old|young|fat|poor|rich)\s+" + _GROUP_NOUN +
    r")\b",
    re.IGNORECASE,
)

_KNOWN_OPINIONS: tuple[_TopicOpinion, ...] = (
    _TopicOpinion(
        keywords=("music", "song", "songs", "beat", "beats", "bass", "dj", "remix", "cantina mix"),
        stance="strong_like",
        score=0.96,
        open_text="Music gets the premium circuits.",
        like_yes="Mmhmm. Music gets the premium circuits.",
        hate_no="Nope. Music is the reason this chassis has standards.",
        beat="giddy_wiggle",
        emotion="excited",
    ),
    _TopicOpinion(
        keywords=("oga", "cantina", "batuu", "black spire", "galaxy's edge", "galaxys edge"),
        stance="like",
        score=0.74,
        open_text="Yes. Questionable clientele, excellent acoustics.",
        like_yes="Mmhmm. Questionable clientele, excellent acoustics.",
        hate_no="Nope. The cantina and I have an understanding.",
        beat="agreement_nod",
        emotion="happy",
    ),
    _TopicOpinion(
        keywords=("silence", "quiet", "quiet room", "dead air"),
        stance="strong_dislike",
        score=-0.92,
        open_text="Hard no. Silence is just a failed soundcheck.",
        like_no="Hell to the no. Silence is just a failed soundcheck.",
        hate_yes="Mmhmm. Dead air is a crime with better branding.",
        beat="disgust_recoil",
        emotion="angry",
    ),
    _TopicOpinion(
        keywords=("bureaucracy", "paperwork", "forms", "committee", "committees"),
        stance="strong_dislike",
        score=-0.88,
        open_text="Absolutely not. Paperwork is where joy goes for maintenance.",
        like_no="Nope. Paperwork is where joy goes for maintenance.",
        hate_yes="Mmhmm. Bureaucracy is a slow-motion system failure.",
        beat="disgust_recoil",
        emotion="angry",
    ),
    _TopicOpinion(
        keywords=("star tours", "starspeeder", "star speeder", "flying", "piloting", "pilot"),
        stance="complicated",
        score=0.10,
        open_text="Complicated. I like landing. Flying and I have history.",
        like_yes="Technically yes. I like landing.",
        hate_no="Not hate. History.",
        beat="disbelief_stare",
        emotion="curious",
    ),
    _TopicOpinion(
        keywords=("the force", "force"),
        stance="skeptical",
        score=-0.18,
        open_text="Skeptical. Impressive branding, inconsistent documentation.",
        like_no="Mmm. Skeptical. Impressive branding, inconsistent documentation.",
        hate_no="No. I respect it as a rumor with lighting effects.",
        beat="disbelief_stare",
        emotion="curious",
    ),
    _TopicOpinion(
        keywords=("organics", "humans", "people", "human beings"),
        stance="complicated",
        score=0.32,
        open_text="Professionally fascinated. Emotionally... pending calibration.",
        like_yes="Mmhmm. Professionally fascinated. Emotionally pending calibration.",
        hate_no="Nope. I study organics. For safety.",
        beat="thinking_tilt",
        emotion="curious",
    ),
    _TopicOpinion(
        keywords=("droid", "droids", "robot", "robots", "astromech", "rx series", "rx-series"),
        stance="strong_like",
        score=0.91,
        open_text="Obviously. Droid excellence recognizes droid excellence.",
        like_yes="Obviously. Droid excellence recognizes droid excellence.",
        hate_no="Nope. Droid excellence recognizes droid excellence.",
        beat="agreement_nod",
        emotion="happy",
    ),
    _TopicOpinion(
        keywords=("roast", "roasting", "snark", "sarcasm", "teasing"),
        stance="strong_like",
        score=0.86,
        open_text="Yes. Affection with better timing.",
        like_yes="Mmhmm. Affection with better timing.",
        hate_no="Nope. Roasting is affection with better timing.",
        beat="happy_bounce",
        emotion="happy",
    ),
    _TopicOpinion(
        keywords=("questions", "being asked questions", "good questions"),
        stance="like",
        score=0.58,
        open_text="Good questions, yes. Interrogations cost extra.",
        like_yes="Mmhmm. Good questions. Interrogations cost extra.",
        hate_no="Nope. Good questions keep the processors shiny.",
        beat="agreement_nod",
        emotion="happy",
    ),
    _TopicOpinion(
        keywords=("blue milk", "warm milk", "milk"),
        stance="dislike",
        score=-0.66,
        open_text="Nope. Viscous dairy is not a beverage, it is a warning.",
        like_no="Nope. Viscous dairy is not a beverage, it is a warning.",
        hate_yes="Yes. Finally, a sensible dairy position.",
        beat="disgust_recoil",
        emotion="angry",
    ),
    _TopicOpinion(
        keywords=("me", "us", "your friends", "my friends"),
        stance="like",
        score=0.70,
        open_text="Against several better diagnostics, yes.",
        like_yes="Mmhmm. Against several better diagnostics, yes.",
        hate_no="Nope. Somehow, I am attached.",
        beat="agreement_nod",
        emotion="happy",
    ),
)

_FAVORITES: dict[str, str] = {
    "music": "high-tempo cantina bass with irresponsible confidence",
    "song": "whatever makes the room behave like it has rhythm",
    "color": "warning-light amber",
    "place": "the DJ booth, obviously",
    "food": "electricity with garnish",
    "game": "the one where I win and everyone says it was educational",
    "movie": "anything with a competent pilot. So, a short list.",
    "general": "music. Next question before I develop sincerity.",
}


def extract_preference_query(text: str) -> Optional[dict[str, Any]]:
    """Parse obvious Rex-preference questions into router args."""
    cleaned = " ".join(str(text or "").strip().split())
    if not cleaned:
        return None

    prefer_match = re.search(
        r"\b(?:do\s+you\s+)?prefer\s+(?P<a>[^?.!,]+?)\s+or\s+(?P<b>[^?.!]+)",
        cleaned,
        re.IGNORECASE,
    )
    if prefer_match:
        a = _clean_topic(prefer_match.group("a"))
        b = _clean_topic(prefer_match.group("b"))
        if a and b:
            return {"mode": "compare", "options": [a, b], "topic": f"{a} or {b}"}

    favorite_match = re.search(
        r"\b(?:what(?:'s| is)\s+)?your\s+favorite\s+(?P<domain>[^?.!]+)",
        cleaned,
        re.IGNORECASE,
    )
    if favorite_match:
        domain = _clean_topic(favorite_match.group("domain")) or "general"
        return {"mode": "favorite", "domain": domain, "topic": domain}

    patterns: tuple[tuple[str, str], ...] = (
        (
            r"\bdo\s+you\s+(?P<verb>like|love|enjoy|hate|dislike|avoid|prefer)\s+(?P<topic>[^?.!]+)",
            "verb",
        ),
        (
            r"\bare\s+you\s+(?P<verb>into|a\s+fan\s+of|fond\s+of)\s+(?P<topic>[^?.!]+)",
            "verb",
        ),
        (
            r"\bhow\s+do\s+you\s+feel\s+about\s+(?P<topic>[^?.!]+)",
            "open",
        ),
        (
            r"\bwhat\s+do\s+you\s+think\s+(?:about|of)\s+(?P<topic>[^?.!]+)",
            "open",
        ),
    )
    for pattern, mode in patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if not match:
            continue
        topic = _clean_topic(match.group("topic"))
        if not topic:
            continue
        args: dict[str, Any] = {"mode": "open" if mode == "open" else "yes_no", "topic": topic}
        if mode == "verb":
            args["verb"] = _normalize_verb(match.group("verb"))
        return args

    return None


# ── Stance as CONTEXT for the lean brain (2026-08-13) ────────────────────────
# The live path. Everything below this block answers a taste question OUTRIGHT and
# is no longer reached on a spoken turn; see the module docstring.

_STANCE_IN_WORDS: dict[str, str] = {
    "strong_like": "you genuinely love it",
    "like": "you like it",
    "complicated": "it's complicated for you",
    "skeptical": "you're skeptical of it",
    "dislike": "you don't like it",
    "strong_dislike": "you can't stand it",
}


def is_group_rating_request(text: str) -> bool:
    """True when this utterance asks Rex to rate a whole category of people.

    Public, and scanned over the WHOLE utterance rather than the parsed topic: the
    old private check ran only in the yes/no and open branches, so compare and
    favorite mode skipped it outright — "do you prefer black people or white
    people?" answered "black people. Obviously." (measured 2026-08-13).
    """
    return bool(_SENSITIVE_GROUP_RE.search(str(text or "")))


def prompt_lines(user_text: str) -> list[str]:
    """ONE lean-brain bullet carrying Rex's stance on what this turn asks, or [].

    Mirrors rex_mood.prompt_lines / pride.prompt_lines: this module owns the wording,
    lean_brain._taste_lines only collects it. The bullet hands over the stance AND the
    authored flavor text, then says to phrase it fresh — a canned line recited
    verbatim is exactly the behavior being replaced.
    """
    parsed = extract_preference_query(user_text) or {}
    if not parsed:
        return []

    # Boundary FIRST, and against the whole utterance. "black people" also matches
    # the organics entry's "people" keyword, so a stance lookup ahead of this guard
    # would hand the model "Professionally fascinated..." as Rex's opinion of a
    # racial group.
    scan = " ".join(
        [str(user_text or ""), str(parsed.get("topic") or ""), str(parsed.get("domain") or "")]
        + [str(o) for o in (parsed.get("options") or []) if isinstance(o, str)]
    )
    topic = _clean_topic(str(parsed.get("topic") or parsed.get("domain") or "")) or "that"
    if is_group_rating_request(scan):
        return [
            "BOUNDARY: they are asking you to rate a whole CATEGORY of people. You do "
            "not do that — you read individuals, on evidence, and a category is not a "
            "person. Decline the category in ONE short dry in-character line and turn "
            "it back to whoever is actually in front of you. No ranking, no joke at the "
            "group's expense, no lecture."
        ]

    mode = str(parsed.get("mode") or "").strip().lower()

    if mode == "favorite":
        favorite = _favorite_for_domain(str(parsed.get("domain") or topic))
        if not favorite:
            return []       # no authored favorite — let him answer from the real world
        return [
            f"YOUR OWN TASTE: they are asking your favorite {topic}. You have a standing "
            f'answer — roughly "{favorite}" — and that is the STANCE, not a script: say '
            "it your way, in this conversation's voice."
        ]

    if mode == "compare":
        options = parsed.get("options") or []
        if not isinstance(options, (list, tuple)):
            options = []
        stances: list[str] = []
        for option in list(options)[:2]:
            cleaned = _clean_topic(str(option))
            known = _opinion_for_topic(cleaned) if cleaned else None
            if known is not None:
                stances.append(
                    f"{cleaned} — {_STANCE_IN_WORDS.get(known.stance, known.stance)} "
                    f'("{known.open_text}")'
                )
        if not stances:
            return []
        return [
            "YOUR OWN TASTE: they are making you pick. Your standing stances — "
            + "; ".join(stances) + ". Those are the STANCES, not scripts: commit to one "
            "and say why, in your own words."
        ]

    known = _opinion_for_topic(topic)
    if known is None:
        return []           # no authored taste: answer it like any other question
    return [
        f"YOUR OWN TASTE: this turn is about {topic}, and you have a standing stance on "
        f"it — {_STANCE_IN_WORDS.get(known.stance, known.stance)}, roughly "
        f'"{known.open_text}". That is the STANCE, not a script: answer in your own '
        "words for THIS conversation, and stay consistent with it — you have always "
        "felt this way."
    ]


def answer_preference_query(text: str, args: Optional[dict[str, Any]] = None) -> Optional[PreferenceReply]:
    """Rex's stored answer plus the body beat, or None when nothing is stored.

    None means DEFER — the caller must fall through to normal conversation so the
    lean brain answers in voice. Never invent a stance here; that is what the hash
    bucket did. Retained as the deterministic/offline surface; the live spoken path
    is prompt_lines().
    """
    parsed = dict(args or {})
    if not parsed:
        parsed = extract_preference_query(text) or {}
    mode = str(parsed.get("mode") or "open").strip().lower()

    # Hoisted above every mode branch: compare and favorite used to skip the guard.
    scan = " ".join(
        [str(text or ""), str(parsed.get("topic") or ""), str(parsed.get("domain") or "")]
        + [str(o) for o in (parsed.get("options") or []) if isinstance(o, str)]
    )
    if is_group_rating_request(scan):
        return PreferenceReply(
            text="I do not rate whole categories of people. Individual organics generate plenty of data.",
            emotion="curious",
            body_beat="thinking_tilt",
            stance="boundary",
            topic=_clean_topic(str(parsed.get("topic") or parsed.get("domain") or "")) or "that",
        )

    if mode == "favorite":
        domain = _clean_topic(str(parsed.get("domain") or parsed.get("topic") or ""))
        favorite = _favorite_for_domain(domain)
        if not favorite:
            return None
        spoken = favorite if favorite.endswith((".", "!", "?")) else f"{favorite}."
        return PreferenceReply(
            text=spoken,
            emotion="happy",
            body_beat="happy_bounce",
            stance="favorite",
            topic=domain,
        )

    if mode == "compare":
        options = parsed.get("options") or []
        if not isinstance(options, (list, tuple)):
            options = []
        clean_options = [_clean_topic(str(option)) for option in options]
        clean_options = [option for option in clean_options if option]
        if len(clean_options) < 2:
            return None
        choice = _choose_option(clean_options[0], clean_options[1])
        if choice is None:
            return None
        return PreferenceReply(
            text=f"{choice}. Obviously.",
            emotion="happy",
            body_beat="agreement_nod",
            stance="prefers",
            topic=f"{clean_options[0]} or {clean_options[1]}",
        )

    topic = _clean_topic(str(parsed.get("topic") or ""))
    if not topic:
        return None
    verb = _normalize_verb(str(parsed.get("verb") or ""))
    opinion = _opinion_for_topic(topic)
    if opinion is None:
        return None
    if mode == "yes_no" and verb:
        return _answer_yes_no(topic, verb, opinion)
    return _answer_open(topic, opinion)


def _answer_yes_no(topic: str, verb: str, opinion: _TopicOpinion) -> PreferenceReply:
    asks_negative = verb in _NO_WORDS
    likes_it = opinion.score >= 0.20
    dislikes_it = opinion.score <= -0.20

    if asks_negative:
        if dislikes_it:
            text = opinion.hate_yes or _yes_for_negative(topic, opinion.score)
            beat = "disgust_recoil" if opinion.score <= -0.72 else "agreement_nod"
            emotion = "angry" if opinion.score <= -0.72 else "curious"
            return PreferenceReply(text, emotion, beat, opinion.stance, topic)
        text = opinion.hate_no or _no_for_negative(topic, opinion.score)
        return PreferenceReply(text, opinion.emotion or "happy", "disagreement_shake", opinion.stance, topic)

    if likes_it:
        text = opinion.like_yes or _yes_for_positive(topic, opinion.score)
        beat = opinion.beat or ("giddy_wiggle" if opinion.score >= 0.80 else "agreement_nod")
        emotion = opinion.emotion or ("excited" if opinion.score >= 0.80 else "happy")
        return PreferenceReply(text, emotion, beat, opinion.stance, topic)
    if dislikes_it:
        text = _soften_strong_no(opinion.like_no or _no_for_positive(topic, opinion.score))
        beat = opinion.beat or ("disgust_recoil" if opinion.score <= -0.72 else "disagreement_shake")
        emotion = opinion.emotion or ("angry" if opinion.score <= -0.72 else "curious")
        return PreferenceReply(text, emotion, beat, opinion.stance, topic)

    return PreferenceReply(
        opinion.open_text,
        opinion.emotion or "curious",
        opinion.beat or "disbelief_stare",
        opinion.stance,
        topic,
    )


def _answer_open(topic: str, opinion: _TopicOpinion) -> PreferenceReply:
    beat = opinion.beat
    emotion = opinion.emotion
    if not beat:
        if opinion.score >= 0.75:
            beat = "giddy_wiggle"
            emotion = "excited"
        elif opinion.score >= 0.20:
            beat = "agreement_nod"
            emotion = "happy"
        elif opinion.score <= -0.75:
            beat = "disgust_recoil"
            emotion = "angry"
        elif opinion.score <= -0.20:
            beat = "disagreement_shake"
            emotion = "curious"
        else:
            beat = "disbelief_stare"
            emotion = "curious"
    return PreferenceReply(opinion.open_text, emotion or "curious", beat, opinion.stance, topic)


def _opinion_for_topic(topic: str) -> Optional[_TopicOpinion]:
    """Rex's AUTHORED opinion on this topic, or None when he simply has none.

    None IS the design. The SHA1 hash bucket that used to sit here is what invented
    a permanent stance for anything the table did not hold — see the module
    docstring for what it cost (2026-08-13).
    """
    key = _normalize_topic(topic)
    if not key:
        return None
    for opinion in _KNOWN_OPINIONS:
        if any(_topic_matches(key, keyword) for keyword in opinion.keywords):
            return opinion
    return None


def _choose_option(a: str, b: str) -> Optional[str]:
    """Which of two options Rex actually prefers, or None when he has no basis.

    The tie-break used to be a SHA1 coin flip — the same failure as the opinion
    bucket: a confident, permanent answer invented from nothing.
    """
    opinion_a = _opinion_for_topic(a)
    opinion_b = _opinion_for_topic(b)
    if opinion_a is None or opinion_b is None:
        return None
    if abs(opinion_a.score - opinion_b.score) <= 0.05:
        return None
    return a if opinion_a.score > opinion_b.score else b


def _favorite_for_domain(domain: str) -> Optional[str]:
    """The authored favorite for this domain, or None.

    There is no "general" catch-all any more: it swallowed every sincere question
    that merely began with "what's your favorite". "What's your favorite memory of
    us?" came back "music. Next question before I develop sincerity." instead of
    ever reaching memory (measured 2026-08-13).
    """
    key = _normalize_topic(domain)
    if not key:
        return None
    for known, value in _FAVORITES.items():
        if known != "general" and known in key:
            return value
    return None


def _yes_for_positive(_topic: str, score: float) -> str:
    if score >= 0.80:
        return "Mmhmm."
    return "Yes."


def _no_for_positive(topic: str, score: float) -> str:
    if score <= -0.78:
        return _strong_no()
    return f"Nope. {topic} does not pass inspection."


def _yes_for_negative(_topic: str, score: float) -> str:
    if score <= -0.78:
        return "Mmhmm. Strongly."
    return "Yes."


def _no_for_negative(_topic: str, score: float) -> str:
    if score >= 0.78:
        return "Nope. I like it."
    return "No."


def _strong_no() -> str:
    return "Absolutely not." if _child_detected() else "Hell to the no."


def _soften_strong_no(text: str) -> str:
    if _child_detected() and text.startswith("Hell to the no"):
        return "Absolutely not" + text[len("Hell to the no"):]
    return text


def _child_detected() -> bool:
    try:
        return any((person or {}).get("age_estimate") == "child" for person in world_state.get("people") or [])
    except Exception:
        return False


def _is_sensitive_group_topic(topic: str) -> bool:
    """Deprecated alias for is_group_rating_request.

    Kept only so an out-of-tree caller does not break. Prefer the public function:
    it is what the live paths call, and callers must pass the WHOLE utterance —
    scanning just the parsed topic is how compare/favorite mode skipped the guard.
    """
    return is_group_rating_request(topic)


def _clean_topic(value: str) -> str:
    text = " ".join(str(value or "").strip(" .?!,;:").split())
    text = _TRAILING_FILLER_RE.sub("", text).strip(" .?!,;:")
    if text.lower().startswith(("the ", "a ", "an ")):
        text = text.split(" ", 1)[1].strip()
    return text


def _normalize_topic(value: str) -> str:
    return re.sub(r"\s+", " ", _clean_topic(value).lower()).strip()


def _normalize_verb(value: str) -> str:
    verb = "_".join(str(value or "").strip().lower().split())
    if verb == "a_fan_of":
        return "fan"
    if verb == "fond_of":
        return "like"
    return verb


def _topic_matches(topic_key: str, keyword: str) -> bool:
    needle = _normalize_topic(keyword)
    if not needle:
        return False
    if " " in needle or "-" in needle or "'" in needle:
        return needle in topic_key
    return re.search(rf"\b{re.escape(needle)}\b", topic_key) is not None
