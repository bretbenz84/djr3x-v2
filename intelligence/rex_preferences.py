"""
rex_preferences.py - deterministic DJ-R3X tastes and opinion replies.

This is intentionally not person memory. It gives Rex a stable character spine
for "do you like X?" questions so he can answer with small utterances and motion
instead of outsourcing every preference to the conversational LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
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
_SENSITIVE_GROUP_RE = re.compile(
    r"\b("
    r"race|races|religion|religions|gender|genders|sex|sexuality|orientation|disabled|disability|"
    r"black|white|asian|latino|latina|hispanic|jewish|muslim|christian|"
    r"gay|lesbian|bisexual|trans|nonbinary|immigrant|immigrants"
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


def answer_preference_query(text: str, args: Optional[dict[str, Any]] = None) -> PreferenceReply:
    """Return Rex's stable preference answer plus the body beat to perform."""
    parsed = dict(args or {})
    if not parsed:
        parsed = extract_preference_query(text) or {}
    mode = str(parsed.get("mode") or "open").strip().lower()

    if mode == "favorite":
        domain = _clean_topic(str(parsed.get("domain") or parsed.get("topic") or "general"))
        favorite = _favorite_for_domain(domain)
        text = favorite if favorite.endswith((".", "!", "?")) else f"{favorite}."
        topic = domain or "general"
        return PreferenceReply(
            text=text,
            emotion="happy",
            body_beat="happy_bounce",
            stance="favorite",
            topic=topic,
        )

    if mode == "compare":
        options = parsed.get("options") or []
        if not isinstance(options, (list, tuple)):
            options = []
        clean_options = [_clean_topic(str(option)) for option in options]
        clean_options = [option for option in clean_options if option]
        if len(clean_options) >= 2:
            choice = _choose_option(clean_options[0], clean_options[1])
            return PreferenceReply(
                text=f"{choice}. Obviously.",
                emotion="happy",
                body_beat="agreement_nod",
                stance="prefers",
                topic=f"{clean_options[0]} or {clean_options[1]}",
            )

    topic = _clean_topic(str(parsed.get("topic") or "")) or "that"
    verb = _normalize_verb(str(parsed.get("verb") or ""))
    opinion = _opinion_for_topic(topic)
    if _is_sensitive_group_topic(topic):
        return PreferenceReply(
            text="I do not rate whole categories of people. Individual organics generate plenty of data.",
            emotion="curious",
            body_beat="thinking_tilt",
            stance="boundary",
            topic=topic,
        )

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


def _opinion_for_topic(topic: str) -> _TopicOpinion:
    key = _normalize_topic(topic)
    for opinion in _KNOWN_OPINIONS:
        if any(_topic_matches(key, keyword) for keyword in opinion.keywords):
            return opinion

    bucket = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 30:
        return _TopicOpinion(
            keywords=(),
            stance="like",
            score=0.52,
            open_text=f"Yes. {topic} passes the vibe inspection.",
            like_yes="Mmhmm.",
            hate_no="Nope.",
        )
    if bucket < 56:
        return _TopicOpinion(
            keywords=(),
            stance="complicated",
            score=0.05,
            open_text=f"Mixed. {topic} requires additional suspicious staring.",
            beat="disbelief_stare",
            emotion="curious",
        )
    if bucket < 84:
        return _TopicOpinion(
            keywords=(),
            stance="dislike",
            score=-0.48,
            open_text=f"Nope. {topic} is not clearing the board.",
            like_no="Nope.",
            hate_yes="Mmhmm.",
        )
    return _TopicOpinion(
        keywords=(),
        stance="strong_dislike",
        score=-0.84,
        open_text=f"Hell to the no. {topic} goes in the airlock of taste.",
        like_no=_strong_no(),
        hate_yes="Mmhmm. Strongly.",
        beat="disgust_recoil",
        emotion="angry",
    )


def _choose_option(a: str, b: str) -> str:
    opinion_a = _opinion_for_topic(a)
    opinion_b = _opinion_for_topic(b)
    if abs(opinion_a.score - opinion_b.score) > 0.05:
        return a if opinion_a.score > opinion_b.score else b
    pair = "|".join(sorted([_normalize_topic(a), _normalize_topic(b)]))
    return a if int(hashlib.sha1(pair.encode("utf-8")).hexdigest()[:8], 16) % 2 == 0 else b


def _favorite_for_domain(domain: str) -> str:
    key = _normalize_topic(domain)
    for known, value in _FAVORITES.items():
        if known in key:
            return value
    return _FAVORITES["general"]


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
    return bool(_SENSITIVE_GROUP_RE.search(topic or ""))


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
