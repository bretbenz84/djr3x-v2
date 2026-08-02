"""
intelligence/web_search.py — current-info web search for Rex's conversational replies.

When a turn needs CURRENT / real-time information — either because the person said
so out loud ("look that up", "what's the latest on…") or because Rex decides on his
own that the question needs live data — this module answers it through OpenAI's
hosted ``web_search`` tool on the Responses API and voices the result in character.

It is a self-contained BRANCH off the normal reply path (see
``interaction._maybe_web_search_reply``): the tuned streaming reply in ``llm.py`` is
left untouched, and everything here is behind ``config.WEB_SEARCH_ENABLED``.

Why the Responses API (not the Chat Completions path the rest of the app uses): the
hosted ``web_search`` tool — where OpenAI does the retrieval, fetching, synthesis,
and citations in one call — lives on the Responses API. We talk to it directly here
(the ``llm_compat`` shim is Chat-Completions-shaped and does not apply), reusing the
existing ``OPENAI_API_KEY`` — no new provider, dependency, or secret.

Triggering:
  - EXPLICIT — any phrase in ``config.WEB_SEARCH_TRIGGER_PHRASES`` (substring,
    case-insensitive) forces a search.
  - AUTONOMOUS — a cheap keyword prefilter (``WEB_SEARCH_AUTONOMOUS_KEYWORDS``)
    narrows to plausibly time-sensitive questions; when
    ``WEB_SEARCH_AUTONOMOUS_GATE_ENABLED`` is on a small gpt-4o-mini classifier then
    confirms before a search is spent. Gate off → the prefilter alone triggers.

Everything is failure-safe: any error in detection or the search call returns a
"not handled / no result" so the caller falls through to a normal from-knowledge
reply rather than leaving Rex silent.
"""

from __future__ import annotations

import logging
import random
import re
import time
from typing import List, NamedTuple, Optional

import config
import apikeys
from intelligence import llm_compat

from openai import OpenAI

_log = logging.getLogger(__name__)

# Dedicated client so the search call's longer timeout never bleeds into the
# realtime reply clients. Reuses the existing OpenAI key — no new secret.
from intelligence import connectivity as _connectivity
_client = _connectivity.guard_client(OpenAI(
    api_key=apikeys.OPENAI_API_KEY,
    timeout=float(getattr(config, "WEB_SEARCH_TIMEOUT_SECS", 20.0)),
    max_retries=int(getattr(config, "LLM_MAX_RETRIES", 2)),
), "web_search")

# Last stall line spoken, so the same one never fires back-to-back.
_last_stall_line: Optional[str] = None


class SearchDecision(NamedTuple):
    """Result of trigger detection. ``forced`` means an explicit out-loud request
    (search no matter what); ``reason`` is a short label for the logs."""
    triggered: bool
    forced: bool
    reason: str


class SearchResult(NamedTuple):
    """Outcome of a search call. ``ok`` False (or empty ``text``) tells the caller to
    fall through to a normal reply."""
    ok: bool
    text: str
    citations: List[str]


_NO_DECISION = SearchDecision(False, False, "")


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def _search_model() -> str:
    """Model that runs the search + voices the answer. Defaults to the conversation
    model (so the answer stays in Rex's voice) but is independently overridable via
    ``WEB_SEARCH_MODEL`` — point it at a search-capable model if the conversation
    model can't host the tool."""
    return getattr(config, "WEB_SEARCH_MODEL", None) or llm_compat.conversation_model()


def _phrases() -> List[str]:
    return [str(p).lower() for p in getattr(config, "WEB_SEARCH_TRIGGER_PHRASES", []) if p]


def _keywords() -> List[str]:
    return [str(k).lower() for k in getattr(config, "WEB_SEARCH_AUTONOMOUS_KEYWORDS", []) if k]


_QUESTION_WORDS = (
    "what", "whats", "what's", "who", "whos", "who's", "when", "where", "which",
    "how", "why", "is", "are", "was", "were", "does", "do", "did", "can", "could",
    "will", "would", "should", "has", "have",
)


# ─────────────────────────────────────────────────────────────────────────────
# Trigger detection
# ─────────────────────────────────────────────────────────────────────────────

def matched_trigger_phrase(text: str) -> Optional[str]:
    """The explicit trigger phrase found in ``text`` (case-insensitive substring), or
    None. Pure + cheap — also handy for tests."""
    low = (text or "").lower()
    for phrase in _phrases():
        if phrase and phrase in low:
            return phrase
    return None


def _looks_like_question(text: str) -> bool:
    low = (text or "").strip().lower()
    if not low:
        return False
    if "?" in low:
        return True
    first = low.split()[0].strip(",.!").rstrip("'")
    return first in _QUESTION_WORDS


def _has_currentness_keyword(text: str) -> bool:
    low = (text or "").lower()
    return any(k in low for k in _keywords())


def _gate_says_needs_search(text: str) -> bool:
    """Small gpt-4o-mini classifier: does answering this REQUIRE current/real-time
    info from the web? Mirrors ``llm.classify_surprise``. Returns False on any error
    so a missed call never forces an unwanted search."""
    prompt = (
        "You decide whether answering a user's message to a voice assistant REQUIRES "
        "looking up CURRENT, real-time, or recent information from the web — news, "
        "latest releases, live scores, prices, today's events, who currently holds a "
        "role, and the like. Stable general knowledge, opinions, chit-chat, and "
        "questions about the user themselves do NOT require a search. Reply with only "
        'the single word "yes" or "no".\n\n'
        f'Message: "{text}"'
    )
    try:
        resp = _client.chat.completions.create(
            model=getattr(config, "WEB_SEARCH_GATE_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=3,
        )
        answer = (resp.choices[0].message.content or "").strip().lower()
        return answer.startswith("y")
    except Exception as exc:
        _log.debug("[web_search] gate classifier failed: %s", exc)
        return False


# Phatic small talk that MENTIONS currentness without asking for information:
# "what's up today" is a greeting, not a news query (field bug 2026-07-06: it
# triggered a 10s search that returned 0 citations and Rex recited hallucinated
# AI-model news). The trailing-filler whitelist keeps real queries searchable —
# "what's happening in Iran today" does NOT match (Iran isn't filler).
_PHATIC_SMALL_TALK_RE = re.compile(
    r"^\s*(?:hey|hi|hello|yo|so)?[,!\s]*"
    r"(?:what'?s\s+(?:up|new|good|happening|going\s+on)"
    r"|how'?s\s+it\s+going|how\s+are\s+(?:you|things)|what\s+is\s+up)"
    r"(?:\s+(?:with\s+you|today|tonight|right\s+now|man|dude|rex|r3x|buddy))*"
    r"\s*[?!.]*\s*$",
    re.IGNORECASE,
)


def should_search(text: str) -> SearchDecision:
    """Decide whether this turn warrants a web search. Explicit phrases always win
    (forced); otherwise the autonomous path (keyword prefilter, then optional LLM
    gate) decides. Failure-safe: any error → no search."""
    if not text or not text.strip():
        return _NO_DECISION
    try:
        phrase = matched_trigger_phrase(text)
        if phrase:
            return SearchDecision(True, True, f"explicit:{phrase}")

        if _PHATIC_SMALL_TALK_RE.match(text):
            return _NO_DECISION   # a greeting idiom, however "current" it sounds

        if not getattr(config, "WEB_SEARCH_AUTONOMOUS_ENABLED", True):
            return _NO_DECISION

        # Cheap prefilter so the gate classifier isn't spent on ordinary chatter:
        # a question-shaped turn that carries a currentness marker.
        if not (_looks_like_question(text) and _has_currentness_keyword(text)):
            return _NO_DECISION

        if not getattr(config, "WEB_SEARCH_AUTONOMOUS_GATE_ENABLED", True):
            return SearchDecision(True, False, "autonomous:keyword")

        if _gate_says_needs_search(text):
            return SearchDecision(True, False, "autonomous:gate")
        return _NO_DECISION
    except Exception as exc:
        _log.debug("[web_search] should_search error: %s", exc)
        return _NO_DECISION


# ─────────────────────────────────────────────────────────────────────────────
# Stall line
# ─────────────────────────────────────────────────────────────────────────────

def pick_stall_line() -> str:
    """One in-character 'let me check' line, never the same one twice running."""
    global _last_stall_line
    lines = [str(s) for s in getattr(config, "WEB_SEARCH_STALL_LINES", []) if s]
    if not lines:
        return ""
    choices = [s for s in lines if s != _last_stall_line] or lines
    chosen = random.choice(choices)
    _last_stall_line = chosen
    return chosen


# ─────────────────────────────────────────────────────────────────────────────
# Recent-search marker (for inquisitive proactive follow-ups)
# ─────────────────────────────────────────────────────────────────────────────
# After Rex looks something up for the person, the proactive/idle loop would
# otherwise keep COMMENTING on the same topic during the lull (re-summarizing it,
# piling on opinions). This marker lets the proactive directive flip those lull
# lines to be INQUISITIVE about the topic instead — "what got you asking about X?".

_recent_search: Optional[dict] = None

# Lead-ins to strip so the stored topic is the subject, not the request wrapper.
_TOPIC_LEADIN_RE = re.compile(
    r"^\s*(?:hey\s+rex[,]?\s*)?"
    r"(?:can|could|would|will)\s+you\s+|"
    r"(?:please|i'?d\s+like\s+you\s+to|i\s+want\s+you\s+to|let'?s)\s+",
    re.I,
)
_TOPIC_VERB_RE = re.compile(
    r"^\s*(?:look\s+(?:that|it)?\s*up|look\s+up|look\s+into|"
    r"search\s+(?:the\s+web|the\s+internet|online)?(?:\s+(?:for|about))?|"
    r"search(?:\s+for)?|google(?:\s+(?:that|it))?|find\s+out(?:\s+for\s+me)?(?:\s+about)?|"
    r"what'?s\s+the\s+latest\s+on|what\s+is\s+the\s+latest\s+on|tell\s+me\s+about)\s*",
    re.I,
)


def _search_topic(query: str) -> str:
    """Reduce the user's request to a short topic phrase for the follow-up prompt
    ("search the web about Star Trek Voyager" -> "Star Trek Voyager"). Best-effort."""
    q = (query or "").strip()
    if not q:
        return ""
    prev = None
    # Peel a leading politeness/request wrapper, then a search verb, possibly twice
    # ("can you look up ...").
    while prev != q:
        prev = q
        q = _TOPIC_LEADIN_RE.sub("", q).strip()
        q = _TOPIC_VERB_RE.sub("", q).strip()
    topic = q.strip(" ?.!,:'\"") or (query or "").strip()
    words = topic.split()
    if len(words) > 12:
        topic = " ".join(words[:12])
    return topic


def note_search(query: str) -> None:
    """Record that Rex just web-searched `query` for the person, so the next idle/lull
    line can be inquisitive about it instead of piling on more facts."""
    global _recent_search
    topic = _search_topic(query)
    if topic:
        _recent_search = {"topic": topic, "at": time.monotonic()}


def recent_search(max_age_secs: Optional[float] = None) -> Optional[str]:
    """Topic of a still-fresh recent web search, else None. Window defaults to
    config.WEB_SEARCH_FOLLOWUP_WINDOW_SECS."""
    if not _recent_search:
        return None
    if not getattr(config, "WEB_SEARCH_FOLLOWUP_INQUISITIVE_ENABLED", True):
        return None
    window = (
        float(max_age_secs)
        if max_age_secs is not None
        else float(getattr(config, "WEB_SEARCH_FOLLOWUP_WINDOW_SECS", 120.0))
    )
    if (time.monotonic() - float(_recent_search.get("at", 0.0))) > window:
        return None
    return _recent_search.get("topic") or None


def clear_recent_search(min_age_secs: float = 0.0) -> None:
    """Clear the recent-search marker (e.g. when the user re-engages with a new turn).
    `min_age_secs` guards against wiping a marker that was set moments ago in the same
    turn (the searched answer can play for several seconds before control returns)."""
    global _recent_search
    if not _recent_search:
        return
    if min_age_secs > 0.0 and (
        time.monotonic() - float(_recent_search.get("at", 0.0)) < min_age_secs
    ):
        return
    _recent_search = None


# ─────────────────────────────────────────────────────────────────────────────
# The search call (OpenAI Responses API + hosted web_search tool)
# ─────────────────────────────────────────────────────────────────────────────

def _build_instructions(person_id: Optional[int]) -> str:
    """Rex's full persona prompt (so the answer stays in voice) plus a short addendum
    telling him he just looked it up. Falls back to the bare addendum if prompt
    assembly fails."""
    addendum = getattr(config, "WEB_SEARCH_PERSONA_ADDENDUM", "")
    try:
        from intelligence import llm
        base = llm.assemble_system_prompt(person_id)
        return f"{base}\n\n---\n\n{addendum}".strip() if addendum else base
    except Exception as exc:
        _log.debug("[web_search] persona prompt assembly failed; using addendum only: %s", exc)
        return addendum


def _extract_text(response) -> str:
    """Pull the assistant's answer text from a Responses API result. Prefers the SDK's
    ``output_text`` aggregator, falling back to walking output message content."""
    text = (getattr(response, "output_text", None) or "").strip()
    if text:
        return text
    parts: List[str] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "message":
            continue
        for block in getattr(item, "content", None) or []:
            chunk = getattr(block, "text", None)
            if chunk:
                parts.append(chunk)
    return "".join(parts).strip()


# URL / link shapes the hosted search likes to fold into the answer. Rex SPEAKS his
# replies, so a read-aloud "https://example.com/long/path" is just noise — strip them.
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")            # [label](url) -> label
_SOURCE_PAREN_RE = re.compile(
    r"\s*[\(\[]\s*(?:source|sources|via|see|read more|ref|link)\b[^)\]]*[\)\]]", re.I
)
_BARE_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)    # http(s):// or www. ...
_BARE_DOMAIN_RE = re.compile(                                 # espn.com, reuters.com/world/...
    r"\b(?:[a-z0-9-]+\.)+(?:com|org|net|io|gov|edu|co|news|tv|me|uk|us|ai|app|dev|info|biz)\b"
    r"(?:/\S*)?",
    re.I,
)
_CITE_MARK_RE = re.compile(r"\s*\[\d+\]")                     # footnote markers [1]
_EMPTY_BRACKET_RE = re.compile(r"[\(\[]\s*[\)\]]")            # leftover () or []


def strip_links(text: str) -> str:
    """Remove anything URL-shaped from a spoken answer — full URLs, markdown links
    (keeps the label, drops the address), bare domains, "(source: …)" citations, and
    footnote markers — then tidy the punctuation/whitespace left behind. Rex reads
    replies aloud, so a link is just characters spelled out at the listener."""
    if not text:
        return text
    out = _MD_LINK_RE.sub(r"\1", text)
    out = _SOURCE_PAREN_RE.sub("", out)
    out = _BARE_URL_RE.sub("", out)
    out = _BARE_DOMAIN_RE.sub("", out)
    out = _CITE_MARK_RE.sub("", out)
    out = _EMPTY_BRACKET_RE.sub("", out)
    # Tidy orphaned punctuation/whitespace where a link used to be.
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    out = re.sub(r"\(\s*\)|\[\s*\]", "", out)
    out = re.sub(r"\s{2,}", " ", out)
    return out.strip(" \t\n,;:")


def _extract_citations(response) -> List[str]:
    """Best-effort list of source URLs from the answer's annotations (for logs)."""
    urls: List[str] = []
    try:
        for item in getattr(response, "output", None) or []:
            if getattr(item, "type", None) != "message":
                continue
            for block in getattr(item, "content", None) or []:
                for ann in getattr(block, "annotations", None) or []:
                    url = getattr(ann, "url", None)
                    if url and url not in urls:
                        urls.append(url)
    except Exception:
        pass
    return urls


def _search_models() -> List[str]:
    """Models to try, in order: the primary (in-voice) model first, then a known
    tool-capable fallback. The fallback rescues the case the primary model can't host
    the web_search tool (it raises) — so an explicit lookup still returns a real result
    instead of silently degrading to a stale from-knowledge answer."""
    primary = _search_model()
    models = [primary]
    fallback = str(getattr(config, "WEB_SEARCH_FALLBACK_MODEL", "") or "").strip()
    if fallback and fallback != primary:
        models.append(fallback)
    return models


def _create_search_response(model: str, *, instructions: str, user_input: str, forced: bool):
    """One Responses-API create() for `model`, with the tool_choice=required→auto SDK
    retry. Returns the response or raises (so the caller can try the next model)."""
    kwargs = {
        "model": model,
        "instructions": instructions,
        "input": user_input,
        "tools": [{"type": "web_search"}],
        "tool_choice": "required" if forced else "auto",
        "max_output_tokens": int(getattr(config, "WEB_SEARCH_MAX_OUTPUT_TOKENS", 600)),
        "timeout": float(getattr(config, "WEB_SEARCH_TIMEOUT_SECS", 20.0)),
    }
    # reasoning_effort is a reasoning-model-only knob; including it for gpt-4o-class
    # models is a 400. The stall line covers the latency, so a little reasoning here
    # buys better synthesis.
    if llm_compat.is_reasoning_model(model):
        effort = getattr(config, "WEB_SEARCH_REASONING_EFFORT", None)
        if effort:
            kwargs["reasoning"] = {"effort": effort}
    try:
        return _client.responses.create(**kwargs)
    except TypeError as exc:
        # An older/narrower SDK that doesn't accept tool_choice="required" — retry
        # once with auto rather than failing the explicit search outright.
        if forced and "tool_choice" in str(exc):
            _log.debug("[web_search] tool_choice=required rejected; retrying auto")
            kwargs["tool_choice"] = "auto"
            return _client.responses.create(**kwargs)
        raise


def answer(text: str, person_id: Optional[int] = None, forced: bool = False) -> SearchResult:
    """Run the hosted web search and return Rex's spoken answer. Never raises — on any
    failure returns ``SearchResult(ok=False, ...)`` so the caller falls through to a
    normal reply."""
    instructions = _build_instructions(person_id)
    user_input = text
    if forced:
        user_input = (
            f"{text}\n\n[The user explicitly asked you to look this up — use web search.]"
        )

    models = _search_models()
    response = None
    for idx, model in enumerate(models):
        try:
            response = _create_search_response(
                model, instructions=instructions, user_input=user_input, forced=forced
            )
            break
        except Exception as exc:
            # The primary model may not support the hosted web_search tool (a 400) —
            # try the fallback rather than silently returning a stale-knowledge reply.
            more = " — trying fallback" if idx + 1 < len(models) else ""
            _log.warning(
                "[web_search] search call failed on model=%s (%s): %s%s",
                model, type(exc).__name__, exc, more,
            )
    if response is None:
        return SearchResult(False, "", [])

    answer_text = _extract_text(response)
    if not answer_text:
        return SearchResult(False, "", [])

    # Strip URLs/links — Rex speaks his replies, so a read-aloud web address is noise.
    if getattr(config, "WEB_SEARCH_STRIP_LINKS", True):
        answer_text = strip_links(answer_text)

    try:
        from intelligence import llm
        answer_text = llm.clean_response_text(answer_text)
    except Exception:
        pass

    if not answer_text.strip():
        # Nothing left after stripping (e.g. a bare-link "answer") — fall through.
        return SearchResult(False, "", [])

    return SearchResult(True, answer_text, _extract_citations(response))
