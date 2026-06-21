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
from typing import List, NamedTuple, Optional

import config
import apikeys
from intelligence import llm_compat

from openai import OpenAI

_log = logging.getLogger(__name__)

# Dedicated client so the search call's longer timeout never bleeds into the
# realtime reply clients. Reuses the existing OpenAI key — no new secret.
_client = OpenAI(
    api_key=apikeys.OPENAI_API_KEY,
    timeout=float(getattr(config, "WEB_SEARCH_TIMEOUT_SECS", 20.0)),
    max_retries=int(getattr(config, "LLM_MAX_RETRIES", 2)),
)

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


def answer(text: str, person_id: Optional[int] = None, forced: bool = False) -> SearchResult:
    """Run the hosted web search and return Rex's spoken answer. Never raises — on any
    failure returns ``SearchResult(ok=False, ...)`` so the caller falls through to a
    normal reply."""
    model = _search_model()
    instructions = _build_instructions(person_id)
    user_input = text
    if forced:
        user_input = (
            f"{text}\n\n[The user explicitly asked you to look this up — use web search.]"
        )

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
        response = _client.responses.create(**kwargs)
    except TypeError as exc:
        # An older/narrower SDK that doesn't accept tool_choice="required" — retry
        # once with auto rather than failing the explicit search outright.
        if forced and "tool_choice" in str(exc):
            _log.debug("[web_search] tool_choice=required rejected; retrying auto")
            kwargs["tool_choice"] = "auto"
            try:
                response = _client.responses.create(**kwargs)
            except Exception as exc2:
                _log.warning("[web_search] search call failed: %s", exc2)
                return SearchResult(False, "", [])
        else:
            _log.warning("[web_search] search call failed: %s", exc)
            return SearchResult(False, "", [])
    except Exception as exc:
        _log.warning("[web_search] search call failed (%s): %s", type(exc).__name__, exc)
        return SearchResult(False, "", [])

    answer_text = _extract_text(response)
    if not answer_text:
        return SearchResult(False, "", [])

    try:
        from intelligence import llm
        answer_text = llm.clean_response_text(answer_text)
    except Exception:
        pass

    return SearchResult(True, answer_text, _extract_citations(response))
