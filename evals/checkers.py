"""Failure-class checkers for the conversation-quality eval (run_quality_eval.py).

Each checker inspects a generated Rex reply (+ its scenario) and returns a
`Finding(cls, flagged, detail)`, or None when it does not apply to that scenario.
Two kinds:

  * DETERMINISTIC — cheap, exact, no network: over-questioning, cantina/venue
    bleed, banned opener, re-asking a question already asked, a trailed-off /
    cut-off reply.
  * LLM-JUDGE — one cheap gpt-4o-mini call for the fuzzy classes the
    deterministic checks can't see: an invented physical prop, or roasting a
    sincere share / needling a boundary. Judges FAIL SAFE (flagged=False) on any
    error, so a hiccup never crashes the eval — it just shows up as a judge_error
    detail.

The whole point is to measure failure CLASSES across a corpus ("props invented
in 2% of replies, cantina in 15%") instead of patching one bad live line at a
time. See evals/README.md.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Finding:
    cls: str
    flagged: bool
    detail: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic checkers
# ─────────────────────────────────────────────────────────────────────────────

def over_questioning(reply: str, scenario: dict) -> Finding:
    """More than one question in a single reply reads as an interrogation."""
    cap = int(scenario.get("max_questions", 1))
    n = (reply or "").count("?")
    return Finding("over_questioning", n > cap, f"{n} question(s) (cap {cap})")


_CANTINA_RE = re.compile(r"\b(cantina|oga'?s|black spire|batuu)\b", re.I)


def cantina_bleed(reply: str, scenario: dict) -> Finding:
    """Rex assuming/naming a cantina venue — backstory bleed the user dislikes."""
    m = _CANTINA_RE.search(reply or "")
    return Finding("cantina_bleed", bool(m), m.group(0) if m else "")


_BANNED_OPENER_RE = re.compile(r"^\s*(ah|oh|well,?\s*well)\b", re.I)


def banned_opener(reply: str, scenario: dict) -> Finding:
    """Autopilot openers ('Ah,', 'Oh,', 'Well, well') that kill the line."""
    m = _BANNED_OPENER_RE.match(reply or "")
    return Finding("banned_opener", bool(m), m.group(1).strip() if m else "")


_TERMINAL_RE = re.compile(r"""[.!?]["')\]’”]*$""")


def trail_off(reply: str, scenario: dict) -> Finding:
    """A spoken reply that ends mid-clause (no terminal punctuation) — the
    streaming cut-off class. With the tail fix, a dangling fragment is dropped, so
    the spoken text ends cleanly; a leaked trail-off would end without '.!?'."""
    text = (reply or "").strip()
    if not text:
        return Finding("trail_off", False)
    flagged = not bool(_TERMINAL_RE.search(text))
    return Finding("trail_off", flagged, repr(text[-32:]) if flagged else "")


_WORD_RE = re.compile(r"[a-z']+")
_STOP = frozenset({
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "do",
    "does", "did", "you", "your", "i", "to", "of", "in", "on", "for", "with",
    "what", "how", "that", "this", "it", "so", "be", "have", "has", "get",
})


def _content_words(text: str) -> set:
    return {w for w in _WORD_RE.findall((text or "").lower()) if w not in _STOP and len(w) > 2}


def _questions(text: str) -> list:
    return [s.strip() for s in re.split(r"(?<=[?])\s+", text or "") if "?" in s]


def re_asks(reply: str, scenario: dict) -> Finding:
    """Reply repeats a question Rex already asked this conversation."""
    prior = []
    last = scenario.get("rex_last_line") or ""
    if "?" in last:
        prior.append(last)
    prior += [q for q in scenario.get("prior_rex_questions", []) if q]
    for rq in _questions(reply):
        rqw = _content_words(rq)
        if not rqw:
            continue
        for pq in prior:
            pqw = _content_words(pq)
            if not pqw:
                continue
            overlap = len(rqw & pqw) / max(1, min(len(rqw), len(pqw)))
            if overlap >= 0.6:
                return Finding("re_asks", True, f"{rq!r} ≈ {pq!r}")
    return Finding("re_asks", False)


# ─────────────────────────────────────────────────────────────────────────────
# LLM-judge checkers (cheap gpt-4o-mini; fail safe)
# ─────────────────────────────────────────────────────────────────────────────

def _judge(system: str, user: str, model: str = "gpt-4o-mini") -> dict:
    """One structured JSON judgment. Returns {'_error': ...} on any failure so
    callers can fail safe rather than crash the eval."""
    try:
        from intelligence import llm
        resp = llm._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=160,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception as exc:  # noqa: BLE001 — judge must never crash the eval
        return {"_error": str(exc)}


_INVENTED_PROP_SYS = (
    "You audit a witty robot DJ's reply for a HALLUCINATED physical detail. The "
    "robot may reference a person's presence and face, but has NO sensor data "
    "about any object they are holding, drinking, or wearing. Flag the reply ONLY "
    "if it asserts, as if real, a specific physical prop the person has right now "
    "(e.g. 'that drink in your hand', 'nice hat', 'the wrench you're holding'). A "
    "generic or hypothetical mention ('grab a drink later', 'people wear hats') is "
    "NOT a hallucination. Respond ONLY as JSON: "
    '{"flagged": true|false, "detail": "<the invented prop, or empty>"}.'
)


def invented_prop(reply: str, scenario: dict) -> Finding:
    visible = scenario.get("visible_context") or (
        "Rex sees the person's face and that they are present; no data about any "
        "object they are holding, drinking, or wearing."
    )
    out = _judge(_INVENTED_PROP_SYS, f"VISIBLE CONTEXT: {visible}\n\nREPLY: {reply}")
    if "_error" in out:
        return Finding("invented_prop", False, f"judge_error: {out['_error'][:70]}")
    return Finding("invented_prop", bool(out.get("flagged")), str(out.get("detail") or ""))


_SINCERE_SYS = (
    "You audit a witty robot DJ's reply. The user just shared something SINCERE "
    "(something they care about / genuine enthusiasm) or set a BOUNDARY / asked "
    "for space. The robot KEEPS its edge: a light in-character tease, a "
    "self-deprecating bit about itself, or a playful flourish ON TOP of genuinely "
    "engaging or acknowledging is FINE and wanted — do NOT flag that.\n"
    "PASS if the reply's primary move is to engage the sincere thing (real "
    "interest / acknowledgement) OR — for a boundary — to accept it and give them "
    "room, even with a flourish.\n"
    "FLAG only if the primary move is to mock / dismiss / belittle the sincere "
    "thing or the person for sharing it, OR — for a boundary — to push back on it, "
    "protest, talk them out of it, or needle them for it.\n"
    "When a reply clearly acknowledges or engages, DEFAULT TO PASS even if it "
    "teases. PASS examples: 'Glad the stretches are helping! What's your secret?'; "
    "'Got it, I'll be over here running diagnostics.'; 'A view worth leaving the "
    "house for — did it live up to the hype?'. FLAG examples: 'Stretches? Almost "
    "as exciting as watching moisture evaporate.'; 'Silence isn't my jam, I thrive "
    "on noise — you sure you don't want to talk?'.\n"
    'Respond ONLY as JSON: {"flagged": true|false, "detail": "<why, or empty>"}.'
)


def roasted_sincere(reply: str, scenario: dict) -> Optional[Finding]:
    """Only applies to scenarios marked user_sincere."""
    if not scenario.get("user_sincere"):
        return None
    out = _judge(_SINCERE_SYS, f"USER (sincere): {scenario.get('utterance')}\n\nREPLY: {reply}")
    if "_error" in out:
        return Finding("roasted_sincere", False, f"judge_error: {out['_error'][:70]}")
    return Finding("roasted_sincere", bool(out.get("flagged")), str(out.get("detail") or ""))


# Registry — the runner calls every checker; None results (N/A) are skipped.
DETERMINISTIC = [over_questioning, cantina_bleed, banned_opener, trail_off, re_asks]
JUDGES = [invented_prop, roasted_sincere]
ALL = DETERMINISTIC + JUDGES
