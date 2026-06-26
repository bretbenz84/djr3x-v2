"""
performance_output.py - execute PerformancePlan objects with injected I/O.

This module centralizes the mechanics of a planned performance moment without
importing hardware, speech, or LLM modules directly. interaction.py supplies the
real functions; tests can supply tiny fakes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import re
from typing import Optional

import config
from intelligence import performance_plan
from intelligence.performance_plan import PerformancePlan


GenerateText = Callable[[str], str]
CleanText = Callable[[str], str]
PlayBodyBeat = Callable[[str], None]

# Delivery styles whose body beat should LAND in the post-line silence (the comedic
# "button" after the line) rather than fire upfront over the front of the line.
_POST_LINE_STYLES = frozenset({"quick_punchline", "consent_roast", "quick_riff"})


@dataclass(frozen=True)
class PerformanceOutput:
    """Result of executing a planned performance moment."""

    text: str
    completed: bool
    action: str
    emotion: str
    body_beat: Optional[str]
    delivery_style: str
    memory_policy: str
    generation_failed: bool = False
    body_beat_failed: bool = False


def _clean(text: str, clean_text: Optional[CleanText]) -> str:
    if clean_text is None:
        return " ".join(str(text or "").strip().split())
    return str(clean_text(text or "") or "").strip()


def _split_setup_question_and_punchline(text: str) -> Optional[tuple[str, str]]:
    """Return setup/punchline when text has a clear question-then-answer shape."""
    cleaned = " ".join(str(text or "").strip().split())
    if not cleaned:
        return None
    match = re.search(r"\?\s+", cleaned)
    if not match:
        return None
    setup = cleaned[: match.start() + 1].strip()
    punchline = cleaned[match.end():].strip()
    if not setup or not punchline:
        return None
    return setup, punchline


def execute_plan(
    plan: PerformancePlan,
    *,
    generate_text: GenerateText,
    speak_text: Callable[..., bool],
    play_body_beat: Optional[PlayBodyBeat] = None,
    play_landing_body_beat: Optional[PlayBodyBeat] = None,
    clean_text: Optional[CleanText] = None,
    on_text: Optional[Callable[[str], None]] = None,
) -> PerformanceOutput:
    """Generate, physically punctuate, and speak one performance plan.

    For the comedic-landing delivery styles (_POST_LINE_STYLES) a `play_landing_body_beat`
    may be supplied: the body beat is then deferred to fire the instant the line's audio
    ends (via the speak layer's on_audio_end hook) so it lands IN the post-line silence —
    "line lands -> beat of silence -> button" — instead of over the front of the line.
    Falls back to the upfront `play_body_beat` when no landing player is given or the
    PERFORMANCE_POST_LINE_BEAT_ENABLED flag is off."""
    raw = ""
    generation_failed = False
    if plan.requires_llm and plan.prompt_contract:
        try:
            raw = generate_text(plan.prompt_contract) or ""
        except Exception:
            raw = ""
            generation_failed = True
    else:
        raw = plan.fallback_text

    text = _clean(raw, clean_text)
    if not text:
        text = _clean(plan.fallback_text, clean_text) or str(plan.fallback_text or "").strip()

    # Surface the line to the transcript the MOMENT it's generated — BEFORE the
    # blocking speak — so the GUI shows it immediately (read-along) instead of after
    # TTS finishes. The spoken lines below pass log_text=False; the caller's later
    # conv_log.log_rex of the returned text dedupes against this one.
    if on_text is not None and text:
        try:
            on_text(text)
        except Exception:
            pass

    # Decide upfront-vs-post-line for the body beat. A "landing" beat is deferred to
    # the END of the line's audio (the comedic button in the silence); everything else
    # keeps firing upfront over the front of the line.
    landing = bool(
        getattr(config, "PERFORMANCE_POST_LINE_BEAT_ENABLED", True)
        and plan.body_beat
        and play_landing_body_beat is not None
        and plan.delivery_style in _POST_LINE_STYLES
    )
    on_audio_end = (lambda: play_landing_body_beat(plan.body_beat)) if landing else None
    # Only the FINAL spoken line (the punchline / the single line) gets the landing —
    # never a setup line, or the button would fire mid-joke.
    _land = {"on_audio_end": on_audio_end} if on_audio_end is not None else {}

    body_beat_failed = False
    if plan.body_beat and play_body_beat is not None and not landing:
        try:
            play_body_beat(plan.body_beat)
        except Exception:
            body_beat_failed = True

    completed = False
    if plan.delivery_style == "quick_punchline":
        split = _split_setup_question_and_punchline(text)
        if split is not None:
            setup, punchline = split
            # Deliver the joke as setup + punchline, but DON'T log each piece —
            # the caller logs the whole joke once (otherwise the conversation log
            # showed the setup, the punchline, AND the full joke = three lines).
            completed = bool(
                speak_text(
                    setup,
                    emotion=plan.emotion,
                    pre_beat_ms=plan.pre_beat_ms,
                    post_beat_ms_override=0,
                    log_text=False,
                )
            )
            if completed:
                pause_ms = max(
                    0,
                    int(getattr(config, "JOKE_SETUP_PUNCHLINE_PAUSE_MS", 700) or 0),
                )
                completed = bool(
                    speak_text(
                        punchline,
                        emotion=plan.emotion,
                        pre_beat_ms=pause_ms,
                        post_beat_ms_override=plan.post_beat_ms,
                        log_text=False,
                        **_land,
                    )
                )
        else:
            completed = bool(
                speak_text(
                    text,
                    emotion=plan.emotion,
                    pre_beat_ms=plan.pre_beat_ms,
                    post_beat_ms_override=plan.post_beat_ms,
                    log_text=False,
                    **_land,
                )
            )
    else:
        completed = bool(
            speak_text(
                text,
                emotion=plan.emotion,
                pre_beat_ms=plan.pre_beat_ms,
                post_beat_ms_override=plan.post_beat_ms,
                log_text=False,
                **_land,
            )
        )
    return PerformanceOutput(
        text=text,
        completed=completed,
        action=plan.action,
        emotion=plan.emotion,
        body_beat=plan.body_beat,
        delivery_style=plan.delivery_style,
        memory_policy=plan.memory_policy,
        generation_failed=generation_failed,
        body_beat_failed=body_beat_failed,
    )


def execute_body_beat_event(
    event: str,
    *,
    play_body_beat: Optional[PlayBodyBeat],
    action: str = "",
    emotion: str = "",
    outcome: str = "",
    repair_kind: str = "",
    body_beat: str = "",
) -> Optional[str]:
    """Play the deterministic body beat for a semantic event, if one exists."""
    if play_body_beat is None:
        return None
    beat = performance_plan.body_beat_for_event(
        event,
        action=action,
        emotion=emotion,
        outcome=outcome,
        repair_kind=repair_kind,
        body_beat=body_beat,
    )
    if not beat:
        return None
    try:
        play_body_beat(beat)
    except Exception:
        return None
    return beat
