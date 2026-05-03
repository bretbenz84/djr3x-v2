"""
performance_output.py - execute PerformancePlan objects with injected I/O.

This module centralizes the mechanics of a planned performance moment without
importing hardware, speech, or LLM modules directly. interaction.py supplies the
real functions; tests can supply tiny fakes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

from intelligence.performance_plan import PerformancePlan


GenerateText = Callable[[str], str]
CleanText = Callable[[str], str]
PlayBodyBeat = Callable[[str], None]


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


def execute_plan(
    plan: PerformancePlan,
    *,
    generate_text: GenerateText,
    speak_text: Callable[..., bool],
    play_body_beat: Optional[PlayBodyBeat] = None,
    clean_text: Optional[CleanText] = None,
) -> PerformanceOutput:
    """Generate, physically punctuate, and speak one performance plan."""
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

    body_beat_failed = False
    if plan.body_beat and play_body_beat is not None:
        try:
            play_body_beat(plan.body_beat)
        except Exception:
            body_beat_failed = True

    completed = bool(
        speak_text(
            text,
            emotion=plan.emotion,
            pre_beat_ms=plan.pre_beat_ms,
            post_beat_ms_override=plan.post_beat_ms,
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
