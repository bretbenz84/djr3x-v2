"""
performance_plan.py - side-effect-free performance choreography for Rex.

The action router decides what kind of moment this is. A PerformancePlan decides
how Rex should perform that moment: prompt contract, emotion, body beat,
delivery style, and memory policy. Execution still belongs to interaction.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


MEMORY_NORMAL = "normal"
MEMORY_DO_NOT_STORE = "do_not_store"


@dataclass(frozen=True)
class PerformancePlan:
    """A small, deterministic contract for performing one routed action."""

    action: str
    prompt_contract: str = ""
    fallback_text: str = ""
    emotion: str = "neutral"
    body_beat: Optional[str] = None
    delivery_style: str = "normal"
    memory_policy: str = MEMORY_NORMAL
    pre_beat_ms: int = 0
    post_beat_ms: int = 0
    requires_llm: bool = True


def _arg_text(args: dict[str, Any] | None, *keys: str) -> str:
    args = args or {}
    for key in keys:
        value = args.get(key)
        if value is None:
            continue
        text = " ".join(str(value).strip().split())
        if text:
            return text
    return ""


def plan_for_action(
    action: str,
    *,
    user_text: str = "",
    args: dict[str, Any] | None = None,
) -> PerformancePlan | None:
    """Return a deterministic performance plan for a stable action key."""
    action = str(action or "").strip()
    text = str(user_text or "").strip()

    if action == "humor.tell_joke":
        return PerformancePlan(
            action=action,
            prompt_contract=(
                "The user explicitly asked for a joke: "
                f"{text!r}. Tell exactly ONE short in-character DJ-R3X joke, pun, "
                "or one-liner. Rex may use droid, Star Tours, Batuu, cantina, DJ, "
                "or organic-life humor. No explanation, no apology, no follow-up "
                "question, no sensitive topics. Deliver the punchline and stop."
            ),
            fallback_text=(
                "I tried writing a joke about my flight record. "
                "The punchline filed an insurance claim."
            ),
            emotion="happy",
            body_beat="dramatic_visor_peek",
            delivery_style="quick_punchline",
            memory_policy=MEMORY_DO_NOT_STORE,
        )

    if action == "humor.roast":
        target = _arg_text(args, "target", "person", "name") or "speaker"
        return PerformancePlan(
            action=action,
            prompt_contract=(
                "The user explicitly asked for a roast: "
                f"{text!r}. Roast target: {target!r}. Deliver exactly ONE playful, "
                "consent-based Rex roast. Keep it affectionate, surface-level, and "
                "about the current vibe, the request, the room, organic indecision, "
                "or Rex's droid perspective. Do NOT joke about body, age, gender, "
                "race, religion, disability, health, money, identity, grief, private "
                "text, trauma, family, or anything intimate. If the target is not "
                "the speaker, keep it extra gentle and public. No question. One "
                "line only."
            ),
            fallback_text=(
                "Fine. Consider yourself roasted: medium rare confidence, "
                "fully cooked decision-making."
            ),
            emotion="curious",
            body_beat="suspicious_glance",
            delivery_style="consent_roast",
            memory_policy=MEMORY_DO_NOT_STORE,
        )

    if action == "humor.free_bit":
        return PerformancePlan(
            action=action,
            prompt_contract=(
                "The user explicitly asked Rex to be funny or do a bit: "
                f"{text!r}. Give exactly ONE short in-character Rex riff. Favor "
                "self-deprecation, cantina/DJ patter, droid irritation, empty-room "
                "absurdity, or broad organic-life observational humor. No explanation, "
                "no follow-up question, no sensitive topics. One line only."
            ),
            fallback_text=(
                "I would do observational comedy, but the room is mostly "
                "observing me fail upward."
            ),
            emotion="happy",
            body_beat="proud_dj_pose",
            delivery_style="quick_riff",
            memory_policy=MEMORY_DO_NOT_STORE,
        )

    if action == "performance.dj_bit":
        return PerformancePlan(
            action=action,
            prompt_contract=(
                "The user asked for DJ-R3X cantina patter, hype, or a station-break "
                f"bit: {text!r}. Give exactly ONE short in-character DJ line. Do not "
                "start music. No follow-up question."
            ),
            fallback_text="Systems nominal, vibes questionable, DJ superiority intact.",
            emotion="happy",
            body_beat="proud_dj_pose",
            delivery_style="dj_stinger",
            memory_policy=MEMORY_DO_NOT_STORE,
        )

    if action == "performance.body_beat":
        beat = _arg_text(args, "body_beat", "beat", "gesture", "pose") or "thinking_tilt"
        return PerformancePlan(
            action=action,
            fallback_text="Physical expression logged. Very advanced. Very unnecessary.",
            emotion="curious",
            body_beat=beat,
            delivery_style="physical_beat",
            memory_policy=MEMORY_DO_NOT_STORE,
            requires_llm=False,
        )

    return None
