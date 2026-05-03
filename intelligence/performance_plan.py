"""
performance_plan.py - side-effect-free performance choreography for Rex.

The action router decides what kind of moment this is. A PerformancePlan decides
how Rex should perform that moment: prompt contract, emotion, body beat,
delivery style, and memory policy. performance_output.py executes the plan with
I/O supplied by interaction.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


MEMORY_NORMAL = "normal"
MEMORY_DO_NOT_STORE = "do_not_store"

BODY_BEAT_NAMES = frozenset({
    "dramatic_visor_peek",
    "offended_recoil",
    "proud_dj_pose",
    "suspicious_glance",
    "thinking_tilt",
    "tiny_victory_dance",
})
MOOD_POSE_NAMES = frozenset({
    "annoyed",
    "embarrassed",
    "happy",
    "offended",
    "proud",
    "suspicious",
    "thinking",
})

_BODY_BEAT_ALIASES = {
    "correct": "tiny_victory_dance",
    "correct_answer": "tiny_victory_dance",
    "dj_pose": "proud_dj_pose",
    "dj_start": "proud_dj_pose",
    "game_correct": "tiny_victory_dance",
    "game_wrong": "suspicious_glance",
    "insult": "offended_recoil",
    "insult_recoil": "offended_recoil",
    "offended": "offended_recoil",
    "proud": "proud_dj_pose",
    "side_eye": "suspicious_glance",
    "suspicious": "suspicious_glance",
    "think": "thinking_tilt",
    "thinking": "thinking_tilt",
    "tiny_dance": "tiny_victory_dance",
    "victory": "tiny_victory_dance",
    "victory_dance": "tiny_victory_dance",
    "visor_peek": "dramatic_visor_peek",
    "wrong": "suspicious_glance",
    "wrong_answer": "suspicious_glance",
}

_BODY_BEAT_FALLBACKS = {
    "dramatic_visor_peek": "Dramatic visor peek. Very subtle. Nobody panic.",
    "offended_recoil": "Offended recoil. Bold choice, organic.",
    "proud_dj_pose": "Proud DJ pose. The booth respects me.",
    "suspicious_glance": "Suspicious glance engaged. I distrust the room professionally.",
    "thinking_tilt": "Thinking tilt. It makes the processors look busy.",
    "tiny_victory_dance": "Tiny victory dance deployed. Try not to be intimidated.",
}

_BODY_BEAT_EMOTIONS = {
    "dramatic_visor_peek": "curious",
    "offended_recoil": "angry",
    "proud_dj_pose": "happy",
    "suspicious_glance": "curious",
    "thinking_tilt": "curious",
    "tiny_victory_dance": "happy",
}
_MOOD_POSE_ALIASES = {
    "bashful": "embarrassed",
    "confused": "thinking",
    "delighted": "happy",
    "excited": "happy",
    "fed_up": "annoyed",
    "insulted": "offended",
    "irritated": "annoyed",
    "sheepish": "embarrassed",
    "skeptical": "suspicious",
    "smug": "proud",
    "thoughtful": "thinking",
}
_MOOD_POSE_BODY_BEATS = {
    "annoyed": "offended_recoil",
    "embarrassed": "dramatic_visor_peek",
    "happy": "tiny_victory_dance",
    "offended": "offended_recoil",
    "proud": "proud_dj_pose",
    "suspicious": "suspicious_glance",
    "thinking": "thinking_tilt",
}
_MOOD_POSE_FALLBACKS = {
    "annoyed": "Annoyed pose. I am mostly dignity and warranty concerns.",
    "embarrassed": "Embarrassed pose. My confidence briefly went into maintenance mode.",
    "happy": "Happy pose. Alarming, but apparently operational.",
    "offended": "Offended pose. I have filed a complaint with myself.",
    "proud": "Proud pose. Try not to applaud the machinery.",
    "suspicious": "Suspicious pose. I trust absolutely everyone, which is to say no one.",
    "thinking": "Thinking pose. Please admire the illusion of wisdom.",
}
_MOOD_POSE_EMOTIONS = {
    "annoyed": "angry",
    "embarrassed": "curious",
    "happy": "happy",
    "offended": "angry",
    "proud": "happy",
    "suspicious": "curious",
    "thinking": "curious",
}

_ACTION_BODY_BEATS = {
    "humor.tell_joke": "dramatic_visor_peek",
    "humor.roast": "suspicious_glance",
    "humor.free_bit": "proud_dj_pose",
    "performance.dj_bit": "proud_dj_pose",
}

_EVENT_BODY_BEATS = {
    "action": None,
    "correction.accepted": "thinking_tilt",
    "dj.bit": "proud_dj_pose",
    "empty.room.joke": "thinking_tilt",
    "game.correct": "tiny_victory_dance",
    "game.loss": "offended_recoil",
    "game.start": "proud_dj_pose",
    "game.thinking": "thinking_tilt",
    "game.timeout": "dramatic_visor_peek",
    "game.win": "tiny_victory_dance",
    "game.wrong": "suspicious_glance",
    "humor.free.bit": "proud_dj_pose",
    "humor.joke": "dramatic_visor_peek",
    "humor.roast": "suspicious_glance",
    "idle.empty.room": "thinking_tilt",
    "insult.detected": "offended_recoil",
    "misunderstanding.correction": "thinking_tilt",
    "repair.factual": "thinking_tilt",
    "repair.misheard": "thinking_tilt",
    "repair.misunderstood": "thinking_tilt",
    "repair.pronoun": "thinking_tilt",
}


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


def _body_key(value: str) -> str:
    return "_".join(str(value or "").strip().lower().replace("-", "_").split())


def _event_key(value: str) -> str:
    text = str(value or "").strip().lower().replace("_", ".").replace("-", ".")
    parts = [part for chunk in text.split(".") for part in chunk.split() if part]
    return ".".join(parts)


def canonical_body_beat(name: str) -> Optional[str]:
    """Return a stable body-beat name for direct names and friendly aliases."""
    key = _body_key(name)
    if key in BODY_BEAT_NAMES:
        return key
    return _BODY_BEAT_ALIASES.get(key)


def canonical_mood_pose(name: str) -> Optional[str]:
    """Return a stable mood-pose name for direct names and friendly aliases."""
    key = _body_key(name)
    if key in MOOD_POSE_NAMES:
        return key
    return _MOOD_POSE_ALIASES.get(key)


def body_beat_for_event(
    event: str,
    *,
    action: str = "",
    emotion: str = "",
    outcome: str = "",
    repair_kind: str = "",
    body_beat: str = "",
) -> Optional[str]:
    """
    Map semantic moments to Rex's named body beats.

    This keeps physical theatre deterministic: code can say "insult.detected"
    or "game.correct" and the servo layer only receives known pose names.
    """
    explicit = canonical_body_beat(body_beat)
    if explicit:
        return explicit

    action_beat = _ACTION_BODY_BEATS.get(str(action or "").strip())
    if action_beat:
        return action_beat

    event_key = _event_key(event)
    if event_key == "action":
        return None

    if event_key == "repair" and repair_kind:
        repair_key = _event_key(f"repair.{repair_kind}")
        beat = _EVENT_BODY_BEATS.get(repair_key)
        if beat:
            return beat

    if event_key == "game" and outcome:
        outcome_key = _event_key(f"game.{outcome}")
        beat = _EVENT_BODY_BEATS.get(outcome_key)
        if beat:
            return beat

    beat = _EVENT_BODY_BEATS.get(event_key)
    if beat:
        return beat

    emotion_key = str(emotion or "").strip().lower()
    if emotion_key in {"happy", "excited", "proud"}:
        return "tiny_victory_dance"
    if emotion_key in {"curious", "confused", "uncertain"}:
        return "thinking_tilt"
    if emotion_key in {"annoyed", "angry", "offended"}:
        return "offended_recoil"
    return None


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
            body_beat=body_beat_for_event("action", action=action),
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
            body_beat=body_beat_for_event("action", action=action),
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
            body_beat=body_beat_for_event("action", action=action),
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
            body_beat=body_beat_for_event("action", action=action),
            delivery_style="dj_stinger",
            memory_policy=MEMORY_DO_NOT_STORE,
        )

    if action == "performance.body_beat":
        beat = _arg_text(args, "body_beat", "beat", "gesture", "pose") or "thinking_tilt"
        canonical = canonical_body_beat(beat) or "thinking_tilt"
        return PerformancePlan(
            action=action,
            fallback_text=_BODY_BEAT_FALLBACKS.get(
                canonical,
                "Physical expression logged. Very advanced. Very unnecessary.",
            ),
            emotion=_BODY_BEAT_EMOTIONS.get(canonical, "curious"),
            body_beat=canonical,
            delivery_style="physical_beat",
            memory_policy=MEMORY_DO_NOT_STORE,
            requires_llm=False,
        )

    if action == "performance.mood_pose":
        mood = _arg_text(args, "mood", "emotion", "pose") or "thinking"
        canonical = canonical_mood_pose(mood) or "thinking"
        beat = _MOOD_POSE_BODY_BEATS.get(canonical, "thinking_tilt")
        return PerformancePlan(
            action=action,
            fallback_text=_MOOD_POSE_FALLBACKS.get(
                canonical,
                "Mood pose engaged. The acting academy remains silent.",
            ),
            emotion=_MOOD_POSE_EMOTIONS.get(canonical, "curious"),
            body_beat=beat,
            delivery_style="mood_pose",
            memory_policy=MEMORY_DO_NOT_STORE,
            requires_llm=False,
        )

    return None
