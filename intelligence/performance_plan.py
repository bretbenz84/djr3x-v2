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
    "agreement_nod",
    "anger_flash",
    "disagreement_shake",
    "disbelief_stare",
    "dramatic_visor_peek",
    "disgust_recoil",
    "giddy_wiggle",
    "happy_bounce",
    "offended_recoil",
    "proud_dj_pose",
    "sad_droop",
    "surprise_pop",
    "suspicious_glance",
    "thinking_tilt",
    "tiny_victory_dance",
})
MOOD_POSE_NAMES = frozenset({
    "agreement",
    "annoyed",
    "angry",
    "disagreement",
    "disbelief",
    "disgusted",
    "embarrassed",
    "giddy",
    "happy",
    "offended",
    "proud",
    "sad",
    "surprised",
    "suspicious",
    "thinking",
})

_BODY_BEAT_ALIASES = {
    "agree": "agreement_nod",
    "agreement": "agreement_nod",
    "anger": "anger_flash",
    "angry": "anger_flash",
    "correct": "tiny_victory_dance",
    "correct_answer": "tiny_victory_dance",
    "disagree": "disagreement_shake",
    "disagreement": "disagreement_shake",
    "disbelief": "disbelief_stare",
    "dj_pose": "proud_dj_pose",
    "dj_start": "proud_dj_pose",
    "furious": "anger_flash",
    "game_correct": "tiny_victory_dance",
    "game_wrong": "suspicious_glance",
    "giddy": "giddy_wiggle",
    "giddy_joy": "giddy_wiggle",
    "grossed_out": "disgust_recoil",
    "happy": "happy_bounce",
    "head_shake": "disagreement_shake",
    "insult": "offended_recoil",
    "insult_recoil": "offended_recoil",
    "joy": "giddy_wiggle",
    "mad": "anger_flash",
    "no": "disagreement_shake",
    "nod": "agreement_nod",
    "offended": "offended_recoil",
    "proud": "proud_dj_pose",
    "sad": "sad_droop",
    "side_eye": "suspicious_glance",
    "shocked": "surprise_pop",
    "surprise": "surprise_pop",
    "surprised": "surprise_pop",
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
    "agreement_nod": "Mmhmm.",
    "anger_flash": "Grr. Systems annoyed.",
    "disagreement_shake": "Nope.",
    "disbelief_stare": "Processing... that was a choice.",
    "dramatic_visor_peek": "Dramatic visor peek. Very subtle. Nobody panic.",
    "disgust_recoil": "Bleh. Rejecting that input with enthusiasm.",
    "giddy_wiggle": "Heh-heh. Systems delighted.",
    "happy_bounce": "Mmhmm. That one gets a happy bounce.",
    "offended_recoil": "Offended recoil. Bold choice, organic.",
    "proud_dj_pose": "Proud DJ pose. The booth respects me.",
    "sad_droop": "Aww. Minor systems droop.",
    "surprise_pop": "Yip.",
    "suspicious_glance": "Suspicious glance engaged. I distrust the room professionally.",
    "thinking_tilt": "Thinking tilt. It makes the processors look busy.",
    "tiny_victory_dance": "Tiny victory dance deployed. Try not to be intimidated.",
}

_BODY_BEAT_EMOTIONS = {
    "agreement_nod": "happy",
    "anger_flash": "angry",
    "disagreement_shake": "curious",
    "disbelief_stare": "curious",
    "dramatic_visor_peek": "curious",
    "disgust_recoil": "angry",
    "giddy_wiggle": "excited",
    "happy_bounce": "happy",
    "offended_recoil": "angry",
    "proud_dj_pose": "happy",
    "sad_droop": "sad",
    "surprise_pop": "excited",
    "suspicious_glance": "curious",
    "thinking_tilt": "curious",
    "tiny_victory_dance": "happy",
}
_MOOD_POSE_ALIASES = {
    "agree": "agreement",
    "delight": "giddy",
    "bashful": "embarrassed",
    "confused": "thinking",
    "delighted": "giddy",
    "disgust": "disgusted",
    "grossed_out": "disgusted",
    "excited": "giddy",
    "fed_up": "annoyed",
    "furious": "angry",
    "mad": "angry",
    "insulted": "offended",
    "irritated": "annoyed",
    "joy": "giddy",
    "joyful": "giddy",
    "no": "disagreement",
    "sheepish": "embarrassed",
    "shocked": "surprised",
    "skeptical": "suspicious",
    "smug": "proud",
    "startled": "surprised",
    "thoughtful": "thinking",
    "yes": "agreement",
}
_MOOD_POSE_BODY_BEATS = {
    "agreement": "agreement_nod",
    "annoyed": "offended_recoil",
    "angry": "anger_flash",
    "disagreement": "disagreement_shake",
    "disbelief": "disbelief_stare",
    "disgusted": "disgust_recoil",
    "embarrassed": "dramatic_visor_peek",
    "giddy": "giddy_wiggle",
    "happy": "happy_bounce",
    "offended": "offended_recoil",
    "proud": "proud_dj_pose",
    "sad": "sad_droop",
    "surprised": "surprise_pop",
    "suspicious": "suspicious_glance",
    "thinking": "thinking_tilt",
}
_MOOD_POSE_FALLBACKS = {
    "agreement": "Mmhmm.",
    "annoyed": "Annoyed pose. I am mostly dignity and warranty concerns.",
    "angry": "Grr. Tiny anger subroutine, tastefully deployed.",
    "disagreement": "Nope.",
    "disbelief": "I am staring in disbelief. Respect the processing time.",
    "disgusted": "Bleh. Strong sensory objection logged.",
    "embarrassed": "Embarrassed pose. My confidence briefly went into maintenance mode.",
    "giddy": "Heh-heh. That was giddy joy. Very controlled. Mostly.",
    "happy": "Happy pose. Alarming, but apparently operational.",
    "offended": "Offended pose. I have filed a complaint with myself.",
    "proud": "Proud pose. Try not to applaud the machinery.",
    "sad": "Sad pose. Yes, even the hardware can slump.",
    "surprised": "Yip.",
    "suspicious": "Suspicious pose. I trust absolutely everyone, which is to say no one.",
    "thinking": "Thinking pose. Please admire the illusion of wisdom.",
}
_MOOD_POSE_EMOTIONS = {
    "agreement": "happy",
    "annoyed": "angry",
    "angry": "angry",
    "disagreement": "curious",
    "disbelief": "curious",
    "disgusted": "angry",
    "embarrassed": "curious",
    "giddy": "excited",
    "happy": "happy",
    "offended": "angry",
    "proud": "happy",
    "sad": "sad",
    "surprised": "excited",
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
    "amusement.detected": "giddy_wiggle",
    "compliment.detected": "proud_dj_pose",
    "correction.accepted": "thinking_tilt",
    "dj.bit": "proud_dj_pose",
    "empty.room.joke": "thinking_tilt",
    "emotion.agreement": "agreement_nod",
    "emotion.anger": "anger_flash",
    "emotion.disagreement": "disagreement_shake",
    "emotion.disbelief": "disbelief_stare",
    "emotion.disgust": "disgust_recoil",
    "emotion.giddy": "giddy_wiggle",
    "emotion.happiness": "happy_bounce",
    "emotion.sadness": "sad_droop",
    "emotion.surprise": "surprise_pop",
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
    "insult.detected": "anger_flash",
    "misunderstanding.correction": "thinking_tilt",
    "preference.negative": "disagreement_shake",
    "preference.positive": "agreement_nod",
    "preference.strong.negative": "disgust_recoil",
    "preference.strong.positive": "giddy_wiggle",
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
    if emotion_key in {"surprised", "surprise", "startled", "shocked"}:
        return "surprise_pop"
    if emotion_key in {"happy", "proud"}:
        return "happy_bounce"
    if emotion_key in {"excited", "giddy", "joy", "joyful"}:
        return "giddy_wiggle"
    if emotion_key in {"curious", "confused", "uncertain", "thinking"}:
        return "thinking_tilt"
    if emotion_key in {"annoyed", "offended"}:
        return "offended_recoil"
    if emotion_key in {"angry", "mad", "furious"}:
        return "anger_flash"
    if emotion_key in {"disgust", "disgusted", "grossed_out"}:
        return "disgust_recoil"
    if emotion_key in {"sad", "sadness", "dejected"}:
        return "sad_droop"
    return None


# Distinct comedic lanes for "tell me a joke". The dispatch rotates through these
# (round-robin) so a rapid joke chain doesn't keep landing on the same premise —
# in the field, four "tell me a joke" requests all returned the same DJ/bad-pilot
# self-own. The self-own is now just one lane of six.
JOKE_ANGLES: tuple[str, ...] = (
    "Riff on organic-life behavior — humans and their indecision, snacks, naps, "
    "or staring at screens.",
    "Riff on droid life — firmware, maintenance, reboots, memory banks, or being "
    "the only one in the room with a warranty.",
    "Riff on music, beats, basslines, or life in the DJ booth.",
    "Riff on Batuu, Black Spire, Star Tours, smugglers, or spaceport life.",
    "Lean on a pun or a piece of absurd wordplay — a groaner is fair game.",
    "Do a self-deprecating Star Tours bit about your flight record or your "
    "programming.",
)


def joke_angle_directive(rotation: Optional[int]) -> str:
    """Pure helper: the comedic-lane directive for a round-robin rotation index."""
    if rotation is None:
        return ""
    return JOKE_ANGLES[int(rotation) % len(JOKE_ANGLES)]


def plan_for_action(
    action: str,
    *,
    user_text: str = "",
    args: dict[str, Any] | None = None,
    joke_rotation: Optional[int] = None,
    joke_avoid_directive: str = "",
) -> PerformancePlan | None:
    """Return a deterministic performance plan for a stable action key.

    ``joke_rotation`` / ``joke_avoid_directive`` are supplied by the dispatch for
    humor.tell_joke so each joke picks a fresh comedic lane and steers clear of
    premises Rex has already spent this conversation. The function stays pure — all
    rotation/recency state lives in the caller.
    """
    action = str(action or "").strip()
    text = str(user_text or "").strip()

    if action == "humor.tell_joke":
        angle = joke_angle_directive(joke_rotation)
        angle_line = f" ANGLE FOR THIS ONE: {angle}" if angle else ""
        avoid_line = f" {joke_avoid_directive}" if joke_avoid_directive else ""
        return PerformancePlan(
            action=action,
            prompt_contract=(
                "The user explicitly asked for a joke: "
                f"{text!r}. Tell exactly ONE short in-character DJ-R3X joke, pun, "
                "or one-liner."
                f"{angle_line} Bring a genuinely NEW joke — a different premise, "
                "setup, and punchline than your recent ones; do NOT keep recycling "
                "the same flight/landing/\"bad pilot\" self-own."
                f"{avoid_line} "
                "No explanation, no apology, no follow-up question, no sensitive "
                "topics. Deliver the punchline and stop."
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
                "absurdity, broad organic-life observational humor, or Star Tours "
                "programming self-owns. No explanation, no follow-up question, no "
                "sensitive topics. One line only."
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
