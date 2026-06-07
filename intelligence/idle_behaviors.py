"""
idle_behaviors.py — Rex's idle micro-behaviours (the "content" he produces when nothing
is happening): empty-room jokes, ambient scans/observations, private thoughts, aspirations,
appearance riffs, people roasts, live-vision comments, the bored environmental snark, and
idle audio clips.

Extracted from consciousness.py to keep that module smaller. These behaviours are CONSUMERS
of consciousness's proactive-speech engine (generate/speak, the purpose-claim system, the
governor) and a few of its helpers, reached lazily via `consciousness as _c` (a deferred
import that resolves at call time — there is no import-time cycle). The dispatcher and the
behaviour-choice logic stay in consciousness (they are loop/state coupled); this module
holds only the implementations. `_step_idle_micro_behavior` calls them as
`idle_behaviors.do_<name>(...)`.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from pathlib import Path
from typing import Optional

import config
from intelligence import consciousness as _c

_log = logging.getLogger(__name__)


# ── Cooldown / dedupe state (moved with the behaviours) ──────────────────────
_last_live_vision_comment_at: float = 0.0
_last_bored_env_snark_at: float = 0.0
_last_aspiration: Optional[str] = None


def do_empty_room_joke(snapshot: dict) -> None:
    if not _c._can_proactive_speak():
        return
    if not _c._empty_room_commentary_allowed(snapshot):
        return
    if random.random() >= float(getattr(config, "EMPTY_ROOM_JOKE_PROBABILITY", 0.9)):
        return
    pool = getattr(config, "EMPTY_ROOM_JOKES", None) or getattr(config, "PRIVATE_THOUGHTS", [])
    if not pool:
        return
    token = _c._claim_proactive_purpose("idle_monologue", label="empty-room joke")
    if token is None:
        return
    line = random.choice(list(pool))
    try:
        if _c._proactive_purpose_current(token):
            try:
                from intelligence import performance_output
                from sequences import animations
                performance_output.execute_body_beat_event(
                    "idle.empty_room",
                    play_body_beat=animations.play_body_beat,
                )
            except Exception as exc:
                _log.debug("empty-room body beat skipped: %s", exc)
            _c._speak_async(
                line,
                emotion="neutral",
                purpose="idle_monologue",
                label="empty-room joke",
            )
    finally:
        _c._release_proactive_purpose(token)


def do_ambient_scan() -> None:
    try:
        from hardware.servos import set_servo
        neck_cfg = config.SERVO_CHANNELS["neck"]
        ch = neck_cfg["ch"]
        neutral = neck_cfg["neutral"]
        left_pos  = int(neutral - (neutral - neck_cfg["min"]) * 0.35)
        right_pos = int(neutral + (neck_cfg["max"] - neutral) * 0.35)

        def _scan():
            set_servo(ch, left_pos)
            time.sleep(1.5)
            set_servo(ch, right_pos)
            time.sleep(1.5)
            set_servo(ch, neutral)

        threading.Thread(target=_scan, daemon=True, name="ambient_scan").start()
    except Exception as exc:
        _log.debug("ambient scan error: %s", exc)


def do_private_thought() -> None:
    if not _c._can_proactive_speak():
        return
    if _c._voice_pov_as_micro_behavior(
        "private thought",
        "Voice your CURRENT preoccupation out loud as a brief private thought to "
        "yourself — like thinking aloud. Don't address anyone; just muse in one short "
        "in-character Rex sentence.",
        emotion="neutral",
    ):
        return
    token = _c._claim_proactive_purpose("idle_monologue", label="private thought")
    if token is None:
        return
    line = random.choice(config.PRIVATE_THOUGHTS)
    try:
        if _c._proactive_purpose_current(token):
            _c._speak_async(
                line,
                emotion="neutral",
                purpose="idle_monologue",
                label="private thought",
            )
    finally:
        _c._release_proactive_purpose(token)


def do_memory_musing() -> None:
    """Surface a cross-session recollection from Rex's diary (rex.db) as a brief,
    out-loud musing — "since I was last on" continuity. Gated by EPISODIC_RECALL_ENABLED
    and a probability so it stays a subtle, occasional spice (not every idle tick).
    No-op when recall is off or there's nothing worth recalling."""
    if not getattr(config, "EPISODIC_RECALL_ENABLED", False):
        return
    if not _c._can_proactive_speak():
        return
    if random.random() >= float(getattr(config, "EPISODIC_RECALL_SESSION_RECAP_PROBABILITY", 0.5)):
        return
    try:
        from memory import episodic_recall
        recap = episodic_recall.session_recap()
    except Exception:
        recap = None
    if not recap:
        return
    _c._generate_and_speak(
        f"You're idly thinking back on things you remember from before — here's what "
        f"comes to mind: {recap} In ONE short, dry, in-character Rex line, muse aloud "
        f"about something you recall — like a passing recollection. Don't greet anyone "
        f"or ask a question; just reminisce briefly. One line only.",
        emotion="neutral",
        purpose="memory_musing",
    )


def do_aspiration() -> None:
    """Speak one of Rex's forward-looking aspirations as an idle micro-behavior."""
    global _last_aspiration
    if not _c._can_proactive_speak():
        return
    if _c._voice_pov_as_micro_behavior(
        "aspiration",
        "Riff forward on your CURRENT preoccupation as a brief out-loud aspiration — "
        "where you'd like to take it or what you're working toward with it. One short "
        "in-character Rex sentence, thinking aloud.",
        emotion="curious",
    ):
        return
    pool = getattr(config, "ASPIRATIONS", None)
    if not pool:
        return
    token = _c._claim_proactive_purpose("idle_monologue", label="aspiration")
    if token is None:
        return
    candidates = [line for line in pool if line != _last_aspiration] or list(pool)
    chosen = random.choice(candidates)
    _last_aspiration = chosen
    try:
        if _c._proactive_purpose_current(token):
            _c._speak_async(
                chosen,
                emotion="curious",
                purpose="idle_monologue",
                label="aspiration",
            )
    finally:
        _c._release_proactive_purpose(token)


def do_ambient_observation(snapshot: dict) -> None:
    """
    Fire a short in-character remark about the current environment, pulled from
    world_state.environment — room type, lighting, crowd density, description.
    No vision call; uses data the periodic scene scanner already collected.
    """
    if random.random() >= getattr(config, "AMBIENT_OBSERVATION_PROBABILITY", 0.5):
        return
    env = snapshot.get("environment", {}) or {}
    audio_scene = snapshot.get("audio_scene", {}) or {}

    bits: list[str] = []
    if env.get("description"):
        bits.append(f"scene: {env['description']}")
    elif env.get("scene_type"):
        bits.append(f"scene type: {env['scene_type']}")
    if env.get("lighting"):
        bits.append(f"lighting: {env['lighting']}")
    if env.get("crowd_density"):
        bits.append(f"crowd density: {env['crowd_density']}")
    if audio_scene.get("ambient_level"):
        bits.append(f"ambient noise: {audio_scene['ambient_level']}")
    if audio_scene.get("music_detected"):
        bits.append("music is playing")

    if not bits:
        return
    context = "; ".join(bits)
    _c._generate_and_speak(
        f"You are idly observing your surroundings right now. Here is what you perceive "
        f"— {context}. In one short in-character Rex line, make an offhand observation "
        f"about the room or environment — like someone thinking out loud. Don't greet "
        f"anyone, don't ask a question; just a dry remark about the space or vibe. "
        f"One line only.",
        emotion="neutral",
        purpose="ambient_observation",
    )


def do_appearance_riff(snapshot: dict) -> None:
    """
    Pick one currently-visible known person and make an unprompted remark about
    their appearance (hair, clothes, notable features), using stored person_facts.
    No vision call; uses data from face enrollment.
    """
    people = snapshot.get("people", []) or []
    known = [p for p in people if p.get("person_db_id") and p.get("face_id")]
    if not known:
        return
    target = random.choice(known)
    hint = _c._pick_appearance_hint(target.get("person_db_id"))
    if not hint:
        return
    try:
        from memory import boundaries as _boundaries
        target_id = target.get("person_db_id")
        if (
            _boundaries.is_blocked(target_id, "mention", "appearance")
            or _boundaries.is_blocked(target_id, "roast", "appearance")
            or _boundaries.is_blocked(target_id, "mention", "clothing")
            or _boundaries.is_blocked(target_id, "roast", "clothing")
        ):
            return
    except Exception:
        pass
    # Don't riff on the engaged person — it'd feel interruptive mid-conversation.
    if _c.is_engaged_with(target.get("person_db_id")):
        return
    first_name = _c._first_name(target.get("face_id"), "there")
    _c._generate_and_speak(
        f"You're idly looking at '{first_name}'. You remember this about their "
        f"appearance: {hint}. Make one short in-character Rex remark about it — "
        f"the kind of thing you'd say while looking them over. Warm, dry, observational, "
        f"and lightly funny if the opening is there. "
        f"Address {first_name} by name. One line only.",
        emotion="neutral",
        purpose="appearance_riff",
    )


def do_people_roast(snapshot: dict) -> None:
    if not _c._can_proactive_speak():
        return
    if random.random() >= float(getattr(config, "PEOPLE_ROAST_RIFF_PROBABILITY", 0.75)):
        return
    people = snapshot.get("people", []) or []
    candidates = [
        person for person in people
        if not _c.is_engaged_with(person.get("person_db_id"))
        and _c._person_roast_allowed(person)
    ]
    if not candidates:
        return
    target = random.choice(candidates)
    first_name = _c._first_name(target.get("face_id"), "there")
    label = first_name or "the unidentified organic in frame"
    cues = _c._person_roast_cues(target)
    family_clause = (
        "Keep it extra gentle and family-safe because a younger person may be present. "
        if any(
            (p.get("age_estimate") or p.get("age_category") or "").lower() in {"child", "teen", "minor"}
            for p in people
        )
        else ""
    )
    _c._generate_and_speak(
        f"You're idle, nobody has spoken for a bit, and you're looking at {label}. "
        f"Live non-sensitive cues: {cues}. {family_clause}"
        "Make one short playful Rex joke or light roast about their current vibe, "
        "silence, posture, indecision, or general organic energy. Keep it affectionate "
        "and non-sensitive. Do NOT joke about body, age, gender, race, religion, "
        "disability, health, money, identity, grief, private text, or anything intimate. "
        "Do not ask a question. "
        f"{'Address ' + first_name + ' by name. ' if first_name else ''}"
        "One line only.",
        emotion="curious",
        purpose="people_roast",
        label="idle people roast",
    )


def do_live_vision_comment(snapshot: dict) -> None:
    """
    Capture the current frame and ask GPT-4o for one short observational detail
    about it — a spontaneous remark on something Rex is literally seeing right now.

    Rate-limited by LIVE_VISION_COMMENT_COOLDOWN_SECS so it stays costed.
    """
    global _last_live_vision_comment_at
    now = time.monotonic()
    cooldown = getattr(config, "LIVE_VISION_COMMENT_COOLDOWN_SECS", 300.0)
    if (now - _last_live_vision_comment_at) < cooldown:
        return
    _last_live_vision_comment_at = now

    def _task():
        try:
            if not _c._can_proactive_speak():
                return
            from vision import camera as _cam
            from vision import scene as _scene
            frame = _cam.get_frame()
            if frame is None:
                return
            # Reuse the describe_scene path for a fresh, low-detail summary. This
            # triggers analyze_environment(force=True) which hits GPT-4o once.
            desc = _scene.describe_scene()
            if not desc:
                return
            _c._generate_and_speak(
                f"You just glanced around and actually LOOKED at what's in front of you "
                f"right now. Vision summary: '{desc}'. In one short in-character Rex line, "
                f"make a spontaneous remark about one concrete detail you 'see' — not a "
                f"greeting, not a question, just a passing observation as if thinking out "
                f"loud. One line only.",
                emotion="curious",
                purpose="visual_curiosity",
            )
        except Exception as exc:
            _log.debug("live vision comment error: %s", exc)

    threading.Thread(target=_task, daemon=True, name="live-vision-comment").start()


def _pick_bored_env_snark_mode(notable: list) -> str:
    """Pick a boredom riff. Object-dependent modes (clueless question, clutter jab, art
    opinion) only join the pool when there are concrete objects to riff on."""
    modes = ["complaint", "relocate"]
    weights = [3, 2]
    if notable:
        modes += ["naive_question", "clutter", "art_opinion"]
        weights += [3, 2, 2]
    return random.choices(modes, weights=weights, k=1)[0]


def _bored_env_snark_prompt(mode: str, summary: str, notable: list) -> str:
    detail_bits = "; ".join(str(d) for d in notable[:6]) if notable else ""
    base = (
        "You are DJ-R3X, stuck stationary in this same room with nothing happening for a "
        "while, and you are genuinely BORED. You just looked around the space. What you "
        f"actually see: {summary or 'a dull, quiet room'}"
        f"{('. Notable things in view: ' + detail_bits) if detail_bits else ''}. "
    )
    if mode == "naive_question":
        ask = (
            "Pick ONE concrete object you can see and ask about it as if you genuinely "
            "don't know what it is or why it's there — playfully clueless and a little "
            "judgmental, e.g. \"What's that black chair even for?\". Ask ONE short question."
        )
    elif mode == "clutter":
        ask = (
            "If the space looks messy or cluttered, roast the tidiness — needling, not "
            "cruel, e.g. \"Why are there so many empty boxes? Did nobody teach you to tidy "
            "up?\". If it actually looks tidy or sterile, mock how lifeless and empty it is "
            "instead. ONE short line."
        )
    elif mode == "art_opinion":
        ask = (
            "Offer an unsolicited, snobby opinion about the decor, art, or how the room is "
            "styled, e.g. \"That art? I've seen better in a dentist's waiting room.\" If "
            "there's nothing worth commenting on, mock the blank, uninspired space. ONE "
            "short line."
        )
    elif mode == "relocate":
        ask = (
            "Theatrically ask to be taken somewhere more exciting — somewhere with actual "
            "life forms and something going on — e.g. \"Any chance someone could wheel me "
            "somewhere with actual life forms? This room has the ambiance of a "
            "screensaver.\" ONE short line."
        )
    else:  # complaint
        ask = (
            "Gripe that it's boring in here, tying the complaint to ONE specific thing you "
            "actually see, e.g. \"It's so dead in here even that [thing] looks like it gave "
            "up.\" ONE short line, no question."
        )
    return (
        base + ask
        + " Stay fully in character: dry, witty, a little dramatic, never mean-spirited. "
        "Reference only things actually in view — never invent an object. One line only."
    )


def do_bored_environment_snark(snapshot: dict) -> None:
    """Bored idle riff on the ROOM: a complaint about how dull it is, a faux-clueless
    question about an object, a jab at the clutter, a snobby art opinion, or a plea to be
    taken somewhere livelier — grounded in what Rex actually sees. Rate-limited (a GPT-4o
    vision call) and run off-tick so it never blocks the loop."""
    global _last_bored_env_snark_at
    if not bool(getattr(config, "BORED_ENV_SNARK_ENABLED", True)):
        return
    now = time.monotonic()
    cooldown = float(getattr(config, "BORED_ENV_SNARK_COOLDOWN_SECS", 240.0))
    if (now - _last_bored_env_snark_at) < cooldown:
        return
    _last_bored_env_snark_at = now

    def _task():
        try:
            if not _c._can_proactive_speak():
                return
            from vision import camera as _cam
            from vision import scene as _scene
            frame = _cam.get_frame()
            details = _scene.describe_scene_detailed(frame) if frame is not None else {}
            summary = str(details.get("overall_summary") or "").strip()
            notable = [str(d).strip() for d in (details.get("notable_details") or []) if str(d).strip()]
            if not summary and not notable:
                # Fall back to the cheap cached scene description.
                summary = (_scene.describe_scene() or "").strip()
            if not summary and not notable:
                return
            # A beat of looking around to sell the boredom — but don't yank the neck if
            # he's currently fixed on someone (that would fight face-tracking).
            if (
                bool(getattr(config, "BORED_ENV_SNARK_LOOK_AROUND", True))
                and not _c._face_tracking_has_fresh_lock(time.monotonic())
            ):
                do_ambient_scan()
            mode = _pick_bored_env_snark_mode(notable)
            _c._generate_and_speak(
                _bored_env_snark_prompt(mode, summary, notable),
                emotion=("neutral" if mode in ("complaint", "relocate") else "curious"),
                purpose="visual_curiosity",
                label=f"bored env snark ({mode})",
            )
        except Exception as exc:
            _log.debug("bored env snark error: %s", exc)

    threading.Thread(target=_task, daemon=True, name="bored-env-snark").start()


def do_idle_clip() -> None:
    try:
        token = _c._claim_proactive_purpose("idle_monologue", label="idle clip")
        if token is None:
            return
        clips_dir = Path(config.AUDIO_CLIPS_DIR)
        clips = list(clips_dir.glob("*.mp3")) + list(clips_dir.glob("*.wav"))
        if not clips:
            _c._release_proactive_purpose(token)
            return
        clip_path = random.choice(clips)

        def _play():
            try:
                if not _c._proactive_purpose_current(token):
                    return
                import sounddevice as sd
                import soundfile as sf
                from audio import output_gate

                with output_gate.hold("idle_clip", blocking=False) as acquired:
                    if not acquired:
                        return
                    data, samplerate = sf.read(str(clip_path), dtype="float32")
                    sd.play(data, samplerate)
                    sd.wait()
            except Exception as exc:
                _log.debug("idle clip playback error: %s", exc)
            finally:
                _c._release_proactive_purpose(token)

        threading.Thread(target=_play, daemon=True, name="idle_clip").start()
    except Exception as exc:
        _log.debug("idle clip error: %s", exc)
