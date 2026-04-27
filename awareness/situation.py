"""
awareness/situation.py — Situation Assessment for DJ-R3X.

SituationAssessor is the judgment layer between "I could say something" and
"I will say something."  It is evaluated once per consciousness loop tick and
before every speech decision.

Public API:
    assessor.evaluate()          → SituationProfile
    assessor.set_vad_active(bool)  — called by interaction.py VAD loop
    assessor.set_rex_speaking(bool)— called by audio/speech_queue.py worker
    assessor.set_interaction_busy(bool)
"""

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
import state as state_module
from state import State
from world_state import world_state


_SOCIAL_MODE_MAP: dict[str, str] = {
    "alone":       "one_on_one",
    "pair":        "one_on_one",
    "small_group": "small_group",
    "group":       "crowd",
    "crowd":       "performance",
}


@dataclass
class SituationProfile:
    conversation_active: bool       # ACTIVE state with recent speech
    user_mid_sentence: bool         # VAD currently detecting speech
    rapid_exchange: bool            # 3+ speech turns in last 30 s
    child_present: bool             # any child/teen in world_state.people
    apparent_departure: bool        # face gone AND VAD silent > DEPARTURE_AUDIO_SILENCE_SECS
    likely_still_present: bool      # face gone BUT user_mid_sentence (moved, not left)
    social_mode: str                # one_on_one / small_group / crowd / performance
    suppress_proactive: bool        # don't fire unsolicited speech right now
    suppress_system_comments: bool  # don't mention CPU/uptime mid-conversation
    force_family_safe: bool         # child present — override all adult content
    being_discussed: bool           # Rex was referenced ABOUT-not-TO recently
    discussion_sentiment: str       # "positive" / "neutral" / "negative" of the last mention
    interaction_busy: bool          # interaction loop is recording/processing/responding


class SituationAssessor:
    """
    Thread-safe singleton.  Reads from WorldState plus externally-pushed VAD and
    speech-queue state, and derives a SituationProfile on each evaluate() call.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        now = time.monotonic()

        # VAD / speech-turn state
        self._vad_active: bool = False
        self._vad_last_active_at: float = 0.0   # monotonic of last speech onset
        self._vad_became_silent_at: float = now  # monotonic of last speech end (init = now)

        # Speech turn history — monotonic timestamps of VAD rising edges (speech onset)
        self._speech_turn_times: list[float] = []

        # Rex's own speech state
        self._rex_speaking: bool = False
        self._rex_stopped_at: float = 0.0       # monotonic when Rex last finished speaking

        # Conversation boundary tracking (for suppress_proactive "ended < 5s ago")
        self._last_seen_state: State = State.IDLE
        self._conversation_ended_at: float = 0.0

        # Interaction pipeline state. True from user speech onset until Rex has
        # finished handling that turn, including transcription/LLM/TTS latency.
        self._interaction_busy: bool = False

    # ── External setters ───────────────────────────────────────────────────────

    def set_vad_active(self, active: bool) -> None:
        """
        Called by the VAD loop in interaction.py on every audio chunk.
        Tracks rising edges (new speech turn) and falling edges (speech end).
        """
        with self._lock:
            was_active = self._vad_active
            self._vad_active = active
            now = time.monotonic()
            if active and not was_active:
                # Rising edge — new speech turn started
                self._vad_last_active_at = now
                self._speech_turn_times.append(now)
            elif not active and was_active:
                # Falling edge — speech ended
                self._vad_became_silent_at = now

    def set_rex_speaking(self, speaking: bool) -> None:
        """Called by audio/speech_queue.py worker at start/end of every playback."""
        with self._lock:
            if not speaking and self._rex_speaking:
                self._rex_stopped_at = time.monotonic()
            self._rex_speaking = speaking

    def set_interaction_busy(self, busy: bool) -> None:
        """Called by interaction.py while a user turn is being handled."""
        with self._lock:
            self._interaction_busy = bool(busy)

    def is_interaction_busy(self) -> bool:
        """Return True while the interaction loop is handling a user turn."""
        with self._lock:
            return self._interaction_busy

    def recent_speech_turn_count(self, window_secs: float = 30.0) -> int:
        """Return the number of user speech onsets within the recent window."""
        cutoff = time.monotonic() - max(0.0, float(window_secs))
        with self._lock:
            self._speech_turn_times = [
                t for t in self._speech_turn_times if t >= cutoff
            ]
            return len(self._speech_turn_times)

    # ── Main evaluation ────────────────────────────────────────────────────────

    def evaluate(self) -> SituationProfile:
        """Derive and return the current SituationProfile.  Thread-safe."""
        now = time.monotonic()

        with self._lock:
            vad_active = self._vad_active
            vad_last_active = self._vad_last_active_at
            vad_became_silent = self._vad_became_silent_at
            rex_speaking = self._rex_speaking
            rex_stopped_at = self._rex_stopped_at
            interaction_busy = self._interaction_busy

            # Prune stale turn history (>30 s old)
            cutoff = now - 30.0
            self._speech_turn_times = [t for t in self._speech_turn_times if t >= cutoff]
            turn_count = len(self._speech_turn_times)

        # Detect ACTIVE → non-ACTIVE transition to track "conversation ended" time.
        # (Runs outside the lock because state_module has its own lock.)
        current_state = state_module.get_state()
        if self._last_seen_state == State.ACTIVE and current_state != State.ACTIVE:
            self._conversation_ended_at = now
        self._last_seen_state = current_state

        # ── Derived fields ─────────────────────────────────────────────────────

        # conversation_active: in ACTIVE state AND user spoke within the window
        conv_window = getattr(config, "CONVERSATION_ACTIVE_WINDOW_SECS", 30)
        conversation_active = (
            current_state == State.ACTIVE
            and vad_last_active > 0.0
            and (now - vad_last_active) < conv_window
        )

        # user_mid_sentence: VAD currently detecting speech
        user_mid_sentence = vad_active

        # rapid_exchange: 3+ speech turns in the last 30 s
        rapid_exchange = turn_count >= 3

        # child_present, face_gone
        child_present = False
        face_gone = True
        try:
            people = world_state.get("people")
            face_gone = not bool(people)
            child_present = any(
                (p.get("age_category") or p.get("age_estimate")) in ("child", "teen")
                for p in people
            )
        except Exception:
            pass

        # apparent_departure: face gone AND VAD has been silent for the required window
        departure_silence = getattr(config, "DEPARTURE_AUDIO_SILENCE_SECS", 3.0)
        if face_gone and not vad_active and vad_became_silent > 0.0:
            apparent_departure = (now - vad_became_silent) >= departure_silence
        else:
            apparent_departure = False

        # likely_still_present: face gone but user is still talking (moved, not left)
        likely_still_present = face_gone and user_mid_sentence

        # social_mode: prefer pose/proxemic crowd interaction mode, fall back
        # to simple count labels from the vision scene counter.
        count_label = "alone"
        interaction_mode = None
        try:
            crowd = world_state.get("crowd")
            interaction_mode = crowd.get("interaction_mode")
            count_label = crowd.get("count_label") or "alone"
        except Exception:
            pass
        social_mode = interaction_mode or _SOCIAL_MODE_MAP.get(count_label, "one_on_one")

        # suppress_proactive: any condition that means Rex should stay quiet right now
        convo_just_ended = (
            self._conversation_ended_at > 0.0
            and (now - self._conversation_ended_at) < 5.0
        )
        rex_just_spoke = (
            not rex_speaking
            and rex_stopped_at > 0.0
            and (now - rex_stopped_at) < 2.0
        )
        state_suppresses = current_state in (State.QUIET, State.SHUTDOWN)
        suppress_proactive = (
            user_mid_sentence
            or interaction_busy
            or convo_just_ended
            or rex_just_spoke
            or state_suppresses
        )

        # suppress_system_comments: conversation active OR speech within silence window
        sys_silence = getattr(config, "SYSTEM_COMMENT_SILENCE_SECS", 60)
        speech_recently = vad_last_active > 0.0 and (now - vad_last_active) < sys_silence
        suppress_system_comments = conversation_active or speech_recently

        # force_family_safe: any child or teen present
        force_family_safe = child_present

        # being_discussed: Rex was referenced ABOUT-not-TO within the active window.
        being_discussed = False
        discussion_sentiment = "neutral"
        try:
            social = world_state.get("social") or {}
            bd = social.get("being_discussed") or {}
            last_at = bd.get("last_mention_at")
            window = float(getattr(config, "BEING_DISCUSSED_ACTIVE_WINDOW_SECS", 30.0))
            if last_at is not None:
                if (time.time() - float(last_at)) <= window:
                    being_discussed = True
                    discussion_sentiment = bd.get("sentiment") or "neutral"
        except Exception:
            pass

        return SituationProfile(
            conversation_active=conversation_active,
            user_mid_sentence=user_mid_sentence,
            rapid_exchange=rapid_exchange,
            child_present=child_present,
            apparent_departure=apparent_departure,
            likely_still_present=likely_still_present,
            social_mode=social_mode,
            suppress_proactive=suppress_proactive,
            suppress_system_comments=suppress_system_comments,
            force_family_safe=force_family_safe,
            being_discussed=being_discussed,
            discussion_sentiment=discussion_sentiment,
            interaction_busy=interaction_busy,
        )


# Module-level singleton — import this everywhere
assessor = SituationAssessor()
