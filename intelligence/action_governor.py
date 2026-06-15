"""
action_governor.py — deterministic candidate registry for proactive behavior.

This is intentionally not another LLM planner. Consciousness can register the
things it wants to say, and the governor scores those candidate moves with
plain rules. In shadow mode the current behavior still runs, but logs show what
the governor would have chosen.

This module is deliberately proactive-only. User-turn routing belongs to
action_router.py and planned output execution belongs to performance_output.py.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from itertools import count
from typing import Any, Callable, Optional

import config

_log = logging.getLogger(__name__)

_ids = count(1)

# Cross-thread proactive intake (Increment 2): mechanisms on threads OTHER than the
# consciousness loop (e.g. interaction's idle banter / memory follow-ups) can't join
# the thread-local cycle, so they submit candidates here instead of speaking. The next
# consciousness tick drains them into its cycle, so ALL proactive speech is arbitrated
# by ONE decider. Lock-guarded; stale entries (older than the TTL) are dropped so a
# paused consciousness loop can't replay an old idle line.
_external_lock = threading.Lock()
_external_candidates: list = []
_EXTERNAL_CANDIDATE_TTL_SECS = 2.0

# Cross-cycle de-dup: topic_key -> monotonic time it was last SELECTED to speak.
# _decide's seen_topics only collapses duplicates WITHIN a single tick; without
# this, a flickering world cue (the crowd label bouncing pair<->alone, an animal
# false-positive, a smile detector) re-selects the SAME proactive line on
# consecutive ticks — the live "now it's just us" line spoken twice in 7s, which a
# bystander read as "your code glitched". Self-pruning via the cooldown window.
# idle_monologue is EXCLUDED: it generates a fresh line each time and paces itself
# via IDLE_BANTER_COOLDOWN_SECS, so a static-key cooldown would wrongly throttle it.
_recent_selected_lock = threading.Lock()
_recent_selected: dict[str, float] = {}
_REPEAT_COOLDOWN_EXCLUDED_PURPOSES = {"idle_monologue"}


def _repeat_cooldown_secs() -> float:
    return float(getattr(config, "ACTION_GOVERNOR_REPEAT_COOLDOWN_SECS", 45.0) or 0.0)


def _topic_recently_selected(topic_key: str, purpose: str) -> bool:
    cooldown = _repeat_cooldown_secs()
    if cooldown <= 0 or purpose in _REPEAT_COOLDOWN_EXCLUDED_PURPOSES:
        return False
    now = time.monotonic()
    with _recent_selected_lock:
        ts = _recent_selected.get(topic_key)
        return ts is not None and (now - ts) < cooldown


def _note_topic_selected(topic_key: str) -> None:
    now = time.monotonic()
    with _recent_selected_lock:
        _recent_selected[topic_key] = now
        cutoff = now - max(_repeat_cooldown_secs() * 2.0, 1.0)
        for stale in [k for k, t in _recent_selected.items() if t < cutoff]:
            _recent_selected.pop(stale, None)


_PURPOSE_PRIORITIES: dict[str, int] = {
    "emotional_checkin": 100,
    "relationship_inquiry": 95,
    "identity_prompt": 92,
    "presence_reaction": 80,
    "overheard_chime_in": 75,
    "third_party_awareness": 72,
    "reengagement": 70,
    "group_turn_invite": 68,
    "personal_space": 67,
    "memory_followup": 65,
    "celebration_checkin": 64,
    "startup_empty_room": 60,
    # Banked-callback humor in a mid-conversation lull: above visual_curiosity
    # (a callback referencing THEM beats commenting on the room), below every
    # sincerity flow (celebration 64 / memory_followup 65 / checkin 100).
    "lull_callback": 58,
    "visual_curiosity": 55,
    "small_talk": 45,
    "world.animal_arrival": 85,
    "weather.proactive_comment": 42,
    "world_reaction": 40,
    "ambient_observation": 30,
    "appearance_riff": 28,
    # 22 (not the old 15) so idle monologue / empty-room lines clear
    # ACTION_GOVERNOR_MIN_SCORE (20). At 15 they were always rejected
    # below_min_score, i.e. unreachable once ACTION_GOVERNOR_ENFORCE was turned on
    # — they could never fire even on an otherwise-empty idle tick. Still low
    # priority: any presence/identity/check-in candidate outranks them.
    "idle_monologue": 22,
    "direct_speech": 20,
}

_LOW_PRIORITY_RAPID_EXCHANGE_CUTOFF = 55
_ACTIVE_CONVERSATION_LOW_PRIORITY = {
    "small_talk",
    "weather.proactive_comment",
    "startup_empty_room",
    "ambient_observation",
    "appearance_riff",
    "idle_monologue",
}
_ACTIVE_CONVERSATION_ALLOWED_SOURCES = {
    "_step_group_lull",
    "_step_group_turn_taking",
    "_step_visual_curiosity",
    "_step_emotional_checkin",
    # Re-engaging a present person who went quiet mid-conversation IS the
    # active-conversation job — so idle banter must NOT take the -35
    # conversation_active penalty (which dropped its priority-50 candidate to 15,
    # below ACTION_GOVERNOR_MIN_SCORE, every cycle). It still loses to any
    # presence/identity/check-in candidate (80-100) and stays below
    # visual_curiosity (55), so it never talks over higher-value speech.
    "interaction._maybe_idle_banter",
}
PROACTIVE_CANDIDATE_KIND = "proactive"


@dataclass
class CandidateMove:
    """A possible proactive move Rex could make."""

    source: str
    purpose: str
    kind: str = PROACTIVE_CANDIDATE_KIND
    label: str = ""
    prompt: str = ""
    suggested_text: str = ""
    emotion: str = "neutral"
    priority: Optional[int] = None
    target_person_id: Optional[int] = None
    target_label: str = ""
    requires_llm: bool = True
    wait_secs: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # When ENFORCING, the cycle resolver invokes this on the winning candidate
    # (and on no one else) — the deferred "generate + speak" work. Ignored in
    # shadow mode, where each mechanism still speaks for itself.
    speak_fn: Optional[Callable[[], None]] = None
    candidate_id: str = field(default_factory=lambda: f"cg-{next(_ids)}")
    created_at: float = field(default_factory=time.monotonic)
    outcome: str = "observed"
    outcome_reason: str = ""


@dataclass
class ScoredCandidate:
    candidate: CandidateMove
    score: int
    rejected: bool
    reasons: list[str]
    selected: bool = False
    skip_reasons: list[str] = field(default_factory=list)


@dataclass
class Decision:
    action: str
    selected: Optional[ScoredCandidate]
    scored: list[ScoredCandidate]
    reason: str


class ActionGovernor:
    """
    Collect and score proactive speech candidates for a single loop tick.

    A thread-local cycle keeps background LLM/TTS threads from accidentally
    adding late results to the active consciousness loop's candidate set.
    """

    def __init__(self) -> None:
        self._local = threading.local()

    @property
    def shadow_mode(self) -> bool:
        return bool(getattr(config, "ACTION_GOVERNOR_SHADOW_MODE", True))

    @property
    def log_candidates(self) -> bool:
        return bool(getattr(config, "ACTION_GOVERNOR_LOG_CANDIDATES", True))

    @property
    def enforcing(self) -> bool:
        """When True the governor is the single decider: candidates are collected
        for the tick and ONLY the winner's `speak_fn` is invoked (losers are
        suppressed). When False (default) it only observes/logs and each mechanism
        speaks for itself (the legacy scattered behavior)."""
        return bool(getattr(config, "ACTION_GOVERNOR_ENFORCE", False))

    def active(self) -> bool:
        return self.shadow_mode or self.log_candidates or self.enforcing

    def has_active_cycle(self) -> bool:
        """True when THIS thread is inside a start_cycle/finish_cycle window (the
        consciousness tick). The cycle is thread-local, so a candidate submitted via
        observe() from any OTHER thread (a spawned worker) would only standalone-log
        and never run its speak_fn — such callers must use submit_external instead."""
        return getattr(self._local, "cycle", None) is not None

    def submit_external(self, candidate: "CandidateMove") -> str:
        """Submit a proactive candidate from a NON-consciousness thread (e.g.
        interaction's idle banter). It is picked up and arbitrated by the next
        consciousness tick (within ~1 cycle). No-op when not enforcing — callers
        fall back to their legacy inline speaking."""
        if not self.enforcing:
            return candidate.candidate_id
        with _external_lock:
            _external_candidates.append(candidate)
        return candidate.candidate_id

    def _drain_external(self) -> list:
        now = time.monotonic()
        with _external_lock:
            fresh = [
                c for c in _external_candidates
                if (now - c.created_at) <= _EXTERNAL_CANDIDATE_TTL_SECS
            ]
            _external_candidates.clear()
        return fresh

    def start_cycle(self, *, profile: Any = None, snapshot: Optional[dict] = None) -> None:
        if not self.active():
            return
        # Pull in any cross-thread candidates so they compete in this tick.
        external = self._drain_external() if self.enforcing else []
        self._local.cycle = {
            "id": f"cycle-{next(_ids)}",
            "started_at": time.monotonic(),
            "profile": profile,
            "snapshot": snapshot or {},
            "candidates": external,
        }

    def observe(self, candidate: CandidateMove) -> str:
        if not self.active():
            return candidate.candidate_id
        cycle = getattr(self._local, "cycle", None)
        if cycle is None:
            scored = self._score(candidate, profile=None)
            if self.log_candidates:
                decision = self._decide([scored])
                self._log_candidate(scored, cycle_id="standalone")
                self._log_decision(decision, cycle_id="standalone")
            return candidate.candidate_id
        cycle["candidates"].append(candidate)
        return candidate.candidate_id

    def mark_outcome(self, candidate_id: Optional[str], outcome: str, reason: str = "") -> None:
        if not candidate_id or not self.active():
            return
        cycle = getattr(self._local, "cycle", None)
        if cycle is None:
            return
        for candidate in cycle.get("candidates", []):
            if candidate.candidate_id == candidate_id:
                candidate.outcome = outcome
                candidate.outcome_reason = reason
                return

    def finish_cycle(self) -> Optional[Decision]:
        cycle = getattr(self._local, "cycle", None)
        if cycle is None:
            return None
        try:
            candidates: list[CandidateMove] = list(cycle.get("candidates", []))
            if not candidates:
                if bool(getattr(config, "ACTION_GOVERNOR_LOG_EMPTY_CYCLES", False)):
                    _log.info("[action_governor] %s no candidates", cycle["id"])
                return None
            scored = [self._score(c, profile=cycle.get("profile")) for c in candidates]
            decision = self._decide(scored)
            # Record the winner so the same proactive cue can't be re-selected on a
            # later flickering tick (cross-cycle de-dup). Only the real cycle path
            # records; standalone observe() logging does not.
            if decision.action == "speak" and decision.selected is not None:
                _note_topic_selected(
                    self._candidate_topic_key(decision.selected.candidate)
                )
            if self.log_candidates:
                for item in scored:
                    self._log_candidate(item, cycle_id=cycle["id"])
                self._log_decision(decision, cycle_id=cycle["id"])
            return decision
        finally:
            self._local.cycle = None

    def _score(self, candidate: CandidateMove, *, profile: Any = None) -> ScoredCandidate:
        priority = candidate.priority
        if priority is None:
            priority = _PURPOSE_PRIORITIES.get(candidate.purpose, 20)
        score = int(priority)
        reasons: list[str] = []

        if candidate.kind != PROACTIVE_CANDIDATE_KIND:
            reasons.append("non_proactive_candidate")

        if candidate.outcome == "dropped":
            reasons.append(candidate.outcome_reason or "dropped_by_current_behavior")

        if profile is not None:
            if getattr(profile, "user_mid_sentence", False):
                reasons.append("user_mid_sentence")
            if getattr(profile, "interaction_busy", False):
                reasons.append("interaction_busy")
            if getattr(profile, "suppress_proactive", False):
                reasons.append("situation_suppresses_proactive")
            if (
                getattr(profile, "rapid_exchange", False)
                and priority < _LOW_PRIORITY_RAPID_EXCHANGE_CUTOFF
            ):
                score -= 25
                reasons.append("rapid_exchange_low_priority")
            if (
                getattr(profile, "conversation_active", False)
                and candidate.purpose in _ACTIVE_CONVERSATION_LOW_PRIORITY
                and candidate.source not in _ACTIVE_CONVERSATION_ALLOWED_SOURCES
            ):
                score -= 35
                reasons.append("conversation_active_low_priority")
            if getattr(profile, "force_family_safe", False):
                if (
                    candidate.metadata.get("family_safe") is False
                    or candidate.metadata.get("adult_only")
                    or candidate.metadata.get("unsafe_for_children")
                ):
                    reasons.append("child_present_family_safe_block")
                else:
                    candidate.metadata.setdefault("family_safe", True)

        if candidate.metadata.get("waiting_for_response"):
            reasons.append("waiting_for_human_response")
        if candidate.metadata.get("proactive_speech_pending"):
            reasons.append("proactive_speech_pending")
        if candidate.metadata.get("game_interruptions_suppressed"):
            reasons.append("game_active_suppresses_proactive")
        if candidate.metadata.get("active_state_proactive_blocked"):
            reasons.append("active_state_proactive_blocked")
        if candidate.metadata.get("speech_queue_speaking"):
            reasons.append("speech_queue_speaking")
        if candidate.metadata.get("output_gate_busy"):
            reasons.append("output_gate_busy")
        if candidate.metadata.get("output_gate_status_error"):
            reasons.append("output_gate_status_error")
        if candidate.metadata.get("can_proactive_speak") is False:
            reasons.append("can_proactive_speak_false")
        if candidate.metadata.get("can_speak") is False:
            reasons.append("can_speak_false")
        # Gates relocated from the conversation_agenda claim (which ENFORCE bypasses)
        # so the governor — the single decider — still honors end-of-thread grace and
        # the question budget for proactive purposes.
        if candidate.metadata.get("grace_suppressed"):
            reasons.append("end_thread_grace_suppressed")
        if candidate.metadata.get("question_budget_exhausted"):
            reasons.append("question_budget_exhausted")
        if candidate.metadata.get("cooldown_active"):
            cooldown_reason = str(candidate.metadata.get("cooldown_reason") or "cooldown_active")
            remaining = candidate.metadata.get("cooldown_remaining_secs")
            if isinstance(remaining, (int, float)) and remaining > 0:
                reasons.append(f"{cooldown_reason}_{remaining:.1f}s")
            else:
                reasons.append(cooldown_reason)

        recent_rex_gap = candidate.metadata.get("seconds_since_rex_spoke")
        min_gap = float(getattr(config, "CONSCIOUSNESS_PROACTIVE_MIN_GAP_SECS", 0.0) or 0.0)
        if (
            isinstance(recent_rex_gap, (int, float))
            and min_gap
            and recent_rex_gap < min_gap
            and not candidate.metadata.get("cooldown_active")
        ):
            score -= 20
            remaining = max(0.0, min_gap - recent_rex_gap)
            reasons.append(f"proactive_cooldown_{remaining:.1f}s")

        rejected = bool(
            candidate.outcome == "dropped"
            or "non_proactive_candidate" in reasons
            or "user_mid_sentence" in reasons
            or "interaction_busy" in reasons
            or "situation_suppresses_proactive" in reasons
            or "child_present_family_safe_block" in reasons
            or "waiting_for_human_response" in reasons
            or "proactive_speech_pending" in reasons
            or "game_active_suppresses_proactive" in reasons
            or "active_state_proactive_blocked" in reasons
            or "speech_queue_speaking" in reasons
            or "output_gate_busy" in reasons
            or "output_gate_status_error" in reasons
            or "can_proactive_speak_false" in reasons
            or "can_speak_false" in reasons
            or "end_thread_grace_suppressed" in reasons
            or "question_budget_exhausted" in reasons
            or candidate.metadata.get("cooldown_active")
        )
        min_score = int(getattr(config, "ACTION_GOVERNOR_MIN_SCORE", 20))
        if score < min_score:
            rejected = True
            reasons.append(f"below_min_score_{min_score}")

        if not reasons:
            reasons.append("eligible")
        return ScoredCandidate(candidate=candidate, score=score, rejected=rejected, reasons=reasons)

    @staticmethod
    def _selection_key(item: ScoredCandidate) -> tuple[int, float]:
        return (
            item.score,
            -item.candidate.created_at,
        )

    @staticmethod
    def _candidate_topic_key(candidate: CandidateMove) -> str:
        explicit = (
            candidate.metadata.get("topic_key")
            or candidate.metadata.get("dedupe_key")
            or candidate.metadata.get("topic")
        )
        if explicit:
            return str(explicit).strip().lower()
        target = candidate.target_person_id
        if target is None:
            target = candidate.target_label.strip().lower()
        label = (candidate.label or candidate.purpose or "").strip().lower()
        return f"{candidate.purpose}:{target or ''}:{label}"

    @classmethod
    def _decide(cls, scored: list[ScoredCandidate]) -> Decision:
        for item in scored:
            item.selected = False
            item.skip_reasons.clear()

        eligible_by_rank = sorted(
            [item for item in scored if not item.rejected],
            key=cls._selection_key,
            reverse=True,
        )
        seen_topics: dict[str, ScoredCandidate] = {}
        for item in eligible_by_rank:
            topic_key = cls._candidate_topic_key(item.candidate)
            if topic_key in seen_topics:
                item.rejected = True
                item.reasons.append("duplicate_topic")
                item.skip_reasons.append("duplicate_topic")
                continue
            if _topic_recently_selected(topic_key, item.candidate.purpose):
                item.rejected = True
                item.reasons.append("topic_repeat_cooldown")
                item.skip_reasons.append("topic_repeat_cooldown")
                continue
            seen_topics[topic_key] = item

        eligible = [item for item in scored if not item.rejected]
        if not eligible:
            for item in scored:
                if not item.skip_reasons:
                    item.skip_reasons.extend(item.reasons)
            return Decision(
                action="wait",
                selected=None,
                scored=scored,
                reason="no eligible candidates",
            )
        selected = max(eligible, key=cls._selection_key)
        selected.selected = True
        for item in scored:
            if item is selected:
                continue
            if item.rejected:
                if not item.skip_reasons:
                    item.skip_reasons.extend(item.reasons)
                continue
            if selected.score > item.score:
                reason = f"lower_priority_than_selected:{selected.candidate.purpose}"
            else:
                reason = f"tie_lost_to_selected:{selected.candidate.purpose}"
            item.skip_reasons.append(reason)
            item.reasons.append(reason)
        return Decision(
            action="speak",
            selected=selected,
            scored=scored,
            reason="highest eligible score",
        )

    @staticmethod
    def _clip(text: str, limit: int = 180) -> str:
        compact = " ".join((text or "").split())
        if len(compact) <= limit:
            return compact
        return compact[: max(0, limit - 3)] + "..."

    def _log_candidate(self, scored: ScoredCandidate, *, cycle_id: str) -> None:
        c = scored.candidate
        reasons = ",".join(scored.reasons)
        skip_reasons = ",".join(scored.skip_reasons)
        payload = c.suggested_text or c.prompt
        _log.info(
            "[action_governor] %s candidate=%s kind=%s purpose=%s source=%s label=%r "
            "score=%s selected=%s skipped=%s rejected=%s outcome=%s reasons=%s "
            "skip_reasons=%s llm=%s target=%s text=%r",
            cycle_id,
            c.candidate_id,
            c.kind,
            c.purpose,
            c.source,
            c.label,
            scored.score,
            scored.selected,
            not scored.selected,
            scored.rejected,
            c.outcome,
            reasons,
            skip_reasons,
            c.requires_llm,
            c.target_person_id or c.target_label or "",
            self._clip(payload),
        )

    def _log_decision(self, decision: Decision, *, cycle_id: str) -> None:
        if decision.selected is None:
            _log.info(
                "[action_governor] %s shadow_decision=WAIT reason=%s candidates=%d",
                cycle_id,
                decision.reason,
                len(decision.scored),
            )
            return
        c = decision.selected.candidate
        _log.info(
            "[action_governor] %s shadow_decision=SPEAK candidate=%s purpose=%s source=%s "
            "score=%s reason=%s",
            cycle_id,
            c.candidate_id,
            c.purpose,
            c.source,
            decision.selected.score,
            decision.reason,
        )


governor = ActionGovernor()
