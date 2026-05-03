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
from typing import Any, Optional

import config

_log = logging.getLogger(__name__)

_ids = count(1)


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
    "visual_curiosity": 55,
    "small_talk": 45,
    "world_reaction": 40,
    "ambient_observation": 30,
    "appearance_riff": 28,
    "idle_monologue": 15,
    "direct_speech": 20,
}

_LOW_PRIORITY_RAPID_EXCHANGE_CUTOFF = 55
_ACTIVE_CONVERSATION_LOW_PRIORITY = {
    "small_talk",
    "ambient_observation",
    "appearance_riff",
    "idle_monologue",
}
_ACTIVE_CONVERSATION_ALLOWED_SOURCES = {
    "_step_group_lull",
    "_step_group_turn_taking",
    "_step_visual_curiosity",
    "_step_emotional_checkin",
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

    def active(self) -> bool:
        return self.shadow_mode or self.log_candidates

    def start_cycle(self, *, profile: Any = None, snapshot: Optional[dict] = None) -> None:
        if not self.active():
            return
        self._local.cycle = {
            "id": f"cycle-{next(_ids)}",
            "started_at": time.monotonic(),
            "profile": profile,
            "snapshot": snapshot or {},
            "candidates": [],
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
