"""
action_governor.py — deterministic candidate registry for proactive behavior.

This is intentionally not another LLM planner. Consciousness can register the
things it wants to say, and the governor scores those candidate moves with
plain rules. In shadow mode the current behavior still runs, but logs show what
the governor would have chosen.
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


@dataclass
class CandidateMove:
    """A possible proactive move Rex could make."""

    source: str
    purpose: str
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
                self._log_candidate(scored, cycle_id="standalone")
                self._log_decision(
                    Decision(
                        action="speak" if not scored.rejected else "wait",
                        selected=scored if not scored.rejected else None,
                        scored=[scored],
                        reason="standalone candidate" if not scored.rejected else "candidate rejected",
                    ),
                    cycle_id="standalone",
                )
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
            if getattr(profile, "force_family_safe", False):
                candidate.metadata.setdefault("family_safe", True)

        if candidate.metadata.get("waiting_for_response"):
            reasons.append("waiting_for_human_response")
        if candidate.metadata.get("can_proactive_speak") is False:
            reasons.append("can_proactive_speak_false")
        if candidate.metadata.get("can_speak") is False:
            reasons.append("can_speak_false")

        recent_rex_gap = candidate.metadata.get("seconds_since_rex_spoke")
        min_gap = float(getattr(config, "CONSCIOUSNESS_PROACTIVE_MIN_GAP_SECS", 0.0) or 0.0)
        if isinstance(recent_rex_gap, (int, float)) and min_gap and recent_rex_gap < min_gap:
            score -= 20
            reasons.append(f"recent_rex_speech_{recent_rex_gap:.1f}s")

        rejected = bool(
            candidate.outcome == "dropped"
            or "user_mid_sentence" in reasons
            or "interaction_busy" in reasons
            or "situation_suppresses_proactive" in reasons
            or "waiting_for_human_response" in reasons
            or "can_proactive_speak_false" in reasons
            or "can_speak_false" in reasons
        )
        min_score = int(getattr(config, "ACTION_GOVERNOR_MIN_SCORE", 20))
        if score < min_score:
            rejected = True
            reasons.append(f"below_min_score_{min_score}")

        if not reasons:
            reasons.append("eligible")
        return ScoredCandidate(candidate=candidate, score=score, rejected=rejected, reasons=reasons)

    @staticmethod
    def _decide(scored: list[ScoredCandidate]) -> Decision:
        eligible = [item for item in scored if not item.rejected]
        if not eligible:
            return Decision(
                action="wait",
                selected=None,
                scored=scored,
                reason="no eligible candidates",
            )
        selected = max(
            eligible,
            key=lambda item: (
                item.score,
                -item.candidate.created_at,
            ),
        )
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
        payload = c.suggested_text or c.prompt
        _log.info(
            "[action_governor] %s candidate=%s purpose=%s source=%s label=%r "
            "score=%s rejected=%s outcome=%s reasons=%s llm=%s target=%s text=%r",
            cycle_id,
            c.candidate_id,
            c.purpose,
            c.source,
            c.label,
            scored.score,
            scored.rejected,
            c.outcome,
            reasons,
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
