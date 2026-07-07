"""
Presence-gated proactive cadence (owner direction 2026-07-06): the gap between
chatter-class proactive lines scales with presence — base while conversation
flows, longer when someone is present but quiet, near-silent in an empty room.
Enforced CENTRALLY in the governor so submit_external candidates (the historical
idle-banter leak: no cooldown metadata, no cadence gate at all) are covered.
"""

import time
import unittest
from types import SimpleNamespace
from unittest import mock

import config
from intelligence import presence_cadence
from intelligence.action_governor import ActionGovernor, CandidateMove


def _profile(**over):
    base = dict(conversation_active=False, user_mid_sentence=False,
                interaction_busy=False, suppress_proactive=False,
                rapid_exchange=False, child_present=False,
                likely_still_present=False)
    base.update(over)
    return SimpleNamespace(**base)


class EffectiveGapTest(unittest.TestCase):
    def _gap(self, *, visible, profile=None):
        with mock.patch.object(presence_cadence, "_visible_person_present",
                               return_value=visible):
            return presence_cadence.effective_min_gap_secs(profile)

    def test_engaged_uses_base(self):
        self.assertEqual(self._gap(visible=True,
                                   profile=_profile(conversation_active=True)),
                         float(config.CONSCIOUSNESS_PROACTIVE_MIN_GAP_SECS))

    def test_present_but_quiet_uses_idle_gap(self):
        self.assertEqual(self._gap(visible=True, profile=_profile()),
                         float(config.PROACTIVE_GAP_PRESENT_IDLE_SECS))

    def test_empty_room_uses_long_gap(self):
        self.assertEqual(self._gap(visible=False, profile=_profile()),
                         float(config.PROACTIVE_GAP_EMPTY_ROOM_SECS))

    def test_briefly_lost_face_counts_as_present(self):
        self.assertEqual(
            self._gap(visible=False, profile=_profile(likely_still_present=True)),
            float(config.PROACTIVE_GAP_PRESENT_IDLE_SECS))

    def test_kill_switch_restores_base(self):
        with mock.patch.object(config, "PROACTIVE_CADENCE_CLAMP_ENABLED", False, create=True):
            self.assertEqual(self._gap(visible=False, profile=_profile()),
                             float(config.CONSCIOUSNESS_PROACTIVE_MIN_GAP_SECS))


class GovernorClampTest(unittest.TestCase):
    """The central hard gate: a chatter-class candidate inside the presence gap is
    REJECTED even with no cooldown metadata (the submit_external leak)."""

    def _score(self, purpose, *, seconds_since_spoke, profile=None, metadata=None):
        gov = ActionGovernor()
        candidate = CandidateMove(source="test", purpose=purpose, priority=50,
                                  label="t", metadata=metadata or {})
        from intelligence import consciousness as c
        with mock.patch.object(c, "_last_proactive_speech_at",
                               time.monotonic() - seconds_since_spoke):
            return gov._score(candidate, profile=profile or _profile())

    def test_chatter_inside_gap_is_rejected_without_metadata(self):
        # room_change: clamped but NOT lean-suppressed (idle_monologue never reaches
        # the clamp under LEAN_BRAIN_ENABLED — the lean gate rejects it first).
        # Present-but-quiet tier (45s); Rex spoke 20s ago; NO cooldown metadata.
        with mock.patch.object(presence_cadence, "_visible_person_present",
                               return_value=True):
            scored = self._score("room_change", seconds_since_spoke=20.0)
        self.assertTrue(scored.rejected)
        self.assertTrue(any(r.startswith("cadence_clamp") for r in scored.reasons))

    def test_idle_monologue_clamped_when_lean_brain_off(self):
        # The classic-brain path (kill switch): idle banter faces the central clamp.
        with (
            mock.patch.object(config, "LEAN_BRAIN_ENABLED", False),
            mock.patch.object(presence_cadence, "_visible_person_present",
                              return_value=True),
        ):
            scored = self._score("idle_monologue", seconds_since_spoke=20.0)
        self.assertTrue(scored.rejected)
        self.assertTrue(any(r.startswith("cadence_clamp") for r in scored.reasons))

    def test_chatter_outside_gap_passes(self):
        with mock.patch.object(presence_cadence, "_visible_person_present",
                               return_value=True):
            scored = self._score("room_change", seconds_since_spoke=90.0)
        self.assertFalse(any(r.startswith("cadence_clamp") for r in scored.reasons))

    def test_engaged_conversation_uses_short_gap(self):
        # 20s since Rex spoke, conversation ACTIVE -> base gap 12s -> passes.
        with mock.patch.object(presence_cadence, "_visible_person_present",
                               return_value=True):
            scored = self._score("room_change", seconds_since_spoke=20.0,
                                 profile=_profile(conversation_active=True))
        self.assertFalse(any(r.startswith("cadence_clamp") for r in scored.reasons))

    def test_event_driven_purposes_are_never_clamped(self):
        with mock.patch.object(presence_cadence, "_visible_person_present",
                               return_value=False):  # even in an empty room
            scored = self._score("presence_reaction", seconds_since_spoke=5.0)
        self.assertFalse(any(r.startswith("cadence_clamp") for r in scored.reasons))

    def test_salient_candidates_bypass(self):
        with mock.patch.object(presence_cadence, "_visible_person_present",
                               return_value=True):
            scored = self._score("room_change", seconds_since_spoke=5.0,
                                 metadata={"salient": True})
        self.assertFalse(any(r.startswith("cadence_clamp") for r in scored.reasons))

    def test_never_spoken_yet_is_not_clamped(self):
        gov = ActionGovernor()
        candidate = CandidateMove(source="test", purpose="room_change",
                                  priority=50, label="t", metadata={})
        from intelligence import consciousness as c
        with (
            mock.patch.object(c, "_last_proactive_speech_at", 0.0),
            mock.patch.object(presence_cadence, "_visible_person_present",
                              return_value=True),
        ):
            scored = gov._score(candidate, profile=_profile())
        self.assertFalse(any(r.startswith("cadence_clamp") for r in scored.reasons))


if __name__ == "__main__":
    unittest.main()
