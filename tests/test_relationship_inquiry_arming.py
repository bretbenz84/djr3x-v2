"""The "who's this?" relationship-inquiry reactor arms its reply window on SPEAK, not submit.

_step_relationship_inquiry asks the engaged person who an unknown visitor is and arms
_pending_relationship_prompt so the next utterance is parsed as the {name, relationship}
answer. Under ACTION_GOVERNOR_ENFORCE, _generate_and_speak returns True at governor
SUBMISSION time, so the old pre-speak arming (+ `if not _generate_and_speak(): clear`
self-heal, which was dead code) left the reply window armed even when the governor then
REJECTED the candidate and nothing was spoken — the next user statement got mis-parsed as
an answer to a question Rex never asked. The fix arms in an on_spoke callback (fires only
after the line enqueues), mirroring the identity_prompt reactor, plus an in-flight latch
with a stale-timeout auto-clear so a rejected candidate can't wedge the reactor.
"""

import unittest
from types import SimpleNamespace as NS
from unittest import mock

import config
from intelligence import consciousness as C


_T = 10_000.0  # fixed monotonic clock for the whole step


def _visitor_snapshot():
    return {
        "people": [
            {"person_db_id": 1, "id": "person_1", "face_id": "Bret Benziger"},
            {"person_db_id": None, "id": "person_2"},  # the unknown visitor
        ],
        "audio_scene": {},
    }


class RelationshipInquiryArmingTest(unittest.TestCase):
    def setUp(self):
        # Snapshot every module global the step reads/writes, and reset to a clean,
        # ripe-to-fire state.
        self._saved = {
            "pending": C._pending_relationship_prompt.is_set(),
            "ctx": dict(C._pending_relationship_context),
            "inflight": C._relationship_prompt_in_flight.is_set(),
            "inflight_at": C._relationship_prompt_in_flight_at,
            "asked": set(C._asked_relationship_slots),
            "first_seen": dict(C._unknown_first_seen_at),
            "last_prompt": C._last_identity_prompt_at,
            "engaged_id": C._engaged_person_id,
            "engaged_touch": C._engaged_last_touch_at,
        }
        self.addCleanup(self._restore)

        C._pending_relationship_prompt.clear()
        C._pending_relationship_context.clear()
        C._relationship_prompt_in_flight.clear()
        C._relationship_prompt_in_flight_at = 0.0
        C._asked_relationship_slots.clear()
        C._unknown_first_seen_at.clear()
        # Unknown seen long enough ago to be "ripe"; cooldown long expired.
        C._unknown_first_seen_at["person_2"] = _T - 100.0
        C._last_identity_prompt_at = _T - 100.0
        C._engaged_person_id = 1
        C._engaged_last_touch_at = _T  # engaged right now

    def _restore(self):
        s = self._saved
        (C._pending_relationship_prompt.set() if s["pending"]
         else C._pending_relationship_prompt.clear())
        C._pending_relationship_context.clear()
        C._pending_relationship_context.update(s["ctx"])
        (C._relationship_prompt_in_flight.set() if s["inflight"]
         else C._relationship_prompt_in_flight.clear())
        C._relationship_prompt_in_flight_at = s["inflight_at"]
        C._asked_relationship_slots.clear()
        C._asked_relationship_slots.update(s["asked"])
        C._unknown_first_seen_at.clear()
        C._unknown_first_seen_at.update(s["first_seen"])
        C._last_identity_prompt_at = s["last_prompt"]
        C._engaged_person_id = s["engaged_id"]
        C._engaged_last_touch_at = s["engaged_touch"]

    def _run(self, gen):
        """Drive the step with the gates open and _generate_and_speak = `gen`."""
        profile = NS(suppress_proactive=False)
        with mock.patch.object(C.time, "monotonic", return_value=_T), \
             mock.patch.object(C, "_can_speak", return_value=True), \
             mock.patch.object(C, "_can_proactive_speak", return_value=True), \
             mock.patch.object(C, "_generate_and_speak", side_effect=gen) as g:
            C._step_relationship_inquiry(_visitor_snapshot(), profile)
        return g

    # ── the core fix ──────────────────────────────────────────────────────────

    def test_spoken_line_arms_the_reply_window(self):
        # generate_and_speak fires on_spoke (line enqueued) and returns True.
        def gen(prompt, **kw):
            kw["on_spoke"]()
            return True
        g = self._run(gen)
        g.assert_called_once()
        self.assertEqual(g.call_args.kwargs.get("purpose"), "relationship_inquiry")
        self.assertTrue(C._pending_relationship_prompt.is_set())
        self.assertEqual(C._pending_relationship_context["engaged_person_id"], 1)
        self.assertEqual(C._pending_relationship_context["slot_id"], "person_2")
        self.assertEqual(C._pending_relationship_context["engaged_name"], "Bret Benziger")
        # cooldown armed on speak; in-flight latch released.
        self.assertEqual(C._last_identity_prompt_at, _T)
        self.assertFalse(C._relationship_prompt_in_flight.is_set())

    def test_submitted_but_not_spoken_does_NOT_arm_reply_window(self):
        # ENFORCE: submission returns True, but the governor later rejects the candidate so
        # on_spoke never fires. The reply window must stay CLOSED (the state-poisoning bug).
        def gen(prompt, **kw):
            return True  # submitted; on_spoke intentionally NOT called
        self._run(gen)
        self.assertFalse(C._pending_relationship_prompt.is_set())
        self.assertEqual(C._pending_relationship_context, {})
        # cooldown NOT burned on a candidate that never spoke.
        self.assertEqual(C._last_identity_prompt_at, _T - 100.0)
        # latch stays set (a later tick's stale-timeout will clear it — see below).
        self.assertTrue(C._relationship_prompt_in_flight.is_set())

    def test_submission_failure_releases_the_latch(self):
        # Legacy/non-enforcing path (or a rejected claim) returns False → release the latch.
        def gen(prompt, **kw):
            return False
        self._run(gen)
        self.assertFalse(C._pending_relationship_prompt.is_set())
        self.assertFalse(C._relationship_prompt_in_flight.is_set())

    # ── the in-flight latch ───────────────────────────────────────────────────

    def test_fresh_inflight_latch_blocks_a_duplicate_submission(self):
        C._relationship_prompt_in_flight.set()
        C._relationship_prompt_in_flight_at = _T  # brand new
        g = self._run(lambda *a, **k: True)
        g.assert_not_called()  # returned early, no second candidate

    def test_stale_inflight_latch_auto_clears_and_retries(self):
        C._relationship_prompt_in_flight.set()
        C._relationship_prompt_in_flight_at = _T - 100.0  # older than the stale window
        calls = {}

        def gen(prompt, **kw):
            calls["fired"] = True
            kw["on_spoke"]()
            return True
        g = self._run(gen)
        g.assert_called_once()  # dead latch cleared → the ask went out
        self.assertTrue(C._pending_relationship_prompt.is_set())

    def test_stale_window_exceeds_llm_inflight_time(self):
        # The relationship line runs the LLM INSIDE the in-flight window (unlike
        # identity_prompt's fixed string), so the stale timeout must clear the worst-case
        # generation time — otherwise a slow-but-legitimate "who's this?" is judged stale and
        # re-asked (adversarial review 2026-07-10). Pin the invariant so the value can't
        # regress back to identity_prompt's enqueue-only 10s.
        self.assertGreater(
            float(config.RELATIONSHIP_PROMPT_INFLIGHT_STALE_SECS),
            float(config.LLM_REQUEST_TIMEOUT_SECS),
        )


if __name__ == "__main__":
    unittest.main()
