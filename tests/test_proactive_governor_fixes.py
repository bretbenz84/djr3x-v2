"""Four ways proactive speech misfired or went silent in the 2026-08-20 run."""

import unittest
from unittest import mock

import config
from intelligence import action_governor, consciousness as C, interaction as I


class LeanAddresseeTests(unittest.TestCase):
    """The lull impulse — the only remaining source of proactive lull speech, since
    the governor hard-rejects the legacy silence-fill taxonomy — went dead the
    moment a second person spoke. JT's first turn at 20:12:05 killed it for the
    rest of the session: ~11 minutes of unbroken `no_known_person`, and 2300 of
    3271 consults (70%) died on that one gate."""

    def setUp(self):
        self._people = set(I._session_person_ids)
        self._turns = dict(I._session_person_turn_counts)
        I._session_person_ids.clear()
        I._session_person_turn_counts.clear()

    def tearDown(self):
        I._session_person_ids.clear(); I._session_person_ids.update(self._people)
        I._session_person_turn_counts.clear()
        I._session_person_turn_counts.update(self._turns)

    def test_single_person_still_uses_the_strict_answer(self):
        I._session_person_ids.add(1)
        self.assertEqual(I._lean_impulse_addressee(), (1, "primary"))

    def test_two_speakers_resolve_to_the_engaged_partner(self):
        I._session_person_ids.update({1, 4})
        with mock.patch.object(C, "get_recent_engagement",
                               return_value={"person_id": 4, "name": "JT"}):
            self.assertEqual(I._lean_impulse_addressee(), (4, "engaged"))

    def test_falls_back_to_the_most_talkative_present_person(self):
        I._session_person_ids.update({1, 4})
        I._session_person_turn_counts.update({1: 12, 4: 3})
        with (
            mock.patch.object(C, "get_recent_engagement", return_value=None),
            mock.patch.object(I, "_lean_impulse_person_present", return_value=True),
        ):
            self.assertEqual(I._lean_impulse_addressee(), (1, "most_turns"))

    def test_skips_the_most_talkative_person_if_they_left(self):
        I._session_person_ids.update({1, 4})
        I._session_person_turn_counts.update({1: 12, 4: 3})
        with (
            mock.patch.object(C, "get_recent_engagement", return_value=None),
            mock.patch.object(I, "_lean_impulse_person_present",
                              side_effect=lambda pid: pid == 4),
        ):
            self.assertEqual(I._lean_impulse_addressee(), (4, "most_turns"))

    def test_empty_room_is_distinguished_from_ambiguity(self):
        """Both were logged `no_known_person`, which is why this hid for so long."""
        with mock.patch.object(C, "get_recent_engagement", return_value=None):
            self.assertEqual(I._lean_impulse_addressee(), (None, "empty_room"))
            I._session_person_ids.update({1, 4})
            with mock.patch.object(I, "_lean_impulse_person_present", return_value=False):
                self.assertEqual(I._lean_impulse_addressee(),
                                 (None, "ambiguous_addressee"))

    def test_never_raises_when_consciousness_misbehaves(self):
        I._session_person_ids.update({1, 4})
        with (
            mock.patch.object(C, "get_recent_engagement", side_effect=RuntimeError),
            mock.patch.object(I, "_lean_impulse_person_present", return_value=False),
        ):
            self.assertEqual(I._lean_impulse_addressee()[0], None)


class TopicCooldownArmsOnSpeechTests(unittest.TestCase):
    """20:16:39 — the animal remark WON the arbitration, yielded to the user
    mid-sentence, said nothing, and still blocked its own topic for the full 45 s
    while three otherwise-ELIGIBLE cycles were rejected `topic_repeat_cooldown`.
    The cooldown exists to stop a flickering cue re-selecting a line that DID
    speak."""

    def setUp(self):
        with action_governor._recent_selected_lock:
            action_governor._recent_selected.clear()
        self.addCleanup(action_governor._recent_selected.clear)

    def _candidate(self):
        return action_governor.CandidateMove(
            source="test", purpose="world.animal_arrival",
            label="animal return: dog", suggested_text="And Max remains on station.",
        )

    def test_winning_alone_does_not_arm_the_cooldown(self):
        cand = self._candidate()
        key = action_governor.ActionGovernor._candidate_topic_key(cand)
        self.assertFalse(
            action_governor._topic_recently_selected(key, cand.purpose))
        # Arbitration happens; nothing speaks.
        self.assertFalse(
            action_governor._topic_recently_selected(key, cand.purpose),
            "a candidate that only WON must not block its own topic",
        )

    def test_speaking_arms_the_cooldown(self):
        cand = self._candidate()
        key = action_governor.ActionGovernor._candidate_topic_key(cand)
        action_governor.note_topic_spoken(cand)
        self.assertTrue(
            action_governor._topic_recently_selected(key, cand.purpose),
            "a line that spoke must still de-dup a flickering cue",
        )

    def test_note_topic_spoken_never_raises(self):
        action_governor.note_topic_spoken(None)
        action_governor.note_topic_spoken(object())

    def test_enforce_path_arms_only_when_speak_fn_reports_speech(self):
        """The ENFORCE winner speaks through a deferred speak_fn AFTER finish_cycle
        has cleared the thread-local cycle, so mark_outcome cannot reach it — the
        arming has to come off speak_fn's return value."""
        from intelligence import speech_engine

        for spoke, expect_armed in ((False, False), (True, True)):
            with self.subTest(spoke=spoke):
                action_governor._recent_selected.clear()
                cand = self._candidate()
                cand.speak_fn = lambda spoke=spoke: spoke
                key = action_governor.ActionGovernor._candidate_topic_key(cand)
                decision = mock.Mock(action="speak",
                                     selected=mock.Mock(candidate=cand))
                with (
                    mock.patch.object(action_governor.governor, "finish_cycle",
                                      return_value=decision),
                    mock.patch.object(type(action_governor.governor), "enforcing",
                                      new_callable=mock.PropertyMock,
                                      return_value=True),
                ):
                    speech_engine.finish_governor_cycle()
                self.assertEqual(
                    action_governor._topic_recently_selected(key, cand.purpose),
                    expect_armed,
                )

    def test_finish_cycle_no_longer_stamps(self):
        import inspect
        body = inspect.getsource(action_governor.ActionGovernor.finish_cycle)
        self.assertNotIn("_note_topic_selected(", body,
                         "arbitration must not arm the repeat cooldown")


class CelebrityGreetingScoreTests(unittest.TestCase):
    """speech_queue.enqueue's `priority` is a small ordinal (2 = "high, clear
    everything below"); CandidateMove.priority is the governor's SCORE on 0-100
    with a floor of 20. Three greeting sites handed the queue ordinal to the
    governor, so a first-sight greeting scored 2 and logged `below_min_score_20`
    in the same second it actually spoke (20:11:43)."""

    def test_presence_reaction_scores_well_above_the_floor(self):
        score = action_governor._PURPOSE_PRIORITIES["presence_reaction"]
        floor = int(getattr(config, "ACTION_GOVERNOR_MIN_SCORE", 20))
        self.assertGreater(score, floor)

    def test_no_call_site_passes_a_score_below_the_governor_floor(self):
        """Structural, across every governor call site in every intelligence
        module — not just the three greetings. The collision is nominal
        (`priority` means two different things), so it will recur unless the
        invariant is checked where it lives."""
        import ast
        import pathlib

        floor = int(getattr(config, "ACTION_GOVERNOR_MIN_SCORE", 20))
        offenders = []
        targets = {"_observe_governor_candidate", "observe_governor_candidate"}
        for path in pathlib.Path("intelligence").glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fn = node.func
                name = getattr(fn, "attr", None) or getattr(fn, "id", None)
                if name not in targets:
                    continue
                for kw in node.keywords:
                    if kw.arg != "priority":
                        continue
                    if isinstance(kw.value, ast.Constant) and isinstance(
                        kw.value.value, int
                    ) and kw.value.value < floor:
                        offenders.append(f"{path}:{node.lineno} priority={kw.value.value}")
        self.assertEqual(
            offenders, [],
            "a speech_queue ordinal is being passed as the governor SCORE "
            f"(must be >= ACTION_GOVERNOR_MIN_SCORE={floor}, or omitted so "
            "_PURPOSE_PRIORITIES supplies it): " + ", ".join(offenders),
        )


class AnimalRemarkSignOffFenceTests(unittest.TestCase):
    """20:24:28 — six seconds after Rex said goodbye and while the session-end
    consolidation was still running, he blurted a one-liner about a cat that had
    been on the couch for minutes."""

    def setUp(self):
        C._pending_animal_arrivals.clear()
        self.addCleanup(C._pending_animal_arrivals.clear)

    def test_pending_remarks_are_dropped_at_sign_off(self):
        C._pending_animal_arrivals["cat"] = {"species": "cat", "last_seen_at": 1.0}
        with mock.patch.object(C, "_session_is_signing_off", return_value=True):
            self.assertFalse(C._fire_pending_animal_arrival_reaction())
        self.assertEqual(C._pending_animal_arrivals, {},
                         "dropped, not deferred — it has outlived its conversation")

    def test_normal_conversation_is_unaffected(self):
        C._pending_animal_arrivals["dog"] = {"species": "dog", "last_seen_at": 1.0}
        with (
            mock.patch.object(C, "_session_is_signing_off", return_value=False),
            mock.patch.object(C, "_animal_remark_covered_by_report", return_value=True),
        ):
            C._fire_pending_animal_arrival_reaction()
        # Reached the normal drop path (covered-by-report), not the sign-off fence.
        self.assertEqual(C._pending_animal_arrivals, {})

    def test_signing_off_probe_never_raises(self):
        with mock.patch.dict("sys.modules", {"intelligence.end_thread": None}):
            self.assertIn(C._session_is_signing_off(), (True, False))


if __name__ == "__main__":
    unittest.main()
