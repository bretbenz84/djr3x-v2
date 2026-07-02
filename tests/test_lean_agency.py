"""Lean agency (Phase 1): under LEAN_BRAIN_ENABLED the old silence-fill proactive is suppressed
and replaced by ONE motivated impulse that Rex chooses to make — or passes on. Covers the
governor suppression gate and the impulse's PASS/act parsing (mocked — no API)."""

import contextlib
import unittest
from types import SimpleNamespace as NS
from unittest import mock

import config
from intelligence import lean_brain
from intelligence.action_governor import ActionGovernor, CandidateMove


class GovernorSuppressionTest(unittest.TestCase):
    def _rejected(self, purpose, lean):
        prev = config.LEAN_BRAIN_ENABLED
        try:
            config.LEAN_BRAIN_ENABLED = lean
            g = ActionGovernor()
            return g._score(CandidateMove(source="t", purpose=purpose, priority=50, label=purpose)).rejected
        finally:
            config.LEAN_BRAIN_ENABLED = prev

    def test_silence_fill_suppressed_only_under_lean(self):
        self.assertTrue(self._rejected("idle_monologue", lean=True))
        self.assertFalse(self._rejected("idle_monologue", lean=False))   # classic path intact

    def test_perception_reactors_never_suppressed(self):
        for reactor in ("presence_reaction", "world.animal_arrival", "world.scenery_change"):
            self.assertFalse(self._rejected(reactor, lean=True), reactor)


def _one_chunk_stream(text):
    """A fake OpenAI stream yielding one delta with `text`."""
    return [NS(choices=[NS(delta=NS(content=text))])]


class ImpulseDecisionParsingTest(unittest.TestCase):
    def test_pass_means_watch(self):
        with mock.patch.object(lean_brain.llm_compat, "create", return_value=_one_chunk_stream("PASS")):
            self.assertEqual(lean_brain.consider_initiating(person_id=None, transcript=[]), "")

    def test_pass_with_trailing_junk_still_watches(self):
        with mock.patch.object(lean_brain.llm_compat, "create", return_value=_one_chunk_stream('PASS.')):
            self.assertEqual(lean_brain.consider_initiating(person_id=None, transcript=[]), "")

    def test_a_real_line_is_spoken(self):
        with mock.patch.object(lean_brain.llm_compat, "create",
                               return_value=_one_chunk_stream("Nice hat, Bret.")):
            self.assertEqual(lean_brain.consider_initiating(person_id=None, transcript=[]),
                             "Nice hat, Bret.")


class ImpulseBackoffTest(unittest.TestCase):
    """Talking into the void: Rex breaks a lull once or twice, then goes quiet until the user
    speaks — he must NOT keep quipping every cooldown-tick (the 'piled 4 lines about your dinner'
    failure). Drives _maybe_lean_impulse through a silence with a fake clock."""

    def test_caps_and_escalates_then_resets_on_user_speech(self):
        import intelligence.interaction as I
        clock = {"t": 1000.0}
        spoken = []
        fire = lambda: I._maybe_lean_impulse(idle_for=5.0, effective_idle_timeout=60.0)

        with contextlib.ExitStack() as es:
            p = es.enter_context
            p(mock.patch.object(I.config, "LEAN_BRAIN_ENABLED", True))
            p(mock.patch.object(I.config, "LEAN_IMPULSE_ENABLED", True))
            p(mock.patch.object(I.config, "LEAN_IMPULSE_QUIET_SECS", 4.0))
            p(mock.patch.object(I.config, "LEAN_IMPULSE_COOLDOWN_SECS", 12.0))
            p(mock.patch.object(I.config, "LEAN_IMPULSE_MAX_UNANSWERED", 2))
            p(mock.patch.object(I.config, "LEAN_IMPULSE_ESCALATION", 1.0))
            p(mock.patch.object(I.config, "PROACTIVE_LINE_MIN_GAP_SECS", 6.0))
            p(mock.patch.object(I.time, "monotonic", lambda: clock["t"]))
            p(mock.patch.object(I, "_game_suppresses_conversation", lambda: False))
            p(mock.patch.object(I, "_directed_context_fresh", lambda: False))
            p(mock.patch.object(I.end_thread, "is_grace_active", lambda: False))
            p(mock.patch.object(I, "_primary_session_person_id", lambda: 1))
            p(mock.patch.object(I.speech_queue, "seconds_since_last_speech", lambda: 5.0))
            p(mock.patch.object(I.speech_queue, "is_speaking", lambda: False))
            p(mock.patch.object(I.consciousness, "is_waiting_for_response", lambda: False))
            p(mock.patch.object(I.output_gate, "is_busy", lambda: False))
            p(mock.patch.object(I.echo_cancel, "is_suppressed", lambda: False))
            p(mock.patch.object(I, "_suppress_proactive_after_heavy", lambda pid: False))
            p(mock.patch.object(I.body_mood, "current_mood", lambda: ("amused", 0.5)))
            p(mock.patch.object(I, "_lean_recent_transcript", lambda s: []))
            p(mock.patch.object(I, "_lean_world", lambda: {}))
            p(mock.patch.object(I, "_line_duplicates_recent_question", lambda line: False))
            p(mock.patch.object(I, "_register_rex_utterance", lambda *a, **k: None))
            p(mock.patch.object(I.conv_memory, "add_to_transcript", lambda *a, **k: None))
            p(mock.patch.object(I.conv_log, "log_rex", lambda *a, **k: None))
            p(mock.patch.object(lean_brain, "consider_initiating", lambda *a, **k: "A line."))

            def speak(line, **k):
                spoken.append((clock["t"], line))
                return True
            p(mock.patch.object(I, "_speak_proactive", speak))

            # Fresh silence.
            I._consecutive_lean_impulses = 0
            I._last_lean_impulse_at = 0.0
            I._last_proactive_line_at = 0.0
            I._floor_held_until = 0.0
            I._interrupted.clear()

            self.assertTrue(fire(), "1st line breaks the lull immediately")
            clock["t"] += 10.0
            self.assertFalse(fire(), "2nd blocked — escalated gap is 24s, only 10s elapsed")
            clock["t"] += 15.0                                   # 25s since the 1st line
            self.assertTrue(fire(), "2nd line fires once the escalated 24s gap passes")
            clock["t"] += 500.0
            self.assertFalse(fire(), "3rd never fires — capped at MAX_UNANSWERED with no user reply")
            self.assertEqual(len(spoken), 2)

            # User speaks → the reset _begin_user_turn performs → fresh allowance.
            I._consecutive_lean_impulses = 0
            I._last_lean_impulse_at = 0.0
            I._last_proactive_line_at = 0.0
            self.assertTrue(fire(), "after the user re-engages, Rex may break the next lull again")
            self.assertEqual(len(spoken), 3)


class ImpulseReengageTest(unittest.TestCase):
    """After the fast lull-break has yielded the floor (cap hit), a LONG silence with the person
    still present should still get ONE patient, fresh-topic re-engagement — bypassing the fast cap,
    routed through the calmer re-engage voice (long_silence=True)."""

    def _run(self, *, quiet_secs, consecutive):
        import intelligence.interaction as I
        captured = {}
        spoken = []
        with contextlib.ExitStack() as es:
            p = es.enter_context
            p(mock.patch.object(I.config, "LEAN_BRAIN_ENABLED", True))
            p(mock.patch.object(I.config, "LEAN_IMPULSE_ENABLED", True))
            p(mock.patch.object(I.config, "LEAN_IMPULSE_QUIET_SECS", 4.0))
            p(mock.patch.object(I.config, "LEAN_IMPULSE_COOLDOWN_SECS", 12.0))
            p(mock.patch.object(I.config, "LEAN_IMPULSE_MAX_UNANSWERED", 2))
            p(mock.patch.object(I.config, "LEAN_IMPULSE_ESCALATION", 1.0))
            p(mock.patch.object(I.config, "LEAN_IMPULSE_REENGAGE_SECS", 40.0))
            p(mock.patch.object(I.config, "PROACTIVE_LINE_MIN_GAP_SECS", 6.0))
            p(mock.patch.object(I.time, "monotonic", lambda: 10_000.0))
            p(mock.patch.object(I, "_game_suppresses_conversation", lambda: False))
            p(mock.patch.object(I, "_directed_context_fresh", lambda: False))
            p(mock.patch.object(I.end_thread, "is_grace_active", lambda: False))
            p(mock.patch.object(I, "_primary_session_person_id", lambda: 1))
            p(mock.patch.object(I.speech_queue, "seconds_since_last_speech", lambda: quiet_secs))
            p(mock.patch.object(I.speech_queue, "is_speaking", lambda: False))
            p(mock.patch.object(I.consciousness, "is_waiting_for_response", lambda: False))
            p(mock.patch.object(I.output_gate, "is_busy", lambda: False))
            p(mock.patch.object(I.echo_cancel, "is_suppressed", lambda: False))
            p(mock.patch.object(I, "_suppress_proactive_after_heavy", lambda pid: False))
            p(mock.patch.object(I.body_mood, "current_mood", lambda: ("relaxed", 0.4)))
            p(mock.patch.object(I, "_lean_recent_transcript", lambda s: []))
            p(mock.patch.object(I, "_lean_world", lambda: {}))
            p(mock.patch.object(I, "_line_duplicates_recent_question", lambda line: False))
            p(mock.patch.object(I, "_register_rex_utterance", lambda *a, **k: None))
            p(mock.patch.object(I.conv_memory, "add_to_transcript", lambda *a, **k: None))
            p(mock.patch.object(I.conv_log, "log_rex", lambda *a, **k: None))

            def consider(*a, **k):
                captured["long_silence"] = k.get("long_silence")
                return "So what's the plan for the long weekend?"
            p(mock.patch.object(lean_brain, "consider_initiating", consider))
            p(mock.patch.object(I, "_speak_proactive", lambda line, **k: spoken.append(line) or True))

            # Fast run already exhausted; last line was long ago.
            I._consecutive_lean_impulses = consecutive
            I._last_lean_impulse_at = 10_000.0 - 100.0
            I._last_proactive_line_at = 0.0
            I._floor_held_until = 0.0
            I._interrupted.clear()
            fired = I._maybe_lean_impulse(idle_for=60.0, effective_idle_timeout=120.0)
        return fired, captured.get("long_silence"), spoken

    def test_long_silence_reengages_past_the_fast_cap(self):
        fired, mode, spoken = self._run(quiet_secs=42.0, consecutive=5)   # cap (2) already blown
        self.assertTrue(fired, "40s+ quiet with the person present → one patient re-engage fires")
        self.assertTrue(mode, "re-engage routes through the calm long_silence voice")
        self.assertEqual(len(spoken), 1)

    def test_short_silence_stays_capped(self):
        fired, mode, _ = self._run(quiet_secs=10.0, consecutive=5)        # < REENGAGE_SECS
        self.assertFalse(fired, "under the re-engage threshold, the fast cap still holds — no pile-on")


class RecentTopicsAwarenessTest(unittest.TestCase):
    """Cross-session 'already discussed, don't re-open' awareness reaches the lean prompt (replies
    AND the silence impulse), and is inert when the flag is off."""

    def _patched_topics(self, topics):
        from memory import episodic_recall
        return mock.patch.object(episodic_recall, "recent_conversation_topics", lambda *a, **k: topics)

    def test_recent_topics_reach_reply_prompt(self):
        with mock.patch.object(config, "RECENT_TOPICS_AWARENESS_ENABLED", True), \
             self._patched_topics(["Bret went to his dad's for fireworks", "Bret rewatched Pirates"]):
            sp = lean_brain._system_prompt(1, {})
        # Reply path frames recent topics as RECALLABLE-when-asked (not a blanket suppression).
        self.assertIn("IN YOUR MEMORY", sp)
        self.assertIn("RECALL and answer", sp)
        self.assertIn("fireworks", sp)
        self.assertIn("Pirates", sp)

    def test_recent_topics_reach_impulse_situation(self):
        with mock.patch.object(config, "RECENT_TOPICS_AWARENESS_ENABLED", True), \
             self._patched_topics(["Bret went to his dad's for fireworks"]):
            block = lean_brain._situation_block(1, {}, quiet_secs=42, mood="amused")
        self.assertIn("ALREADY COVERED", block)
        self.assertIn("fireworks", block)

    def test_flag_off_is_inert(self):
        with mock.patch.object(config, "RECENT_TOPICS_AWARENESS_ENABLED", False), \
             self._patched_topics(["Bret went to his dad's for fireworks"]):
            sp = lean_brain._system_prompt(1, {})
        self.assertNotIn("IN YOUR MEMORY", sp)


class OneVoiceTest(unittest.TestCase):
    """Phase 4: proactive/greeting/reaction generation shares the lean persona — llm.stream_response
    routes to lean_brain.stream_directive — EXCEPT the reply-path classic fallbacks (classic=True)
    and whenever the lean brain is disabled."""

    def test_one_voice_active_gating(self):
        from intelligence import llm
        with mock.patch.object(llm.config, "LEAN_BRAIN_ENABLED", True), \
             mock.patch.object(llm.config, "LEAN_ONE_VOICE_ENABLED", True):
            self.assertTrue(llm._one_voice_active(classic=False))
            self.assertFalse(llm._one_voice_active(classic=True))    # reply fallback stays classic
        with mock.patch.object(llm.config, "LEAN_BRAIN_ENABLED", False), \
             mock.patch.object(llm.config, "LEAN_ONE_VOICE_ENABLED", True):
            self.assertFalse(llm._one_voice_active(classic=False))   # lean brain off → all classic

    def test_stream_response_routes_to_lean(self):
        from intelligence import llm
        used = {"lean": False, "classic": False}

        def fake_directive(instruction, person_id=None, world=None, transcript=None):
            used["lean"] = True
            yield "lean line."

        def fake_assemble(*a, **k):
            used["classic"] = True
            return "CLASSIC"

        with mock.patch.object(llm.config, "LEAN_BRAIN_ENABLED", True), \
             mock.patch.object(llm.config, "LEAN_ONE_VOICE_ENABLED", True), \
             mock.patch.object(llm, "assemble_system_prompt", fake_assemble), \
             mock.patch.object(llm, "_one_voice_world", lambda: None), \
             mock.patch.object(llm, "_one_voice_transcript", lambda: None), \
             mock.patch("intelligence.lean_brain.stream_directive", fake_directive):
            out = "".join(llm.stream_response("greet Bret warmly", 1))
        self.assertEqual(out, "lean line.")
        self.assertTrue(used["lean"])
        self.assertFalse(used["classic"])   # never touched the classic assembled prompt

    def test_classic_flag_bypasses_lean(self):
        from intelligence import llm
        used = {"lean": False}

        def fake_directive(*a, **k):
            used["lean"] = True
            yield "x"

        with mock.patch.object(llm.config, "LEAN_BRAIN_ENABLED", True), \
             mock.patch.object(llm.config, "LEAN_ONE_VOICE_ENABLED", True), \
             mock.patch.object(llm, "assemble_system_prompt", lambda *a, **k: "CLASSIC"), \
             mock.patch.object(llm.llm_compat, "create", side_effect=RuntimeError("no api")), \
             mock.patch("intelligence.lean_brain.stream_directive", fake_directive):
            list(llm.stream_response("hi", 1, classic=True))   # reply-path fallback
        self.assertFalse(used["lean"])      # classic=True must never route to the lean voice


if __name__ == "__main__":
    unittest.main()
