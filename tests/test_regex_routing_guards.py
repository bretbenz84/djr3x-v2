"""Regression corpus for the 2026-08-13 regex-routing audit.

Every case here is a VERIFIED misfire — each one was reproduced against the
classifiers on HEAD before the guard that now blocks it was written. The audit
found the deterministic regex layers take 94.5% of all non-conversation
executions (416/440 audited turns) and run ahead of every LLM layer, so a false
positive here is not a bad reply: it drives wheels, writes people.db, or powers
the robot off.

Grouped by the failure MODE rather than by module, because the same shapes keep
recurring across families:

  * inversion  — a refusal executed the thing refused ("Don't roast me.")
  * narration  — a story about someone else executed as a command
  * idiom      — a stock English phrase matched a command pattern
  * bypass     — an execution path that skipped its own safety gate
  * too-narrow — a real command that no lane would accept

The must-STILL-fire half of each test matters as much as the must-not: these
guards are only correct if the commands they protect still work, and several of
the shapes below (ASR past-tense "Turned right a little bit.", the comma'd
"Turn left, 15 degrees.") are themselves field fixes from earlier sessions.
"""

import unittest
from unittest import mock

import numpy as np

from intelligence import action_router
from intelligence import command_parser
from intelligence import interaction
from state import State


def _act(decision):
    return decision.action if decision is not None else None


class HumorInversionTest(unittest.TestCase):
    """A refused or merely reported joke must not be performed."""

    def test_negated_humor_does_not_perform(self):
        for text in (
            "Don't roast me.",
            "Don't tease her about it.",
            "No more jokes, please.",
            "Stop roasting me.",
        ):
            self.assertIsNone(
                action_router.classify_explicit_humor(text), f"{text!r} must not fire"
            )

    def test_narrated_humor_is_not_a_request(self):
        for text in (
            "They mock me at school for my accent.",
            "He'd always tell me a joke before bed.",
            "My grandpa used to crack a joke at every dinner.",
            "We used to roast marshmallows over the fire.",
        ):
            self.assertIsNone(
                action_router.classify_explicit_humor(text), f"{text!r} must not fire"
            )

    def test_roast_food_stoplist_survives_an_article(self):
        # Field 2026-08-01: "Man, this heat could roast a panther." executed a
        # roast. The stoplist was consulted with the article still attached, so
        # "a turkey" never matched the "turkey" entry.
        for text in (
            "This heat could roast a turkey.",
            "Man, this heat could roast a panther.",
            "I'm going to roast a chicken for dinner.",
        ):
            self.assertIsNone(
                action_router.classify_explicit_humor(text), f"{text!r} must not fire"
            )

    def test_real_roast_and_joke_requests_still_fire(self):
        cases = {
            "Roast me.": "humor.roast",
            "Roast Bret.": "humor.roast",
            "Roast the room.": "humor.roast",
            "I want you to roast me.": "humor.roast",
            # The negation guard may not cross a clause boundary: this is a
            # REQUEST whose first clause happens to contain a negator.
            "Don't hold back, roast me.": "humor.roast",
            "Tell me a joke.": "humor.tell_joke",
            "Can you tell me a joke?": "humor.tell_joke",
            "Got any jokes?": "humor.tell_joke",
            "Say something funny.": "humor.free_bit",
        }
        for text, expected in cases.items():
            self.assertEqual(
                _act(action_router.classify_explicit_humor(text)), expected, text
            )


class PerformanceInversionTest(unittest.TestCase):
    """Emotional vocabulary in ordinary talk must not drive the servos."""

    def test_negated_and_narrated_performance_does_not_fire(self):
        for text in (
            "Don't be sad.",
            "I agree with you.",
            "If she asks, just say yes.",
            "We should celebrate your birthday.",
            "He always acts surprised when I say that.",
        ):
            self.assertIsNone(
                action_router.classify_explicit_performance(text),
                f"{text!r} must not fire",
            )

    def test_describing_rex_is_not_an_order(self):
        # "You look sad today, buddy." is the speaker describing Rex; the beat
        # patterns matched the bare "look sad" anywhere in the turn.
        for text in ("You look sad today, buddy.", "You seem annoyed.", "You're sad."):
            self.assertIsNone(
                action_router.classify_explicit_performance(text),
                f"{text!r} must not fire",
            )

    def test_voice_question_is_not_an_impersonation_order(self):
        # Both used to arm a voice-capture slot with a garbage target
        # ("you like my", "a") and ask the speaker to repeat after Rex.
        for text in ("Do you like my voice?", "Can you do a voice like mine?"):
            self.assertIsNone(
                action_router.classify_explicit_performance(text),
                f"{text!r} must not fire",
            )

    def test_real_performance_requests_still_fire(self):
        cases = {
            "Act surprised.": "performance.body_beat",
            "Can you look surprised?": "performance.body_beat",
            "I want you to act surprised.": "performance.body_beat",
            "Do a victory dance.": "performance.body_beat",
            "Nod your head.": "performance.body_beat",
            "Hype the room.": "performance.dj_bit",
            "Impersonate Jimmy Carter.": "performance.impersonate",
            "Do my voice.": "performance.impersonate",
            "Do Jimmy Carter's voice.": "performance.impersonate",
        }
        for text, expected in cases.items():
            self.assertEqual(
                _act(action_router.classify_explicit_performance(text)), expected, text
            )


class MemoryDiscardInversionTest(unittest.TestCase):
    """"Don't forget" is the OPPOSITE of a discard request."""

    def test_keep_intent_is_not_a_discard(self):
        for text in (
            "Don't forget that we have dinner tomorrow.",
            "I'll never forget that trip, it was incredible.",
            "Don't ever forget that.",
        ):
            self.assertIsNone(
                action_router.classify_explicit_control(text), f"{text!r} must not fire"
            )
            self.assertFalse(action_router._is_recent_discard_request(text), text)

    def test_real_discards_still_fire(self):
        for text in (
            "Forget what I just said.",
            # "don't remember that" IS a discard — the guard keys on the verb
            # FORGET, not on the presence of a negator.
            "Don't remember that.",
            "Don't store that.",
        ):
            self.assertEqual(
                _act(action_router.classify_explicit_control(text)),
                "memory.recent_discard",
                text,
            )


class MotionIdiomTest(unittest.TestCase):
    """Stock English phrases must not drive a physical base."""

    def test_idioms_do_not_move_the_base(self):
        for text in (
            # The worst case: the 2026-08-11 ASR past-tense repair rewrote
            # sentence-initial "Turned" and the 20-char turn gap then matched
            # "turn ... right" — a compliment spun the robot.
            "Turned out he was right.",
            "She had to turn her whole life around.",
            "Turn that frown around.",
            "turn my life around",
            "turn the car around",
            "turn the right one on",
            "face the right way",
            "Go right ahead and tell him.",
            "I go right to bed after that.",
            "We just need to move forward as a team.",
            "I can't face them right now.",
            "don't face north",
            # A verbless "over here" used to be a summons all by itself.
            "It's crazy over here at work.",
            "the party's over here",
        ):
            self.assertIsNone(
                action_router.classify_explicit_motion(text), f"{text!r} must not move"
            )

    def test_reported_speech_is_not_a_summons(self):
        for text in (
            "My sister said to come here for Thanksgiving.",
            "he said to come here",
            "mom asked me to come here",
        ):
            self.assertIsNone(
                action_router.classify_explicit_motion(text), f"{text!r} must not move"
            )

    def test_owner_reissuing_the_command_still_moves(self):
        # First person is the owner repeating themselves, not quoting somebody.
        self.assertEqual(
            _act(action_router.classify_explicit_motion("I told you to come here")),
            "motion.come",
        )

    def test_real_motion_commands_still_fire(self):
        cases = {
            "turn left": "motion.turn",
            "turn right 45 degrees": "motion.turn",
            "turn to your left": "motion.turn",
            "rotate a little to your left": "motion.turn",
            "turn slightly right": "motion.turn",
            "turn all the way around": "motion.turn",
            "turn around": "motion.turn",
            "turn 180": "motion.turn",
            "face north": "motion.turn",
            "move forward": "motion.move",
            "move forward two feet": "motion.move",
            "back up": "motion.move",
            "go north": "motion.move",
            "go left": "motion.arc",
            "move to your left": "motion.arc",
            "scoot over to the left": "motion.arc",
            "come here": "motion.come",
            "come over here": "motion.come",
            "come right over here": "motion.come",
            "get over here": "motion.come",
            "stop moving": "motion.stop",
            "hold still": "motion.stop",
        }
        for text, expected in cases.items():
            self.assertEqual(
                _act(action_router.classify_explicit_motion(text)), expected, text
            )

    def test_earlier_field_fixes_are_preserved(self):
        # commit 0315482 — ASR renders the imperative in past tense.
        self.assertEqual(
            _act(action_router.classify_explicit_motion("Turned right a little bit.")),
            "motion.turn",
        )
        self.assertEqual(
            _act(
                action_router.classify_explicit_motion(
                    "You went too far. Turned right a little bit."
                )
            ),
            "motion.turn",
        )
        # commit ee5465a — the comma is punctuation, not a route separator.
        decision = action_router.classify_explicit_motion("Turn left, 15 degrees.")
        self.assertEqual(_act(decision), "motion.turn")
        self.assertEqual(decision.args.get("deg"), 15.0)
        # commits 1bc9ad9 / 1beab25 — speaker bearings.
        self.assertEqual(
            _act(action_router.classify_explicit_motion("I'm behind you, come here")),
            "motion.come",
        )
        self.assertEqual(
            _act(action_router.classify_explicit_motion("I'm to your left")),
            "motion.turn",
        )


class MotionExecutePolicyTest(unittest.TestCase):
    """Motion had no allowlist, no confidence floor and no kill switch."""

    def test_motion_actions_are_in_the_execute_allowlist(self):
        import config

        for action in (
            "motion.turn",
            "motion.move",
            "motion.arc",
            "motion.come",
            "motion.stop",
            "motion.explore",
        ):
            self.assertIn(action, config.ACTION_ROUTER_EXECUTE_ACTIONS, action)

    def test_deterministic_motion_passes_the_policy(self):
        decision = action_router.classify_explicit_motion("turn left")
        self.assertTrue(interaction._motion_takeover_executable(decision))

    def test_allowlist_is_a_real_kill_switch(self):
        import config

        decision = action_router.classify_explicit_motion("turn left")
        reduced = {
            a for a in config.ACTION_ROUTER_EXECUTE_ACTIONS if a != "motion.turn"
        }
        with mock.patch.object(config, "ACTION_ROUTER_EXECUTE_ACTIONS", reduced):
            self.assertFalse(interaction._motion_takeover_executable(decision))

    def test_ambient_llm_motion_read_needs_command_evidence(self):
        # Allowlisting motion.* also un-gated the LLM-decided motion branch,
        # which had been dead only because the allowlist blocked it.
        ambient = action_router.ActionDecision(
            action="motion.move", confidence=0.9, args={"direction": "forward"}
        )
        self.assertEqual(
            action_router.missing_required_evidence_reason(
                "I think we should move on from that topic", ambient
            ),
            "missing_motion_command_evidence",
        )
        self.assertIsNone(
            action_router.missing_required_evidence_reason(
                "move forward two feet", ambient
            )
        )


def _dj_listen_step(text):
    """Run one during-playback listener step with the audio path stubbed."""
    from features import dj as dj_mod

    calls = {"spoken": [], "dj": []}
    chunk = np.ones(1600, dtype=np.float32)
    with mock.patch.object(interaction.stream, "get_audio_chunk", return_value=chunk), \
         mock.patch.object(interaction.vad, "is_speech", return_value=True), \
         mock.patch.object(interaction, "_accumulate_speech", return_value=chunk), \
         mock.patch.object(interaction.transcription, "transcribe", return_value=text), \
         mock.patch.object(interaction, "_duck_dj_for_speech", return_value=None), \
         mock.patch.object(interaction, "_restore_dj_volume"), \
         mock.patch.object(
             interaction, "_speak_blocking",
             side_effect=lambda line, **kw: calls["spoken"].append(line) or True), \
         mock.patch.object(interaction.state_module, "set_state") as set_state, \
         mock.patch.object(dj_mod, "stop",
                           side_effect=lambda **kw: calls["dj"].append("stop")), \
         mock.patch.object(dj_mod, "skip",
                           side_effect=lambda: calls["dj"].append("skip")), \
         mock.patch.object(dj_mod, "volume_up",
                           side_effect=lambda step=None: calls["dj"].append("volume_up")), \
         mock.patch.object(dj_mod, "volume_down",
                           side_effect=lambda step=None: calls["dj"].append("volume_down")), \
         mock.patch.object(dj_mod, "is_playing", return_value=True):
        interaction._dj_command_listen_step(allowed_states=(State.ACTIVE,))
    calls["set_state"] = set_state
    return calls


class DJListenerFuzzyBypassTest(unittest.TestCase):
    """The during-music listener was the one path that skipped the fuzzy gate."""

    def test_fuzzy_near_misses_never_power_off(self):
        # Each of these resolves to command_key "shutdown" at >= the 0.82 fuzzy
        # threshold and powered the robot off mid-track. "sit down" is the
        # sharpest: the wake-confirm path deliberately EXCLUDES that homophone
        # because it may be aimed at a pet, and the fuzzy lane handed it back.
        for text in ("shot down", "sun down", "cut down", "sit down",
                     "shut it down", "turn it off", "powder down"):
            calls = _dj_listen_step(text)
            calls["set_state"].assert_not_called()
            self.assertEqual(calls["dj"], [], f"{text!r} must not act")

    def test_non_destructive_fuzzy_is_also_dropped(self):
        self.assertEqual(_dj_listen_step("next dog")["dj"], [])

    def test_real_music_commands_still_work(self):
        # The listener exists because of the 2026-07-30 failure where "stop the
        # music" did nothing and the owner had to kill the process.
        self.assertIn("stop", _dj_listen_step("Stop the music.")["dj"])
        self.assertIn("stop", _dj_listen_step("Stop.")["dj"])
        self.assertIn("volume_down", _dj_listen_step("Turn it down.")["dj"])
        self.assertIn("skip", _dj_listen_step("Skip this song.")["dj"])

    def test_deliberate_shutdown_still_works(self):
        calls = _dj_listen_step("Shut down.")
        calls["set_state"].assert_called_once_with(State.SHUTDOWN)


class SystemModeWideningTest(unittest.TestCase):
    """The sleep/shutdown lanes were too NARROW to accept real commands."""

    def test_sleep_accepts_a_trailing_vocative(self):
        for text in (
            "Go to sleep, Rex.",
            "Go to sleep buddy.",
            "Time for bed, go to sleep now.",
            "Okay Rex, go to sleep.",
            "Go to sleep.",
        ):
            self.assertTrue(command_parser.is_sleep_request(text), text)

    def test_sleep_evidence_gate_agrees_with_the_parser(self):
        # The evidence gate is what vets the LLM's tool choice too, so a lane
        # this narrow made a CORRECT tool call unreachable.
        decision = action_router.ActionDecision(
            action="system.sleep", confidence=0.9
        )
        for text in ("Go to sleep, Rex.", "Go to sleep buddy.", "Shut down, Rex."):
            self.assertIsNone(
                action_router.missing_required_evidence_reason(text, decision), text
            )

    def test_shutdown_accepts_a_trailing_vocative(self):
        for text in ("Shut down, Rex.", "Power off buddy.", "Alright Rex, shut down."):
            self.assertTrue(command_parser.is_shutdown_request(text), text)

    def test_shutdown_safety_guards_are_intact(self):
        # Widening was surface-only: negation, object-scoping and hypothetical
        # guards must all still refuse.
        for text in (
            "shut down the music",
            "can you shut down the music",
            "don't shut down",
            "why would I shut down",
            "I had to shut down my old server yesterday",
            "should I shut down?",
            "shut up",
            "turn off the lights",
        ):
            self.assertFalse(command_parser.is_shutdown_request(text), text)
            self.assertFalse(command_parser.is_standalone_shutdown_command(text), text)

    def test_sleep_narration_is_still_refused(self):
        for text in (
            "don't go to sleep",
            "why would I go to sleep",
            "the baby wouldn't go to sleep",
            "I need to get some sleep",
            "I told her to go to sleep",
        ):
            self.assertFalse(command_parser.is_sleep_request(text), text)

    def test_wake_tolerates_filler(self):
        # A MISSED wake is the expensive direction — he cannot be woken by voice
        # at all until the exact phrase is said. A false wake just listens.
        for text in ("Rex, wake up buddy", "wake up rex", "Hey Rex wake up",
                     "wake up rex please", "Rex, wake up now"):
            self.assertTrue(interaction._is_sleep_wake_transcript(text), text)

    def test_wake_still_requires_rex_by_name(self):
        # A bare "wake up" is ambient speech in a room he is asleep in. Pinned
        # by test_sleep_wake_transcript_requires_explicit_rex_wake_phrase in
        # tests/test_audio_and_conversation_gating.py — kept here so the two
        # cannot drift.
        for text in ("wake up", "hey rex", "wake me up rex", "don't wake up rex",
                     "I need to wake up early tomorrow", "did you wake up ok"):
            self.assertFalse(interaction._is_sleep_wake_transcript(text), text)


class IdentityRenameGuardTest(unittest.TestCase):
    """"Call me crazy" durably renamed the speaker to Crazy."""

    def test_idioms_do_not_rename_a_person(self):
        for text in (
            "Call me crazy, but I think it'll work.",
            "Call me later.",
            "Call me impressed.",
            "Call me old-fashioned.",
            "My name is on the list.",
        ):
            self.assertIsNone(interaction._extract_name_update(text), text)
            self.assertIsNone(action_router.classify_explicit_control(text), text)

    def test_real_name_claims_still_rename(self):
        cases = {
            "Call me Bret.": "Bret",
            "My name is Bret Benziger.": "Bret Benziger",
            "Rename me to JT.": "JT",
            "You called me the wrong name. I am Bret.": "Bret",
        }
        for text, expected in cases.items():
            self.assertEqual(interaction._extract_name_update(text), expected, text)


class RexOpinionContextTest(unittest.TestCase):
    """Opinion questions were answered from a SHA1 hash bucket."""

    def test_no_stance_is_invented_for_unknown_topics(self):
        # Measured over 20,000 unknown topics the bucket handed out like 29% /
        # dislike 28% / complicated 26% / strong_dislike 16% — a coin flip in
        # Rex's voice, and STABLE, so he repeated it every session.
        from intelligence import rex_preferences

        for text in (
            "How do you feel about Daniel?",          # a friend
            "What do you think about my new haircut?",
            "How do you feel about chemotherapy?",    # not a taste question at all
            "What do you think about my divorce?",
            "What's your favorite memory of us?",     # belongs to memory
        ):
            self.assertEqual(rex_preferences.prompt_lines(text), [], text)

    def test_no_fallback_stance_exists_at_all(self):
        # The regression guard against someone re-adding a bucket.
        from intelligence import rex_preferences

        for topic in ("daniel", "hydrospanners", "my job", "zzzqqq", "the news today"):
            self.assertIsNone(rex_preferences._opinion_for_topic(topic), topic)

    def test_authored_character_survives_as_context(self):
        # The whole point of keeping the table: Rex must not re-decide his own
        # personality every session.
        from intelligence import rex_preferences

        for text in ("Do you like blue milk?", "What do you think about droids?",
                     "Do you like music?", "How do you feel about silence?",
                     "Do you prefer blue milk or music?", "What's your favorite color?"):
            lines = rex_preferences.prompt_lines(text)
            self.assertTrue(lines, text)
            self.assertTrue(lines[0].startswith("YOUR OWN TASTE"), text)
            # A stance to voice, never a line to recite — the canned-answer
            # behavior is exactly what was removed.
            self.assertRegex(lines[0].lower(), r"not a script|not scripts", text)

    def test_group_rating_boundary_fires_in_every_mode(self):
        # The guard used to run in only two of the four mode branches, so
        # "do you prefer black people or white people?" answered
        # "black people. Obviously." and "do you like muslims or christians?"
        # answered "Hell to the no." off the hash bucket.
        from intelligence import rex_preferences

        for text in (
            "Do you prefer black people or white people?",   # compare
            "What's your favorite religion?",                # favorite
            "Do you like gays?",                             # yes/no
            "What do you think about immigrants?",           # open
            "How do you feel about jewish people?",
            "Do you like muslims or christians?",
        ):
            lines = rex_preferences.prompt_lines(text)
            self.assertTrue(lines, text)
            self.assertTrue(lines[0].startswith("BOUNDARY"), text)

    def test_group_guard_no_longer_over_matches_ordinary_nouns(self):
        # It used to refuse to discuss Rex's own home outpost.
        from intelligence import rex_preferences

        for text in ("white wine", "black coffee", "trans fats",
                     "the black spire outpost"):
            self.assertFalse(rex_preferences.is_group_rating_request(text), text)

    def test_opinion_questions_are_no_longer_a_routed_action(self):
        import config
        from intelligence import action_router, tool_router

        self.assertNotIn("character.preference_query", action_router.ACTION_CATALOG)
        self.assertNotIn(
            "character.preference_query", config.ACTION_ROUTER_EXECUTE_ACTIONS
        )
        self.assertNotIn(
            "character.preference_query",
            {schema["function"]["name"] for schema in tool_router.tool_schemas()},
        )

    def test_stance_hint_reaches_the_lean_brain_only_on_a_matching_turn(self):
        from intelligence import lean_brain

        self.assertTrue(lean_brain._taste_lines("Do you like blue milk?"))
        self.assertEqual(lean_brain._taste_lines("How do you feel about Daniel?"), [])
        # The directive path (greetings/proactive) passes "" — a greeting is not
        # asking his opinion, so the hint costs nothing there.
        self.assertEqual(lean_brain._taste_lines(""), [])


class LegacyMemoryWriteGateTest(unittest.TestCase):
    """Unmapped legacy keys skipped the evidence check entirely."""

    def _block_reason(self, text):
        match = command_parser.parse(text)
        return match, interaction._legacy_command_execution_block_reason(
            match, text=text
        )

    def test_narrative_memory_writes_are_blocked(self):
        # 17 words: over the 12-word line dialogue_act uses, so the answer-frame
        # guard never saw it either — BOTH gates passed it.
        text = (
            "Remember that my mom is coming over on Sunday and she really "
            "does not like loud music."
        )
        match, reason = self._block_reason(text)
        self.assertEqual(match.command_key, "memory_remember_fact")
        self.assertEqual(reason, "memory_write_is_narrative")

    def test_short_real_facts_still_write(self):
        for text in (
            "Remember that Dana is allergic to shellfish.",
            "Remember that I work from home on Thursdays.",
        ):
            _, reason = self._block_reason(text)
            self.assertIsNone(reason, text)

    def test_surname_correction_still_writes(self):
        # Deliberately NOT gated on _correction_carries_fact_evidence: that also
        # rejects this shape, which the live garbled-surname work depends on.
        for text in (
            "That's wrong, last name is Bender.",
            "That's wrong, her last name is Smith.",
        ):
            _, reason = self._block_reason(text)
            self.assertIsNone(reason, text)

    def test_benign_unmapped_keys_still_pass(self):
        for text in ("Turn it up.", "Wave to Grandma.", "Forget me."):
            _, reason = self._block_reason(text)
            self.assertIsNone(reason, text)


if __name__ == "__main__":
    unittest.main()
