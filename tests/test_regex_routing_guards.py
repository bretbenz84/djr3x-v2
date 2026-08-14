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

import time
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


class HumorPerformanceToolMigrationTest(unittest.TestCase):
    """The classifier was the whole decision; now it is only a detector."""

    OFF_PATTERN = [
        ("humor.tell_joke", "Give me a zinger."),
        ("humor.tell_joke", "Know any good ones?"),
        ("humor.free_bit", "Hit me with something."),
        ("humor.free_bit", "Do that thing you do."),
        ("humor.roast", "Be mean to me for a second."),
        ("humor.roast", "Let me have it."),
        ("performance.dj_bit", "Give us some hype."),
        ("performance.dj_bit", "Work the crowd."),
        ("performance.body_beat", "Look like you just saw a ghost."),
        ("performance.mood_pose", "Pull a face."),
        ("performance.mood_pose", "Show me shocked."),
        ("performance.impersonate", "Sound like my brother."),
    ]

    REFUSALS = [
        ("humor.roast", "Don't roast me."),
        ("humor.roast", "They mock me at school for my accent."),
        ("humor.tell_joke", "Don't tell me a joke."),
        ("humor.tell_joke", "He'd always tell me a joke before bed."),
        ("humor.tell_joke", "That was a joke."),
        ("performance.mood_pose", "Don't be sad."),
        ("performance.body_beat", "You look sad today, buddy."),
        ("performance.body_beat", "You're sad."),
        ("performance.body_beat", "I agree with you."),
        ("performance.impersonate", "That was a good impression."),
        ("performance.impersonate", "Do you like my voice?"),
        ("performance.impersonate", "Don't impersonate me."),
    ]

    def test_evidence_gate_no_longer_caps_the_tool_router(self):
        # The gate used to re-run the very classifiers the migration demotes, so
        # a CORRECT tool call for an off-pattern request was vetoed by the regex
        # it was meant to replace — 9 of these 12 were blocked before.
        for action, text in self.OFF_PATTERN:
            decision = action_router.ActionDecision(action=action, confidence=0.95)
            self.assertIsNone(
                action_router.missing_required_evidence_reason(text, decision),
                f"{action} {text!r}",
            )

    def test_refusal_and_narration_guards_still_gate_the_tool_path(self):
        # Dropping the POSITIVE pattern must not drop the negative guards: a model
        # that decides to roast on "they mock me at school" has to be stopped.
        for action, text in self.REFUSALS:
            decision = action_router.ActionDecision(action=action, confidence=0.95)
            self.assertIsNotNone(
                action_router.missing_required_evidence_reason(text, decision),
                f"{action} {text!r}",
            )

    def test_online_the_classifier_only_detects(self):
        """A match hands the turn to the reply call instead of claiming it."""
        with mock.patch("intelligence.llm_compat.create") as create:
            for text in ("tell me a joke", "roast me", "say something funny",
                         "do a victory dance", "act surprised", "hype the room",
                         "impersonate Jimmy Carter"):
                self.assertEqual(
                    action_router.decide(text, {}).action, "conversation.reply", text
                )
            # And it still skips the ~0.8s JSON-prose router call, because the
            # regex match is itself proof the turn is actionable.
            self.assertFalse(create.called)

    def test_offline_the_classifier_still_claims_the_turn(self):
        # docs/tool_router_scope.md 2.4 — the local reply model gets no tools, so
        # with the link down the deterministic lane is all Rex has.
        with mock.patch("intelligence.connectivity.is_offline", return_value=True):
            for text, expected in (("tell me a joke", "humor.tell_joke"),
                                   ("do a victory dance", "performance.body_beat"),
                                   ("impersonate Jimmy Carter", "performance.impersonate")):
                self.assertEqual(action_router.decide(text, {}).action, expected, text)

    def test_kill_switch_restores_pre_migration_routing(self):
        import config

        with mock.patch.object(config, "TOOL_ROUTER_LIVE_ENABLED", False, create=True):
            self.assertEqual(
                action_router.decide("tell me a joke", {}).action, "humor.tell_joke"
            )

    def test_tool_calls_reach_the_real_executors(self):
        cases = [
            ("humor.tell_joke", {}, "give me a zinger", "perf"),
            ("humor.roast", {"target": "speaker"}, "be mean to me", "perf"),
            ("performance.body_beat", {"body_beat": "tiny_victory_dance"},
             "pull a face", "perf"),
            ("performance.mood_pose", {"mood": "surprised"}, "show me shocked", "perf"),
            ("performance.impersonate", {"target": "speaker"}, "sound like me", "imp"),
        ]
        for action, args, text, which in cases:
            with mock.patch.object(interaction, "_handle_router_performance_action",
                                   return_value="PERF") as perf, \
                 mock.patch.object(interaction, "_handle_router_impersonation",
                                   return_value="IMP") as imp, \
                 mock.patch.object(interaction.llm, "get_response", return_value="prose"), \
                 mock.patch.object(interaction, "_speak_blocking", return_value=True):
                out = interaction._execute_tool_routed_action(action, args, text, None)
            self.assertEqual(out, "PERF" if which == "perf" else "IMP", action)
            self.assertEqual(perf.called, which == "perf", action)
            self.assertEqual(imp.called, which == "imp", action)

    def test_invented_gesture_declines_instead_of_shrugging(self):
        # performance_plan coerces an unknown beat to thinking_tilt, so without
        # the arg check an invented pose would silently perform a head tilt.
        with mock.patch.object(interaction, "_handle_router_performance_action",
                               return_value="PERF") as perf, \
             mock.patch.object(interaction.llm, "get_response", return_value="prose"), \
             mock.patch.object(interaction, "_speak_blocking", return_value=True):
            out = interaction._execute_tool_routed_action(
                "performance.body_beat", {"body_beat": "spin the mystery servo"},
                "do the thing", None,
            )
        self.assertEqual(out, "prose")
        perf.assert_not_called()

    def test_dispatcher_applies_the_refusal_guard(self):
        with mock.patch.object(interaction, "_handle_router_performance_action",
                               return_value="PERF") as perf, \
             mock.patch.object(interaction.llm, "get_response", return_value="prose"), \
             mock.patch.object(interaction, "_speak_blocking", return_value=True):
            out = interaction._execute_tool_routed_action(
                "humor.roast", {"target": "speaker"},
                "They mock me at school for my accent.", None,
            )
        self.assertEqual(out, "prose")
        perf.assert_not_called()


class IntentClassifierDemotionTest(unittest.TestCase):
    """The keyword nets were the largest remaining first-claim lane."""

    # Each of these EXECUTED before the demotion, and each also satisfied the
    # evidence gate, so nothing downstream was going to stop them.
    MISFIRES = [
        ("did you see the game last night", "query_what_do_you_see"),
        ("I've been running a lot lately", "query_uptime"),
        ("help me understand what you meant", "query_memory"),
        ("my coffee's cold", "query_weather"),
    ]

    def test_loose_nets_are_handed_to_the_reply_call(self):
        from intelligence import intent_classifier

        for text, intent in self.MISFIRES:
            self.assertEqual(intent_classifier.classify_deterministic(text), intent, text)
            self.assertEqual(
                interaction._intent_execution_block_reason(intent, text=text),
                "tool_router_owns_action",
                text,
            )

    def test_zero_llm_handlers_keep_their_instant_lane(self):
        # time/date answer with NO model call, so demoting them would trade Rex's
        # fastest answer for a round trip and buy nothing.
        from intelligence import intent_classifier

        for text, intent in (("what time is it", "query_time"),
                             ("what's the date", "query_date")):
            self.assertEqual(intent_classifier.classify_deterministic(text), intent, text)
            self.assertIsNone(
                interaction._intent_execution_block_reason(intent, text=text), text
            )

    def test_offline_the_intent_lane_claims_as_before(self):
        from intelligence import intent_classifier

        with mock.patch("intelligence.connectivity.is_offline", return_value=True):
            for text in ("what do you see", "play some jazz", "what's your battery"):
                intent = intent_classifier.classify_deterministic(text)
                self.assertIsNone(
                    interaction._intent_execution_block_reason(intent, text=text), text
                )

    def test_owned_intents_no_longer_pay_the_json_router_call(self):
        # The stricter evidence regex used to decide whether to burn ~0.8s on the
        # way to the same conversation.reply.
        for text in ("something about the weather maybe", "play some jazz",
                     "what do you remember about me?", "what do you see?"):
            with mock.patch("intelligence.llm_compat.create") as create:
                decision = action_router.decide(text, {})
            self.assertEqual(decision.action, "conversation.reply", text)
            self.assertFalse(create.called, text)

    def test_who_is_speaking_tool_gets_biometric_evidence(self):
        # The tool path was the only caller not threading raw_best_* through, so
        # a tool-routed "who's speaking?" always took the no-match branch.
        with mock.patch.object(interaction, "_current_turn_speaker_evidence",
                               {"raw_best_id": 7, "raw_best_name": "Bret",
                                "raw_best_score": 0.91}), \
             mock.patch.object(interaction, "_handle_classified_intent",
                               return_value="ok") as handler, \
             mock.patch.object(interaction, "_speak_blocking", return_value=True):
            interaction._execute_tool_routed_action(
                "identity.who_is_speaking", {}, "who's speaking?", None
            )
        self.assertTrue(handler.called)
        kwargs = handler.call_args.kwargs
        self.assertEqual(kwargs.get("raw_best_id"), 7)
        self.assertEqual(kwargs.get("raw_best_name"), "Bret")
        self.assertAlmostEqual(kwargs.get("raw_best_score"), 0.91)

    def test_compound_game_turn_keeps_full_routing(self):
        # game.start is not a live tool yet, so skipping the router would answer
        # the weather half and silently drop the game.
        self.assertIsNone(
            action_router._deterministic_self_query_intent(
                "what's the weather? let's play trivia", {}
            )
        )


class MemoryWriteMigrationTest(unittest.TestCase):
    """The last regex-owned deletes, and the delete that had no confirmation."""

    def test_dismissal_idioms_never_reach_a_delete(self):
        # These are the measured blast radius, not hypotheticals: "Forget it,
        # I'll do it myself." produced the SUBSTRING search term "i'll", which
        # matches nearly every stored conversation summary, across ten tables,
        # with no undo and (until 2026-08-13) no confirmation.
        for text in ("Forget it, I'll do it myself.",
                     "Forget the traffic, we made it!"):
            self.assertIsNotNone(
                action_router.memory_boundary_refusal_reason(
                    text, "memory.forget_specific"), text)

    def test_housekeeping_verbs_are_not_memory_commands(self):
        # delete/remove/erase/wipe/clear are ordinary English about ordinary
        # objects; only "forget" is inherently about memory.
        for text in ("Remove the lid before microwaving.",
                     "Delete the extra whitespace in that file.",
                     "Clear the table when you're done."):
            self.assertIsNotNone(
                action_router.memory_boundary_refusal_reason(
                    text, "memory.forget_specific"), text)

    def test_real_forget_requests_pass_including_off_pattern(self):
        for text in ("Forget about my dog Scout.",
                     "forget what I told you about my job",
                     "scrub the dog from your memory",
                     "erase what you know about my ex"):
            self.assertIsNone(
                action_router.memory_boundary_refusal_reason(
                    text, "memory.forget_specific"), text)

    def test_discard_inversion_guard_survives_the_migration(self):
        # c7ef872's "don't forget = keep, not discard" must hold on every route.
        for text in ("Don't forget that we have dinner tomorrow.",
                     "I'll never forget that trip."):
            self.assertIsNotNone(
                action_router.memory_boundary_refusal_reason(
                    text, "memory.recent_discard"), text)
        for text in ("Forget what I just said.", "Don't remember that."):
            self.assertIsNone(
                action_router.memory_boundary_refusal_reason(
                    text, "memory.recent_discard"), text)

    def test_releasing_a_boundary_never_mints_one(self):
        self.assertIsNotNone(
            action_router.memory_boundary_refusal_reason(
                "you can ask about that again", "emotional.boundary"))

    def test_a_tool_routed_delete_arms_a_confirmation_instead_of_deleting(self):
        with mock.patch.object(interaction, "_execute_command",
                               return_value="deleted") as execute, \
             mock.patch.object(interaction, "_speak_blocking", return_value=True):
            interaction._pending_specific_forget = None
            line = interaction._execute_tool_routed_action(
                "memory.forget_specific", {"target": "my dog Scout"},
                "forget about my dog Scout", 1,
            )
        execute.assert_not_called()
        slot = interaction._pending_specific_forget
        self.assertIsNotNone(slot)
        self.assertEqual(slot["target"], "my dog Scout")
        self.assertIn("scout", (line or "").lower())

    def test_confirmed_delete_runs_the_same_executor_the_regex_ran(self):
        interaction._pending_specific_forget = {
            "person_id": 1, "target": "my dog Scout", "asked_at": time.monotonic(),
        }
        with mock.patch.object(interaction, "_execute_command",
                               return_value="Forgotten.") as execute, \
             mock.patch.object(interaction, "_speak_blocking", return_value=True):
            line = interaction._handle_specific_forget_confirmation("yes", 1)
        execute.assert_called_once()
        self.assertEqual(line, "Forgotten.")

    def test_forget_it_is_not_a_yes(self):
        # The single most likely way to mean the opposite at a delete prompt.
        interaction._pending_specific_forget = {
            "person_id": 1, "target": "my dog Scout", "asked_at": time.monotonic(),
        }
        with mock.patch.object(interaction, "_execute_command") as execute, \
             mock.patch.object(interaction, "_speak_blocking", return_value=True):
            interaction._handle_specific_forget_confirmation("forget it", 1)
        execute.assert_not_called()

    def test_boundary_write_is_the_models_call_online(self):
        from memory import boundaries as boundary_memory

        with mock.patch.object(boundary_memory, "apply_detected_boundary") as write, \
             mock.patch.object(interaction, "_apply_topic_boundary_side_effects") as ban, \
             mock.patch.object(interaction, "_speak_blocking", return_value=True):
            out = interaction._handle_conversation_boundary(
                1, "Don't ask me how I got it, long story.")
        write.assert_not_called()      # no durable row from a regex
        self.assertTrue(ban.called)    # the reversible ban still fires
        self.assertIsNone(out)         # and the turn reaches the reply call

    def test_offline_the_memory_lanes_claim_as_before(self):
        with mock.patch("intelligence.connectivity.is_offline", return_value=True):
            for text in ("Forget about my dog Scout.", "forget what I just said"):
                match = command_parser.parse(text)
                self.assertIsNone(
                    interaction._legacy_command_execution_block_reason(
                        match, text=text), text)


class GamesMigrationTest(unittest.TestCase):
    """command_parser was the only thing that ever started a game."""

    def test_off_pattern_start_requests_are_no_longer_vetoed(self):
        # Every one of these was a correct start request the positive-pattern
        # gate blocked, so the game simply never began.
        for text in ("quiz me", "how about a game", "game time",
                     "fire up trivia", "deal me in"):
            self.assertIsNone(
                action_router.game_request_refusal_reason(text, "game.start"), text)

    def test_off_pattern_stop_requests_are_no_longer_vetoed(self):
        for text in ("can we quit this", "I'm done with this game", "wrap it up"):
            self.assertIsNone(
                action_router.game_request_refusal_reason(text, "game.stop"), text)

    def test_narration_and_refusal_still_blocked(self):
        for action, text in (("game.start", "we played trivia last night"),
                             ("game.start", "he's playing games with my head"),
                             ("game.stop", "don't stop the game")):
            self.assertIsNotNone(
                action_router.game_request_refusal_reason(text, action),
                f"{action} {text!r}")

    def test_mid_game_stop_stays_deterministic(self):
        """The player's only exit must not depend on a model call.

        Mid-game the game owns the turn before any reply call happens, and if the
        model answers in prose the tool is dropped entirely — the player would be
        stuck inside the game with a chatty non-answer. Same rule the scope doc
        gives bare "stop" during motion.
        """
        match = command_parser.parse("stop the game")
        with mock.patch.object(interaction, "_game_active_for_router",
                               return_value=True):
            self.assertIsNone(
                interaction._legacy_command_execution_block_reason(
                    match, text="stop the game"))
        with mock.patch.object(interaction, "_game_active_for_router",
                               return_value=False):
            self.assertEqual(
                interaction._legacy_command_execution_block_reason(
                    match, text="stop the game"),
                "tool_router_owns_action")

    def test_start_game_is_handed_over_online_and_claimed_offline(self):
        match = command_parser.parse("let's play trivia")
        self.assertEqual(match.command_key, "start_game")
        self.assertEqual(
            interaction._legacy_command_execution_block_reason(
                match, text="let's play trivia"),
            "tool_router_owns_action")
        with mock.patch("intelligence.connectivity.is_offline", return_value=True):
            self.assertIsNone(
                interaction._legacy_command_execution_block_reason(
                    match, text="let's play trivia"))

    def test_a_hedged_right_answer_is_not_recorded_as_a_pass(self):
        # "I don't know, Paris?" is a guess with a disclaimer bolted to the front,
        # but the hedge pattern matches ANYWHERE, so the answer was thrown away.
        from features import jeopardy

        self.assertEqual(jeopardy.strip_pass_hedge("I don't know, Paris?"), "paris")
        self.assertEqual(jeopardy.strip_pass_hedge("no idea, maybe Lincoln"), "lincoln")

    def test_a_bare_pass_is_still_a_pass(self):
        # The residual may only ever PROMOTE to correct, never demote a shrug into
        # a wrong answer with a deduction.
        from features import jeopardy

        for text in ("I don't know", "no idea", "pass", "beats me"):
            self.assertEqual(jeopardy.strip_pass_hedge(text), "", text)


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
