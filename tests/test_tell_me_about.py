"""Tests for the "tell me about someone" pre-briefing flow.

Covers intent detection phrasings, tone/done parsing, the heuristic detail
classifier, the full multi-turn flow against a temp people.db (person row
pre-created, secondhand facts with gossip/kindness labels, relationship edge),
prompt-safety hedging for gossip, and the person_facts schema migration.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def _make_db(path: Path) -> None:
    from setup_assets import DB_SCHEMA

    with sqlite3.connect(path) as conn:
        conn.executescript(DB_SCHEMA)


class DetectionTests(unittest.TestCase):
    def test_positive_phrasings(self):
        from intelligence import tell_me_about

        cases = [
            ("I'd like to tell you about my coworker Daniel", "Daniel", "coworker"),
            ("We'd like to tell you about our friend Sarah", "Sarah", "friend"),
            ("Let me tell you about my father Jeff", "Jeff", "father"),
            ("I want to tell you about a friend of mine", None, "friend"),
            ("Can I tell you about someone?", None, None),
            ("I'm going to tell you about my neighbor Karen", "Karen", "neighbor"),
            ("You should know about Marcus", "Marcus", None),
            ("Let me fill you in on my sister", None, "sister"),
            ("I want to tell you some facts about my boss", None, "boss"),
        ]
        for text, name, rel in cases:
            parsed = tell_me_about.detect(text)
            self.assertTrue(parsed.is_tell_about, f"should detect: {text!r} ({parsed.reason})")
            self.assertEqual(parsed.name, name, f"name for {text!r}")
            self.assertEqual(parsed.relationship, rel, f"relationship for {text!r}")

    def test_gossip_hint_sets_default_kind(self):
        from intelligence import tell_me_about

        parsed = tell_me_about.detect("I've got some tea on my neighbor Karen")
        self.assertTrue(parsed.is_tell_about)
        self.assertTrue(parsed.gossip_hint)
        self.assertEqual(parsed.name, "Karen")

        parsed = tell_me_about.detect("Wanna hear some gossip about my boss?")
        self.assertTrue(parsed.is_tell_about, parsed.reason)
        self.assertTrue(parsed.gossip_hint)

    def test_negatives(self):
        from intelligence import tell_me_about

        negatives = [
            "I'd like to tell you about my weekend",
            "Let me tell you about my trip",
            "Did Jennifer tell you about Daniel?",
            "Did he fill you in on the project?",
            "I'd like you to meet my sister",   # live introduction, not a briefing
            "This is my coworker Daniel",        # live introduction
            "Let me tell you about my dog Rex",  # pet, not a person row
            "I want to tell you about it",
            "Never mind, forget it",
            "let me tell you about everything that happened",
        ]
        for text in negatives:
            parsed = tell_me_about.detect(text)
            self.assertFalse(parsed.is_tell_about, f"should NOT detect: {text!r}")


class ReplyParsingTests(unittest.TestCase):
    def test_tone_parsing(self):
        from intelligence import tell_me_about

        self.assertEqual(tell_me_about.parse_tone_reply("juicy gossip please"), "gossip")
        self.assertEqual(tell_me_about.parse_tone_reply("definitely the tea"), "gossip")
        self.assertEqual(tell_me_about.parse_tone_reply("just boring facts"), "fact")
        self.assertEqual(tell_me_about.parse_tone_reply("the basics"), "fact")
        self.assertIsNone(tell_me_about.parse_tone_reply("what do you mean?"))

    def test_done_detection(self):
        from intelligence import tell_me_about

        self.assertTrue(tell_me_about.is_done("that's it"))
        self.assertTrue(tell_me_about.is_done("That's all I've got"))
        self.assertTrue(tell_me_about.is_done("nothing else comes to mind"))
        self.assertTrue(tell_me_about.is_done("nope", allow_bare_no=True))
        self.assertFalse(tell_me_about.is_done("nope", allow_bare_no=False))
        self.assertFalse(tell_me_about.is_done("he also plays drums"))

    def test_acks_always_invite_more(self):
        """A bare 'Logged.' reads as Rex going silent (live-tested) — every
        ack must carry a continuation cue."""
        from intelligence import tell_me_about

        invite_tokens = ("?", "keep going", "continue", "what else", "you got")
        for kind in ("fact", "gossip"):
            for i in range(40):
                line = tell_me_about.ack_line("Daniel", kind, i).lower()
                self.assertTrue(
                    any(tok in line for tok in invite_tokens),
                    f"non-inviting ack: {line!r}",
                )

    def test_reanchor_parsing(self):
        from intelligence import tell_me_about

        self.assertTrue(tell_me_about.is_affirmative("yes"))
        self.assertTrue(tell_me_about.is_affirmative("yeah, still going"))
        self.assertTrue(tell_me_about.is_negative("nope"))
        self.assertTrue(tell_me_about.is_negative("moved on"))
        self.assertFalse(tell_me_about.is_affirmative("nope"))
        self.assertTrue(tell_me_about.mentions_subject("he plays drums", "Daniel"))
        self.assertTrue(tell_me_about.mentions_subject("Daniel hates jazz", "Daniel"))
        self.assertFalse(tell_me_about.mentions_subject("I was smiling at the dog", "Daniel"))

    def test_blank_offer_and_gender(self):
        from intelligence import tell_me_about

        self.assertTrue(tell_me_about.is_blank_offer("um, I don't know"))
        self.assertTrue(tell_me_about.is_blank_offer("nothing"))
        self.assertFalse(tell_me_about.is_blank_offer("he hates jazz"))
        self.assertEqual(tell_me_about.parse_gender("he's a man"), "man")
        self.assertEqual(tell_me_about.parse_gender("she's a girl"), "girl")
        self.assertIsNone(tell_me_about.parse_gender("about forty years old"))


class ExitAndPivotParsingTests(unittest.TestCase):
    def test_explicit_exits(self):
        from intelligence import tell_me_about

        positives = [
            "exit gossip mode",
            "stop gossiping",
            "enough about Joe",
            "that's enough about him",
            "we're done with Joe",
            "let's talk about something else",
            "change of subject",
            "moving on",
            "close the file",
            "stop",
        ]
        for text in positives:
            self.assertTrue(tell_me_about.is_exit(text), f"should exit: {text!r}")

        negatives = [
            "he wouldn't stop talking about volleyball",
            "stop me if you've heard this one, he once",
            "he quit his job at the file room",
        ]
        for text in negatives:
            self.assertFalse(tell_me_about.is_exit(text), f"should NOT exit: {text!r}")

    def test_request_to_rex_vs_detail(self):
        from intelligence import tell_me_about

        requests = [
            "Can you give me a recipe for or something to make with chicken noodle soup.",
            "what's the weather like today",
            "play some music",
            "hey rex, what time is it",
            "tell me a joke",
            "let's play jeopardy",
        ]
        for text in requests:
            self.assertTrue(
                tell_me_about.looks_like_request_to_rex(text, "Joe"),
                f"should pivot: {text!r}",
            )

        details = [
            "Can you believe he held a dildo out the car window",
            "he likes volleyball",
            "Joe is a 40 year old gay man",
            "what's funny is he collects spoons",
            "loves karaoke and hates jazz",
        ]
        for text in details:
            self.assertFalse(
                tell_me_about.looks_like_request_to_rex(text, "Joe"),
                f"should stay a detail: {text!r}",
            )


class HeuristicClassifierTests(unittest.TestCase):
    def test_heuristic_when_llm_disabled(self):
        import config
        from intelligence import tell_me_about

        with mock.patch.object(config, "TELL_ABOUT_CLASSIFY_LLM_ENABLED", False, create=True):
            mean = tell_me_about.classify_detail(
                "He's a terrible liar and he stole my sandwich", "Daniel", None
            )
            self.assertEqual(mean["kind"], "gossip")
            self.assertLess(mean["kindness"], 0)

            kind = tell_me_about.classify_detail(
                "She's the sweetest person I know", "Ana", None
            )
            self.assertEqual(kind["kind"], "fact")
            self.assertGreater(kind["kindness"], 0)

            labeled = tell_me_about.classify_detail(
                "He works at the hardware store", "Daniel", "gossip"
            )
            self.assertEqual(labeled["kind"], "gossip")


class FlowIntegrationTests(unittest.TestCase):
    """Drive interaction's handlers against a temp people.db."""

    def setUp(self):
        import config
        from memory import database as db
        from intelligence import interaction

        self._tmp = tempfile.TemporaryDirectory()
        db_path = Path(self._tmp.name) / "people.db"
        _make_db(db_path)
        self._patches = [
            mock.patch.object(db, "_DB_FILE", db_path),
            mock.patch.object(config, "TELL_ABOUT_CLASSIFY_LLM_ENABLED", False, create=True),
        ]
        for p in self._patches:
            p.start()
        interaction._pending_tell_about = None
        from memory import people as people_memory
        self.teller_id = people_memory.enroll_person("Bret")

    def tearDown(self):
        from intelligence import interaction

        interaction._pending_tell_about = None
        for p in self._patches:
            p.stop()
        self._tmp.cleanup()

    def test_full_flow_with_name_gossip_and_close(self):
        from intelligence import interaction
        from memory import facts as facts_memory
        from memory import people as people_memory
        from memory import database as db

        r1 = interaction._handle_tell_about_turn(
            "I'd like to tell you about my coworker Daniel", self.teller_id, "Bret"
        )
        self.assertIsNotNone(r1)
        self.assertEqual(interaction._pending_tell_about["step"], "awaiting_tone")
        daniel = people_memory.find_person_by_name("Daniel")
        self.assertIsNotNone(daniel)

        rel_row = db.fetchone(
            "SELECT relationship FROM person_relationships WHERE from_person_id = ? AND to_person_id = ?",
            (self.teller_id, daniel["id"]),
        )
        self.assertIsNotNone(rel_row)
        self.assertEqual(rel_row["relationship"], "coworker")

        r2 = interaction._handle_tell_about_turn("juicy gossip obviously", self.teller_id, "Bret")
        self.assertIsNotNone(r2)
        self.assertEqual(interaction._pending_tell_about["step"], "collecting")
        self.assertEqual(interaction._pending_tell_about["default_kind"], "gossip")

        r3 = interaction._handle_tell_about_turn(
            "Apparently he's a terrible driver and got fired from the bowling alley",
            self.teller_id,
            "Bret",
        )
        self.assertIsNotNone(r3)
        facts = facts_memory.get_facts(int(daniel["id"]))
        gossip = [f for f in facts if f.get("fact_kind") == "gossip"]
        self.assertEqual(len(gossip), 1)
        self.assertEqual(gossip[0]["source"], "secondhand")
        self.assertEqual(int(gossip[0]["told_by"]), int(self.teller_id))
        self.assertLess(float(gossip[0]["kindness"]), 0)

        r4 = interaction._handle_tell_about_turn("that's it", self.teller_id, "Bret")
        self.assertIsNotNone(r4)
        self.assertIn("Daniel", r4)
        self.assertIsNone(interaction._pending_tell_about)

    def test_no_name_path_asks_then_captures(self):
        from intelligence import interaction
        from memory import people as people_memory

        r1 = interaction._handle_tell_about_turn(
            "I want to tell you about someone", self.teller_id, "Bret"
        )
        self.assertIsNotNone(r1)
        self.assertEqual(interaction._pending_tell_about["step"], "awaiting_name")

        r2 = interaction._handle_tell_about_turn("His name is Marcus", self.teller_id, "Bret")
        self.assertIsNotNone(r2)
        self.assertEqual(interaction._pending_tell_about["step"], "awaiting_tone")
        self.assertIsNotNone(people_memory.find_person_by_name("Marcus"))

    def test_pointed_gender_question_stores_gender_fact(self):
        from intelligence import interaction
        from memory import facts as facts_memory
        from memory import people as people_memory

        interaction._handle_tell_about_turn(
            "Let me tell you about my father Jeff", self.teller_id, "Bret"
        )
        interaction._handle_tell_about_turn("boring facts", self.teller_id, "Bret")
        stall = interaction._handle_tell_about_turn("um, I don't know", self.teller_id, "Bret")
        self.assertIn("man or a woman", stall)
        self.assertEqual(interaction._pending_tell_about["last_pointed"], "gender")

        interaction._handle_tell_about_turn("he's a man", self.teller_id, "Bret")
        jeff = people_memory.find_person_by_name("Jeff")
        facts = {f["key"]: f for f in facts_memory.get_facts(int(jeff["id"]))}
        self.assertIn("gender", facts)
        self.assertEqual(facts["gender"]["value"], "man")

    def test_other_speaker_turns_are_not_consumed(self):
        from intelligence import interaction
        from memory import people as people_memory

        other_id = people_memory.enroll_person("Gloria")
        interaction._handle_tell_about_turn(
            "I'd like to tell you about my coworker Daniel", self.teller_id, "Bret"
        )
        self.assertIsNone(
            interaction._handle_tell_about_turn("juicy gossip", other_id, "Gloria")
        )
        self.assertEqual(interaction._pending_tell_about["step"], "awaiting_tone")

    def test_rex_request_releases_turn_and_closes_flow(self):
        """The chicken-soup regression: a request to Rex mid-collection must
        fall through to normal routing, not get filed as gossip."""
        from intelligence import interaction
        from memory import facts as facts_memory
        from memory import people as people_memory

        interaction._handle_tell_about_turn(
            "I'd like to tell you about my friend Joe", self.teller_id, "Bret"
        )
        interaction._handle_tell_about_turn("juicy gossip", self.teller_id, "Bret")
        interaction._handle_tell_about_turn(
            "he likes volleyball", self.teller_id, "Bret"
        )

        released = interaction._handle_tell_about_turn(
            "Can you give me a recipe for or something to make with chicken noodle soup.",
            self.teller_id,
            "Bret",
        )
        self.assertIsNone(released)
        self.assertIsNone(interaction._pending_tell_about)
        joe = people_memory.find_person_by_name("Joe")
        values = [f["value"] for f in facts_memory.get_facts(int(joe["id"]))]
        self.assertFalse(
            any("soup" in v for v in values),
            f"soup request must not be filed as gossip: {values}",
        )

    def test_exit_gossip_mode_closes_flow(self):
        from intelligence import interaction

        interaction._handle_tell_about_turn(
            "I'd like to tell you about my friend Joe", self.teller_id, "Bret"
        )
        interaction._handle_tell_about_turn("juicy gossip", self.teller_id, "Bret")
        interaction._handle_tell_about_turn(
            "he likes volleyball", self.teller_id, "Bret"
        )
        closed = interaction._handle_tell_about_turn(
            "exit gossip mode", self.teller_id, "Bret"
        )
        self.assertIsNotNone(closed)
        self.assertIn("Joe", closed)
        self.assertIsNone(interaction._pending_tell_about)

    def _open_collecting_flow(self, interaction):
        interaction._handle_tell_about_turn(
            "I'd like to tell you about my boss Daniel", self.teller_id, "Bret"
        )
        interaction._handle_tell_about_turn("just boring facts", self.teller_id, "Bret")
        interaction._handle_tell_about_turn(
            "Daniel is a supremely talented programmer", self.teller_id, "Bret"
        )

    def test_proactive_barge_triggers_reanchor_question(self):
        """The smile-reaction regression: consciousness barging in mid-briefing
        must be followed by 'still telling me about X?', and yes/no routes."""
        from intelligence import interaction

        self._open_collecting_flow(interaction)
        with mock.patch.object(interaction.speech_queue, "enqueue") as enq:
            interaction.tell_about_on_external_rex_line("smile_reaction")
            self.assertTrue(enq.called)
            spoken = enq.call_args.args[0]
        self.assertIn("Daniel", spoken)
        self.assertEqual(interaction._pending_tell_about["step"], "reanchor")

        # A second barge while already re-anchored must not stack questions.
        with mock.patch.object(interaction.speech_queue, "enqueue") as enq:
            interaction.tell_about_on_external_rex_line("idle_banter")
            self.assertFalse(enq.called)

        # "yes" resumes collecting.
        resumed = interaction._handle_tell_about_turn("yes", self.teller_id, "Bret")
        self.assertIsNotNone(resumed)
        self.assertEqual(interaction._pending_tell_about["step"], "collecting")

    def test_reanchor_no_closes_with_memory_banks_line(self):
        from intelligence import interaction

        self._open_collecting_flow(interaction)
        with mock.patch.object(interaction.speech_queue, "enqueue"):
            interaction.tell_about_on_external_rex_line("smile_reaction")
        closed = interaction._handle_tell_about_turn("nope", self.teller_id, "Bret")
        self.assertIsNotNone(closed)
        self.assertIn("memory banks", closed)
        self.assertIn("Daniel", closed)
        self.assertIsNone(interaction._pending_tell_about)

    def test_reanchor_detail_reply_keeps_collecting(self):
        from intelligence import interaction
        from memory import facts as facts_memory
        from memory import people as people_memory

        self._open_collecting_flow(interaction)
        with mock.patch.object(interaction.speech_queue, "enqueue"):
            interaction.tell_about_on_external_rex_line("smile_reaction")
        ack = interaction._handle_tell_about_turn(
            "yeah and he also coaches little league", self.teller_id, "Bret"
        )
        self.assertIsNotNone(ack)
        self.assertEqual(interaction._pending_tell_about["step"], "collecting")
        daniel = people_memory.find_person_by_name("Daniel")
        values = [f["value"] for f in facts_memory.get_facts(int(daniel["id"]))]
        self.assertTrue(any("little league" in v for v in values))

    def test_flow_own_lines_do_not_reanchor(self):
        from intelligence import interaction

        self._open_collecting_flow(interaction)
        with mock.patch.object(interaction.speech_queue, "enqueue") as enq:
            interaction.tell_about_on_external_rex_line("tell_about")
            self.assertFalse(enq.called)
        self.assertEqual(interaction._pending_tell_about["step"], "collecting")

    def test_no_flow_external_line_is_noop(self):
        from intelligence import interaction

        interaction._pending_tell_about = None
        with mock.patch.object(interaction.speech_queue, "enqueue") as enq:
            interaction.tell_about_on_external_rex_line("smile_reaction")
            self.assertFalse(enq.called)

    def test_told_about_teller_name_for_unmet_subject(self):
        from intelligence import interaction
        from memory import people as people_memory

        interaction._handle_tell_about_turn(
            "I'd like to tell you about my coworker Daniel", self.teller_id, "Bret"
        )
        daniel = people_memory.find_person_by_name("Daniel")
        self.assertEqual(interaction._told_about_teller_name(int(daniel["id"])), "Bret")
        # The teller himself has no secondhand file.
        self.assertIsNone(interaction._told_about_teller_name(int(self.teller_id)))


class FactSafetyTests(unittest.TestCase):
    def setUp(self):
        from memory import database as db

        self._tmp = tempfile.TemporaryDirectory()
        db_path = Path(self._tmp.name) / "people.db"
        _make_db(db_path)
        self._patch = mock.patch.object(db, "_DB_FILE", db_path)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def test_secondhand_does_not_overwrite_explicit(self):
        from memory import facts as facts_memory
        from memory import people as people_memory

        pid = people_memory.enroll_person("Daniel")
        facts_memory.add_fact(pid, "work", "job", "engineer", "explicit")
        facts_memory.add_fact(
            pid, "work", "job", "barista", "secondhand", told_by=None
        )
        facts = {f["key"]: f for f in facts_memory.get_facts(pid)}
        self.assertEqual(facts["job"]["value"], "engineer")

    def test_explicit_overwrites_secondhand(self):
        from memory import facts as facts_memory
        from memory import people as people_memory

        pid = people_memory.enroll_person("Daniel")
        facts_memory.add_fact(pid, "work", "job", "barista", "secondhand")
        facts_memory.add_fact(pid, "work", "job", "engineer", "explicit")
        facts = {f["key"]: f for f in facts_memory.get_facts(pid)}
        self.assertEqual(facts["job"]["value"], "engineer")

    def test_mean_gossip_prompt_format_forbids_reciting(self):
        from memory import facts as facts_memory
        from memory import people as people_memory

        pid = people_memory.enroll_person("Daniel")
        facts_memory.add_fact(
            pid,
            "story",
            "driving_record",
            "terrible driver, hit three mailboxes",
            "secondhand",
            fact_kind="gossip",
            kindness=-0.6,
            told_by=None,
        )
        fact = facts_memory.get_facts(pid)[0]
        line = facts_memory.format_fact_for_prompt(fact)
        self.assertIn("secondhand", line)
        self.assertIn("NEVER repeat", line)

    def test_neutral_gossip_prompt_format_hedges(self):
        from memory import facts as facts_memory
        from memory import people as people_memory

        pid = people_memory.enroll_person("Ana")
        facts_memory.add_fact(
            pid,
            "story",
            "karaoke",
            "secretly amazing at karaoke",
            "secondhand",
            fact_kind="gossip",
            kindness=0.4,
        )
        fact = facts_memory.get_facts(pid)[0]
        line = facts_memory.format_fact_for_prompt(fact)
        self.assertIn("gossip", line)
        self.assertNotIn("NEVER repeat", line)


class MigrationTests(unittest.TestCase):
    def test_old_person_facts_table_gains_new_columns(self):
        from memory import database as db

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "people.db"
            with sqlite3.connect(db_path) as conn:
                conn.execute("CREATE TABLE people (id INTEGER PRIMARY KEY, name TEXT)")
                conn.execute(
                    """CREATE TABLE person_facts (
                        id INTEGER PRIMARY KEY, person_id INTEGER, category TEXT,
                        key TEXT, value TEXT, confidence REAL, source TEXT,
                        created_at DATETIME, updated_at DATETIME
                    )"""
                )
                conn.execute(
                    """CREATE TABLE person_events (
                        id INTEGER PRIMARY KEY, person_id INTEGER,
                        event_name TEXT, event_date DATE, event_notes TEXT,
                        mentioned_at DATETIME, followed_up BOOLEAN DEFAULT FALSE,
                        follow_up_at DATETIME, outcome TEXT
                    )"""
                )
            with mock.patch.object(db, "_DB_FILE", db_path):
                db._run_migrations()
                with db.connection() as conn:
                    cols = {
                        row["name"]
                        for row in conn.execute("PRAGMA table_info(person_facts)")
                    }
        for col in ("fact_kind", "kindness", "told_by"):
            self.assertIn(col, cols)


class RequestPivotTests(unittest.TestCase):
    """The subject-pivot release must only fire on actual REQUESTS to Rex.

    Live-logged regression: "Rex is a very close friend of mine" — a statement
    ABOUT Rex offered as a detail — matched the bare vocative and closed the
    flow with zero details filed."""

    def test_statement_about_rex_is_not_a_pivot(self):
        from intelligence import tell_me_about

        for text in (
            "Rex is a very close friend of mine",
            "Rex really made Joy laugh last week",
            "rex would get along with Joy",
            "Rex",  # bare address: ambiguous — stay in the flow
        ):
            self.assertFalse(
                tell_me_about.looks_like_request_to_rex(text, "Karen"), text
            )

    def test_requests_still_pivot(self):
        from intelligence import tell_me_about

        for text in (
            "Rex, play some music",
            "hey rex what's the weather like",
            "can you give me a recipe for chicken soup",
            "rex can you play some jazz",
            "okay rex tell me a joke",
        ):
            self.assertTrue(
                tell_me_about.looks_like_request_to_rex(text, "Karen"), text
            )

    def test_subject_or_pronoun_marks_detail_even_in_request_shape(self):
        from intelligence import tell_me_about

        for text in (
            "can you believe he got arrested",
            "what's Karen's deal with karaoke",
        ):
            self.assertFalse(
                tell_me_about.looks_like_request_to_rex(text, "Karen"), text
            )


class FlowGuardTests(unittest.TestCase):
    """Proactive-speech suppression + the 30s spoken inactivity timeout."""

    def setUp(self):
        import time
        from intelligence import interaction

        self.time = time
        self.interaction = interaction
        interaction._pending_tell_about = None

    def tearDown(self):
        self.interaction._pending_tell_about = None

    def _open_flow(self, *, age_secs=0.0, subject="Joy", details=1):
        now = self.time.monotonic() - age_secs
        self.interaction._pending_tell_about = {
            "step": "collecting",
            "teller_id": 1,
            "teller_name": "Bret",
            "subject_id": 5,
            "subject_name": subject,
            "details_count": details,
            "created_at": now,
            "asked_at": now,
        }

    def test_flow_active_tracks_state_and_ttl(self):
        self.assertFalse(self.interaction.tell_about_flow_active())
        self._open_flow()
        self.assertTrue(self.interaction.tell_about_flow_active())
        self._open_flow(age_secs=9999)  # past TELL_ABOUT_STEP_TTL_SECS
        self.assertFalse(self.interaction.tell_about_flow_active())

    def test_proactive_speech_blocked_while_flow_open(self):
        from intelligence import speech_engine

        self._open_flow()
        with mock.patch("intelligence.consciousness._can_speak", return_value=True):
            self.assertFalse(speech_engine.can_proactive_speak())
            self.assertFalse(speech_engine.can_proactive_speak(salient=True))

    def test_timeout_closes_flow_out_loud(self):
        self._open_flow(age_secs=31.0)
        with (
            mock.patch.object(self.interaction.speech_queue, "is_speaking", return_value=False),
            mock.patch.object(self.interaction.output_gate, "is_busy", return_value=False),
            mock.patch.object(self.interaction.speech_queue, "enqueue") as enqueue,
            mock.patch.object(self.interaction.conv_memory, "add_to_transcript"),
            mock.patch.object(self.interaction.conv_log, "log_rex"),
        ):
            self.assertTrue(self.interaction._maybe_tell_about_timeout())
        self.assertIsNone(self.interaction._pending_tell_about)
        enqueue.assert_called_once()
        spoken_line = enqueue.call_args.args[0]
        self.assertIn("Joy", spoken_line)

    def test_timeout_waits_while_fresh_or_rex_speaking(self):
        # Fresh flow: nothing happens.
        self._open_flow(age_secs=5.0)
        self.assertFalse(self.interaction._maybe_tell_about_timeout())
        self.assertIsNotNone(self.interaction._pending_tell_about)
        # Stale, but Rex still holds the floor: hold off, keep the flow.
        self._open_flow(age_secs=31.0)
        with mock.patch.object(
            self.interaction.speech_queue, "is_speaking", return_value=True
        ):
            self.assertFalse(self.interaction._maybe_tell_about_timeout())
        self.assertIsNotNone(self.interaction._pending_tell_about)


if __name__ == "__main__":
    unittest.main()
