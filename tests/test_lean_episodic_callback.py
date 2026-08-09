"""Episodic shared-memory callback on the LEAN reply path (2026-08-08).

The recall Phase 2 "we have history" hook ("I made you laugh", "we played
trivia") was injected only by llm.assemble_system_prompt — the CLASSIC prompt.
With the lean brain live as the primary voice, the diary reached replies only on
the classic fallback path, so the feature almost never fired.

Fix: lean_brain._person_lines now calls the SAME llm._pick_episodic_callback
(shared probability roll + shared once-per-session dedup set) and renders the
same SHARED-MEMORY HOOK line. Pins:
  * the hook renders on a reply turn when a callback is picked;
  * the directive path (user_text="") never rolls the callback;
  * LEAN_EPISODIC_CALLBACK_ENABLED=False disables it;
  * a picker exception never breaks the prompt;
  * the lean path uses llm's picker (shared dedup), not a private copy.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config
from intelligence import lean_brain, llm
from memory import database as db


MEMORY = "I played Trivia with Test User — scored 4 out of 5"


class _TempDb(unittest.TestCase):
    def setUp(self):
        from setup_assets import DB_SCHEMA
        self._tmp = tempfile.TemporaryDirectory()
        self._path = Path(self._tmp.name) / "people.db"
        with sqlite3.connect(self._path) as conn:
            conn.executescript(DB_SCHEMA)
            conn.execute("INSERT INTO people (id, name) VALUES (1, 'Test User')")
        self._p = mock.patch.object(db, "_DB_FILE", self._path)
        self._p.start()

    def tearDown(self):
        self._p.stop()
        self._tmp.cleanup()


class LeanEpisodicCallbackTest(_TempDb):
    def test_reply_turn_renders_shared_memory_hook(self):
        with mock.patch.object(llm, "_pick_episodic_callback", return_value=MEMORY) as picker:
            lines = lean_brain._person_lines(1, "want another round of trivia?")
        picker.assert_called_once()
        self.assertEqual(picker.call_args.args, (1,))
        hook = [l for l in lines if l.startswith("SHARED-MEMORY HOOK")]
        self.assertEqual(len(hook), 1)
        self.assertIn(MEMORY, hook[0])

    def test_directive_path_never_rolls_the_callback(self):
        # user_text="" is the stream_directive path — the proactive cue owns that
        # turn; a memory hook must not compete with it.
        with mock.patch.object(llm, "_pick_episodic_callback", return_value=MEMORY) as picker:
            lines = lean_brain._person_lines(1, "")
        picker.assert_not_called()
        self.assertFalse(any(l.startswith("SHARED-MEMORY HOOK") for l in lines))

    def test_kill_switch_disables_the_hook(self):
        with mock.patch.object(config, "LEAN_EPISODIC_CALLBACK_ENABLED", False), \
             mock.patch.object(llm, "_pick_episodic_callback", return_value=MEMORY) as picker:
            lines = lean_brain._person_lines(1, "hey rex")
        picker.assert_not_called()
        self.assertFalse(any(l.startswith("SHARED-MEMORY HOOK") for l in lines))

    def test_no_pick_means_no_hook_line(self):
        with mock.patch.object(llm, "_pick_episodic_callback", return_value=None):
            lines = lean_brain._person_lines(1, "hey rex")
        self.assertFalse(any(l.startswith("SHARED-MEMORY HOOK") for l in lines))

    def test_picker_exception_is_swallowed(self):
        with mock.patch.object(llm, "_pick_episodic_callback", side_effect=RuntimeError("boom")):
            lines = lean_brain._person_lines(1, "hey rex")
        self.assertTrue(any(l.startswith("You're talking with") for l in lines))
        self.assertFalse(any(l.startswith("SHARED-MEMORY HOOK") for l in lines))

    def test_shared_session_dedup_with_classic_path(self):
        # The lean path must consume llm's picker (and therefore its
        # _episodic_callbacks_used_this_session set) — not keep a private copy.
        # Simulate the classic path having already surfaced the memory this
        # session: the real picker then returns None for it, and lean shows no hook.
        with mock.patch.object(config, "EPISODIC_RECALL_ENABLED", True), \
             mock.patch.object(llm.random, "random", return_value=0.0), \
             mock.patch("memory.episodic_recall.person_episodes", return_value=[MEMORY]):
            llm._episodic_callbacks_used_this_session.add(f"1:{MEMORY}")
            try:
                lines = lean_brain._person_lines(1, "hey rex")
            finally:
                llm._episodic_callbacks_used_this_session.discard(f"1:{MEMORY}")
        self.assertFalse(any(l.startswith("SHARED-MEMORY HOOK") for l in lines))


if __name__ == "__main__":
    unittest.main()
