"""
Cross-session memory for recurring UNKNOWN voices (memory/voice_signatures.py)
and its integration with the session anonymous-speaker slots.

A persisted "voice signature" lets Rex recognize a voice he has no name for
across sessions, and links its samples to a person the moment they're named —
without ever creating a nameless person row.
"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from memory import database as db
from memory import voice_signatures as vs


def _unit(*idx) -> np.ndarray:
    """A normalized basis-ish vector that is distinct per index set."""
    v = np.zeros(256, dtype=np.float32)
    for i in idx:
        v[i] = 1.0
    return v / np.linalg.norm(v)


class _TempPeopleDb(unittest.TestCase):
    def setUp(self):
        from setup_assets import DB_SCHEMA

        self._tmp = tempfile.TemporaryDirectory()
        self._path = Path(self._tmp.name) / "people.db"
        with sqlite3.connect(self._path) as conn:
            conn.executescript(DB_SCHEMA)
        self._patch = mock.patch.object(db, "_DB_FILE", self._path)
        self._patch.start()
        vs.reset_table_cache()

    def tearDown(self):
        self._patch.stop()
        vs.reset_table_cache()
        self._tmp.cleanup()


class VoiceSignatureDataLayerTest(_TempPeopleDb):
    def test_record_and_match_same_voice(self):
        emb = _unit(1, 2, 3)
        sid = vs.record(emb, label="unknown_voice_1")
        self.assertIsNotNone(sid)
        m = vs.match(emb)
        self.assertIsNotNone(m)
        self.assertEqual(m["id"], sid)
        self.assertGreaterEqual(m["score"], 0.99)

    def test_distinct_voice_does_not_match(self):
        vs.record(_unit(1, 2, 3))
        self.assertIsNone(vs.match(_unit(10, 11, 12)))

    def test_bump_increments_turns(self):
        emb = _unit(4, 5)
        sid = vs.record(emb)
        vs.bump(sid, emb)
        vs.bump(sid, emb)
        self.assertEqual(vs.match(emb)["turns"], 3)

    def test_attach_person_links_signature(self):
        emb = _unit(6, 7)
        sid = vs.record(emb)
        vs.attach_person(sid, 42)
        self.assertEqual(vs.match(emb)["person_id"], 42)

    def test_disabled_flag_is_a_noop(self):
        import config

        with mock.patch.object(config, "VOICE_SIGNATURE_PERSIST_ENABLED", False):
            self.assertIsNone(vs.record(_unit(1)))
            self.assertIsNone(vs.match(_unit(1)))


class AnonymousSlotPersistenceTest(_TempPeopleDb):
    """The session slots persist/recognize/promote via voice_signatures."""

    def setUp(self):
        super().setUp()
        from intelligence import interaction as I

        self.I = I
        I._clear_anonymous_speaker_slots()
        # Make speaker_id.get_embedding return whatever embedding the test injects.
        self._emb = _unit(1, 2, 3)
        self._p = mock.patch.object(I.speaker_id, "get_embedding", side_effect=lambda a: self._emb)
        self._p.start()

    def tearDown(self):
        self._p.stop()
        self.I._clear_anonymous_speaker_slots()
        super().tearDown()

    def _speak(self, emb):
        self._emb = emb
        return self.I._resolve_anonymous_speaker_slot(
            np.zeros(16000, dtype=np.float32),
            person_id=None,
            raw_best_id=None,
            raw_best_name=None,
            raw_best_score=0.0,
        )

    def test_recurring_voice_persists_a_signature(self):
        emb = _unit(1, 2, 3)
        label1, _ = self._speak(emb)   # turn 1: creates slot, not yet persisted
        self.assertIsNone(vs.match(emb))
        label2, _ = self._speak(emb)   # turn 2: hits MIN_TURNS -> persisted
        self.assertEqual(label1, label2)
        self.assertIsNotNone(vs.match(emb))

    def test_voice_from_prior_session_is_recognized(self):
        emb = _unit(8, 9)
        vs.record(emb, label="from_last_time")  # simulate a prior session
        label, _ = self._speak(emb)
        slot = next(s for s in self.I._anonymous_speaker_slots if s.label == label)
        self.assertTrue(slot.recognized_across_sessions)
        self.assertIsNotNone(slot.signature_id)

    def test_promotion_links_signature_to_named_person(self):
        emb = _unit(3, 4, 5)
        self._speak(emb)
        label, _ = self._speak(emb)  # persisted now
        self.I._retire_anonymous_speaker_slot(label, person_id=99, person_name="Dana")
        self.assertEqual(vs.match(emb)["person_id"], 99)


if __name__ == "__main__":
    unittest.main()
