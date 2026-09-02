"""
tests/test_identity_instrumentation.py — measurement-before-behaviour for speaker ID
(2026-09-02). Every audio turn logs one [identity_decision] record with the full
scoreboard, voiced/buffer seconds, the accept tier and decision outcome, the visual
latch, the engaged person, the continuity anchor age and the previous speaker; the
scan line prints every candidate with durations; every enrollment logs its provenance.
"""

import json
import logging
import unittest
from unittest import mock

import numpy as np

import config
from audio import speaker_id
from intelligence import interaction as I


def _burst(secs_voiced=1.0, secs_silence=1.0, rate=16000):
    rng = np.random.default_rng(1)
    voiced = (rng.standard_normal(int(secs_voiced * rate)) * 0.1).astype(np.float32)
    silence = np.zeros(int(secs_silence * rate), dtype=np.float32)
    return np.concatenate([silence, voiced, silence])


class DurationTest(unittest.TestCase):
    def test_voiced_excludes_the_padding(self):
        audio = _burst(1.0, 1.0)
        self.assertAlmostEqual(speaker_id.buffer_secs(audio), 3.0, places=2)
        self.assertLess(abs(speaker_id.voiced_secs(audio) - 1.0), 0.1)

    def test_silence_is_zero(self):
        self.assertEqual(speaker_id.voiced_secs(np.zeros(16000, dtype=np.float32)), 0.0)
        self.assertEqual(speaker_id.voiced_secs(None), 0.0)


class ScoreboardLogTest(unittest.TestCase):
    RANKED = [(7, "PJ Thomas", 0.748, 5), (1, "Bret Benziger", 0.733, 8),
              (8, "Jade Smith", 0.383, 5), (4, "Jeremy Thomas", 0.310, 1)]

    def test_every_row_and_the_durations_are_logged(self):
        with mock.patch.object(speaker_id.logger, "info") as info:
            speaker_id._log_scoreboard(self.RANKED, voiced=1.23, buffer=2.9)
        rendered = info.call_args[0][0] % info.call_args[0][1:]
        for name in ("PJ Thomas#7=0.748(5p)", "Bret Benziger#1=0.733(8p)",
                     "Jade Smith#8=0.383(5p)", "Jeremy Thomas#4=0.310(1p)"):
            self.assertIn(name, rendered)
        self.assertIn("voiced=1.23s", rendered)
        self.assertIn("buffer=2.90s", rendered)

    def test_old_callers_without_durations_still_work(self):
        with mock.patch.object(speaker_id.logger, "info") as info:
            speaker_id._log_scoreboard(self.RANKED)
        rendered = info.call_args[0][0] % info.call_args[0][1:]
        self.assertNotIn("voiced=", rendered)
        self.assertIn("Jeremy Thomas#4", rendered)


class EnrollProvenanceTest(unittest.TestCase):
    def test_enroll_logs_scoreboard_and_durations_before_storing(self):
        vec = (np.ones(192) / np.sqrt(192)).astype(np.float32)
        order = []
        with mock.patch.object(speaker_id, "get_embedding", return_value=vec), \
             mock.patch.object(speaker_id, "rank_embedding",
                               return_value=[(1, "Bret Benziger", 0.604, 8), (7, "PJ Thomas", 0.552, 5)]), \
             mock.patch.object(speaker_id.people, "add_biometric",
                               side_effect=lambda *a, **k: order.append("stored")), \
             mock.patch.object(speaker_id.logger, "info",
                               side_effect=lambda msg, *a: order.append(msg % a)):
            ok = speaker_id.enroll_voice(7, _burst(1.5, 0.5), source="introduction",
                                         transcript="I didn't leave. I just turned around.")
        self.assertTrue(ok)
        prov = [o for o in order if "enroll provenance" in o]
        self.assertEqual(len(prov), 1)
        self.assertLess(order.index(prov[0]), order.index("stored"), "provenance must precede the write")
        self.assertIn("source=introduction", prov[0])
        self.assertIn("Bret Benziger#1=0.604(8p)", prov[0])
        self.assertIn("words=7", prov[0])
        self.assertIn("voiced=", prov[0])

    def test_rank_embedding_matches_rank_speakers(self):
        vec = (np.ones(192) / np.sqrt(192)).astype(np.float32)
        rows = [{"person_id": 1, "encoding": vec.tobytes()}]
        with mock.patch.object(speaker_id, "get_embedding", return_value=vec), \
             mock.patch.object(speaker_id.db, "fetchall", return_value=rows), \
             mock.patch.object(speaker_id.people, "get_person", return_value={"name": "Bret"}):
            a = speaker_id.rank_speakers(np.zeros(16000, dtype=np.float32))
            b = speaker_id.rank_embedding(vec)
        self.assertEqual(a, b)
        self.assertEqual(a[0][0], 1)


class DecisionPayloadTest(unittest.TestCase):
    def _payload(self, **kw):
        base = dict(
            turn_id=8, text="Yes, we just talked about that.", text_input=False,
            transcript_trusted=True, scan_secs={"buffer": 2.95, "voiced": 1.6},
            scoreboard=[(7, "PJ Thomas", 0.748, 5), (1, "Bret Benziger", 0.733, 8)],
            raw_best_id=7, raw_best_name="PJ Thomas", speaker_score=0.748,
            speaker_margin=0.016, required_margin=0.07, accept_tier=None,
            decision_outcome="off_screen_unknown", identity_resolution=None,
            visible_known_ids=[1], visual_latch=None, mouth_still=False,
            engaged={"person_id": 1, "name": "Bret Benziger"},
            continuity_anchor_age=41.2,
            previous_speaker={"person_id": 1, "person_name": "Bret Benziger", "label": None, "at": 0.0},
            person_id=None, person_name=None, anonymous_label="unknown_voice_1",
            anonymous_score=None, off_camera_unknown=True,
        )
        base.update(kw)
        return I._identity_decision_payload(**base)

    def test_field_case_record_is_complete_and_serializable(self):
        p = self._payload()
        json.dumps(p)   # one INFO line — must serialize
        self.assertEqual(p["words"], 6)
        self.assertEqual(p["voiced_secs"], 1.6)
        self.assertEqual([r["person_id"] for r in p["scoreboard"]], [7, 1])
        self.assertEqual(p["scoreboard"][1]["prints"], 8)
        self.assertEqual(p["margin"], 0.016)
        self.assertEqual(p["required_margin"], 0.07)
        self.assertIsNone(p["accept_tier"])
        self.assertEqual(p["decision"], "off_screen_unknown")
        self.assertEqual(p["visible_known_ids"], [1])
        self.assertEqual(p["engaged"]["person_id"], 1)
        self.assertEqual(p["previous_speaker"]["person_id"], 1)
        self.assertGreater(p["previous_speaker"]["age_secs"], 0)
        self.assertEqual(p["final"]["label"], "unknown_voice_1")
        self.assertTrue(p["final"]["off_camera_unknown"])

    def test_visual_latch_is_summarised(self):
        import time
        p = self._payload(visual_latch={"person_db_id": 1, "confidence": 0.8, "at": time.time() - 1.0},
                          mouth_still=False)
        self.assertEqual(p["visual_latch"]["person_id"], 1)
        self.assertGreaterEqual(p["visual_latch"]["age_secs"], 0.9)

    def test_text_input_turn_has_no_audio_durations(self):
        p = self._payload(text_input=True, scan_secs={})
        self.assertTrue(p["text_input"])
        self.assertEqual(p["voiced_secs"], 0.0)


class ProcessAudioRecordsDurationsTest(unittest.TestCase):
    def test_scan_secs_and_full_ranking_are_kept(self):
        from audio import transcription
        ranked = [(7, "PJ", 0.748, 5), (1, "Bret", 0.733, 8), (8, "Jade", 0.383, 5), (4, "Jeremy", 0.31, 1)]
        with mock.patch.object(transcription, "transcribe", return_value="hi there"), \
             mock.patch.object(speaker_id, "rank_speakers", return_value=ranked), \
             mock.patch.object(speaker_id.logger, "info"):
            I._process_audio(_burst(1.0, 0.5))
        self.assertEqual(len(I._last_scan_ranked), 4)
        self.assertAlmostEqual(I._last_scan_secs["buffer"], 2.0, places=2)
        self.assertLess(abs(I._last_scan_secs["voiced"] - 1.0), 0.1)


if __name__ == "__main__":
    unittest.main()
