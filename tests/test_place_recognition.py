"""
Unit tests for perception.place_recognition (Visual Place Recognition).

No model and no camera: the injected ``embed_fn`` is the identity, so "frames" ARE
unit-ish vectors and the exact cosine geometry is controlled by the test. Every case
uses an in-memory ``places.db`` and a fake world_state/emit sink, so the suite is fast,
hermetic, and cannot perturb the global world_state singleton the rest of the suite uses.

Covers: top-k scoring + classification, temporal-hysteresis flip, the motion freeze
gate, the once-per-episode unknown_place event, the enrollment state machine (heading
diversity, auto-commit, timeout commit/abort), duplicate detect + confirm merge/new,
the per-place cap, person-occlusion skip, incremental refresh, persistence, and model_tag
isolation.
"""

import math
import unittest

import numpy as np

from perception import place_recognition as P
from perception.place_recognition import MotionState, PlaceRecognizer, _circular_sep

E = np.eye(16, dtype=np.float32)


def q_cos(i: int, c: float, perp: int = 10) -> np.ndarray:
    """Unit vector with cosine ``c`` to basis vector ``i`` (perp component along ``perp``)."""
    return (c * E[i] + math.sqrt(max(0.0, 1.0 - c * c)) * E[perp]).astype(np.float32)


class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def adv(self, d):
        self.t += d
        return self.t


class _FakeWorldState:
    def __init__(self):
        self.value = None  # mirrors world_state._DEFAULTS["current_place"]

    def update(self, field, value):
        self.value = value

    def get(self, field):
        return self.value


class PlaceRecognitionTest(unittest.TestCase):
    def _make(self, db_path=":memory:", model_tag="t", **kw):
        clock = _Clock()
        events = []
        boxes = {"heading": None, "motion": MotionState(), "occ": 0.0}
        ws = _FakeWorldState()
        pr = PlaceRecognizer(
            embed_fn=lambda f: np.asarray(f, dtype=np.float32),
            get_heading=lambda: boxes["heading"],
            get_motion_state=lambda: boxes["motion"],
            get_person_occlusion=lambda: boxes["occ"],
            world_state=ws,
            emit_event=lambda n, p: events.append((n, p)),
            db_path=db_path,
            model_tag=model_tag,
            clock=clock,
            **kw,
        )
        self.addCleanup(pr.close)
        return pr, clock, events, boxes, ws

    def _n_events(self, events, name):
        return sum(1 for n, _ in events if n == name)

    # ── geometry ──
    def test_circular_sep_wraps(self):
        self.assertAlmostEqual(_circular_sep(350, 10), 20.0)
        self.assertAlmostEqual(_circular_sep(10, 350), 20.0)
        self.assertAlmostEqual(_circular_sep(0, 180), 180.0)

    # ── scoring / classification ──
    def test_topk_mean_and_classification(self):
        pr, clk, ev, box, ws = self._make()
        pr.enroll_from_frames("office", [E[0]] * 5)
        pr.enroll_from_frames("living", [E[1]] * 5)
        r = pr.score_frame(E[0])
        self.assertEqual(r.best.name, "office")
        self.assertAlmostEqual(r.best.score, 1.0, places=5)
        self.assertEqual(r.classification, P.CONFIDENT)
        self.assertEqual(pr.score_frame(q_cos(0, 0.75)).classification, P.TENTATIVE)
        self.assertEqual(pr.score_frame(E[7]).classification, P.UNKNOWN)

    def test_empty_gallery_is_unknown(self):
        pr, *_ = self._make()
        r = pr.score_frame(E[0])
        self.assertIsNone(r.best)
        self.assertEqual(r.classification, P.UNKNOWN)

    # ── hysteresis + world_state ──
    def test_hysteresis_flip_writes_world_state(self):
        pr, clk, ev, box, ws = self._make()
        pr.enroll_from_frames("office", [E[0]] * 5)
        box["motion"] = MotionState(wheels_moving=True)
        for _ in range(10):
            clk.adv(2.0)
            pr.observe(E[0])
            if (ws.get("current_place") or {}).get("name") == "office":
                break
        cp = ws.get("current_place")
        self.assertEqual(cp["name"], "office")
        self.assertGreater(cp["score"], 0.99)
        self.assertEqual(cp["place_id"], pr.current_place()["place_id"])

    def test_single_frame_never_flips(self):
        pr, clk, ev, box, ws = self._make()
        pr.enroll_from_frames("office", [E[0]] * 5)
        box["motion"] = MotionState(wheels_moving=True)
        clk.adv(2.0)
        pr.observe(E[0])  # one confident frame < majority(3)
        self.assertIsNone(pr.current_place())

    def test_motion_freeze_then_unfreeze(self):
        pr, clk, ev, box, ws = self._make()
        pr.enroll_from_frames("office", [E[0]] * 5)
        pr.enroll_from_frames("living", [E[1]] * 5)
        box["motion"] = MotionState(wheels_moving=True)
        for _ in range(10):
            clk.adv(2.0)
            pr.observe(E[0])
            if (ws.get("current_place") or {}).get("name") == "office":
                break  # break at the confirming frame -> no motion after confirm
        box["motion"] = MotionState()  # stationary + accel quiet -> frozen
        for _ in range(6):
            clk.adv(2.0)
            pr.observe(E[1])
        self.assertEqual(ws.get("current_place")["name"], "office")
        box["motion"] = MotionState(wheels_moving=True)  # moved -> can re-evaluate
        for _ in range(6):
            clk.adv(2.0)
            pr.observe(E[1])
        self.assertEqual(ws.get("current_place")["name"], "living")

    def test_unknown_place_fires_once_and_rearms(self):
        pr, clk, ev, box, ws = self._make()
        pr.enroll_from_frames("office", [E[0]] * 5)
        box["motion"] = MotionState(wheels_moving=True)
        for _ in range(5):
            clk.adv(2.0)
            pr.observe(E[0])  # confirm + arm
        ev.clear()
        for _ in range(8):
            clk.adv(2.0)
            pr.observe(E[7])  # 8 unknowns while moving
        self.assertEqual(self._n_events(ev, P.EVENT_UNKNOWN_PLACE), 1)
        for _ in range(5):
            clk.adv(2.0)
            pr.observe(E[7])
        self.assertEqual(self._n_events(ev, P.EVENT_UNKNOWN_PLACE), 1)  # not re-fired
        for _ in range(5):
            clk.adv(2.0)
            pr.observe(E[0])  # confident -> re-arm
        ev.clear()
        for _ in range(8):
            clk.adv(2.0)
            pr.observe(E[7])
        self.assertEqual(self._n_events(ev, P.EVENT_UNKNOWN_PLACE), 1)

    def test_unknown_place_suppressed_while_stationary(self):
        pr, clk, ev, box, ws = self._make()
        pr.enroll_from_frames("office", [E[0]] * 5)
        box["motion"] = MotionState(wheels_moving=True)
        for _ in range(10):
            clk.adv(2.0)
            pr.observe(E[0])
            if pr.current_place():  # break at the confirm -> no motion after it
                break
        box["motion"] = MotionState()  # parked, quiet since the confirm
        ev.clear()
        for _ in range(12):
            clk.adv(2.0)
            pr.observe(E[7])
        self.assertEqual(self._n_events(ev, P.EVENT_UNKNOWN_PLACE), 0)

    # ── enrollment state machine ──
    def test_enroll_heading_diversity_and_autocommit(self):
        pr, clk, ev, box, ws = self._make()
        pr.enroll("garage")
        self.assertEqual(pr.state, P.COLLECTING)
        for i in range(12):
            box["heading"] = (i * 40.0) % 360  # spaced 40 > 35
            clk.adv(0.1)
            pr.observe(E[2] + 0.001 * E[3])
            if pr.state == P.IDLE:
                break
        self.assertEqual(pr.state, P.IDLE)
        self.assertEqual(self._n_events(ev, P.EVENT_PLACE_ENROLLED), 1)
        self.assertEqual(ws.get("current_place")["name"], "garage")

    def test_enroll_rejects_headings_too_close(self):
        pr, clk, ev, box, ws = self._make()
        pr.enroll("den")
        box["heading"] = 100.0
        pr.observe(E[2])
        box["heading"] = 110.0  # 10 deg < 35 -> reject
        pr.observe(E[2])
        self.assertEqual(len(pr._enroll.vectors), 1)

    def test_enroll_timeout_commits_when_enough(self):
        pr, clk, ev, box, ws = self._make()
        pr.enroll("attic")
        box["heading"] = None
        for _ in range(4):  # 4 frames (>=3, <8), time-sep fallback
            clk.adv(5.0)
            pr.observe(E[6])
        clk.adv(100.0)
        pr.tick()
        self.assertEqual(self._n_events(ev, P.EVENT_PLACE_ENROLLED), 1)
        self.assertIn("attic", pr.place_names())

    def test_enroll_timeout_aborts_when_too_few(self):
        pr, clk, ev, box, ws = self._make()
        pr.enroll("closet")
        box["heading"] = None
        pr.observe(E[5])
        clk.adv(5.0)
        pr.observe(E[5])  # only 2 frames
        clk.adv(100.0)
        pr.tick()
        self.assertEqual(self._n_events(ev, P.EVENT_ENROLLMENT_FAILED), 1)
        self.assertNotIn("closet", pr.place_names())  # empty created row dropped
        self.assertEqual(pr.state, P.IDLE)

    def test_cancel_drops_empty_created_place(self):
        pr, clk, ev, box, ws = self._make()
        pr.enroll("temp")
        pr.cancel_enrollment()
        self.assertEqual(pr.state, P.IDLE)
        self.assertNotIn("temp", pr.place_names())

    # ── duplicate detection ──
    def test_duplicate_confirm_merges(self):
        pr, clk, ev, box, ws = self._make()
        pr.enroll_from_frames("office", [E[0]] * 5)
        pr.enroll("office_dup")
        box["heading"] = None
        for _ in range(9):
            clk.adv(5.0)
            pr.observe(E[0])
            if pr.state == P.CONFIRMING:
                break
        self.assertTrue(any(n == P.EVENT_POSSIBLE_DUPLICATE for n, _ in ev))
        self.assertEqual(pr.state, P.CONFIRMING)
        self.assertTrue(pr.confirm_duplicate(True))
        self.assertNotIn("office_dup", pr.place_names())
        self.assertEqual(pr.state, P.IDLE)

    def test_duplicate_reject_keeps_new(self):
        pr, clk, ev, box, ws = self._make()
        pr.enroll_from_frames("office", [E[0]] * 5)
        pr.enroll("office_dup")
        box["heading"] = None
        for _ in range(9):
            clk.adv(5.0)
            pr.observe(E[0])
            if pr.state == P.CONFIRMING:
                break
        self.assertTrue(pr.confirm_duplicate(False))
        self.assertIn("office_dup", pr.place_names())

    # ── cap / occlusion / refresh ──
    def test_per_place_cap(self):
        pr, clk, ev, box, ws = self._make()
        res = pr.enroll_from_frames("big", [E[0] + 0.01 * E[(i % 5) + 8] for i in range(20)])
        self.assertEqual(res.provided, 20)
        self.assertEqual(res.committed, 15)

    def test_occlusion_skips_query_and_collect(self):
        pr, clk, ev, box, ws = self._make()
        pr.enroll_from_frames("office", [E[0]] * 5)
        box["motion"] = MotionState(wheels_moving=True)
        box["occ"] = 0.5  # > 0.35
        for _ in range(5):
            clk.adv(5.0)
            r = pr.observe(E[0])
        self.assertTrue(r.skipped and r.skip_reason == "person_occlusion")
        self.assertIsNone(pr.current_place())
        pr.enroll("hall")
        box["occ"] = 0.9
        box["heading"] = None
        clk.adv(5.0)
        pr.observe(E[3])
        self.assertEqual(len(pr._enroll.vectors), 0)

    def test_incremental_refresh_band(self):
        # Needs a SECOND gallery: refresh is only safe when there is a runner-up to
        # discriminate against (see test_solo_gallery_never_refreshes).
        pr, clk, ev, box, ws = self._make()
        pr.enroll_from_frames("office", [E[0]] * 5)
        pr.enroll_from_frames("attic", [E[5]] * 5)      # far from office; a real runner-up
        box["motion"] = MotionState(wheels_moving=True)
        for _ in range(5):
            clk.adv(2.0)
            pr.observe(E[0])  # confirm office
        pid = pr._name_to_id["office"]
        n0 = int((pr._emb_pids == pid).sum())
        clk.adv(2.0)
        r = pr.observe(q_cos(0, 0.75))  # in [0.70, 0.78] band for the believed place
        self.assertEqual(r.classification, P.TENTATIVE)
        self.assertEqual(int((pr._emb_pids == pid).sum()), n0 + 1)
        clk.adv(2.0)
        pr.observe(E[0])  # confident, out of band -> no append
        self.assertEqual(int((pr._emb_pids == pid).sum()), n0 + 1)

    def test_solo_gallery_never_refreshes(self):
        # THE CONTAMINATION ENGINE (field 2026-07-25): with one room enrolled the
        # margin guard was satisfied by default (`len(scores) < 2`), so every in-band
        # frame was appended no matter where the robot actually was. 12 of the
        # workshop's 15 embeddings turned out to be dining-room views, dragging the
        # galleries together until a DIFFERENT room scored 0.91 against "workshop".
        # With nothing to compare against there is no evidence the frame belongs here.
        pr, clk, ev, box, ws = self._make()
        pr.enroll_from_frames("workshop", [E[0]] * 5)
        box["motion"] = MotionState(wheels_moving=True)
        for _ in range(6):
            clk.adv(2.0)
            pr.observe(E[0])
        pid = pr._name_to_id["workshop"]
        n0 = int((pr._emb_pids == pid).sum())
        clk.adv(2.0)
        pr.observe(q_cos(0, 0.75))      # squarely inside the refresh band
        self.assertEqual(int((pr._emb_pids == pid).sum()), n0,
                         "a solo gallery must not grow from an unverifiable frame")

    def test_solo_gallery_needs_a_stricter_score_to_be_confident(self):
        # Measured on the robot: the correct room scores 0.85-0.88 but a DIFFERENT
        # room in the same house still scores 0.75-0.82 — straddling
        # PLACE_MATCH_CONFIDENT. With no runner-up the margin proves nothing, so the
        # bar rises to PLACE_MATCH_SOLO_CONFIDENT. This is why the dining room was
        # announced as "the workshop".
        pr, clk, ev, box, ws = self._make()
        pr.enroll_from_frames("workshop", [E[0]] * 5)
        self.assertEqual(pr.score_frame(q_cos(0, 0.82)).classification, P.TENTATIVE)
        self.assertEqual(pr.score_frame(q_cos(0, 0.90)).classification, P.CONFIDENT)

    def test_human_denial_drops_the_belief(self):
        # "This is not the workshop" must clear the belief, not draw "Yep, the
        # workshop. I recognize it." (field 2026-07-24).
        pr, clk, ev, box, ws = self._make()
        pr.enroll_from_frames("workshop", [E[0]] * 5)
        box["motion"] = MotionState(wheels_moving=True)
        for _ in range(6):
            clk.adv(2.0)
            pr.observe(E[0])
            if pr.current_place():
                break
        self.assertIsNotNone(pr.current_place())
        self.assertTrue(pr.reject_belief("workshop"))
        self.assertIsNone(pr.current_place())
        self.assertIsNone(ws.get("current_place"))
        # A denial naming a DIFFERENT room says nothing about this belief.
        for _ in range(6):
            clk.adv(2.0)
            pr.observe(E[0])
            if pr.current_place():
                break
        self.assertIsNotNone(pr.current_place())
        self.assertFalse(pr.reject_belief("garage"))
        self.assertIsNotNone(pr.current_place())
        # No belief held -> nothing to reject.
        pr.reject_belief("workshop")
        self.assertFalse(pr.reject_belief())

    def test_refresh_never_fires_on_ambiguous_frames(self):
        # Field regression (2026-07-21): refresh on frames where ANOTHER room out-scores
        # (or nearly ties) the believed room cross-pollinates the galleries — the
        # believed room absorbs views of the room he's actually looking at, and two
        # look-alike galleries converge further. Refresh requires believed == top match
        # BY the confidence margin.
        pr, clk, ev, box, ws = self._make()
        pr.enroll_from_frames("office", [E[0]] * 5)
        qvec = (0.75 * E[0] + math.sqrt(1 - 0.75 ** 2) * E[2]).astype("float32")
        pr.enroll_from_frames("hall", [qvec] * 5)
        box["motion"] = MotionState(wheels_moving=True)
        for _ in range(6):
            clk.adv(2.0)
            pr.observe(E[0])  # office (1.0) beats hall (0.75) -> belief = office
            if pr.current_place():
                break
        self.assertEqual(pr.current_place()["name"], "office")
        pid = pr._name_to_id["office"]
        n0 = int((pr._emb_pids == pid).sum())
        # A frame where hall wins while office sits in its refresh band: NO refresh —
        # this is exactly the contamination vector (it would append a hall-looking
        # frame to office's gallery).
        r = pr.score_frame(qvec)
        self.assertEqual(r.best.name, "hall")
        clk.adv(2.0)
        pr.observe(qvec)
        self.assertEqual(int((pr._emb_pids == pid).sum()), n0)

    def test_confident_requires_margin_over_runner_up(self):
        # Two look-alike galleries: a high absolute score with a whisker-thin lead is
        # TENTATIVE, not confident (the 2026-07-21 flip-flop).
        pr, clk, ev, box, ws = self._make()
        base = E[0]
        near = (0.995 * E[0] + math.sqrt(1 - 0.995 ** 2) * E[3]).astype("float32")
        near /= np.linalg.norm(near)
        pr.enroll_from_frames("living", [base] * 5)
        pr.enroll_from_frames("dining", [near] * 5)   # twin gallery
        r = pr.score_frame(base)
        self.assertEqual(r.best.name, "living")
        self.assertGreater(r.best.score, 0.95)                     # high absolute score
        self.assertEqual(r.classification, P.TENTATIVE)            # but no margin -> tentative
        # A frame matching a UNIQUE room is still confident.
        pr.enroll_from_frames("garage", [E[5]] * 5)
        self.assertEqual(pr.score_frame(E[5]).classification, P.CONFIDENT)

    def test_belief_context_reports_ambiguity(self):
        pr, clk, ev, box, ws = self._make()
        pr.enroll_from_frames("living", [E[0]] * 5)
        near = (0.995 * E[0] + math.sqrt(1 - 0.995 ** 2) * E[3]).astype("float32")
        near /= np.linalg.norm(near)
        pr.enroll_from_frames("dining", [near] * 5)
        box["motion"] = MotionState(wheels_moving=True)
        clk.adv(2.0)
        pr.observe(E[0])                       # one query -> last_query snapshot
        ctx = pr.belief_context()
        self.assertTrue(ctx["ambiguous"])      # twins within the margin
        self.assertEqual(ctx["known_rooms"], 2)
        self.assertEqual(len(ctx["top"]), 2)
        self.assertIsNone(ctx["belief"])       # nothing confirmed yet

    def test_no_unknown_place_on_stationary_cold_boot(self):
        # Regression (review): a stationary boot in an unenrolled room must NOT emit
        # unknown_place (no motion has ever occurred).
        pr, clk, ev, box, ws = self._make()
        pr.enroll_from_frames("office", [E[0]] * 5)
        box["motion"] = MotionState()  # never moved
        for _ in range(12):
            clk.adv(2.0)
            pr.observe(E[7])  # all unknown
        self.assertEqual(self._n_events(ev, P.EVENT_UNKNOWN_PLACE), 0)

    def test_mixed_dim_gallery_degrades_not_crashes(self):
        # Regression (review): ragged rows under one model_tag (a model swap that reused
        # the tag) must degrade to the majority dim, not crash np.stack at load.
        import os
        import sqlite3
        import tempfile

        db = os.path.join(tempfile.mkdtemp(), "places.db")
        emb = lambda f: np.asarray(f, dtype=np.float32)  # noqa: E731
        pr = PlaceRecognizer(emb, db_path=db, model_tag="A")
        pr.enroll_from_frames("office", [E[0]] * 3)  # dim 16
        pr.close()
        # Inject a rogue 8-dim row under the same tag directly.
        con = sqlite3.connect(db)
        v8 = np.ones(8, dtype=np.float32) / np.sqrt(8)
        pid = con.execute("SELECT place_id FROM places WHERE name='office'").fetchone()[0]
        con.execute(
            "INSERT INTO place_embeddings (place_id, vector, dim, heading_deg, captured_at, model_tag) "
            "VALUES (?,?,?,?,?,?)",
            (pid, v8.tobytes(), 8, None, "2020-01-01T00:00:00.000000", "A"),
        )
        con.commit()
        con.close()
        pr2 = PlaceRecognizer(emb, db_path=db, model_tag="A")  # must not raise
        self.addCleanup(pr2.close)
        self.assertEqual(pr2._dim, 16)                      # majority dim kept
        self.assertEqual(pr2._matrix.shape[1], 16)
        self.assertEqual(pr2._matrix.shape[0], 3)           # rogue 8-dim row ignored
        # rows still on disk (nothing deleted)
        con = sqlite3.connect(db)
        self.assertEqual(con.execute("SELECT count(*) FROM place_embeddings").fetchone()[0], 4)
        con.close()

    def test_collect_drops_frame_for_superseded_session(self):
        # Regression (review): an in-flight embed for session A must not land in a
        # session B created by a re-enroll while embedding. Simulated by embedding for
        # A, then swapping to B before the phase-2 append.
        swapped = {"done": False}

        def embed(f):
            v = np.asarray(f, dtype=np.float32)
            if not swapped["done"]:
                swapped["done"] = True
                pr.enroll("kitchen")  # replace session A with B mid-embed
            return v

        pr, clk, ev, box, ws = self._make()
        # rebind embed_fn to our racing stub
        pr._embed_fn = embed
        pr.enroll("office")
        box["heading"] = None
        pr.observe(E[0])  # embeds for office, but swaps to kitchen before append
        self.assertEqual(pr._enroll.name, "kitchen")
        self.assertEqual(len(pr._enroll.vectors), 0)  # office frame NOT appended to kitchen

    # ── robot-lens review regressions ──
    def test_motion_signal_unavailable_disables_freeze(self):
        # No drive base -> get_motion_state returns None -> the freeze gate must be OFF
        # (a robot with no wheel sensor can still be carried between rooms).
        pr, clk, ev, box, ws = self._make()
        box["motion"] = None
        pr.enroll_from_frames("office", [E[0]] * 5)
        pr.enroll_from_frames("living", [E[1]] * 5)
        for _ in range(6):
            clk.adv(2.0)
            pr.observe(E[0])
        self.assertEqual(pr.current_place()["name"], "office")
        for _ in range(6):
            clk.adv(2.0)
            pr.observe(E[1])   # carried to the living room; no motion ever reported
        self.assertEqual(pr.current_place()["name"], "living")

    def test_static_flip_streak_overrides_freeze(self):
        # Base attached but silent (carried): sustained confident evidence for another
        # room must eventually beat the frozen gate (PLACE_STATIC_FLIP_STREAK).
        pr, clk, ev, box, ws = self._make()
        pr.enroll_from_frames("office", [E[0]] * 5)
        pr.enroll_from_frames("living", [E[1]] * 5)
        box["motion"] = MotionState(wheels_moving=True)
        for _ in range(10):
            clk.adv(2.0)
            pr.observe(E[0])
            if pr.current_place():
                break
        box["motion"] = MotionState()          # wheels still — freeze engaged
        for _ in range(pr._static_flip_max + pr._hysteresis_frames + 2):
            clk.adv(2.0)
            pr.observe(E[1])
        self.assertEqual(pr.current_place()["name"], "living")

    def test_sustained_unknown_clears_belief_to_lost(self):
        # Carried to an UNENROLLED room with a silent base: keep claiming "office"
        # forever was wrong — after PLACE_LOST_STREAK unknowns the belief drops to None
        # (which re-arms the ask-what-room cue).
        pr, clk, ev, box, ws = self._make()
        pr.enroll_from_frames("office", [E[0]] * 5)
        box["motion"] = MotionState(wheels_moving=True)
        for _ in range(10):
            clk.adv(2.0)
            pr.observe(E[0])
            if pr.current_place():
                break
        box["motion"] = MotionState()
        for _ in range(pr._lost_streak_max + 1):
            clk.adv(2.0)
            pr.observe(E[7])
        self.assertIsNone(pr.current_place())
        self.assertIsNone(ws.get("current_place"))

    def test_stuck_heading_falls_back_to_time_gate(self):
        # A parked head (or stuck compass) yields a CONSTANT heading; enrollment must
        # degrade to time-separated captures, not starve into enrollment_failed.
        pr, clk, ev, box, ws = self._make()
        pr.enroll("den")
        box["heading"] = 100.0                  # never changes
        for _ in range(12):
            clk.adv(5.0)                        # > PLACE_ENROLL_MIN_TIME_SEP_S
            pr.observe(E[2])
            if pr.state == P.IDLE:
                break
        self.assertEqual(self._n_events(ev, P.EVENT_PLACE_ENROLLED), 1)
        self.assertIn("den", pr.place_names())

    def test_enrolling_name_accessor(self):
        pr, clk, ev, box, ws = self._make()
        self.assertIsNone(pr.enrolling_name())
        pr.enroll("garage")
        self.assertEqual(pr.enrolling_name(), "garage")
        pr.cancel_enrollment()
        self.assertIsNone(pr.enrolling_name())

    # ── persistence / tag isolation ──
    def test_persistence_and_tag_isolation(self):
        import os
        import tempfile

        d = tempfile.mkdtemp()
        db = os.path.join(d, "places.db")
        emb = lambda f: np.asarray(f, dtype=np.float32)  # noqa: E731
        a = PlaceRecognizer(emb, db_path=db, model_tag="A")
        a.enroll_from_frames("office", [E[0]] * 4)
        a.close()

        a2 = PlaceRecognizer(emb, db_path=db, model_tag="A")
        self.assertEqual(a2.place_names(), ["office"])
        self.assertEqual(a2.score_frame(E[0]).best.name, "office")
        a2.close()

        b = PlaceRecognizer(emb, db_path=db, model_tag="B")
        self.assertEqual(b._matrix.shape[0], 0)      # other tag ignored
        self.assertEqual(b.place_names(), ["office"])  # but not deleted
        b.enroll_from_frames("kitchen", [E[1]] * 3)
        b.close()

        a3 = PlaceRecognizer(emb, db_path=db, model_tag="A")
        self.assertEqual(int((a3._emb_pids == a3._name_to_id["office"]).sum()), 4)
        a3.close()


if __name__ == "__main__":
    unittest.main()
