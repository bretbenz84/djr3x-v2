"""Game roster names -> stored identity -> visible face -> normal head centering."""
import sqlite3
import unittest
from unittest import mock

import config
from features import games, jeopardy
from intelligence import consciousness as C
from memory import people
from setup_assets import DB_SCHEMA
from state import State


class RosterNamesTest(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.db.row_factory = sqlite3.Row
        self.db.executescript(DB_SCHEMA)
        self.db.executemany("INSERT INTO people(id, name, nickname) VALUES (?, ?, ?)", [
            (1, "Bret Benziger", "Bretmichael"), (3, "Joy Jackson", "Exudica Marbles"),
            (4, "Jeremy Thomas", "JT"), (7, "PJ Thomas", "Peaceful-P"),
            (8, "Jade Smith", "DJ Smitty"),
        ])
        self.db.execute("INSERT INTO person_aliases(person_id,alias,alias_norm) VALUES (3, 'TJoy', 'tjoy')")
        self.addCleanup(self.db.close)
        self.enterContext(mock.patch.object(people.db, "fetchall", side_effect=lambda sql, params=(): self.db.execute(sql, params).fetchall()))
        self.enterContext(mock.patch.object(people.db, "fetchone", side_effect=lambda sql, params=(): self.db.execute(sql, params).fetchone()))
        self.enterContext(mock.patch.object(people, "has_voice_biometric", return_value=True))
        self.create = self.enterContext(mock.patch.object(people, "find_or_create_person", side_effect=AssertionError("Known players must not create duplicates")))

    def test_spoken_first_names_link_full_names(self):
        roster, _ = games._jeopardy_prepare_players(jeopardy.parse_player_names("Bret, Jeremy, PJ, and Jade"))
        self.assertEqual([p["person_id"] for p in roster], [1, 4, 7, 8])
        self.assertEqual([p["name"] for p in roster], ["Bret", "Jeremy", "PJ", "Jade"])
        self.create.assert_not_called()

    def test_nicknames_and_dictated_initials_link_existing_players(self):
        roster, _ = games._jeopardy_prepare_players(jeopardy.parse_player_names("Bretmichael, J T, Peaceful-P, and DJ Smitty"))
        self.assertEqual([p.get("person_id") for p in roster], [1, 4, 7, 8])
        self.create.assert_not_called()

    def test_alias_and_multiword_nickname_resolve(self):
        self.assertEqual(games._jeopardy_find_or_create_player("TJoy")[0], 3)
        self.assertEqual(games._jeopardy_find_or_create_player("Exudica Marbles")[0], 3)

    def test_stored_nickname_beats_builtin_guess(self):
        self.db.executemany("INSERT INTO people(id,name,nickname) VALUES (?,?,?)", [
            (10, "Robert Jones", "Bill"), (11, "William Smith", None),
        ])
        self.assertEqual(games._jeopardy_find_or_create_player("Bill")[0], 10)

    def test_duplicate_nicknames_do_not_choose_arbitrary_person(self):
        self.db.execute("INSERT INTO people(id,name,nickname) VALUES (10,'Another Person','JT')")
        self.assertIsNone(people.find_person_by_nickname("JT"))


class GazePlayerTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(mock.patch.object(games, "_active_game", "jeopardy"))
        self.enterContext(mock.patch.object(games, "_game_state", {
            "players": [{"name": "Bret", "person_id": 1}, {"name": "PJ", "person_id": 7}],
            "phase": "selecting", "current_player_idx": 0,
        }))

    def test_selecting_answering_and_wagering_follow_current_player(self):
        for phase in ("selecting", "awaiting_answer", "awaiting_wager"):
            games._game_state.update(phase=phase, current_player_idx=1)
            self.assertEqual(games.active_game_gaze_player_id(), 7)

    def test_final_and_voice_checks_follow_their_own_queue(self):
        for phase, key in (("final_answer", "final_queue"), ("final_wager", "final_queue"), ("voice_enroll", "voice_enroll_queue")):
            games._game_state.update(phase=phase, **{key: [1, 0]})
            self.assertEqual(games.active_game_gaze_player_id(), 7)
            games._game_state[key].pop(0)
            self.assertEqual(games.active_game_gaze_player_id(), 1)
            games._game_state[key].clear()
            self.assertIsNone(games.active_game_gaze_player_id())

    def test_solo_gaze_does_not_change_voice_attribution_prior(self):
        games._game_state["players"] = games._game_state["players"][:1]
        self.assertEqual(games.active_game_gaze_player_id(), 1)
        self.assertIsNone(games.active_game_current_player_id())

    def test_unknown_or_ended_game_has_no_gaze_target(self):
        games._game_state["players"][0].pop("person_id")
        self.assertIsNone(games.active_game_gaze_player_id())
        games._active_game = None
        self.assertIsNone(games.active_game_gaze_player_id())


class TrackingTest(unittest.TestCase):
    def setUp(self):
        from tests.test_face_tracking import FaceTrackingTests
        fixture = FaceTrackingTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        fixture._set_servo_positions()
        self.frame = fixture.frame
        self.faces = [
            {"id": "bret", "person_db_id": 1, "face_id": "Bret", "face_visible": True, "face_box": (900, 180, 200, 200)},
            {"id": "pj", "person_db_id": 7, "face_id": "PJ", "face_visible": True, "face_box": (100, 160, 120, 120)},
        ]
        self.enterContext(mock.patch.object(games, "_active_game", "jeopardy"))
        self.enterContext(mock.patch.object(games, "_game_state", {
            "players": [{"person_id": 1}, {"person_id": 7}], "phase": "awaiting_answer", "current_player_idx": 1,
        }))
        self.enterContext(mock.patch.object(C, "_face_tracking_lock", {"key": "db:1", "person_id": 1, "last_seen_at": 199.}))
        self.enterContext(mock.patch.object(C, "_face_tracking_suspended_until", 0.))
        self.enterContext(mock.patch.object(C, "_idle_wander", {"active": False}))
        self.enterContext(mock.patch.object(C, "_gaze_drive", {"phase": "away", "anchor": (7000,6000,4320)}))
        self.enterContext(mock.patch.object(C.state_module, "get_state", return_value=State.ACTIVE))
        self.enterContext(mock.patch.object(C.time, "monotonic", return_value=200.))
        self.enterContext(mock.patch("intelligence.exploration.active", return_value=False))
        self.manual = self.enterContext(mock.patch("hardware.servos.manual_override_enabled", return_value=False))
        self.enterContext(mock.patch("hardware.servos.listening_motion_active", return_value=False))
        self.speech = self.enterContext(mock.patch("hardware.servos.speech_motion_active", return_value=False))
        self.move = self.enterContext(mock.patch("hardware.servos.set_servos"))
        self.enterContext(mock.patch("hardware.servos.set_motion_profile"))
        self.enterContext(mock.patch("hardware.servos.set_face_tracking_baseline"))
        self.gaze = self.enterContext(mock.patch.object(C, "_maybe_drive_gaze", return_value=True))
        self.enterContext(mock.patch.object(C, "_drive_object_glance", return_value=False))
        self.enterContext(mock.patch.object(C, "_object_glance_release"))
        self.enterContext(mock.patch.object(config, "GAME_PLAYER_GAZE_ENABLED", True))

    def test_current_player_wins_over_previous_lock_and_larger_face(self):
        C._step_face_tracking(self.frame, self.faces)
        self.assertEqual(C._face_tracking_lock["person_id"], 7)
        self.assertLess(self.move.call_args.args[0][config.SERVO_CHANNELS["neck"]["ch"]], 6000)
        self.assertEqual(C._gaze_drive["phase"], "idle")
        self.gaze.assert_not_called()

    def test_turn_change_moves_to_new_player_without_new_speech(self):
        C._step_face_tracking(self.frame, self.faces)
        games._game_state["current_player_idx"] = 0
        C._step_face_tracking(self.frame, self.faces)
        self.assertEqual(C._face_tracking_lock["person_id"], 1)
        self.assertGreater(self.move.call_args.args[0][config.SERVO_CHANNELS["neck"]["ch"]], 6000)

    def test_reading_clue_keeps_gentle_centering_on_player(self):
        self.speech.return_value = True
        C._step_face_tracking(self.frame, self.faces)
        self.assertEqual(C._face_tracking_lock["person_id"], 7)
        self.move.assert_called()

    def test_missing_or_unrecognized_player_does_not_get_game_focus(self):
        for change in ({"face_visible": False}, {"face_missing": True}, {"face_id": None}):
            faces = [self.faces[0], {**self.faces[1], **change}]
            self.assertIsNone(C._game_player_gaze_candidate(C._visible_face_tracking_candidates(faces)))
            C._step_face_tracking(self.frame, faces)
        self.move.assert_not_called()
        self.gaze.assert_called()

    def test_game_focus_does_not_rewrite_or_acquire_speaker_intent(self):
        intent = {"person_id": 1, "search_requested": True}
        with mock.patch.object(C, "_speaker_gaze_intent", intent), \
             mock.patch.object(C, "_speaker_gaze_note_acquired") as acquired, \
             mock.patch.object(C, "_speaker_gaze_request_search") as search:
            C._step_face_tracking(self.frame, self.faces)
        self.assertEqual(intent, {"person_id": 1, "search_requested": True})
        acquired.assert_not_called()
        search.assert_not_called()

    def test_game_focus_ends_idle_wander_without_regreeting(self):
        C._idle_wander["active"] = True
        with mock.patch.object(C, "_drive_idle_head_wander") as wander:
            C._step_face_tracking(self.frame, self.faces)
        self.assertFalse(C._idle_wander["active"])
        self.assertFalse(C._idle_wander["pending_regreet"])
        wander.assert_not_called()
        self.assertEqual(C._face_tracking_lock["person_id"], 7)

    def test_config_can_restore_normal_gaze_during_game(self):
        with mock.patch.object(config, "GAME_PLAYER_GAZE_ENABLED", False):
            C._step_face_tracking(self.frame, self.faces)
        self.gaze.assert_called_once()
        self.move.assert_not_called()

    def test_manual_and_explicit_look_commands_keep_control(self):
        self.manual.return_value = True
        C._step_face_tracking(self.frame, self.faces)
        self.manual.return_value = False
        with mock.patch.object(C, "directed_gaze_hold_active", return_value=True):
            C._step_face_tracking(self.frame, self.faces)
        self.move.assert_not_called()

    def test_game_end_restores_normal_gaze(self):
        games._active_game = None
        C._step_face_tracking(self.frame, self.faces)
        self.gaze.assert_called_once()
        self.move.assert_not_called()


if __name__ == "__main__":
    unittest.main()
