"""A declared game roster changes who can plausibly be speaking.

Voice attribution's 1:1 heuristics assume that anyone the voice points at is
probably absent, so a marginal cross-match must be a print artifact and the one
visible face should win. A game roster inverts that: those people announced
themselves into the room, take turns, and mostly sit OFF camera.

Field 2026-08-26 20:12-20:15 — PJ was the only recognized face in frame and was
logged as the speaker for four different contestants' answers in a row, and two
registered players became unknown_voice_1 / unknown_voice_2 mid-game.
"""

import unittest
from unittest import mock

from features import games
from intelligence import interaction

BRET, TJOY, JEREMY, PJ, JADE = 1, 3, 4, 7, 8
ROSTER = frozenset({BRET, TJOY, JEREMY, PJ})


class VoicePrimaryRosterDecisionTest(unittest.TestCase):
    """_voice_primary_face_decision is pure — drive it directly."""

    BASE = dict(
        person_id=JEREMY, raw_best_id=JEREMY, speaker_score=0.654, ws_pid=PJ,
        single_visible=True, engaged_is_visible=False,
        unknown_visible=False, other_known_recently=False,
    )

    def test_without_a_roster_the_visible_face_still_wins(self):
        self.assertEqual(
            interaction._voice_primary_face_decision(**self.BASE),
            "voice_weak_face_wins",
        )

    def test_a_roster_member_keeps_their_own_marginal_turn(self):
        self.assertEqual(
            interaction._voice_primary_face_decision(**self.BASE, roster_ids=ROSTER),
            "voice_over_face_roster",
        )

    def test_someone_off_the_roster_does_not_get_the_relief(self):
        decision = interaction._voice_primary_face_decision(
            **dict(self.BASE, person_id=JADE, raw_best_id=JADE), roster_ids=ROSTER)
        self.assertEqual(decision, "voice_weak_face_wins")

    def test_a_confident_voice_is_unaffected(self):
        decision = interaction._voice_primary_face_decision(
            **dict(self.BASE, speaker_score=0.90), roster_ids=ROSTER)
        self.assertEqual(decision, "voice_over_face")

    def test_the_mouth_still_veto_still_wins(self):
        # The camera positively watched the visible face NOT talking: the roster
        # must not resurrect the face-wins path (field 2026-08-02 12:37).
        decision = interaction._voice_primary_face_decision(
            **self.BASE, visual_mouth_still=True, roster_ids=ROSTER)
        self.assertEqual(decision, "off_screen_unknown")

    def test_rex_never_stops_the_board_to_ask_who_is_speaking(self):
        # Marginal match ON the visible face with no voice credibility.
        base = dict(
            person_id=PJ, raw_best_id=PJ, speaker_score=0.60, ws_pid=PJ,
            single_visible=True, engaged_is_visible=False,
            unknown_visible=False, other_known_recently=False,
        )
        self.assertEqual(
            interaction._voice_primary_face_decision(**base), "challenge_identity")
        self.assertEqual(
            interaction._voice_primary_face_decision(**base, roster_ids=ROSTER),
            "voice_agrees_no_refresh",
        )

    def test_a_mid_game_mouth_still_veto_is_not_overridden(self):
        base = dict(
            person_id=PJ, raw_best_id=PJ, speaker_score=0.40, ws_pid=PJ,
            single_visible=True, engaged_is_visible=False,
            unknown_visible=False, other_known_recently=False,
            visual_mouth_still=True,
        )
        self.assertEqual(
            interaction._voice_primary_face_decision(**base, roster_ids=ROSTER),
            "challenge_identity",
        )


class RosterAmbiguityReliefTest(unittest.TestCase):
    """The margin guard exists to stop an unenrolled STRANGER being named. In a
    declared game every close candidate is already a registered player, so a
    thin gap cannot mean "stranger" — it means two contestants who sound alike."""

    def setUp(self):
        games._active_game = "jeopardy"
        games._game_state = {
            "players": [
                {"name": "Bret", "person_id": BRET},
                {"name": "PJ", "person_id": PJ},
                {"name": "Jeremy", "person_id": JEREMY},
                {"name": "Tjoy", "person_id": TJOY},
            ],
            "current_player_idx": 0,
        }
        self._saved_scan = interaction._last_scan_ranked

    def tearDown(self):
        interaction._last_scan_ranked = self._saved_scan
        games._game_state = {}
        games._active_game = None

    def _accept(self, ranked, floor=0.45):
        interaction._last_scan_ranked = ranked
        return interaction._game_roster_ambiguity_accept(ranked[0][0], ranked[0][2], floor)

    def test_two_contestants_in_a_near_tie_are_accepted(self):
        # Field 20:22:08: PJ 0.783 lost to Bret 0.758 by 0.025 and became
        # "unknown_voice_2" while answering a clue.
        self.assertTrue(self._accept([
            (PJ, "PJ Thomas", 0.783, 5),
            (BRET, "Bret Benziger", 0.758, 5),
            (JADE, "Jade Smith", 0.412, 5),
        ]))

    def test_a_non_player_in_the_contention_band_keeps_the_strict_guard(self):
        self.assertFalse(self._accept([
            (PJ, "PJ Thomas", 0.783, 5),
            (JADE, "Jade Smith", 0.770, 5),
        ]))

    def test_a_non_player_winner_gets_nothing(self):
        self.assertFalse(self._accept([
            (JADE, "Jade Smith", 0.80, 5),
            (PJ, "PJ Thomas", 0.70, 5),
        ]))

    def test_below_the_known_floor_is_still_unknown(self):
        self.assertFalse(self._accept([
            (PJ, "PJ Thomas", 0.40, 5),
            (BRET, "Bret Benziger", 0.30, 5),
        ]))

    def test_no_game_no_relief(self):
        games._active_game = None
        self.assertFalse(self._accept([
            (PJ, "PJ Thomas", 0.783, 5),
            (BRET, "Bret Benziger", 0.758, 5),
        ]))

    def test_the_current_player_is_the_tie_break_inside_the_band(self):
        interaction._last_scan_ranked = [
            (PJ, "PJ Thomas", 0.783, 5),
            (BRET, "Bret Benziger", 0.758, 5),
            (JADE, "Jade Smith", 0.412, 5),
        ]
        # Bret's turn, and Bret is inside the ambiguity band -> tie-breakable.
        self.assertEqual(games.active_game_current_player_id(), BRET)
        self.assertIsNotNone(interaction._game_roster_tied_candidate(BRET))
        # Jeremy is nowhere near the band.
        self.assertIsNone(interaction._game_roster_tied_candidate(JEREMY))

    def test_a_clear_winner_is_not_tie_broken(self):
        interaction._last_scan_ranked = [
            (PJ, "PJ Thomas", 0.90, 5),
            (BRET, "Bret Benziger", 0.40, 5),
        ]
        self.assertIsNone(interaction._game_roster_tied_candidate(BRET))


class EagerProbeDuringGamesTest(unittest.TestCase):
    """The eager motion probe runs a second full ASR decode per turn and takes
    MLX_LOCK ahead of the real one. A game turn is never a drive command."""

    def tearDown(self):
        games._active_game = None

    def test_a_moving_base_keeps_the_stop_safety_cut(self):
        # MOTION_HOLD_DURING_GAMES parks only the SOCIAL lanes — the flinch
        # reflex and an explicit come-here still drive, so "stop" must still cut
        # eagerly while the wheels are turning.
        with mock.patch.object(interaction.motion_controller, "available",
                               return_value=True), \
             mock.patch.object(interaction.config, "MOTION_EAGER_ENDPOINT_REQUIRE_AEC",
                               False, create=True), \
             mock.patch.object(interaction.motion_controller, "is_moving",
                               return_value=True):
            games._active_game = "jeopardy"
            self.assertTrue(interaction._eager_motion_endpoint_enabled())

    def test_probe_is_off_while_a_game_is_running(self):
        # Everything downstream of the game gate is platform-dependent (it needs
        # a drive base and hardware AEC), so stub the later gates and assert
        # only that the game itself turns the probe off.
        with mock.patch.object(interaction.motion_controller, "available",
                               return_value=True), \
             mock.patch.object(interaction.config, "MOTION_EAGER_ENDPOINT_REQUIRE_AEC",
                               False, create=True), \
             mock.patch.object(interaction.motion_controller, "is_moving",
                               return_value=False):
            games._active_game = None
            self.assertTrue(interaction._eager_motion_endpoint_enabled())
            games._active_game = "jeopardy"
            self.assertFalse(interaction._eager_motion_endpoint_enabled())

    def test_the_game_gate_is_switchable(self):
        with mock.patch.object(interaction.motion_controller, "available",
                               return_value=True), \
             mock.patch.object(interaction.config, "MOTION_EAGER_ENDPOINT_REQUIRE_AEC",
                               False, create=True), \
             mock.patch.object(interaction.config, "MOTION_EAGER_ENDPOINT_DURING_GAMES",
                               True, create=True), \
             mock.patch.object(interaction.motion_controller, "is_moving",
                               return_value=False):
            games._active_game = "jeopardy"
            self.assertTrue(interaction._eager_motion_endpoint_enabled())


if __name__ == "__main__":
    unittest.main()
