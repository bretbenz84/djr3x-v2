"""
Rex's day mood (intelligence/rex_mood.py).

Owner gripe 2026-08-05: asked "how are you?" — directly, or bounced back as "how
about you?" after Rex asked first — he always answered "operating within normal
parameters". He had no self-state to answer FROM, and REX_CORE_PROMPT hands the model
"systems nominal" as a droid verbal tic, so a status report was the only attractor.

These lock in: one mood per LOCAL day, seeded by real day signals, drifting within
bounds, surviving a reboot, and NOT repeating within the anti-repeat window.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest import mock

import config
from intelligence import rex_mood


# Captured at import time, BEFORE any test patches config, so the shipped-pool guard
# below validates the real authored pool rather than the test fixture.
_SHIPPED_SEEDS = list(getattr(config, "REX_MOOD_SEEDS", []) or [])

_TEST_SEEDS = [
    {"id": "up-a", "label": "buoyant", "valence": 0.8, "energy": 0.7,
     "line": "Annoyingly good, and I cannot source it.", "fits": ["bright", "up"]},
    {"id": "up-b", "label": "bright", "valence": 0.7, "energy": 0.75,
     "line": "Genuinely good, and riding it.", "fits": ["bright"]},
    {"id": "down-a", "label": "flat", "valence": -0.25, "energy": 0.2,
     "line": "Flat. Not bad, flat.", "fits": ["flat", "low"]},
    {"id": "down-b", "label": "worn", "valence": -0.4, "energy": 0.2,
     "line": "Worn down, and awake for all of it.", "fits": ["drained"]},
    {"id": "mid", "label": "patient", "valence": 0.3, "energy": 0.4,
     "line": "Steady enough to let you finish a sentence.", "fits": ["any"]},
]


class _MoodTestCase(unittest.TestCase):
    """Patches the pool + flags and wipes module state on both ends, since one
    unittest process runs every class in this module."""

    def setUp(self) -> None:
        self._patches = [
            mock.patch.object(config, "REX_MOOD_ENABLED", True),
            mock.patch.object(config, "REX_MOOD_PERSIST_ENABLED", True),
            mock.patch.object(config, "REX_MOOD_SEEDS", _TEST_SEEDS),
            mock.patch.object(config, "REX_MOOD_RECENT_MEMORY_DAYS", 3),
            mock.patch.object(config, "REX_MOOD_DRIFT_LIMIT", 0.35),
            # Pacing off by default here so drift tests can step deterministically;
            # DriftPacingTests turns it back on and owns that behavior.
            mock.patch.object(config, "REX_MOOD_DRIFT_MIN_INTERVAL_SECS", 0.0),
            # No day signals unless a test installs one — otherwise live weather /
            # the real news cache would make these tests depend on the outside world.
            mock.patch.object(rex_mood, "_SIGNALS", ()),
        ]
        for p in self._patches:
            p.start()
        rex_mood.clear()

    def tearDown(self) -> None:
        rex_mood.clear()
        for p in self._patches:
            p.stop()


class MintTests(_MoodTestCase):

    def test_mood_is_minted_once_per_day_and_is_stable(self):
        first = rex_mood.ensure_today()
        self.assertIsNotNone(first)
        for _ in range(5):
            self.assertIs(rex_mood.ensure_today(), first)

    def test_same_date_always_rolls_the_same_mood(self):
        # Day-seeded RNG: a crash-and-restart before the state file is written must
        # resume the SAME mood, not re-roll a new personality.
        when = datetime(2026, 8, 5, 9, 0, 0)
        a = rex_mood.ensure_today(when).seed_id
        rex_mood.clear()
        b = rex_mood.ensure_today(when).seed_id
        self.assertEqual(a, b)

    def test_a_new_day_mints_a_new_mood(self):
        day1 = datetime(2026, 8, 5, 9, 0, 0)
        day2 = datetime(2026, 8, 6, 9, 0, 0)
        first = rex_mood.ensure_today(day1)
        second = rex_mood.ensure_today(day2)
        self.assertEqual(second.date, "2026-08-06")
        self.assertNotEqual(first.date, second.date)

    def test_disabled_mints_nothing_and_stays_silent(self):
        with mock.patch.object(config, "REX_MOOD_ENABLED", False):
            self.assertIsNone(rex_mood.ensure_today())
            self.assertEqual(rex_mood.prompt_lines(), [])
            self.assertEqual(rex_mood.prompt_section(), "")

    def test_empty_pool_degrades_to_no_mood_rather_than_raising(self):
        with mock.patch.object(config, "REX_MOOD_SEEDS", []):
            self.assertIsNone(rex_mood.ensure_today())

    def test_malformed_seeds_are_skipped_not_fatal(self):
        pool = [{"id": "", "line": "x"}, {"id": "ok", "line": ""}, _TEST_SEEDS[0]]
        with mock.patch.object(config, "REX_MOOD_SEEDS", pool):
            self.assertEqual([s["id"] for s in rex_mood._seeds()], ["up-a"])


class AntiRepeatTests(_MoodTestCase):
    """The bug this class exists for: anti-repeat was first written as a WEIGHT
    reduction, but the RNG is seeded on the date, so a fixed uniform draw just slid
    the cumulative bands under a static cursor and landed back on the de-weighted
    seed — three identical moods in a row from a fresh state. It is an exclusion now.
    """

    def _run_days(self, n: int) -> list:
        out = []
        for i in range(n):
            day = (date(2026, 8, 5) + timedelta(days=i)).isoformat()
            seed = rex_mood._choose((), day, rex_mood._recent_ids)
            rex_mood._remember(day, seed["id"])
            out.append(seed["id"])
        return out

    def test_no_mood_repeats_inside_the_memory_window(self):
        window = int(config.REX_MOOD_RECENT_MEMORY_DAYS)
        got = self._run_days(20)
        for i in range(len(got)):
            recent = got[max(0, i - window):i]
            self.assertNotIn(
                got[i], recent,
                f"mood {got[i]!r} at day {i} repeats within the {window}-day window: {got}",
            )

    def test_never_two_days_running(self):
        got = self._run_days(20)
        for a, b in zip(got, got[1:]):
            self.assertNotEqual(a, b, f"consecutive repeat in {got}")

    def test_memory_window_is_trimmed_to_the_configured_length(self):
        self._run_days(10)
        self.assertLessEqual(len(rex_mood._recent_ids),
                             int(config.REX_MOOD_RECENT_MEMORY_DAYS))

    def test_exclusion_recycles_instead_of_returning_nothing(self):
        # Memory window >= pool size would empty the candidate list.
        with mock.patch.object(config, "REX_MOOD_RECENT_MEMORY_DAYS", 99):
            self._run_days(len(_TEST_SEEDS))
            self.assertIsNotNone(
                rex_mood._choose((), "2026-09-09", rex_mood._recent_ids))


class DaySignalTests(_MoodTestCase):

    def test_matching_tags_bias_the_roll(self):
        # Over many days a "drained/low/flat" day should skew low-energy, and a
        # "bright" day high-valence. Asserted as an aggregate, not per-draw — the
        # tags are a thumb on the scale, not a lookup table.
        def mean(tags, field):
            rex_mood.clear()
            vals = []
            for i in range(30):
                day = (date(2026, 1, 1) + timedelta(days=i)).isoformat()
                seed = rex_mood._choose(tags, day, rex_mood._recent_ids)
                rex_mood._remember(day, seed["id"])
                vals.append(seed[field])
            return sum(vals) / len(vals)

        self.assertLess(mean(("drained", "low", "flat"), "energy"),
                        mean(("bright", "up"), "energy"))
        self.assertLess(mean(("drained", "low", "flat"), "valence"),
                        mean(("bright", "up"), "valence"))

    def test_clock_signal_reads_the_calendar(self):
        tags, because = rex_mood._clock_signal(datetime(2026, 8, 3, 10, 0, 0))  # Monday
        self.assertIn("flat", tags)
        self.assertIn("Monday", because)

        tags, _ = rex_mood._clock_signal(datetime(2026, 8, 5, 3, 0, 0))         # 3am
        self.assertIn("drained", tags)

        tags, _ = rex_mood._clock_signal(datetime(2026, 8, 7, 16, 0, 0))        # Fri pm
        self.assertIn("up", tags)

        tags, _ = rex_mood._clock_signal(datetime(2026, 8, 8, 11, 0, 0))        # Saturday
        self.assertIn("loose", tags)

    def test_first_firing_signal_owns_the_reason(self):
        def occasion(now=None, allow_blocking=False):
            return (("occasion", "up"), "it's a holiday today")

        def clock(now=None, allow_blocking=False):
            return (("flat",), "it's a Monday")

        with mock.patch.object(rex_mood, "_SIGNALS",
                               (("occasion", occasion), ("clock", clock))):
            tags, because, kind = rex_mood._day_tags(datetime(2026, 8, 5, 10, 0))
        self.assertEqual(kind, "occasion")
        self.assertEqual(because, "it's a holiday today")
        # ...but every signal still contributes tags to the roll.
        self.assertIn("flat", tags)
        self.assertIn("occasion", tags)

    def test_a_signal_that_raises_does_not_break_minting(self):
        def boom(now=None, allow_blocking=False):
            raise RuntimeError("weather feed exploded")

        with mock.patch.object(rex_mood, "_SIGNALS", (("weather", boom),)):
            self.assertIsNotNone(rex_mood.ensure_today())

    def test_the_boot_mint_never_blocks_on_the_holiday_feed(self):
        # holidays.get_holidays() does a 5s-timeout network fetch on the first miss
        # per year, and the mint runs on the FOREGROUND boot path, before
        # consciousness.start(). It must abstain rather than stall the robot's boot.
        from awareness import holidays
        with (mock.patch.object(holidays, "_cache", {}),
              mock.patch.object(holidays, "next_relevant_holiday",
                                side_effect=AssertionError("boot mint blocked on the network")) as nrh):
            self.assertEqual(rex_mood._occasion_signal(datetime(2026, 8, 5)), ((), ""))
            nrh.assert_not_called()

    def test_the_background_pass_may_pay_for_the_holiday_fetch(self):
        from awareness import holidays
        with (mock.patch.object(holidays, "_cache", {}),
              mock.patch.object(holidays, "next_relevant_holiday",
                                return_value={"name": "Labor Day", "days_until": 0})):
            tags, because = rex_mood._occasion_signal(
                datetime(2026, 9, 7), allow_blocking=True)
        self.assertIn("occasion", tags)
        self.assertIn("Labor Day", because)

    def test_a_warm_holiday_cache_is_used_on_the_boot_path(self):
        from awareness import holidays
        with (mock.patch.object(holidays, "_cache", {2026: [{"name": "x"}]}),
              mock.patch.object(holidays, "next_relevant_holiday",
                                return_value={"name": "Independence Day", "days_until": 1})):
            tags, because = rex_mood._occasion_signal(datetime(2026, 7, 3))
        self.assertIn("occasion", tags)
        self.assertIn("tomorrow", because)

    def test_a_holiday_further_out_than_tomorrow_does_not_vote(self):
        from awareness import holidays
        with (mock.patch.object(holidays, "_cache", {2026: [{"name": "x"}]}),
              mock.patch.object(holidays, "next_relevant_holiday",
                                return_value={"name": "Thanksgiving", "days_until": 9})):
            self.assertEqual(rex_mood._occasion_signal(datetime(2026, 11, 17)), ((), ""))


class DriftTests(_MoodTestCase):

    def test_events_move_the_mood(self):
        before = rex_mood.ensure_today().valence
        rex_mood.note("complimented")
        self.assertGreater(rex_mood.current().valence, before)
        rex_mood.note("insulted")
        rex_mood.note("insulted")
        self.assertLess(rex_mood.current().valence, before)

    def test_drift_is_clamped_so_the_day_keeps_its_character(self):
        rex_mood.ensure_today()
        for _ in range(200):
            rex_mood.note("complimented")
        limit = float(config.REX_MOOD_DRIFT_LIMIT)
        self.assertLessEqual(rex_mood.current().drift_valence, limit + 1e-9)
        self.assertLessEqual(rex_mood.current().drift_energy, limit + 1e-9)

    def test_unknown_event_kinds_are_ignored(self):
        mood = rex_mood.ensure_today()
        rex_mood.note("nonsense-event")
        self.assertEqual(mood.drift_valence, 0.0)
        self.assertEqual(mood.drift_energy, 0.0)

    def test_event_log_is_bounded(self):
        rex_mood.ensure_today()
        for _ in range(200):
            rex_mood.note("long_quiet")
        self.assertLessEqual(len(rex_mood.current().events), 40)

    def test_late_hours_take_energy_off_the_top(self):
        # He shouldn't still claim to be wired at 1am because he woke up wired at 9.
        day = datetime(2026, 8, 5, 10, 0, 0)
        rex_mood.ensure_today(day)
        daytime = rex_mood.effective_energy(day)
        late = rex_mood.effective_energy(datetime(2026, 8, 5, 23, 30, 0))
        self.assertLess(late, daytime)

    def test_late_hour_taper_is_not_stored(self):
        day = datetime(2026, 8, 5, 10, 0, 0)
        mood = rex_mood.ensure_today(day)
        base = mood.energy
        rex_mood.effective_energy(datetime(2026, 8, 5, 23, 30, 0))
        self.assertEqual(rex_mood.current().energy, base)


class DriftPacingTests(_MoodTestCase):
    """Drift callers live on polling loops — `long_quiet` is evaluated from the lull
    path, which can be consulted repeatedly inside ONE stretch of silence. Without
    per-kind pacing a single quiet afternoon drives the drift to its clamp in seconds
    and every later event that day becomes a no-op."""

    def test_repeat_of_the_same_kind_is_paced(self):
        with mock.patch.object(config, "REX_MOOD_DRIFT_MIN_INTERVAL_SECS", 600.0):
            rex_mood.ensure_today()
            for _ in range(50):
                rex_mood.note("long_quiet")
            self.assertEqual(rex_mood.current().events.count("long_quiet"), 1)

    def test_pacing_is_per_kind_not_global(self):
        with mock.patch.object(config, "REX_MOOD_DRIFT_MIN_INTERVAL_SECS", 600.0):
            rex_mood.ensure_today()
            rex_mood.note("long_quiet")
            rex_mood.note("complimented")
            events = rex_mood.current().events
            self.assertIn("long_quiet", events)
            self.assertIn("complimented", events)

    def test_pacing_expires(self):
        with mock.patch.object(config, "REX_MOOD_DRIFT_MIN_INTERVAL_SECS", 600.0):
            rex_mood.ensure_today()
            rex_mood.note("long_quiet")
            # Reach back in time rather than sleeping ten minutes.
            rex_mood._last_note_at["long_quiet"] -= 601.0
            rex_mood.note("long_quiet")
            self.assertEqual(rex_mood.current().events.count("long_quiet"), 2)

    def test_pacing_state_is_wiped_by_clear(self):
        with mock.patch.object(config, "REX_MOOD_DRIFT_MIN_INTERVAL_SECS", 600.0):
            rex_mood.ensure_today()
            rex_mood.note("long_quiet")
            self.assertTrue(rex_mood._last_note_at)
            rex_mood.clear()
            self.assertEqual(rex_mood._last_note_at, {})


class PromptTests(_MoodTestCase):

    def test_prompt_line_carries_the_state_and_bans_the_status_report(self):
        rex_mood.ensure_today()
        lines = rex_mood.prompt_lines()
        self.assertEqual(len(lines), 1, "stays one line — it rides on EVERY lean call")
        line = lines[0]
        self.assertIn("YOUR OWN STATE TODAY", line)
        self.assertIn(rex_mood.current().label, line)
        # The exact failure mode being locked out.
        self.assertIn("systems nominal", line)
        self.assertIn("normal parameters", line)
        self.assertIn("uptime", line)
        # The reciprocal case the owner actually hit.
        self.assertIn("bouncing your own question back", line)

    def test_prompt_line_marks_the_authored_answer_as_an_example_not_a_script(self):
        # Otherwise this just relocates "the same line every time" one level down.
        rex_mood.ensure_today()
        line = rex_mood.prompt_lines()[0]
        self.assertIn("EXAMPLE", line)
        self.assertIn("never a script", line)
        self.assertIn("freshly", line)

    def test_prompt_line_tells_him_not_to_make_it_the_topic(self):
        rex_mood.ensure_today()
        line = rex_mood.prompt_lines()[0]
        self.assertIn("Don't announce your mood unprompted", line)

    def test_reason_is_included_when_the_day_supplied_one(self):
        def newsy(now=None, allow_blocking=False):
            return (("chewing",), 'you\'ve had "a thing" rattling around all day')

        with mock.patch.object(rex_mood, "_SIGNALS", (("news", newsy),)):
            rex_mood.ensure_today()
        self.assertIn("rattling around all day", rex_mood.prompt_lines()[0])

    def test_drift_shows_up_as_how_the_day_has_gone(self):
        rex_mood.ensure_today()
        plain = rex_mood.prompt_lines()[0]
        for _ in range(4):
            rex_mood.note("insulted")
        drifted = rex_mood.prompt_lines()[0]
        self.assertNotEqual(plain, drifted)
        self.assertIn("chipping away", drifted)

    def test_classic_section_wraps_the_same_content(self):
        rex_mood.ensure_today()
        section = rex_mood.prompt_section()
        self.assertTrue(section.startswith("Rex's own state today:"))
        self.assertIn(rex_mood.prompt_lines()[0], section)

    def test_describe_reports_the_live_numbers(self):
        rex_mood.ensure_today()
        rex_mood.note("complimented")
        d = rex_mood.describe()
        self.assertEqual(d["label"], rex_mood.current().label)
        self.assertIn("complimented", d["events"])
        self.assertEqual(d["date"], rex_mood.current().date)


class SpokenLockTests(_MoodTestCase):

    def test_enrich_attaches_a_reason_to_a_causeless_mood(self):
        rex_mood.ensure_today()               # no signals -> no reason
        self.assertEqual(rex_mood.current().because, "")

        def late_weather(now=None, allow_blocking=False):
            return (("bright",), "it's clear out")

        with mock.patch.object(rex_mood, "_SIGNALS", (("weather", late_weather),)):
            self.assertTrue(rex_mood.enrich())
        self.assertEqual(rex_mood.current().because, "it's clear out")
        self.assertEqual(rex_mood.current().seed_kind, "weather")

    def test_enrich_will_not_retcon_a_mood_he_already_explained(self):
        rex_mood.ensure_today()
        rex_mood.note_spoken()

        def late_weather(now=None, allow_blocking=False):
            return (("bright",), "it's clear out")

        with mock.patch.object(rex_mood, "_SIGNALS", (("weather", late_weather),)):
            self.assertFalse(rex_mood.enrich())
        self.assertEqual(rex_mood.current().because, "")

    def test_enrich_leaves_an_existing_reason_alone(self):
        def newsy(now=None, allow_blocking=False):
            return (("chewing",), "the original reason")

        with mock.patch.object(rex_mood, "_SIGNALS", (("news", newsy),)):
            rex_mood.ensure_today()

        def other(now=None, allow_blocking=False):
            return (("bright",), "a different reason")

        with mock.patch.object(rex_mood, "_SIGNALS", (("weather", other),)):
            self.assertFalse(rex_mood.enrich())
        self.assertEqual(rex_mood.current().because, "the original reason")

    def test_a_line_voicing_the_mood_arms_the_lock(self):
        rex_mood.ensure_today()
        label = rex_mood.current().label
        self.assertTrue(rex_mood.note_spoken_if_voiced(f"Honestly? {label}, since you ask."))
        self.assertEqual(rex_mood.current().spoken, 1)

    def test_an_unrelated_line_does_not_arm_the_lock(self):
        rex_mood.ensure_today()
        self.assertFalse(rex_mood.note_spoken_if_voiced("The bass on that track is criminal."))
        self.assertEqual(rex_mood.current().spoken, 0)

    def test_label_matches_whole_tokens_only(self):
        # Review find 2026-08-05: the first cut substring-matched the label against a
        # sorted word-soup, so "worn" inside "sworn" locked the spoken flag off an
        # unrelated line — silently killing the unprompted share AND the enrich pass
        # for the rest of the day.
        worn = {"id": "worn", "label": "worn", "valence": -0.4, "energy": 0.2,
                "line": "Worn down, and awake for all of it.", "fits": ["any"]}
        with mock.patch.object(config, "REX_MOOD_SEEDS", [worn]):
            rex_mood.clear()
            rex_mood.ensure_today()
            self.assertFalse(
                rex_mood.note_spoken_if_voiced("I could have sworn you said Tuesday."))
            self.assertEqual(rex_mood.current().spoken, 0)
            self.assertTrue(
                rex_mood.note_spoken_if_voiced("Honestly? Worn out, since you ask."))

    def test_multi_word_labels_require_every_token(self):
        keyed = {"id": "keyed-up", "label": "keyed-up", "valence": 0.25, "energy": 0.95,
                 "line": "Something's about to happen and my systems have opinions.",
                 "fits": ["any"]}
        with mock.patch.object(config, "REX_MOOD_SEEDS", [keyed]):
            rex_mood.clear()
            rex_mood.ensure_today()
            self.assertFalse(rex_mood.note_spoken_if_voiced("The keyed lock jammed again."))
            self.assertTrue(rex_mood.note_spoken_if_voiced("Bit keyed up today, honestly."))

    def test_note_spoken_if_voiced_never_mints(self):
        # An idle line must not create the day's mood as a side effect.
        self.assertIsNone(rex_mood.current())
        self.assertFalse(rex_mood.note_spoken_if_voiced("buoyant bright flat worn patient"))
        self.assertIsNone(rex_mood.current())


class ShareCueTests(_MoodTestCase):
    """Volunteering the mood unprompted (owner 2026-08-05: "real people do that").
    rex_mood owns only "is there something worth saying" — the roll, the relationship
    gate, and the session cap live in interaction._lean_mood_share_cue."""

    def _force(self, seed: dict):
        return mock.patch.object(config, "REX_MOOD_SEEDS", [seed])

    def test_a_notable_mood_is_worth_mentioning(self):
        with self._force(_TEST_SEEDS[0]):          # buoyant, valence 0.8
            rex_mood.clear()
            self.assertTrue(rex_mood.is_notable())
            cue = rex_mood.share_cue()
        self.assertIsNotNone(cue)
        self.assertEqual(cue["label"], "buoyant")
        self.assertIn("Annoyingly good", cue["line"])

    def test_a_middling_mood_is_kept_to_himself(self):
        # Nobody volunteers "I feel exactly average."
        with self._force(_TEST_SEEDS[4]):          # patient, valence 0.3 / energy 0.4
            rex_mood.clear()
            self.assertFalse(rex_mood.is_notable())
            self.assertIsNone(rex_mood.share_cue())

    def test_low_energy_alone_makes_a_mood_mentionable(self):
        with self._force(_TEST_SEEDS[3]):          # worn, valence -0.4 / energy 0.2
            rex_mood.clear()
            self.assertLess(abs(rex_mood.ensure_today().valence),
                            float(config.REX_MOOD_SHARE_MIN_INTENSITY))
            self.assertTrue(rex_mood.is_notable())

    def test_notability_is_measured_live_so_drift_counts(self):
        # A bland morning that the day has since ground down becomes mentionable:
        # energy 0.55 is mid, but a very quiet afternoon takes it to the clamp at
        # 0.20, under REX_MOOD_SHARE_LOW_ENERGY.
        mid = {"id": "mid-day", "label": "even", "valence": 0.1, "energy": 0.55,
               "line": "Perfectly ordinary.", "fits": ["any"]}
        with self._force(mid):
            rex_mood.clear()
            self.assertFalse(rex_mood.is_notable())
            for _ in range(10):
                rex_mood.note("long_quiet")
            self.assertTrue(rex_mood.is_notable())
            self.assertIsNotNone(rex_mood.share_cue())

    def test_already_voiced_today_means_no_unprompted_share(self):
        # If he told you he was worn out when you ASKED, he doesn't then announce it.
        with self._force(_TEST_SEEDS[0]):
            rex_mood.clear()
            self.assertIsNotNone(rex_mood.share_cue())
            rex_mood.note_spoken()
            self.assertIsNone(rex_mood.share_cue())

    def test_the_spend_survives_a_restart(self):
        # The per-DAY gate is persisted, so rebooting this afternoon does not re-arm
        # an announcement he already made this morning.
        day = datetime(2026, 8, 5, 9, 0, 0)
        with self._force(_TEST_SEEDS[0]):
            rex_mood.clear()
            rex_mood.ensure_today(day)
            rex_mood.note_spoken()
            snap = rex_mood.snapshot_state()
            rex_mood.clear()
            self.assertTrue(rex_mood.restore_state(snap, now=day))
            self.assertIsNone(rex_mood.share_cue(day))

    def test_share_cue_carries_the_reason_and_the_shade(self):
        def newsy(now=None, allow_blocking=False):
            return (("chewing",), "you've had a thing rattling around all day")

        with self._force(_TEST_SEEDS[0]), \
             mock.patch.object(rex_mood, "_SIGNALS", (("news", newsy),)):
            rex_mood.clear()
            rex_mood.ensure_today()
            for _ in range(3):
                rex_mood.note("insulted")
            cue = rex_mood.share_cue()
        self.assertIn("rattling around", cue["because"])
        self.assertIn("chipping away", cue["shade"])

    def test_disabled_shares_nothing(self):
        with self._force(_TEST_SEEDS[0]), \
             mock.patch.object(config, "REX_MOOD_ENABLED", False):
            rex_mood.clear()
            self.assertIsNone(rex_mood.share_cue())
            self.assertFalse(rex_mood.is_notable())

    def test_thresholds_are_configurable(self):
        with self._force(_TEST_SEEDS[4]):          # patient — not notable by default
            rex_mood.clear()
            self.assertFalse(rex_mood.is_notable())
            with mock.patch.object(config, "REX_MOOD_SHARE_MIN_INTENSITY", 0.1):
                self.assertTrue(rex_mood.is_notable())

    def test_the_late_hour_taper_does_not_manufacture_notability(self):
        # The taper is a DELIVERY adjustment (don't claim to be wired at 1am), not a
        # property of the day. Letting it feed notability made the CLOCK a reason to
        # talk about himself: every mid-energy mood crossed the low bar after 8pm.
        mid = {"id": "mid-energy", "label": "even", "valence": 0.1, "energy": 0.34,
               "line": "Perfectly ordinary.", "fits": ["any"]}
        with self._force(mid):
            rex_mood.clear()
            late = datetime(2026, 8, 5, 23, 30, 0)
            rex_mood.ensure_today(late)
            # The taper genuinely drops the SPOKEN energy under the bar...
            self.assertLessEqual(rex_mood.effective_energy(late),
                                 float(config.REX_MOOD_SHARE_LOW_ENERGY))
            # ...but the day is still an ordinary one, so he keeps it to himself.
            self.assertFalse(rex_mood.is_notable(late))
            self.assertIsNone(rex_mood.share_cue(late))


class PersistenceTests(_MoodTestCase):

    def test_todays_mood_survives_a_restart_with_its_drift(self):
        day = datetime(2026, 8, 5, 9, 0, 0)
        mood = rex_mood.ensure_today(day)
        rex_mood.note("insulted", day)
        seed_id, drift = mood.seed_id, mood.drift_valence

        snap = rex_mood.snapshot_state()
        rex_mood.clear()
        self.assertTrue(rex_mood.restore_state(snap, now=day))

        resumed = rex_mood.current()
        self.assertEqual(resumed.seed_id, seed_id)
        self.assertAlmostEqual(resumed.drift_valence, drift, places=4)

    def test_yesterdays_mood_is_not_resumed_but_anti_repeat_memory_is(self):
        yesterday = datetime(2026, 8, 5, 9, 0, 0)
        mood = rex_mood.ensure_today(yesterday)
        snap = rex_mood.snapshot_state()
        rex_mood.clear()

        today = datetime(2026, 8, 6, 9, 0, 0)
        self.assertFalse(rex_mood.restore_state(snap, now=today))
        self.assertIsNone(rex_mood.current())
        # The memory that prevents waking up in the same mood two days running.
        self.assertIn(mood.seed_id, {r["seed_id"] for r in rex_mood._recent_ids})

    def test_a_removed_seed_is_dropped_rather_than_resurrected(self):
        day = datetime(2026, 8, 5, 9, 0, 0)
        stale = {
            "mood": {"date": "2026-08-05", "seed_id": "seed-that-no-longer-exists"},
            "recent": [{"date": "2026-08-04", "seed_id": "also-gone"}],
        }
        self.assertFalse(rex_mood.restore_state(stale, now=day))
        self.assertIsNone(rex_mood.current())
        self.assertEqual(rex_mood._recent_ids, [])

    def test_seed_wording_is_re_read_from_the_pool_not_the_disk(self):
        # Editing a seed's line should take effect immediately, not next time the
        # state file happens to be rewritten.
        day = datetime(2026, 8, 5, 9, 0, 0)
        rex_mood.ensure_today(day)
        snap = rex_mood.snapshot_state()
        seed_id = rex_mood.current().seed_id
        rex_mood.clear()

        edited = [dict(s, line="REWRITTEN.", label="rewritten") if s["id"] == seed_id
                  else s for s in _TEST_SEEDS]
        with mock.patch.object(config, "REX_MOOD_SEEDS", edited):
            self.assertTrue(rex_mood.restore_state(snap, now=day))
            self.assertEqual(rex_mood.current().line, "REWRITTEN.")
            self.assertEqual(rex_mood.current().label, "rewritten")

    def test_garbage_snapshots_are_survivable(self):
        for junk in (None, [], "nope", {"mood": "not-a-dict"}, {}):
            self.assertFalse(rex_mood.restore_state(junk))

    def test_file_round_trip_with_a_temp_path(self):
        # A non-default path makes _file_io_suppressed() False, so real file I/O runs
        # even under the test runner.
        day = datetime(2026, 8, 5, 9, 0, 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rex_mood_state.json"
            with mock.patch.object(config, "REX_MOOD_STATE_PATH", str(path)):
                rex_mood.ensure_today(day)
                rex_mood.note("complimented", day)
                seed_id = rex_mood.current().seed_id
                rex_mood.persist()
                self.assertTrue(path.exists())
                # Atomic write leaves no .tmp sibling behind.
                self.assertFalse(path.with_suffix(".json.tmp").exists())
                self.assertIn("mood", json.loads(path.read_text()))

                rex_mood.clear()
                self.assertTrue(rex_mood.load_persisted(now=day))
                self.assertEqual(rex_mood.current().seed_id, seed_id)

    def test_default_path_file_io_is_suppressed_under_the_test_runner(self):
        self.assertTrue(rex_mood._file_io_suppressed())
        path = rex_mood._default_state_path()
        before = path.read_bytes() if path.exists() else None
        rex_mood.ensure_today()
        rex_mood.persist()
        after = path.read_bytes() if path.exists() else None
        self.assertEqual(before, after, "the suite must never write the real state file")

    def test_state_path_and_default_path_compare_equal_by_construction(self):
        # If these ever diverge (one resolving, the other not), _file_io_suppressed
        # silently fails open and the suite writes the robot's real mood file.
        with mock.patch.object(config, "REX_MOOD_STATE_PATH", None):
            self.assertEqual(rex_mood._state_path(), rex_mood._default_state_path())

    def test_relative_override_resolves_against_the_project_root(self):
        with mock.patch.object(config, "REX_MOOD_STATE_PATH", "assets/memory/x.json"):
            self.assertTrue(rex_mood._state_path().is_absolute())
            self.assertTrue(str(rex_mood._state_path()).endswith("assets/memory/x.json"))

    def test_persist_is_a_noop_when_persistence_is_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rex_mood_state.json"
            with (mock.patch.object(config, "REX_MOOD_STATE_PATH", str(path)),
                  mock.patch.object(config, "REX_MOOD_PERSIST_ENABLED", False)):
                rex_mood.ensure_today()
                rex_mood.persist()
                self.assertFalse(path.exists())


class ShippedPoolTests(unittest.TestCase):
    """Guards on the AUTHORED pool, validated against the shipped values captured at
    import time — not the test fixture."""

    def test_pool_is_big_enough_to_stay_interesting(self):
        self.assertGreaterEqual(len(_SHIPPED_SEEDS), 12)

    def test_ids_are_unique(self):
        ids = [s["id"] for s in _SHIPPED_SEEDS]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_seed_is_well_formed(self):
        for seed in _SHIPPED_SEEDS:
            with self.subTest(seed=seed.get("id")):
                self.assertTrue(str(seed.get("id") or "").strip())
                self.assertTrue(str(seed.get("label") or "").strip())
                self.assertTrue(str(seed.get("line") or "").strip())
                self.assertGreaterEqual(float(seed["valence"]), -1.0)
                self.assertLessEqual(float(seed["valence"]), 1.0)
                self.assertGreaterEqual(float(seed["energy"]), 0.0)
                self.assertLessEqual(float(seed["energy"]), 1.0)

    def test_pool_spans_good_and_bad_days(self):
        valences = [float(s["valence"]) for s in _SHIPPED_SEEDS]
        self.assertGreater(max(valences), 0.4, "needs genuinely good days")
        self.assertLess(min(valences), -0.2, "needs genuinely off days")
        energies = [float(s["energy"]) for s in _SHIPPED_SEEDS]
        self.assertGreater(max(energies), 0.7)
        self.assertLess(min(energies), 0.3)

    def test_no_seed_answers_with_a_status_report(self):
        # The exact phrasing this whole feature exists to replace.
        banned = ("systems nominal", "normal parameters", "all systems",
                  "fully operational", "diagnostic")
        for seed in _SHIPPED_SEEDS:
            line = str(seed.get("line") or "").lower()
            for phrase in banned:
                with self.subTest(seed=seed["id"], phrase=phrase):
                    self.assertNotIn(phrase, line)

    def test_lines_are_short_enough_to_say_out_loud(self):
        for seed in _SHIPPED_SEEDS:
            with self.subTest(seed=seed["id"]):
                self.assertLessEqual(len(str(seed["line"]).split()), 25)

    def test_drift_table_kinds_are_all_two_tuples(self):
        for kind, delta in (getattr(config, "REX_MOOD_DRIFT", {}) or {}).items():
            with self.subTest(kind=kind):
                self.assertEqual(len(tuple(delta)), 2)


if __name__ == "__main__":
    unittest.main()
