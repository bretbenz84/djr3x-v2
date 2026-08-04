"""Shared reset for the lean-impulse module globals.

`intelligence.interaction` keeps the proactive-speech pacing state in MODULE
globals: unanswered-run counters, cooldown anchors, the engagement-probe /
snooze latches, and a rolling rate-cap deque. A test that drives
`_maybe_lean_impulse` writes several of them, and unittest runs every module in
ONE process — so whatever a test leaves behind silently changes the next test's
answer.

That is not hypothetical: `ImpulseReengageTest` left `_engagement_probe_at` set
(the probe path arms it), which made every later `_maybe_lean_impulse` return
False at the "probe outstanding — wait quietly" gate, and it appended to
`_lean_impulse_spoken_times` until the 5-per-window rate cap tripped. Between
them those two leaks broke 8 tests across three other modules, all of which
passed in isolation.

Call `reset_impulse_state(self)` from setUp. It snapshots every one of those
globals, restores them on cleanup, and zeroes them for the test.
"""

from __future__ import annotations

# Scalar globals: (name, value to install for the test).
_SCALARS = {
    "_engagement_probe_at": 0.0,
    "_engagement_probed_this_silence": False,
    "_impulse_snooze_until": 0.0,
    "_impulse_snooze_reason": "",
    "_impulse_snooze_person": None,
    "_consecutive_lean_impulses": 0,
    "_last_lean_impulse_at": 0.0,
    "_last_proactive_line_at": 0.0,
    "_floor_held_until": 0.0,
    "_last_user_content_at": 0.0,
    "_last_rex_line_was_question": False,
    "_lean_memory_mused_this_session": False,
    "_lean_news_mentioned_this_session": False,
    "_awaiting_followup_event": None,
}

# Mutable containers must be restored IN PLACE — other modules hold references.
_CONTAINERS = ("_lean_impulse_spoken_times",)
# Dict-shaped state, also restored in place. `_lean_cue_cooldowns` joined the
# globals with the cue-drop-cooldown feature (2026-08-02): a test that drops a
# generated line benches its cue for 600s of MOCKED time, which silently
# starved the same cue in every later test (2 wiring tests broke module-wide
# while passing in isolation — the exact leak this helper exists to stop).
_DICTS = ("_lean_cue_cooldowns", "_workday_checkin_rolls")


def reset_impulse_state(testcase) -> None:
    """Snapshot + restore the lean-impulse globals around one test."""
    from intelligence import interaction as I

    for name, fresh in _SCALARS.items():
        if not hasattr(I, name):
            continue
        saved = getattr(I, name)
        testcase.addCleanup(lambda n=name, v=saved: setattr(I, n, v))
        setattr(I, name, fresh)

    for name in _CONTAINERS:
        container = getattr(I, name, None)
        if container is None:
            continue
        saved_items = list(container)
        testcase.addCleanup(
            lambda c=container, v=saved_items: c.__setitem__(slice(None), v)
        )
        container[:] = []

    def _restore_dict(d, items):
        d.clear()
        d.update(items)

    for name in _DICTS:
        mapping = getattr(I, name, None)
        if mapping is None:
            continue
        saved_map = dict(mapping)
        testcase.addCleanup(lambda d=mapping, v=saved_map: _restore_dict(d, v))
        mapping.clear()
