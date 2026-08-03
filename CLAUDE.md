# DJ-R3X v2 — notes for Claude Code sessions

## Testing

Run tests **per module**, never with one-process discover:

```bash
venv/bin/python -m unittest tests.test_<module>
```

`unittest discover` across the whole tests/ tree in one process is NOT reliable
here and has burned hours repeatedly:

- It **hangs** at `test_local_tts.SpeakLocalPlaybackTest` — real audio playback
  wedges under AEC/speech-queue state left behind by earlier modules.
- Cross-module state leaks flip results: `intelligence/interaction.py` keeps
  proactive-speech pacing in module globals, and a test that touches them
  changes the next module's answers (`tests/_lean_impulse_state.py` exists for
  exactly this — call `reset_impulse_state(self)` in setUp of any test driving
  `_maybe_lean_impulse`). `test_idle_head_wander` and `test_listening_motion`
  pass alone but fail under discover.

### Known pre-existing failures (as of 2026-08-02)

These fail on a clean checkout of main, per-module, and are NOT caused by your
change (tracked as their own fix-up task):

- `tests/test_audio_and_conversation_gating.py` — 3 failures
  (furry-animal surprise frame, direct-shutdown LED clip, +1)
- `tests/test_body_mood.py` — `test_visor_released_to_lens_clear_floor_when_mood_ends`
- `tests/test_holiday_plan_cue.py` — `test_spoken_lean_holiday_cue_marks_only_that_person`

**Before investigating any failure as a regression, check it against HEAD
first** (`git stash -u && venv/bin/python -m unittest tests.<module> && git
stash pop`). If it fails there too, it is pre-existing — note it, move on, and
do not spend time on it unless the user asks.

### Date-rot

Several fixtures hardcode event dates ("2026-06-01") that silently cross the
`FOLLOWUP_DATED_MAX_AGE_DAYS` staleness horizon as real time passes, breaking
tests months after they were written. Write fixture dates relative to
`date.today()`. If a followup/plan test fails on a date comparison, suspect
rot before suspecting the code.
