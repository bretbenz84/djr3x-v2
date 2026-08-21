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

On the robot Mac, tests drive the REAL Maestro, so a sweep leaves the head in a
random pose and the next boot jerks from it. `tests/test_zzz_servo_park.py`
(named to sort last) glides everything back to the shutdown rest pose — always
let a full sweep include it, and run it manually after any servo-touching test
session:

```bash
venv/bin/python -m unittest tests.test_zzz_servo_park
```

### Known pre-existing failures (as of 2026-08-02)

These fail on a clean checkout of main, per-module, and are NOT caused by your
change (tracked as their own fix-up task):

- `tests/test_audio_and_conversation_gating.py` — 3 failures
  (`test_direct_shutdown_clip_leaves_leds_off`,
  `test_existing_common_name_prompt_logs_human_turn_before_returning`,
  `test_first_sight_sparse_profile_uses_basic_profile_question` — the third
  verified pre-existing against clean HEAD 2026-08-05, solo run too; the
  furry-animal surprise-frame failure was fixed 2026-08-03 by the animal
  presence-ledger rework — the old species cooldown was leaking between tests)
- `tests/test_face_tracking.py` — FIXED 2026-08-07 (the 4 failures were test rot:
  assertions compared against the neck neutral 5472 while `_set_servo_positions`
  starts the head at 6000 — updated alongside the head-tilt anti-hunting work)
- `tests/test_body_mood.py` — `test_visor_released_to_lens_clear_floor_when_mood_ends`
- `tests/test_holiday_plan_cue.py` — `test_spoken_lean_holiday_cue_marks_only_that_person`
- `tests/test_rex_supervisor.py` — 3 failures (`test_chime_uses_afplay_when_available`,
  `test_consecutive_frame_default_and_override`, `test_threshold_env_override`;
  verified against clean HEAD 2026-08-04)

Found by a full per-module sweep of all 241 modules on 2026-08-05 and each verified
against clean HEAD the same day:

- `tests/test_lean_memory_musing.py` — `test_spoken_musing_sets_once_per_session_flag`
- `tests/test_proactive_discipline.py` — `test_idle_monologue_is_excluded_from_the_cooldown`
- `tests/test_pose_face_guard.py` — `test_no_pose_anchor_keeps_all`
- `tests/test_rfdetr_backend.py` — `test_object_records_apply_exclusions`
- `tests/test_speaker_challenge.py` — `test_cold_signature_needs_strict_bar`
- `tests/test_vision_panel_skeleton.py` — 2 failures
  (`test_coarse_hand_points_render`,
  `test_skeleton_does_not_bleed_past_video_edges`)

Two more appeared between then and 2026-08-20 (the tree is 284 modules now, not
241). Both verified pre-existing that day by swapping the pre-batch
`intelligence/action_router.py` back in and re-running — they fail identically:

- `tests/test_reaction_awareness.py` — 3 failures in `NewsDigestContractTests`
  (`test_bans_the_closing_fetch_menu`, `test_bans_the_press_release_tics`,
  `test_caps_the_spoken_length`)
- `tests/test_review_regressions.py` —
  `test_router_keeps_known_named_person_topic_as_memory_query`

**A full sweep should now come back with exactly these 12 modules and nothing
else** (sweep of all 284 modules, 2026-08-20, 390s).

Note the trap that cost time on 2026-08-20: `git stash -u` is a NO-OP when your
work is already committed, so the usual "stash, test, pop" baseline check silently
re-tests your own code and every failure looks pre-existing. When the tree is
clean, compare against a specific commit instead — swap the single file back in
with `git show <base>:path > path`, test, then restore. A `git worktree` checkout
is NOT a good baseline here: it lacks the untracked `.env` / `apikeys.py` /
`assets/`, so modules fail to import for reasons that have nothing to do with the
change.

A full sweep is easiest as a small Python runner that shells out per module with a
timeout (macOS has no `timeout(1)`), skipping `test_local_tts`. Budget ~12 minutes.

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
