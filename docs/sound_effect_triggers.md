# Expressive sound effects

The main app selects one accent for a spoken line. Explicit event cues take
precedence over delivery tags, comedy modes, and the usual emotion chirp.

| Clip / group | Trigger |
| --- | --- |
| `Droid_Affectionate` (`warm`) | A direct affectionate turn such as “I love you, Rex” or “You're my best friend.” |
| `Droid_Sassy_smug` (`sarcastic`) | Friendly-roast or smug-superiority comedy, an explicit roast performance, or a `[sarcastic]` delivery cue. |
| `Droid_Confused_wobbly` | Misheard/misunderstood repair replies and a requested confused pose. |
| `Droid_Disappointed` | Rex exhausts his guesses and loses 20 Questions. |
| `Droid_Scared_startled` | An admitted startle reaction to a bang, crash, scream, or other configured startle sound. |
| `Droid_Embarrassed` | Self-own comedy, acknowledging a factual/other repair, or a requested embarrassed pose. |
| `Droid_Mischievous` | Appliance-conspiracy comedy or a `[mischievously]` delivery cue. |
| `Droid_Goodbye_gentle` | The first reply to the current explicit farewell, as recognized by `end_thread`. Soft topic closures do not qualify. |
| `Error_buzz,_short` | Fake-system-error comedy or the fallback for a failed generated performance. |
| `Droid_Laughing_rapid_1/2` | Joke punchlines, `[laughs]` delivery cues, and the existing transition into an amused body mood. The two recordings rotate. |
| `angry_robot` | A detected insult: the fast detector attaches it to the first reply chunk; a later semantic detection can use the offended-mood transition. |

Joke setups suppress their generic happy chirp so they do not consume the
laughter cooldown. Affection and fast insult reactions similarly reserve their
accent for the queued reply rather than spending the cooldown on an early mood
chirp. Explicit cues stay attached to their speech item: a dropped or stale item
does not play its cue. Rex's 20 Questions loss is an immediate event accent.

The existing sound enable flags, speech-family cooldown (6 seconds, with a
12-second same-key interval), speaker-busy checks, and listening-window protection
still apply. Impersonation-tagged speech remains chirp-free. Serious speech
emotions take precedence over comedy accents. No extra model calls are needed.

Restart the main app after code or same-filename audio replacements; decoded
sounds are cached for the process lifetime.

## Clip loudness

All 68 MP3s in `assets/audio/sound_effects/` (including `thinking/` and
`excitement/`) were normalized toward **-20 LUFS in the mono mix used by Rex**.
The measured range is now -20.73 to -19.91 LUFS, versus -43.63 to -9.46 before.
This target is close to the original library's median (-19.39 LUFS).

Processing uses a constant gain per clip and a lookahead peak limiter at 192 kHz,
then encodes from the original source to high-quality MP3. Every encoded result
was remeasured: the highest true peak is -2.27 dBTP. Filenames, sample counts,
48 kHz sample rate, and both stereo channels are preserved. No timing edits or
silence trimming were applied. Existing playback gain and cooldown settings
continue to apply.

[Per-file before/after measurements](sound_effect_loudness.csv) include the final
file hashes. Newly replaced clips should be measured against the same profile.
