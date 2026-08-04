# Presidential voice references

Fourteen `assets/voices/famous/<slug>.{wav,txt}` reference clips for the
impersonation feature (`features/impersonation.py`), covering deceased US
presidents from Theodore Roosevelt to George H. W. Bush.

Rebuild with:

```bash
venv/bin/python tools/build_presidential_refs.py
```

The clips are gitignored (`.gitignore:247`, `assets/voices/*` minus `rex/`), so
the build script — not the audio — is the tracked artifact. Sources are cached
under `assets/voices/_src_presidents/` (also gitignored); `--no-fetch` rebuilds
from that cache without re-downloading.

## Scope: deceased presidents only

Every voice here is of someone dead. That is the line this set is drawn on, and
it is worth keeping if the set is ever extended: a convincing clone of a *living*
public figure — especially a serving politician — is a disinformation tool, and
the interesting uses of one are mostly bad. Dead presidents are the long-standing
territory of comic impressions, and the audio is public-domain government work.

Two consequences worth knowing before adding anyone:

- **Nothing before ~1900 is possible.** Audio recording did not exist for
  Washington through Arthur, and no authentic recording of McKinley survives.
  (The Library of Congress item that looks like McKinley's last speech is Len
  Spencer, a contemporary recording artist, *reciting* it — cloning that would
  put an actor's voice behind a president's name.) `find_famous_ref` already
  returns a graceful refusal for anyone missing, so Lincoln simply gets
  "I'd need to actually hear Abraham Lincoln first."
- **Calvin Coolidge is missing** despite being findable in principle — he made
  ~40 radio broadcasts and was the first president to use the medium, but LOC
  search for him is swamped by the unrelated "Coolidge Auditorium" (named for
  Elizabeth Sprague Coolidge), and no clean solo item surfaced. Worth another try.

## Licensing basis

- **Miller Center** (`millercenter.org`, 10 of 14): presidential addresses are
  US Government works, public domain under 17 U.S.C. §105.
- **Internet Archive** (Hoover 1932, T. Roosevelt 1901): both items carry an
  explicit `creativecommons.org/licenses/publicdomain/` mark.
- **LOC National Jukebox** (Wilson 1912, Taft 1912): these are 1912 Victor
  discs. The collection-wide rights note describes a permission arrangement with
  the rightsholder, but sound recordings fixed before 1 Jan 1923 entered the
  public domain on 1 Jan 2022 under the Music Modernization Act, which covers
  both of these.

## Picking a span

Each span was chosen from a word-timestamped Whisper scan, and must be
**contiguous solo speech by the president himself**. Several sources are not
that for their first minute, and a naive "take the first 20 seconds" would clone
the wrong person outright:

| Source | Who else is on it |
|---|---|
| Eisenhower 1961 | radio announcer, 4–12 s |
| Kennedy 1961 | Chief Justice Warren administering the oath, 0–31 s |
| Ford 1974 | Chief Justice Burger administering the oath, 14–59 s |
| T. Roosevelt 1901 | modern Vincent Voice Library narrator, 73–79 s |

Applause is the other hazard — LBJ and Bush are before live audiences, so the
spans sit inside contiguous stretches with only rhetorical pauses. `spe_1989_0120_bush.mp3`
also runs 63 minutes and contains more than the inaugural; the chosen span is
inside the inaugural address proper.

Targets: 15–22 s (Rex's own reference is 19.5 s), complete sentences, ending on a
sentence boundary where possible.

## Verification

`tools/build_presidential_refs.py` re-transcribes each **final** clip rather than
reusing the scan text, so the `.txt` provably describes the audio the cloner is
handed. A round-trip check (synthesize a fixed line in each voice, transcribe it
back) scored 94–100% word recall across all fourteen at ~1.0x real time — the
94% cases were `I am` rendered as `I'm`, not errors. The 1912 acoustic discs and
the 1901 cylinder clone as intelligibly as the broadcast-era material, which was
not a given.

## Aliases

`find_famous_ref` matches an exact slug first, then loosely on the surname token,
so nicknames without a surname in them would miss. These are symlinked:

    fdr -> franklin-roosevelt      jfk -> john-kennedy      ike -> dwight-eisenhower
    lbj -> lyndon-johnson          teddy-roosevelt -> theodore-roosevelt

Note that "Roosevelt" alone is genuinely ambiguous between Franklin and
Theodore; the loose match breaks the tie by glob order. Say which one.
