"""
Impersonation — Rex clones a voice and performs a short, affectionate parody.

"Rex, do an impersonation of me / of Jimmy Carter." Rex clones a voice (via the
selected on-device engine, audio/local_tts.py) and delivers a brief LLM-written
parody in that voice, framed by his own stall/outro lines. Two reference sources:

  * Known people — accepted voice-enrollment recordings or ordinary live speech, saved
    under VOICES_DIR/people/<person_id>.{wav,txt,json}. The parody script is mined
    from that person's memory, with sensitive/boundary topics hard-excluded.
  * Famous people — user-supplied VOICES_DIR/famous/<slug>.{wav,txt} clips; the
    script comes from the LLM's general knowledge.

This module owns the target resolution, the reference files, the parody-script
generation (memory read + boundary exclusion + one-off LLM call), and the spoken
performance. interaction.py owns only the router dispatch and the live-capture
pending slot; it calls resolve_target() / perform() / save_person_capture() here.
"""

from __future__ import annotations

import json
import hashlib
import math
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

import config
from audio import local_tts

logger = logging.getLogger(__name__)

_SELF_WORDS = {"me", "myself", "i", "my", "speaker", "my voice", "myvoice", "yourself of me"}
_CANCEL_RE = re.compile(
    r"^\s*(?:no(?:\s+thanks)?|nope|not now|later|never ?mind|forget it|stop|cancel|wait|hold on|"
    r"don'?t\s+(?:do|want|keep)|actually[,\s]+no|(?:i'?m|we'?re)\s+done|"
    r"he'?s\s+not\s+(?:gonna|going to)\s+be\s+able)\b",
    re.IGNORECASE,
)
_STOPWORDS = {"the", "a", "an", "of", "mr", "mrs", "ms", "dr", "president", "sir", "madam"}

# Comic angles for a famous-person bit, one picked at random per request.
# Temperature alone kept landing the model on the same "greatest hit" joke for a
# given figure (the avoid-list only rules out what was said, not the lane it was
# said in), so the lane is chosen for it.
#
# None of these mention a party. The earlier set did, and the model took it as a
# standing fact about the world — every bit came back about partygoers, snacks
# and the dance floor even when Rex was sitting in a quiet room (field
# 2026-08-04). The only setting the script may assume is that a droid is doing
# an impression and someone is listening.
_FAMOUS_ANGLES = (
    "bend their single most-quoted line so it lands on the droid",
    "have them lodge a dignified complaint about a machine borrowing their voice",
    "have them address the droid directly, as an inferior they intend to instruct",
    "have them try, and fail, to stay dignified about being mimicked by a machine",
    "have them take entirely too much credit for how good the impression is",
    "have them campaign — earnestly, to nobody who asked — for something absurd",
    "have them treat the impression as a matter of national importance",
    "have them be delighted, and slightly too impressed with the technology",
)


# ── Paths / availability ──────────────────────────────────────────────────────

def _project_root() -> Path:
    return Path(config.__file__).resolve().parent


def _voices_dir() -> Path:
    return _project_root() / getattr(config, "VOICES_DIR", "assets/voices")


def is_enabled() -> bool:
    """Feature on AND the on-device engine installed (ElevenLabs can't clone)."""
    return bool(getattr(config, "IMPERSONATION_ENABLED", True)) and local_tts.is_available()


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower())
    return s.strip("-")


# ── Reference clips ───────────────────────────────────────────────────────────

def person_ref(person_id: int) -> Optional[local_tts.VoiceRef]:
    """Use an existing clone reference, then accepted enrollment recordings."""
    base = _voices_dir() / "people"
    ref = local_tts.voice_ref_from_files(
        base / f"{person_id}.wav", base / f"{person_id}.txt", f"person:{person_id}"
    )
    if ref is not None:
        try:
            meta = json.loads((base / f'{person_id}.json').read_text())
        except (OSError, ValueError):
            meta = {}
        if meta.get('complete') is not False:
            return ref
    return enrollment_ref(person_id)


def enrollment_ref(person_id: int) -> Optional[local_tts.VoiceRef]:
    """Build a derived reference from verified originals; never enroll from it.

    Full clips keep their actual transcripts. Content-addressed files live in
    the speech cache and are invalidated when the source voice data is cleared.
    No recognizer or synthesis model is loaded to prepare this reference.
    """
    from memory import voice_recordings
    from scipy.signal import resample_poly
    try:
        takes, texts, digests = [], [], []
        voiced = duration = 0.
        sr = int(config.AUDIO_SAMPLE_RATE)
        rows = voice_recordings.list_recordings(int(person_id))
        for row in rows:
            meta = json.loads(row['metadata'])
            text = str(meta.get('transcript') or '').strip()
            if (meta.get('source') != 'conversational_enrollment'
                    or meta.get('anchor_and_cluster_verified') is not True or not text):
                continue
            rate, audio = voice_recordings.decode(row)
            if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all() or rate <= 0:
                continue
            if np.issubdtype(audio.dtype, np.integer):
                audio = audio.astype(np.float32) / max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
            audio = np.asarray(audio, dtype=np.float32)
            if rate != sr:
                divisor = math.gcd(int(rate), sr)
                audio = resample_poly(audio, sr // divisor, int(rate) // divisor)
            secs = float(meta.get('voiced_secs') or 0.)
            if (not math.isfinite(secs) or secs < 2. or secs > len(audio) / sr + .1
                    or duration + len(audio)/sr > 20.):
                continue
            takes.append(_trim_silence(audio, sr)); texts.append(text); digests.append(row['digest'])
            voiced += secs; duration += len(audio)/sr
            if voiced >= float(config.IMPERSONATION_CAPTURE_MIN_VOICED_SECS):
                break
        if voiced < float(config.IMPERSONATION_CAPTURE_MIN_VOICED_SECS):
            return None
        key = hashlib.sha256((str(sr) + ':'.join(digests)).encode()).hexdigest()[:16]
        base = Path(config.TTS_CACHE_DIR) / 'voice_refs'
        base.mkdir(parents=True, exist_ok=True)
        stem = base / f'person-{int(person_id)}-{key}'
        wav, txt = stem.with_suffix('.wav'), stem.with_suffix('.txt')
        if not wav.exists() or not txt.exists():
            import soundfile as sf
            gap = np.zeros(int(.3 * sr), dtype=np.float32)
            joined = np.concatenate([piece for n, take in enumerate(takes)
                                     for piece in ([gap, take] if n else [take])])
            sf.write(str(wav), _pad_tail(joined, sr), sr, subtype='PCM_16')
            txt.write_text(' '.join(texts), encoding='utf-8')
        logger.info('[impersonation] using enrollment recordings person_id=%s clips=%d voiced=%.2f',
                    person_id, len(takes), voiced)
        return local_tts.voice_ref_from_files(wav, txt, f'person:{person_id}')
    except Exception as exc:
        logger.warning('[impersonation] enrollment reference unavailable for %s: %s', person_id, exc)
        return None


def capture_progress(person_id):
    """A saved incomplete sample can survive cancellation, expiry and restart."""
    if person_id is None:
        return None
    base = _voices_dir() / 'people'
    try:
        meta = json.loads((base / f'{int(person_id)}.json').read_text())
        if meta.get('complete') is not False:
            return None
        import soundfile as sf
        audio, rate = sf.read(base / f'{int(person_id)}.wav', dtype='float32')
        text = (base / f'{int(person_id)}.txt').read_text().strip()
        if rate != int(config.AUDIO_SAMPLE_RATE) or audio.ndim != 1 or not text:
            return None
        return {'takes': [audio], 'take_texts': [text], 'voiced_secs': float(meta['voiced_secs'])}
    except (OSError, ValueError, TypeError, KeyError):
        return None


def find_famous_ref(name: str) -> Optional[local_tts.VoiceRef]:
    """Locate a user-supplied famous-person clip by name. Tries an exact slug, then
    a loose surname match (last name token present in a candidate slug)."""
    base = _voices_dir() / "famous"
    if not base.exists():
        return None
    slug = slugify(name)
    if slug:
        exact = local_tts.voice_ref_from_files(
            base / f"{slug}.wav", base / f"{slug}.txt", f"famous:{slug}"
        )
        if exact is not None:
            return exact
    # Loose match: pick a candidate whose token set contains the query's last
    # (surname) token — so "Carter" / "President Carter" both find "jimmy-carter".
    q_tokens = [t for t in slug.split("-") if t and t not in _STOPWORDS]
    if not q_tokens:
        return None
    surname = q_tokens[-1]
    if len(surname) < 3:
        return None
    best = None
    best_overlap = 0
    for wav in base.glob("*.wav"):
        cand = wav.stem
        cand_tokens = set(cand.split("-"))
        if surname in cand_tokens:
            overlap = len(set(q_tokens) & cand_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best = cand
    if best is None:
        return None
    return local_tts.voice_ref_from_files(
        base / f"{best}.wav", base / f"{best}.txt", f"famous:{best}"
    )


def _trim_silence(arr: np.ndarray, sr: int) -> np.ndarray:
    """Strip sub-threshold head/tail from a capture before it becomes a ref.

    Capture segments ride in padded buffers: seconds of room tone (and Rex's
    own −17 dB AEC residual) bracket the take, and the cloner treats all of it
    as "this is what the voice sounds like". Keeps a small margin each side;
    returns the array unchanged when nothing voiced is found."""
    if arr.size == 0:
        return arr
    frame = max(1, int(0.03 * sr))
    usable = (arr.size // frame) * frame
    if usable <= 0:
        return arr
    frames = np.abs(arr[:usable].astype(np.float32)).reshape(-1, frame)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    peak = float(rms.max()) if rms.size else 0.0
    if peak <= 0.0:
        return arr
    floor = max(0.004, 0.1 * peak)
    voiced = np.flatnonzero(rms >= floor)
    if voiced.size == 0:
        return arr
    margin = int(0.15 * sr)
    start = max(0, int(voiced[0]) * frame - margin)
    end = min(arr.size, (int(voiced[-1]) + 1) * frame + margin)
    return arr[start:end]


def _pad_tail(arr: np.ndarray, sr: int) -> np.ndarray:
    """Ensure the clip ends with at least IMPERSONATION_CAPTURE_END_PAD_SECS of
    silence, so the clone isn't clipped on the final phoneme. Measures the trailing
    silence already present and only appends the shortfall."""
    target = int(round(float(getattr(config, "IMPERSONATION_CAPTURE_END_PAD_SECS", 0.5)) * sr))
    if target <= 0 or arr.size == 0:
        return arr
    # Where does audible signal end? (float32 in [-1, 1]; ~-60 dBFS threshold.)
    nonsilent = np.flatnonzero(np.abs(arr) > 1e-3)
    trailing = arr.size - (int(nonsilent[-1]) + 1) if nonsilent.size else arr.size
    shortfall = target - trailing
    if shortfall <= 0:
        return arr
    return np.concatenate([arr, np.zeros(shortfall, dtype=np.float32)])


def save_person_capture(
    person_id: Optional[int], audio_array: np.ndarray, transcript: str, *,
    voiced_secs: Optional[float] = None, complete: bool = True,
) -> Optional[local_tts.VoiceRef]:
    """Persist a live capture as the person's reference clip: a 16 kHz mono PCM_16
    WAV + the transcript + a JSON sidecar. Returns a VoiceRef, or None on failure.

    audio_array is float32 mono in [-1, 1] at config.AUDIO_SAMPLE_RATE (16000 Hz);
    the clone resamples it internally, so 16 kHz as-is is fine. The saved transcript
    is what was ACTUALLY said (describes the audio), which is what the clone needs.

    person_id=None is an anonymous guest: the ref is written to a session-scoped
    slot ("anon-latest", overwritten each capture) so a stranger can be cloned for
    the bit without minting a durable per-person voice file.
    """
    ref_text = " ".join((transcript or "").split())
    if not ref_text:
        return None
    base = _voices_dir() / "people"
    base.mkdir(parents=True, exist_ok=True)
    stem = "anon-latest" if person_id is None else str(person_id)
    label = "person:anon" if person_id is None else f"person:{person_id}"
    wav_path = base / f"{stem}.wav"
    txt_path = base / f"{stem}.txt"
    json_path = base / f"{stem}.json"
    sr = int(getattr(config, "AUDIO_SAMPLE_RATE", 16000))
    try:
        import soundfile as sf
        arr = np.asarray(audio_array, dtype=np.float32).reshape(-1)
        arr = _trim_silence(arr, sr)
        arr = _pad_tail(arr, sr)
        sf.write(str(wav_path), arr, sr, subtype="PCM_16")
        txt_path.write_text(ref_text, encoding="utf-8")
        json_path.write_text(
            json.dumps({
                "person_id": person_id,
                "duration_secs": round(len(arr) / float(sr), 2),
                "transcript": ref_text,
                "voiced_secs": voiced_secs,
                "complete": complete,
            }),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("[impersonation] failed to save capture for %s: %s", person_id, exc)
        return None
    return local_tts.voice_ref_from_files(wav_path, txt_path, label)


def save_person_capture_parts(
    person_id: Optional[int],
    takes: "list[np.ndarray]",
    transcripts: "list[str]",
    *, voiced_secs: Optional[float] = None, complete: bool = True,
) -> Optional[local_tts.VoiceRef]:
    """Join accepted speech clips and their actual transcripts into one reference."""
    takes = [t for t in (takes or []) if isinstance(t, np.ndarray) and t.size]
    if not takes:
        return None
    sr = int(getattr(config, "AUDIO_SAMPLE_RATE", 16000))
    gap = np.zeros(int(0.3 * sr), dtype=np.float32)
    pieces: "list[np.ndarray]" = []
    for i, take in enumerate(takes):
        if i:
            pieces.append(gap)
        pieces.append(_trim_silence(np.asarray(take, dtype=np.float32).reshape(-1), sr))
    joined = np.concatenate(pieces)
    transcript = " ".join(
        " ".join(str(t or "").split()) for t in (transcripts or []) if str(t or "").strip()
    )
    return save_person_capture(person_id, joined, transcript, voiced_secs=voiced_secs, complete=complete)


# ── Lines ─────────────────────────────────────────────────────────────────────

def _pick(lines, fallback: str) -> str:
    import random
    lines = list(lines or [])
    return random.choice(lines) if lines else fallback


def _pick_cycling(lines, fallback: str, state_name: str, config_key: str) -> str:
    """Like _pick, but walks the WHOLE list before repeating and never lands on
    the same line twice running (utils.phrase_cycler, state under assets/state/).

    A random pick over three intros meant "loading the impression module" opened
    most bits; the frame around the joke got stale faster than the joke did.
    Falls back to a plain random pick if the cycler is unavailable.
    """
    lines = [str(x).strip() for x in (lines or []) if str(x).strip()]
    if not lines:
        return fallback
    try:
        from utils import phrase_cycler
        state_path = str(getattr(config, config_key, "") or "") or str(
            _project_root() / "assets" / "state" / state_name
        )
        picked = phrase_cycler.select_cycling_line(lines, state_path)
        if picked:
            return picked
    except Exception as exc:
        logger.debug("[impersonation] line cycling unavailable (%s)", exc)
    return _pick(lines, fallback)


def capture_prompt(name: object) -> str:
    first = str(name or "").strip().split(" ")[0] if name else ""
    who = f"{first}, " if first else ""
    return (f"{who}tell me a little about something you enjoyed today, in your own words. "
            "A few seconds of just your voice will give me something to work with.")


def who_line() -> str:
    """The ask when a request named nobody ("Impersonate.")."""
    return _pick(
        getattr(config, "IMPERSONATION_WHO_LINES", []),
        "Impersonate who? I can do you, or somebody famous.",
    )


def intro_line() -> str:
    """The stall line while the clone renders. Cycles — see _pick_cycling."""
    return _pick_cycling(
        getattr(config, "IMPERSONATION_INTRO_LINES", []),
        "Okay, clearing my vocal buffers. Ahem.",
        "impersonation_intro.json",
        "IMPERSONATION_INTRO_STATE_PATH",
    )


def outro_line() -> Optional[str]:
    """Rex's bow after the bit. Cycles independently of the intro."""
    if not bool(getattr(config, "IMPERSONATION_OUTRO_ENABLED", True)):
        return None
    return _pick_cycling(
        getattr(config, "IMPERSONATION_OUTRO_LINES", []),
        "Thank you, thank you. You're too kind.",
        "impersonation_outro.json",
        "IMPERSONATION_OUTRO_STATE_PATH",
    )


# ── Target resolution ─────────────────────────────────────────────────────────

@dataclass
class Resolution:
    kind: str                                   # 'perform' | 'capture' | 'refuse'
    ref: Optional[local_tts.VoiceRef] = None    # for 'perform'
    person_id: Optional[int] = None
    name: str = ""
    is_self: bool = False
    line: str = ""                              # capture prompt or refusal line


def _is_self(target: str) -> bool:
    t = (target or "").strip().lower()
    return t in _SELF_WORDS or t == ""


def is_self_target(target: str) -> bool:
    """Public wrapper: does this target mean the current speaker ("me")?"""
    return _is_self(target)


def resolve_target(
    target: str, speaker_person_id: Optional[int], speaker_name: Optional[str]
) -> Resolution:
    """Decide what to do for a 'do an impersonation of <target>' request."""
    if not is_enabled():
        if not bool(getattr(config, "IMPERSONATION_ENABLED", True)):
            return Resolution("refuse", line="I'll keep my impressions to myself for now.")
        return Resolution(
            "refuse",
            line="My mimicry circuits aren't installed on this rig — no impressions today.",
        )

    # "me" / "myself" → the current speaker. A KNOWN person gets a persistent ref
    # + memory-mined material. An UNKNOWN speaker (a guest/stranger) still gets
    # the bit — voice cloning needs only the captured clip, not a dossier: the
    # capture runs with person_id=None, the ref is session-scoped, and the parody
    # is a generic playful tease (live-requested 2026-07-19: Rex refused a guest).
    if _is_self(target):
        name = speaker_name or "you"
        if speaker_person_id is None:
            return Resolution("capture", person_id=None, name="", is_self=True,
                              line=capture_prompt(None))
        ref = person_ref(speaker_person_id)
        if ref is not None:
            return Resolution("perform", ref=ref, person_id=speaker_person_id, name=name, is_self=True)
        return Resolution("capture", person_id=speaker_person_id, name=name, is_self=True,
                          line=capture_prompt(name))

    # Stored names/aliases, then unique nicknames, before famous clips.
    from memory import people as people_db
    person = None
    try:
        person = people_db.find_person_by_name(target)
        if person is None:
            person = people_db.find_person_by_nickname(target)
    except Exception as exc:
        logger.debug("[impersonation] person lookup for %r failed: %s", target, exc)
    if person is not None:
        pid = person.get("id")
        name = person.get("name") or target
        ref = person_ref(pid) if pid is not None else None
        if ref is not None:
            return Resolution("perform", ref=ref, person_id=pid, name=name)
        # Known but never captured — offer to capture (they may be in the room).
        return Resolution("capture", person_id=pid, name=name, line=capture_prompt(name))

    # Famous-clip fallback.
    famous = find_famous_ref(target)
    if famous is not None:
        return Resolution("perform", ref=famous, person_id=None, name=target.strip())

    # ASR can render Jeffrey as Jaffrey. Resolve a unique close spelling for
    # this request only; do not rename the person or learn an identity alias.
    import difflib
    query = slugify(target)
    if len(query) >= 5:
        matches = []
        for name in people_db.list_person_names():
            candidate = slugify(name)
            if '-' not in query:
                candidate = candidate.split('-')[0]
            score = difflib.SequenceMatcher(None, query, candidate).ratio()
            if score >= .85:
                matches.append((score, name))
        matches.sort(reverse=True)
        if matches and (len(matches) == 1 or matches[0][0] - matches[1][0] >= .1):
            name = matches[0][1]
            person = people_db.find_person_by_name(name)
            if person is not None:
                pid = person['id']
                ref = person_ref(pid)
                return Resolution('perform' if ref else 'capture', ref=ref,
                                  person_id=pid, name=name, line='' if ref else capture_prompt(name))

    return Resolution(
        "refuse",
        line=(
            f"I'd need to actually hear {target.strip()} first — bring them over, "
            "or drop me a clip and I'll study up."
        ),
    )


# ── Script generation ─────────────────────────────────────────────────────────

_MATERIAL_CATEGORIES = {
    "job", "occupation", "career", "hobby", "interest", "project", "skill",
    "habit", "preference", "pet", "music", "food", "sport", "sports", "travel",
}
_MATERIAL_IDENTITY_KEYS = {"job", "job_title", "occupation", "current_project"}
_PERSONAL_CONTEXT_CATEGORIES = {
    "worldview", "belief", "religion", "politics", "health", "appearance",
    "family", "relationship", "boundary",
}


def _reliable_material(record: dict) -> bool:
    """Firsthand, well-supported material only; a parody cannot hedge gossip."""
    try:
        return (record.get("source") in {"explicit", "corrected"}
                and float(record.get("confidence") or 0) >= .85
                and (record.get("fact_kind") or "fact") == "fact")
    except (TypeError, ValueError):
        return False


def _gather_material(person_id: int) -> tuple[list[str], list[str]]:
    """Return (material_lines, do_not_lines) for a known person: what to riff on,
    and what is strictly off-limits (boundaries + heavy emotional events)."""
    material: list[str] = []
    do_not: list[str] = []
    from intelligence.onboarding import tidy_value
    try:
        from memory import boundaries as boundaries_db
        mute_terms = boundaries_db.muted_topic_terms(person_id)
    except Exception:
        mute_terms = set()
    try:
        from memory import facts as facts_db
        # Filter a wider pool before truncating: identity/appearance metadata
        # otherwise outranks the person's actual interests and work.
        for fact in facts_db.get_prompt_worthy_facts(person_id, limit=60, mute_terms=mute_terms):
            category = str(fact.get("category") or "").lower()
            key = str(fact.get("key") or "").lower()
            if not _reliable_material(fact):
                continue
            if category not in _MATERIAL_CATEGORIES and not (
                category == "identity" and key in _MATERIAL_IDENTITY_KEYS
            ):
                continue
            if not tidy_value(str(fact.get("value") or ""), "fact"):
                continue
            line = facts_db.format_fact_for_prompt(fact)
            if line:
                material.append(line)
    except Exception as exc:
        logger.debug("[impersonation] facts read failed: %s", exc)
    try:
        from memory import interests as interests_db
        for it in interests_db.get_interests_for_prompt(person_id, limit=6):
            topic = tidy_value(str(it.get("name") or ""), "interest")
            if (_reliable_material(it) and topic
                    and it.get("category") not in _PERSONAL_CONTEXT_CATEGORIES):
                # Freeform notes can contain quoted questions and remarks about
                # Rex. Use the interest label, not that conversation.
                material.append("interest: " + topic)
    except Exception as exc:
        logger.debug("[impersonation] interests read failed: %s", exc)
    try:
        from memory import preferences as prefs_db
        for pref in prefs_db.get_preferences_for_prompt(person_id, limit=8):
            line = prefs_db.format_preference_for_prompt(pref)
            if not line:
                continue
            if str(pref.get("preference_type") or "") == "boundary":
                do_not.append(line)
            elif (_reliable_material(pref) and pref.get("domain") not in _PERSONAL_CONTEXT_CATEGORIES
                  and tidy_value(str(pref.get("value") or pref.get("key") or ""), "fact")):
                material.append(line)
    except Exception as exc:
        logger.debug("[impersonation] preferences read failed: %s", exc)
    try:
        from memory import boundaries as boundaries_db
        summary = boundaries_db.summarize_for_prompt(person_id)
        if summary:
            do_not.append(summary)
    except Exception as exc:
        logger.debug("[impersonation] boundaries read failed: %s", exc)
    try:
        from memory import emotional_events as ee_db
        for ev in ee_db.get_active_events(person_id, limit=10):
            if ee_db.is_heavy_event(ev):
                desc = str(ev.get("description") or ev.get("summary") or "").strip()
                if desc:
                    do_not.append(desc)
    except Exception as exc:
        logger.debug("[impersonation] emotional events read failed: %s", exc)
    return material, do_not


def _recent_scripts(
    subject_name: str, person_id: Optional[int], limit: int = 6, *,
    voice_key: Optional[str] = None,
) -> list[str]:
    """The last few parodies Rex did of THIS subject, newest first.

    Asked twice in a row, the model happily returns the same joke off the same
    memory material — which reads as a cached bit even though it was generated
    fresh. Feeding the previous takes back as a do-not-repeat list is what makes
    a second "do me again" actually new.

    ``voice_key`` is the resolved VoiceRef label ("famous:richard-nixon"), and it
    is what makes that work for famous people. Matching on the SPOKEN name did
    not: "do Nixon", "do Richard Nixon" and "do President Nixon" all land on one
    voice clip but recorded three different subjects, so each phrasing saw an
    empty history and the model was free to tell its favourite joke again.
    Episodes written before the key existed still fall back to the name.
    """
    want = str(subject_name or "").strip().lower()
    out: list[str] = []
    try:
        from memory import episodes
        rows = episodes.recent_episodes(limit=60, kind="impersonation")
    except Exception as exc:
        logger.debug("[impersonation] recent script lookup failed: %s", exc)
        return out
    for row in rows or []:
        try:
            if person_id is not None:
                if row["person_id"] != person_id:
                    continue
            detail = row["detail"]
            if not detail:
                continue
            payload = json.loads(detail)
            if person_id is None:
                prior_voice = str(payload.get("voice") or "")
                if voice_key and prior_voice:
                    if prior_voice != voice_key:
                        continue
                elif str(payload.get("subject") or "").strip().lower() != want:
                    continue
            prior = " ".join(str(payload.get("script") or "").split())
        except Exception:
            continue
        if prior and prior not in out:
            out.append(prior)
        if len(out) >= limit:
            break
    return out


def _script_prompt(
    name: str, material: list[str], do_not: list[str], *,
    is_self: bool, famous: bool, stranger: bool = False,
    avoid: Optional[list[str]] = None, angle: Optional[str] = None,
    context: Optional[str] = None,
) -> str:
    who = "yourself" if is_self else name
    parts = [
        f"You are DJ-R3X, a witty Star Wars droid, doing a live comedic impression of {name} "
        f"out loud for whoever is in earshot. Write ONLY the words you would SAY while impersonating "
        f"{who}, in {name}'s own first-person voice.",
        "Rules: 2 to 3 short sentences. Affectionate exaggeration — play up their catchphrases, "
        "obsessions, and signature quirks for a warm laugh, never mean. PG. No stage directions, "
        "no quotation marks, no emoji, no bracketed tags, no preamble — just the spoken parody.",
    ]
    if context:
        # Unprompted bit (features/organic_impersonation.py): the subject came up
        # in conversation, so the bit must be a CAMEO IN that conversation — the
        # figure butting in on what was just being discussed — not the standalone
        # "a droid borrowed my voice" act the requested flow does. Field
        # 2026-08-19: with the standard famous block, Bret's trip to Plains got a
        # generic peanut line + "shared by a droid!" that felt tacked on.
        parts.append(
            "Nobody asked for this impression — it is a surprise cameo. Here is the "
            "live conversation it is interrupting (last lines, oldest first):\n"
            f"{context}\n"
            f"Write {name} BUTTING INTO that conversation: he has been listening and "
            "has an opinion about the SPECIFIC thing under discussion — the place, "
            "the plan, the claim, the person mentioned. React to their details by "
            "name; give advice, take credit, correct the record, or invite himself "
            "along, in his own famous voice and fixations. The listener should feel "
            "he heard them. Do NOT mention droids, robots, voice-borrowing, or being "
            "an impression, and do NOT retell his greatest-hits bio — one signature "
            "tic or fixation woven into the reply is enough to be unmistakably him. "
            "Keep it SHORT: 2 sentences at most."
        )
    if context and famous:
        # The cameo block above replaces the standard famous framing entirely —
        # its droid-collision and angle instructions are the requested-flow act.
        famous = False
    if stranger:
        parts.append(
            "You met this person SECONDS ago — you know absolutely nothing about them except "
            "the sound of their voice and the one line they just performed for you. Do NOT "
            "invent personal facts. Riff on the mystery itself: the bold move of handing a "
            "droid your voice, generic delightful human quirks, first-person mock-confidence "
            "('I'm the kind of person who...'). Keep it warm and silly."
        )
    elif famous:
        parts.append(
            f"This is a well-known public figure, and the bit is HALF him, HALF where he has "
            f"ended up. Both halves are required:\n"
            f"1. Anchor it in something unmistakably {name} — a line he is genuinely famous "
            f"for, a known fixation, a verbal tic, the thing every impressionist does. Someone "
            f"who knows nothing else about him should still recognise him from it. A generic "
            f"dignified old statesman is a failed take.\n"
            f"2. Collide that with the fact that a Star Wars droid has borrowed his voice and "
            f"is doing an impression of him right now, out loud, to whoever is listening. The "
            f"classic move is his own famous line, bent so it lands on the droid.\n"
            "Setting: assume NOTHING beyond that. There may be one person here or ten, in a "
            "kitchen or a workshop, at any hour. No party, no crowd, no guests, no dance "
            "floor, no music, no snacks or drinks — inventing an occasion is a failed take.\n"
            f"Direction matters: the DROID is impersonating {name}, not the other way round. "
            "He is a man, reacting to a machine that took his voice — outraged, flattered, or "
            "magnificently oblivious. He may mock the droid, deny being one, or refuse to "
            "believe in one, but he NEVER calls himself a droid or claims to be one.\n"
            "Vary the anchor between takes. If an earlier take opened on his single most "
            "famous line, reach for different material this time — another catchphrase, a "
            "policy he would not shut up about, a mannerism, a known appetite or vanity.\n"
            "Keep it light and playful — no politics-of-the-day or cheap shots, and nothing "
            "about how he died."
        )
        if angle:
            parts.append(
                f"Angle for THIS take: {angle}. The angle only shapes HOW the bit goes — it "
                "never excuses dropping the recognisable-him half or the droid half above. A "
                "take that follows the angle but could have come out of any old politician's "
                "mouth has failed."
            )
    if not famous and not stranger and not context:
        parts.append(
            "This is a real person in the room. Choose ONE coherent, well-supported "
            "interest, occupation, or habit from the material; use at most two related "
            "details. Exaggerate how they might talk or approach that activity, not "
            "their identity. Do not stitch a dossier into a biography, introduce yourself "
            "by full name, or list hometown, age, relatives, religion or politics. "
            "Do not invent a restaurant, event, audience or setting. A passing place "
            "remark does not establish the scene. Preserve who said what ABOUT WHOM: "
            "a comment aimed at Rex describes Rex, never the person being impersonated. "
            "One question or teasing remark is not a catchphrase, obsession, job or "
            "personality trait. Never turn their criticism of Rex into a self-insult. "
            "If reliable material is sparse, make the droid borrowing their voice the "
            "joke; do not invent personal traits or draw them from prior parodies."
        )
    if material:
        parts.append("Things you know about " + name + " (riff on these):\n- " + "\n- ".join(material[:14]))
    if do_not:
        parts.append(
            "NEVER reference, hint at, or joke about any of the following — these are hard "
            "boundaries, not material:\n- " + "\n- ".join(do_not[:12])
        )
    if avoid:
        parts.append(
            "The following are prior GENERATED PARODIES, not evidence about this person. "
            "They may contain inventions or attribution mistakes; never adopt those as facts. "
            "You have already done this impression before. Write a DIFFERENT bit — new angle, "
            "new detail, new punchline. Reusing a joke, a premise, or a closing beat from these "
            "is a failure, even reworded:\n- " + "\n- ".join(avoid[:6])
        )
    return "\n\n".join(parts)


def build_parody_script(
    subject_name: str, person_id: Optional[int] = None, *,
    is_self: bool = False, stranger: bool = False,
    voice_key: Optional[str] = None,
    context: Optional[str] = None,
    max_words: Optional[int] = None,
) -> Optional[str]:
    """Generate the short parody line via a one-off LLM completion. Returns the
    cleaned script text, or None on failure. `stranger` = an anonymous live guest
    (no memory, no famous framing — a generic warm tease).

    Nothing here is cached: every call re-asks the model, and the avoid-list plus
    a randomly drawn angle push it off the take it gave last time.
    """
    import random

    material, do_not = ([], [])
    if person_id is not None:
        material, do_not = _gather_material(person_id)
    famous = (person_id is None and not stranger)
    prompt = _script_prompt(
        subject_name, material, do_not, is_self=is_self,
        famous=famous, stranger=stranger,
        avoid=_recent_scripts(subject_name, person_id, voice_key=voice_key),
        angle=(random.choice(_FAMOUS_ANGLES) if (famous and not context) else None),
        **({"context": context} if context else {}),
    )
    try:
        from intelligence import llm
        resp = llm._client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.85,
            max_tokens=180,
        )
        text = (resp.choices[0].message.content or "").strip()
        return _cap_script_words(llm.clean_response_text(text), max_words=max_words) or None
    except Exception as exc:
        logger.warning("[impersonation] script generation failed: %s", exc)
        return None


def _cap_script_words(text: Optional[str], *, max_words: Optional[int] = None) -> Optional[str]:
    """Hard cap the parody length at sentence boundaries. The prompt asks for
    2-3 short sentences but the model sometimes runs long, and every extra word
    is more local-synthesis time the room spends listening to the thinking loop
    (and was more stutter, before takes were prewarmed)."""
    if not text:
        return text
    if max_words is None:
        max_words = int(getattr(config, "IMPERSONATION_SCRIPT_MAX_WORDS", 45))
    max_words = int(max_words)
    words_so_far = 0
    kept: list[str] = []
    for sentence in re.split(r"(?<=[.!?…])\s+", text.strip()):
        n = len(sentence.split())
        if kept and words_so_far + n > max_words:
            break
        kept.append(sentence)
        words_so_far += n
    capped = " ".join(kept).strip()
    if capped != text.strip():
        logger.info("[impersonation] script capped %d -> %d words",
                    len(text.split()), words_so_far)
    return capped or text


# ── Performance ───────────────────────────────────────────────────────────────

def perform(
    ref: local_tts.VoiceRef, subject_name: str, person_id: Optional[int], *, is_self: bool = False
) -> str:
    """Speak the full bit: Rex-voice stall line → the parody in the cloned voice →
    optional Rex-voice bow. Blocks until each line finishes. Logs an episode.
    Returns the parody text. All three lines are logged HERE, in spoken order —
    the caller's own write of the return value dedupes away (see claim_rex_line).
    On a synthesis/script miss, covers in Rex's voice and returns that.

    An anonymous guest (label 'person:anon' — captured live, no person row) gets
    the stranger script mode: no invented facts, no famous framing.
    """
    from audio import speech_queue

    # An explicit request outranks any unprompted bit still waiting for its
    # moment — and start_take below would evict its parked take anyway.
    try:
        from features import organic_impersonation
        organic_impersonation.cancel("explicit_request")
    except Exception:
        pass

    stranger = (getattr(ref, "label", "") == "person:anon")
    if stranger and not (subject_name or "").strip():
        subject_name = "my mystery guest"

    def _say(line: str, emotion: str, *, voice_ref=None, log_text: bool) -> bool:
        try:
            done = speech_queue.enqueue(
                line, emotion, priority=1, tag="impersonation",
                voice_ref=voice_ref, log_text=log_text,
            )
            # A bounded Breeze take can contain up to 60 seconds of audio.
            # Allow it to drain, including device latency, before releasing it.
            timeout = max(45.0, float(config.BREEZE_TTS_MAX_TOKENS) / 12.5 + 10.0)
            return bool(done.wait(timeout=timeout) and getattr(done, "played", False))
        except Exception as exc:
            logger.debug("[impersonation] enqueue failed: %s", exc)
            return False

    # 1. Script first. Qwen may render its buffered take behind an online intro;
    # Breeze waits until that intro finishes, then prepares the complete take.
    # Nothing is cached: each request generates fresh script and speech.
    voice_key = getattr(ref, "label", "") or None
    script = build_parody_script(
        subject_name, person_id, is_self=is_self, stranger=stranger, voice_key=voice_key,
    )

    # The player keys the parked take on the text it will actually synthesize.
    speech_text = script or ""
    if script:
        try:
            from audio import tts as tts_module
            speech_text = tts_module.spoken_form(script) or script
        except Exception as exc:
            logger.debug("[impersonation] spoken_form failed: %s", exc)
            speech_text = script

    take = None
    if speech_text:
        # In --local-tts mode the INTRO also needs the engine, and synthesis is
        # serialized — starting the take first would block the intro. Order
        # around it.
        # A bounded Breeze stream can park while holding its engine. Wait
        # for the intro even online: it may itself fall back to local speech.
        from audio import tts as tts_module
        start_before_intro = not (local_tts.streams_clones() or tts_module._use_local_backend())
        if start_before_intro:
            try:
                take = local_tts.start_take(speech_text, ref)
            except Exception as exc:
                logger.debug("[impersonation] take launch failed: %s", exc)

    # 2. Stall/setup line in Rex's own voice (covers model load + synthesis).
    _say(intro_line(), "excited", log_text=True)

    if not script:
        cover = "...huh. My impression module just blew a fuse. We'll try that again later."
        _say(cover, "sheepish", log_text=False)
        return cover

    if take is None:
        try:
            take = local_tts.start_take(speech_text, ref)
        except Exception as exc:
            logger.debug("[impersonation] take launch failed: %s", exc)

    # 3. Cover the wait for the first playable audio: the full take for Breeze,
    #    a buffered unit for Qwen. Stop the loop before opening voice playback.
    if take is not None and not take.first_ready.is_set():
        loop_handle = None
        try:
            from audio import sound_effects
            loop_handle = sound_effects.start_loop("thinking")
        except Exception as exc:
            logger.debug("[impersonation] thinking loop failed: %s", exc)
        take.first_ready.wait(
            timeout=float(getattr(config, "IMPERSONATION_FIRST_UNIT_TIMEOUT_SECS", 45.0))
        )
        try:
            from audio import sound_effects
            sound_effects.stop_loop(loop_handle)
        except Exception:
            pass

    def _release_take() -> None:
        """Unpark and stop the take. Harmless once the player has claimed it;
        essential when the line never reached playback (shutdown, busy output
        gate), where an unclaimed renderer would otherwise keep running."""
        if take is None:
            return
        try:
            local_tts.pop_take(speech_text, ref)
            take.close()
        except Exception as exc:
            logger.debug("[impersonation] take release failed: %s", exc)

    if take is None or take.failed or not take.first_ready.is_set():
        _release_take()
        cover = "...huh. My impression module just blew a fuse. We'll try that again later."
        _say(cover, "sheepish", log_text=False)
        return cover

    # 4. The parody in the cloned voice.
    #
    # Logged HERE, as it is spoken, rather than left to the caller. The caller
    # logs whatever perform() returns, which happens after the outro has already
    # been written — so the transcript and the GUI showed intro, bow, then the
    # punchline last (field 2026-08-04). Logging the SCRIPT, not speech_text:
    # spoken_form() may have expanded numbers and abbreviations for the
    # synthesizer, and the caller will later try to log the script itself.
    try:
        from utils import conv_log
        conv_log.log_rex(script)
    except Exception as exc:
        logger.debug("[impersonation] parody log failed: %s", exc)

    try:
        completed = _say(speech_text, "excited", voice_ref=ref, log_text=False)
    finally:
        _release_take()

    if not completed:
        logger.warning("[impersonation] playback incomplete (voice=%s)", voice_key)
        cover = "My impression cut out before I finished. Sorry about that."
        _say(cover, "sheepish", log_text=False)
        return cover

    # 5. Optional Rex-voice button — Rex steps back out and takes his bow.
    outro = outro_line()
    if outro:
        _say(outro, "amused", log_text=True)

    # The caller logs the returned script; claim it so that write dedupes away
    # instead of repeating the parody below the bow. Must come after the outro —
    # only the previous line is compared.
    try:
        from utils import conv_log
        conv_log.claim_rex_line(script)
    except Exception as exc:
        logger.debug("[impersonation] parody log claim failed: %s", exc)

    # 4. Episodic memory of the bit.
    try:
        from memory import episodes
        episodes.record_episode(
            "impersonation",
            f"I did an impersonation of {subject_name}.",
            person_id=person_id,
            person_name=(subject_name if person_id is not None else None),
            detail={"subject": subject_name, "voice": voice_key, "script": script},
            salience=0.75,
        )
    except Exception as exc:
        logger.debug("[impersonation] episode log failed: %s", exc)

    return script


def sounds_like_cancel(text: str) -> bool:
    """True when a captured reply is the person backing out of the impression."""
    return bool(_CANCEL_RE.search(text or ""))
