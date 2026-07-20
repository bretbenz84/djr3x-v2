"""
Impersonation — Rex clones a voice and performs a short, affectionate parody.

"Rex, do an impersonation of me / of Jimmy Carter." Rex clones a voice (via the
on-device Qwen3-TTS engine, audio/local_tts.py) and delivers a brief LLM-written
parody in that voice, framed by his own stall/outro lines. Two reference sources:

  * Known people — captured live (Rex asks them to repeat a fixed line), saved
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
    r"\b(no|nope|not now|later|never ?mind|forget it|stop|cancel|wait|hold on|"
    r"don'?t|actually no)\b",
    re.IGNORECASE,
)
_STOPWORDS = {"the", "a", "an", "of", "mr", "mrs", "ms", "dr", "president", "sir", "madam"}


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
    """Existing saved reference for a known person, or None."""
    base = _voices_dir() / "people"
    return local_tts.voice_ref_from_files(
        base / f"{person_id}.wav", base / f"{person_id}.txt", f"person:{person_id}"
    )


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
    person_id: Optional[int], audio_array: np.ndarray, transcript: str
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
        arr = _pad_tail(arr, sr)
        sf.write(str(wav_path), arr, sr, subtype="PCM_16")
        txt_path.write_text(ref_text, encoding="utf-8")
        json_path.write_text(
            json.dumps({
                "person_id": person_id,
                "duration_secs": round(len(arr) / float(sr), 2),
                "transcript": ref_text,
            }),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("[impersonation] failed to save capture for %s: %s", person_id, exc)
        return None
    return local_tts.voice_ref_from_files(wav_path, txt_path, label)


# ── Lines ─────────────────────────────────────────────────────────────────────

def _pick(lines, fallback: str) -> str:
    import random
    lines = list(lines or [])
    return random.choice(lines) if lines else fallback


def capture_line() -> str:
    return _pick(
        getattr(config, "IMPERSONATION_CAPTURE_LINES", []),
        "Mary had a little lamb, its fleece was white as snow. "
        "And everywhere that Mary went, the lamb was sure to go.",
    )


def intro_line() -> str:
    return _pick(
        getattr(config, "IMPERSONATION_INTRO_LINES", []),
        "Okay, clearing my vocal buffers. Ahem.",
    )


def outro_line() -> Optional[str]:
    if not bool(getattr(config, "IMPERSONATION_OUTRO_ENABLED", True)):
        return None
    return _pick(getattr(config, "IMPERSONATION_OUTRO_LINES", []), "I do not sound like that.")


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
            return Resolution(
                "capture", person_id=None, name="", is_self=True,
                line=capture_line(),
            )
        ref = person_ref(speaker_person_id)
        if ref is not None:
            return Resolution("perform", ref=ref, person_id=speaker_person_id, name=name, is_self=True)
        return Resolution(
            "capture", person_id=speaker_person_id, name=name, is_self=True,
            line=capture_line(),
        )

    # A named target. Known people take precedence over famous clips.
    from memory import people as people_db
    person = None
    try:
        person = people_db.find_person_by_name(target)
    except Exception as exc:
        logger.debug("[impersonation] find_person_by_name(%r) failed: %s", target, exc)
    if person is not None:
        pid = person.get("id")
        name = person.get("name") or target
        ref = person_ref(pid) if pid is not None else None
        if ref is not None:
            return Resolution("perform", ref=ref, person_id=pid, name=name)
        # Known but never captured — offer to capture (they may be in the room).
        return Resolution("capture", person_id=pid, name=name, line=capture_line())

    # Famous-clip fallback.
    famous = find_famous_ref(target)
    if famous is not None:
        return Resolution("perform", ref=famous, person_id=None, name=target.strip())

    return Resolution(
        "refuse",
        line=(
            f"I'd need to actually hear {target.strip()} first — bring them over, "
            "or drop me a clip and I'll study up."
        ),
    )


# ── Script generation ─────────────────────────────────────────────────────────

def _gather_material(person_id: int) -> tuple[list[str], list[str]]:
    """Return (material_lines, do_not_lines) for a known person: what to riff on,
    and what is strictly off-limits (boundaries + heavy emotional events)."""
    material: list[str] = []
    do_not: list[str] = []
    try:
        from memory import boundaries as boundaries_db
        mute_terms = boundaries_db.muted_topic_terms(person_id)
    except Exception:
        mute_terms = set()
    try:
        from memory import facts as facts_db
        for fact in facts_db.get_prompt_worthy_facts(person_id, limit=12, mute_terms=mute_terms):
            # Belt-and-suspenders: never feed unkind gossip into a parody.
            if str(fact.get("fact_kind") or "") == "gossip" and float(fact.get("kindness", 0.0)) <= -0.25:
                continue
            line = facts_db.format_fact_for_prompt(fact)
            if line:
                material.append(line)
    except Exception as exc:
        logger.debug("[impersonation] facts read failed: %s", exc)
    try:
        from memory import interests as interests_db
        for it in interests_db.get_interests_for_prompt(person_id, limit=6):
            line = interests_db.format_interest_for_prompt(it)
            if line:
                material.append(line)
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
            else:
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


def _script_prompt(
    name: str, material: list[str], do_not: list[str], *,
    is_self: bool, famous: bool, stranger: bool = False,
) -> str:
    who = "yourself" if is_self else name
    parts = [
        f"You are DJ-R3X, a witty Star Wars droid, doing a live comedic impression of {name} "
        f"for a small room of friends. Write ONLY the words you would SAY while impersonating "
        f"{who}, in {name}'s own first-person voice.",
        "Rules: 2 to 3 short sentences. Affectionate exaggeration — play up their catchphrases, "
        "obsessions, and signature quirks for a warm laugh, never mean. PG. No stage directions, "
        "no quotation marks, no emoji, no bracketed tags, no preamble — just the spoken parody.",
    ]
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
            "This is a well-known public figure; riff on their famous mannerisms and speaking "
            "style. Keep it light and playful — no politics-of-the-day or cheap shots."
        )
    if material:
        parts.append("Things you know about " + name + " (riff on these):\n- " + "\n- ".join(material[:14]))
    if do_not:
        parts.append(
            "NEVER reference, hint at, or joke about any of the following — these are hard "
            "boundaries, not material:\n- " + "\n- ".join(do_not[:12])
        )
    return "\n\n".join(parts)


def build_parody_script(
    subject_name: str, person_id: Optional[int] = None, *,
    is_self: bool = False, stranger: bool = False,
) -> Optional[str]:
    """Generate the short parody line via a one-off LLM completion. Returns the
    cleaned script text, or None on failure. `stranger` = an anonymous live guest
    (no memory, no famous framing — a generic warm tease)."""
    material, do_not = ([], [])
    if person_id is not None:
        material, do_not = _gather_material(person_id)
    prompt = _script_prompt(
        subject_name, material, do_not, is_self=is_self,
        famous=(person_id is None and not stranger), stranger=stranger,
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
        return llm.clean_response_text(text) or None
    except Exception as exc:
        logger.warning("[impersonation] script generation failed: %s", exc)
        return None


# ── Performance ───────────────────────────────────────────────────────────────

def perform(
    ref: local_tts.VoiceRef, subject_name: str, person_id: Optional[int], *, is_self: bool = False
) -> str:
    """Speak the full bit: Rex-voice stall line → the parody in the cloned voice →
    optional Rex-voice button. Blocks until each line finishes. Logs an episode.
    Returns the parody text (the caller logs it once as Rex's turn); the intro/outro
    self-log. On a synthesis/script miss, covers in Rex's voice and returns that.

    An anonymous guest (label 'person:anon' — captured live, no person row) gets
    the stranger script mode: no invented facts, no famous framing.
    """
    from audio import speech_queue

    stranger = (getattr(ref, "label", "") == "person:anon")
    if stranger and not (subject_name or "").strip():
        subject_name = "my mystery guest"

    def _say(line: str, emotion: str, *, voice_ref=None, log_text: bool) -> None:
        try:
            done = speech_queue.enqueue(
                line, emotion, priority=1, tag="impersonation",
                voice_ref=voice_ref, log_text=log_text,
            )
            done.wait(timeout=45.0)
        except Exception as exc:
            logger.debug("[impersonation] enqueue failed: %s", exc)

    # 1. Stall/setup line in Rex's own voice (also covers cold model-load latency).
    _say(intro_line(), "excited", log_text=True)

    # 2. The parody in the cloned voice.
    script = build_parody_script(subject_name, person_id, is_self=is_self, stranger=stranger)
    if not script:
        cover = "...huh. My impression module just blew a fuse. We'll try that again later."
        _say(cover, "sheepish", log_text=False)
        return cover
    _say(script, "excited", voice_ref=ref, log_text=False)

    # 3. Optional Rex-voice button.
    outro = outro_line()
    if outro:
        _say(outro, "amused", log_text=True)

    # 4. Episodic memory of the bit.
    try:
        from memory import episodes
        episodes.record_episode(
            "impersonation",
            f"I did an impersonation of {subject_name}.",
            person_id=person_id,
            person_name=(subject_name if person_id is not None else None),
            detail={"subject": subject_name, "script": script},
            salience=0.75,
        )
    except Exception as exc:
        logger.debug("[impersonation] episode log failed: %s", exc)

    return script


def sounds_like_cancel(text: str) -> bool:
    """True when a captured reply is the person backing out of the impression."""
    return bool(_CANCEL_RE.search(text or ""))
