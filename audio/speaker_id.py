import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import config
from audio import voice_score
from memory import database as db
from memory import people

logger = logging.getLogger(__name__)

_encoder = None
_UNAVAILABLE = False

# Resolved at first load: "ecapa" or "resemblyzer" (after any fallback).
_active_backend: Optional[str] = None


def _load_ecapa():
    """ECAPA-TDNN speaker embeddings (SpeechBrain, 192-dim) — far wider
    genuine/impostor separation than Resemblyzer (the root cause of every
    ambiguity incident: JT's print sat 0.45-0.49 from ALL of Bret's).
    Model files live in config.ECAPA_MODEL_DIR (setup_assets.py downloads them;
    ~80MB). Returns the encoder or None."""
    try:
        import torch  # noqa: F401 — fail fast if torch is broken
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError as exc:
        logger.warning("speechbrain unavailable (%s)", exc)
        return None
    model_dir = str(getattr(config, "ECAPA_MODEL_DIR", "assets/models/ecapa"))
    if not (Path(model_dir) / "hyperparams.yaml").exists():
        logger.warning("ECAPA model missing at %s — run setup_assets.py", model_dir)
        return None
    try:
        encoder = EncoderClassifier.from_hparams(
            source=model_dir, savedir=model_dir, run_opts={"device": "cpu"},
        )
        logger.info("ECAPA-TDNN speaker encoder loaded from %s (192-dim)", model_dir)
        return encoder
    except Exception as exc:
        logger.warning("ECAPA load failed (%s)", exc)
        return None


def _load_resemblyzer():
    try:
        from resemblyzer import VoiceEncoder
        weights = Path(config.RESEMBLYZER_MODEL_DIR) / "pretrained.pt"
        encoder = VoiceEncoder(weights_fpath=weights)
        logger.info("Resemblyzer encoder loaded from %s", weights)
        return encoder
    except Exception as exc:
        logger.warning("Resemblyzer unavailable (%s)", exc)
        return None


def _get_encoder():
    global _encoder, _UNAVAILABLE, _active_backend
    if _UNAVAILABLE:
        return None
    if _encoder is not None:
        return _encoder

    backend = str(getattr(config, "VOICE_EMBEDDER", "ecapa") or "ecapa").lower()
    if backend == "ecapa":
        _encoder = _load_ecapa()
        if _encoder is not None:
            _active_backend = "ecapa"
            voice_score.set_active_backend("ecapa")
            return _encoder
        logger.warning("ECAPA unavailable — falling back to Resemblyzer embedder")

    _encoder = _load_resemblyzer()
    if _encoder is not None:
        _active_backend = "resemblyzer"
        voice_score.set_active_backend("resemblyzer")
        return _encoder

    logger.warning("no voice embedder available — speaker identification disabled")
    _UNAVAILABLE = True
    return None


def active_backend() -> Optional[str]:
    """The embedder actually in use ("ecapa"/"resemblyzer"), or None if unloaded."""
    return _active_backend


def _embed_ecapa(encoder, audio_array: np.ndarray) -> Optional[np.ndarray]:
    import torch
    sample_rate = int(getattr(config, "AUDIO_SAMPLE_RATE", 16000) or 16000)
    wav = np.asarray(audio_array, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sample_rate != 16000:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(16000, sample_rate)
        wav = resample_poly(wav, 16000 // g, sample_rate // g).astype(np.float32)
    if wav.size < 800:   # <50ms of audio — nothing to embed
        return None
    with torch.no_grad():
        emb = encoder.encode_batch(torch.from_numpy(wav).unsqueeze(0))
    emb = emb.squeeze().cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm <= 1e-10:
        return None
    return emb / norm


def preload() -> bool:
    """Load the voice encoder before the first live speech turn."""
    start = time.monotonic()
    encoder = _get_encoder()
    if encoder is None:
        return False
    try:
        # Warm the full embedding path too; the first live turn needs it.
        sample_rate = int(getattr(config, "AUDIO_SAMPLE_RATE", 16000) or 16000)
        samples = max(1, int(sample_rate * 0.75))
        t = np.arange(samples, dtype=np.float32) / float(sample_rate)
        dummy = (0.001 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
        if _active_backend == "ecapa":
            _embed_ecapa(encoder, dummy)
        else:
            from resemblyzer import preprocess_wav
            wav = preprocess_wav(dummy, source_sr=sample_rate)
            encoder.embed_utterance(wav)
    except Exception as exc:
        logger.warning("[speaker_id] preload failed while warming embedding path: %s", exc)
    logger.info(
        "[speaker_id] preloaded %s encoder in %.3fs",
        _active_backend or "no", time.monotonic() - start,
    )
    _warn_if_all_prints_are_other_backend()
    return True


def _warn_if_all_prints_are_other_backend() -> None:
    """One loud line when every stored voice print belongs to the OTHER embedder —
    the operator would otherwise just see everyone become a mystery voice."""
    try:
        native_bytes = (192 if _active_backend == "ecapa" else 256) * 4
        rows = db.fetchall(
            "SELECT LENGTH(encoding) AS n FROM biometrics WHERE type = 'voice'"
        )
        if not rows:
            return
        native = sum(1 for r in rows if int(r["n"]) == native_bytes)
        if native == 0:
            logger.warning(
                "[speaker_id] %d stored voice print(s), NONE from the active '%s' "
                "embedder — everyone will read as unknown until they re-enroll "
                "(tools/test_voice_id.py --enroll NAME --replace)",
                len(rows), _active_backend,
            )
    except Exception as exc:
        logger.debug("[speaker_id] print-dimension audit failed: %s", exc)


def get_embedding(audio_array: np.ndarray) -> Optional[np.ndarray]:
    """Preprocess audio and return a normalized float32 embedding, or None on failure.

    Dimension depends on the active backend: 192 (ECAPA) or 256 (Resemblyzer).
    The two are INCOMPATIBLE — matchers skip stored rows of the other dimension,
    so people enrolled under one embedder must re-enroll after switching.
    """
    encoder = _get_encoder()
    if encoder is None:
        return None
    try:
        if _active_backend == "ecapa":
            return _embed_ecapa(encoder, audio_array)
        from resemblyzer import preprocess_wav
        wav = preprocess_wav(audio_array, source_sr=config.AUDIO_SAMPLE_RATE)
        embedding = encoder.embed_utterance(wav)
        norm = np.linalg.norm(embedding)
        return (embedding / (norm + 1e-10)).astype(np.float32)
    except Exception as exc:
        logger.error("get_embedding failed: %s", exc)
        return None


def buffer_secs(audio_array: Optional[np.ndarray]) -> float:
    """Wall length of a capture buffer in seconds (pre-roll and padding included)."""
    if not isinstance(audio_array, np.ndarray) or len(audio_array) <= 0:
        return 0.0
    rate = float(getattr(config, "AUDIO_SAMPLE_RATE", 16000) or 16000)
    return float(len(audio_array)) / max(rate, 1.0)


def voiced_secs(audio_array: Optional[np.ndarray]) -> float:
    """Seconds of actual SPEECH in a buffer: 30 ms frames counted voiced against an
    amplitude-relative floor (same rule as interaction._voiced_duration_secs). The
    embedder pools statistics over the WHOLE buffer, which carries ~0.45 s pre-roll
    and ~0.65 s silence timeout around every utterance, so this — not buffer length —
    is the duration a score should be read against. Instrumentation 2026-09-02."""
    if not isinstance(audio_array, np.ndarray) or len(audio_array) <= 0:
        return 0.0
    rate = float(getattr(config, "AUDIO_SAMPLE_RATE", 16000) or 16000)
    frame = max(1, int(0.03 * rate))
    usable = (len(audio_array) // frame) * frame
    if usable <= 0:
        return 0.0
    frames = np.abs(audio_array[:usable].astype(np.float32)).reshape(-1, frame)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    peak = float(rms.max()) if rms.size else 0.0
    if peak <= 0.0:
        return 0.0
    floor = max(0.004, 0.1 * peak)
    return float(int((rms >= floor).sum()) * frame / rate)


def rank_speakers(audio_array: np.ndarray) -> list[tuple[int, str, float, int]]:
    """Return [(person_id, name, similarity, n_prints), ...] sorted by similarity desc,
    ONE entry per person, scoring the query against that person's CENTROID (mean of all
    their enrolled voice embeddings, renormalized). n_prints is how many enrolled clips
    back that centroid — the maturity signal required_ambiguity_margin uses.

    Per-person centroids fix two problems with the old max-over-rows approach: (1) a
    person no longer appears multiple times in the candidate list (a weak duplicate row
    could outrank a different person), and (2) averaging several clips is a higher-SNR,
    better-generalizing speaker representation than any single noisy clip — which raises
    true-speaker scores. Returns [] if the embedding can't be computed or nothing is
    enrolled.
    """
    embedding = get_embedding(audio_array)
    if embedding is None:
        return []
    return rank_embedding(embedding)


def window_evidence(audio_array: np.ndarray) -> list[dict]:
    """Bounded audio-only check for a speaker change inside one capture.

    Reuse the resident voice encoder on up to three non-overlapping windows.
    This can detect conflicting enrolled voices without a camera or DoA array;
    absence of a conflict is not proof of one speaker. Scores are similarities.
    """
    sr = int(config.AUDIO_SAMPLE_RATE)
    if audio_array is None or len(audio_array) < int(2.4 * sr):
        return []
    frame = max(1, int(.03 * sr))
    n = len(audio_array) // frame
    rms = np.sqrt(np.mean(np.square(audio_array[:n*frame].reshape(n, frame)), axis=1))
    active = np.flatnonzero(rms >= max(.004, .1 * float(rms.max())))
    if not len(active):
        return []
    lo, hi = int(active[0]*frame), min(len(audio_array), int((active[-1]+1)*frame))
    if hi - lo < int(2.4 * sr):
        return []
    width = min(int(1.5*sr), (hi-lo)//2)
    starts = [lo, hi-width]
    if hi-lo >= 3*width:
        starts.insert(1, (lo+hi-width)//2)
    rows = []
    previous_embedding = None
    for start in starts:
        stop = start + width
        clip = audio_array[start:stop]
        if voiced_secs(clip) < .6:
            continue
        embedding = get_embedding(clip)
        if embedding is None:
            continue
        ranked = rank_embedding(embedding)
        if not ranked:
            continue
        pid, name, score, _ = ranked[0]
        margin = score - (ranked[1][2] if len(ranked) > 1 else -1.)
        trusted = (score >= float(config.SPEAKER_ID_SIMILARITY_THRESHOLD)
                   and margin >= required_ambiguity_margin(ranked))
        from audio import voice_score
        pair_similarity = (voice_score.map_similarity(float(np.dot(previous_embedding, embedding)))
                           if previous_embedding is not None else None)
        changed = (pair_similarity is not None
                   and pair_similarity < float(config.SPEAKER_ID_SIMILARITY_THRESHOLD))
        rows.append({"start": start/sr, "end": stop/sr, "person_id": pid if trusted else None,
                     "name": name if trusted else None, "score": float(score), "margin": float(margin),
                     "previous_similarity": pair_similarity, "change_suspected": changed})
        previous_embedding = embedding
    return rows


def rank_embedding(embedding: np.ndarray) -> list[tuple[int, str, float, int]]:
    """rank_speakers for an ALREADY-computed embedding (the enrollment provenance log
    scores the clip about to be stored without embedding it twice)."""
    if embedding is None:
        return []
    rows = db.fetchall(
        "SELECT person_id, encoding FROM biometrics WHERE type = 'voice'"
    )
    if not rows:
        return []

    query = np.asarray(embedding, dtype=np.float32)
    query_norm = query / (np.linalg.norm(query) + 1e-10)

    per_person: dict[int, list[np.ndarray]] = {}
    for row in rows:
        stored = np.frombuffer(bytes(row["encoding"]), dtype=np.float32)
        if stored.shape != query.shape:
            # Other-embedder enrollment — expected during migration, not an error.
            logger.debug(
                "voice embedding shape mismatch: stored %s vs query %s",
                stored.shape, query.shape,
            )
            continue
        stored_norm = stored / (np.linalg.norm(stored) + 1e-10)
        per_person.setdefault(row["person_id"], []).append(stored_norm)

    scored: list[tuple[int, str, float, int]] = []
    for pid, vecs in per_person.items():
        centroid = np.mean(np.stack(vecs), axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
        # Mapped onto the Resemblyzer-calibrated threshold scale (see voice_score).
        sim = voice_score.map_similarity(float(np.dot(centroid, query_norm)))
        person = people.get_person(pid)
        nm = (person.get("name") if person else None) or "?"
        scored.append((pid, nm, sim, len(vecs)))

    scored.sort(key=lambda t: t[2], reverse=True)
    return scored


def required_ambiguity_margin(ranked: list) -> float:
    """Gap the top candidate must have over the runner-up before the match counts
    as unambiguous, given THIS scoreboard.

    Base is SPEAKER_ID_KNOWN_MARGIN. It is REDUCED (thin-challenger relief) when:
      - the runner-up's centroid is built from <= SPEAKER_ID_THIN_PRINT_MAX_ROWS
        clips (a single unverified print is high-variance — it shouldn't carry
        full veto power over a mature multi-print match), AND
      - the top candidate has a mature (> thin) print set, AND
      - the top score clears SPEAKER_ID_THIN_RUNNER_MIN_TOP_SCORE, high enough
        that a cross-match impostor is unlikely (field data: JT's live voice hit
        Bret's centroid at 0.529; Bret's own short greeting hits 0.558+).

    Field log 2026-07-06-19-23: Bret (6 prints) 0.558 vs JT (1 print) 0.502 —
    margin 0.056 < 0.07 challenged the OWNER as a mystery voice. With relief the
    required margin halves (0.035) and Bret resolves. The reverse case (JT top
    0.563 vs Bret runner-up 0.529, log 2026-07-05-21-22) keeps the FULL margin
    because the runner-up (Bret) is mature — the who's-that challenge still fires
    while JT's print is thin, which is how his print gets confirmed and grown.
    """
    base = float(getattr(config, "SPEAKER_ID_KNOWN_MARGIN", 0.07) or 0.0)
    if len(ranked) < 2:
        return base
    thin_max = int(getattr(config, "SPEAKER_ID_THIN_PRINT_MAX_ROWS", 1) or 1)
    factor = float(getattr(config, "SPEAKER_ID_THIN_RUNNER_MARGIN_FACTOR", 0.5) or 0.5)
    min_top = float(getattr(config, "SPEAKER_ID_THIN_RUNNER_MIN_TOP_SCORE", 0.55) or 0.55)
    top, runner = ranked[0], ranked[1]
    top_sim, top_count = float(top[2]), int(top[3])
    runner_count = int(runner[3])
    if runner_count <= thin_max and top_count > thin_max and top_sim >= min_top:
        return base * factor
    return base


def format_scoreboard(scored: list) -> str:
    """'Name#id=score(nP)' per row, EVERY row — the runner-up gap as a function of
    voiced seconds is the number the whole margin design hinges on, and the old
    three-row cut hid the tail (instrumentation 2026-09-02)."""
    return ", ".join(f"{nm}#{pid}={sim:.3f}({n}p)" for pid, nm, sim, n in scored)


def _log_scoreboard(
    scored: list, *, voiced: Optional[float] = None, buffer: Optional[float] = None,
) -> None:
    # Same "Name#id=score" format tooling/test_voice_id.py read (one row per person).
    dur = ""
    if voiced is not None or buffer is not None:
        dur = f" voiced={float(voiced or 0.0):.2f}s buffer={float(buffer or 0.0):.2f}s"
    logger.info(
        "[speaker_id] scan — threshold=%.3f,%s candidates: %s",
        config.SPEAKER_ID_SIMILARITY_THRESHOLD,
        dur,
        format_scoreboard(scored),
    )


def identify_speaker_raw(
    audio_array: np.ndarray,
) -> Tuple[Optional[int], Optional[str], float]:
    """Return the TOP per-person centroid voice match without a threshold filter.

    Returns (best_id, best_name, best_sim), or (None, None, 0.0) if no match could be
    computed. The low-level primitive — callers apply their own acceptance logic.
    """
    scored = rank_speakers(audio_array)
    if not scored:
        return (None, None, 0.0)
    _log_scoreboard(scored)
    best_id, name, best_sim, _n = scored[0]
    return (best_id, name, float(best_sim))


def identify_speaker(
    audio_array: np.ndarray,
) -> Tuple[Optional[int], Optional[str], float]:
    """Return (person_id, name, score) for a confident voice match, else (None, None, 0.0).

    Accepts the top centroid match only when it clears SPEAKER_ID_SIMILARITY_THRESHOLD
    AND beats the next-closest DIFFERENT person by SPEAKER_ID_KNOWN_MARGIN. The margin
    guard lets the threshold sit low (0.50, where a real returning speaker actually
    scores) without false-matching a different known voice when two candidates are close.
    """
    scored = rank_speakers(audio_array)
    if not scored:
        return (None, None, 0.0)
    _log_scoreboard(scored)
    best_id, name, best_sim, _n = scored[0]
    if best_sim < config.SPEAKER_ID_SIMILARITY_THRESHOLD:
        return (None, None, 0.0)
    second = scored[1][2] if len(scored) > 1 else -1.0
    margin = required_ambiguity_margin(scored)
    if (best_sim - second) < margin:
        logger.info(
            "[speaker_id] ambiguous: %s#%s=%.3f vs next=%.3f (margin %.3f < %.2f) — no match",
            name, best_id, best_sim, second, best_sim - second, margin,
        )
        return (None, None, 0.0)
    if best_sim < 0.80:
        logger.warning(
            "[speaker_id] LOW-CONFIDENCE match person_id=%s name=%r score=%.3f (< 0.80) — treat with caution",
            best_id, name, best_sim,
        )
    else:
        logger.info(
            "[speaker_id] matched person_id=%s name=%r score=%.3f",
            best_id, name, best_sim,
        )
    return (best_id, name, float(best_sim))


_BACKEND_EMBED_DIMS = {"ecapa": 192, "resemblyzer": 256}


def comparable_print_count(person_id) -> int:
    """How many enrolled voice clips this person has under the ACTIVE embedder.

    0 is the VOICELESS-FACE signature: this person's speech cannot match their
    own row, so it necessarily lands on someone ELSE's print — at whatever score
    the embedder hands their nearest acoustic neighbor (field 2026-08-23 21:08:
    print-less PJ scored 0.79-0.94 on Bret's centroid, turn after turn, while
    PJ's recognized face was on camera). Rows from the other embedder don't
    count: they can never match a live query."""
    try:
        rows = db.fetchall(
            "SELECT encoding FROM biometrics WHERE type = 'voice' AND person_id = ?",
            (int(person_id),),
        )
    except Exception:
        return 0
    try:
        backend = voice_score.active_backend()
    except Exception:
        backend = None
    dim = _BACKEND_EMBED_DIMS.get(str(backend or "").lower())
    if dim is None:
        return len(rows)
    count = 0
    for row in rows:
        arr = np.frombuffer(bytes(row["encoding"]), dtype=np.float32)
        if arr.shape[0] == dim:
            count += 1
    return count


def enroll_voice(
    person_id: int,
    audio_array: np.ndarray,
    *,
    source: str = "",
    transcript: Optional[str] = None,
) -> bool:
    """Compute an embedding from audio and store it as a voice biometric for person_id.

    Logs the clip's PROVENANCE before storing it — voiced/buffer seconds, word count,
    and how the clip scored against every existing print — so a poisoned print can
    be traced to the turn that made it. Field 2026-08-29 11:21: the introduction
    capture enrolled a clip as PJ that the scoreboard read as Bret 0.604 / PJ 0.552,
    with Bret alone in the room; nothing recorded that at the time."""
    embedding = get_embedding(audio_array)
    if embedding is None:
        logger.warning(
            "[speaker_id] enroll_voice: could not compute embedding for person_id=%s",
            person_id,
        )
        return False
    try:
        words = len(str(transcript or "").split()) if transcript else None
        logger.info(
            "[speaker_id] enroll provenance person_id=%s source=%s voiced=%.2fs "
            "buffer=%.2fs words=%s scored_against_existing: %s",
            person_id, source or "unspecified", voiced_secs(audio_array),
            buffer_secs(audio_array), "n/a" if words is None else words,
            format_scoreboard(rank_embedding(embedding)) or "(no prints yet)",
        )
    except Exception as exc:
        logger.debug("[speaker_id] enroll provenance log failed: %s", exc)
    people.add_biometric(person_id, "voice", embedding)
    logger.info("[speaker_id] enrolled voice biometric for person_id=%s", person_id)
    return True
