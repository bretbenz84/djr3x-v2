import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import config
from memory import database as db
from memory import people

logger = logging.getLogger(__name__)

_encoder = None
_UNAVAILABLE = False


def _get_encoder():
    global _encoder, _UNAVAILABLE
    if _UNAVAILABLE:
        return None
    if _encoder is not None:
        return _encoder
    try:
        from resemblyzer import VoiceEncoder
        weights = Path(config.RESEMBLYZER_MODEL_DIR) / "pretrained.pt"
        _encoder = VoiceEncoder(weights_fpath=weights)
        logger.info("Resemblyzer encoder loaded from %s", weights)
    except Exception as exc:
        logger.warning(
            "Resemblyzer unavailable (%s) — speaker identification disabled", exc
        )
        _UNAVAILABLE = True
    return _encoder


def preload() -> bool:
    """Load the voice encoder before the first live speech turn."""
    start = time.monotonic()
    encoder = _get_encoder()
    if encoder is None:
        return False
    try:
        # Warm this import too; get_embedding() needs it on the first turn.
        from resemblyzer import preprocess_wav
        sample_rate = int(getattr(config, "AUDIO_SAMPLE_RATE", 16000) or 16000)
        samples = max(1, int(sample_rate * 0.75))
        t = np.arange(samples, dtype=np.float32) / float(sample_rate)
        dummy = (0.001 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
        wav = preprocess_wav(dummy, source_sr=sample_rate)
        encoder.embed_utterance(wav)
    except Exception as exc:
        logger.warning("[speaker_id] preload failed while warming embedding path: %s", exc)
    logger.info("[speaker_id] preloaded encoder in %.3fs", time.monotonic() - start)
    return True


def get_embedding(audio_array: np.ndarray) -> Optional[np.ndarray]:
    """Preprocess audio and return a normalized float32 embedding, or None on failure."""
    encoder = _get_encoder()
    if encoder is None:
        return None
    try:
        from resemblyzer import preprocess_wav
        wav = preprocess_wav(audio_array, source_sr=config.AUDIO_SAMPLE_RATE)
        embedding = encoder.embed_utterance(wav)
        norm = np.linalg.norm(embedding)
        return (embedding / (norm + 1e-10)).astype(np.float32)
    except Exception as exc:
        logger.error("get_embedding failed: %s", exc)
        return None


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
    rows = db.fetchall(
        "SELECT person_id, encoding FROM biometrics WHERE type = 'voice'"
    )
    if not rows:
        return []

    query = embedding.astype(np.float32)
    query_norm = query / (np.linalg.norm(query) + 1e-10)

    per_person: dict[int, list[np.ndarray]] = {}
    for row in rows:
        stored = np.frombuffer(bytes(row["encoding"]), dtype=np.float32)
        if stored.shape != query.shape:
            logger.warning(
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
        sim = float(np.dot(centroid, query_norm))
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


def _log_scoreboard(scored: list) -> None:
    # Same "Name#id=score" format tooling/test_voice_id.py read (now one row per person).
    parts = [f"{nm}#{pid}={sim:.3f}" for pid, nm, sim, _n in scored[:3]]
    logger.info(
        "[speaker_id] scan — threshold=%.3f, candidates: %s",
        config.SPEAKER_ID_SIMILARITY_THRESHOLD,
        ", ".join(parts),
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


def enroll_voice(person_id: int, audio_array: np.ndarray) -> bool:
    """Compute an embedding from audio and store it as a voice biometric for person_id."""
    embedding = get_embedding(audio_array)
    if embedding is None:
        logger.warning(
            "[speaker_id] enroll_voice: could not compute embedding for person_id=%s",
            person_id,
        )
        return False
    people.add_biometric(person_id, "voice", embedding)
    logger.info("[speaker_id] enrolled voice biometric for person_id=%s", person_id)
    return True
