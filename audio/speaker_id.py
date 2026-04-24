import logging
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


def identify_speaker(
    audio_array: np.ndarray,
) -> Tuple[Optional[int], Optional[str], float]:
    """Return (person_id, name, score) for the best voice match above threshold.

    Returns (None, None, 0.0) if Resemblyzer is unavailable, no biometrics are stored,
    or the best match falls below SPEAKER_ID_SIMILARITY_THRESHOLD.
    """
    embedding = get_embedding(audio_array)
    if embedding is None:
        return (None, None, 0.0)

    # find_by_voice in people.py doesn't expose the similarity score, so we do the
    # biometric scan here directly — same query pattern people.py uses internally.
    rows = db.fetchall(
        "SELECT person_id, encoding FROM biometrics WHERE type = 'voice'"
    )
    if not rows:
        return (None, None, 0.0)

    # embedding is already normalized by get_embedding; normalize defensively anyway.
    query = embedding.astype(np.float32)
    query_norm = query / (np.linalg.norm(query) + 1e-10)

    # Score every stored voice so we can log a ranked diagnostic line.
    scored: list[tuple[int, float]] = []  # (person_id, similarity)
    for row in rows:
        stored = np.frombuffer(bytes(row["encoding"]), dtype=np.float32)
        if stored.shape != query.shape:
            logger.warning(
                "voice embedding shape mismatch: stored %s vs query %s",
                stored.shape, query.shape,
            )
            continue
        stored_norm = stored / (np.linalg.norm(stored) + 1e-10)
        sim = float(np.dot(stored_norm, query_norm))
        scored.append((row["person_id"], sim))

    scored.sort(key=lambda t: t[1], reverse=True)

    # Always log the top candidates with their scores — makes voice-print
    # tuning possible from the live log, not just unit tests.
    top_summary_parts = []
    for pid, sim in scored[:3]:
        person = people.get_person(pid)
        nm = (person.get("name") if person else None) or "?"
        top_summary_parts.append(f"{nm}#{pid}={sim:.3f}")
    top_summary = ", ".join(top_summary_parts) if top_summary_parts else "(no voice prints enrolled)"
    logger.info(
        "[speaker_id] scan — threshold=%.3f, candidates: %s",
        config.SPEAKER_ID_SIMILARITY_THRESHOLD, top_summary,
    )

    if not scored:
        return (None, None, 0.0)

    best_id, best_sim = scored[0]
    if best_sim < config.SPEAKER_ID_SIMILARITY_THRESHOLD:
        return (None, None, 0.0)

    person = people.get_person(best_id)
    name = person.get("name") if person else None
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
