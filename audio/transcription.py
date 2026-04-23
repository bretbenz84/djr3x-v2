import io
import logging
import re
import wave
from pathlib import Path

import numpy as np

import config

logger = logging.getLogger(__name__)
_WHISPER_LOCAL_DIR = (Path(__file__).resolve().parents[1] / config.WHISPER_MODEL_DIR).resolve()
_WARNED_MISSING_LOCAL_MODEL = False

try:
    import mlx_whisper
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False


def _float32_to_wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _apply_corrections(text: str) -> str:
    for pattern, replacement in config.WHISPER_CORRECTIONS.items():
        # re.escape so literal dots/parens in keys work; IGNORECASE for case-insensitive match;
        # replacement value is used verbatim so its casing is always preserved.
        text = re.sub(re.escape(pattern), replacement, text, flags=re.IGNORECASE)
    return text


def _is_hallucination(text: str) -> bool:
    lower = text.lower().strip()
    # Compare against full-utterance matches only (after basic normalization).
    # Substring matching is too aggressive and can hide valid speech.
    normalized = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9'\s]", " ", lower)).strip()
    for phrase in config.HALLUCINATION_BLOCKLIST:
        phrase_norm = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9'\s]", " ", phrase.lower())).strip()
        if normalized == phrase_norm:
            return True
    # Minimum meaningful content check — discard pure punctuation/whitespace junk.
    stripped = re.sub(r"[^a-z0-9]", "", normalized)
    if len(stripped) < config.WHISPER_MIN_CHARS:
        return True
    # Minimum meaningful word count — words longer than 2 characters are considered
    # substantive; short tokens like "uh", "um", "ah" do not count.
    meaningful = [w for w in re.findall(r"[a-zA-Z0-9']+", normalized) if len(w) > 2]
    if len(meaningful) < config.WHISPER_MIN_WORDS:
        return True

    # Repetition pattern: any single word appearing more than threshold times is a loop artifact.
    words = [w.lower() for w in re.findall(r"[a-zA-Z0-9']+", normalized)]
    if words:
        from collections import Counter
        if max(Counter(words).values()) > config.WHISPER_REPETITION_THRESHOLD:
            return True
    # Non-Latin alphabetic characters (e.g. Japanese, Chinese, Arabic) indicate
    # Whisper hallucinating in another language on near-silence or ambient noise.
    # U+024F is the last code point in Latin Extended-B; anything higher that is
    # also alphabetic is a non-Latin script character.
    if any(c.isalpha() and ord(c) > 0x024F for c in text):
        return True
    return False


def transcribe(audio_array: np.ndarray) -> str:
    """Transcribe a float32 numpy array (16 kHz mono) and return a cleaned string.

    Tries mlx_whisper first; falls back to the OpenAI Whisper API if unavailable
    or if the local call raises. Returns an empty string on failure.
    """
    raw = ""
    backend = "none"
    local_model_ready = (_WHISPER_LOCAL_DIR / "config.json").exists()

    if _MLX_AVAILABLE:
        if local_model_ready:
            try:
                result = mlx_whisper.transcribe(
                    audio_array,
                    path_or_hf_repo=str(_WHISPER_LOCAL_DIR),
                    initial_prompt=config.WHISPER_INITIAL_PROMPT,
                    language=config.WHISPER_LANGUAGE,
                )
                raw = result.get("text", "").strip()
                backend = "mlx_whisper"
            except Exception as exc:
                logger.warning("mlx_whisper failed (%s), falling back to OpenAI Whisper", exc)
        else:
            global _WARNED_MISSING_LOCAL_MODEL
            if not _WARNED_MISSING_LOCAL_MODEL:
                logger.warning(
                    "Local Whisper model missing at %s (config.json not found). "
                    "Run setup_assets.py; falling back to OpenAI Whisper.",
                    _WHISPER_LOCAL_DIR,
                )
                _WARNED_MISSING_LOCAL_MODEL = True

    if not raw:
        try:
            import apikeys
            from openai import OpenAI

            client = OpenAI(api_key=apikeys.OPENAI_API_KEY)
            wav_bytes = _float32_to_wav_bytes(audio_array)
            buf = io.BytesIO(wav_bytes)
            buf.name = "audio.wav"
            response = client.audio.transcriptions.create(
                model=config.WHISPER_FALLBACK_MODEL,
                file=buf,
                prompt=config.WHISPER_INITIAL_PROMPT,
                language=config.WHISPER_LANGUAGE,
            )
            raw = response.text.strip()
            backend = "openai_whisper"
        except Exception as exc:
            logger.error("OpenAI Whisper fallback failed: %s", exc)
            return ""

    if not raw:
        return ""

    if _is_hallucination(raw):
        logger.info(
            "[transcription] hallucination filtered | backend=%s | raw=%r",
            backend, raw,
        )
        return ""

    cleaned = _apply_corrections(raw)
    logger.info(
        "[transcription] backend=%s | raw=%r | cleaned=%r",
        backend, raw, cleaned,
    )
    return cleaned
