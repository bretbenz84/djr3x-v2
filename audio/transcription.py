import io
import logging
import re
import wave

import numpy as np

import config

logger = logging.getLogger(__name__)

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
    return False


def transcribe(audio_array: np.ndarray) -> str:
    """Transcribe a float32 numpy array (16 kHz mono) and return a cleaned string.

    Tries mlx_whisper first; falls back to the OpenAI Whisper API if unavailable
    or if the local call raises. Returns an empty string on failure.
    """
    raw = ""
    backend = "none"

    if _MLX_AVAILABLE:
        try:
            result = mlx_whisper.transcribe(
                audio_array,
                path_or_hf_repo=config.WHISPER_LOCAL_MODEL,
                initial_prompt=config.WHISPER_INITIAL_PROMPT,
            )
            raw = result.get("text", "").strip()
            backend = "mlx_whisper"
        except Exception as exc:
            logger.warning("mlx_whisper failed (%s), falling back to OpenAI Whisper", exc)

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
