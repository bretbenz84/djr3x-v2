import io
import logging
import re
import time
import wave
from collections import Counter
from pathlib import Path

import numpy as np

import config

logger = logging.getLogger(__name__)
_WHISPER_LOCAL_DIR = (Path(__file__).resolve().parents[1] / config.WHISPER_MODEL_DIR).resolve()
_WARNED_MISSING_LOCAL_MODEL = False

try:
    import mlx_whisper
    _MLX_AVAILABLE = True
except Exception as exc:
    mlx_whisper = None
    _MLX_AVAILABLE = False
    logger.warning("mlx_whisper unavailable; local Whisper disabled: %s", exc)


def _local_model_ready() -> bool:
    return (_WHISPER_LOCAL_DIR / "config.json").exists()


def _mlx_decode_options() -> dict:
    return {
        "initial_prompt": config.WHISPER_INITIAL_PROMPT,
        "language": config.WHISPER_LANGUAGE,
        "temperature": getattr(config, "WHISPER_TEMPERATURE", 0.0),
        "condition_on_previous_text": bool(
            getattr(config, "WHISPER_CONDITION_ON_PREVIOUS_TEXT", False)
        ),
    }


def preload() -> bool:
    """Warm local MLX Whisper so the first live utterance does not pay setup cost."""
    if not _MLX_AVAILABLE:
        return False
    if not _local_model_ready():
        return False
    try:
        start = time.monotonic()
        dummy = np.zeros(int(config.AUDIO_SAMPLE_RATE * 0.25), dtype=np.float32)
        mlx_whisper.transcribe(
            dummy,
            path_or_hf_repo=str(_WHISPER_LOCAL_DIR),
            **_mlx_decode_options(),
        )
        logger.info("[transcription] preloaded local Whisper in %.3fs", time.monotonic() - start)
        return True
    except Exception as exc:
        logger.warning("[transcription] local Whisper preload failed: %s", exc)
        return False


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


# Whisper hallucinates subtitle/credit boilerplate on silence or noise
# ("Subs by www.zeoranger.co.uk", "Thanks for watching", "Amara.org"). These
# phrases / URLs never occur in real speech to Rex, so a substring match here is
# safe (and important: such a hallucination was being attributed to the user,
# which both derailed Rex and kept his "still engaged" timer alive so he never
# noticed the user had left the frame).
_SUBTITLE_HALLUCINATION_RE = re.compile(
    r"\b(?:subs?|subtitle[sd]?|caption[sd]?|transcription|translation)\s+by\b|"
    r"\b(?:closed\s+caption|cc)\s+by\b|"
    r"\bthank(?:s| you)\s+for\s+watching\b|"
    # YouTube-outro family ("and more. I hope you enjoyed this video. I'll see
    # you in the next video." — live hallucination 2026-07-06-22-39, spoken by
    # NOBODY; Whisper's training data leaks video sign-offs onto near-silence).
    r"\bhope\s+you\s+enjoyed\s+(?:this|the|my|that)\s+video\b|"
    r"\bsee\s+you\s+in\s+the\s+next\s+(?:video|one|episode)\b|"
    r"\bsee\s+you\s+(?:guys\s+)?in\s+the\s+next\b|"
    r"\b(?:like\s+and\s+subscribe|(?:please|plz|pls)\.?\s+subscribe|"
    r"don'?t\s+forget\s+to\s+subscribe|subscribe\s+to\s+(?:my|our|the)\s+channel)\b|"
    r"\bamara\.org\b|"
    r"\bwww\.\S+|"
    r"\bhttps?://\S+|"
    r"\b\S+\.(?:com|org|net|io|tv|co\.uk)\b",
    re.IGNORECASE,
)


def _is_hallucination(text: str) -> bool:
    lower = text.lower().strip()
    # Subtitle/credit boilerplate and stray URLs are always hallucinations.
    if _SUBTITLE_HALLUCINATION_RE.search(lower):
        return True
    # Compare against full-utterance matches only (after basic normalization).
    # Substring matching is too aggressive and can hide valid speech.
    normalized = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9'\s]", " ", lower)).strip()
    for phrase in config.HALLUCINATION_BLOCKLIST:
        phrase_norm = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9'\s]", " ", phrase.lower())).strip()
        if normalized == phrase_norm:
            return True
    short_allowlist = {
        re.sub(r"\s+", " ", re.sub(r"[^a-z0-9'\s]", " ", item.lower())).strip()
        for item in getattr(config, "WHISPER_SHORT_UTTERANCE_ALLOWLIST", [])
    }
    if normalized in short_allowlist:
        return False
    filler_blocklist = {
        re.sub(r"\s+", " ", re.sub(r"[^a-z0-9'\s]", " ", item.lower())).strip()
        for item in getattr(config, "WHISPER_FILLER_UTTERANCE_BLOCKLIST", [])
    }
    if normalized in filler_blocklist:
        return True
    # Minimum meaningful content check — discard pure punctuation/whitespace junk.
    stripped = re.sub(r"[^a-z0-9]", "", normalized)
    if len(stripped) < config.WHISPER_MIN_CHARS:
        return True

    # Character-loop artifacts can arrive as one very long token rather than a
    # repeated word, e.g. "Zzzzzzzzzzzzzzzzzzz" on near-silence.
    repeated_char_min = max(
        1,
        int(getattr(config, "WHISPER_REPEATED_CHAR_MIN_RUN", 16) or 16),
    )
    repeated_char_dominance = float(
        getattr(config, "WHISPER_REPEATED_CHAR_DOMINANCE", 0.90) or 0.90
    )
    if not 0 < repeated_char_dominance <= 1:
        repeated_char_dominance = 0.90
    if len(stripped) >= repeated_char_min:
        dominant_count = max(Counter(stripped).values())
        if (
            dominant_count >= repeated_char_min
            and dominant_count / len(stripped) >= repeated_char_dominance
        ):
            return True

    # Minimum meaningful word count — words longer than 2 characters are considered
    # substantive; short tokens like "uh", "um", "ah" do not count.
    meaningful = [w for w in re.findall(r"[a-zA-Z0-9']+", normalized) if len(w) > 2]
    if len(meaningful) < config.WHISPER_MIN_WORDS:
        return True

    # Repetition pattern: a real Whisper loop repeats ONE word until it dominates the
    # utterance ("you you you you"). A varied sentence that naturally reuses a function
    # word ("I like Bach, I like Beethoven, I like Bach") is NOT a loop — counting raw
    # occurrences wrongly discarded real speech. Require BOTH: the top word exceeds the
    # count threshold AND it makes up a large fraction of all words.
    words = [w.lower() for w in re.findall(r"[a-zA-Z0-9']+", normalized)]
    if words:
        top_count = max(Counter(words).values())
        dominance = top_count / len(words)
        dominance_min = float(getattr(config, "WHISPER_REPETITION_DOMINANCE", 0.5) or 0.5)
        if top_count > config.WHISPER_REPETITION_THRESHOLD and dominance >= dominance_min:
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
    local_decoded_ok = False
    local_model_ready = _local_model_ready()

    if _MLX_AVAILABLE:
        if local_model_ready:
            try:
                result = mlx_whisper.transcribe(
                    audio_array,
                    path_or_hf_repo=str(_WHISPER_LOCAL_DIR),
                    **_mlx_decode_options(),
                )
                raw = result.get("text", "").strip()
                backend = "mlx_whisper"
                local_decoded_ok = True
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

    if (
        not raw
        and local_decoded_ok
        and not bool(getattr(config, "WHISPER_FALLBACK_ON_EMPTY", False))
    ):
        # Local Whisper RAN and decoded nothing — that's an answer ("nobody said
        # anything intelligible"), not a failure. Asking the API for a second
        # opinion on near-silence is how the YouTube-outro hallucination reached
        # the reply path ("I hope you enjoyed this video. I'll see you in the
        # next video.", live 2026-07-06-22-39) — and it costs ~2s + a network
        # call per silence. The fallback is for a BROKEN local path only.
        logger.info(
            "[transcription] EMPTY result — segment dropped | backend=%s | "
            "local decoded silence; API fallback skipped", backend,
        )
        return ""

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
        # Both engines heard nothing intelligible. Log it — a silently dropped
        # turn reads as "the robot ignored me" in the field (log 2026-07-06-19-20:
        # one utterance vanished here with no trace). RMS separates true silence
        # (gain/AEC problem) from real speech the models couldn't decode.
        try:
            dur = len(audio_array) / float(config.AUDIO_SAMPLE_RATE)
            rms = float(np.sqrt(np.mean(np.square(audio_array))))
        except Exception:
            dur, rms = -1.0, -1.0
        logger.info(
            "[transcription] EMPTY result — segment dropped | backend=%s | %.2fs audio, rms=%.4f",
            backend, dur, rms,
        )
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
