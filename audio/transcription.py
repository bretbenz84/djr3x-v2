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


# ── Qwen3-ASR backend (primary since 2026-07-31; see TRANSCRIPTION_BACKEND) ──
_QWEN_LOCAL_DIR = (Path(__file__).resolve().parents[1]
                   / getattr(config, "QWEN_ASR_MODEL_DIR", "assets/models/qwen_asr")).resolve()
_qwen_model = None
_QWEN_LOAD_FAILED = False


def _qwen_backend_selected() -> bool:
    return str(getattr(config, "TRANSCRIPTION_BACKEND", "whisper")).strip().lower() == "qwen3"


def _qwen_ready() -> bool:
    return (_QWEN_LOCAL_DIR / "config.json").exists()


def _get_qwen_model():
    """Lazy-load the Qwen3-ASR model once; a load failure latches so a broken
    install degrades to the whisper path instead of retrying every segment."""
    global _qwen_model, _QWEN_LOAD_FAILED
    if _qwen_model is not None or _QWEN_LOAD_FAILED:
        return _qwen_model
    try:
        from mlx_audio.stt.utils import load_model
        from utils.mlx_lock import MLX_LOCK
        with MLX_LOCK:
            _qwen_model = load_model(str(_QWEN_LOCAL_DIR))
        logger.info("[transcription] Qwen3-ASR loaded from %s", _QWEN_LOCAL_DIR)
    except Exception as exc:
        _QWEN_LOAD_FAILED = True
        logger.warning("[transcription] Qwen3-ASR load failed (%s) — "
                       "falling back to whisper for this run", exc)
    return _qwen_model


def _qwen_transcribe(audio_array: np.ndarray) -> "tuple[str, float | None]":
    """Decode with Qwen3-ASR, returning (text, mean per-token logprob).

    The public mlx_audio generate() discards logprobs, so this walks the
    streaming API and keeps them: the mean token logprob is the .confident
    signal (see Transcript) — calibrated 2026-07-31, clean decodes sit at
    0.0..-0.03 and truncated/garbage captures at -0.75 and below."""
    model = _get_qwen_model()
    if model is None:
        raise RuntimeError("Qwen3-ASR model unavailable")
    from utils.mlx_lock import MLX_LOCK
    tokens: "list[int]" = []
    logps: "list[float]" = []
    max_tokens = int(getattr(config, "QWEN_ASR_MAX_TOKENS", 256))
    # MLX_LOCK: same shared-Metal-runtime rule as mlx_whisper below — concurrent
    # evaluation with the local TTS engine is a fatal native crash.
    with MLX_LOCK:
        for token, logprobs in model.stream_generate(
            audio_array,
            language=str(getattr(config, "WHISPER_LANGUAGE", "en") or "en"),
            max_tokens=max_tokens,
        ):
            t = int(token)
            tokens.append(t)
            try:
                logps.append(float(logprobs[t]))
            except Exception:
                pass
        import mlx.core as mx
        mx.synchronize()
    text = model._tokenizer.decode(tokens).strip()
    return text, (sum(logps) / len(logps)) if logps else None


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
    """Warm the active local ASR backend so the first live utterance does not
    pay setup cost. With TRANSCRIPTION_BACKEND="qwen3" this loads Qwen3-ASR and
    runs a short silent decode; whisper preloads as before (and remains the
    fallback, so a missing qwen model degrades instead of failing)."""
    if _qwen_backend_selected() and _qwen_ready():
        try:
            start = time.monotonic()
            dummy = np.zeros(int(config.AUDIO_SAMPLE_RATE * 0.25), dtype=np.float32)
            _qwen_transcribe(dummy)
            logger.info("[transcription] preloaded Qwen3-ASR in %.3fs",
                        time.monotonic() - start)
            return True
        except Exception as exc:
            logger.warning("[transcription] Qwen3-ASR preload failed: %s — "
                           "whisper path will be used", exc)
    elif _qwen_backend_selected():
        logger.warning(
            "[transcription] TRANSCRIPTION_BACKEND=qwen3 but model missing at %s "
            "(run setup_assets.py) — whisper path will be used", _QWEN_LOCAL_DIR,
        )
    if not _MLX_AVAILABLE:
        return False
    if not _local_model_ready():
        return False
    try:
        start = time.monotonic()
        dummy = np.zeros(int(config.AUDIO_SAMPLE_RATE * 0.25), dtype=np.float32)
        from utils.mlx_lock import MLX_LOCK
        with MLX_LOCK:   # serialize vs the local-TTS engine (shared MLX runtime)
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
        # Word-boundary anchored when the key starts/ends with a word character —
        # a bare substring replace corrupted embedded matches ("vibrate" contains
        # "brat", "breadth" contains "bread").
        escaped = re.escape(pattern)
        if pattern and pattern[0].isalnum():
            escaped = r"\b" + escaped
        if pattern and pattern[-1].isalnum():
            escaped = escaped + r"\b"
        text = re.sub(escaped, replacement, text, flags=re.IGNORECASE)
    return text


# Optional address/politeness wrappers allowed around a standalone-corrected
# phrase — "Rex, roast meat." / "please roast meat" are still the bare command.
_STANDALONE_LEAD_RE = (
    r"^(?P<lead>(?:(?:hey|ok|okay)\s+)?(?:rex\s*[,!]?\s*)?(?:please\s+)?)"
)
_STANDALONE_TAIL_RE = r"(?P<tail>(?:\s*,?\s*please)?[\s.!?]*)$"


def _apply_standalone_corrections(text: str) -> str:
    """Whole-utterance homophone fixes (config.WHISPER_STANDALONE_CORRECTIONS).

    Unlike WHISPER_CORRECTIONS these NEVER rewrite the phrase inside a longer
    sentence — "roast meat" is a perfectly real thing to talk about; only a
    bare "Roast meat." (optionally wrapped in an address/'please') is the ASR
    mangling the command "roast me" (field 2026-08-02, qwen3 backend)."""
    stripped = (text or "").strip()
    if not stripped:
        return text
    for phrase, replacement in getattr(
        config, "WHISPER_STANDALONE_CORRECTIONS", {}
    ).items():
        pattern = re.compile(
            _STANDALONE_LEAD_RE + re.escape(phrase) + _STANDALONE_TAIL_RE,
            re.IGNORECASE,
        )
        m = pattern.match(stripped)
        if m:
            fix = replacement
            # Preserve sentence-initial capitalization ("Roast meat." → "Roast me.")
            if not m.group("lead") and stripped[:1].isupper() and fix:
                fix = fix[0].upper() + fix[1:]
            fixed = (m.group("lead") + fix + m.group("tail")).strip()
            logger.info(
                "[transcription] standalone correction: %r -> %r", stripped, fixed
            )
            return fixed
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


class Transcript(str):
    """The cleaned text, plus how much Whisper actually believed it.

    Subclasses str so every existing caller keeps working unchanged — the extra
    fields are there for the ones that need to decide whether this turn is solid
    enough to LEARN from.

    ``confident`` is False when Whisper's own decode statistics say it was
    guessing. Whisper does not fail loudly: handed a single quiet word or a
    half-captured phrase it emits a fluent, plausible sentence with no outward
    sign anything went wrong. Field 2026-07-25: "wine" was transcribed
    "I'm going to split it.", "This is the workshop room" became "Shop room.",
    and an utterance that produced "Spice it." enrolled a PERSON NAMED SPICE.
    Every one of those was written to durable memory and mined for proactive
    questions days later. avg_logprob / no_speech_prob are the signal that
    separates them, and they were being thrown away.
    """

    avg_logprob: "float | None" = None
    no_speech_prob: "float | None" = None
    confident: bool = True
    backend: str = "none"

    def __new__(cls, text: str, *, avg_logprob=None, no_speech_prob=None,
                confident: bool = True, backend: str = "none"):
        obj = super().__new__(cls, text)
        obj.avg_logprob = avg_logprob
        obj.no_speech_prob = no_speech_prob
        obj.confident = bool(confident)
        obj.backend = backend
        return obj


def _decode_stats(result: dict) -> "tuple[float | None, float | None]":
    """Mean avg_logprob and max no_speech_prob across the decoded segments."""
    segments = (result or {}).get("segments") or []
    logps = [s.get("avg_logprob") for s in segments if isinstance(s, dict)
             and isinstance(s.get("avg_logprob"), (int, float))]
    nsps = [s.get("no_speech_prob") for s in segments if isinstance(s, dict)
            and isinstance(s.get("no_speech_prob"), (int, float))]
    return (
        (sum(logps) / len(logps)) if logps else None,
        max(nsps) if nsps else None,
    )


def _is_confident(avg_logprob, no_speech_prob, backend: str = "mlx_whisper") -> bool:
    """Whether this decode is solid enough to LEARN from (not to act on).

    Deliberately permissive: the far-field SNR here is 13-15 dB and genuine
    speech routinely scores poorly, so a strict gate would make Rex deaf. A
    failing turn is still heard, replied to, and acted on — it just doesn't
    become a durable fact, a person's name, or a room.

    The floor is backend-specific — Whisper's avg_logprob and Qwen3's mean
    token logprob live on different scales (Qwen3 at temperature 0 is far more
    peaked: clean decodes ~0.0, garbage below -0.7)."""
    if backend == "qwen3_asr":
        floor = float(getattr(config, "QWEN_ASR_TRUST_MIN_AVG_LOGPROB", -0.35))
    else:
        floor = float(getattr(config, "WHISPER_TRUST_MIN_AVG_LOGPROB", -0.85))
    ceiling = float(getattr(config, "WHISPER_TRUST_MAX_NO_SPEECH_PROB", 0.5))
    if avg_logprob is not None and avg_logprob < floor:
        return False
    if no_speech_prob is not None and no_speech_prob > ceiling:
        return False
    return True


def transcribe(audio_array: np.ndarray) -> "Transcript":
    """Transcribe a float32 numpy array (16 kHz mono) and return a cleaned string.

    Tries mlx_whisper first; falls back to the OpenAI Whisper API if unavailable
    or if the local call raises. Returns an empty string on failure. The result is
    a Transcript (a str) carrying Whisper's decode confidence — see that class.
    """
    raw = ""
    backend = "none"
    avg_logprob = no_speech_prob = None
    local_decoded_ok = False
    local_model_ready = _local_model_ready()

    if _qwen_backend_selected() and _qwen_ready() and not _QWEN_LOAD_FAILED:
        try:
            raw, avg_logprob = _qwen_transcribe(audio_array)
            raw = raw.strip()
            backend = "qwen3_asr"
            local_decoded_ok = True
        except Exception as exc:
            logger.warning("Qwen3-ASR failed (%s), falling back to local whisper", exc)

    if not local_decoded_ok and _MLX_AVAILABLE:
        if local_model_ready:
            try:
                # MLX_LOCK: mlx_whisper shares the MLX/Metal runtime with the
                # local Qwen3-TTS engine; concurrent evaluation from two threads
                # is a fatal native crash (PyThreadState_Get / trap 5, observed
                # live 2026-07-19). TTS holds the lock per ~0.3s chunk, so a
                # transcription waits at most one chunk, never a whole utterance.
                from utils.mlx_lock import MLX_LOCK
                with MLX_LOCK:
                    result = mlx_whisper.transcribe(
                        audio_array,
                        path_or_hf_repo=str(_WHISPER_LOCAL_DIR),
                        **_mlx_decode_options(),
                    )
                raw = result.get("text", "").strip()
                avg_logprob, no_speech_prob = _decode_stats(result)
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
        return Transcript("", backend=backend)

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
            return Transcript("", backend=backend)

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
        return Transcript("", backend=backend)

    if _is_hallucination(raw):
        logger.info(
            "[transcription] hallucination filtered | backend=%s | raw=%r",
            backend, raw,
        )
        return Transcript("", backend=backend)

    cleaned = _apply_standalone_corrections(_apply_corrections(raw))
    confident = _is_confident(avg_logprob, no_speech_prob, backend)
    logger.info(
        "[transcription] backend=%s | raw=%r | cleaned=%r | avg_logprob=%s "
        "no_speech_prob=%s trusted=%s",
        backend, raw, cleaned,
        "n/a" if avg_logprob is None else f"{avg_logprob:.2f}",
        "n/a" if no_speech_prob is None else f"{no_speech_prob:.2f}",
        confident,
    )
    if not confident:
        logger.info(
            "[transcription] LOW CONFIDENCE — Rex will answer this but not learn "
            "from it (no facts, no names, no rooms): %r", cleaned,
        )
    return Transcript(cleaned, avg_logprob=avg_logprob, no_speech_prob=no_speech_prob,
                      confident=confident, backend=backend)
