import difflib
import io
import logging
import re
import threading
import time
import wave
from collections import Counter, deque
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


# ── Context biasing (Qwen3-ASR) ──────────────────────────────────────────────
# Qwen3-ASR accepts a system prompt of context text and biases decoding toward
# vocabulary in it. Rex's own recent lines are the highest-value context: a
# reply usually re-uses the entities he just named (field 2026-08-02: Rex said
# "Lake Folsom today ..." and the user's answer "we're not going to Lake
# Folsom anymore" decoded as "like falsum" — so the trip cancellation never
# reached the memory layer). Static vocab (names/places) rides along.
_context_lock = threading.Lock()
_recent_rex_lines: "deque[str]" = deque(maxlen=4)


def note_rex_line(text: str) -> None:
    """Record a line Rex spoke, for ASR context biasing. Cheap; called from
    utils.conv_log.log_rex on every spoken line."""
    cleaned = (text or "").strip()
    if not cleaned:
        return
    with _context_lock:
        _recent_rex_lines.append(cleaned)


def _asr_context_prompt() -> "str | None":
    """Build the Qwen3-ASR biasing context, or None when disabled/empty."""
    if not bool(getattr(config, "QWEN_ASR_CONTEXT_BIAS_ENABLED", True)):
        return None
    vocab = [str(v) for v in getattr(config, "QWEN_ASR_CONTEXT_VOCAB", ()) if v]
    n_lines = int(getattr(config, "QWEN_ASR_CONTEXT_REX_LINES", 2))
    with _context_lock:
        lines = list(_recent_rex_lines)[-n_lines:] if n_lines > 0 else []
    parts = []
    if vocab:
        parts.append("Names and places that may occur: " + ", ".join(vocab) + ".")
    if lines:
        # NEWEST line first: the user's reply usually re-uses the entities from
        # the line Rex JUST spoke, and the max_chars cap truncates from the end
        # — so the freshest context must never be the part that gets cut.
        parts.append(
            "The audio replies to a droid who just said: "
            + " ".join(reversed(lines))
        )
    if not parts:
        return None
    prompt = ("This audio is one side of a live spoken conversation. "
              + " ".join(parts))
    # Prefill cost is linear in prompt length (~0.5ms/char measured 2026-08-02:
    # 363ch of live context = +0.18s per decode) — the cap bounds worst-case
    # added latency, not just token count.
    max_chars = int(getattr(config, "QWEN_ASR_CONTEXT_MAX_CHARS", 400))
    prompt = prompt[:max_chars]
    # Remember the EXACT string we sent. On near-silence Qwen3-ASR emits its own
    # system prompt back verbatim at logprob 0.0 (the documented 2026-08-02
    # behavior), and the echo guard's candidate set was built for the two known
    # shapes — a Rex line copied back, and the vocab list copied back — so the
    # fixed preamble was never in it. Field 2026-08-20 20:16:38: the whole prompt
    # came back as a 54-word "utterance" and only the speaking-rate backstop caught
    # it. On a longer capture (a 10s VAD segment is ordinary) that backstop does
    # not fire, and the decode lands trusted=True carrying every name in
    # QWEN_ASR_CONTEXT_VOCAB — exactly the shape that poisons people.db and
    # resurfaces later as a proactive question.
    global _last_context_prompt
    with _context_lock:
        _last_context_prompt = prompt
    return prompt


# The exact biasing prompt most recently handed to the decoder (see
# _asr_context_prompt). Read by _context_echo_hallucination.
_last_context_prompt: "str | None" = None

# Fragments of the fixed preamble that can never be real speech TO Rex. Cheap
# belt-and-braces alongside the similarity/coverage checks, and independent of how
# much of the prompt the model chose to recite.
_CONTEXT_PROMPT_MARKERS = (
    "one side of a live spoken conversation",
    "names and places that may occur",
    "the audio replies to a droid who just said",
)


def _norm_for_echo(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", (text or "").lower()).strip()


def _context_echo_hallucination(text: str) -> bool:
    """True when a Qwen3-ASR decode is just the biasing context copied back.

    Measured 2026-08-02: on silence/noise the model outputs the context text
    VERBATIM at full confidence (logprob 0.0) — so a biased decode that is
    near-identical to a context line or the vocab list is a hallucination,
    not speech. This also acts as the reference-text guard for the echo
    capture seam: a transcript that IS Rex's own recent line is his residual,
    not the user.

    Two checks: (1) per-candidate similarity, and (2) COVERAGE — field
    2026-08-02 12:36: a 1.9s echo capture decoded as BOTH startup lines
    concatenated (44 words), so each single line only matched ~0.5 and the
    ratio check waved it through. Stripping every known line out of the
    transcript and rejecting when almost nothing remains catches multi-line
    echo verbatim."""
    norm = _norm_for_echo(text)
    if not norm:
        return False
    # Structural tell first: no human says the preamble to a droid, at any length.
    for marker in _CONTEXT_PROMPT_MARKERS:
        if _norm_for_echo(marker) in norm:
            logger.info(
                "[transcription] context-PROMPT regurgitation rejected "
                "(contains %r): %r", marker, text)
            return True
    candidates = [", ".join(
        str(v) for v in getattr(config, "QWEN_ASR_CONTEXT_VOCAB", ()) if v)]
    with _context_lock:
        rex_lines = list(_recent_rex_lines)
        prompt = _last_context_prompt
    if prompt:
        candidates.insert(0, prompt)
    candidates.extend(rex_lines)
    for cand in candidates:
        cand_norm = _norm_for_echo(cand)
        if not cand_norm:
            continue
        ratio = difflib.SequenceMatcher(None, norm, cand_norm).ratio()
        if ratio >= float(getattr(config, "QWEN_ASR_CONTEXT_ECHO_RATIO", 0.85)):
            logger.info(
                "[transcription] context-echo hallucination rejected "
                "(ratio %.2f vs %r): %r", ratio, cand[:60], text)
            return True
    # Coverage: remove every recent Rex line from the transcript; if what's
    # left is a small fraction, the "utterance" is composed of his own lines.
    residue = norm
    matched_any = False
    for cand in ([prompt] if prompt else []) + rex_lines:
        cand_norm = _norm_for_echo(cand)
        if cand_norm and len(cand_norm) >= 12 and cand_norm in residue:
            residue = residue.replace(cand_norm, " ")
            matched_any = True
    if matched_any:
        residue = re.sub(r"\s+", " ", residue).strip()
        max_residue = float(getattr(config, "QWEN_ASR_ECHO_MAX_RESIDUE_FRAC", 0.2))
        if len(residue) <= max(12, int(len(norm) * max_residue)):
            logger.info(
                "[transcription] context-echo hallucination rejected "
                "(coverage; residue %r): %r", residue[:40], text)
            return True
    return False


def _impossible_speaking_rate(text: str, duration_secs: float) -> bool:
    """True when the decode packs more words than the audio could physically
    hold (field 2026-08-02 12:36: 44 words 'heard' in a 1.89s echo capture —
    the biased decoder completed Rex's startup lines from faint residual at
    logprob 0.0). Human speech tops out ~4-5 words/sec; the default cap of 6
    only rejects the physically impossible."""
    if duration_secs <= 0.5:
        return False
    words = len((text or "").split())
    if words < 8:
        return False
    max_wps = float(getattr(config, "ASR_MAX_WORDS_PER_SEC", 6.0))
    if words / duration_secs > max_wps:
        logger.info(
            "[transcription] impossible speaking rate rejected "
            "(%d words in %.2fs = %.1f wps): %r",
            words, duration_secs, words / duration_secs, (text or "")[:80])
        return True
    return False


def _rex_speech_word_set() -> "set[str]":
    """Content words (4+ letters) from Rex's recent lines and the biasing prompt."""
    with _context_lock:
        lines = list(_recent_rex_lines)
        prompt = _last_context_prompt
    words: "set[str]" = set()
    for cand in ([prompt] if prompt else []) + lines:
        words.update(w for w in _norm_for_echo(cand).split() if len(w) >= 4)
    return words


def _overlaps_recent_rex_speech(text: str) -> bool:
    """True when a rejected decode is built out of REX'S OWN recent words.

    Field 2026-08-27 13:34:07: the post-boot capture seam decoded as his ready
    line doubled up (33 words in 1.98s) and only the speaking-rate backstop
    caught it — the similarity guard missed because doubling halves every ratio
    and the imperfect re-decode broke the verbatim coverage strip. The softer
    sibling of _context_echo_hallucination: that one has to be sure enough to
    throw a transcript away, this one only has to be sure enough to distrust a
    RETRY of audio the decoder already read Rex's lines out of.
    """
    try:
        rex_words = _rex_speech_word_set()
        if not rex_words:
            return False
        words = [w for w in _norm_for_echo(text).split() if len(w) >= 4]
        if not words:
            return False
        hits = sum(1 for w in words if w in rex_words)
        return (hits / len(words)) >= float(
            getattr(config, "ASR_ECHO_CLASS_WORD_OVERLAP", 0.6))
    except Exception as exc:
        # Whatever got here was already rejected as physically impossible, so it
        # is junk either way — if the probe breaks, call it echo and skip the
        # rescue rather than hand a fabrication to the turn pipeline.
        logger.debug("[transcription] rex-overlap probe failed: %s", exc)
        return True


def _qwen_transcribe(
    audio_array: np.ndarray, *, use_context: bool = True,
) -> "tuple[str, float | None]":
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
    context = _asr_context_prompt() if use_context else None
    wait_started = time.monotonic()
    with MLX_LOCK:
        lock_wait = time.monotonic() - wait_started
        decode_started = time.monotonic()
        for token, logprobs in model.stream_generate(
            audio_array,
            language=str(getattr(config, "WHISPER_LANGUAGE", "en") or "en"),
            max_tokens=max_tokens,
            system_prompt=context,
        ):
            t = int(token)
            tokens.append(t)
            try:
                logps.append(float(logprobs[t]))
            except Exception:
                pass
        import mlx.core as mx
        mx.synchronize()
        decode_secs = time.monotonic() - decode_started
    # Slow-decode diagnostic: separate "sat behind another MLX_LOCK holder" from
    # "the decode itself crawled" (memory pressure, GPU contention). Field
    # 2026-09-01 23:05: a 3s utterance took 6.9s here while speaker-ID on the
    # same clip finished instantly, and nothing at INFO could say which it was.
    try:
        slow_after = float(getattr(config, "ASR_SLOW_DECODE_LOG_SECS", 1.5) or 0.0)
        if slow_after > 0.0 and (lock_wait + decode_secs) >= slow_after:
            audio_secs = len(audio_array) / float(
                getattr(config, "AUDIO_SAMPLE_RATE", 16000) or 16000)
            logger.info(
                "[transcription] slow decode — waited %.2fs for MLX_LOCK, decoded "
                "%.1fs of audio in %.2fs (%d tokens, context=%s)",
                lock_wait, audio_secs, decode_secs, len(tokens),
                "on" if context else "off",
            )
    except Exception:
        pass
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
            duration = len(audio_array) / float(
                getattr(config, "AUDIO_SAMPLE_RATE", 16000))
            raw, avg_logprob = _qwen_transcribe(audio_array)
            raw = raw.strip()
            reject_reason = ""
            if raw:
                if _context_echo_hallucination(raw):
                    reject_reason = "context-echo"
                elif _impossible_speaking_rate(raw, duration):
                    reject_reason = "impossible-rate"
            if reject_reason:
                # ECHO-CLASS: the words the biased decoder produced were REX'S
                # OWN, so his voice is provably in this audio and a second decode
                # of it is a second look at his residual. Field 2026-08-27
                # 13:34:09 — the seam after "Ready to go. Statistically, one of us
                # is about to say something interesting." was rejected at 16.6
                # wps, the unbiased retry read the same residue as a bare "Okay."
                # (-0.69), and Rex answered himself with "Okay what, exactly?"
                # while Bret sat there saying "I didn't say anything".
                echo_class = (
                    reject_reason == "context-echo"
                    or _overlaps_recent_rex_speech(raw)
                )
                raw = ""
                # The BIAS is what broke this decode — the model completed the
                # prompt instead of transcribing — so the audio deserves one
                # unbiased look before the turn is thrown away. Without it, a
                # rejected decode silently costs the whole utterance: field
                # 2026-08-26, a Jeopardy run lost 10 of 59 turns this way, every
                # one of them next to a regurgitation rejection, because the
                # clue Rex had just read was the prompt.
                if bool(getattr(config, "ASR_RETRY_WITHOUT_CONTEXT_ON_ECHO", True)):
                    try:
                        retry, retry_logprob = _qwen_transcribe(
                            audio_array, use_context=False)
                        retry = retry.strip()
                    except Exception as exc:
                        logger.debug("[transcription] unbiased retry failed: %s", exc)
                        retry, retry_logprob = "", None
                    retry_ok = bool(retry) and not _context_echo_hallucination(
                        retry
                    ) and not _impossible_speaking_rate(retry, duration)
                    if (
                        retry_ok
                        and echo_class
                        and bool(getattr(
                            config, "ASR_ECHO_RETRY_REQUIRE_TRUSTED", True))
                        and not (
                            retry_logprob is not None
                            and _is_confident(retry_logprob, None, "qwen3_asr")
                        )
                    ):
                        # Dropping the bias cannot turn Rex's voice into somebody
                        # else's words. All 11 retries of the 2026-08-27 13:33 run
                        # sat behind an echo-class rejection and every one came
                        # back a low-trust fragment ("Oh.", "Okay." -0.69, "Look
                        # me what you got me." -0.55) that he then answered.
                        logger.info(
                            "[transcription] unbiased retry DISCARDED — the "
                            "rejected decode (%s) was Rex's own voice and the "
                            "retry is low-trust (avg_logprob=%s): %r",
                            reject_reason,
                            "n/a" if retry_logprob is None
                            else f"{retry_logprob:.2f}",
                            retry[:80],
                        )
                        retry_ok = False
                    if retry_ok:
                        logger.info(
                            "[transcription] unbiased retry recovered a rejected "
                            "decode (%s): %r", reject_reason, retry[:80],
                        )
                        raw, avg_logprob = retry, retry_logprob
                    else:
                        logger.info(
                            "[transcription] unbiased retry decoded nothing usable "
                            "— segment stays dropped",
                        )
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
