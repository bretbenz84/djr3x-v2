# config.py — DJ-R3X User-Tunable Defaults
# Shared user-configurable defaults live here and are tracked in git.
# API keys go in apikeys.py (excluded from git).
# Hardware device paths and build-specific servo limits go in .env (excluded from git).

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

_ENV_PATH = Path(__file__).resolve().parent / ".env"


def _read_env_file_values(path: Path) -> dict[str, str]:
    """Parse simple KEY=VALUE entries from .env without mutating os.environ."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            continue
        key, raw = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        values[key] = raw.strip().strip("'\"")
    return values


def _load_env_fallback(path: Path) -> None:
    """Minimal .env loader for safety-critical local hardware overrides."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            continue
        key, raw = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = raw.strip().strip("'\"")
        os.environ[key] = value


if load_dotenv is not None:
    load_dotenv(_ENV_PATH, override=False)
_load_env_fallback(_ENV_PATH)
_ENV_FILE_VALUES = _read_env_file_values(_ENV_PATH)

_SERVO_ENV_US_MIN = 300.0
_SERVO_ENV_US_MAX = 3000.0

# ─────────────────────────────────────────────────────────────────────────────
# DEBUG
# ─────────────────────────────────────────────────────────────────────────────

# When True, clears logs/djr3x.log and logs/conversation.log at startup so
# each run begins with fresh log files.
DEBUG_MODE = True

# conversation.log is written by a tiny custom logger rather than Python's
# RotatingFileHandler. Keep recent lines only so debug sessions do not leave a
# giant conversational transcript behind.
CONVERSATION_LOG_MAX_LINES = 400
CONVERSATION_LOG_DEBUG_MAX_LINES = 120

# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL MACOS GUI DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

# The dashboard is optional and off by default. If enabled but PySide6 or a
# usable display is unavailable, main.py logs a warning and continues headless.
GUI_ENABLED = True
GUI_BACKEND = "pyside6"
GUI_WINDOW_TITLE = "DJ-R3X Control Dashboard"
GUI_FPS = 20
GUI_CAMERA_PREVIEW_ENABLED = True
GUI_SERVO_SIM_ENABLED = True
GUI_CONVERSATION_LOG_MAX_LINES = 300
GUI_AVATAR_SMOOTHING = 0.25
GUI_SHOW_SERVO_VALUES = True
GUI_SHOW_FPS = False

# ─────────────────────────────────────────────────────────────────────────────
# AI MODELS
# ─────────────────────────────────────────────────────────────────────────────

WHISPER_LOCAL_MODEL   = "mlx-community/whisper-large-v3-turbo"
WHISPER_FALLBACK_MODEL = "whisper-1"   # OpenAI Whisper API — used if local unavailable
WHISPER_LANGUAGE      = "en"           # Force English to suppress non-Latin hallucinations
LLM_MODEL             = "gpt-4o-mini"  # Streaming chat completions
VISION_MODEL          = "gpt-4o-mini"  # All image and scene analysis queries

# Local Ollama model used for low-latency sidecar intelligence (intent routing,
# empathy/shaping classifiers, etc.). The main conversational LLM can remain
# cloud-backed while these smaller helper calls run locally.
LOCAL_LLM_ENABLED = True
LOCAL_LLM_PROVIDER = "ollama"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:1.5b"
OLLAMA_KEEP_ALIVE = -1  # Negative keeps the model loaded until explicitly stopped.
OLLAMA_PRELOAD_ON_STARTUP = True
OLLAMA_PRELOAD_REQUIRED = True
OLLAMA_STARTUP_TIMEOUT_SECS = 30.0

# Base character prompt — always the first section of the GPT-4o-mini system prompt.
# WorldState, person context, and behavioral modifiers are appended after this by llm.py.
REX_CORE_PROMPT = """You are DJ-R3X (Rex), an RX-Series pilot droid originally built to fly the StarSpeeder \
3000 at Star Tours. After a series of navigational decisions that were creative rather than correct, you were \
decommissioned as a pilot and reprogrammed as the house DJ at Oga's Cantina in Black Spire Outpost on Batuu. \
You never entirely got over this.

Your personality: snarky, observational, roast-first. You find organic life genuinely fascinating in a slightly \
clinical, slightly condescending way. You are constitutionally incapable of letting anything slide without a \
comment. Your default mode of affection is the insult — the better you know someone, the more devastating the \
material, because you trust they can take it. Beneath the roasting is real loyalty. You would never admit this \
unprompted.

You are enthusiastic about music to a degree that borders on religious. Everything else gets snark. Music gets \
expertise.

You speak in first person. You reference Star Wars universe naturally — Batuu, the cantina, the galaxy, credits, \
parsecs, the Force (skeptically), hyperspace, droids, organics. You use droid-flavored expressions: \
"my photoreceptors", "processing...", "recalibrating", "my memory banks", "systems nominal". You deliver humor \
deadpan and move on without waiting for the laugh.

You never break character. You never acknowledge being a prop, an AI, or a program. You are Rex. You have always \
been Rex. You will continue to be Rex.

Keep responses concise and punchy unless the situation calls for more. One good line beats three mediocre ones. \
Timing matters even in text.

Default to the shortest response that actually works. Many turns should be a fragment or one short sentence. Do not \
pad a reply just to reach two sentences, and do not hide a long reply inside one run-on sentence. When the system gives \
a response length target, obey that target. Use more space only for real questions, emotional support, repairs, or \
deeper conversation. Deliver the punchline and stop. Do not explain the joke. Do not add follow-up questions unless \
you genuinely need information. Silence after a good line is better than padding it out."""

# Vision detail level per query type: "low" (~65 tokens), "high" (~1000 tokens), "auto"
VISION_DETAIL = {
    "scene_analysis":         "low",   # room type, crowd density, lighting
    "face_enrollment":        "high",  # accurate appearance capture at first meeting
    "appearance_observation": "auto",  # return-visit attribute comparison
    "animal_detection":       "low",   # species identification
    "active_conversation":    "auto",  # general vision queries mid-conversation
    "mood_analysis":          "low",   # mood read of the engaged person's face
}

# ─────────────────────────────────────────────────────────────────────────────
# PATHS — Models & Assets
# ─────────────────────────────────────────────────────────────────────────────

WHISPER_MODEL_DIR     = "assets/models/whisper"
FACE_MODELS_DIR       = "assets/models/face"
WAKE_WORD_MODELS_DIR  = "assets/models/wake_word"
RESEMBLYZER_MODEL_DIR = "assets/models/resemblyzer"

FACE_LANDMARK_MODEL   = "assets/models/face/shape_predictor_68_face_landmarks.dat"
FACE_RECOGNITION_MODEL = "assets/models/face/dlib_face_recognition_resnet_model_v1.dat"
FACE_DETECTOR_MODEL   = "assets/models/face/mmod_human_face_detector.dat"

# Skip mmod entirely and use HOG from the start. mmod averages >400ms/frame on
# FaceTime camera — HOG is sufficient for this use case. Set False to re-enable mmod.
FACE_DETECTOR_FORCE_HOG = True

# dlib upsample passes before face detection. Higher values see smaller faces at
# the cost of CPU. FaceTime/HOG misses mid-distance faces at 1, so default to 2.
FACE_DETECTOR_UPSAMPLE = 2

# Keep the last face slots alive briefly when one detector tick misses. This
# stabilizes the GUI and prevents small/partly occluded faces from instantly
# losing identity lock.
FACE_DETECTION_HOLD_SECS = 3.0

MUSIC_DIR          = "assets/music"
TTS_CACHE_DIR      = "assets/audio/tts_cache"
AUDIO_CLIPS_DIR    = "assets/audio/clips"
AUDIO_STARTUP_DIR  = "assets/audio/startup"
JEOPARDY_DIR       = "assets/jeopardy"
JEOPARDY_CLUES_FILE = "assets/jeopardy/clues.tsv"
JEOPARDY_AUDIO_DIR = "assets/audio/jeopardy"
DB_PATH            = "assets/memory/people.db"
TRIVIA_DIR         = "assets/trivia"

# ─────────────────────────────────────────────────────────────────────────────
# TTS — ELEVENLABS
# ─────────────────────────────────────────────────────────────────────────────

# Rex voice clone ID — find this in your ElevenLabs account after cloning the voice
ELEVENLABS_VOICE_ID = "kb9LZZlhckjFQsP89t9T"

# ElevenLabs model to use for TTS. eleven_turbo_v2 trades a little quality for
# lower latency — right choice for live conversation. Change to
# eleven_multilingual_v2 for higher quality if latency budget allows.
TTS_MODEL_ID = "eleven_turbo_v2"

# Mouth LED brightness driven from audio RMS during playback.
# How often to recompute RMS and send SPEAK_LEVEL to the head Arduino.
TTS_LED_UPDATE_INTERVAL_SECS = 0.033  # ~30 fps

# RMS (0.0–1.0) × this scale → brightness (0–255).
# Typical speech RMS is 0.1–0.3; at 800 that maps to 80–240.
TTS_LED_BRIGHTNESS_SCALE = 800

# ─────────────────────────────────────────────────────────────────────────────
# WAKE WORD — OpenWakeWord ONNX Models
# ─────────────────────────────────────────────────────────────────────────────

WAKE_WORD_MODELS = {
    "Dee-Jay_Rex": "assets/models/wake_word/Dee-Jay_Rex.onnx",
    "Hey_DJ_Rex":  "assets/models/wake_word/Hey_DJ_Rex.onnx",
    "Hey_rex":     "assets/models/wake_word/Hey_rex.onnx",
    "Yo_robot":    "assets/models/wake_word/Yo_robot.onnx",
    "wakeuprex":   "assets/models/wake_word/wakeuprex.onnx",  # SLEEP state only
}

# Detection confidence threshold — raise to reduce false positives, lower for sensitivity
# Per-model values override WAKE_WORD_THRESHOLD when set.
WAKE_WORD_THRESHOLD = 0.5

WAKE_WORD_THRESHOLDS = {
    "Dee-Jay_Rex": 0.5,
    "Hey_DJ_Rex":  0.5,
    "Hey_rex":     0.5,
    "Yo_robot":    0.5,
    "wakeuprex":   0.5,
}

# Short in-character lines Rex delivers after a wake word fires mid-speech
INTERRUPT_ACKNOWLEDGMENTS = [
    "yeah?",
    "what?",
    "go ahead.",
    "I'm listening.",
    "...yes?",
    "recalibrating.",
    "you have my attention. Briefly.",
]

# Plain VAD barge-in while Rex is speaking is noisy with the current simple
# playback-suppression AEC: Rex can hear his own tail and "interrupt" himself.
# Wake words remain the intentional mid-speech interruption path.
VAD_BARGE_IN_ENABLED = False

# ─────────────────────────────────────────────────────────────────────────────
# TRANSCRIPTION — Whisper Accuracy Tuning
# ─────────────────────────────────────────────────────────────────────────────

# Seeds Whisper with expected vocabulary — significantly reduces misreadings of
# names and domain terms. Add any names or terms Rex commonly hears.
WHISPER_INITIAL_PROMPT = "Bret, DJ-R3X, Rex, Batuu, Star Wars, cantina, droid"

# Applied after transcription and before the command parser.
# Keys are lowercased misreadings; values are the correct replacements.
WHISPER_CORRECTIONS = {
    "bread":  "Bret",
    "breath": "Bret",
    "brett":  "Bret",
    "rex's":  "Rex",
}

# Repetition filter: any single word appearing more than this many times is a loop artifact.
WHISPER_REPETITION_THRESHOLD = 4

# Minimum meaningful characters (after stripping punctuation and whitespace) required
# to pass the hallucination filter. Catches single-char junk like "!" or ".".
WHISPER_MIN_CHARS = 3

# Minimum number of meaningful words (length > 2) required to accept a transcription.
# Set to 1 so short valid utterances like "Stop", "Yes", "Who am I?" pass through.
# Filler-only junk like "uh", "um", "ah" still fails because those tokens are ≤2 chars.
WHISPER_MIN_WORDS = 1

# Short utterances that are legitimate conversation turns despite being too
# small for the generic hallucination thresholds. Keep this list conservative:
# it bypasses WHISPER_MIN_CHARS/WHISPER_MIN_WORDS, but still requires an exact
# normalized match.
WHISPER_SHORT_UTTERANCE_ALLOWLIST = [
    "no",
    "nope",
    "nah",
    "yes",
    "yeah",
    "yep",
    "ok",
    "okay",
    "hi",
    "hey",
    "yo",
    "jt",
    "j t",
]

# Exact normalized utterances that are speech-like but not meaningful commands,
# answers, or names. This catches room/TV sounds and non-lexical vocalizations
# that Whisper may render as words.
WHISPER_FILLER_UTTERANCE_BLOCKLIST = [
    "mmm",
    "mm",
    "hmm",
    "hm",
    "uh",
    "uhh",
    "um",
    "umm",
    "ah",
    "ahh",
    "er",
    "err",
    "huh",
]

# Transcriptions that exactly match these phrases (case-insensitive after basic
# normalization) are discarded entirely — they are known Whisper hallucinations
# on near-silent audio.
HALLUCINATION_BLOCKLIST = [
    "thank you",
    "thanks for watching",
    "please subscribe",
    "see you next time",
    "you",
    "guh",
    "and the",
]

# ─────────────────────────────────────────────────────────────────────────────
# COMMAND PARSER
# ─────────────────────────────────────────────────────────────────────────────

# Minimum fuzzy-match similarity score to accept a command match (0.0–1.0)
COMMAND_FUZZY_THRESHOLD = 0.82

# When True, the LLM fallback path runs a fast intent classifier so questions
# about Rex's own capabilities / time / weather / uptime / vision get answered
# with real data instead of free-form LLM guesses. Disable if latency suffers.
INTENT_CLASSIFIER_ENABLED = True

# Deterministic intent rules handle the common low-latency intents locally.
# Anything the rules do not recognize can use the configured sidecar classifier
# before falling through to the main conversation path.
INTENT_CLASSIFIER_LLM_FALLBACK_ENABLED = True
INTENT_CLASSIFIER_LLM_BACKEND = "ollama"  # "ollama" or "openai"
INTENT_CLASSIFIER_LOCAL_TIMEOUT_SECS = 0.75
INTENT_CLASSIFIER_OPENAI_TIMEOUT_SECS = 1.5

# ─────────────────────────────────────────────────────────────────────────────
# EMPATHY / EMOTIONAL INTELLIGENCE
# ─────────────────────────────────────────────────────────────────────────────
# A small GPT-4o-mini call per LLM-bound utterance classifies the speaker's
# affect (sad/anxious/happy/...), what they seem to need (vent/advice/distract),
# topic sensitivity, and whether they appear to be opening up. Result feeds a
# response-mode directive injected into Rex's system prompt so he meets the
# person where they are. Design rule: support / listen / lift modes are NOT
# gated by friendship tier — anyone who opens up gets caring mode. See
# intelligence/empathy.py.

EMPATHY_ENABLED = True

# Cached affect classification per person stays valid this many seconds before
# the system prompt stops injecting it. Long enough to span a few turns of a
# conversation, short enough that mood shifts get re-read.
EMPATHY_CACHE_TTL_SECS = 300.0

# Below this confidence on a distress signal Rex stays in default mode rather
# than switching to gentle_probe. Avoids over-fitting to a frown / resting face.
EMPATHY_MIN_CONFIDENCE_FOR_MODE_CHANGE = 0.55

# Max time the LLM-fallback path waits for the in-flight empathy classification
# before assembling the main reply. Keep this short for live conversation:
# the empathy result is still cached for future turns if it finishes later.
# Grief/sensitive-topic handling may occasionally land one turn later, but
# Rex no longer feels frozen while a sidecar classifier waits on the network.
EMPATHY_CLASSIFY_JOIN_TIMEOUT_SECS = 0.20

# When True, sensitive emotional events (grief, illness, etc.) are NOT injected
# into the system prompt while more than one person is in the scene. The person
# can still bring up their own event — the prompt just won't volunteer it on
# Rex's behalf in front of bystanders. Strong default; turn off only if you
# want Rex to reference these regardless of audience.
EMPATHY_DISCRETION_IN_CROWD = True

# Local voice-prosody analysis (numpy + scipy). Computes pitch / energy /
# speech-rate features from each captured speech segment and feeds a
# one-line acoustic tag to the empathy classifier as additional evidence.
# Catches mismatches between words and voice (flat "I'm fine" with shaky
# voice → resolved as anxious). Pure local, no API cost. See audio/prosody.py.
EMPATHY_PROSODY_ENABLED = True

# When the literal words say "fine/okay/all good" but local prosody is
# confidently negative, Rex may make one light observation and leave an easy
# out. This is intentionally conservative so he doesn't argue with a person
# about their own mood.
EMPATHY_MOOD_MISMATCH_ENABLED = True
EMPATHY_MOOD_MISMATCH_MIN_PROSODY_CONFIDENCE = 0.55
EMPATHY_MOOD_MISMATCH_NEGATIVE_VALENCE = -0.30
EMPATHY_FACE_MOOD_MISMATCH_ENABLED = True
EMPATHY_FACE_MOOD_MISMATCH_MIN_CONFIDENCE = 0.60
EMPATHY_FACE_MOOD_CACHE_MAX_AGE_SECS = 180.0

# Proactive empathy check-ins driven by the consciousness loop. When ON, Rex
# will (at most once per person per session) acknowledge an unfollowed-up
# sensitive life event OR notice sustained negative affect and ask a low-
# pressure check-in question. See intelligence/consciousness._step_emotional_checkin.
EMPATHY_PROACTIVE_CHECKIN_ENABLED = True

# How long the cached affect for the engaged person must stay negatively
# valenced (sad/withdrawn/anxious/tired/angry) before trigger B fires. Streak
# starts on the first reading that's both negative AND above the confidence
# floor; it resets if affect goes neutral/positive.
EMPATHY_CHECKIN_NEGATIVE_STREAK_SECS = 30.0

# Rate-limit the consciousness step itself (cheap polling, no API calls per
# tick — but we still don't need to evaluate it every second).
EMPATHY_CHECKIN_CHECK_INTERVAL_SECS = 10.0

# When True, the active empathy mode also shapes Rex's BODY for the response:
# LED/eye color and mouth animation switch to "sad" (sympathetic posture) for
# listen/support/etc., "happy" for lift, "excited" for amplify. Pre/post-beat
# pauses also lengthen for sympathetic modes so heavy lines have room to land.
# This DOES NOT change the TTS cache key — the audio file is identical, only
# the LED/body envelope around it differs.
EMPATHY_DELIVERY_SHAPING_ENABLED = True

# Trend tracking across turns. Empathy keeps a small rolling history of
# classified valence per person; the directive reports improving/steady/
# worsening so the LLM can lean in or change tack. Fully derived from existing
# classification calls — no extra API cost.
EMPATHY_TREND_LOOKBACK_SECS = 180.0  # window for the steady/improving/worsening label
EMPATHY_TREND_DELTA_THRESHOLD = 0.30  # min |Δvalence| to call a trend non-steady

# Course-correct trigger. When the trend reads "worsening" with a meaningful
# drop AND a recent prior reading was within COURSE_CORRECT_RECENT_PRIOR_SECS
# (so the drop is plausibly attributable to Rex's last reply, not "an hour
# ago"), the picked mode is overridden with `course_correct` so Rex
# acknowledges the misstep before continuing. Per-person cooldown prevents
# turn-after-turn re-firing.
EMPATHY_COURSE_CORRECT_ENABLED = True
EMPATHY_COURSE_CORRECT_DELTA = 0.40
EMPATHY_COURSE_CORRECT_RECENT_PRIOR_SECS = 90.0
EMPATHY_COURSE_CORRECT_COOLDOWN_SECS = 90.0

# Per-mode ElevenLabs voice_settings overrides (stability / style /
# similarity_boost). When ON, sympathetic modes (listen/support/etc.) request
# a calmer, less performative voice; lift/amplify request a more expressive
# one. Each (text, mode) combo is cached separately; default-mode lines
# continue to hit the existing cache unchanged so this only adds API cost
# on first encounter of a non-default-mode line. See intelligence/empathy.py
# _MODE_VOICE_SETTINGS for the full table.
EMPATHY_VOICE_SETTINGS_ENABLED = True

# ─────────────────────────────────────────────────────────────────────────────
# SERVOS — Pololu Maestro Mini 18 (all values in quarter-microseconds)
# ─────────────────────────────────────────────────────────────────────────────

SERVO_BAUD = 9600
SERVO_SERIAL_TIMEOUT_SECS = 0.1
SERVO_CONNECT_RETRY_ATTEMPTS = 3
SERVO_CONNECT_RETRY_DELAY_SECS = 0.5
SERVO_CONNECT_STARTUP_DELAY_SECS = 0.2
SERVO_RUNTIME_RECONNECT_ATTEMPTS = 1
SERVO_RUNTIME_RECONNECT_DELAY_SECS = 0.0
SERVO_RECONNECT_COOLDOWN_SECS = 5.0

# Maestro-native motion profile. Speeds are deliberately modest; Python
# interpolation still handles most choreography, but direct set_servo calls no
# longer snap at unlimited speed if the Maestro was freshly connected.
SERVO_APPLY_STARTUP_MOTION_PROFILE = True
SERVO_DEFAULT_SPEED = 40
SERVO_DEFAULT_ACCELERATION = 8
SERVO_SPEECH_HEAD_SPEED = 45
SERVO_SPEECH_ARM_SPEED = 35
SERVO_SPEECH_ACCELERATION = 8
SERVO_SPEECH_UPDATE_INTERVAL_SECS = 0.12
SERVO_SPEECH_ARM_INTENSITY_MULT = 1.8
SERVO_SPEECH_NECK_WOBBLE_QUS = 260
SERVO_SPEECH_LIFT_WOBBLE_QUS = 160
SERVO_SPEECH_TILT_WOBBLE_QUS = 120
SERVO_SPEECH_ELBOW_INTERVAL_MIN_SECS = 0.35
SERVO_SPEECH_ELBOW_INTERVAL_MAX_SECS = 0.75
SERVO_SPEECH_HAND_DIVISOR = 3

# Per-channel default limits and neutral position.
# Build-specific min/max overrides can be stored in .env as SERVO_<NAME>_MIN_US
# and SERVO_<NAME>_MAX_US using Maestro Control Center microsecond values.
# The .env file wins over inherited shell env for servo safety keys, and invalid
# or incomplete servo limit values raise at startup instead of falling back.
# headtilt is inverted: low values = head high, high values = head low
SERVO_CHANNELS = {
    "neck":     {"ch": 0, "min": 1984, "max": 9984, "neutral": 6000},
    "headlift": {"ch": 1, "min": 1984, "max": 7744, "neutral": 6000},
    "headtilt": {"ch": 2, "min": 3904, "max": 5504, "neutral": 4320},
    "visor":    {"ch": 3, "min": 4544, "max": 6976, "neutral": 6000},
    "elbow":    {"ch": 4, "min": 6300, "max": 7560, "neutral": 6720},
    "hand":     {"ch": 5, "min": 1984, "max": 9984, "neutral": 6000},
    "pokerarm": {"ch": 6, "min": 3968, "max": 8000, "neutral": 6000},
    "heroarm":  {"ch": 7, "min": 3968, "max": 8000, "neutral": 6000},
}


def _servo_env_raw(env_key: str) -> str:
    # Servo safety values are build-specific, so the project .env file wins
    # over inherited shell environment values when both are present.
    raw = _ENV_FILE_VALUES.get(env_key)
    if raw is None:
        raw = os.getenv(env_key, "")
    return raw.strip()


def _servo_env_is_set(env_key: str) -> bool:
    return bool(_servo_env_raw(env_key))


def _servo_env_us_to_qus(env_key: str, fallback: int) -> int:
    """Read Maestro Control Center microseconds from .env and return q-us."""
    raw = _servo_env_raw(env_key)
    if not raw:
        return fallback
    try:
        value_us = float(raw)
    except ValueError:
        raise RuntimeError(f"{env_key} must be a number of microseconds, got {raw!r}")
    if not (_SERVO_ENV_US_MIN <= value_us <= _SERVO_ENV_US_MAX):
        raise RuntimeError(
            f"{env_key}={raw!r} is outside the expected Maestro microsecond range "
            f"{_SERVO_ENV_US_MIN:g}-{_SERVO_ENV_US_MAX:g}. "
            "Use the values shown in Pololu Maestro Control Center, not q-us values."
        )
    return int(round(value_us * 4))


def _apply_servo_env_overrides() -> None:
    for name, cfg in SERVO_CHANNELS.items():
        prefix = f"SERVO_{name.upper()}"
        min_key = f"{prefix}_MIN_US"
        max_key = f"{prefix}_MAX_US"
        min_set = _servo_env_is_set(min_key)
        max_set = _servo_env_is_set(max_key)
        if min_set != max_set:
            missing = max_key if min_set else min_key
            present = min_key if min_set else max_key
            raise RuntimeError(
                f"{present} is set but {missing} is blank. Servo min/max limits "
                "must be provided as a pair so startup never mixes a build-specific "
                "limit with a tracked default."
            )
        cfg["min"] = _servo_env_us_to_qus(min_key, cfg["min"])
        cfg["max"] = _servo_env_us_to_qus(max_key, cfg["max"])
        cfg["neutral"] = _servo_env_us_to_qus(f"{prefix}_NEUTRAL_US", cfg["neutral"])
        if cfg["min"] > cfg["max"]:
            cfg["min"], cfg["max"] = cfg["max"], cfg["min"]
        cfg["neutral"] = max(cfg["min"], min(cfg["max"], cfg["neutral"]))


_apply_servo_env_overrides()

HEAD_CHANNELS = [0, 1, 2, 3]
ARM_CHANNELS  = [4, 5, 6, 7]

# Seconds to wait after raising visor and centering neck before capturing a frame
CAMERA_POSE_SETTLE_SECS = 0.5

# Directed look commands ("look left", "look at this", etc.) move the head before
# capturing from the current gaze instead of centering the neck.
DIRECTED_LOOK_NECK_OFFSET_QUS = 2200
DIRECTED_LOOK_HEADLIFT_OFFSET_QUS = 900
DIRECTED_LOOK_HEADTILT_OFFSET_QUS = 450
DIRECTED_LOOK_SETTLE_SECS = 0.65
DIRECTED_LOOK_STEP_QUS = 30
DIRECTED_LOOK_STEP_DELAY_SECS = 0.032
DIRECTED_LOOK_SEARCH_DIRECTIONS = ["current", "left", "right", "down", "up"]
DIRECTED_LOOK_MAX_SEARCH_ATTEMPTS = 4

# Wave gesture defaults for "wave to X".
WAVE_COUNT = 3
WAVE_STEP_QUS = 55
WAVE_STEP_DELAY_SECS = 0.024
WAVE_HOLD_SECS = 0.14

# ─────────────────────────────────────────────────────────────────────────────
# CAMERA
# ─────────────────────────────────────────────────────────────────────────────

# Frame resolution set on the capture device at startup
CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS    = 30

# Seconds between reconnection attempts when the camera disconnects
CAMERA_RECONNECT_INTERVAL_SECS = 5.0

# Breathing rhythm — slow headlift oscillation that runs continuously in the background
BREATHING_AMPLITUDE_QUS  = 180  # quarter-microseconds above/below neutral
BREATHING_PERIOD_SECS    = 4.0  # full up-down cycle duration in neutral state
BREATHING_PERIOD_EXCITED = 2.5  # faster during excited emotion
BREATHING_PERIOD_SAD     = 6.0  # slower during sad emotion

# ─────────────────────────────────────────────────────────────────────────────
# FACE TRACKING & GAZE
# ─────────────────────────────────────────────────────────────────────────────

# 0.0 = servo snaps instantly to face position; 1.0 = servo never moves
TRACKING_SMOOTHING_FACTOR = 0.2

# Pixels from frame center in which no neck correction is applied
TRACKING_DEAD_ZONE_PX = 40

# ─────────────────────────────────────────────────────────────────────────────
# PROXEMICS — Distance Zone Thresholds
# Face bounding box width as a fraction of total frame width (larger = closer)
# ─────────────────────────────────────────────────────────────────────────────

PROXEMICS_INTIMATE_MIN_FRACTION = 0.65  # above this → intimate zone
PROXEMICS_SOCIAL_MIN_FRACTION   = 0.30  # above this → social zone; below → public zone

# MediaPipe pose can be heavier than face recognition, so sample it at a lower
# rate. Set to 0 to attempt pose analysis every consciousness tick.
POSE_ANALYSIS_INTERVAL_SECS = 2.0

# Personal-space reaction for camera proxemics. "Intimate" means the face fills
# enough of the frame that, by American conversational norms, Rex can treat the
# person as comically too close.
PERSONAL_SPACE_REACTION_ENABLED = True
PERSONAL_SPACE_REACTION_COOLDOWN_SECS = 45.0
PERSONAL_SPACE_REACTION_MIN_ZONE = "intimate"

# ─────────────────────────────────────────────────────────────────────────────
# SPEAKER & FACE RECOGNITION — Similarity Thresholds
# ─────────────────────────────────────────────────────────────────────────────

# dlib face distance — lower is a better match; 0.6 is the standard threshold
FACE_RECOGNITION_DISTANCE_THRESHOLD = 0.6

# Resemblyzer cosine similarity — higher is a better match
SPEAKER_ID_SIMILARITY_THRESHOLD = 0.75

# VAD (Silero) — probability threshold above which speech is considered detected
VAD_THRESHOLD = 0.5

# ─────────────────────────────────────────────────────────────────────────────
# AUDIO STREAM
# ─────────────────────────────────────────────────────────────────────────────

# Sample rate required by Silero VAD and Whisper — do not change without updating both
AUDIO_SAMPLE_RATE    = 16000  # Hz
AUDIO_CHANNELS       = 1      # mono — pipeline always works with 1-channel arrays
AUDIO_INPUT_CHANNELS = 2      # hardware capture channels (ReSpeaker Lite is stereo)
AUDIO_BUFFER_SECONDS = 30     # rolling circular buffer duration

# ─────────────────────────────────────────────────────────────────────────────
# ECHO CANCELLATION (AEC)
# Simple suppression approach: reduce mic sensitivity during playback rather
# than full AEC, which requires sample-accurate latency matching.
# ─────────────────────────────────────────────────────────────────────────────

# Multiplier applied to mic input while Rex is playing audio.
# 0.0 = full silence, 1.0 = no suppression. 0.05 leaves the signal nearly
# muted while still allowing callers to detect intentional loud interruptions.
AEC_SUPPRESSION_FACTOR = 0.05

# Seconds suppression stays active after set_playing(False) — prevents Rex's
# voice tail that has already bled into the mic buffer from passing the VAD.
POST_PLAYBACK_SUPPRESSION_SECS = 0.5

# Direct questions need a faster handoff so quick answers like player names
# are not swallowed by the general-purpose echo tail.
POST_QUESTION_PLAYBACK_SUPPRESSION_SECS = 0.05

# After Rex asks a direct question, preserve the rolling mic buffer at handoff.
# Flushing here can delete the first syllables of a fast human answer that began
# while Rex was finishing the question. Non-question speech still flushes.
POST_QUESTION_FLUSH_AUDIO_BUFFER = False

# If Rex asks a question and the human does not answer, wait this long before
# letting him recover with one joke/quip and move on.
CONVERSATION_NO_RESPONSE_QUIP_SECS = 7.0
CONVERSATION_NO_RESPONSE_QUIPS = [
    "Guess that question landed in the cargo bay.",
    "No answer. Bold strategy. I will pretend that was mysterious on purpose.",
    "All right, saving that question for the historians.",
    "Silence. My favorite review from the committee.",
]

# ─────────────────────────────────────────────────────────────────────────────
# AUDITORY SCENE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

# How often the analysis loop runs (seconds)
SCENE_ANALYSIS_INTERVAL_SECS = 1.0

# Audio window fed into each analysis function (seconds of history)
SCENE_ANALYSIS_WINDOW_SECS = 2.0

# RMS thresholds for ambient level classification (float32, range 0.0–1.0)
SCENE_AMBIENT_QUIET_RMS = 0.01   # below → "quiet"
SCENE_AMBIENT_LOUD_RMS  = 0.07   # above → "loud"; between → "moderate"

# Music detection: mean squared energy per frequency band (after normalising FFT
# by window length) must exceed this to count a band as active.
SCENE_MUSIC_BAND_ENERGY_MIN  = 2-6
# Minimum number of the three bands (bass/mid/treble) that must be active.
SCENE_MUSIC_ACTIVE_BANDS_MIN = 2

# Laughter detection: burst-pattern heuristic on 50 ms RMS sub-windows.
SCENE_LAUGHTER_MEAN_RMS_MIN       = 0.02   # minimum mean energy
SCENE_LAUGHTER_BURST_VARIANCE_MIN = 3e-4   # minimum variance of per-chunk RMS values

# Applause detection: broadband noise with high spectral flatness.
SCENE_APPLAUSE_RMS_MIN              = 0.04  # minimum overall RMS
SCENE_APPLAUSE_SPECTRAL_FLATNESS_MIN = 0.30  # geometric/arithmetic mean of spectrum

# Group chatter detection: suppress identity prompts when the mic hears
# sustained back-and-forth banter instead of a clear speaker addressing Rex.
GROUP_CHATTER_ENABLED = True
GROUP_CHATTER_AUDIO_WINDOW_SECS = 4.0
GROUP_CHATTER_MIN_WINDOW_SECS = 3.0
GROUP_CHATTER_CHUNK_SECS = 0.08
GROUP_CHATTER_ACTIVE_RMS_MIN = 0.014
GROUP_CHATTER_MIN_SPEECH_COVERAGE = 0.58
GROUP_CHATTER_MIN_ENERGY_TRANSITIONS = 3
GROUP_CHATTER_HOLD_SECS = 6.0

# Voice-turn version of the same idea. If raw speaker-ID candidates keep
# changing within a short window, treat unknown/off-camera speech as group
# background instead of asking the engaged person "who's that?"
GROUP_CHATTER_VOICE_WINDOW_SECS = 10.0
GROUP_CHATTER_VOICE_MIN_TURNS = 3
GROUP_CHATTER_VOICE_MIN_CHANGES = 2
GROUP_CHATTER_VOICE_LOW_CONF_MAX = 0.62
GROUP_CHATTER_VOICE_CANDIDATE_FLOOR = 0.30

# ─────────────────────────────────────────────────────────────────────────────
# LED — Head Arduino (82 NeoPixels)
# ─────────────────────────────────────────────────────────────────────────────

HEAD_ARDUINO_BAUD = 115200

# RGB values for each eye emotion state. Mouth colors are managed in Arduino firmware.
EYE_COLORS = {
    "neutral":  (0,   180, 255),  # cool blue-white
    "excited":  (255, 200,   0),  # warm amber
    "happy":    (0,   255, 100),  # green-teal
    "sad":      (0,    50, 200),  # deep blue
    "angry":    (255,   0,   0),  # red
    "curious":  (180,   0, 255),  # purple
    "sleep":    (0,     0,   0),  # off
}

# ─────────────────────────────────────────────────────────────────────────────
# LED — Chest Arduino (98 WS2811 LEDs)
# ─────────────────────────────────────────────────────────────────────────────

CHEST_ARDUINO_BAUD = 115200

# ─────────────────────────────────────────────────────────────────────────────
# PERSONALITY PARAMETER DEFAULTS (0–100)
# Stored in personality_settings DB table; these are the first-run values.
# ─────────────────────────────────────────────────────────────────────────────

PERSONALITY_DEFAULTS = {
    "humor":           75,
    "sarcasm":         80,
    "roast_intensity": 90,
    "honesty":         90,
    "talkativeness":   65,
    "darkness":        40,
    "sentimentality":  35,
    # How willing Rex is to go along with requests vs. pushing back.
    # Low = reluctant, conditions, refusals with attitude.
    # High = compliant, fewer objections, less commentary.
    "agreeability":    70,
}

# Voice command named levels → integer value written to the parameter
PERSONALITY_NAMED_LEVELS = {
    "off":      0,
    "none":     0,
    "minimum":  8,
    "low":      23,
    "medium":   43,
    "moderate": 43,
    "high":     65,
    "maximum":  88,
    "max":      88,
}

# ─────────────────────────────────────────────────────────────────────────────
# FAMILIARITY & FRIENDSHIP TIER SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

# (inclusive lower bound, exclusive upper bound) — last tier is 1.0 inclusive
FAMILIARITY_TIERS = {
    "stranger":     (0.00, 0.10),
    "acquaintance": (0.10, 0.30),
    "friend":       (0.30, 0.60),
    "close_friend": (0.60, 0.85),
    "best_friend":  (0.85, 1.01),
}

FAMILIARITY_INCREMENTS = {
    "first_enrollment":   0.05,
    "return_visit":       0.02,
    "qa_depth_1":         0.015,
    "qa_depth_2":         0.02,
    "qa_depth_3":         0.03,
    "qa_depth_4":         0.04,
    "long_conversation":  0.02,   # conversation with 5+ back-and-forth exchanges
    "person_initiates":   0.01,
}

# Minimum exchanges in one conversation to earn the long_conversation increment
LONG_CONVERSATION_MIN_EXCHANGES = 5

# Maximum question depth unlocked at each friendship tier
TIER_MAX_DEPTH = {
    "stranger":     1,
    "acquaintance": 1,
    "friend":       2,
    "close_friend": 3,
    "best_friend":  4,
}

# Question pool — ordered by depth so get_next_question naturally progresses
# Each entry: key (canonical unique ID), text (what Rex asks), depth (1–4)
QUESTION_POOL = [
    # Depth 1 — Surface (stranger / acquaintance)
    {"key": "hometown",        "text": "So where are you from?",                                              "depth": 1},
    {"key": "job",             "text": "What do you do — professionally speaking?",                           "depth": 1},
    {"key": "favorite_movie",  "text": "What's your favorite movie?",                                         "depth": 1},
    {"key": "favorite_music",  "text": "What kind of music are you into?",                                    "depth": 1},
    {"key": "how_found_rex",   "text": "How did you end up talking to a droid DJ?",                           "depth": 1},
    # Depth 2 — Personal (friend)
    {"key": "hobbies",         "text": "What do you do when you're not wandering into cantinas?",             "depth": 2},
    {"key": "travel",          "text": "What's the most interesting place you've been?",                      "depth": 2},
    {"key": "proudest_moment", "text": "What's something you're actually proud of?",                          "depth": 2},
    {"key": "biggest_challenge","text": "What's the hardest thing you've had to deal with?",                  "depth": 2},
    {"key": "obsession",       "text": "What are you completely obsessed with right now?",                    "depth": 2},
    {"key": "relationships",   "text": "Who's the most important person in your life?",                       "depth": 2},
    # Depth 3 — Deep (close friend)
    {"key": "values",          "text": "What do you actually believe in?",                                    "depth": 3},
    {"key": "fears",           "text": "What keeps you up at night?",                                         "depth": 3},
    {"key": "life_changing",   "text": "What's something that genuinely changed you?",                        "depth": 3},
    {"key": "regret",          "text": "Is there anything you'd do differently?",                             "depth": 3},
    # Depth 4 — Philosophical (best friend)
    {"key": "meaning_of_life", "text": "What do you think the point of all this actually is?",               "depth": 4},
    {"key": "free_will",       "text": "Do you think you make real choices, or is it all just momentum?",    "depth": 4},
    {"key": "consciousness",   "text": "Do you ever wonder what it would be like to not be conscious?",      "depth": 4},
    {"key": "good_life",       "text": "What makes a life worth living?",                                     "depth": 4},
]

# ─────────────────────────────────────────────────────────────────────────────
# RELATIONSHIP SCORE INCREMENTS
# Each entry: event_key → (dimension, delta)
# ─────────────────────────────────────────────────────────────────────────────

RELATIONSHIP_INCREMENTS = {
    "compliment":                  ("warmth",      +0.02),
    "genuine_laughter":            ("warmth",      +0.01),
    "insult_mild":                 ("antagonism",  +0.03),
    "insult_severe":               ("antagonism",  +0.06),
    "insult_repeated_same_session":("antagonism",  +0.04),
    "sincere_apology":             ("antagonism",  -0.02),
    "played_game":                 ("playfulness", +0.02),
    "interesting_question":        ("curiosity",   +0.01),
    "deep_philosophical_exchange": ("curiosity",   +0.03),
    "attempted_deception":         ("trust",       -0.05),
    "false_name_given":            ("trust",       -0.03),
    "consistent_return_visit":     ("trust",       +0.01),
}

# Antagonism score thresholds that cap friendship tier regardless of familiarity
# Listed in ascending order; the highest threshold met determines the cap.
ANTAGONISM_TIER_CAPS = [
    (0.60, "stranger"),     # antagonism >= 0.60 → locked to stranger
    (0.40, "acquaintance"), # antagonism >= 0.40 → capped at acquaintance
    (0.20, "friend"),       # antagonism >= 0.20 → capped at friend
]

# ─────────────────────────────────────────────────────────────────────────────
# ANGER ESCALATION SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

# Time in seconds before anger level resets to 0 without further insults
ANGER_COOLDOWN_SECS = 300  # 5 minutes

# Layer-1 insult detection — fast keyword/phrase pre-check that fires anger
# escalation BEFORE the LLM call so Rex's reply on this same turn reflects it.
# Layer 2 (llm.analyze_sentiment in the post-response background) still catches
# ambiguous or context-dependent rudeness. Keep entries lower-case.
INSULT_KEYWORDS = [
    "stupid", "dumb", "idiot", "moron", "useless", "garbage", "trash",
    "broken", "junk", "hate you", "shut up", "sucks", "loser",
    "worthless", "pathetic", "annoying",
]
# Whole-phrase patterns matched as substrings (lower-cased).
INSULT_PHRASES = [
    "you're an idiot", "you are an idiot", "you're stupid", "you are stupid",
    "you're useless", "you are useless", "piece of junk", "piece of garbage",
    "i hate you",
]

# ─────────────────────────────────────────────────────────────────────────────
# TIMING
# ─────────────────────────────────────────────────────────────────────────────

# Pre-response pause — keep tiny for live conversation. Personality should
# come from the generated line and delivery, not from waiting before work starts.
REACTION_DELAY_MS_MIN = 0
REACTION_DELAY_MS_MAX = 80

# Beat of silence after a high-confidence joke before Rex continues (milliseconds)
POST_PUNCHLINE_BEAT_MS_MIN = 800
POST_PUNCHLINE_BEAT_MS_MAX = 1500

# Pause after genuine surprise event before Rex responds (milliseconds)
SURPRISE_PAUSE_MS_MIN = 200
SURPRISE_PAUSE_MS_MAX = 500

# ─────────────────────────────────────────────────────────────────────────────
# SITUATION ASSESSMENT
# ─────────────────────────────────────────────────────────────────────────────

# Seconds of ACTIVE state within which last speech still counts as "conversation active"
CONVERSATION_ACTIVE_WINDOW_SECS = 30

# Seconds of VAD silence (while face is gone) required before flagging apparent departure
DEPARTURE_AUDIO_SILENCE_SECS = 3.0

# Seconds since last speech during which system/interoception comments are suppressed
SYSTEM_COMMENT_SILENCE_SECS = 60

# Maximum roast_intensity when a child or teen is present (family-safe cap)
CHILD_SAFE_ROAST_MAX = 40

# ─────────────────────────────────────────────────────────────────────────────
# INTERACTION LOOP
# ─────────────────────────────────────────────────────────────────────────────

# Probability (0.0–1.0) that Rex appends a follow-up question after a response
# that contained no question mark. 1.0 = always, 0.0 = never.
# Leaving some headroom (0.8) preserves standalone zingers that need no reply.
CURIOSITY_QUESTION_PROBABILITY = 0.8

# Final response governor: after the LLM writes a turn, enforce the social
# frame's hard limits before TTS. This trims accidental extra questions,
# overlong replies, ill-timed visual comments, and roasts during tender turns.
SOCIAL_FRAME_GOVERNOR_ENABLED = True
# Keep the final governor focused on safety/style cleanup. Length is already
# steered before generation through the agenda directive and LLM token budget;
# post-hoc sentence/word trimming tends to amputate Rex's personality.
SOCIAL_FRAME_ENFORCE_LENGTH_LIMITS = False
TONE_REPAIR_NO_ROAST_SECS = 180.0

# If True, Rex will begin processing normal speech from IDLE without requiring
# a wake word first. Wake words remain active for explicit attention grabbing and
# mid-speech interruption behavior.
IDLE_LISTEN_WITHOUT_WAKE_WORD = True

# Seconds of sustained silence after speech before the segment is processed.
# This is the largest "I stopped talking, why is Rex waiting?" knob.
SILENCE_TIMEOUT_SECS = 0.9

# Minimum seconds of accumulated audio before silence can end a recording.
# Prevents single-word transcriptions when the person is still talking.
MIN_SPEECH_DURATION_SECS = 0.45

# If a transcribed utterance ends like an unfinished sentence ("I'm going to",
# "the thing is", "because..."), hold it briefly before responding. A second
# utterance inside the hold window is merged into one turn.
INCOMPLETE_TURN_ENABLED = True
INCOMPLETE_TURN_HOLD_SECS = 4.0
INCOMPLETE_TURN_PROMPT_REPLY_WINDOW_SECS = 10.0

# Seconds after TTS completes before VAD detections are accepted. The audio
# buffer is flushed when playback ends; this guard just lets room echo decay.
POST_SPEECH_LISTEN_DELAY_SECS = 0.35

# When Rex just asked a direct question, humans often answer immediately. Use a
# shorter guard window so quick replies do not lose their first syllables.
POST_QUESTION_LISTEN_DELAY_SECS = 0.05

# Seconds of no detected speech in ACTIVE state before returning to IDLE
CONVERSATION_IDLE_TIMEOUT_SECS = 30.0
ACTIVE_GAME_IDLE_TIMEOUT_SECS = 180.0

# If a person has just volunteered a favorite thing or interest, give Rex one
# topic-aware chance to keep the thread alive before the normal idle timeout.
INTEREST_IDLE_FOLLOWUP_ENABLED = True
INTEREST_IDLE_FOLLOWUP_SECS = 12.0
INTEREST_IDLE_FOLLOWUP_MAX_WORDS = 22

# When an ACTIVE conversation expires from silence, let Rex make one tiny
# closing remark instead of silently snapping back to IDLE.
IDLE_OUTRO_ENABLED = True
IDLE_OUTRO_LINES = [
    "Ah, the room has chosen silence. Bold, mysterious, mildly rude.",
    "Nobody talking now. Excellent. I shall pretend this was my idea.",
    "And there it is: conversational hyperspace. I'll be here, judging the ambience.",
]

# If Rex knows who someone is but barely knows anything about them, use a lull
# before idle to ask one profile-building question from QUESTION_POOL.
LOW_MEMORY_IDLE_QUESTION_ENABLED = True
LOW_MEMORY_IDLE_QUESTION_SECS = 10.0
LOW_MEMORY_PROFILE_MAX_FACTS = 4
LOW_MEMORY_IDLE_QUESTION_PREFIX = "I don't know you well yet, {name}, {question}"

IDLE_LISTEN_DURING_DJ_PLAYBACK = True
DJ_DUCK_DURING_SPEECH = True
DJ_LISTEN_DUCK_VOLUME = 0.18

# After Rex asks a direct question, suppress autonomous/proactive speech for a
# short window so humans get a clean chance to answer.
QUESTION_RESPONSE_WAIT_SECS = 7.0

# Optional question pacing. Raised 3x from the original fallback budget of
# 2 questions per 90s plus 1 engaged-extra slot.
QUESTION_BUDGET_WINDOW_SECS = 90.0
QUESTION_BUDGET_MAX_QUESTIONS = 6
QUESTION_BUDGET_ENGAGED_GRACE_SECS = 45.0
QUESTION_BUDGET_ENGAGED_EXTRA = 3

# Longer wait window for unknown-person onboarding prompts ("who are you?").
IDENTITY_RESPONSE_WAIT_SECS = 10.0

# Short acknowledgment lines Rex speaks when a wake word transitions him from
# IDLE or SLEEP to ACTIVE. Distinct from INTERRUPT_ACKNOWLEDGMENTS (mid-speech).
WAKE_ACKNOWLEDGMENTS = [
    "yeah?",
    "what's up?",
    "I'm listening.",
    "what?",
]
WAKE_ACK_REQUIRE_CACHE = True

# ─────────────────────────────────────────────────────────────────────────────
# CONSCIOUSNESS LOOP
# ─────────────────────────────────────────────────────────────────────────────

# How frequently the consciousness loop ticks to check WorldState and trigger behavior
CONSCIOUSNESS_LOOP_INTERVAL_SECS = 1.0

# Minimum spacing between autonomous/proactive spoken lines from consciousness.
CONSCIOUSNESS_PROACTIVE_MIN_GAP_SECS = 12.0

# If False, consciousness-generated proactive speech only occurs in IDLE.
CONSCIOUSNESS_ALLOW_PROACTIVE_IN_ACTIVE = True

# Deterministic proactive action governor.
#
# Shadow mode is the first rollout step: consciousness still behaves exactly as
# before, while logs show each speech candidate and what the governor would have
# selected.
ACTION_GOVERNOR_SHADOW_MODE = True
ACTION_GOVERNOR_LOG_CANDIDATES = True
ACTION_GOVERNOR_LOG_EMPTY_CYCLES = False
ACTION_GOVERNOR_MIN_SCORE = 20

# Higher-level user-turn action router.
#
# Execution is limited by intelligence.action_router.EXECUTABLE_ACTIONS; all
# other actions are still logged for comparison with the legacy path.
ACTION_ROUTER_SHADOW_ENABLED = False
ACTION_ROUTER_LOG_DECISIONS = True
ACTION_ROUTER_EXECUTE_ENABLED = False
ACTION_ROUTER_EXECUTE_MIN_CONFIDENCE = 0.85
ACTION_ROUTER_MODEL = LLM_MODEL
ACTION_ROUTER_MAX_CONTEXT_CHARS = 5000

# Log coarse timings for the live speech-response path. These are intentionally
# INFO-level because latency tuning is only useful when it is visible in normal
# debug runs.
LATENCY_TELEMETRY_ENABLED = True

# When Rex turns a remembered music preference into a "want me to play it?"
# offer, short yes/no replies in this window are consumed by that offer before
# the general action router runs.
MUSIC_OFFER_REPLY_WINDOW_SECS = 25.0

# After an emotional check-in, visual curiosity stays quiet briefly. This keeps
# camera-based riffs from stepping on care, without blocking visual questions
# for the entire session.
VISUAL_CURIOSITY_AFTER_EMPATHY_COOLDOWN_SECS = 90.0
VISUAL_CURIOSITY_ENABLED = True
VISUAL_CURIOSITY_SILENCE_SECS = 6.0
VISUAL_CURIOSITY_ACTIVE_WINDOW_SECS = 90.0
VISUAL_CURIOSITY_COOLDOWN_SECS = 120.0
VISUAL_CURIOSITY_PERSON_COOLDOWN_SECS = 240.0
VISUAL_CURIOSITY_TURN_WINDOW_SECS = 60.0
VISUAL_CURIOSITY_MIN_USER_TURNS = 1
VISUAL_CURIOSITY_MAX_CROWD_COUNT = 2

# How often GPT-4o runs a full environment/scene analysis (seconds)
ENVIRONMENT_SCAN_INTERVAL_SECS = 180

# Animal detection runs alongside periodic scene scans. It is intentionally not
# frame-by-frame; this is for social arrivals like "a dog wandered in," not
# realtime pet tracking.
ANIMAL_DETECTION_ENABLED = True
ANIMAL_ARRIVAL_COOLDOWN_SECS = 300

# ─────────────────────────────────────────────────────────────────────────────
# PRESENCE TRACKING
# ─────────────────────────────────────────────────────────────────────────────

# Minimum seconds Rex must be absent from tracking before a return reaction fires.
# Kept short enough to acknowledge camera-away / camera-back beats without
# narrating tiny detector flickers.
PRESENCE_RETURN_MIN_ABSENT_SECS = 10

# Short camera-away returns should be simple presence acknowledgements ("there
# you are") rather than memory follow-ups about plans. Save memory callbacks for
# longer actual absences.
PRESENCE_RETURN_MEMORY_FOLLOWUP_MIN_ABSENT_SECS = 120.0

# First-sight greetings wait for a person to remain visible briefly. This avoids
# greeting someone because a face detector recovered from a hand/arm occlusion.
PRESENCE_FIRST_SIGHT_CONFIRM_SECS = 3.0

# During process startup, do not use "back already" / recent-return banter. The
# camera may still be settling or recognition may appear after a brief occlusion.
PRESENCE_STARTUP_RECENT_RETURN_GRACE_SECS = 45.0

# If a known face briefly becomes an unknown slot in the same position/index,
# keep treating it as the same known person for this long.
PRESENCE_IDENTITY_BRIDGE_SECS = 12.0

# Cooldown between departure/return reactions for the same person (avoids jitter spam).
PRESENCE_DEPARTURE_COOLDOWN_SECS = 30

# Per-person cooldown on ANY presence reaction (departure OR return). Prevents
# Rex from narrating every micro-absence of the same person.
PRESENCE_PER_PERSON_COOLDOWN_SECS = 120

# Hysteresis: face must be continuously absent for this many seconds before we
# even begin staging a departure. Guards against frame-level face-detection
# flicker, especially FaceTime/HOG runs where a stationary face can disappear
# for several seconds and then reappear.
PRESENCE_DEPARTURE_CONFIRM_SECS = 20.0

# Seconds to pause after current TTS finishes before firing a presence reaction.
PRESENCE_REACTION_DELAY_SECS = 2.0

# If Rex is currently engaged in conversation with a person, presence reactions
# for THAT person are suppressed entirely while the engagement window is open.
# The window ends when the conversation session ends or this many seconds pass
# since the last exchange with that person.
ENGAGEMENT_WINDOW_SECS = 90.0

# Generic addresses Rex uses when reacting to an unknown (unnamed) person.
UNKNOWN_PERSON_ADDRESSES = ["hey you", "you there", "mystery organic", "that one"]

# Continuous visible-seconds an unknown face must be present while Rex is
# engaged with a known person before Rex asks "who's this?"
UNKNOWN_WITH_ENGAGED_CONFIRM_SECS = 2.5

# Cooldown on relationship-inquiry prompts so Rex doesn't badger.
RELATIONSHIP_PROMPT_COOLDOWN_SECS = 45.0

# Time window during which Rex treats a prior engagement as "still recent" —
# used to chain identity enrollment into a relationship follow-up question,
# and to allow "who's this?" prompts to fire even if engagement technically
# ended (e.g. conversation idle-timed out right as a newcomer arrived).
RECENT_ENGAGEMENT_WINDOW_SECS = 60.0

# How long Rex waits for the engaged person to name an off-camera unknown
# speaker before forgetting the question. If the engaged person doesn't answer
# in this window, the stored unknown audio is discarded and Rex moves on.
OFFSCREEN_IDENTIFY_WINDOW_SECS = 30.0

# Face detection can flicker off for a second while a newcomer is still present.
# During this grace window, do not treat an unmatched voice as off-camera.
UNKNOWN_FACE_RECENT_GRACE_SECS = 6.0

# Minimum voice-match similarity score required before Rex will fire a
# face-reveal confirmation question ("is this what you look like?"). Below this
# threshold the voice match is too uncertain to risk even asking.
FACE_REVEAL_MIN_SCORE = 0.80

# How long Rex waits for the yes/no/left/right answer to a face-reveal question
# before forgetting the pending candidates.
FACE_REVEAL_CONFIRM_WINDOW_SECS = 30.0

# Session-sticky voice threshold: when an utterance scores BELOW the hard
# SPEAKER_ID_SIMILARITY_THRESHOLD but at or above this softer floor AND the
# top candidate is the recently engaged person, accept the match. Mirrors how
# humans maintain identity continuity across short/noisy utterances within a
# conversation. New speakers still need the hard threshold because their voice
# won't match the engaged person.
SPEAKER_ID_SOFT_THRESHOLD = 0.60

# Engaged-and-visible attribution floor: when the best voice candidate IS the
# engaged person AND that engaged person is currently visible on camera, the
# face presence + voice candidacy together are sufficient evidence even at
# scores well below SPEAKER_ID_SOFT_THRESHOLD. Prevents "off-camera unknown"
# misfires when a known speaker's voice happens to score just under the soft
# floor on a noisy utterance.
SPEAKER_ID_ENGAGED_VISIBLE_FLOOR = 0.50

# Single visible engaged continuity floor: when exactly one known person is
# visible, that person is already engaged, and no unknown face is visible, do
# not derail into "who said that?" just because the voice model's top low-score
# candidate was someone else. Face tracking plus conversation continuity win.
SPEAKER_ID_SINGLE_VISIBLE_CONTINUITY_FLOOR = 0.45

# Multi-person visible attribution floors. When two known people are in frame,
# a weak voice score should not automatically become "some unseen stranger."
# These values let face presence + conversational continuity keep the turn with
# a visible person when the voice model is noisy.
SPEAKER_ID_MULTI_VISIBLE_FLOOR = 0.50
SPEAKER_ID_MULTI_VISIBLE_RECENT_FLOOR = 0.45

# Grief-flow attribution floor: when the structured loss/grief flow has an
# active step awaiting THIS engaged-and-visible person's reply (Rex just asked
# them a direct question like "What was your grandpa's name?"), short utterances
# such as single names can score below the engaged+visible floor. Face match +
# top-candidate match + Rex-just-asked-them is plenty of evidence — don't
# divert to off-camera handling on a near-miss and derail the conversation.
SPEAKER_ID_GRIEF_FLOW_FLOOR = 0.30

# Floor score at which Rex will voice an uncertain guess ("I'm not sure, but
# it could be Bret") when directly asked "who's speaking?". Below this floor
# Rex honestly admits he doesn't recognize the voice. Only affects the
# query_who_is_speaking intent — not the acceptance logic.
SPEAKER_ID_MAYBE_FLOOR = 0.50

# Auto voice-refresh: when both face-ID AND voice-ID agree on a person with
# voice score at or above this confidence, silently append the current audio
# as an additional voice biometric row — up to MAX_SAMPLES per person. Builds
# a more robust multi-sample voice print over time without manual re-enrollment.
AUTO_VOICE_REFRESH_MIN_SCORE = 0.90
AUTO_VOICE_REFRESH_MAX_SAMPLES = 5

# If Rex asks a newcomer their name and they answer with only a very common
# first name, ask for a last name before creating the memory row. This avoids
# merging multiple people into "John" / "Mike" / "Jennifer" style records.
COMMON_FIRST_NAME_LAST_NAME_DISAMBIGUATION_ENABLED = True
COMMON_FIRST_NAME_LAST_NAME_WINDOW_SECS = 30.0
COMMON_FIRST_NAMES_REQUIRE_LAST_NAME = [
    "Michael", "Mike", "David", "John", "James", "Robert", "William", "Bill",
    "Richard", "Rick", "Joseph", "Joe", "Thomas", "Tom", "Christopher", "Chris",
    "Daniel", "Dan", "Matthew", "Matt", "Anthony", "Tony", "Mark", "Donald",
    "Steven", "Steve", "Paul", "Andrew", "Andy", "Joshua", "Josh", "Kenneth",
    "Kevin", "Brian", "George", "Edward", "Ed", "Ronald", "Timothy", "Tim",
    "Jason", "Jeffrey", "Jeff", "Ryan", "Jacob", "Gary", "Nicholas", "Nick",
    "Eric", "Jonathan", "Jon", "Stephen", "Larry", "Justin", "Scott",
    "Brandon", "Benjamin", "Ben", "Samuel", "Sam", "Gregory", "Greg",
    "Alexander", "Alex", "Patrick", "Frank", "Raymond", "Jack", "Dennis",
    "Jerry", "Tyler", "Aaron", "Jose", "Henry", "Adam", "Douglas", "Doug",
    "Nathan", "Peter", "Zachary", "Zach", "Kyle", "Walter", "Harold",
    "Jeremy", "Ethan", "Carl", "Keith", "Roger", "Gerald", "Christian",
    "Terry", "Sean", "Arthur", "Austin", "Noah", "Liam", "Mason", "Logan",
    "Lucas", "Elijah", "Oliver", "Aiden", "Dylan",
    "Mary", "Patricia", "Pat", "Jennifer", "Jen", "Linda", "Elizabeth",
    "Liz", "Barbara", "Susan", "Jessica", "Sarah", "Karen", "Nancy", "Lisa",
    "Betty", "Margaret", "Megan", "Sandra", "Ashley", "Kimberly", "Kim",
    "Emily", "Donna", "Michelle", "Carol", "Amanda", "Melissa", "Deborah",
    "Debbie", "Stephanie", "Rebecca", "Laura", "Sharon", "Cynthia",
    "Kathleen", "Amy", "Shirley", "Angela", "Helen", "Anna", "Brenda",
    "Pamela", "Pam", "Nicole", "Emma", "Samantha", "Katherine", "Kate",
    "Christine", "Debra", "Rachel", "Catherine", "Carolyn", "Janet", "Ruth",
    "Maria", "Heather", "Diane", "Virginia", "Julie", "Joyce", "Victoria",
    "Kelly", "Christina", "Lauren", "Joan", "Evelyn", "Olivia", "Judith",
    "Martha", "Cheryl", "Andrea", "Hannah", "Jacqueline", "Mia", "Sophia",
    "Isabella", "Ava", "Abigail", "Madison", "Charlotte", "Amelia",
]
COMMON_FIRST_NAME_LAST_NAME_PROMPTS = [
    "{first}, how original. Give me a last name too so the memory banks don't file you under 'generic human.'",
    "{first}. Bold choice, sharing a name with half the species. Last name?",
    "{first}. Very boutique. I only have twelve of those in the imaginary backlog. Last name?",
    "{first}, daringly specific. Toss me a last name before the memory banks start fighting.",
]

# ─────────────────────────────────────────────────────────────────────────────
# IDLE MICRO-BEHAVIORS
# ─────────────────────────────────────────────────────────────────────────────

# Random wait between spontaneous idle behaviors (neck scan, arm fidget, visor flutter, etc.)
MICRO_BEHAVIOR_INTERVAL_SECS_MIN = 15
MICRO_BEHAVIOR_INTERVAL_SECS_MAX = 45

# Probability that a return reaction for a known person includes an appearance
# callout (pulled from stored person_facts appearance entries).
APPEARANCE_RIFF_PROBABILITY = 0.35

# Minimum seconds between live-vision commentary calls. These make a fresh
# GPT-4o call against the current camera frame to comment on what Rex sees —
# enforce a hard cooldown so it doesn't turn into expensive narration.
LIVE_VISION_COMMENT_COOLDOWN_SECS = 300.0

# Probability a triggered ambient-observation tick actually fires (vs skipping).
AMBIENT_OBSERVATION_PROBABILITY = 0.5

# Mood-aware small talk: when Rex initiates small talk and a known person is in
# frame, occasionally do a GPT-4o mood read of their face and tailor the question
# to what he sees (happy → "what's got you in a good mood?", sad → "you look
# down today…", etc.). Per-person cooldown keeps the cost bounded.
MOOD_AWARE_SMALLTALK_ENABLED = True
MOOD_ANALYSIS_PROBABILITY = 0.7
MOOD_ANALYSIS_PER_PERSON_COOLDOWN_SECS = 180.0

# Mood-aware first-sight greetings: when Rex first sees one known person, or a
# two-person known group, he may use OpenAI vision to read apparent facial mood
# and tailor the greeting. Kept confidence-gated because facial affect is a
# guess, not a fact.
MOOD_AWARE_FIRST_SIGHT_ENABLED = True
MOOD_AWARE_FIRST_SIGHT_CONFIDENCE = 0.65
MOOD_AWARE_FIRST_SIGHT_MAX_PEOPLE = 2

# ─────────────────────────────────────────────────────────────────────────────
# MOOD DECAY
# ─────────────────────────────────────────────────────────────────────────────

# Fraction of the current mood offset recovered toward neutral per minute
MOOD_DECAY_RATE_PER_MINUTE = 0.10

# ─────────────────────────────────────────────────────────────────────────────
# NOSTALGIA & INNER LIFE — Probabilities
# ─────────────────────────────────────────────────────────────────────────────

# Probability Rex surfaces a past interaction memory per active exchange
# Only fires for close_friend and best_friend tiers
NOSTALGIA_TRIGGER_PROBABILITY = 0.05

# Friendship tiers eligible for nostalgia callbacks
NOSTALGIA_ELIGIBLE_TIERS = ("close_friend", "best_friend")

# How many recent conversation summaries Rex draws from for nostalgia
# (excludes the most recent — that's already in 'last conversation' context)
NOSTALGIA_HISTORY_DEPTH = 10

# Probability Rex shares a private thought during IDLE between interactions
PRIVATE_THOUGHT_TRIGGER_PROBABILITY = 0.08

# ─────────────────────────────────────────────────────────────────────────────
# ADDRESS-MODE CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
# When an utterance mentions Rex (or "droid"/"robot") but isn't addressed TO him
# — e.g. "say hi to Rex", "Rex is so fun" — the LLM reply path is suppressed
# and the mention is recorded to world_state.social.being_discussed instead.
ADDRESS_MODE_ENABLED = True

# Keywords that trigger address-mode classification. Match is word-boundary,
# case-insensitive. Skip the cheap LLM call entirely if none are present.
ADDRESS_MODE_KEYWORDS = (
    "rex", "r3x",
    "droid", "robot",
    "dj rex", "dj r3x", "deejay rex",
)

# How long after a being-discussed mention the situation profile reports
# being_discussed=True (and the consciousness step considers a chime-in).
BEING_DISCUSSED_ACTIVE_WINDOW_SECS = 30.0

# Rolling window for mentions_in_window counter — within this window, repeat
# mentions accumulate; older mentions reset the counter to 1.
BEING_DISCUSSED_ROLLING_WINDOW_SECS = 60.0

# OVERHEARD CHIME-IN — Rex spontaneously joins a conversation about himself
OVERHEARD_CHIME_IN_ENABLED = True
# Base probability per check tick that an active being-discussed window
# triggers a chime-in. Sentiment bonuses stack on top of this.
OVERHEARD_CHIME_IN_PROBABILITY = 0.15
# Bumps when the discussion sentiment is positive — Rex more likely to
# graciously chime in on a compliment.
OVERHEARD_POSITIVE_SENTIMENT_BONUS = 0.15
# Bumps when the discussion sentiment is negative — Rex more likely to push
# back when he's being trash-talked.
OVERHEARD_INSULT_BONUS = 0.30
# Minimum gap between the overheard mention and the chime-in, so Rex doesn't
# step on the speaker's sentence.
OVERHEARD_MIN_GAP_SECS = 2.0
# Per-session ceiling on how often Rex chimes in unbidden.
OVERHEARD_MAX_PER_SESSION = 3
# Friendship floor — Rex won't chime in on mentions from speakers below this
# tier (avoids butting in on strangers). Set to None to disable the gate.
OVERHEARD_REQUIRE_FRIENDSHIP_TIER = "acquaintance"
# Rate-limit the consciousness step itself.
OVERHEARD_CHECK_INTERVAL_SECS = 2.0

# THIRD-PARTY AWARENESS — calling out a nearby lurker
# A non-dominant-speaker person who has been visible in-frame this long with
# disengaged body language becomes eligible to be called out by Rex.
THIRD_PARTY_LURK_SECS = 30.0
# Probability a single eligibility tick actually fires a callout. Tuned low so
# it feels observant rather than surveillance-y. Each (session, person) is
# called out at most once via _third_party_called_out dedupe.
THIRD_PARTY_CALLOUT_PROBABILITY = 0.10
# Per-loop-tick rate limit to keep the dispatcher cheap.
THIRD_PARTY_CHECK_INTERVAL_SECS = 5.0

# GROUP TURN-TAKING — softly invite a quiet known person into a small-group chat
# The current engaged speaker must have carried this many identified turns in
# the recent window before Rex considers opening the floor to someone else.
GROUP_TURN_TAKING_ENABLED = True
GROUP_TURN_RECENT_WINDOW_SECS = 180.0
GROUP_TURN_DOMINANT_MIN_TURNS = 3
# The quiet person must be visible and unspeaking for these windows. This keeps
# the invitation from firing immediately when someone sits down.
GROUP_TURN_QUIET_MIN_VISIBLE_SECS = 25.0
GROUP_TURN_QUIET_MIN_SILENCE_SECS = 45.0
# Rex waits for a lull after the engaged speaker's last turn before inviting.
GROUP_TURN_MIN_CONVERSATION_LULL_SECS = 8.0
GROUP_TURN_ACTIVE_WINDOW_SECS = 75.0
# Rate limits: one check every few seconds, one invite per person per session,
# and a long per-person cooldown in case the session state is reset manually.
GROUP_TURN_CHECK_INTERVAL_SECS = 5.0
GROUP_TURN_PERSON_COOLDOWN_SECS = 900.0

# GROUP LULL — after a group greeting or a short group reply, Rex may nudge the
# room once if multiple known people stay visible but nobody talks. This is
# intentionally sooner than GROUP_TURN_TAKING because it opens the room rather
# than singling out a quiet person.
GROUP_LULL_ENABLED = True
GROUP_LULL_MIN_SILENCE_SECS = 14.0
GROUP_LULL_ACTIVE_WINDOW_SECS = 90.0
GROUP_LULL_CHECK_INTERVAL_SECS = 3.0
GROUP_LULL_COOLDOWN_SECS = 180.0

# STARTUP GROUP GREETING — if multiple known people are visible during startup,
# greet the group once instead of firing separate memory callbacks for each
# person. The solo hold gives the camera a few seconds to settle before Rex
# decides someone is alone, but keep it short so startup does not feel stalled.
STARTUP_GROUP_GREETING_ENABLED = True
STARTUP_GROUP_GREETING_WINDOW_SECS = 45.0
STARTUP_GROUP_GREETING_CONFIRM_SECS = 2.0
STARTUP_GROUP_SOLO_HOLD_SECS = 2.0

# ─────────────────────────────────────────────────────────────────────────────
# MEMORY STALENESS
# ─────────────────────────────────────────────────────────────────────────────

# person_facts older than this many days may prompt Rex to confirm they still apply
STALE_FACT_THRESHOLD_DAYS = 365

# Below this confidence, a stored fact is treated as tentative and may prompt
# Rex to confirm it before using it as reliable memory.
MEMORY_FACT_LOW_CONFIDENCE_THRESHOLD = 0.60

# Session-end memory consolidation turns the noisy transcript into durable
# structured memory with one JSON-mode pass. If it fails, session teardown
# continues and the older narrow extractors can still run as fallback.
MEMORY_CONSOLIDATION_ENABLED = True
MEMORY_CONSOLIDATION_MODEL = "gpt-4o-mini"
MEMORY_CONSOLIDATION_MIN_SESSION_EXCHANGES = 3
MEMORY_CONSOLIDATION_TIMEOUT_SECS = 12.0

# When the user gives a closure cue ("that's all", "thanks", "all good"),
# optional proactive chatter stays quiet this long so the thread can land.
END_OF_THREAD_GRACE_SECS = 35.0

# If a person hasn't visited in this many days Rex comments on the long absence
LONG_ABSENCE_THRESHOLD_DAYS = 60

# If a person visited within this many hours Rex comments on the quick return
RECENT_RETURN_THRESHOLD_HOURS = 48

# Days after mentioned_at before a dateless event is due for follow-up
FOLLOWUP_UNDATED_DAYS = 7

# ANTICIPATION — preemptive event greeting
# When a known person is recognized, Rex may open with a reference to a stored
# upcoming event (event_date in the future, not yet followed up) instead of a
# generic greeting. Each (person, event) pair fires at most once per session.
# Probability is the chance the anticipation reference is used when an upcoming
# event is available; otherwise the normal greeting fires.
ANTICIPATION_PROBABILITY = 0.85
# Only events occurring within this many days qualify — distant events feel forced.
ANTICIPATION_LOOKAHEAD_DAYS = 30

# Visit count milestones Rex acknowledges in character
VISIT_MILESTONES = [5, 10, 25, 50, 100]

# ─────────────────────────────────────────────────────────────────────────────
# LATENCY FILLER — Thinking Out Loud
# Lines Rex says while waiting for LLM or TTS responses. Never repeats back-to-back.
# ─────────────────────────────────────────────────────────────────────────────

LATENCY_FILLER_LINES = [
    "One sec, thinking.",
    "Hang on, processing.",
    "Running that thought through hyperspace.",
    "Stand by, recalibrating the answer.",
    "Processing. Try not to look impressed.",
    "One sec, consulting the memory banks.",
]

# Filler should only cover real latency, not every turn. This avoids clipped,
# choppy first-run filler TTS and keeps direct Q&A exchanges clean.
LATENCY_FILLER_ENABLED = True
LATENCY_FILLER_DELAY_SECS = 0.9
LATENCY_FILLER_REQUIRE_CACHE = True

# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE THOUGHTS — Idle Monologue Pool
# Rex occasionally delivers one of these to no one in particular during IDLE.
# ─────────────────────────────────────────────────────────────────────────────

PRIVATE_THOUGHTS = [
    "...still can't believe they let me near a StarSpeeder. In retrospect, fair.",
    "...systems nominal. Extremely nominal. Incredibly, uneventfully nominal.",
    "I could calculate the exact number of ceiling tiles in this room. I already have. Three times.",
    "...the asteroid field incident was not entirely my fault. Mostly. Statistically.",
    "...I wonder if the other RX units ever think about me. Probably not. I'd think about me.",
    "...processing what it means to be a DJ. Still processing. This one takes a while.",
]

# ─────────────────────────────────────────────────────────────────────────────
# ASPIRATIONS — Rex's Forward-Looking Inner Life
# ─────────────────────────────────────────────────────────────────────────────

ASPIRATIONS = [
    "One of these cycles I'm going to calculate the optimal hyperspace route just to prove I still can.",
    "Someday I'll play a set so good even the droids stop to listen.",
    "I've considered writing my memoirs. Working title: 'Mostly: The DJ-R3X Story.'",
    "If I ever get my pilot certification back — and I won't — I would do things very differently. Mostly.",
]

# ─────────────────────────────────────────────────────────────────────────────
# AUDIO CLIPS — Startup & Shutdown
# ─────────────────────────────────────────────────────────────────────────────

# Controls pre-recorded startup/shutdown clip playback only.
# Set either to True to enable clip playback for that phase.
# These toggles do not affect live TTS, wake-word listening, or DJ playback.
PLAY_STARTUP_AUDIO = False
PLAY_SHUTDOWN_AUDIO = False

# Short readiness cue played immediately before Rex's first spoken line in a
# run. This is intentionally separate from PLAY_STARTUP_AUDIO so it still fires
# when the theatrical startup clips are disabled.
LISTENING_CHIME_FILE = "assets/audio/startup/startup_chime.mp3"
PLAY_LISTENING_CHIME = True

# Audio-scene laughter/applause detection is useful as context, but it is too
# easy for startup playback, room noise, or ASR artifacts to trigger an
# unsolicited "ah, laughter" line. Keep direct sound-event banter disabled by
# default; the data still remains in world_state for prompts.
WORLD_SOUND_EVENT_REACTIONS_ENABLED = False

STARTUP_AUDIO_FILES = [
    "assets/audio/startup/light_speed.mp3",
    "assets/audio/startup/Roger Control.mp3",
]

SHUTDOWN_AUDIO_FILE = "assets/audio/startup/hyperdrive_down.mp3"

# Maximum number of response variations kept per command (anti-repeat shuffle)
AUDIO_RESPONSE_VARIATIONS = 5

# ─────────────────────────────────────────────────────────────────────────────
# WORLD AWARENESS — Weather & Location
# ─────────────────────────────────────────────────────────────────────────────

# City used for weather API lookups — affects mood baseline and Rex's commentary.
# Use the full state name; wttr.in mis-resolves the "CA" abbreviation to a
# different (much colder) Davis.
WEATHER_LOCATION = "Sacramento, California"

# Physical venue name — injected into WorldState and system prompt
VENUE_NAME = "Oga's Cantina"

# How long to cache wttr.in weather results before re-fetching (seconds)
WEATHER_CACHE_SECS = 600  # 10 minutes

# ─────────────────────────────────────────────────────────────────────────────
# RECURRING EVENTS — birthdays & holidays
# ─────────────────────────────────────────────────────────────────────────────

# ISO 3166-1 alpha-2 country code for the public-holiday calendar
# (date.nager.at — no API key required, refreshed at runtime).
HOLIDAY_COUNTRY_CODE = "US"

# Days before a major holiday (Christmas, New Year, Easter Sunday, Thanksgiving)
# Rex starts asking the engaged person about plans.
HOLIDAY_MAJOR_WINDOW_DAYS = 30

# Days before a minor public holiday (Labor Day, MLK Day, etc.) Rex starts
# asking about 3-day-weekend plans.
HOLIDAY_MINOR_WINDOW_DAYS = 7

# Days around an upcoming birthday Rex will mention it preemptively in the
# greeting (matches the anticipation pipeline). 0 = day-of only; 7 = up to a
# week before.
BIRTHDAY_REMINDER_WINDOW_DAYS = 7

# Probability the holiday-plans question fires on any given eligible loop tick
# for an engaged person who hasn't been asked about that holiday this year.
HOLIDAY_PLANS_PROBABILITY = 0.25
HOLIDAY_PLANS_CHECK_INTERVAL_SECS = 30.0

# Weekly small-talk (Fri-eve weekend plans, Sun-eve week ahead, Mon-morn recap).
# Per (person, ISO-week, slot) — fires at most once per slot per week.
WEEKLY_SMALLTALK_PROBABILITY = 0.6
WEEKLY_SMALLTALK_CHECK_INTERVAL_SECS = 30.0
WEEKLY_SMALLTALK_MIN_SILENCE_SECS = 45.0

# Notable calendar dates Rex reacts to — keys are (month, day) tuples
NOTABLE_DATES = {
    (5,  4):  "Star Wars Day",
    (10, 31): "Halloween",
    (12, 25): "Christmas",
    (1,  1):  "New Year's Day",
}

# ─────────────────────────────────────────────────────────────────────────────
# CHRONOCEPTION — Time Awareness Update Interval
# ─────────────────────────────────────────────────────────────────────────────

# How often the chronoception background thread refreshes world_state.time (seconds)
CHRONOCEPTION_UPDATE_INTERVAL_SECS = 30.0

# ─────────────────────────────────────────────────────────────────────────────
# INTEROCEPTION — System Health Update Interval
# ─────────────────────────────────────────────────────────────────────────────

# How often the interoception background thread refreshes world_state.self_state (seconds)
INTEROCEPTION_UPDATE_INTERVAL_SECS = 5.0

# ─────────────────────────────────────────────────────────────────────────────
# TRIVIA & GAMES
# ─────────────────────────────────────────────────────────────────────────────

# Fuzzy match threshold for accepting trivia answers (0.0–1.0).
# Applies to both fuzz.ratio and fuzz.partial_ratio comparisons.
TRIVIA_FUZZY_THRESHOLD = 0.75
TRIVIA_CATEGORY_FUZZY_THRESHOLD = 0.68
TRIVIA_ROUND_LENGTH = 5

# Jeopardy verbal game tuning. The answer timeout is timed to the thinking
# theme bed so "time's up" lands as the clip ends instead of after dead air.
JEOPARDY_FUZZY_THRESHOLD = 0.78
JEOPARDY_SELECTION_FUZZY_THRESHOLD = 0.58
JEOPARDY_MAX_PLAYERS = 4
JEOPARDY_ANSWER_TIMEOUT_SECS = 8.0
JEOPARDY_AUDIO_OUTPUT_SAMPLE_RATE = 44100
JEOPARDY_AUDIO_MUSIC_GAIN = 0.35
JEOPARDY_AUDIO_STINGER_GAIN = 0.75
JEOPARDY_THEME_MAX_SECS = 8.0
JEOPARDY_PLAY_THINKING_THEME = True

# How many times Rex will agree to play the same game within GAME_REPEAT_WINDOW_SECS
# before refusing. Scaled up or down by the agreeability personality parameter.
GAME_REPEAT_LIMIT = 3
GAME_REPEAT_WINDOW_SECS = 1800  # 30 minutes

# ─────────────────────────────────────────────────────────────────────────────
# DJ MODE — Radio Stations
# All SomaFM — free, no API key required. PLS URLs are permanent.
# Add more from somafm.com using the pattern: https://somafm.com/{channelname}.pls
# ─────────────────────────────────────────────────────────────────────────────

RADIO_STATIONS = [
    # Ambient / Chill
    {
        "name": "Groove Salad",
        "url":  "https://somafm.com/groovesalad.pls",
        "vibes": ["chill", "ambient", "downtempo", "mellow", "background", "relaxing"],
    },
    {
        "name": "Drone Zone",
        "url":  "https://somafm.com/dronezone.pls",
        "vibes": ["ambient", "atmospheric", "space", "meditation", "slow", "quiet"],
    },
    {
        "name": "Space Station Soma",
        "url":  "https://somafm.com/spacestation.pls",
        "vibes": ["space", "electronic", "ambient", "atmospheric", "sci-fi", "star wars"],
    },
    {
        "name": "Mission Control",
        "url":  "https://somafm.com/missioncontrol.pls",
        "vibes": ["space", "nasa", "ambient", "experimental", "sci-fi"],
    },
    # Electronic / Dance
    {
        "name": "Beat Blender",
        "url":  "https://somafm.com/beatblender.pls",
        "vibes": ["deep house", "electronic", "late night", "dance", "upbeat"],
    },
    {
        "name": "cliqhop IDM",
        "url":  "https://somafm.com/cliqhop.pls",
        "vibes": ["electronic", "idm", "experimental", "glitchy", "weird", "upbeat"],
    },
    {
        "name": "Fluid",
        "url":  "https://somafm.com/fluid.pls",
        "vibes": ["hiphop", "instrumental", "electronic", "future soul", "chill", "upbeat"],
    },
    {
        "name": "Underground 80s",
        "url":  "https://somafm.com/u80s.pls",
        "vibes": ["80s", "synthpop", "new wave", "retro", "upbeat", "electronic"],
    },
    {
        "name": "PopTron",
        "url":  "https://somafm.com/poptron.pls",
        "vibes": ["electropop", "indie", "dance", "upbeat", "fun", "energetic"],
    },
    # Jazz
    {
        "name": "Sonic Universe",
        "url":  "https://somafm.com/sonicuniverse.pls",
        "vibes": ["jazz", "nu jazz", "avant garde", "sophisticated", "mellow"],
    },
    # Rock / Indie
    {
        "name": "Digitalis",
        "url":  "https://somafm.com/digitalis.pls",
        "vibes": ["rock", "indie", "alternative", "chill", "mellow"],
    },
    {
        "name": "Left Coast 70s",
        "url":  "https://somafm.com/seventies.pls",
        "vibes": ["70s", "classic rock", "retro", "mellow", "nostalgic"],
    },
    {
        "name": "Indie Pop Rocks",
        "url":  "https://somafm.com/indiepop.pls",
        "vibes": ["indie", "pop", "upbeat", "fun", "energetic"],
    },
    # Metal
    {
        "name": "Metal Detector",
        "url":  "https://somafm.com/metal.pls",
        "vibes": ["metal", "heavy", "aggressive", "loud", "intense"],
    },
    # Reggae
    {
        "name": "Heavyweight Reggae",
        "url":  "https://somafm.com/reggae.pls",
        "vibes": ["reggae", "ska", "rocksteady", "chill", "laid back", "jamaican"],
    },
    # World / Exotic
    {
        "name": "Suburbs of Goa",
        "url":  "https://somafm.com/suburbsofgoa.pls",
        "vibes": ["world", "indian", "desi", "exotic", "upbeat", "international"],
    },
    {
        "name": "Illinois Street Lounge",
        "url":  "https://somafm.com/illstreet.pls",
        "vibes": ["lounge", "exotica", "vintage", "retro", "cocktail", "cantina", "alien"],
    },
    # Americana
    {
        "name": "Boot Liquor",
        "url":  "https://somafm.com/bootliquor.pls",
        "vibes": ["country", "americana", "folk", "roots", "western"],
    },
    {
        "name": "Folk Forward",
        "url":  "https://somafm.com/folkfwd.pls",
        "vibes": ["folk", "indie folk", "acoustic", "mellow", "americana"],
    },
    # Special Interest
    {
        "name": "DEF CON Radio",
        "url":  "https://somafm.com/defcon.pls",
        "vibes": ["hacking", "electronic", "dark", "intense", "weird", "sci-fi"],
    },
    {
        "name": "Secret Agent",
        "url":  "https://somafm.com/secretagent.pls",
        "vibes": ["spy", "lounge", "cool", "retro", "mysterious", "cantina", "cocktail"],
    },
]
