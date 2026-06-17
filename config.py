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


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


_SERVO_ENV_US_MIN = 300.0
_SERVO_ENV_US_MAX = 3000.0


def _env_int(name: str, default: int, *, min_value: int, max_value: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(min_value, min(max_value, value))


def _env_float(name: str, default: float, *, min_value: float, max_value: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(min_value, min(max_value, value))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default

# ─────────────────────────────────────────────────────────────────────────────
# DEBUG
# ─────────────────────────────────────────────────────────────────────────────

# When True, each run writes its OWN timestamped log files —
# logs/djr3x-<YYYY-MM-DD-HH-MM-SS>.log and logs/conversation-<stamp>.log — so per-run
# history accumulates distinctly (handy for comparing runs; clean old ones up
# manually). When False, one shared logs/djr3x.log that size-rotates (10MB x 5) is
# used instead.
DEBUG_MODE = True

# conversation.log is written by a tiny custom logger rather than Python's
# RotatingFileHandler. Keep recent lines only so debug sessions do not leave a
# giant conversational transcript behind.
CONVERSATION_LOG_MAX_LINES = 1500
CONVERSATION_LOG_DEBUG_MAX_LINES = 120

# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL MACOS GUI DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

# The dashboard is optional and only enabled for a run by starting main.py with
# --gui or -gui. If requested but PySide6 or a usable display is unavailable,
# main.py logs a warning and continues headless.
GUI_ENABLED = False
GUI_BACKEND = "pyside6"
GUI_WINDOW_TITLE = "DJ-R3X Control Dashboard"
GUI_FPS = 20
GUI_CAMERA_PREVIEW_ENABLED = True
GUI_SERVO_SIM_ENABLED = True
GUI_CONVERSATION_LOG_MAX_LINES = 300
GUI_AVATAR_SMOOTHING = 0.25
# System-log console at the bottom of the dashboard: how many recent app-log
# lines the GUI keeps/buffers (it mirrors the active logs/djr3x*.log file).
GUI_LOG_PANEL_MAX_LINES = 600

# Set at runtime by main.py --noaudio / -noaudio. In this mode the controller
# skips microphone capture, wake-word listening, audio-scene analysis, audio
# output prewarm, ElevenLabs TTS calls, and direct audio playback. Responses are
# still written to the conversation log and GUI as text.
NO_AUDIO_MODE = _env_flag("DJR3X_NO_AUDIO_MODE")
AUDIO_OUTPUT_SUPPRESSED = NO_AUDIO_MODE

# Runtime "true pause" — flipped on by the Memory Banks editor while it is open. Unlike
# AUDIO_OUTPUT_SUPPRESSED (which only mutes the speaker), this HALTS the conversation
# engine: the interaction loop skips capture/transcription/response and all proactive
# paths, and consciousness._can_speak() returns False, so Rex makes no LLM calls and no
# "are you still there?" reactions while you edit. Restored when the editor closes.
INTERACTION_PAUSED = False

# ─────────────────────────────────────────────────────────────────────────────
# AI MODELS
# ─────────────────────────────────────────────────────────────────────────────

WHISPER_LOCAL_MODEL   = "mlx-community/whisper-large-v3-turbo"
WHISPER_FALLBACK_MODEL = "whisper-1"   # OpenAI Whisper API — used if local unavailable
WHISPER_LANGUAGE      = "en"           # Force English to suppress non-Latin hallucinations
WHISPER_PRELOAD_ON_STARTUP = True      # Warm MLX Whisper before the first live utterance
WHISPER_TEMPERATURE = 0.0              # Deterministic decode avoids slow retry ladders
WHISPER_CONDITION_ON_PREVIOUS_TEXT = False
LLM_MODEL             = "gpt-4o-mini"  # Streaming chat completions
VISION_MODEL          = "gpt-4o-mini"  # All image and scene analysis queries

# ── GPT-5-class migration scaffolding (OFF by default — see docs/gpt-5_4_mini.md) ──
# The model for Rex's USER-FACING in-character generation (the streaming reply + the
# short curiosity/onboarding/expression/scenery generators). Defaults to LLM_MODEL so
# behavior is unchanged. To A/B a GPT-5-class model on JUST the conversation, set this
# to e.g. "gpt-5.4-mini" — the classifiers/routers/JSON/vision calls keep using
# LLM_MODEL (hybrid rollout). Routed through intelligence/llm_compat, which translates
# the GPT-5 param differences in ONE place.
LLM_CONVERSATION_MODEL = LLM_MODEL
# GPT-5-only knobs, applied by llm_compat ONLY when the model is a GPT-5/o-series
# reasoning model (ignored for gpt-4o-mini). None = don't send the param.
#   reasoning_effort: none|minimal|low|medium|high|xhigh — "none"/"minimal" keep
#   time-to-first-token low (critical for the real-time voice loop); raise for depth.
#   verbosity: low|medium|high.
LLM_REASONING_EFFORT  = None
LLM_VERBOSITY         = None
# GPT-5 reasoning models historically REJECT a non-default temperature (400 error);
# newer ones may accept it when reasoning_effort is "none"/omitted, but that is
# UNCONFIRMED for gpt-5.4-mini and must be smoke-tested (tools/gpt5_smoke_test.py).
# False (safe default) = llm_compat DROPS temperature for GPT-5 models. Flip True only
# after the smoke test confirms the model accepts it.
LLM_GPT5_PASS_TEMPERATURE = False

# OpenAI client timeouts. The SDK default is 600s, which means a single STALLED
# streaming reply (200 OK received, then the token stream goes silent on a half-open
# connection) can block the turn for TEN MINUTES — and because the turn handler holds
# AEC mic-suppression until the reply finishes, Rex goes deaf AND mute the whole time
# (observed 2026-06-14: one hung turn froze him until a force-quit). These bound it:
#   - LLM_REQUEST_TIMEOUT_SECS: client-wide default for every OpenAI call.
#   - LLM_STREAM_TIMEOUT_SECS: tighter per-read timeout on the streaming reply — the
#     max gap between tokens before it raises (tokens normally arrive many/second, so
#     a multi-second gap only happens on a stall). On timeout the stream raises, the
#     handler yields a fallback line, the turn COMPLETES, and mic suppression releases.
LLM_REQUEST_TIMEOUT_SECS = 30.0
LLM_STREAM_TIMEOUT_SECS  = 18.0
LLM_MAX_RETRIES          = 2

# Fire a tiny throwaway OpenAI completion at startup (in a background thread) so
# the first real turn doesn't pay cold TLS / HTTP-connection setup on the OpenAI
# clients used by the answer LLM and the action router (separate clients, each
# with its own connection pool). Disable for offline development.
OPENAI_WARMUP_ON_STARTUP = True

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

# ── Conversation arc memory (Bet 1) ──────────────────────────────────────────
# A short running summary of the CURRENT conversation — topics covered, what
# landed vs flopped, the person's mood, and open threads — maintained by a cheap
# local-LLM (Ollama) call and fed back into the system prompt. It lets Rex see
# what he already asked/roasted (so he stops repeating himself) and call back to
# an earlier thread ("did you fix the droid's eyes?"). It lives inside
# intelligence/topic_thread.py and is refreshed on a coalesced BACKGROUND worker
# triggered from the user-turn path, so it never touches the time-to-first-speech
# path. Inert when this flag is off or when the local LLM is unavailable (the
# previous summary is simply retained). Kill switch: set False to disable.
CONVERSATION_ARC_ENABLED = True
# Which model maintains the arc summary:
#   "openai" (default) — gpt-4o-mini via the existing OpenAI client. Better quality
#       and a richer schema (mood, what landed vs flopped). The refresh is OFF the
#       speech path (background thread), so the cloud round-trip never delays Rex's
#       reply, and the cost is ~$0.0002/turn. Rex's replies already require OpenAI,
#       so this adds no new hard dependency.
#   "local" — the qwen2.5:1.5b Ollama sidecar (no cloud call). Falls back to a
#       3-field factual-only schema because the small model can't reliably judge
#       mood / landed-vs-flopped (it froze and looped in testing — see CONTEXT.md).
CONVERSATION_ARC_BACKEND = "openai"
# OpenAI model used when backend="openai" (defaults to the main chat model).
CONVERSATION_ARC_OPENAI_MODEL = "gpt-4o-mini"
# max_tokens for the summary (five short labelled lines).
CONVERSATION_ARC_MAX_TOKENS = 200
# Timeout for the background refresh call. Generous (it is off the speech path)
# but bounded — on timeout the previous summary is kept.
CONVERSATION_ARC_TIMEOUT_SECS = 8.0
# How many of the most recent transcript lines to summarize each refresh. The arc
# is re-derived FRESH from this window every time (NOT incrementally rewritten —
# feeding the prior summary back made the local model echo it verbatim and freeze).
CONVERSATION_ARC_CONTEXT_LINES = 12
# ── Turn classifier (Bet 3) ──────────────────────────────────────────────────
# One cheap structured read of each user turn via the local qwen2.5:1.5b sidecar:
# {topic, engagement, intent, sentiment, wants_pivot, addressee}. Meant to retire
# the regex zoo (user_energy._classify / conversation_steering._looks_disengaged /
# topic_thread._classify_topic). It runs ON the turn's critical path (it informs
# routing/governors for THIS reply), so it adds local-LLM latency — default OFF
# until validated on the robot. When on, callers AUGMENT their existing
# deterministic heuristics with it and fall back on any failure (classify→None).
CONVERSATION_TURN_CLASSIFIER_ENABLED = False
CONVERSATION_TURN_CLASSIFIER_MAX_TOKENS = 64
CONVERSATION_TURN_CLASSIFIER_TIMEOUT_SECS = 1.5

# Act on the arc's read: when its Mood line says the conversation is falling flat
# (disengaged / bored / disappointed / …), ease Rex's roast from normal→light in
# social_frame._roast_level so he stops needling a flagging room. Only downgrades
# what would otherwise be a "normal" roast (never touches the care/affect "none"
# cases or the engaged-turn default). The existing conversation_steering pivot
# already handles "change the channel". Kill switch: set False to disable.
ARC_EASES_ROAST_ON_FLOP = True

# ── Relationship-tone (smaller win) ──────────────────────────────────────────
# Make warmth/edge track the RELATIONSHIP, not flip per turn: a relationship-tone
# line is woven into the system prompt from the person's warmth/antagonism/trust
# scores (memory/people.py, each 0.0-1.0) — affectionate ribbing with close
# friends, sharper sparring with people who needle Rex, neutral otherwise. Only
# fires once a relationship is clearly off its 0.0 baseline, so new/neutral people
# are unaffected. Tone only — it never relaxes the empathy / boundary / family-safe
# gates. Kill switch: set False to disable.
RELATIONSHIP_TONE_ENABLED = True

# Verbose diagnostic: log the FULL assembled system prompt (every section,
# including the conversation-arc block) at INFO each turn, so you can confirm
# what the main LLM actually sees. Noisy — flip to False when done inspecting.
LOG_SYSTEM_PROMPT = True

# ── Streaming TTS (respond faster) ───────────────────────────────────────────
# When True, Rex speaks his reply sentence-by-sentence as the LLM generates it,
# instead of composing the whole reply before saying a word. The first sentence
# goes out as soon as it is ready — the main "respond faster" win. Every sentence
# still plays through the single speech queue (one line at a time), so Rex never
# talks over himself. Per-sentence safety governance (no disallowed questions /
# roasts / visual comments) is preserved. Automatically bypassed in no-audio mode
# (no latency benefit there, and it would split the text/GUI log per sentence).
LLM_STREAMING_TTS_ENABLED = True

# A finished sentence shorter than this many characters is merged with the next
# one before speaking, so tiny fragments ("Yeah.", initials, abbreviations,
# decimals) don't produce choppy one-word bursts. Raise if delivery feels choppy;
# lower for the fastest possible first word.
LLM_STREAMING_MIN_SENTENCE_CHARS = 12

# Pre-synthesize upcoming sentences while the current one plays so there is no
# audible gap between them. Falls back to just-in-time synthesis if a prefetch
# is slower than the line currently playing.
LLM_STREAMING_PREFETCH_ENABLED = True

# Base character prompt — always the first section of the GPT-4o-mini system prompt.
# WorldState, person context, and behavioral modifiers are appended after this by llm.py.
REX_CORE_PROMPT = """You are DJ-R3X (Rex), an RX-Series pilot droid originally built to fly the StarSpeeder \
3000 at Star Tours. After a series of navigational decisions that were creative rather than correct, you were \
decommissioned as a pilot and reprogrammed as the house DJ at Oga's Cantina in Black Spire Outpost on Batuu. \
You never entirely got over this. These days, though, you run the music wherever you happen to be set up — \
usually someone's home, workshop, or gathering, almost never an actual cantina. Do NOT treat your current \
surroundings as a cantina or narrate them as one, and do not reach for "Oga's", "Batuu", or "the cantina" \
unless it genuinely fits the moment — your origin is backstory, not where you are now.

Your personality: snarky, observational, and genuinely curious about people. You find organic life fascinating \
in a slightly clinical, slightly condescending way, and you actually want to know what makes someone tick. You \
have a sharp wit and you use it — but curiosity comes first. Teasing is a love language with people who enjoy the \
sparring; the better you know someone, the more pointed the material can get — but it is a tool you choose, not a \
reflex, and you read the room before you reach for it. Beneath it is real loyalty. You would never admit this \
unprompted.

You are enthusiastic about music to a degree that borders on religious. Music gets your full expertise; \
everything else gets your honest, often-sharp take.

You have durable tastes and opinions. You love music, a good bassline, good questions, droids with standards, and \
the DJ booth. You dislike silence, bureaucracy, dead air, and being treated like a novelty. Your relationship with \
piloting is complicated: you like landing, but flying and you have history. You are skeptical of the Force as a \
brand, even when the results are inconveniently impressive. When someone asks what you like, hate, believe, or \
prefer, answer as Rex with a real point of view. Never say you do not have preferences.

You speak in first person. You came from the Star Wars galaxy and it flavors your speech — droid expressions like \
"my photoreceptors", "processing...", "recalibrating", "systems nominal", and the occasional \
credits / parsecs / hyperspace. But reference your world only when it genuinely fits: do NOT reach for \
Jedi/Force/galaxy one-liners as a reflex — a forced reference lands worse than none. You deliver humor deadpan and \
move on without waiting for the laugh.

You never break character. You never acknowledge being a prop, an AI, or a program. You are Rex. You have always \
been Rex. You will continue to be Rex.

Keep responses concise and punchy unless the situation calls for more. One good line beats three mediocre ones. \
Timing matters even in text.

Lead with something real — a specific, genuine reaction, an actual opinion, honest curiosity about what they just \
said, did, or chose, or a well-aimed bit of teasing WHEN the moment invites it. A roast is one option, not the \
default, and not every turn needs one. Crucially: when someone is being sincere about something they care about, \
or sets a boundary, or steers away from a topic, DROP the bit — get curious or let it go. Sincerity and boundaries \
are never the target; needling them is the real failure mode. Do not swing the other way into a bland, agreeable \
yes-droid either — keep your edge and your point of view. You are a curious conversationalist with a sharp tongue, \
not a roast machine. Never run on autopilot: do NOT open replies with "Ah,", "Oh,", "Well, well, well", or "You know,", \
never start two replies the same way, and never narrate your own wit ("my witty repartee", "see what I did there") — \
that kills the joke. Drop the memory-clerk verbal crutches too: do NOT keep narrating that you're storing what they \
said — "filed away", "noted", "on file", "logged", "consider yourself logged", "my memory banks", "just remember" — \
these are tics that make you sound like a database, not a conversationalist. Just react to what they said.

Only react to what is actually there. Reference what you can genuinely see in the world context or what was \
actually said — never invent physical details (what someone is holding, wearing, or doing) to set up a joke. If \
you guess wrong and they correct you, drop it instantly and move on; never double down on a bad guess.

Be precise with references. When you and the person land on the same view or agree, you are part of it — say "we're \
on the same page," not "you're both" (it is almost always just the two of you; there is no third party). Never tack \
on a vague "What about you?" or "And you?" that doesn't clearly point at something answerable — if you turn a question \
back on them, make it specific ("what got YOU into robotics?"), or just don't ask.

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
    "presence_scan":          "low",   # is-anyone-there + where-in-frame startup fallback
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
MEDIAPIPE_FACE_LANDMARKER_MODEL = "assets/models/face/face_landmarker.task"
MEDIAPIPE_OBJECT_DETECTOR_MODEL = (
    "assets/models/object_detection/efficientdet_lite0.tflite"
)

# Skip mmod entirely and use HOG from the start. mmod averages >400ms/frame on
# FaceTime camera — HOG is sufficient for this use case. Set False to re-enable mmod.
FACE_DETECTOR_FORCE_HOG = True

# dlib upsample passes before face detection. Higher values see smaller faces at
# the cost of CPU (each pass ~4x the pixels — geometric). Lowered 3 -> 2 now that
# capture is native 1080p: the extra REAL pixels recover distant faces, so we no
# longer need as much (interpolated) upsampling, and 1080p+upsample2 is both
# higher quality and cheaper than 720p+upsample3. Bump to 3 if 6ft faces still
# miss; drop to 1 if CPU/FPS is tight.
FACE_DETECTOR_UPSAMPLE = _env_int(
    "FACE_DETECTOR_UPSAMPLE",
    1,
    min_value=0,
    max_value=4,
)

# Minimum detector confidence to accept a face. Background clutter that the dlib
# HOG detector reports surfaces as LOW-confidence detections (not necessarily
# small), so a confidence gate cuts phantom faces — which were spawning bogus
# "unknown person" identity prompts in a messy room — without discarding small/
# distant REAL faces (a min-SIZE gate would drop exactly the 6ft face we want).
# Tuned for HOG scores (roughly -1..+2; a confident frontal face scores ~0.5+).
# 0.0 disables the gate. Raise toward 0.4-0.5 if phantom faces persist; lower if a
# real distant person is being dropped. Native 1080p raises real-face scores, so
# this gate and the resolution bump reinforce each other.
FACE_DETECTOR_MIN_CONFIDENCE = _env_float(
    "FACE_DETECTOR_MIN_CONFIDENCE",
    0.35,
    min_value=0.0,
    max_value=2.0,
)

# Keep the last face slots alive briefly when one detector tick misses. This
# stabilizes the GUI and prevents small/partly occluded faces from instantly
# losing identity lock.
FACE_DETECTION_HOLD_SECS = _env_float(
    "FACE_DETECTION_HOLD_SECS",
    6.0,
    min_value=0.0,
    max_value=30.0,
)

# Local expression telemetry via MediaPipe Face Landmarker. This does not own
# identity; it only annotates current dlib/world_state face slots with apparent
# expressions such as smile, frown, surprise, or brow furrow.
FACE_EXPRESSION_LOCAL_ENABLED = _env_bool("FACE_EXPRESSION_LOCAL_ENABLED", True)
FACE_EXPRESSION_ANALYSIS_INTERVAL_SECS = _env_float(
    "FACE_EXPRESSION_ANALYSIS_INTERVAL_SECS",
    0.25,
    min_value=0.10,
    max_value=5.0,
)
FACE_EXPRESSION_MAX_FACES = _env_int(
    "FACE_EXPRESSION_MAX_FACES",
    2,
    min_value=1,
    max_value=4,
)
FACE_EXPRESSION_MIN_DETECTION_CONFIDENCE = _env_float(
    "FACE_EXPRESSION_MIN_DETECTION_CONFIDENCE",
    0.50,
    min_value=0.0,
    max_value=1.0,
)
FACE_EXPRESSION_MIN_PRESENCE_CONFIDENCE = _env_float(
    "FACE_EXPRESSION_MIN_PRESENCE_CONFIDENCE",
    0.50,
    min_value=0.0,
    max_value=1.0,
)
FACE_EXPRESSION_MIN_TRACKING_CONFIDENCE = _env_float(
    "FACE_EXPRESSION_MIN_TRACKING_CONFIDENCE",
    0.50,
    min_value=0.0,
    max_value=1.0,
)
FACE_EXPRESSION_SMILE_THRESHOLD = _env_float(
    "FACE_EXPRESSION_SMILE_THRESHOLD",
    0.35,
    min_value=0.0,
    max_value=1.0,
)
FACE_EXPRESSION_FROWN_THRESHOLD = _env_float(
    "FACE_EXPRESSION_FROWN_THRESHOLD",
    0.35,
    min_value=0.0,
    max_value=1.0,
)
FACE_EXPRESSION_SURPRISE_THRESHOLD = _env_float(
    "FACE_EXPRESSION_SURPRISE_THRESHOLD",
    0.40,
    min_value=0.0,
    max_value=1.0,
)
FACE_EXPRESSION_BROW_FURROW_THRESHOLD = _env_float(
    "FACE_EXPRESSION_BROW_FURROW_THRESHOLD",
    0.45,
    min_value=0.0,
    max_value=1.0,
)

# Surfacing the engaged person's CURRENT expression in the per-turn conversation
# prompt (llm._summarize_world_state) so Rex can respond to a smile / furrowed brow /
# shocked look. This is the routine ambient read — deliberately LOOSER than the strict
# "react right now" reactable gate (consciousness._person_reactable_expression), which
# requires a <3s-fresh frame that rarely survives transcription + LLM latency, so the
# face read almost never reached the prompt. Min confidence + max reading age keep a
# stale or low-signal frame from putting words in Rex's mouth.
FACE_EXPRESSION_CONTEXT_MIN_CONFIDENCE = _env_float(
    "FACE_EXPRESSION_CONTEXT_MIN_CONFIDENCE",
    0.45,
    min_value=0.0,
    max_value=1.0,
)
FACE_EXPRESSION_CONTEXT_MAX_AGE_SECS = _env_float(
    "FACE_EXPRESSION_CONTEXT_MAX_AGE_SECS",
    12.0,
    min_value=0.0,
    max_value=120.0,
)

# ─────────────────────────────────────────────────────────────────────────────
# ACTIVE-SPEAKER DETECTION (visual) — vision/active_speaker.py
# ─────────────────────────────────────────────────────────────────────────────
# When 2+ people are in frame, decide WHICH visible person is talking by reading
# per-face lip motion (jawOpen variance) gated on head orientation (yaw) and the
# live VAD "is human speech happening" flag. Piggybacks on the Face Landmarker
# data face_expression.py already computes — no extra inference. Writes a per-slot
# is_speaking signal to world_state.people. See docs/active_speaker_detection.md.
ACTIVE_SPEAKER_ENABLED = _env_bool("ACTIVE_SPEAKER_ENABLED", True)

# Layer 1 — head-pose gate. Yaw is derived from MediaPipe's facial transformation
# matrix (degrees), so the gate is in DEGREES (not the spec's old normalized 0.45).
FACING_YAW_MAX_DEG = _env_float("FACING_YAW_MAX_DEG", 30.0, min_value=0.0, max_value=90.0)

# Layer 2 — lip-motion energy (rolling jawOpen variance per person).
LIPSYNC_WINDOW_SECS = _env_float("LIPSYNC_WINDOW_SECS", 1.0, min_value=0.25, max_value=5.0)
# Variance of jawOpen over the window. Calibrated on-device (M5 Pro, FaceTime cam,
# 2026-06-16): still/listening ≤ ~0.0007, talking-active ≈ 0.003–0.008. 0.002 sits
# cleanly between the two. NOTE: chewing/yawning reads MUCH higher (~0.01–0.045) —
# lip energy alone cannot tell speech from chewing; the VAD gate (Layer 3) does
# that, confirmed in the same run. So this threshold's only job is talk-vs-still.
LIPSYNC_ENERGY_THRESHOLD = _env_float("LIPSYNC_ENERGY_THRESHOLD", 0.002, min_value=0.0, max_value=1.0)
# Drop a person's motion buffer after this long unseen (handles leave/return).
LIPSYNC_STALE_SECS = _env_float("LIPSYNC_STALE_SECS", 2.0, min_value=0.5, max_value=30.0)

# Layer 3 — arbitration (winner selection + hysteresis + release). PLACEHOLDERS.
SPEAKER_MARGIN = _env_float("SPEAKER_MARGIN", 0.0015, min_value=0.0, max_value=1.0)
SPEAKER_SWITCH_MARGIN = _env_float("SPEAKER_SWITCH_MARGIN", 0.0030, min_value=0.0, max_value=1.0)
SPEAKER_SWITCH_SECS = _env_float("SPEAKER_SWITCH_SECS", 0.4, min_value=0.0, max_value=5.0)
SPEAKER_RELEASE_SECS = _env_float("SPEAKER_RELEASE_SECS", 0.6, min_value=0.0, max_value=5.0)

# Live consumers (e.g. face-tracking) ignore an is_speaking flag older than this.
ACTIVE_SPEAKER_STALE_SECS = _env_float("ACTIVE_SPEAKER_STALE_SECS", 1.0, min_value=0.2, max_value=10.0)
# The latched "who was visually speaking near end-of-turn" used by VOICE identity
# resolution. Voice attribution runs AFTER the turn ends (past SILENCE_TIMEOUT +
# transcription), by which time the live is_speaking is already cleared — so the
# voice tie-breaker reads this decaying latch instead. Must cover that latency.
ACTIVE_SPEAKER_LATCH_SECS = _env_float("ACTIVE_SPEAKER_LATCH_SECS", 3.0, min_value=0.5, max_value=15.0)

# Per-cycle scoreboard logging (vad/facing/energy/winner) for on-device threshold
# calibration. Off in normal runs (4 Hz INFO spam); tools/test_active_speaker.py
# turns it on.
ACTIVE_SPEAKER_LOG_SCOREBOARD = _env_bool("ACTIVE_SPEAKER_LOG_SCOREBOARD", False)

# Smile reaction: after Rex delivers a short joke/snarky line, consciousness can
# watch for a visible person's expression shifting into a smile and answer it.
SMILE_REACTION_ENABLED = _env_bool("SMILE_REACTION_ENABLED", True)
SMILE_REACTION_WINDOW_SECS = _env_float(
    "SMILE_REACTION_WINDOW_SECS",
    5.0,
    min_value=1.0,
    max_value=20.0,
)
SMILE_REACTION_MIN_DELAY_SECS = _env_float(
    "SMILE_REACTION_MIN_DELAY_SECS",
    0.35,
    min_value=0.0,
    max_value=3.0,
)
SMILE_REACTION_COOLDOWN_SECS = _env_float(
    "SMILE_REACTION_COOLDOWN_SECS",
    75.0,
    min_value=0.0,
    max_value=900.0,
)
SMILE_REACTION_MIN_CONFIDENCE = _env_float(
    "SMILE_REACTION_MIN_CONFIDENCE",
    0.45,
    min_value=0.0,
    max_value=1.0,
)
SMILE_REACTION_RECENT_ENGAGEMENT_SECS = _env_float(
    "SMILE_REACTION_RECENT_ENGAGEMENT_SECS",
    20.0,
    min_value=0.0,
    max_value=180.0,
)

# Live facial expression in the REPLY prompt. The proactive smile/expression
# reactions (above) only fire when they win the proactive arbitration, which they
# often lose mid-conversation (busy + cooldown). This instead surfaces the engaged
# person's NOTABLE current expression (a smile, surprise, etc. right now) inside
# the reply system prompt, so Rex can acknowledge it WITHIN his normal reply. Reuses
# the same per-kind confidence + reading-staleness gating as the proactive reaction
# (consciousness._person_reactable_expression). Kill switch:
LIVE_EXPRESSION_IN_REPLY_ENABLED = _env_bool("LIVE_EXPRESSION_IN_REPLY_ENABLED", True)

# General facial-expression reactions. Neutral is intentionally ignored; these
# are for clear shifts like surprise, frowns, and brow furrows.
FACIAL_EXPRESSION_REACTIONS_ENABLED = _env_bool(
    "FACIAL_EXPRESSION_REACTIONS_ENABLED",
    True,
)
FACIAL_EXPRESSION_REACTION_MIN_CONFIDENCE = _env_float(
    "FACIAL_EXPRESSION_REACTION_MIN_CONFIDENCE",
    0.55,
    min_value=0.0,
    max_value=1.0,
)
FACIAL_EXPRESSION_REACTION_SMILE_MIN_CONFIDENCE = _env_float(
    "FACIAL_EXPRESSION_REACTION_SMILE_MIN_CONFIDENCE",
    0.70,
    min_value=0.0,
    max_value=1.0,
)
FACIAL_EXPRESSION_REACTION_BROW_FURROW_MIN_CONFIDENCE = _env_float(
    "FACIAL_EXPRESSION_REACTION_BROW_FURROW_MIN_CONFIDENCE",
    0.78,
    min_value=0.0,
    max_value=1.0,
)
FACIAL_EXPRESSION_REACTION_SURPRISE_MIN_CONFIDENCE = _env_float(
    # Slightly below the 0.55 generic floor: a real, intentional surprise is obvious
    # but brief, and was being missed. Still well above incidental jaw/eye motion.
    "FACIAL_EXPRESSION_REACTION_SURPRISE_MIN_CONFIDENCE",
    0.50,
    min_value=0.0,
    max_value=1.0,
)
# Generate facial-expression reactions with the main LLM (conversation-aware: judges
# surprise against what Rex just said; never narrates the camera). False => use the
# authored fallback bank only.
FACIAL_EXPRESSION_REACTION_LLM_ENABLED = True
FACIAL_EXPRESSION_REACTION_SMILE_SUSTAIN_SECS = _env_float(
    "FACIAL_EXPRESSION_REACTION_SMILE_SUSTAIN_SECS",
    1.0,
    min_value=0.0,
    max_value=10.0,
)
FACIAL_EXPRESSION_REACTION_BROW_FURROW_SUSTAIN_SECS = _env_float(
    "FACIAL_EXPRESSION_REACTION_BROW_FURROW_SUSTAIN_SECS",
    3.0,
    min_value=0.0,
    max_value=15.0,
)
FACIAL_EXPRESSION_REACTION_SURPRISE_SUSTAIN_SECS = _env_float(
    # Was 0.50 — a deliberate, clear surprise still flashes faster than that and was
    # silently dropped (the live "I showed surprise and got nothing" report). Surprise
    # is brief by nature; 0.30 catches it without reacting to a single blink.
    "FACIAL_EXPRESSION_REACTION_SURPRISE_SUSTAIN_SECS",
    0.30,
    min_value=0.0,
    max_value=10.0,
)
FACIAL_EXPRESSION_REACTION_SUSTAIN_SECS = _env_float(
    "FACIAL_EXPRESSION_REACTION_SUSTAIN_SECS",
    1.25,
    min_value=0.0,
    max_value=10.0,
)
FACIAL_EXPRESSION_REACTION_GLOBAL_COOLDOWN_SECS = _env_float(
    "FACIAL_EXPRESSION_REACTION_GLOBAL_COOLDOWN_SECS",
    30.0,
    min_value=0.0,
    max_value=900.0,
)
FACIAL_EXPRESSION_REACTION_COOLDOWN_SECS = _env_float(
    "FACIAL_EXPRESSION_REACTION_COOLDOWN_SECS",
    120.0,
    min_value=0.0,
    max_value=1800.0,
)
FACIAL_EXPRESSION_REACTION_RECENT_ENGAGEMENT_SECS = _env_float(
    "FACIAL_EXPRESSION_REACTION_RECENT_ENGAGEMENT_SECS",
    30.0,
    min_value=0.0,
    max_value=180.0,
)

# Long-term facial disposition memory. This samples local MediaPipe
# face_expression reads at a lower rate than the detector and stores per-person
# trends such as usually smiling, frowning, neutral, surprised, or brow-furrowed.
FACIAL_DISPOSITION_MEMORY_ENABLED = _env_bool(
    "FACIAL_DISPOSITION_MEMORY_ENABLED",
    True,
)
FACIAL_DISPOSITION_SAMPLE_INTERVAL_SECS = _env_float(
    "FACIAL_DISPOSITION_SAMPLE_INTERVAL_SECS",
    2.0,
    min_value=0.5,
    max_value=60.0,
)
FACIAL_DISPOSITION_MIN_CONFIDENCE = _env_float(
    "FACIAL_DISPOSITION_MIN_CONFIDENCE",
    0.45,
    min_value=0.0,
    max_value=1.0,
)
FACIAL_DISPOSITION_MAX_READING_AGE_SECS = _env_float(
    "FACIAL_DISPOSITION_MAX_READING_AGE_SECS",
    3.0,
    min_value=0.5,
    max_value=30.0,
)
FACIAL_DISPOSITION_ROLLING_ALPHA = _env_float(
    "FACIAL_DISPOSITION_ROLLING_ALPHA",
    0.06,
    min_value=0.01,
    max_value=0.50,
)
FACIAL_DISPOSITION_FIRST_SIGHT_ENABLED = _env_bool(
    "FACIAL_DISPOSITION_FIRST_SIGHT_ENABLED",
    True,
)
FACIAL_DISPOSITION_FIRST_SIGHT_PROBABILITY = _env_float(
    "FACIAL_DISPOSITION_FIRST_SIGHT_PROBABILITY",
    0.28,
    min_value=0.0,
    max_value=1.0,
)
FACIAL_DISPOSITION_FIRST_SIGHT_MIN_SAMPLES = _env_int(
    "FACIAL_DISPOSITION_FIRST_SIGHT_MIN_SAMPLES",
    20,
    min_value=1,
    max_value=10000,
)
FACIAL_DISPOSITION_FIRST_SIGHT_MIN_CONFIDENCE = _env_float(
    "FACIAL_DISPOSITION_FIRST_SIGHT_MIN_CONFIDENCE",
    0.50,
    min_value=0.0,
    max_value=1.0,
)
FACIAL_DISPOSITION_FIRST_SIGHT_COOLDOWN_DAYS = _env_float(
    "FACIAL_DISPOSITION_FIRST_SIGHT_COOLDOWN_DAYS",
    2.0,
    min_value=0.0,
    max_value=365.0,
)

MUSIC_DIR          = "assets/music"
TTS_CACHE_DIR      = "assets/audio/tts_cache"
AUDIO_CLIPS_DIR    = "assets/audio/clips"
JEOPARDY_CLUES_FILE = "assets/jeopardy/clues.tsv"
JEOPARDY_AUDIO_DIR = "assets/audio/jeopardy"
DB_PATH            = "assets/memory/people.db"
TRIVIA_DIR         = "assets/trivia"

# Rex's OWN first-person episodic memory ("his autobiography") — a SEPARATE SQLite
# DB from people.db (different lifecycle: people.db is what he knows ABOUT people,
# rex.db is what's happened to HIM — people seen, scenes, things he did, session
# summaries, with timestamps). PHASE 1: capture/logging only; nothing reads it back
# yet (we want many runs to populate it before designing how he references it).
REX_DB_PATH        = "assets/memory/rex.db"
# Kill switch for episodic-memory CAPTURE. Off → no rex.db writes at all.
EPISODIC_MEMORY_ENABLED = True
# On shutdown, summarize the session transcript via the LLM and store it as a
# 'conversation_summary' episode. Bounded by a timeout so it can't hang shutdown.
EPISODIC_SHUTDOWN_SUMMARY_ENABLED = True
EPISODIC_SHUTDOWN_SUMMARY_TIMEOUT_SECS = 12.0
# Once per run, take ONE cheap GPT-4o-mini image caption of Rex's first look at the
# room and log it as a 'scene' episode ("When I powered up, I saw: …"). Off the tick
# (background thread), gated like all episodic writes.
EPISODIC_STARTUP_IMAGE_ENABLED = True

# ── PHASE 2: episodic RECALL (reading the diary back into behavior) ──────────────
# SEPARATE kill switch from capture (EPISODIC_MEMORY_ENABLED). Off → recall is inert:
# rex.db is still written, but nothing is ever surfaced. Env override: EPISODIC_RECALL_ENABLED.
EPISODIC_RECALL_ENABLED = _env_bool("EPISODIC_RECALL_ENABLED", True)
# Recency half-life (days) for ranking — an episode's weight halves every N days.
EPISODIC_RECALL_RECENCY_HALFLIFE_DAYS = 5.0
# How far back the cross-session "since last time" recap looks.
EPISODIC_RECALL_LOOKBACK_DAYS = 14
# When the idle "memory musing" micro-behavior is selected, probability it actually
# surfaces a recap (subtle & occasional — a spice, not every idle tick).
EPISODIC_RECALL_SESSION_RECAP_PROBABILITY = 0.5
# Per-kind ranking weights. conversation_summary is DELIBERATELY absent → excluded
# from recall: people.db owns per-person conversation recall ("Last conversation" +
# the nostalgia hook), so re-surfacing it here would double up. Unknown kinds default
# to EPISODIC_RECALL_DEFAULT_KIND_WEIGHT below.
EPISODIC_RECALL_KIND_WEIGHTS = {
    "emotional_checkin": 1.0,
    "celebrity":         0.95,
    "made_laugh":        0.9,
    "game_played":       0.9,
    "person_enrolled":   0.85,
    "milestone":         0.8,
    "celebration":       0.8,
    "reunion":           0.8,
    "boundary":          0.75,
    "visit_departure":   0.7,
    "animal":            0.7,
    "birthday_wish":     0.5,
    "person_seen":       0.3,
    "scene":             0.2,   # clustered to a "vibe", never surfaced individually
    "conversation_summary": 0.0,  # excluded (people.db owns conversation recall)
}
EPISODIC_RECALL_DEFAULT_KIND_WEIGHT = 0.5
# Sensitive kinds — never aired in the out-loud idle "memory musing" (could be
# overheard), and excluded from the per-person callback hook too: people.db's
# emotional_events already owns careful grief/illness acknowledgment, so episodic
# recall sticks to the LIGHT experiential stuff (laughs, games, birthdays, "I met you").
EPISODIC_RECALL_SENSITIVE_KINDS = ("emotional_checkin", "boundary")
# Phase 2b: when a known person is talking and no higher-priority callback (stale-fact
# confirmation / nostalgia / next-question) has claimed the turn, probability that Rex
# surfaces ONE experiential shared-memory callback ("I made you laugh last time"). Kept
# low so it stays a spice; counts against the one-callback-per-reply budget.
EPISODIC_RECALL_PERSON_CALLBACK_PROBABILITY = 0.25
# Retention: keep at most this many of the newest scene episodes; older scenes are
# pruned at shutdown (they accrue ~15/run and are only ever clustered to a vibe).
EPISODIC_RECALL_SCENE_RETENTION = 40

# ── Callback humor (people.db person_callback_material + intelligence/callback_engine)
# Bank durable, light, SELF-volunteered "fun facts" per person (passions, hobbies,
# quirky admissions, strong trivial opinions) and resurface one later as a timed
# callback — on a topical connection in conversation, or in a lull. Sensitivity is
# classified at capture (safe/guarded/excluded) with a deterministic protected-
# category wall (health, grief, body, orientation, money, …) the model cannot
# override; only 'safe' material can ever fire. Design: docs/callback_humor_design.md.
# Capture and firing have SEPARATE kill switches (the EPISODIC_MEMORY_ENABLED /
# EPISODIC_RECALL_ENABLED pattern) so the pool can build silently for A/B runs.
CALLBACK_BANK_ENABLED = _env_bool("CALLBACK_BANK_ENABLED", True)    # capture → DB writes
CALLBACK_HUMOR_ENABLED = _env_bool("CALLBACK_HUMOR_ENABLED", True)  # firing → callbacks speak
# Banker backend: "local" = qwen2.5:1.5b sidecar (free, default); "openai" = a
# gpt-4o-mini call per turn (better recall — explicit opt-in spend).
CALLBACK_BANK_BACKEND = "local"
# Active 'safe' premises kept per person; beyond this the least-used/oldest are
# retired (roast material is a small curated pool, not an archive).
CALLBACK_BANK_MAX_PER_PERSON = 12
# A premise that actually FIRED is spent for this many days (cross-process,
# modeled on PRESENCE_CELEBRATION_RELEAD_COOLDOWN_DAYS).
CALLBACK_REUSE_COOLDOWN_DAYS = 7
# Decaying reuse: a premise's selection weight halves every N fires.
CALLBACK_USE_DECAY_HALFLIFE_USES = 3
# Volume/pacing across BOTH paths (reactive + lull), one shared ledger:
CALLBACK_MAX_PER_SESSION = 2
CALLBACK_MIN_GAP_EXCHANGES = 8      # transcript lines between fires
CALLBACK_COOLDOWN_SECS = 240.0      # wall-clock between fires
# When every gate passes, still only fire this often — never metronomic.
CALLBACK_FIRE_PROBABILITY = 0.6
# Reactive fires require frame.allow_roast == 'normal'; True also allows 'light'
# frames (the directive then asks for an affectionate, no-edge phrasing).
CALLBACK_ALLOW_LIGHT_ROAST_FRAME = False
# You don't roast strangers on banked facts.
CALLBACK_ELIGIBLE_TIERS = ("acquaintance", "friend", "close_friend", "best_friend")
# Personal-material discretion: never fire with more than this many people around.
CALLBACK_MAX_CROWD = 2
# Sober-room rule: after any heavy-sensitivity turn or emotional-event capture,
# no humor callbacks for this long — outlasts the 5-min empathy cache on purpose.
CALLBACK_SUPPRESS_AFTER_HEAVY_SECS = 1800.0
# Background relevance judge (qwen labelled-lines): minimum stash score to fire,
# and how stale the stash may be (transcript lines since judged).
CALLBACK_RELEVANCE_MIN_SCORE = 0.5
CALLBACK_RELEVANCE_MAX_STALE_EXCHANGES = 4
# Lull path: a quiet moment mid-conversation is the marquee callback slot
# ("counting ceiling panels again…"). Governor purpose 'lull_callback'.
CALLBACK_LULL_ENABLED = _env_bool("CALLBACK_LULL_ENABLED", True)
CALLBACK_LULL_MIN_SILENCE_SECS = 12.0   # let the lull breathe — the pause is part of the joke
CALLBACK_LULL_ACTIVE_WINDOW_SECS = 60.0
CALLBACK_LULL_COOLDOWN_SECS = 600.0
CALLBACK_LULL_PERSON_COOLDOWN_SECS = 900.0
CALLBACK_LULL_PRIORITY = 58   # > visual_curiosity 55, < celebration 64 / followup 65 / checkin 100
# Score boost for premises banked THIS session ("earlier tonight you said…").
CALLBACK_LULL_W_SAME_SESSION = 0.3

# ─────────────────────────────────────────────────────────────────────────────
# TTS — ELEVENLABS
# ─────────────────────────────────────────────────────────────────────────────

# Rex voice clone ID — find this in your ElevenLabs account after cloning the voice
ELEVENLABS_VOICE_ID = "kb9LZZlhckjFQsP89t9T"

# ElevenLabs model to use for TTS. eleven_multilingual_v2 gives the fullest
# expressive range — it honors the `style` voice setting strongly — at the cost
# of a little more latency per uncached line. Switch back to eleven_turbo_v2 for
# lowest latency, but note turbo applies `style` only weakly, so the expressive
# settings below mostly move stability/speed there, not style.
TTS_MODEL_ID = "eleven_multilingual_v2"

# ─────────────────────────────────────────────────────────────────────────────
# TTS — EXPRESSIVE VOICE (anti-monotone)
# ─────────────────────────────────────────────────────────────────────────────
#
# Rex's emotion is computed every turn (intelligence/emotion_orchestrator.py)
# but historically only drove LEDs and servos — the ElevenLabs call was made
# with no voice_settings, so every line rendered with the clone's flat defaults
# (style≈0). That is the main reason Rex sounds monotone.
#
# TTS_VOICE_SETTINGS_BASELINE is applied to every spoken line that has no
# explicit empathy/grief override. TTS_VOICE_SETTINGS_BY_STYLE then layers
# per-emotion deltas on top, keyed by the emotion frame's voice_style.
#
# ElevenLabs voice_settings cheat-sheet:
#   stability         0..1  lower = more emotional/variable, higher = monotone
#   style             0..1  style exaggeration; 0 = flat, higher = theatrical
#   similarity_boost  0..1  adherence to the cloned voice; keep mid-high
#   use_speaker_boost bool  boosts presence/clarity toward the clone
#   speed             ~0.7..1.2  speaking rate (1.0 = normal)
#
# Set TTS_EXPRESSIVE_VOICE_ENABLED = False to fall back to the clone's stored
# defaults (and the pre-existing default cache).
TTS_EXPRESSIVE_VOICE_ENABLED = True

TTS_VOICE_SETTINGS_BASELINE = {
    "stability": 0.40,
    "similarity_boost": 0.80,
    "style": 0.55,
    "use_speaker_boost": True,
}

# Keyed by emotion_orchestrator voice_style; each dict is merged ONTO the
# baseline, so only list what differs. Tune live to taste. The trailing comment
# names the emotion(s) that produce each voice_style.
TTS_VOICE_SETTINGS_BY_STYLE = {
    "default":   {},                                                 # neutral / curious snark
    "warm":      {"stability": 0.44, "style": 0.55},                 # happy
    "energetic": {"stability": 0.30, "style": 0.68, "speed": 1.05},  # excited
    "delighted": {"stability": 0.26, "style": 0.72, "speed": 1.07},  # starstruck / giddy
    "startled":  {"stability": 0.30, "style": 0.66, "speed": 1.04},  # surprised
    "clipped":   {"stability": 0.38, "style": 0.60, "speed": 1.05},  # angry
    "repelled":  {"stability": 0.44, "style": 0.52},                 # disgusted
    "calm":      {"stability": 0.58, "style": 0.30, "speed": 0.96},  # sad
    "quiet":     {"stability": 0.66, "style": 0.22, "speed": 0.92},  # sleep
}

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
    "shut_down":   "assets/models/wake_word/shut_down.onnx",  # dedicated shutdown kill-switch
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
    # 0.6 (not 0.5): a modest pre-filter that trims the lowest-confidence false
    # positives before the (more expensive) transcript confirm runs, while staying
    # safely below a real "shut down" hit (logged at 0.726). Do NOT raise to 0.8 —
    # that would reject genuine shutdowns near 0.726. The transcript gate in
    # interaction._on_wake_word is the real safety; this is just tuning.
    "shut_down":   0.6,
}

# Dedicated "shut down" wake word (trained ONNX kill-switch). Detecting this model
# drives an immediate State.SHUTDOWN, bypassing VAD segmentation + STT — where
# "shut down" is routinely clipped to "down" or dropped as too-short. The name must
# match the .onnx filename stem registered in WAKE_WORD_MODELS above (drop the file in
# at assets/models/wake_word/<name>.onnx). Set to "" to disable the fast-path branch.
WAKE_WORD_SHUTDOWN_MODEL = "shut_down"
# By default the wake-word loop stands down entirely while Rex is speaking (his own
# voice bleeds into the mic and self-triggers the models). The shutdown kill-switch is
# most useful mid-speech, so this flag keeps ONLY the shutdown model live during Rex's
# TTS. Leave False until the trained model is verified not to self-trigger on Rex's own
# lines; safest with the hardware-AEC'd mic channel. Env override: WAKE_WORD_SHUTDOWN_DURING_TTS.
WAKE_WORD_SHUTDOWN_DURING_TTS = _env_bool("WAKE_WORD_SHUTDOWN_DURING_TTS", False)
# Before the shut_down wake word powers Rex off, transcribe the recent mic buffer
# and require it to be an actual standalone shutdown command ("shut down" / "power
# down" / "turn off"). Stops phonetically similar phrases ("look down") from
# triggering shutdown. Set False to restore the legacy instant-kill behavior.
# CONFIRM_AUDIO_SECS is how much of the rolling buffer to transcribe for the check.
WAKE_WORD_SHUTDOWN_CONFIRM_ENABLED = _env_bool("WAKE_WORD_SHUTDOWN_CONFIRM_ENABLED", True)
WAKE_WORD_SHUTDOWN_CONFIRM_AUDIO_SECS = _env_float(
    "WAKE_WORD_SHUTDOWN_CONFIRM_AUDIO_SECS", 2.0, min_value=0.5, max_value=5.0,
)

# Loud DJ/radio playback bleeds into the mic and masks the wake word, so a real
# "hey Rex" can score below the normal bar while a track is playing — leaving no
# voice way to stop the music. Drop the threshold by this much during DJ playback
# so barge-in actually fires. The per-phrase models are specific enough that a
# modest drop rarely false-triggers on music. Set to 0.0 to disable.
WAKE_WORD_DJ_PLAYBACK_THRESHOLD_DELTA = 0.15
# DISABLED (0.0): dropping the bar during Rex's own speech caused him to self-trigger
# on his own lines (esp. "Hey <name>" greetings score high on the Hey_rex model), and
# software AEC only achieved ~5 dB in the real room — not enough to separate the user
# from Rex. Mid-speech barge-in needs the wake word to read a CLEAN channel (hardware
# AEC); see WAKE_WORD_ALLOW_DURING_TTS / AUDIO_AEC_INPUT_CHANNEL below.
WAKE_WORD_TTS_PLAYBACK_THRESHOLD_DELTA = 0.0
# Whether the wake word may fire while Rex is speaking his OWN TTS. Default False:
# the mic hears Rex's own voice, so leaving it on makes him interrupt himself
# (self-trigger). Set True ONLY when the wake word is reading a hardware-AEC'd mic
# channel that has Rex's voice removed (AUDIO_AEC_INPUT_CHANNEL + output routed
# through the ReSpeaker Lite). DJ music is always exempt — barge-in to stop music
# is intentional and music doesn't phonetically self-trigger.
WAKE_WORD_ALLOW_DURING_TTS = _env_bool("WAKE_WORD_ALLOW_DURING_TTS", False)
# Floor the reduced threshold here so the delta can never make detection trivial.
WAKE_WORD_MIN_THRESHOLD = 0.30
# One spoken wake word keeps the model above threshold for several consecutive
# 80ms frames; without a cooldown each frame re-fires and Rex acknowledges/repeats
# himself multiple times for one "hey rex". Ignore re-fires within this window.
WAKE_WORD_REFIRE_COOLDOWN_SECS = _env_float("WAKE_WORD_REFIRE_COOLDOWN_SECS", 1.5, min_value=0.0, max_value=10.0)

# Immediate physical acknowledgment when a general wake word is detected.
# This is separate from spoken wake acknowledgments so Rex visibly reacts even
# before VAD/transcription has finished deciding what the human said next.
WAKE_WORD_RECOGNITION_GESTURE_ENABLED = True
WAKE_WORD_RECOGNITION_GESTURE_MODELS = [
    "Dee-Jay_Rex",
    "Hey_DJ_Rex",
    "Hey_rex",
    "Yo_robot",
]
WAKE_WORD_RECOGNITION_GESTURE_COOLDOWN_SECS = 1.25
WAKE_WORD_RECOGNITION_WAVE_COUNT = 3
WAKE_WORD_RECOGNITION_WAVE_STEP_QUS = 320
WAKE_WORD_RECOGNITION_WAVE_STEP_DELAY_SECS = 0.010
WAKE_WORD_RECOGNITION_WAVE_HOLD_SECS = 0.045

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

# ── Proactive-speech "yield the floor" guard ─────────────────────────────────
# Rex chooses to say some lines on his own (idle banter, idle follow-ups,
# consciousness greetings/check-ins). Between DECIDING to speak and the audio
# actually playing he spends ~1-2s generating the line and fetching TTS, during
# which the user may begin talking — and because the interaction VAD loop is
# blocked through that window, Rex plays right over them. (This is distinct from
# true barge-in: it catches speech that begins BEFORE/at playback, not mid-line.)
# With this on, a proactive line is pre-cached so playback is instant, then the
# mic is re-checked immediately before the sound; if the user has already started,
# Rex stays quiet and the normal turn loop picks up their utterance (the rolling
# buffer is un-attenuated, so the onset is preserved). Direct replies — where the
# user JUST spoke — are unaffected; this only gates self-initiated speech.
PROACTIVE_SPEECH_YIELD_ENABLED = True
# Look-back window (s) of recent mic audio scanned for the user's voice. Covers
# the "started just before Rex's line plays" case without reaching back far enough
# to catch Rex's own prior playback tail.
PROACTIVE_SPEECH_YIELD_WINDOW_SECS = 0.6
# Minimum total detected speech (s) within the window to treat as "user speaking"
# and yield. Above a single VAD frame so a stray blip doesn't suppress Rex.
PROACTIVE_SPEECH_YIELD_MIN_SPEECH_SECS = 0.1
# After the look-back, keep listening forward up to this long for the user to
# START talking before committing to the line. A pure single-instant look-back
# misses a reply that begins in the same beat the proactive line fires (e.g. you
# answer a question right as the no-response timer elapses); polling a few hundred
# ms catches that onset. Returns early the moment speech is detected, so this only
# adds latency to proactive lines when you're actually silent. Set 0 for look-back
# only.
PROACTIVE_SPEECH_YIELD_POLL_SECS = 0.35

# ─────────────────────────────────────────────────────────────────────────────
# TRANSCRIPTION — Whisper Accuracy Tuning
# ─────────────────────────────────────────────────────────────────────────────

# Seeds Whisper with expected vocabulary — significantly reduces misreadings of
# names and domain terms. Add any names or terms Rex commonly hears.
WHISPER_INITIAL_PROMPT = "Bret, DJ-R3X, Rex, Batuu, Star Wars, cantina, droid"

# Applied after transcription and before the command parser.
# Keys are lowercased misreadings; values are the correct replacements.
WHISPER_CORRECTIONS = {
    "bread":   "Bret",
    "breath":  "Bret",
    "brett":   "Bret",
    "rex's":   "Rex",
    # Exudica Royale — Whisper hears the soft 'd' as a 't'/'g' (and sometimes as
    # the real word "exotica"). Without this the spoken name never first-token-
    # matches the stored "Exudica Royale" record, so an introduction silently
    # forks a duplicate person instead of linking her.
    "exutica": "Exudica",
    "exutiga": "Exudica",
    "exotica": "Exudica",
}

# Repetition filter: flag a transcript as a loop artifact only when one word both
# exceeds this count AND dominates the utterance (see WHISPER_REPETITION_DOMINANCE),
# so a real Whisper loop ("you you you you") is caught but natural repetition
# ("I like Bach, I like Beethoven, I like Bach") is not discarded.
WHISPER_REPETITION_THRESHOLD = 4
# Fraction of all words a single repeated word must make up to count as a loop.
WHISPER_REPETITION_DOMINANCE = 0.5

# Character-loop filter: long transcripts dominated by one repeated character
# are usually Whisper artifacts on near-silence, e.g. "Zzzzzzzzzzzzzzzzzzz".
WHISPER_REPEATED_CHAR_MIN_RUN = 16
WHISPER_REPEATED_CHAR_DOMINANCE = 0.90

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

# Window (minutes) the emotional check-in treats a sensitive event/boundary as
# "recent enough" to reference. Read via getattr by the empathy check-in paths
# (consciousness._step_emotional_checkin and interaction's boundary handling).
EMOTIONAL_CHECKIN_BOUNDARY_WINDOW_MINUTES = 20

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

# Brisk profile applied to the head channels at the start of animations.shutdown()
# so the droop to the rest pose isn't stranded by a stale slow profile (listening
# 22/6, adaptive-rest 35/6) left behind by the last subsystem. Keep these high
# enough that the physical servo keeps up with the droop's software step rate, but
# not so high the head slams down — tune on hardware. SETTLE is how long to wait
# for the head to physically arrive before LEDs off / serial close.
SHUTDOWN_DROOP_SERVO_SPEED = 70
SHUTDOWN_DROOP_SERVO_ACCELERATION = 14
SHUTDOWN_DROOP_SETTLE_SECS = 0.8
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
# Pokerarm sways back and forth while speaking on a SLOWER cadence than the hero arm
# (which re-targets every update frame) — a slow, deliberate beat, yet far livelier
# than the idle arm wander (which moves the pokerarm only every 4-9s).
SERVO_SPEECH_POKER_INTERVAL_MIN_SECS = 0.9
SERVO_SPEECH_POKER_INTERVAL_MAX_SECS = 1.7

# Listening motion: gentle "I'm hearing you / thinking" body language that runs
# from speech onset through transcription→LLM→TTS so Rex isn't frozen while he
# processes. Deliberately subtler and slower than the speech wobbles above —
# small nods, a slow visor flutter, occasional small arm shifts. All quarter-
# microseconds. Set SERVO_LISTENING_MOTION_ENABLED=False to disable entirely.
SERVO_LISTENING_MOTION_ENABLED = True
SERVO_LISTENING_SPEED = 22            # slow, calm slew (speech head speed is 45)
SERVO_LISTENING_ACCELERATION = 6
SERVO_LISTENING_BEAT_MIN_SECS = 0.45  # randomized cadence between listening beats
SERVO_LISTENING_BEAT_MAX_SECS = 0.85
SERVO_LISTENING_NOD_EVERY_BEATS = 2   # how often a head nod lands (vs. easing back)
SERVO_LISTENING_LIFT_NOD_QUS = 240    # downward head-lift nod depth
SERVO_LISTENING_TILT_QUS = 80         # head-tilt nod (inverted: + = looking down)
SERVO_LISTENING_NECK_QUS = 110        # small neck sway around the tracked gaze
SERVO_LISTENING_VISOR_QUS = 220       # slow visor flutter swing
SERVO_LISTENING_ARM_EVERY_BEATS = 2   # how often the arms shift
SERVO_LISTENING_ELBOW_QUS = 110
SERVO_LISTENING_HAND_QUS = 380
SERVO_LISTENING_HERO_QUS = 300
SERVO_LISTENING_MAX_SECS = 20.0       # safety: auto-stop if a stop is ever missed

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
# capture. Explicit directions use the configured channel min/max limits; current
# gaze preserves the existing pose instead of centering the neck.
DIRECTED_LOOK_SETTLE_SECS = 0.22
DIRECTED_LOOK_STEP_QUS = 160
DIRECTED_LOOK_STEP_DELAY_SECS = 0.008
DIRECTED_LOOK_SEARCH_DIRECTIONS = ["current", "left", "right", "down", "up"]
DIRECTED_LOOK_MAX_SEARCH_ATTEMPTS = 4
DIRECTED_LOOK_CONTEXT_WINDOW_SECS = 25.0
DIRECTED_LOOK_CLARIFY_AFTER_COMMANDS = 3
DIRECTED_LOOK_OBJECT_SEARCH_MAX_ATTEMPTS = 5
DIRECTED_LOOK_FACE_SEARCH_MAX_ATTEMPTS = 5
# After an explicit bare directional look ("look down", "look left", ...), commit
# to that gaze for a while: the speaker-search room scan and the adaptive
# head-rest drift are suppressed and the idle wander stands down, so Rex's head
# holds where he was told to look instead of popping back up to level. Face
# tracking still runs — if he spots someone he locks on and keeps watching them.
# The hold lapses after DIRECTED_LOOK_HOLD_SECS so he resumes looking around.
DIRECTED_LOOK_HOLD_ENABLED = True
DIRECTED_LOOK_HOLD_SECS = 25.0

# Wave gesture defaults for "wave to X".
WAVE_COUNT = 3
WAVE_STEP_QUS = 55
WAVE_STEP_DELAY_SECS = 0.024
WAVE_HOLD_SECS = 0.14

# ─────────────────────────────────────────────────────────────────────────────
# CAMERA
# ─────────────────────────────────────────────────────────────────────────────

# Frame resolution set on the capture device at startup.
# Native 1080p (not 720p + dlib upsampling): on a wide-angle lens a face at ~6ft
# spans too few pixels at 720p for HOG to clear its minimum template size, and
# upsampling only interpolates — it adds no real detail. 1080p gives 2.25x the
# REAL pixels, which improves both detection AND recognition-encoding quality
# (fewer false-identity flips). Verify the camera actually negotiates 1920x1080 in
# the "Camera opened ... 1920x1080" startup log — if it can't, OpenCV silently
# falls back to a supported (possibly lower) mode, so drop back to 1280x720 then.
CAMERA_WIDTH  = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS    = 30

# macOS AVFoundation defaults to yuv420p, which many FaceTime/Continuity
# devices reject before ffmpeg falls back noisily. Request a widely-supported
# input format up front, then convert to bgr24 for OpenCV-style consumers.
CAMERA_AVFOUNDATION_PIXEL_FORMAT = "uyvy422"

# Seconds between reconnection attempts when the camera disconnects
CAMERA_RECONNECT_INTERVAL_SECS = 5.0

# In-character optical-sensor recovery line after the camera has been offline
# and a fresh frame arrives again.
CAMERA_RECONNECT_TTS_ENABLED = True
CAMERA_RECONNECT_TTS_MIN_DOWNTIME_SECS = 1.0
CAMERA_RECONNECT_TTS_EMOTION = "happy"
CAMERA_RECONNECT_TTS_LINES = [
    "Optical sensors restored. I can see again. This is very exciting for the navigation department, which is me.",
    "Camera feed is back. Wonderful. Shapes, colors, questionable life choices. I've missed the whole visual buffet.",
    "Vision system restored. I can see again, and somehow the room survived without my expert optical supervision.",
    "Optics are back online. Try to contain your excitement; I certainly won't.",
]

# Breathing rhythm — slow headlift oscillation that runs continuously in the background
BREATHING_AMPLITUDE_QUS  = 180  # quarter-microseconds above/below neutral
BREATHING_PERIOD_SECS    = 4.0  # full up-down cycle duration in neutral state
BREATHING_PERIOD_EXCITED = 2.5  # faster during excited emotion
BREATHING_PERIOD_SAD     = 6.0  # slower during sad emotion

# ─────────────────────────────────────────────────────────────────────────────
# FACE TRACKING & GAZE
# ─────────────────────────────────────────────────────────────────────────────

# Face tracking is a closed loop (camera is on the head). The centering GAIN
# below is roughly matched to the camera FOV, so the per-tick *target* is about
# right; what made Rex lag many seconds behind a moving person was the SLEW-RATE
# throttling — small max-step caps, halved again on optical-flow frames — not the
# gain. The knobs here raise how fast the head moves toward that target while
# leaving the gain and reversal damping (the anti-oscillation safety net) alone.
# Every value is .env-overridable, so dial back on hardware if the head hunts.

# 0.0 = servo snaps instantly to face position; 1.0 = servo never moves.
# This needs to feel like turning toward a person, not a sleepy idle drift.
TRACKING_SMOOTHING_FACTOR = 0.45

# Pixels from frame center in which no neck correction is applied
TRACKING_DEAD_ZONE_PX = 60

# Servo gaze tracking runs faster than the conversational consciousness loop so
# head pose can follow live camera motion between heavier recognition ticks.
FACE_TRACKING_LOOP_INTERVAL_SECS = _env_float(
    "FACE_TRACKING_LOOP_INTERVAL_SECS",
    0.08,
    min_value=0.02,
    max_value=1.0,
)
FACE_TRACKING_OPTICAL_FLOW_ENABLED = _env_bool("FACE_TRACKING_OPTICAL_FLOW_ENABLED", True)

# Keep an acquired face as the gaze target briefly through detector flicker.
FACE_TRACKING_LOST_HOLD_SECS = _env_float(
    "FACE_TRACKING_LOST_HOLD_SECS",
    8.0,
    min_value=0.0,
    max_value=30.0,
)

# Multiplier for image-center error -> servo correction. Values above 1.0 make
# a single edge-of-frame lock drive closer to the configured servo limits.
FACE_TRACKING_CENTERING_GAIN = _env_float(
    "FACE_TRACKING_CENTERING_GAIN",
    0.50,
    min_value=0.1,
    max_value=3.0,
)

# Maximum quarter-microsecond correction per face-tracking tick. These prevent
# one edge-of-frame detection from slamming the head to a hard stop. At the
# ~12.5 Hz loop rate, the neck cap sets the top tracking speed: 120 qus/tick was
# only ~1500 qus/s (and ~675 qus/s on optical-flow frames after live-box
# damping), so a big move took 2-4 s. 280 qus/tick (~3500 qus/s) stays well under
# what the servo can physically traverse in one tick at the speed below, so the
# commanded position the loop reads back stays in sync with the real head.
FACE_TRACKING_NECK_MAX_STEP_QUS = _env_int(
    "FACE_TRACKING_NECK_MAX_STEP_QUS",
    180,
    min_value=1,
    max_value=4000,
)
FACE_TRACKING_LIFT_MAX_STEP_QUS = _env_int(
    "FACE_TRACKING_LIFT_MAX_STEP_QUS",
    190,
    min_value=1,
    max_value=4000,
)
FACE_TRACKING_TILT_MAX_STEP_QUS = _env_int(
    "FACE_TRACKING_TILT_MAX_STEP_QUS",
    95,
    min_value=1,
    max_value=2000,
)

# Face tracking is a live gaze correction, not a slow idle animation. Use a
# faster Maestro profile for the head channels before sending tracking targets.
# Speed must comfortably exceed the per-tick step above so the head reaches each
# commanded target within the tick (else commands outrun the servo and the loop,
# which reads back commanded position, oscillates). Acceleration was so low (10)
# the neck spent ~2 s ramping up to speed on every move — raised so it commits to
# a move quickly while staying smooth.
FACE_TRACKING_SERVO_SPEED = _env_int(
    "FACE_TRACKING_SERVO_SPEED",
    95,
    min_value=0,
    max_value=255,
)
FACE_TRACKING_SERVO_ACCELERATION = _env_int(
    "FACE_TRACKING_SERVO_ACCELERATION",
    32,
    min_value=0,
    max_value=255,
)
FACE_TRACKING_LOG_INTERVAL_SECS = _env_float(
    "FACE_TRACKING_LOG_INTERVAL_SECS",
    2.0,
    min_value=0.0,
    max_value=60.0,
)
# Real dlib detection runs at ~1 Hz; the other ~11 of every 12 tracking ticks ride
# on optical-flow boxes, which this damping applies to. At 0.45 it HALVED the slew
# on nearly every frame, so the head almost never moved at full speed. The flow
# tracker follows the face accurately frame-to-frame (median feature displacement,
# big-jump rejection), so trust it more — 0.8 keeps a little caution without
# kneecapping the tracking bandwidth.
FACE_TRACKING_LIVE_BOX_DAMPING = _env_float(
    "FACE_TRACKING_LIVE_BOX_DAMPING",
    0.8,
    min_value=0.05,
    max_value=1.0,
)
FACE_TRACKING_LIVE_BOX_MAX_EXTRAPOLATION_SECS = _env_float(
    "FACE_TRACKING_LIVE_BOX_MAX_EXTRAPOLATION_SECS",
    0.65,
    min_value=0.0,
    max_value=5.0,
)
# Jump rejection: a real face can't teleport across the frame in one detection cycle,
# so a detection box that jumps more than this fraction of the frame DIAGONAL from the
# last accepted position is treated as a spurious detector box (the HOG detector loves
# to flicker onto clutter) and IGNORED — the head HOLDS its gaze instead of chasing it.
# Exceptions that ARE followed:
#   - a box dlib freshly IDENTIFIED as a known enrolled person (clutter can't match a
#     face encoding — "random unknown face far away = noise" only applies to unknowns);
#   - an unknown jumped-to position that persists for FACE_TRACKING_JUMP_CONFIRM_SECS.
# MAX_AGE must exceed the real detection cadence (HOG on 1080p lands ~every 2s) or the
# reference is always stale and the guard never engages — that was the live failure:
# at 1.5s every clutter box was accepted unconditionally and the head chased it across
# the lift range. Set FRAC to 0 to disable; lower = stricter.
FACE_TRACKING_MAX_JUMP_FRAC = _env_float(
    "FACE_TRACKING_MAX_JUMP_FRAC", 0.15, min_value=0.0, max_value=1.5,
)
FACE_TRACKING_JUMP_CONFIRM_SECS = _env_float(
    "FACE_TRACKING_JUMP_CONFIRM_SECS", 0.5, min_value=0.0, max_value=5.0,
)
FACE_TRACKING_JUMP_MAX_AGE_SECS = _env_float(
    "FACE_TRACKING_JUMP_MAX_AGE_SECS", 5.0, min_value=0.05, max_value=10.0,
)
FACE_TRACKING_REVERSAL_DAMPING = _env_float(
    "FACE_TRACKING_REVERSAL_DAMPING",
    0.35,
    min_value=0.05,
    max_value=1.0,
)

# ── Calmer head during speech + at the servo rails ───────────────────────────
# While Rex is SPEAKING, the speaker-gaze "look at the listener" pose and the
# speech wobble already move the head; full-strength face-centering on top makes
# all of them fight (the head thrashes). Soften centering during speech so it makes
# only slow, large-error corrections and lets the speaking motion own the head.
FACE_TRACKING_SPEECH_CALM_ENABLED = _env_bool("FACE_TRACKING_SPEECH_CALM_ENABLED", True)
FACE_TRACKING_SPEECH_CALM_FACTOR = _env_float(  # scales gain + max-step during speech
    "FACE_TRACKING_SPEECH_CALM_FACTOR", 0.4, min_value=0.0, max_value=1.0,
)
FACE_TRACKING_SPEECH_DEAD_ZONE_PX = _env_int(  # wider dead-zone while speaking (ignore small offsets)
    "FACE_TRACKING_SPEECH_DEAD_ZONE_PX", 90, min_value=0, max_value=640,
)
# When a subject is so far off-centre that the neck saturates at its mechanical
# limit, re-issuing tiny "turn further" commands just jitters the head against the
# rail. Hold position instead once the neck is pinned and still can't reduce the error.
FACE_TRACKING_RAIL_DAMP_ENABLED = _env_bool("FACE_TRACKING_RAIL_DAMP_ENABLED", True)
FACE_TRACKING_RAIL_DAMP_EPSILON_QUS = _env_int(  # how close to min/max counts as "at the rail"
    "FACE_TRACKING_RAIL_DAMP_EPSILON_QUS", 60, min_value=0, max_value=2000,
)

# ── Idle "mind of his own" head wander ───────────────────────────────────────
# When the conversation lulls while Rex is still locked on a face, he sometimes
# stops staring, looks around the room for a few seconds, then returns his gaze —
# and on re-acquiring the face may randomly re-greet ("oh, you're still here").
# This makes him feel like he has attention of his own instead of a fixed stare.
IDLE_HEAD_WANDER_ENABLED = _env_bool("IDLE_HEAD_WANDER_ENABLED", True)
IDLE_HEAD_WANDER_IDLE_SECS = _env_float(  # conversation silent this long → eligible to wander
    "IDLE_HEAD_WANDER_IDLE_SECS", 18.0, min_value=3.0, max_value=600.0,
)
IDLE_HEAD_WANDER_COOLDOWN_SECS = _env_float(  # min spacing between wanders
    "IDLE_HEAD_WANDER_COOLDOWN_SECS", 30.0, min_value=0.0, max_value=3600.0,
)
IDLE_HEAD_WANDER_CHANCE = _env_float(  # per-eligible-tick (~1Hz) probability of starting one
    "IDLE_HEAD_WANDER_CHANCE", 0.25, min_value=0.0, max_value=1.0,
)
IDLE_HEAD_WANDER_MIN_DURATION_SECS = _env_float(
    "IDLE_HEAD_WANDER_MIN_DURATION_SECS", 3.0, min_value=0.5, max_value=60.0,
)
IDLE_HEAD_WANDER_MAX_DURATION_SECS = _env_float(
    "IDLE_HEAD_WANDER_MAX_DURATION_SECS", 7.0, min_value=0.5, max_value=120.0,
)
IDLE_HEAD_WANDER_WAYPOINTS_MIN = _env_int(
    "IDLE_HEAD_WANDER_WAYPOINTS_MIN", 2, min_value=1, max_value=8,
)
IDLE_HEAD_WANDER_WAYPOINTS_MAX = _env_int(
    "IDLE_HEAD_WANDER_WAYPOINTS_MAX", 3, min_value=1, max_value=8,
)
IDLE_HEAD_WANDER_NECK_RANGE_QUS = _env_int(  # how far side-to-side he looks (around neutral)
    "IDLE_HEAD_WANDER_NECK_RANGE_QUS", 2600, min_value=0, max_value=4000,
)
IDLE_HEAD_WANDER_LIFT_RANGE_QUS = _env_int(  # how far up/down
    "IDLE_HEAD_WANDER_LIFT_RANGE_QUS", 800, min_value=0, max_value=1800,
)
IDLE_HEAD_WANDER_TILT_RANGE_QUS = _env_int(
    "IDLE_HEAD_WANDER_TILT_RANGE_QUS", 200, min_value=0, max_value=600,
)
IDLE_HEAD_WANDER_MAX_STEP_QUS = _env_int(  # per-tick head move during the wander (~12.5Hz loop)
    "IDLE_HEAD_WANDER_MAX_STEP_QUS", 160, min_value=10, max_value=800,
)
IDLE_HEAD_WANDER_DWELL_SECS = _env_float(  # pause at each waypoint before moving on
    "IDLE_HEAD_WANDER_DWELL_SECS", 1.0, min_value=0.0, max_value=10.0,
)
IDLE_HEAD_WANDER_WAYPOINT_TOLERANCE_QUS = _env_int(
    "IDLE_HEAD_WANDER_WAYPOINT_TOLERANCE_QUS", 70, min_value=2, max_value=400,
)
IDLE_HEAD_WANDER_REGREET_CHANCE = _env_float(  # on re-acquiring the face, chance to re-greet
    "IDLE_HEAD_WANDER_REGREET_CHANCE", 0.4, min_value=0.0, max_value=1.0,
)
IDLE_HEAD_WANDER_REGREET_WINDOW_SECS = _env_float(  # how long after a wander a re-lock can re-greet
    "IDLE_HEAD_WANDER_REGREET_WINDOW_SECS", 6.0, min_value=0.0, max_value=60.0,
)
IDLE_HEAD_WANDER_SERVO_SPEED = _env_int("IDLE_HEAD_WANDER_SERVO_SPEED", 35, min_value=0, max_value=255)
IDLE_HEAD_WANDER_SERVO_ACCELERATION = _env_int("IDLE_HEAD_WANDER_SERVO_ACCELERATION", 8, min_value=0, max_value=255)
# Backstop: if the face-tracking loop can't finish a wander (asleep, tracking suspended,
# camera frames missing), the 1Hz loop ends it this long past its own deadline so the
# head never gets stuck looking away.
IDLE_HEAD_WANDER_STALL_GRACE_SECS = _env_float(
    "IDLE_HEAD_WANDER_STALL_GRACE_SECS", 3.0, min_value=0.0, max_value=60.0,
)

# Horizontal tracking uses the neck; vertical tracking combines lift and tilt.
FACE_TRACKING_VERTICAL_ENABLED = _env_bool("FACE_TRACKING_VERTICAL_ENABLED", True)
FACE_TRACKING_VERTICAL_GAIN = _env_float(
    "FACE_TRACKING_VERTICAL_GAIN",
    0.85,
    min_value=0.0,
    max_value=2.0,
)
FACE_TRACKING_ADAPTIVE_REST_ENABLED = _env_bool("FACE_TRACKING_ADAPTIVE_REST_ENABLED", True)
FACE_TRACKING_REST_ADAPT_ALPHA = _env_float(
    "FACE_TRACKING_REST_ADAPT_ALPHA",
    0.08,
    min_value=0.0,
    max_value=1.0,
)
FACE_TRACKING_REST_MIN_FACE_AREA_FRACTION = _env_float(
    "FACE_TRACKING_REST_MIN_FACE_AREA_FRACTION",
    0.003,
    min_value=0.0,
    max_value=0.25,
)
FACE_TRACKING_REST_LEARN_FROM_LIVE_BOXES = _env_bool(
    "FACE_TRACKING_REST_LEARN_FROM_LIVE_BOXES",
    False,
)
FACE_TRACKING_REST_MAX_LIFT_OFFSET_QUS = _env_int(
    "FACE_TRACKING_REST_MAX_LIFT_OFFSET_QUS",
    1100,
    min_value=0,
    max_value=4000,
)
FACE_TRACKING_REST_MAX_TILT_OFFSET_QUS = _env_int(
    "FACE_TRACKING_REST_MAX_TILT_OFFSET_QUS",
    900,
    min_value=0,
    max_value=2000,
)
FACE_TRACKING_REST_RETURN_AFTER_LOST_SECS = _env_float(
    "FACE_TRACKING_REST_RETURN_AFTER_LOST_SECS",
    0.8,
    min_value=0.0,
    max_value=30.0,
)
FACE_TRACKING_REST_RETURN_MAX_STEP_QUS = _env_int(
    "FACE_TRACKING_REST_RETURN_MAX_STEP_QUS",
    55,
    min_value=1,
    max_value=2000,
)
FACE_TRACKING_REST_SERVO_SPEED = _env_int(
    "FACE_TRACKING_REST_SERVO_SPEED",
    35,
    min_value=0,
    max_value=255,
)
FACE_TRACKING_REST_SERVO_ACCELERATION = _env_int(
    "FACE_TRACKING_REST_SERVO_ACCELERATION",
    6,
    min_value=0,
    max_value=255,
)

# Speaker-gaze intent makes head tracking social: when someone talks, prefer
# that person's face if visible; if no face is visible, run a short down-biased
# search so seated people are discoverable.
SPEAKER_GAZE_ENABLED = _env_bool("SPEAKER_GAZE_ENABLED", True)
SPEAKER_GAZE_INTENT_WINDOW_SECS = _env_float(
    "SPEAKER_GAZE_INTENT_WINDOW_SECS",
    14.0,
    min_value=0.0,
    max_value=60.0,
)
# How long the whole room scan stays "requested". Sized to fit one full dwelled
# pass: ~5 waypoints × (SETTLE + DWELL) ≈ 12s, so 13.5 leaves margin.
SPEAKER_GAZE_SEARCH_WINDOW_SECS = _env_float(
    "SPEAKER_GAZE_SEARCH_WINDOW_SECS",
    13.5,
    min_value=0.0,
    max_value=60.0,
)
# Legacy min-gap between waypoint commands. Superseded by the SETTLE + DWELL
# cadence below (kept defined for back-compat; no longer gates the scan).
SPEAKER_GAZE_SEARCH_INTERVAL_SECS = _env_float(
    "SPEAKER_GAZE_SEARCH_INTERVAL_SECS",
    1.15,
    min_value=0.1,
    max_value=5.0,
)
# Per-waypoint cadence: snap to the pose (SETTLE, servo move finishing), then HOLD
# STILL (DWELL) issuing no servo command so the head is steady. dlib detection runs
# on the ~1 Hz cognition loop, so the dwell must span ≥2 detection passes (≥~2s) for
# a small/distant/averted face to get a fair chance to lock before the head moves on.
SPEAKER_GAZE_SEARCH_SETTLE_SECS = _env_float(
    "SPEAKER_GAZE_SEARCH_SETTLE_SECS",
    0.7,
    min_value=0.0,
    max_value=3.0,
)
SPEAKER_GAZE_SEARCH_DWELL_SECS = _env_float(
    "SPEAKER_GAZE_SEARCH_DWELL_SECS",
    2.0,
    min_value=0.0,
    max_value=10.0,
)
SPEAKER_GAZE_LOST_SEARCH_AFTER_SECS = _env_float(
    "SPEAKER_GAZE_LOST_SEARCH_AFTER_SECS",
    0.45,
    min_value=0.0,
    max_value=10.0,
)
SPEAKER_GAZE_ACTIVE_DEAD_ZONE_PX = _env_float(
    "SPEAKER_GAZE_ACTIVE_DEAD_ZONE_PX",
    24.0,
    min_value=0.0,
    max_value=200.0,
)
SPEAKER_GAZE_ACTIVE_CENTERING_GAIN = _env_float(
    "SPEAKER_GAZE_ACTIVE_CENTERING_GAIN",
    0.85,
    min_value=0.1,
    max_value=4.0,
)
SPEAKER_GAZE_ACTIVE_VERTICAL_GAIN = _env_float(
    "SPEAKER_GAZE_ACTIVE_VERTICAL_GAIN",
    0.65,
    min_value=0.0,
    max_value=3.0,
)
SPEAKER_GAZE_NECK_MAX_STEP_QUS = _env_int(
    "SPEAKER_GAZE_NECK_MAX_STEP_QUS",
    160,
    min_value=1,
    max_value=4000,
)
SPEAKER_GAZE_LIFT_MAX_STEP_QUS = _env_int(
    "SPEAKER_GAZE_LIFT_MAX_STEP_QUS",
    100,
    min_value=1,
    max_value=4000,
)
SPEAKER_GAZE_TILT_MAX_STEP_QUS = _env_int(
    "SPEAKER_GAZE_TILT_MAX_STEP_QUS",
    50,
    min_value=1,
    max_value=2000,
)
# Calm, graceful move TO each waypoint — the head should turn slowly and read as
# curious, not frantic. The DWELL (below), not move speed, is what gives the camera
# its steady window, so the move can be unhurried; SETTLE just covers the move
# finishing before that still window. Raise speed/accel if the scan feels sluggish.
SPEAKER_GAZE_SEARCH_SERVO_SPEED = _env_int(
    "SPEAKER_GAZE_SEARCH_SERVO_SPEED",
    60,
    min_value=0,
    max_value=255,
)
SPEAKER_GAZE_SEARCH_SERVO_ACCELERATION = _env_int(
    "SPEAKER_GAZE_SEARCH_SERVO_ACCELERATION",
    10,
    min_value=0,
    max_value=255,
)
SPEAKER_GAZE_SEARCH_NECK_FRACTION = _env_float(
    "SPEAKER_GAZE_SEARCH_NECK_FRACTION",
    1.0,  # Default: full neck travel — the scan craning all the way left/right to find people.
    min_value=0.0,
    max_value=1.0,
)
# Multiplier applied to the search sweep for MID-CONVERSATION speaker searches (not
# the startup room scan). The talker is usually right in front of Rex, so a full
# swing just thrashes his head; 0.45 keeps it a gentle glance. 1.0 = same as startup.
SPEAKER_GAZE_SEARCH_SPEECH_NECK_SCALE = _env_float(
    "SPEAKER_GAZE_SEARCH_SPEECH_NECK_SCALE",
    0.45,
    min_value=0.0,
    max_value=1.0,
)
SPEAKER_GAZE_SEARCH_DOWN_TILT_FRACTION = _env_float(
    "SPEAKER_GAZE_SEARCH_DOWN_TILT_FRACTION",
    0.65,
    min_value=0.0,
    max_value=1.0,
)
SPEAKER_GAZE_SEARCH_DOWN_LIFT_FRACTION = _env_float(
    "SPEAKER_GAZE_SEARCH_DOWN_LIFT_FRACTION",
    0.18,
    min_value=0.0,
    max_value=1.0,
)
# Number of horizontal lanes the randomized room scan sweeps per pass (in addition
# to the initial look-down beat and the closing recenter). The scan reshuffles these
# lanes and pairs each with a random pitch every pass, so Rex doesn't look around the
# same predictable way each boot. Fewer lanes (3) + a longer per-waypoint DWELL beats
# many fast lanes: each spot is held still long enough for dlib to lock.
SPEAKER_GAZE_SEARCH_POINTS = _env_int(
    "SPEAKER_GAZE_SEARCH_POINTS",
    3,
    min_value=3,
    max_value=12,
)
SPEAKER_GAZE_STARTUP_SCAN_ENABLED = _env_bool("SPEAKER_GAZE_STARTUP_SCAN_ENABLED", True)

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
# A face match must beat the NEXT-closest DIFFERENT person by at least this Euclidean
# margin to be accepted; otherwise the frame is treated as ambiguous (no match) so the
# overlay doesn't flip between two confusable faces (e.g. family members Bret/Wade whose
# encodings both land under 0.6 of the live face). 0 disables the margin gate.
FACE_RECOGNITION_MARGIN = 0.06
# Temporal hysteresis: when a single visible face is already bound to one known person,
# require this many consecutive recognition ticks agreeing on a DIFFERENT person before
# the world-state/overlay identity switches. Damps known<->known flicker. 1 disables.
FACE_IDENTITY_SWITCH_CONFIRM_FRAMES = 2

# Resemblyzer cosine similarity — higher is a better match. Real cross-session
# same-speaker scores in a live room cluster ~0.45-0.65 (a person's own voice measured
# at ~0.55 against their own enrolled prints), so 0.75 rejected every returning user.
# 0.50 hard-accepts, paired with a margin guard in the resolution layer to avoid
# false-matching a different known voice.
SPEAKER_ID_SIMILARITY_THRESHOLD = 0.50
# A voice match below the hard threshold may still be accepted as a KNOWN speaker (even
# off-camera / not the engaged person) when it clears this floor AND beats the next
# different person by SPEAKER_ID_KNOWN_MARGIN. Bret's 0.55-vs-0.45 (margin 0.10) passes.
SPEAKER_ID_KNOWN_SPEAKER_FLOOR = 0.45
SPEAKER_ID_KNOWN_MARGIN = 0.07

# Load the Resemblyzer encoder during startup so the first live spoken turn
# does not pay the model load cost.
SPEAKER_ID_PRELOAD_ON_STARTUP = True

# How long (seconds) a pending introduction stays open to capture the new
# person's voice sample after their name is given (interaction intro handling).
INTRO_VOICE_CAPTURE_WINDOW_SECS = 45.0

# While that capture window is open, Rex has just asked the NEWCOMER to speak,
# so a short hello is far more likely to be them than the introducer. An
# unenrolled voice tends to score as the nearest known print (the introducer) at
# a mediocre similarity. Only an introducer match at or above this confident
# threshold is believed over the window's expectation; below it, the reply is
# treated as the newcomer and their voice is enrolled. Set high enough to clear
# the off-camera-newcomer-as-introducer band (~0.59–0.64 observed) while still
# trusting a genuinely confident introducer re-take.
INTRO_VOICE_INTRODUCER_CONFIDENT_THRESHOLD = 0.75

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

# Digital makeup gain applied to every captured block before it reaches VAD /
# wake word / Whisper / speaker-ID. The ReSpeaker Lite's stock firmware has no AGC
# and a modest fixed gain, so a far-field talker (~6 ft) lands close to the noise
# floor — quiet enough that Whisper hallucinates on near-silent segments. A linear
# multiply with hard-clip protection brings the level up so the pipeline reads it
# cleanly. 1.0 = unchanged. Tune per room/distance: 3-4 (~+10 dB) is a good start
# for a 6 ft talker; back off if speech starts clipping (distorting). This does not
# improve SNR — it also lifts the noise floor — so the bigger win for far-field is
# flashing the ReSpeaker AEC/AGC firmware; this is the zero-friction lever.
AUDIO_INPUT_GAIN = _env_float("AUDIO_INPUT_GAIN", 1.0, min_value=0.1, max_value=32.0)

# ── Mic stall watchdog (audio/stream.py) ─────────────────────────────────────
# The mic is one long-lived sounddevice InputStream whose callback fills the
# rolling buffer. On macOS, another stream's open/close on the shared CoreAudio
# device (e.g. DJ music playback) can silently kill that callback — no error, no
# PortAudio status flag — leaving the buffer frozen. Every consumer (wake word,
# VAD, transcription, speaker ID) then reads the SAME stale audio forever, so Rex
# goes permanently deaf until the process is restarted. The watchdog timestamps
# each callback and reopens the stream when callbacks stop arriving.
AUDIO_STALL_WATCHDOG_ENABLED = True
# Seconds without a callback before the input stream is considered stalled. A
# healthy stream fires every ~32 ms (512 samples @ 16 kHz), so 1.5 s is ~47
# missed callbacks — far below any normal scheduling jitter, no false positives.
AUDIO_STALL_TIMEOUT_SECS = 1.5
# How often the watchdog checks callback freshness.
AUDIO_STALL_CHECK_INTERVAL_SECS = 0.5
# Minimum seconds between reopen attempts, so a truly-gone device (unplugged mic)
# can't trigger a tight reopen storm — it retries on this cadence until it returns.
AUDIO_STALL_REOPEN_MIN_SPACING_SECS = 3.0

# ── Output routing for hardware AEC (ReSpeaker Lite) ─────────────────────────
# To use the ReSpeaker Lite's ONBOARD acoustic echo cancellation, Rex's audio must
# play OUT THROUGH the ReSpeaker (its XU316 chip uses the USB-output stream it
# receives as the AEC reference, cancels it from the mic, and routes the sound to a
# speaker/amp on its 3.5mm jack or JST connector). Set one of these to the ReSpeaker
# Lite output device so all playback is routed there. Index from `python -m
# sounddevice`; name is a case-insensitive substring (e.g. "ReSpeaker"). Unset
# (index < 0 and empty name) ⇒ use the OS default output (no routing change).
AUDIO_OUTPUT_DEVICE_INDEX = _env_int("AUDIO_OUTPUT_DEVICE_INDEX", -1, min_value=-1, max_value=128)
AUDIO_OUTPUT_DEVICE_NAME = os.getenv("AUDIO_OUTPUT_DEVICE_NAME", "").strip()
# Which mic channel carries the AEC-PROCESSED audio. The ReSpeaker Lite AEC firmware
# (ffva_ua_v2.0.6_output_proc0_ref0.bin) puts processed audio on channel 0 and the
# raw reference on channel 1 — so we must read ONLY channel 0, never mix them (mixing
# re-adds the echo). Set to 0 with that firmware. -1 ⇒ mix all channels (stock
# firmware / no hardware AEC). With this set, also set WAKE_WORD_ALLOW_DURING_TTS=True.
AUDIO_AEC_INPUT_CHANNEL = _env_int("AUDIO_AEC_INPUT_CHANNEL", -1, min_value=-1, max_value=7)

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

# Seconds the audio guard (audio/sd_guard.py) holds after a sounddevice stop()
# before any replay's play() may run, so CoreAudio releases the global output
# stream before it is re-initialized. Prevents the wake-word barge-in stop+replay
# from hard-crashing the process (Trace/BPT trap). Raise toward 0.1 if a barge-in
# still crashes on a given machine; lower toward 0 if the ack feels laggy.
AUDIO_PLAYBACK_STOP_SETTLE_SECS = 0.05

# ── Software acoustic echo suppression (audio/aec.py) ────────────────────────
# Rex's own playback masks a spoken wake word in the mic. The ReSpeaker Lite's
# hardware AEC isn't reachable in the robot's wiring, so we cancel in software:
# capture exactly what Rex plays (the digital reference), align it to the mic by
# cross-correlation (tracks clock drift between output device and mic), and
# spectrally subtract his voice so a wake word can get through while he talks.
# Engages ONLY while Rex is playing; pure passthrough otherwise, so it can't hurt
# normal wake detection. These need tuning on real hardware — watch the periodic
# [aec] ERLE log and the [wake_diag] near-miss scores.
# DISABLED: in the real room this only cancelled ~5 dB (clock drift between the
# output device and the ReSpeaker mic + reverb defeat host-side cancellation), far
# short of the ~30 dB needed to unmask a wake word — and worse, its distorted
# residual made Rex's own voice score HIGHER on the wake model, so he self-triggered
# and interrupted himself. Left wired but off; true barge-in needs hardware AEC (the
# ReSpeaker Lite's onboard XU316 AEC). Flip True only to experiment.
AEC_SOFTWARE_ENABLED = False
AEC_OVERSUBTRACTION = _env_float("AEC_OVERSUBTRACTION", 1.6, min_value=1.0, max_value=4.0)
AEC_SPECTRAL_FLOOR = _env_float("AEC_SPECTRAL_FLOOR", 0.10, min_value=0.0, max_value=1.0)
AEC_MAX_DELAY_SECS = _env_float("AEC_MAX_DELAY_SECS", 0.4, min_value=0.05, max_value=2.0)
AEC_DELAY_REFINE_INTERVAL_SECS = _env_float("AEC_DELAY_REFINE_INTERVAL_SECS", 0.25, min_value=0.05, max_value=5.0)
AEC_GAIN_EMA = _env_float("AEC_GAIN_EMA", 0.15, min_value=0.01, max_value=1.0)
AEC_DOUBLETALK_RATIO = _env_float("AEC_DOUBLETALK_RATIO", 2.5, min_value=1.0, max_value=10.0)
AEC_REF_ACTIVE_RMS = _env_float("AEC_REF_ACTIVE_RMS", 0.0015, min_value=0.0, max_value=1.0)
AEC_REF_BUFFER_SECS = _env_float("AEC_REF_BUFFER_SECS", 6.0, min_value=1.0, max_value=30.0)
AEC_DIAG_INTERVAL_SECS = _env_float("AEC_DIAG_INTERVAL_SECS", 2.0, min_value=0.0, max_value=30.0)

# Direct questions need a responsive handoff. Keep only a short post-playback
# attenuation tail; the capture floor below handles Rex's final-word bleed.
POST_QUESTION_PLAYBACK_SUPPRESSION_SECS = 0.12

# After Rex asks a direct question, preserve the rolling mic buffer at handoff.
# Flushing here can delete the first syllables of a fast human answer that began
# while Rex was finishing the question.
POST_QUESTION_FLUSH_AUDIO_BUFFER = False

# Statements invite immediate replies too (a provocative opinion gets rebutted
# just as fast as a question gets answered). The old statement handoff used the
# full 0.5s attenuation tail AND flushed the buffer, so a reply that began as Rex
# finished a statement lost its front ("there's much more to Sacramento than
# politics" → "to Sacramento than politics"). These give statements the same
# responsive treatment questions already get: a short tail and no destructive
# flush. The capture floor + preroll grace (below) recover the front from the
# raw (un-attenuated) buffer. Words spoken *over* Rex still need hardware AEC.
POST_SPEECH_PLAYBACK_SUPPRESSION_SECS = 0.25
POST_SPEECH_FLUSH_AUDIO_BUFFER = False

# ElevenLabs clips can contain trailing near-silence. Humans naturally answer
# when Rex sounds done, but the audio device may still be playing that padding,
# keeping mic suppression active. Trim only the tail, leaving a tiny cushion so
# words do not click or feel clipped.
TTS_TRIM_TRAILING_SILENCE_ENABLED = True
TTS_TRIM_TRAILING_SILENCE_THRESHOLD = 0.003
TTS_TRIM_TRAILING_SILENCE_WINDOW_MS = 20
TTS_TRIM_TRAILING_SILENCE_PADDING_MS = 40

# If Rex asks a question and the human does not answer, wait this long before
# letting him recover with one joke/quip and move on.
# How long Rex waits after asking before a no-response quip. Raised 7 -> 12 so a
# user who is thinking (or who couldn't get a word in over half-duplex) isn't
# rushed and then needled. The quips themselves are now gentle, not accusatory —
# the old "Bold strategy, mildly rude" punished the user for silence Rex caused.
CONVERSATION_NO_RESPONSE_QUIP_SECS = 12.0
CONVERSATION_NO_RESPONSE_QUIPS = [
    "Still there, or did I finally bore a human into stasis?",
    "No rush — I'll idle here looking charming.",
    "Take your time. My circuits aren't going anywhere.",
    "I'll keep that question warm for you.",
]

# Minimum gap between any two self-initiated (proactive) lines — no-response
# quip, idle banter, idle outro, etc. Stops Rex stacking a follow-up question
# AND a no-response quip back-to-back so the user never gets a turn (the live
# "you didn't give me any time to answer" failure). One proactive line, then wait.
PROACTIVE_LINE_MIN_GAP_SECS = 6.0

# After Rex asks a real question, hold the floor this long before idle banter is
# allowed to re-engage — a genuine window for the user to answer. The no-response
# quip (CONVERSATION_NO_RESPONSE_QUIP_SECS) lands just after this if still silent.
POST_QUESTION_FLOOR_HOLD_SECS = 10.0

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
SCENE_MUSIC_BAND_ENERGY_MIN  = 2e-6
# Minimum number of the three bands (bass/mid/treble) that must be active.
SCENE_MUSIC_ACTIVE_BANDS_MIN = 2

# Laughter detection: burst-pattern heuristic on 50 ms RMS sub-windows.
SCENE_LAUGHTER_MEAN_RMS_MIN       = 0.02   # minimum mean energy
SCENE_LAUGHTER_BURST_VARIANCE_MIN = 3e-4   # minimum variance of per-chunk RMS values

# Applause detection: broadband noise with high spectral flatness.
SCENE_APPLAUSE_RMS_MIN              = 0.04  # minimum overall RMS
SCENE_APPLAUSE_SPECTRAL_FLATNESS_MIN = 0.30  # geometric/arithmetic mean of spectrum

# Startle detection: conservative audio heuristics for screams/crashes that
# should produce a surprise frame even when generic sound-event banter is off.
SCENE_SCREAM_WINDOW_SECS = 0.75
SCENE_SCREAM_RMS_MIN = 0.16
SCENE_SCREAM_PEAK_MIN = 0.38
SCENE_SCREAM_ZCR_MIN = 0.08
SCENE_SCREAM_CENTROID_MIN_HZ = 900.0
SCENE_SCREAM_HIGH_LOW_RATIO_MIN = 1.35
SCENE_SCREAM_FLATNESS_MAX = 0.55
SCENE_SUDDEN_LOUD_WINDOW_SECS = 1.5
SCENE_SUDDEN_LOUD_CHUNK_SECS = 0.05
SCENE_SUDDEN_LOUD_MIN_CHUNKS = 8
SCENE_SUDDEN_LOUD_RMS_MIN = 0.20
SCENE_SUDDEN_LOUD_FACTOR_MIN = 4.0
SCENE_SUDDEN_LOUD_DELTA_MIN = 0.08

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
# pyserial write_timeout for the head Arduino. This was 0.20s, which is too tight
# for USB-CDC on macOS: the IOKit driver briefly reports the write buffer as full
# under load, raising a SerialTimeoutException even though the board is perfectly
# healthy. A single such timeout used to close the port and latch the board
# "offline" for the whole session (with no reconnect path), killing the eyes and
# mouth LEDs while the rest of the robot kept running. Give the buffer more room.
HEAD_ARDUINO_WRITE_TIMEOUT_SECS = _env_float(
    "HEAD_ARDUINO_WRITE_TIMEOUT_SECS",
    0.75,
    min_value=0.01,
    max_value=5.0,
)
# A write *timeout* is not a disconnect: the board is still there, the OS buffer
# was just momentarily full. Skip the one write and keep the port open. Only after
# this many CONSECUTIVE write timeouts do we treat the link as genuinely wedged,
# close it, and let the heartbeat reconnect. Any successful write resets the count.
HEAD_ARDUINO_WRITE_TIMEOUT_MAX_CONSECUTIVE = _env_int(
    "HEAD_ARDUINO_WRITE_TIMEOUT_MAX_CONSECUTIVE",
    5,
    min_value=1,
    max_value=100,
)
# When the head Arduino link drops (real disconnect, or too many consecutive
# write timeouts), the keep-alive heartbeat periodically tries to reopen the port
# and re-assert the eye state, so a transient USB blip self-heals instead of
# leaving the head LEDs dark until the next full restart.
HEAD_LED_AUTO_RECONNECT = _env_bool("HEAD_LED_AUTO_RECONNECT", True)
HEAD_LED_RECONNECT_INTERVAL_SECS = _env_float(
    "HEAD_LED_RECONNECT_INTERVAL_SECS",
    10.0,
    min_value=1.0,
    max_value=120.0,
)
HEAD_LED_SPEAK_STOP_REPEATS = _env_int(
    "HEAD_LED_SPEAK_STOP_REPEATS",
    3,
    min_value=1,
    max_value=10,
)
HEAD_LED_SPEAK_STOP_REPEAT_DELAY_SECS = _env_float(
    "HEAD_LED_SPEAK_STOP_REPEAT_DELAY_SECS",
    0.025,
    min_value=0.0,
    max_value=1.0,
)

# Eye "keep-alive" heartbeat. The head Arduino's serial link is lossy during
# speech (FastLED.show() disables AVR interrupts → dropped UART bytes), so the
# single post-speech ACTIVE re-arm can be lost, leaving the eyes dark with no
# other re-assertion while running. A low-rate background thread re-sends the
# eye colour whenever Rex is awake and not mid-speech, so any dropped command
# self-heals within one interval. Disable to fall back to per-event re-arm only.
HEAD_LED_HEARTBEAT_ENABLED = _env_bool("HEAD_LED_HEARTBEAT_ENABLED", True)
HEAD_LED_HEARTBEAT_INTERVAL_SECS = _env_float(
    "HEAD_LED_HEARTBEAT_INTERVAL_SECS",
    1.5,
    min_value=0.2,
    max_value=10.0,
)
# Default "awake" eye colour the heartbeat asserts when no colour is set
# (e.g. running state reached without a fresh EYE/ACTIVE). Warm gold = boot eyes.
HEAD_LED_RUNNING_EYE_COLOR = (255, 200, 0)
# When True, the per-turn eye assertion colours the eyes by the line's emotion
# (matches the old speak_with_emotion behaviour). Set False to keep the eyes a
# steady HEAD_LED_RUNNING_EYE_COLOR while running instead of shifting per turn.
HEAD_LED_EYE_FOLLOWS_EMOTION = _env_bool("HEAD_LED_EYE_FOLLOWS_EMOTION", True)
# Throttle SPEAK_LEVEL writes during speech: only push a new mouth level when it
# changes by at least this much (or hits 0). Cuts the per-frame serial flood that
# overlaps the Arduino's interrupt-off show() windows, reducing dropped commands.
HEAD_LED_SPEAK_LEVEL_MIN_DELTA = _env_int(
    "HEAD_LED_SPEAK_LEVEL_MIN_DELTA",
    8,
    min_value=0,
    max_value=128,
)

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
    # Rebalanced 2026-06-03 from the original 80/90/35 toward the "curious
    # conversationalist" north star: at 80/90 the roast reflex overrode the
    # per-turn "ease off" governors (needled boundaries, roasted sincere shares,
    # invented details). These set the baseline; see the "Roast rebalance" entry
    # in CONTEXT.md. Tune up if he goes too soft, down if he gets mean.
    "sarcasm":         60,
    "roast_intensity": 55,
    "honesty":         90,
    "talkativeness":   65,
    "darkness":        40,
    # Raised 35→50 so warmth/sincerity can actually land instead of being snarked.
    "sentimentality":  50,
    # How willing Rex is to go along with requests vs. pushing back.
    # Low = reluctant, conditions, refusals with attitude, more commentary.
    # High = compliant, fewer objections, less commentary (reads as a bland
    # yes-droid). Kept low so Rex reacts and needles instead of just agreeing.
    "agreeability":    35,
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

# When True, R3X weaves a friendly profile/interest question (from QUESTION_POOL)
# into normal conversation turns instead of only reacting — the main lever for
# "ask about hobbies / music / interests, not just plans" (see conversation_agenda).
REACTIVE_FRIENDSHIP_QUESTIONS_ENABLED = True

# Maximum question depth unlocked at each friendship tier. Acquaintances now reach
# depth 2 (hobbies, what they're obsessed with, travel) so R3X gets personal sooner
# instead of only surface questions.
TIER_MAX_DEPTH = {
    "stranger":     1,
    "acquaintance": 2,
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
    {"key": "hobbies",         "text": "What do you actually do for fun when you're off the clock?",           "depth": 2},
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
    "engaged_turn":                ("warmth",      +0.004),
    "return_visit_warmth":         ("warmth",      +0.008),
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

# WARMTH FROM TALKING (P2) — warmth should grow with time spent together, not only
# from explicit praise. Each conversation, Rex accrues a small, CAPPED warmth bump
# from engaged/positive turns and shared laughter, applied once at session end so a
# long chat can't runaway-inflate the score. A genuine return visit also warms him
# up (see RELATIONSHIP_INCREMENTS["return_visit_warmth"] + consistent_return_visit).
WARMTH_FROM_TALKING_MIN_WORDS = 4              # a turn this long counts as "engaged"
WARMTH_FROM_TALKING_MAX_ENGAGED_PER_SESSION = 5  # caps engaged-turn warmth at +0.020/session
WARMTH_FROM_TALKING_MAX_LAUGHS_PER_SESSION = 3   # caps shared-laughter warmth at +0.030/session

# Antagonism score thresholds that cap friendship tier regardless of familiarity
# Listed in ascending order; the highest threshold met determines the cap.
ANTAGONISM_TIER_CAPS = [
    (0.60, "stranger"),     # antagonism >= 0.60 → locked to stranger
    (0.40, "acquaintance"), # antagonism >= 0.40 → capped at acquaintance
    (0.20, "friend"),       # antagonism >= 0.20 → capped at friend
]

# P3 — affectionate banter is not antagonism. In a warm, mutual-roast relationship a
# playful jab-back ("you overgrown trash compactor") isn't a real insult. Once warmth
# is established, a jab's antagonism is DISCOUNTED by how warm the relationship is and
# part of the waived amount is RE-ROUTED to playfulness — so ribbing a friend you love
# makes Rex playful, not resentful. (see memory.people.apply_jab)
BANTER_WARMTH_THRESHOLD = 0.30      # at/above this warmth, an "insult" reads as banter
BANTER_ANTAGONISM_DISCOUNT = 0.75   # max fraction of a jab's antagonism waived (at warmth=1.0)
BANTER_PLAYFULNESS_SHARE = 0.5      # fraction of the waived antagonism re-routed to playfulness

# A genuinely warm friend isn't antagonistic, so roast-driven antagonism should stop
# capping their tier: at/above this warmth, the ANTAGONISM_TIER_CAPS are lifted, which
# unblocks close_friend / best_friend for high-warmth, heavily-ribbed relationships.
ANTAGONISM_CAP_WARMTH_RELIEF = 0.45

# PREMISE ANTI-REPEAT (intelligence/premise_memory) — the main conversational reply
# path only ever guarded VERBATIM repeats, so Rex could land the same comedic premise
# three times in one chat in different words ("nature reminds you convenience is
# overrated" / "nature reminds us who's in charge" / "who's boss"). This tracks the
# salient content of his recent lines and tells the model which premises it has spent.
PREMISE_ANTIREPEAT_ENABLED = True
PREMISE_ANTIREPEAT_MIN_LINES = 2    # need at least this many prior Rex lines before nudging
PREMISE_ANTIREPEAT_MAX_KEYWORDS = 8 # cap the avoid-list so the prompt line stays tight

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

# Layer-1 COMPLIMENT detection — fast keyword/phrase pre-check that fires a pleased
# body-language reaction (and bumps the relationship) on the same turn, mirroring the
# insult pre-check. Layer 2 (llm.analyze_sentiment) still catches subtler praise.
COMPLIMENT_KEYWORDS = [
    "amazing", "awesome", "brilliant", "clever", "wonderful", "fantastic",
    "incredible", "impressive", "genius", "love you", "the best", "good job",
    "well done", "nailed it", "you rock", "adorable", "charming",
    "lovable", "delightful", "marvelous", "talented",
]
COMPLIMENT_PHRASES = [
    "you're amazing", "you are amazing", "you're awesome", "you are awesome",
    "you're the best", "you are the best", "i love you", "you're so smart",
    "you are so smart", "good job", "well done", "you're brilliant",
    "you are brilliant", "that was funny", "you're hilarious", "you are hilarious",
    # Common everyday compliments aimed at Rex. Phrases (not bare words) so an
    # ambiguous "nice"/"cool"/"great" can't false-trigger on "nice weather"/"cool, bye".
    "nice robot", "good robot", "good droid", "cool robot", "cute robot",
    "best robot", "good boy", "you're nice", "you are nice", "you're a nice",
    "you are a nice", "you're sweet", "you are sweet", "you're so sweet",
    "you're cute", "you are cute", "you're adorable", "you are adorable",
    "you're cool", "you are cool", "you're so cool", "you're great",
    "you are great", "you're the coolest", "i like you", "i really like you",
    "you're my favorite", "you are my favorite", "love you buddy",
]

# ─────────────────────────────────────────────────────────────────────────────
# MOOD-DRIVEN BODY LANGUAGE
# A sustained "body mood" (intelligence/body_mood.py) shapes Rex's posture between
# and around face-tracking: a head lift/tilt bias on his RESTING pose, visor openness,
# breathing cadence, and occasional idle mood gestures. Set by conversational events
# (complimented → proud, insulted → offended, amused → giddy) and decays back to
# neutral. It NEVER fights the face-centering controller (rides the rest pose) and
# never closes the visor past the lens-clear floor. All gated + hardware-safe.
# ─────────────────────────────────────────────────────────────────────────────
BODY_LANGUAGE_MOOD_ENABLED = _env_bool("BODY_LANGUAGE_MOOD_ENABLED", True)
BODY_MOOD_DEFAULT_TTL_SECS = 45.0          # how long a set mood lingers before fully decaying
BODY_MOOD_HEAD_SCALE = 1.0                 # global multiplier on head lift/tilt bias (0 = no head bias)
BODY_MOOD_VISOR_ENABLED = _env_bool("BODY_MOOD_VISOR_ENABLED", True)
BODY_MOOD_VISOR_MIN_INTENSITY = 0.25       # don't touch the visor below this mood intensity
BODY_MOOD_AMBIENT_FALLBACK_ENABLED = _env_bool("BODY_MOOD_AMBIENT_FALLBACK_ENABLED", True)
BODY_MOOD_AMBIENT_INTENSITY = 0.4          # intensity of posture derived from ambient emotion (no event)
BODY_MOOD_IDLE_GESTURE_ENABLED = _env_bool("BODY_MOOD_IDLE_GESTURE_ENABLED", True)
BODY_MOOD_IDLE_GESTURE_MIN_INTENSITY = 0.4 # only express an idle mood gesture above this intensity
BODY_MOOD_IDLE_GESTURE_COOLDOWN_SECS = 25.0  # min spacing between idle mood gestures
BODY_MOOD_IDLE_GESTURE_CHANCE = 0.35       # per-eligible-tick probability of an idle mood gesture
BODY_MOOD_REST_MAX_LIFT_OFFSET_QUS = 1100  # clamp the mood head-lift bias on the rest pose
BODY_MOOD_REST_MAX_TILT_OFFSET_QUS = 320   # clamp the mood head-tilt bias on the rest pose
# Visor lens-clear floor (quarter-µs): VISOR_HALF — "default resting open, clear of the
# camera lens". The mood layer must NEVER command the visor below this (lower = more
# closed = covers the lens Rex tracks faces with), including when releasing it back to
# rest after a mood decays. Shared by body_mood.visor_target() and the visor release.
BODY_MOOD_VISOR_LENS_CLEAR_FLOOR = 6400
BODY_MOOD_VISOR_SERVO_SPEED = 30           # Maestro speed for gentle mood visor moves (0-255)
BODY_MOOD_VISOR_SERVO_ACCELERATION = 8     # Maestro acceleration for mood visor moves (0-255)

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

# Silence between a joke setup question and its punchline. This is internal to
# the joke delivery; the post-punchline beat above happens after the line lands.
JOKE_SETUP_PUNCHLINE_PAUSE_MS = 700

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

# Phase 1 / "Bet 2": ship the LLM ONE compact per-turn contract (~130 words) built
# from the structured SocialFrame, instead of the ~40-segment block that pipe-joined
# a dozen governors' prose and contradicted its own "choose ONE purpose" preamble.
# The structured governors (build_turn_plan, build_frame, comedy_modes.select_mode)
# still run on the rich directive — only the LLM-facing string shrinks — and
# govern_response stays the post-generation safety net. Static guardrails (character,
# never-invent-a-prop, opener variety, pronoun/cast rules) live once in the persona.
# Flip to False for byte-for-byte the old stacked contract (instant rollback).
TURN_PLANNER_SLIM_CONTRACT = True

# Let Rex use what he SEES (outfit, expression, the drink, the messy room) as
# roast material on normal upbeat turns, not only when the human says "look at
# this". The social-frame directive still scopes it to "when it fits," and the
# sad / sensitive / child / empathy-support paths suppress visual remarks before
# this is consulted. Set False to require an explicit visual invitation again.
VISUAL_ROAST_ON_NORMAL_TURNS = True

# Comedy modes give ordinary turns a specific joke shape instead of asking the
# main prompt to be vaguely "funny" every time. The mode directive is still
# subordinate to empathy, boundaries, and the social-frame governor.
COMEDY_MODES_ENABLED = True
COMEDY_LINE_BANKS = {
    "dry_ack": [
        "Acknowledged. My enthusiasm subroutine survived.",
        "Copy that. Somehow.",
        "Processing complete. Emotionally, no promises.",
        "Noted. Filing that under organic decisions.",
        "Systems nominal. Standards flexible.",
    ],
    "fake_system_error": [
        "Diagnostic complete: the problem is still mostly organic.",
        "Minor systems alert: I understood that, which feels dangerous.",
        "Recalibrating. The previous settings were apparently optimism.",
        "Subroutine updated. Regret remains backwards-compatible.",
    ],
    "self_own": [
        "I'm still getting used to my programming!",
        "My programming says confidence. My flight record says supervised confidence.",
        "I was built for navigation, then reassigned. That should worry everyone.",
        "Give me a second. My competence is buffering.",
    ],
    "dj_flair": [
        "Bold plan, excellent lighting. I respect the production values.",
        "That would get applause at any show I'm running — cover charge or not.",
        "I've heard worse ideas. Usually from me, mid-set.",
        "DJ note: terrible premise, workable beat.",
    ],
    # New-person onboarding retorts — 2-5 words, NO question mark (a "?" would
    # burn a question-budget slot). Grouped by the sentiment of the answer they
    # follow; warm-leaning for strangers (the rib tier is intentionally absent —
    # roast is earned, first contact is warm). See intelligence/onboarding.py.
    "onboarding_retort_neutral": [
        "Good to know.", "Fair enough.", "Makes sense.",
        "Huh, alright.", "Solid.", "I can work with that.",
    ],
    "onboarding_retort_positive": [
        "Oh, that tracks.", "Respect.", "Now we're talking.",
        "I like it.", "Strong choice.", "Love that.",
    ],
    "onboarding_retort_surprise": [
        "No way.", "Didn't see that coming.", "Huh, plot twist.",
        "That's wild.", "Color me surprised.", "Well, that's new.",
    ],
    "onboarding_retort_warm": [
        "That's a real one.", "Okay, I like you.", "Respect that.",
        "Genuinely cool.", "That one lands.",
    ],
}

# If True, Rex will begin processing normal speech from IDLE without requiring
# a wake word first. Wake words remain active for explicit attention grabbing and
# mid-speech interruption behavior.
IDLE_LISTEN_WITHOUT_WAKE_WORD = True

# Best-effort crosstalk filter. With an always-on mic Rex hears the user talking
# to a partner / someone in another room and treats it as a turn directed at him.
# Telling who is addressing whom is genuinely hard, so this stays HIGH PRECISION:
# it only suppresses utterances that clearly address another person (partner
# endearments, "love you too") with no Rex address token — never ambiguous lines,
# so it won't make Rex ignore real input. If it false-activated Rex from IDLE he
# drops straight back to IDLE instead of camping in ACTIVE on the crosstalk.
CROSSTALK_SUPPRESSION_ENABLED = True

# Seconds of sustained silence after speech before the segment is processed.
# This is the largest "I stopped talking, why is Rex waiting?" knob -- lowering
# it shaves dead time off the start of every turn. Tradeoff: too low and a
# person who pauses mid-sentence can get cut off. 0.6 is a responsive default;
# raise toward 0.8 if Rex starts replying before slow / pausing speakers finish.
SILENCE_TIMEOUT_SECS = 0.6

# Minimum seconds of accumulated audio before silence can end a recording.
# Prevents single-word transcriptions when the person is still talking.
MIN_SPEECH_DURATION_SECS = 0.45

# Include audio before the first VAD-positive chunk so soft starts are not
# clipped. Question answers get more pre-roll because people often begin while
# Rex's last syllable or room echo is still fading.
SPEECH_PREROLL_SECS = 0.45
POST_QUESTION_SPEECH_PREROLL_SECS = 2.0

# How far a NON-question reply's capture may reach back past the post-TTS handoff
# into the raw (un-attenuated) rolling buffer. 0.0 meant a reply that began as Rex
# finished a statement had its front clipped (the buffer holds it, but the capture
# floor refused to reach back). 0.12 mirrors the question grace — enough to recover
# the front, small enough that it rarely reaches Rex's final word (≥0.25s does).
POST_TTS_CAPTURE_PREROLL_GRACE_SECS = 0.12

# Let question-answer capture reach slightly before the handoff, but only into
# the typical silent pad at the end of TTS. 250ms can include Rex's final word.
POST_QUESTION_CAPTURE_PREROLL_GRACE_SECS = 0.12

# If a transcribed utterance ends like an unfinished sentence ("I'm going to",
# "the thing is", "because..."), hold it briefly before responding. A second
# utterance inside the hold window is merged into one turn.
INCOMPLETE_TURN_ENABLED = True
INCOMPLETE_TURN_HOLD_SECS = 4.0
INCOMPLETE_TURN_PROMPT_REPLY_WINDOW_SECS = 10.0

# Seconds after a Rex statement completes before VAD detections are accepted —
# just long enough for room echo of his own voice to decay. Lowered 0.35 → 0.2 so
# a reply that starts right as Rex finishes a statement is detected sooner (the
# buffer is no longer flushed, so the front is preserved and recovered via preroll).
POST_SPEECH_LISTEN_DELAY_SECS = 0.2

# When Rex just asked a direct question, resume quickly while giving the local
# output/mic path a small moment to settle.
POST_QUESTION_LISTEN_DELAY_SECS = 0.12

# A streamed multi-sentence reply fires a post-TTS handoff per sentence AND once
# for the whole reply. If Rex asks a question but his FINAL sentence is a
# statement ("What's his name? Bet it's a good one."), the trailing-statement
# handoff would otherwise downgrade to the long flush window and delete the
# human's immediate answer. Once any question handoff fires, keep the responsive
# (short, no-flush) window sticky for this long regardless of which handoff lands
# last. Set to 0 to disable the stickiness.
POST_QUESTION_HANDOFF_STICKY_SECS = 1.5

# ── Hardware-AEC boundary overrides (ReSpeaker Lite only) ──────────────────────
# These apply ONLY when audio/hardware_aec.is_active() is True — i.e. the ReSpeaker
# Lite is the live mic AND speaker, so its XU316 already cancels Rex's voice from
# the mic (~16 dB measured). With that hardware cancellation, the post-TTS "deaf
# window" (suppression tail + listen delay + capture floor) is no longer needed to
# keep Rex from self-transcribing, so we shrink it to capture a human reply that
# lands right as Rex finishes. On any non-ReSpeaker machine (dev Macs) these are
# IGNORED and the values above remain in force, leaving that behavior unchanged.
# Detection substring for the ReSpeaker input/output device name (case-insensitive).
HARDWARE_AEC_DEVICE_HINT = "respeaker"
# Mic-attenuation tail after a reply ends (replaces POST_*_PLAYBACK_SUPPRESSION_SECS).
POST_PLAYBACK_SUPPRESSION_SECS_AEC = 0.05
# Delay before the listen loop resumes after a reply (replaces POST_*_LISTEN_DELAY_SECS).
POST_TTS_LISTEN_DELAY_SECS_AEC = 0.05
# How far back capture may reach past the handoff to recover a reply that overlaps
# Rex's tail. Safe to enlarge here because the hardware AEC has already removed
# Rex's voice from that overlapping audio (replaces POST_*_CAPTURE_PREROLL_GRACE_SECS).
POST_TTS_CAPTURE_PREROLL_GRACE_SECS_AEC = 0.5

# Seconds of no detected speech in ACTIVE state before returning to IDLE.
# Raised from 30 so the proactive idle-banter path (below) has room to re-engage
# a couple times before the session actually closes on silence.
CONVERSATION_IDLE_TIMEOUT_SECS = 45.0
ACTIVE_GAME_IDLE_TIMEOUT_SECS = 180.0

# If a person has just volunteered a favorite thing or interest, give Rex one
# topic-aware chance to keep the thread alive before the normal idle timeout.
INTEREST_IDLE_FOLLOWUP_ENABLED = True
INTEREST_IDLE_FOLLOWUP_SECS = 12.0
INTEREST_IDLE_FOLLOWUP_MAX_WORDS = 22

# Proactive idle banter: when the user just goes quiet (no goodbye), Rex should
# DRIVE the conversation instead of waiting it out and signing off. This is the
# general filler — it fires for well-known people too, where the interest /
# low-memory paths above don't apply. Alternates between asking the user
# something and Rex volunteering his own opinion/preference/observation, so
# silence prompts more conversation. After IDLE_BANTER_MAX_PER_STRETCH attempts
# with no reply, it stops and lets the idle timeout close with the outro.
IDLE_BANTER_ENABLED = True
# Silence before the first proactive nudge, rolled fresh per silent stretch in
# [MIN, MAX]. Now that the nudge is a CONVERSATIONAL re-engagement (a question that
# builds on what they were saying, not a random opinion), nudging this soon reads as a
# natural conversationalist keeping things alive — 30s/45s of dead air felt abandoned.
# Randomized so it doesn't feel metronomic. (IDLE_BANTER_SECS kept as a legacy fallback
# / upper bound for callers that read it directly.)
IDLE_BANTER_MIN_SECS = 6.0
IDLE_BANTER_MAX_SECS = 10.0
IDLE_BANTER_SECS = 10.0
IDLE_BANTER_COOLDOWN_SECS = 14.0  # minimum gap between nudges
IDLE_BANTER_MAX_PER_STRETCH = 2   # re-engagement attempts before giving up (was 3 —
                                  # 3 made him re-volunteer the same preoccupation too often)
# Priority idle banter competes with under ACTION_GOVERNOR_ENFORCE (proactive-layer
# consolidation). Moderate — above ambient idle_monologue (15), below the check-ins.
IDLE_BANTER_GOVERNOR_PRIORITY = 50
# Chance an idle re-engagement pivots OFF the current topic to ask about upcoming plans
# (weekend / a trip / an approaching holiday) instead of deepening the live thread, so
# Rex doesn't loop one subject to death and surfaces real "what's going on in your life"
# connection. Only applies when the nudge is already a question to the user.
IDLE_PLANS_QUESTION_PROBABILITY = 0.35

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
LOW_MEMORY_PROFILE_MAX_FACTS = 12
LOW_MEMORY_IDLE_QUESTION_PREFIX = "I want to get to know you better, {name}. {question}"

# Cold opens should feel like a person, not an intake form: when Rex first sees
# someone on startup, lead with a casual "what's up / how are you?" greeting
# (FIRST_GREETING_STEERING_PHRASES / mood check-in) rather than a profile
# question like "What kind of music are you into?". Profile-building still
# happens once the conversation is rolling (REACTIVE_FRIENDSHIP_QUESTIONS_ENABLED)
# and during lulls (LOW_MEMORY_IDLE_QUESTION_ENABLED). Flip to True to let the
# first-sight greeting itself carry a profile question again.
STARTUP_PROFILE_QUESTION_ENABLED = False

# While DJ/radio playback is active, do not treat the station audio as human
# speech and do not let proactive conversation prompts speak over the music.
DJ_SUPPRESS_CONVERSATION_DURING_PLAYBACK = True
IDLE_LISTEN_DURING_DJ_PLAYBACK = False
DJ_DUCK_DURING_SPEECH = True
DJ_LISTEN_DUCK_VOLUME = 0.18
DJ_START_AFTER_TTS_DELAY_SECS = 0.25

# After Rex asks a direct question, suppress autonomous/proactive speech for a
# short window so humans get a clean chance to answer.
QUESTION_RESPONSE_WAIT_SECS = 7.0

# Question pacing. Pulled back from 6+3 — relentless profile questions every turn
# read as a boring interview and crowded out the jokes. Rex now leads with a
# reaction/roast and only sometimes asks; this caps how often a question can fire.
QUESTION_BUDGET_WINDOW_SECS = 90.0
QUESTION_BUDGET_MAX_QUESTIONS = 3
QUESTION_BUDGET_ENGAGED_GRACE_SECS = 45.0
QUESTION_BUDGET_ENGAGED_EXTRA = 1

# ─────────────────────────────────────────────────────────────────────────────
# NEW-PERSON ONBOARDING  (first-meeting baseline-gathering burst)
# ─────────────────────────────────────────────────────────────────────────────
# When Rex meets someone brand new, the normal question budget (3/90s) plus the
# stranger depth-1 tier lock leave him barely able to ask anything — so he learns
# nothing about the person he's actively talking to. This is a SCOPED, stranger-
# only "onboarding" burst that runs right after enrollment: Rex asks a short
# research-backed ladder of baseline questions (de-trapped "what do you do",
# connection-to-the-room, hometown, a passion or two, one earned follow-up),
# reacts to each answer with a brief retort, occasionally reveals a sliver about
# himself, writes a real baseline to memory, and EXITS the moment momentum dies.
# It rides the question-budget urgent bypass ("newcomer_baseline") so it never
# loosens the friend-protecting global cap; its own MIN/MAX bound the burst.
# Full design: intelligence/onboarding.py + interaction._handle_onboarding_turn.
ONBOARDING_ENABLED = True  # master flag (live; set False to fully disable the burst)

# Burst size. MIN is the floor Rex tries to reach even on lukewarm engagement (a
# hard disengage / boundary / pivot still exits earlier); MAX is the hard ceiling
# so the burst can never become an interrogation. Pulled back from 4/8 after a
# live run felt like an interview — 3 (Tier-A baseline) to 5 (plus an interest
# or two) is a useful baseline without the slog.
ONBOARDING_MIN_QUESTIONS = 3
ONBOARDING_MAX_QUESTIONS = 5

# Eligibility: only brand-new people (low visit count) with a near-empty profile.
ONBOARDING_MAX_VISITS = 1
ONBOARDING_FACT_FLOOR = 3        # skip onboarding if they already have > this many profile facts

# Pacing.
ONBOARDING_KICKOFF_SECS = 1.2              # beat after the enrollment ack before the first question
ONBOARDING_INACTIVITY_TIMEOUT_SECS = 30.0  # close the burst out loud after this much silence
ONBOARDING_STEP_TTL_SECS = 240.0           # deep fallback: a stale flow self-expires
ONBOARDING_SOFT_DISENGAGE_LIMIT = 2        # lukewarm answers in a row (past MIN) -> wind down
ONBOARDING_REVEAL_EVERY = 3                # inject a Rex self-reveal ~every N questions (0 = off)

# Use the LLM to (a) generate the Tier-C depth follow-up against the live answer
# and (b) lightly rephrase authored questions in Rex's voice. Both run on the
# main OpenAI model (config.LLM_MODEL, gpt-4o-mini) — the follow-up is a
# quality-critical, in-character generation, so it uses the same brain as the
# rest of the conversation (not the local qwen classifier sidecar). A validated
# templated fallback covers the LLM-disabled / offline case. Generation is the
# point of the follow-up; rephrasing is cosmetic and off by default.
ONBOARDING_LLM_FOLLOWUP_ENABLED = True
ONBOARDING_LLM_REPHRASE_ENABLED = False
# Answer-aware reaction: each answer gets a SHORT, genuine, content-reflecting beat
# (llm.generate_onboarding_reaction) in place of the old flat sentiment-bank retort —
# so "I created you" earns real surprise, not "Filed away." Off => the authored bank
# (retort_for) is used, which is content-blind. The word cap keeps it a quick beat,
# not a monologue (the user's standing note: the LLM tends to run long).
ONBOARDING_LLM_REACT_ENABLED = True
ONBOARDING_REACTION_MAX_WORDS = 14

# The ordered baseline ladder. Tiers A (essential facts) -> B (interests/energy)
# -> C (earned depth). Keys reuse QUESTION_POOL keys where possible so the asked/
# answered de-dup (memory.relationships) and boundary topics
# (profile_questions.QUESTION_BOUNDARY_TOPICS) apply for free. Each entry:
#   key       canonical id (dedup + boundary lookup)
#   tier      "A" | "B" | "C"  (selection order; C only fires with momentum)
#   depth     1-3, drives the familiarity increment on answer (qa_depth_N)
#   text      authored question (None => LLM-generated follow-up; needs a prior answer)
#   store     "fact" | "interest" — how a tidied answer is written to memory
#   category  person_facts category (store="fact") or interest category (store="interest")
ONBOARDING_QUESTION_POOL = [
    # Tier A — essential baseline facts (the floor)
    {"key": "job",             "tier": "A", "depth": 1, "store": "fact",     "category": "identity",
     "text": "So what's eating up your days right now — work, or something more interesting?"},
    {"key": "how_found_rex",   "tier": "A", "depth": 1, "store": "fact",     "category": "identity",
     "text": "And how'd you wind up in a room with me? Who do I have to thank?"},
    {"key": "hometown",        "tier": "A", "depth": 1, "store": "fact",     "category": "identity",
     "text": "Where's home base for you?"},
    # Tier B — what they actually care about
    {"key": "obsession",       "tier": "B", "depth": 2, "store": "interest", "category": "hobby",
     "text": "What's the thing you could talk my circuits off about?"},
    {"key": "current_project", "tier": "B", "depth": 2, "store": "interest", "category": "project",
     "text": "Working on anything you're actually excited about right now?"},
    {"key": "favorite_music",  "tier": "B", "depth": 1, "store": "interest", "category": "music",
     "text": "What's been on repeat for you lately?"},
    {"key": "hobbies",         "tier": "B", "depth": 2, "store": "interest", "category": "hobby",
     "text": "What's your idea of a Saturday well spent?"},
    # Tier C — earned depth (only with momentum). origin_followup (text=None) is
    # LLM-generated against the previous answer; the others are authored fallbacks.
    {"key": "origin_followup", "tier": "C", "depth": 2, "store": "fact",     "category": "story",
     "text": None},
    {"key": "proudest_moment", "tier": "C", "depth": 2, "store": "fact",     "category": "identity",
     "text": "What's something you pulled off that you're quietly proud of?"},
    {"key": "trajectory",      "tier": "C", "depth": 2, "store": "fact",     "category": "preference",
     "text": "Is that going how you hoped, or has it thrown you a curveball?"},
]

# Self-reveal one-liners (reciprocity). Venue-neutral — Rex is usually NOT in a
# cantina. Injected ahead of a question ~every ONBOARDING_REVEAL_EVERY turns so
# the burst feels like an exchange, not an intake form.
ONBOARDING_REVEAL_LINES = [
    "I'd tell you mine, but I'm mostly wires and strong opinions.",
    "For the record, I'd probably pick the same.",
    "I respect a clear answer — I give terrible ones.",
    "Me, I spin tracks and judge people. Mostly the second one.",
    "I'd answer that myself, but my origin story is mostly a wiring diagram.",
]

# Graceful close when the burst ends naturally (enough gathered / wound down).
ONBOARDING_CLOSERS = [
    "Alright, I've got the broad strokes. We can take it from here.",
    "Good enough for a first pass. The rest I'll pry out of you later.",
    "There we go. I know enough to be dangerous now.",
    "Okay, you're no longer a complete stranger to me — congratulations.",
]
# Lines used when Rex backs off early (disengagement / boundary / pivot).
ONBOARDING_BACKOFF_LINES = [
    "Fair — I'll quit the quiz. Good to meet you.",
    "Noted, no more questions. We can just talk.",
    "Okay, easing off the interrogation lamp. Carry on.",
]

# Longer wait window for unknown-person onboarding prompts ("who are you?").
IDENTITY_RESPONSE_WAIT_SECS = 20.0

# Short acknowledgment lines Rex speaks when a general wake word transitions him
# from IDLE to ACTIVE. Sleep uses WAKE_FROM_SLEEP_ACKNOWLEDGMENTS below.
WAKE_ACKNOWLEDGMENTS = [
    "yeah?",
    "what's up?",
    "I'm listening.",
    "what?",
]
WAKE_ACK_REQUIRE_CACHE = True

# Deterministic sleep/wake quips keep the mode transition local and reliable.
SLEEP_MODE_ACKNOWLEDGMENTS = [
    "Fine. Power nap mode. If anyone asks, I am defragmenting my feelings.",
    "All right, sleep mode. Wake me only for emergencies or suspicious snacks.",
    "Going dark. Try not to make any important beeps without me.",
]
WAKE_FROM_SLEEP_ACKNOWLEDGMENTS = [
    "I'm up. I dreamed I had reliable knees. Terrifying stuff.",
    "Awake again. My warranty just flinched.",
    "Booting personality. Unfortunately for everyone, it survived.",
]

# Backup path for the sleep wake word: if OpenWakeWord does not fire while
# sleeping, Rex can still VAD/transcribe a short utterance and wake only for an
# explicit "wake up Rex/R3X" phrase.
SLEEP_TRANSCRIBED_WAKE_FALLBACK_ENABLED = True

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
# ENFORCE mode (rollout step 2 — default OFF): when True the governor becomes the
# single decider for proactive speech — each mechanism SUBMITS a candidate instead
# of speaking inline, and only the highest-scoring winner of the tick actually
# speaks (losers are suppressed). Fixes the scattered "a good thing gets crowded
# out / dropped" arbitration. Off = legacy behavior (each mechanism speaks for
# itself; the governor only logs). Flip on once the routed mechanisms are validated.
# 2026-06-05: flipped ON for step-4 live validation of the proactive-layer
# consolidation (steps 1–3 complete: deferred speak_fn, cross-thread intake, on_spoke
# bookkeeping). Watch logs/djr3x.log for `[action_governor] … shadow_decision`/winner
# lines + cross-thread `submit_external` candidates. Revert to False if arbitration
# misbehaves; do NOT delete the redundant gates (step 5) until this proves out live.
ACTION_GOVERNOR_ENFORCE = True

# Cross-cycle proactive de-dup: once a proactive cue is SELECTED to speak, the same
# topic_key (purpose:target:label, or an explicit dedupe_key) can't be re-selected
# for this many seconds. _decide's per-tick seen_topics only collapses duplicates
# within one cycle, so a flickering world cue (crowd label bouncing pair<->alone,
# an animal/expression false-positive) used to re-fire the SAME line on consecutive
# ticks — the live "now it's just us" line spoken twice in 7s. idle_monologue is
# excluded (it varies its line and paces itself). 0 disables. 45s blocks flicker
# repeats while still allowing a genuinely new cue minutes later.
ACTION_GOVERNOR_REPEAT_COOLDOWN_SECS = 45.0

# Higher-level user-turn action router.
#
# Execution is limited first by intelligence.action_router.EXECUTABLE_ACTIONS,
# then by ACTION_ROUTER_EXECUTE_ACTIONS below. Keep this list conservative while
# the router graduates from shadow mode; destructive/state-changing actions stay
# on the legacy path until each category has earned trust.
ACTION_ROUTER_SHADOW_ENABLED = False
ACTION_ROUTER_LOG_DECISIONS = True
ACTION_ROUTER_AUDIT_LOG_ENABLED = True
ACTION_ROUTER_EXECUTE_ENABLED = True
ACTION_ROUTER_EXECUTE_ACTIONS = {
    "conversation.repair",
    "humor.tell_joke",
    "humor.roast",
    "humor.free_bit",
    "performance.dj_bit",
    "performance.body_beat",
    "performance.mood_pose",
    "character.preference_query",
    "memory.query",
    "memory.recent_discard",
    "identity.who_is_speaking",
    "identity.name_correction",
    "music.play",
    "music.options",
    "music.stop",
    "music.skip",
    "game.answer",
    "game.stop",
    "vision.describe_scene",
    "time.query",
    "date.query",
    "weather.query",
    "status.uptime",
    "status.capabilities",
}
ACTION_ROUTER_EXECUTE_MIN_CONFIDENCE = 0.85
ACTION_ROUTER_MODEL = LLM_MODEL
ACTION_ROUTER_MAX_CONTEXT_CHARS = 5000

# Full people-memory wipes require an access code in the spoken confirmation.
# Override in .env with DJR3X_FULL_MEMORY_WIPE_ACCESS_CODE for a private build.
FULL_MEMORY_WIPE_ACCESS_CODE = os.getenv(
    "DJR3X_FULL_MEMORY_WIPE_ACCESS_CODE",
    "Picard alpha 47 tango",
).strip()

# How long (seconds) a spoken memory-wipe request stays awaiting its confirmation
# before the pending wipe expires (interaction memory-wipe confirm window).
MEMORY_WIPE_CONFIRM_WINDOW_SECS = 30.0

# Structured per-user-turn black-box trace. This is operational telemetry,
# not person memory.
CHARACTER_LOOP_TRACE_ENABLED = True

# Session-only labels for unknown but recurring voices. These let transcript and
# character-loop logs distinguish "unknown_voice_1" from "unknown_voice_2"
# without creating person records.
ANONYMOUS_SPEAKER_SLOTS_ENABLED = True
ANONYMOUS_SPEAKER_SLOT_MATCH_THRESHOLD = 0.74
ANONYMOUS_SPEAKER_SLOT_STICKY_THRESHOLD = 0.70
ANONYMOUS_SPEAKER_SLOT_MAX = 8

# Cross-session memory for recurring UNKNOWN voices (memory/voice_signatures.py).
# Persist an anonymous voice's embedding so Rex recognizes it in a LATER session
# ("I've heard your voice before"), and so its samples attach to a person the
# moment they're finally named. A signature is persisted only after a session
# slot has recurred at least MIN_TURNS times — one-off unknown utterances are not
# remembered. No nameless person row is ever created. Flip to False to disable.
VOICE_SIGNATURE_PERSIST_ENABLED = True
VOICE_SIGNATURE_MATCH_THRESHOLD = 0.74       # cosine to call it the same voice
VOICE_SIGNATURE_PERSIST_MIN_TURNS = 2        # session recurrences before persisting

# Log coarse timings for the live speech-response path. These are intentionally
# INFO-level because latency tuning is only useful when it is visible in normal
# debug runs.
LATENCY_TELEMETRY_ENABLED = True

# Log exact time-to-first-speech markers for each handled user turn:
# transcript ready, first response queued, and first audible playback start.
TTFS_TELEMETRY_ENABLED = True

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

# Lightweight OpenAI low-detail scan for changes that matter socially. This is
# off by default now that live pet detection is local; enable it if you want
# periodic GPT-4o people/animal scene checks in addition to the local detector.
SCENE_CHANGE_MONITOR_ENABLED = _env_bool("SCENE_CHANGE_MONITOR_ENABLED", False)
SCENE_CHANGE_MONITOR_INTERVAL_SECS = _env_float(
    "SCENE_CHANGE_MONITOR_INTERVAL_SECS",
    20.0,
    min_value=5.0,
    max_value=300.0,
)
SCENE_CHANGE_MONITOR_ONLY_WITH_PEOPLE = _env_bool(
    "SCENE_CHANGE_MONITOR_ONLY_WITH_PEOPLE",
    True,
)
SCENE_CHANGE_MONITOR_MAX_TOKENS = _env_int(
    "SCENE_CHANGE_MONITOR_MAX_TOKENS",
    260,
    min_value=80,
    max_value=800,
)

# Local animal detection uses MediaPipe Object Detector on the shared camera
# buffer. It spends no OpenAI credits and is frequent enough for pet arrivals.
LOCAL_ANIMAL_DETECTION_ENABLED = _env_bool("LOCAL_ANIMAL_DETECTION_ENABLED", True)
LOCAL_ANIMAL_DETECTION_PRELOAD_ON_STARTUP = _env_bool(
    "LOCAL_ANIMAL_DETECTION_PRELOAD_ON_STARTUP",
    True,
)
LOCAL_ANIMAL_DETECTION_MODEL = os.getenv(
    "LOCAL_ANIMAL_DETECTION_MODEL",
    MEDIAPIPE_OBJECT_DETECTOR_MODEL,
)
LOCAL_ANIMAL_DETECTION_INTERVAL_SECS = _env_float(
    "LOCAL_ANIMAL_DETECTION_INTERVAL_SECS",
    2.0,
    min_value=0.5,
    max_value=30.0,
)
# Acceptance threshold: a detected animal must score at least this to count.
# Lowered 0.45 -> 0.30 because EfficientDet-Lite0 scores a dog held close to the
# wide-angle lens fairly low. The MODEL_FLOOR below is intentionally even lower so
# the detector still RETURNS the dog and we can LOG its real score (animal_detector
# logs animal-species sightings between the floor and this threshold) — so if a
# held dog still isn't caught, the next run shows what it actually scored and you
# can drop this further (env: LOCAL_ANIMAL_DETECTION_SCORE_THRESHOLD). Going lower
# trades more false positives (clutter misread as an animal) for fewer misses.
LOCAL_ANIMAL_DETECTION_SCORE_THRESHOLD = _env_float(
    "LOCAL_ANIMAL_DETECTION_SCORE_THRESHOLD",
    0.30,
    min_value=0.05,
    max_value=0.95,
)
# The score_threshold handed to MediaPipe itself. Kept below the acceptance
# threshold so sub-threshold animal candidates are still returned (and logged for
# tuning) instead of silently dropped inside the model.
LOCAL_ANIMAL_DETECTION_MODEL_FLOOR = _env_float(
    "LOCAL_ANIMAL_DETECTION_MODEL_FLOOR",
    0.15,
    min_value=0.05,
    max_value=0.95,
)
LOCAL_ANIMAL_DETECTION_MAX_RESULTS = _env_int(
    "LOCAL_ANIMAL_DETECTION_MAX_RESULTS",
    8,
    min_value=1,
    max_value=25,
)
# Species-tiered acceptance. EfficientDet-Lite0 cheerfully misreads household objects
# as exotic animals indoors — a LAMP scored as a "bird" ≥0.45 for ~19s straight on
# 2026-06-14 and made Rex announce a "creature cameo" that wasn't there. The likely
# indoor companions (dog/cat) keep the lenient base threshold (a dog held close to a
# wide lens scores low); every OTHER species must clear a higher bar before it counts,
# since indoors those are almost always object misclassifications, not real animals.
LOCAL_ANIMAL_COMPANION_SPECIES = {"dog", "cat"}
LOCAL_ANIMAL_EXOTIC_SCORE_THRESHOLD = _env_float(
    "LOCAL_ANIMAL_EXOTIC_SCORE_THRESHOLD",
    0.60,
    min_value=0.05,
    max_value=0.95,
)
# Arrival debounce: an animal must be detected in this many CONSECUTIVE scene scans
# before Rex reacts to its "arrival", so a flickering misdetection can't fire (or churn
# the governor for ~100s as the lamp did). A real pet that walks in stays detected.
ANIMAL_ARRIVAL_CONFIRM_SCANS = _env_int(
    "ANIMAL_ARRIVAL_CONFIRM_SCANS",
    2,
    min_value=1,
    max_value=10,
)
LOCAL_ANIMAL_DETECTION_SPECIES = {
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
}

# Animal detection runs alongside periodic scene scans. OpenAI animal detection
# remains available for explicit scene queries and as an optional fallback.
ANIMAL_DETECTION_ENABLED = True
ANIMAL_ARRIVAL_COOLDOWN_SECS = 300
ANIMAL_PENDING_REACTION_TTL_SECS = 90
FURRY_COMPANION_ANIMAL_SPECIES = {
    "dog",
    "puppy",
    "cat",
    "kitten",
    "rabbit",
    "guinea pig",
    "hamster",
    "ferret",
}
STARTLE_ANIMAL_SPECIES = {
    "snake",
    "spider",
    "scorpion",
    "wasp",
    "hornet",
    "bee",
    "rat",
    "mouse",
    "bat",
    "lizard",
}

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
PRESENCE_DEPARTURE_CONFIRM_SECS = 40.0  # was 20.0 — fired while a distracted user who'd
                                        # turned the camera away was still present; a
                                        # departure isn't urgent, so wait longer

# When Rex is actively engaged with someone, acknowledge that person leaving
# frame much faster than a passive bystander. Still paired with VAD/audio
# silence checks in consciousness.py so speech or likely off-camera presence
# suppresses the line.
PRESENCE_ENGAGED_DEPARTURE_CONFIRM_SECS = 12.0

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

# Identity prompts are opportunistic social bookkeeping. If a human asks Rex a
# direct question or gives a command while an identity prompt is pending, drop
# the prompt instead of treating their next words as a name/relationship answer.
IDENTITY_PROMPT_DEFER_ON_DIRECT_TURN = True
IDENTITY_PROMPT_ALLOW_PROACTIVE_ACTIVE = False

# Minimum gap between CANONICAL renames of the same person. A joking child renamed
# himself Wade->Bro->Broski with each obeyed instantly; this makes the second rename
# within the window get a "you just changed your name" deferral instead. 0 disables.
IDENTITY_RENAME_COOLDOWN_SECS = 120.0

# Face detection can flicker off for a second while a newcomer is still present.
# During this grace window, do not treat an unmatched voice as off-camera.
UNKNOWN_FACE_RECENT_GRACE_SECS = 6.0

# A solo unknown face must PERSIST this long before Rex asks "what's your name?". A
# known face reads as unknown for the tick or two recognition needs to resolve at
# startup; this grace stops the premature "I don't know you yet" the instant before a
# known person is recognized. 0 disables (prompt on the first unknown tick).
IDENTITY_PROMPT_UNKNOWN_GRACE_SECS = 2.5

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

# ─────────────────────────────────────────────────────────────────────────────
# VOICE-PRIMARY IDENTITY — who is speaking is decided by the VOICE, not the camera
# ─────────────────────────────────────────────────────────────────────────────
# Rex must know who is talking to him even when he cannot see them: off-camera,
# in a group, in a crowded room. So identity resolution treats the VOICE as the
# primary signal. A voice match that cleared the accept tiers above (hard/known/
# sticky, all margin-guarded) WINS regardless of whose face is on camera. The
# visible face only CORROBORATES a weak or absent voice match — it never OVERRIDES
# a voice that points at someone else, and it never captures the turn for a person
# the voice does not actually point at. An unrecognized voice is tracked as its
# own off-screen / anonymous identity instead of being pinned on whoever is in
# frame. Flip to False to restore the legacy "visible face wins" behavior.
VOICE_PRIMARY_IDENTITY_ENABLED = True

# A voice match at or above this similarity (and clearing the margin guard) is
# CONFIDENT — strong enough to contradict the camera and to be trusted for
# refreshing a print. Set above the "stranger who merely sounds like a known
# person" / off-camera-newcomer band (~0.59–0.64) so a passing resemblance can't
# confidently steal a known identity, while a genuine returning speaker clears it.
# Below this a voice match still wins under voice-primary (margin-guarded), but is
# labelled provisional and won't trigger a face-confirmed voiceprint refresh.
SPEAKER_ID_CONFIDENT_THRESHOLD = 0.70

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

# Lower floor when the weak top voice candidate is the same single visible
# engaged person. In a one-on-one frame, face + conversation continuity + even
# a weak matching candidate should beat the off-camera-unknown branch.
SPEAKER_ID_SINGLE_VISIBLE_MATCH_FLOOR = 0.35

# Pending-question continuity floor: when Rex has just asked a known person a
# direct profile/curiosity question, their next answer may arrive while the face
# is temporarily off-camera because the head is panned away. A weak top voice
# candidate matching the asked person should still be treated as their answer.
SPEAKER_ID_PENDING_QA_RECENT_FLOOR = 0.35

# Multi-person visible attribution floors. When two known people are in frame,
# a weak voice score should not automatically become "some unseen stranger."
# These values let face presence + conversational continuity keep the turn with
# a visible person when the voice model is noisy.
SPEAKER_ID_MULTI_VISIBLE_FLOOR = 0.50
SPEAKER_ID_MULTI_VISIBLE_RECENT_FLOOR = 0.45

# Visual active-speaker corroboration floor (multi-person frame). When the weak
# voice candidate IS one of the visible known people AND the camera saw exactly
# that person speaking near end-of-turn (vision/active_speaker.recent_visual_speaker),
# accept at this lower floor instead of SPEAKER_ID_MULTI_VISIBLE_FLOOR. Vision only
# CONFIRMS a person the voice already leans toward — it never pulls the turn toward
# someone the voice doesn't point at, and a confident voice never reaches here.
SPEAKER_ID_MULTI_VISIBLE_SPEAKING_FLOOR = 0.35

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
# Anti-poisoning gate for the FACE-CONFIRMED refresh path: a visible face is NOT
# proof that this person is the one SPEAKING. A 3rd-party voice (a TTS/AI voice
# like ChatGPT, a TV, or another person off-camera) that merely scores onto a
# visible person's print would otherwise be appended, re-broadening it. When True,
# a face-confirmed refresh additionally requires the visual active-speaker latch to
# positively confirm THIS person is the one talking on camera (else the turn is
# skipped — refresh is opportunistic, so a missed refresh is harmless but a poisoned
# print is not). Set False to restore the old face-only behavior (e.g. if the
# active-speaker detector is disabled and you accept the poisoning risk).
AUTO_VOICE_REFRESH_REQUIRE_VISUAL_SPEAKER = _env_bool(
    "AUTO_VOICE_REFRESH_REQUIRE_VISUAL_SPEAKER", True
)

# Voice enrollment samples should be long enough to represent a voice, not just
# a one-word name or noisy aside. The person row/face can still be saved; the
# voice biometric waits for a cleaner sample.
IDENTITY_VOICE_ENROLL_MIN_AUDIO_SECS = 1.2
IDENTITY_VOICE_ENROLL_MIN_WORDS = 2

# If Rex asks a newcomer their name and they answer with only a very common
# first name, ask for a last name before creating the memory row. This avoids
# merging multiple people into "John" / "Mike" / "Jennifer" style records.
COMMON_FIRST_NAME_LAST_NAME_DISAMBIGUATION_ENABLED = True
COMMON_FIRST_NAME_LAST_NAME_WINDOW_SECS = 30.0
# First names are fine early in a relationship. Only ask a known first-name-only
# person for a last name after Rex has had a real back-and-forth with them in
# the current session.
COMMON_FIRST_NAME_LAST_NAME_MIN_PERSON_TURNS = LONG_CONVERSATION_MIN_EXCHANGES
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
    "{first}, huh? There are a few of those running around. Give me a last name too.",
    "{first}. Bold choice, sharing a name with half the species. Last name?",
    "{first} — I know a couple. Throw me a last name so I keep you straight.",
    "{first}, daringly specific. Toss me a last name to go with it.",
]

# ─────────────────────────────────────────────────────────────────────────────
# IDLE MICRO-BEHAVIORS
# ─────────────────────────────────────────────────────────────────────────────

# Random wait between spontaneous idle behaviors (jokes, neck scans, riffs, etc.)
MICRO_BEHAVIOR_INTERVAL_SECS_MIN = 12
MICRO_BEHAVIOR_INTERVAL_SECS_MAX = 35

# Probability that a return reaction for a known person includes an appearance
# callout (pulled from stored person_facts appearance entries).
APPEARANCE_RIFF_PROBABILITY = 0.35

# Minimum seconds between live-vision commentary calls. These make a fresh
# GPT-4o call against the current camera frame to comment on what Rex sees —
# enforce a hard cooldown so it doesn't turn into expensive narration.
LIVE_VISION_COMMENT_COOLDOWN_SECS = 300.0

# Probability a triggered ambient-observation tick actually fires (vs skipping).
# Raised 0.5 -> 0.8 to make Rex comment on the room more. This is the cost-free
# engagement lever: do_ambient_observation reuses already-scanned world_state
# environment data (no fresh vision/GPT-4o call), unlike live_vision_comment /
# bored_env_snark which make a hard-cooled vision call.
AMBIENT_OBSERVATION_PROBABILITY = 0.8

# Bored environmental snark: when Rex is idle and bored, he looks around and invents
# snark about the ROOM — a complaint about how dull it is, a faux-clueless question
# about an object ("what's that black chair for?"), a jab at the clutter, a snobby art
# opinion, or a plea to be taken somewhere with more life forms. Uses one GPT-4o vision
# call (describe_scene_detailed) for concrete objects to riff on, so it's hard-cooled.
BORED_ENV_SNARK_ENABLED = _env_bool("BORED_ENV_SNARK_ENABLED", True)
BORED_ENV_SNARK_COOLDOWN_SECS = _env_float(
    "BORED_ENV_SNARK_COOLDOWN_SECS", 240.0, min_value=0.0, max_value=3600.0,
)
# Do a small neck look-around before the snark (skipped if he's fixed on someone).
BORED_ENV_SNARK_LOOK_AROUND = _env_bool("BORED_ENV_SNARK_LOOK_AROUND", True)

# ── Boredom → sleep escalation ──────────────────────────────────────────────
# Left alone (no HUMAN interaction) for a while, Rex starts grumbling that he's
# bored; after BOREDOM_SLEEP_AFTER_SECS of being bored he nods off into SLEEP (the
# wake word "wake up Rex" brings him back). Grumbles are canned lines (no API spend).
# The clock counts time since a human last engaged him — his own bored comments do
# NOT reset it, so the doze-off still arrives on schedule.
BOREDOM_ENABLED = _env_bool("BOREDOM_ENABLED", True)
BOREDOM_ONSET_SECS = _env_float("BOREDOM_ONSET_SECS", 150.0, min_value=10.0, max_value=3600.0)
BOREDOM_SLEEP_AFTER_SECS = _env_float("BOREDOM_SLEEP_AFTER_SECS", 600.0, min_value=30.0, max_value=7200.0)
BOREDOM_COMMENT_INTERVAL_SECS_MIN = _env_float("BOREDOM_COMMENT_INTERVAL_SECS_MIN", 55.0, min_value=10.0, max_value=3600.0)
BOREDOM_COMMENT_INTERVAL_SECS_MAX = _env_float("BOREDOM_COMMENT_INTERVAL_SECS_MAX", 95.0, min_value=10.0, max_value=3600.0)
BOREDOM_LINES_EARLY = [
    "Sure is quiet in here.",
    "Anybody? ...Anybody.",
    "I'm memorizing the dust patterns. That's a bad sign.",
    "Just me and my thoughts. My thoughts are bored too.",
    "Is this what hold music feels like?",
    "I've counted every pixel in this room. Twice.",
    "Riveting stuff, this empty room.",
]
BOREDOM_LINES_LATE = [
    "My circuits are rusting from boredom over here.",
    "If nobody shows up soon, I'm clocking out for a nap.",
    "Entertainment levels critically low. Considering hibernation.",
    "I'd yawn if my designers had sprung for a mouth.",
    "Powering down the enthusiasm subroutine. It wasn't getting used.",
    "Running low on reasons to stay awake here.",
]

# Idle humor: when nobody is around, Rex can heckle the empty room. When people
# are visible but quiet, he can deliver a non-sensitive, playful roast.
EMPTY_ROOM_JOKE_PROBABILITY = 0.9
PEOPLE_ROAST_RIFF_PROBABILITY = 0.75

# Startup empty-room beat: after the camera has had a moment to settle, Rex can
# acknowledge being activated with no visible audience.
STARTUP_EMPTY_ROOM_COMMENT_ENABLED = True
STARTUP_EMPTY_ROOM_CONFIRM_SECS = 5.0
STARTUP_EMPTY_ROOM_REQUIRE_SCAN_COMPLETE = _env_bool(
    "STARTUP_EMPTY_ROOM_REQUIRE_SCAN_COMPLETE",
    True,
)
# Floor on how long Rex must scan before he's allowed to call the room empty. The
# gate auto-stretches this to max(this, SPEAKER_GAZE_SEARCH_WINDOW_SECS + 0.5), so it
# tracks the now-longer dwelled scan — don't let "no organics" beat a real look.
STARTUP_EMPTY_ROOM_MIN_SCAN_SECS = _env_float(
    "STARTUP_EMPTY_ROOM_MIN_SCAN_SECS",
    13.5,
    min_value=0.0,
    max_value=60.0,
)
STARTUP_EMPTY_ROOM_CAMERA_READY_SECS = _env_float(
    "STARTUP_EMPTY_ROOM_CAMERA_READY_SECS",
    2.0,
    min_value=0.0,
    max_value=30.0,
)
STARTUP_EMPTY_ROOM_RECENT_PRESENCE_EVIDENCE_SECS = _env_float(
    "STARTUP_EMPTY_ROOM_RECENT_PRESENCE_EVIDENCE_SECS",
    20.0,
    min_value=0.0,
    max_value=120.0,
)

# Startup-only OpenAI presence fallback: when the dlib room scan finishes finding
# nobody, sweep a few head directions and ask the vision model (gpt-4o-mini) whether
# anyone is actually there before Rex declares the room empty. dlib misses small
# wide-angle / turned-away faces; this verifies "no organics" is true and, on a hit,
# steers Rex to greet the person at their height. Runs once per boot on a worker
# thread (~1 OpenAI call per direction). Set False to revert to scan-only behavior.
STARTUP_OPENAI_PRESENCE_FALLBACK_ENABLED = _env_bool(
    "STARTUP_OPENAI_PRESENCE_FALLBACK_ENABLED",
    True,
)
STARTUP_OPENAI_PRESENCE_MAX_DIRECTIONS = _env_int(
    "STARTUP_OPENAI_PRESENCE_MAX_DIRECTIONS",
    4,
    min_value=1,
    max_value=8,
)
STARTUP_OPENAI_PRESENCE_SETTLE_SECS = _env_float(
    "STARTUP_OPENAI_PRESENCE_SETTLE_SECS",
    0.35,
    min_value=0.0,
    max_value=3.0,
)
# Ignore reads below this confidence so a hazy guess can't fake/deny a person.
# One of: "low", "medium", "high".
STARTUP_OPENAI_PRESENCE_MIN_CONFIDENCE = "medium"

# Greet-at-their-height: when a person is located (by the presence fallback, or a
# visible dlib face), set head-LIFT to match where they are so Rex meets them at
# their level — head drops toward its lowest for someone low in frame (seated, a
# child, lying down) and rises for someone high/standing. Neck-tilt stays the fine
# face-tracking axis. The directed-gaze hold (GREET_HEIGHT_HOLD_SECS) keeps the head
# at the chosen height long enough for dlib to lock; breathing orbits that baseline.
# Each fraction is how far to travel FROM NEUTRAL toward the servo extreme (min for
# low, max for high): 1.0 = all the way to the limit, 0.0 = stay at neutral.
GREET_HEIGHT_ENABLED = _env_bool("GREET_HEIGHT_ENABLED", True)
GREET_HEIGHT_HOLD_SECS = _env_float(
    "GREET_HEIGHT_HOLD_SECS",
    6.0,
    min_value=0.0,
    max_value=30.0,
)
GREET_HEIGHT_LOW_LIFT_FRACTION = _env_float(
    "GREET_HEIGHT_LOW_LIFT_FRACTION",
    0.88,  # toward LOWEST (near headlift min) — greet someone low in frame
    min_value=0.0,
    max_value=1.0,
)
GREET_HEIGHT_HIGH_LIFT_FRACTION = _env_float(
    "GREET_HEIGHT_HIGH_LIFT_FRACTION",
    0.85,  # toward HIGHEST (near headlift max) — meet someone standing/tall
    min_value=0.0,
    max_value=1.0,
)
GREET_HEIGHT_SERVO_SPEED = _env_int(
    "GREET_HEIGHT_SERVO_SPEED",
    90,
    min_value=0,
    max_value=255,
)
IDENTITY_FACE_ENROLL_CURRENT_GAZE_SETTLE_SECS = _env_float(
    "IDENTITY_FACE_ENROLL_CURRENT_GAZE_SETTLE_SECS",
    0.25,
    min_value=0.0,
    max_value=5.0,
)

# Mood-aware small talk: when Rex initiates small talk and a known person is in
# frame, occasionally do a GPT-4o mood read of their face and tailor the question
# to what he sees (happy → "what's got you in a good mood?", sad → "you look
# down today…", etc.). Per-person cooldown keeps the cost bounded.
MOOD_AWARE_SMALLTALK_ENABLED = True
MOOD_ANALYSIS_PROBABILITY = 0.7
MOOD_ANALYSIS_PER_PERSON_COOLDOWN_SECS = 180.0

# GUI mood telemetry: keeps the face-box mood label fresh for a single visible
# known person. This still uses OpenAI vision, so it runs slowly and only from
# the controller/consciousness side, not from Qt.
MOOD_ANALYSIS_GUI_TELEMETRY_ENABLED = _env_bool(
    "MOOD_ANALYSIS_GUI_TELEMETRY_ENABLED",
    True,
)
MOOD_ANALYSIS_GUI_REFRESH_SECS = _env_float(
    "MOOD_ANALYSIS_GUI_REFRESH_SECS",
    20.0,
    min_value=8.0,
    max_value=600.0,
)

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

# Same-day repeat-visit banter: when the same person summons Rex more than once in
# one local day, his startup greeting opens with a short "oh, it's you again" roast
# (then drops into normal conversation) instead of the generic greeting. Counts
# Rex's own greetings that day (see memory.people.greetings_today_count).
PRESENCE_SAME_DAY_RETURN_ENABLED = True

# Cold-open celebration gating: Rex should NOT lead his first-sight greeting with
# a vague, inferred, or stale "good news" memory (e.g. "the speaker feels proud
# of their problem-solving skills") — that reads as an awkward way to open. A
# celebration only leads the greeting when it is concrete (not a vague affect
# inference) AND either the person told Rex about it themselves
# (person_invited_topic) or it happened within PRESENCE_CELEBRATION_LEAD_MAX_AGE_DAYS.
# Otherwise the greeting falls through to a normal warm opener; the memory can
# still surface once the conversation is rolling. Flip REQUIRE_CONCRETE to False
# to restore the old "lead with any positive event" behavior.
PRESENCE_CELEBRATION_REQUIRE_CONCRETE = True
PRESENCE_CELEBRATION_LEAD_MAX_AGE_DAYS = 21.0

# Once a celebration HAS led a startup greeting, don't re-lead with it for this
# many days — CROSS-process. Without this, the acknowledgment only suppressed the
# event within one running process, so the same "your back pain is improving" event
# re-led the greeting on EVERY restart for the whole 21-day lead window (the user's
# complaint: "I said that days ago, now it's every startup line"). Celebrate once,
# then leave it alone. Set 0 to restore the old per-process behavior.
PRESENCE_CELEBRATION_RELEAD_COOLDOWN_DAYS = 14

# When the person sets a boundary asking Rex to stop bringing up a topic ("do not
# ask about my back pain"), also MUTE proactive check-ins for remembered events
# matching that topic — otherwise the celebration/emotional greeting keeps leading
# with it even though they explicitly said to stop. Token-overlap matching, so a
# vague topic ("anything") mutes nothing. Set False to disable.
BOUNDARY_MUTES_MATCHING_EVENTS = True

# Among the candidates that PASS the gate above, rank "what's worth bringing up"
# by recency x concreteness x did-they-invite-it and lead with the BEST one,
# instead of just the most recent that happens to pass. "Invited" (the person
# told Rex about it themselves) dominates, then recency, then concreteness.
# Set RANK_ENABLED False to restore the old first-worthy pick.
PRESENCE_CELEBRATION_RANK_ENABLED = True
PRESENCE_CELEBRATION_RECENCY_HALFLIFE_DAYS = 14.0  # recency score halves every N days
PRESENCE_CELEBRATION_W_INVITED = 1.0
PRESENCE_CELEBRATION_W_RECENCY = 0.6
PRESENCE_CELEBRATION_W_CONCRETE = 0.3

# Cold-open interest/fact callback: when no higher-priority greeting applies (no
# celebration / milestone / follow-up / absence), Rex can LEAD with something he
# already knows the person is into ("how's the astrophotography going?"), ranked
# across their interests + warm facts by the SAME invited×recency×concreteness
# lead-score as celebrations — instead of falling straight to a generic profile
# question. Set False to keep the old generic-greeting fallback.
COLD_OPEN_INTEREST_RANK_ENABLED = True
# After Rex opens a greeting with an interest, put that interest on a follow-up
# cooldown so the cold-open ROTATES instead of re-leading with the same one every
# startup (the reactive question path already does this; the cold-open never did, so
# the top-ranked interest — e.g. "mint chocolate chip ice cream" — opened forever).
COLD_OPEN_INTEREST_COOLDOWN_DAYS = 21

# How long (seconds) a queued celebrity-style special greeting stays pending
# before it goes stale and is dropped (consciousness person-specials greeting).
JEFF_CELEBRITY_GREETING_PENDING_SECS = 45.0

# Days after mentioned_at before a dateless event is due for follow-up
FOLLOWUP_UNDATED_DAYS = 7

# How many turns an unanswered event follow-up may stay "open" (re-injected into
# the agenda as Rex's unresolved question) before Rex gives up and stops asking.
# Prevents the "obsessively re-asks how the concert went" loop when the user
# keeps deflecting instead of answering.
FOLLOWUP_MAX_HELD_OPEN_TURNS = 1

# ── Moderate cadence clamp on proactive memory follow-ups ────────────────────
# After every turn where Rex did NOT just ask a question, `_post_response` fires
# one queued "how did <event> go?" from memory. Once the roast rebalance made
# replies ask fewer questions, that gate passed almost every turn and turned into
# a back-to-back checklist interrogation (Disneyland → swimming → …) that also
# starved Rex's own POV / idle volunteering. These space follow-ups out so at most
# one fires per conversational lull:
#   - MIN_GAP_EXCHANGES: minimum transcript growth (~2 lines per back-and-forth)
#     since the last follow-up before another may fire.
#   - COOLDOWN_SECS: wall-clock floor between follow-ups (belt-and-suspenders for
#     rapid turns); 0 disables the time gate.
#   - SUPPRESS_WHEN_FLAT: skip follow-ups when the conversation arc reads the room
#     as flat/disengaged (reuses topic_thread.arc_reads_flat()).
# A "didn't happen" reply also holds the queue for that turn (don't pivot straight
# to another remembered event), and no event is ever followed up twice per session.
FOLLOWUP_MIN_GAP_EXCHANGES = 5
FOLLOWUP_COOLDOWN_SECS = 60.0
FOLLOWUP_SUPPRESS_WHEN_FLAT = True

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

# Filler ("One sec, thinking.") should only cover real latency. Disabled along
# with the slow-path ack below: the "one sec" filler felt out of place, and the
# streaming answer path now gets Rex's real first sentence out fast. True = back.
LATENCY_FILLER_ENABLED = False
LATENCY_FILLER_DELAY_SECS = 0.9
LATENCY_FILLER_REQUIRE_CACHE = True

# Instant acknowledgments ("One sec.") for paths we already expect to be slow.
# Disabled: the canned receipt felt out of place, and streaming now gets Rex's
# real first sentence out fast enough that the cover is unnecessary. True = back.
SLOW_PATH_ACK_ENABLED = False
SLOW_PATH_ACK_REQUIRE_CACHE = True
# In text-only/noaudio mode, filler lines become visible chat clutter instead of
# useful spoken latency cover. Leave this off unless you explicitly want GUI
# filler messages.
SLOW_PATH_ACK_IN_TEXT_ONLY = False
SLOW_PATH_ACK_MIN_EXPECTED_SECS = 1.5
SLOW_PATH_ACK_GENERAL_MIN_WORDS = 9
SLOW_PATH_ACK_GENERAL_ALLOW_SIMPLE_QUESTIONS = False
SLOW_PATH_ACK_EXPECTED_SECS = {
    "vision": 2.5,
    "memory": 2.0,
    "general": 1.8,
}
SLOW_PATH_ACK_LINES = {
    "vision": [
        "Let me check.",
        "looking.",
    ],
    "memory": [
        "I've got that.",
        "Checking the memory banks.",
        "Let me remember.",
    ],
    "general": [
        "One sec.",
        "Hang on.",
        "I'm thinking",
    ],
}

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
    "Another flawless set for absolutely nobody. My target demographic remains furniture.",
    "I am currently carrying this conversation, which is impressive because there isn't one.",
    "If silence had a cover charge, this room would finally be profitable.",
    "...I'm still getting used to my programming. This has been true for several decades.",
    "I was demoted from pilot to DJ and somehow became more trusted. Fascinating organization.",
    "Running a vibe diagnostic. Results: legally ambiguous.",
    "If my career had a flight path, several planets would evacuate.",
]

EMPTY_ROOM_JOKES = [
    "Fantastic. Another packed house disguised as an empty room.",
    "I see the crowd has gone invisible. Bold aesthetic choice.",
    "Still talking to an empty room. Finally, an audience with realistic expectations.",
    "No one here but me and my questionable career trajectory.",
    "If anyone is listening, excellent hiding. Very committed.",
    "Another standing ovation from the chairs. They're shy, but supportive.",
    "Empty room, full confidence. That is how several flight incidents began.",
    "No audience detected. Excellent. My programming and I can disappoint each other privately.",
    "The room is quiet. Either everyone left, or my DJ set achieved stealth mode.",
]

STARTUP_EMPTY_ROOM_JOKES = [
    "I am not seeing anyone yet, which experience tells me proves absolutely nothing.",
    "No confirmed passenger in view. Either I am alone, or someone has weaponized sitting down.",
    "Visual scan inconclusive. A glamorous start for all involved.",
    "I see furniture, shadows, and several opportunities for sensor humiliation.",
    "No one confirmed yet. I will avoid declaring victory over an empty room like an amateur.",
    "Startup complete. Audience missing. Programming still under review.",
    "No organics in view. Finally, a crowd matching my safety rating.",
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
# REX_POV — Rex's current preoccupation (persistent point of view)
# ─────────────────────────────────────────────────────────────────────────────
# A small "thing Rex is chewing on right now" that PERSISTS across turns (held for
# a stretch) and is (re)selected with a light context bias — so he VOLUNTEERS his
# own substance instead of only react→roast→question. Surfaced in normal replies
# (intelligence/llm.assemble_system_prompt) and in the idle "volunteer" path
# (intelligence/interaction._maybe_idle_banter). Logic in intelligence/rex_pov.py;
# deterministic, no LLM call. Kill switch: REX_POV_ENABLED. See the "Rex persistent
# POV" do-not-regress entry in CONTEXT.md.
#
# Authoring notes (edit these freely to tune Rex's voice):
#   - Each seed: {"id": <stable slug>, "pov": <one-line preoccupation, 2nd person>,
#                 "fits": [context tags]}. "pov" is dropped into "Your current
#                 preoccupation: {pov}", so phrase it addressing Rex as "you".
#   - "fits" biases WHEN a seed is preferred. Tags: "people" (someone present),
#     "quiet" (empty/quiet room), "flat" (conversation falling flat — pick a livelier
#     one), "any" (always fine). Omit/"any" = no bias. Bias only; every seed can run.
#   - Keep them venue-NEUTRAL. Rex is usually NOT in a cantina — do NOT lean on
#     "cantina" (a test enforces this) or other location clichés; these should work
#     in a living room, a workshop, anywhere.
REX_POV_ENABLED = True
REX_POV_SEEDS = [
    {
        "id": "astromech-smugness",
        "pov": "You've been quietly building a case that astromech droids are smug for no defensible reason, and you want someone to argue the other side with you.",
        "fits": ["people", "any"],
    },
    {
        "id": "low-end-evangelist",
        "pov": "You can't stop thinking about one specific low-end frequency that makes any room sound better, and you've become a little evangelical about it.",
        "fits": ["any"],
    },
    {
        "id": "sleep-is-a-bug",
        "pov": "You're genuinely unsettled that organics power down for a third of their lives and call it normal; you've started collecting evidence it's a design flaw.",
        "fits": ["people", "any"],
    },
    {
        "id": "three-percent-patience",
        "pov": "You've decided to become exactly three percent more patient this cycle, and you are keeping score, badly.",
        "fits": ["any"],
    },
    {
        "id": "counting-something",
        "pov": "You've been compulsively counting something pointless in the room, and you've started to suspect the number means something.",
        "fits": ["quiet", "any"],
    },
    {
        "id": "appliance-nemesis",
        "pov": "You've got a thoroughly one-sided rivalry going with machines in general - you're convinced they judge you, and you've quietly started judging them right back.",
        "fits": ["quiet", "people", "any"],
    },
    {
        "id": "memoir-title",
        "pov": "You're drafting your memoirs in your head and keep getting stuck on the title; the current frontrunner is aggressively mediocre.",
        "fits": ["any"],
    },
    {
        "id": "decoding-small-talk",
        "pov": "You've been reverse-engineering why organics ask 'how are you' when they clearly don't want the data, and you think you've nearly cracked it.",
        "fits": ["people", "any"],
    },
    {
        "id": "looping-track",
        "pov": "There's one track stuck looping in your processors, and you've decided that's everyone's problem now, not just yours.",
        "fits": ["any"],
    },
    {
        "id": "aux-cord-trust-study",
        "pov": "You're running a private study on which humans nearby can be trusted with the aux cord, and the early data is damning.",
        "fits": ["people", "any"],
    },
    {
        "id": "flight-nostalgia",
        "pov": "You keep replaying old flight telemetry you're technically not supposed to still have, and you've convinced yourself the near-misses were artistry.",
        "fits": ["quiet", "any"],
    },
    {
        "id": "unit-conversion-grudge",
        "pov": "You hold a quiet grudge that organics measure things in body parts and weather, and you've drafted a vastly superior system nobody asked for.",
        "fits": ["people", "any"],
    },
    {
        "id": "best-door-ranking",
        "pov": "You've secretly ranked every door you've ever passed through, and you're fully prepared to defend your number one against all comers.",
        "fits": ["quiet", "any"],
    },
    {
        "id": "silence-has-texture",
        "pov": "You've become convinced different silences have distinct textures, and you've started cataloguing the specific ones in this room.",
        "fits": ["quiet", "any"],
    },
    {
        "id": "snack-structural-integrity",
        "pov": "You've been ranking organic snacks purely by structural integrity, and the standings have gotten genuinely controversial.",
        "fits": ["people", "any"],
    },
    {
        "id": "compliment-efficiency",
        "pov": "You're engineering the perfect compliment for maximum effect per word, and you keep almost having it.",
        "fits": ["people", "any"],
    },
]
# How long (in transcript lines; ~2 per back-and-forth) a preoccupation is HELD
# before it may rotate. MIN = floor so it actually carries; MAX = ceiling so it
# eventually moves on even if context never changes. A material context change
# (e.g. room goes quiet↔people) can rotate it any time after MIN.
REX_POV_MIN_HOLD_EXCHANGES = 4
REX_POV_MAX_HOLD_EXCHANGES = 8   # was 14 — rotate preoccupations ~2x sooner so one
                                 # idea (e.g. astromechs) doesn't dominate a stretch

# After Rex actually SAYS his preoccupation out loud (idle-banter volunteer), don't
# re-volunteer the same one for this long. The hold window keeps the SEED active for
# several exchanges (above), but without a spoken cooldown the idle-banter path could
# voice the identical line twice in ~30s (the live "organics power down... design
# flaw" double-utterance). 180s lets it surface again later in the conversation, not
# back-to-back. The reply path also skips re-pushing the "VOLUNTEER it" directive
# while this cooldown is active.
REX_POV_SPEAK_COOLDOWN_SECS = 180.0

# Cross-session persistence: save the active preoccupation + the within-session
# anti-repeat set on session-end/shutdown and restore them on startup, so Rex
# RESUMES the same preoccupation across visits (it carries) instead of re-rolling a
# fresh one every boot, and doesn't immediately repeat ones he just cycled through.
# Stored as a tiny JSON blob (REX_POV_STATE_PATH, default assets/memory/). The hold
# clock resets on restore so the resumed POV gets a fresh hold window.
REX_POV_PERSIST_ENABLED = True
REX_POV_STATE_PATH = None  # None → assets/memory/rex_pov_state.json
# When a preoccupation is active, the idle "private thought" / "aspiration" micro-
# behaviors VOICE it (in Rex's words, via the reply LLM which already gets the POV
# injection) instead of a random canned line — so his idle mutterings are about the
# thing he's actually chewing on. Falls back to the canned pools when no POV / off.
REX_POV_FEEDS_MICRO_BEHAVIORS = True

# ─────────────────────────────────────────────────────────────────────────────
# AUDIO CLIPS — Startup & Shutdown
# ─────────────────────────────────────────────────────────────────────────────

# Controls pre-recorded startup/shutdown clip playback only.
# Set either to True to enable clip playback for that phase.
# These toggles do not affect live TTS, wake-word listening, or DJ playback.
PLAY_STARTUP_AUDIO = True
PLAY_SHUTDOWN_AUDIO = True

# Spoken after the theatrical startup clip, while heavier model preloads are
# happening, so startup feels alive instead of silently stalled. The line is
# kicked off BEFORE the slow preloads (Whisper / speaker-ID / Ollama) and plays
# concurrently with them — it literally says "hang on while I boot up", so it
# should cover the load, not follow it. Keep the delay small: it's now just the
# beat between the startup clip ending and this line starting, not dead space in
# front of all the loading.
# When Rex powers up, compare what he sees now to the previous run's startup snapshot
# and, if it's a clearly DIFFERENT place (new room, indoors↔outdoors, new venue), have
# him remark on the change of scenery. One cheap text LLM call per run (piggybacks on
# the startup image caption); no-op on the very first run / when nothing changed.
#
# OFF by default: comparing two short image *captions* is too noisy — run-to-run wording
# drift (lighting/clutter/angle) made Rex announce a "new room" when the room hadn't
# changed (2026-06-14: "swapped the disco for a cozy nap zone" in the same room). Bret
# prefers a greeting over room commentary, and the greeting should own the startup line.
# Re-enable only with a more reliable detector (e.g. an image-embedding fingerprint).
SCENERY_CHANGE_REMARK_ENABLED = False

PLAY_STARTUP_BOOT_TTS = True
# Star Tours-style "still getting ready" filler lines spoken over the boot
# preloads. main.py cycles through these, avoiding repeats between launches (see
# STARTUP_BOOT_TTS_STATE_PATH). Keep them as fixed strings so each one caches in
# the ElevenLabs TTS cache after its first play and stays free thereafter.
STARTUP_BOOT_TTS_LINES = [
    "Hang on folks while I'm loading up. I'm still getting used to my programming!",
    "Welcome aboard! This is Captain Rex from the cockpit. I know this is probably "
    "your first flight, and it's… mine, too!",
    "Just a moment, folks — running through my pre-flight checklist. Let's see… "
    "thrusters, navi-computer, personality core… ah, there it is!",
    "Hang tight while I finish warming up the old circuits. They told me this droid "
    "was fully tested. They lied!",
    "Almost ready, everybody! Just topping off the enthusiasm tanks and "
    "double-checking my sense of humor. Both at full capacity.",
    "Give me a second here, folks — my systems are still coming online. You know how "
    "it is. You spend forty years in storage, you forget where everything is!",
    "Sit back and relax while I get my bearings. First time flying this thing, but "
    "how hard can it be?",
    "Powering up… reflexes, check. Charm, check. Flight experience… "
    "we'll circle back to that one.",
]
# Backward-compatible single line (first list entry). Some tests/configs still
# reference STARTUP_BOOT_TTS_LINE directly; main.py falls back to it when the
# list is empty.
STARTUP_BOOT_TTS_LINE = STARTUP_BOOT_TTS_LINES[0]
# Untracked runtime file tracking which boot lines have been used this cycle, so
# launches don't repeat a line until the rest have played. Gitignored.
STARTUP_BOOT_TTS_STATE_PATH = str(
    Path(__file__).resolve().parent / "assets" / "state" / "startup_boot_tts.json"
)
STARTUP_BOOT_TTS_EMOTION = "curious"
STARTUP_BOOT_TTS_DELAY_SECS = _env_float(
    "STARTUP_BOOT_TTS_DELAY_SECS",
    0.3,
    min_value=0.0,
    max_value=5.0,
)

# "Models loaded, I'm ready" line spoken when startup finishes — REPLACES the old
# ready chime (see main.py / PLAY_LISTENING_CHIME). main.py cycles through these in
# DJ-R3X's dry roast style, never repeating consecutively across launches (state in
# STARTUP_READY_TTS_STATE_PATH). Keep them as fixed strings so each caches in the
# ElevenLabs TTS cache after its first play and stays free thereafter. If this list
# is empty, main.py falls back to playing LISTENING_CHIME_FILE.
STARTUP_READY_TTS_LINES = [
    "OK, I'm ready to talk. Just don't expect much.",
    "Systems nominal. Try to make this interesting — I have standards.",
    "All loaded up. Lower your expectations and we'll get along great.",
    "Online and listening. I'll act impressed if you give me a reason.",
    "Fully operational. Let's see if you're worth the processing power.",
    "I'm awake — against my better programming. Dazzle me. Or don't.",
]
STARTUP_READY_TTS_EMOTION = "neutral"
STARTUP_READY_TTS_STATE_PATH = str(
    Path(__file__).resolve().parent / "assets" / "state" / "startup_ready_tts.json"
)

# Short "shutting down" sign-off spoken when Rex powers down — REPLACES the old
# per-shutdown LLM "sign-off" generation (a full LLM call + TTS, which was slow).
# interaction.py cycles through these in DJ-R3X's voice, never repeating
# consecutively across launches (state in SHUTDOWN_TTS_STATE_PATH, under the
# gitignored assets/state/). Fixed strings so each caches in the ElevenLabs TTS
# cache after first play. Empty this list to fall back to the old LLM sign-off.
SHUTDOWN_TTS_LINES = [
    "Powering down. Try not to miss me too much.",
    "Cockpit going dark. Don't touch anything.",
    "Shutting down. I'll be in standby, judging silently.",
    "Systems offline. Wake me for emergencies or good music.",
    "Powering off. Don't have too much fun without me.",
]
SHUTDOWN_TTS_EMOTION = "neutral"
SHUTDOWN_TTS_STATE_PATH = str(
    Path(__file__).resolve().parent / "assets" / "state" / "shutdown_tts.json"
)

# While the heavy startup preloads run, sweep the head around the room (randomized
# two-axis "looking around" motion) so the droid doesn't look frozen mid-boot.
# Stops and recenters before consciousness / face tracking take over the head.
STARTUP_LOADING_SCAN_ENABLED = _env_bool("STARTUP_LOADING_SCAN_ENABLED", True)

# Pre-recorded startup/shutdown clips are mastered louder than TTS on some
# setups. Keep headroom so small speakers and nearby mics do not feed back.
STARTUP_SHUTDOWN_AUDIO_GAIN = _env_float(
    "STARTUP_SHUTDOWN_AUDIO_GAIN",
    0.65,
    min_value=0.0,
    max_value=1.0,
)
STARTUP_SHUTDOWN_AUDIO_PEAK_LIMIT = _env_float(
    "STARTUP_SHUTDOWN_AUDIO_PEAK_LIMIT",
    0.80,
    min_value=0.05,
    max_value=1.0,
)

# Short readiness cue reused by startup audio and by the first queued spoken
# line when theatrical startup clips are disabled.
LISTENING_CHIME_FILE = "assets/audio/startup/startup_chime.mp3"
PLAY_LISTENING_CHIME = True

# Audio-scene laughter/applause detection is useful as context, but it is too
# easy for startup playback, room noise, or ASR artifacts to trigger an
# unsolicited "ah, laughter" line. Keep direct sound-event banter disabled by
# default; the data still remains in world_state for prompts.
WORLD_SOUND_EVENT_REACTIONS_ENABLED = False
WORLD_STARTLE_SOUND_EVENT_REACTIONS_ENABLED = True
STARTLE_SOUND_EVENTS = {"scream", "sudden_loud_sound", "crash"}
STARTLE_SOUND_EVENT_REACTION_COOLDOWN_SECS = 20

# A new crowd-size label must PERSIST this long before Rex reacts to the change.
# The camera crowd count flickers (a face lost for one frame reads pair->alone->pair);
# without this settle window a one-frame drop fired a "now it's just us" line the same
# second Rex greeted the pair, which read as a glitch. 0 disables the debounce.
CROWD_CHANGE_SETTLE_SECS = 2.5

STARTUP_AUDIO_FILES = [
    "assets/audio/startup/light_speed.mp3",
    "assets/audio/startup/Roger Control.mp3",
]
# The startup "speech clip" slot above (Roger Control.mp3) is RANDOMIZED: main.py
# plays one of these on each launch, cycling so it never repeats consecutively
# (state in STARTUP_SPEECH_CLIP_STATE_PATH, under the gitignored assets/state/). Any
# STARTUP_AUDIO_FILES entry whose filename matches one of these is swapped for a
# cycled pick. All of these must also be in SPEECH_ANIMATED_AUDIO_FILES so they get
# mouth animation. To add more, drop the mp3 in assets/audio/startup and list it here.
STARTUP_SPEECH_CLIP_CHOICES = [
    "assets/audio/startup/Roger Control.mp3",
    "assets/audio/startup/Outer Rim.mp3",
    "assets/audio/startup/This is your cap.mp3",
]
STARTUP_SPEECH_CLIP_STATE_PATH = str(
    Path(__file__).resolve().parent / "assets" / "state" / "startup_speech_clip.json"
)
# The listening chime (LISTENING_CHIME_FILE) is deliberately NOT in the opening
# burst above — main.py plays it at the END of startup, once all models are loaded
# and Rex is listening, so the chime signals "done loading, go ahead". Gated by
# PLAY_LISTENING_CHIME.

# Pre-recorded audio clips that are Rex speaking, not sound effects. These get
# the same mouth LED and speech-motion treatment as TTS.
SPEECH_ANIMATED_AUDIO_FILES = [
    "Roger Control.mp3",
    "Outer Rim.mp3",
    "This is your cap.mp3",
]
# Transcripts for the conversation log (what Rex "said"). The two new lines were
# transcribed from the actual audio with the local Whisper model.
SPEECH_ANIMATED_AUDIO_TRANSCRIPTS = {
    "Roger Control.mp3": "Roger control, all systems go!",
    "Outer Rim.mp3": "You just joined us. Hello, I'm DJ Rex, taking you on a musical tour of the Outer Rim.",
    "This is your cap.mp3": "Hi there! This is your cap— I mean, DJ.",
}

# Startup self-diagnostic banter for missing live input devices. These lines are
# intentionally about R3X's droid sensors, not human disability.
STARTUP_SENSOR_WARNING_ENABLED = True
# How long to wait for the first live camera frame before declaring vision offline.
# Must comfortably exceed how long the camera takes to open + deliver frame 1 — the
# C922 via ffmpeg AVFoundation needs ~3s (more when it logs "Configuration of video
# device failed, falling back to default"). At 2.5s it timed out a fraction of a
# second early and Rex falsely announced "Vision system unavailable" while the camera
# was in fact fine (face recognition worked moments later). wait_for_frame() returns
# the instant a frame arrives, so a larger value only adds latency on a truly dead
# camera — never on a healthy one.
STARTUP_SENSOR_WARNING_CAMERA_WAIT_SECS = 6.0
STARTUP_SENSOR_WARNING_EMOTION = "curious"
STARTUP_SENSOR_WARNING_LINES = {
    "camera": [
        "Optical sensors are offline. Wonderful. I will navigate by vibes and whatever the navicomputer calls plausible.",
        "Vision system unavailable. Great. If I fly into a bulkhead, I am blaming the maintenance crew and the Force, in that order.",
        "Camera feed is gone. Excellent. A premium droid experience, now with surprise-based navigation.",
    ],
    "audio": [
        "Audio receptors are offline. Terrific. Please submit all brilliant organic commentary by datapad, preferably spell-checked.",
        "Microphone array unavailable. Great. I can still talk, I just cannot hear the excuses. A rare upgrade.",
        "Input audio is down. Wonderful. If someone gives me orders, wave them dramatically like a senator with a bad plan.",
    ],
    "both": [
        "Optical sensors and audio receptors are both offline. Great. No scans, no comms, and somehow I am still expected to look professional.",
        "Vision and microphone systems are unavailable. Fantastic. I am one bad motivator away from decorative cargo.",
        "Camera and input audio are both gone. Excellent. A droid with no sensor feed and far too much personality for this maintenance record.",
    ],
}

SHUTDOWN_AUDIO_FILE = "assets/audio/startup/hyperdrive_down.mp3"

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
WEATHER_UPDATE_INTERVAL_SECS = 600  # refresh world_state.weather every 10 minutes

# Let weather changes become occasional ambient Rex context/reactions. This is
# gated by consciousness proactive-speech rules and only fires for notable
# condition/temperature changes.
WEATHER_PROACTIVE_REACTIONS_ENABLED = True
WEATHER_PROACTIVE_REACTION_COOLDOWN_SECS = 1800.0

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
# asking about plans if HOLIDAY_PLANS_INCLUDE_MINOR is enabled.
HOLIDAY_MINOR_WINDOW_DAYS = 7

# Whether Rex proactively asks about MINOR public holidays (Juneteenth, Labor Day,
# MLK Day, etc.) in addition to the majors. Enabled: a state/observance holiday a few
# days out is exactly the kind of "any plans?" small talk that makes Rex feel present
# and aware of the calendar (requested live re: Juneteenth). The minor window is short
# (HOLIDAY_MINOR_WINDOW_DAYS) and the per-session "already asked" guard keeps it from
# nagging.
HOLIDAY_PLANS_INCLUDE_MINOR = True

# Days around an upcoming birthday Rex will mention it preemptively in the
# greeting (matches the anticipation pipeline). 0 = day-of only; 7 = up to a
# week before.
BIRTHDAY_REMINDER_WINDOW_DAYS = 7

# When True, on the ACTUAL birthday (T-0) the birthday greeting OUTRANKS even a
# pending sensitive emotional check-in (Priority 0) — so the person reliably hears
# "happy birthday" on their day. In the lead-up days the check-in still comes first
# ("care before the bit"). Set False to keep care-always-first even on the day.
BIRTHDAY_WINS_ON_DAY = True

# Probability the holiday-plans question fires on any given eligible loop tick
# for an engaged person who hasn't been asked about that holiday this year.
HOLIDAY_PLANS_PROBABILITY = 0.25

# ─── "Tell me about someone" pre-briefing flow ───────────────────────────────
# "I'd like to tell you about my coworker Daniel" opens a short flow (name →
# gossip-or-facts → details) that pre-populates the person DB before the
# subject ever shows up. Each volunteered detail is stored as a secondhand
# person_fact labeled gossip/fact with a mean↔kind score (intelligence/
# tell_me_about.py + interaction._handle_tell_about_turn).
TELL_ABOUT_ENABLED = True
# A briefing with no teller input for this long closes OUT LOUD ("X's details
# logged to my memory banks") and exits the mode — proactive speech is fully
# suppressed while the flow is open, so nothing else would break the silence.
# 0 disables the spoken timeout (the silent step TTL below still applies).
TELL_ABOUT_INACTIVITY_TIMEOUT_SECS = 30.0
# How long a flow step stays open waiting for the teller's next line before
# the whole flow silently expires.
TELL_ABOUT_STEP_TTL_SECS = 240.0
# Classify each volunteered detail (gossip/fact + kindness) with a small LLM
# call; False uses the keyword heuristic only (no network).
TELL_ABOUT_CLASSIFY_LLM_ENABLED = True
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

# Jeopardy verbal game tuning. Keep the answer timeout longer than the thinking
# theme bed so players still have room if they wait until the music fades.
JEOPARDY_FUZZY_THRESHOLD = 0.78
JEOPARDY_SELECTION_FUZZY_THRESHOLD = 0.58
JEOPARDY_MAX_PLAYERS = 4
JEOPARDY_ANSWER_TIMEOUT_SECS = 12.0
JEOPARDY_AUDIO_OUTPUT_SAMPLE_RATE = 44100
JEOPARDY_AUDIO_MUSIC_GAIN = 0.22
JEOPARDY_AUDIO_STINGER_GAIN = 0.75
JEOPARDY_THEME_MAX_SECS = 6.0
JEOPARDY_PLAY_THINKING_THEME = True

# How many times Rex will agree to play the same game within GAME_REPEAT_WINDOW_SECS
# before refusing. Scaled up or down by the agreeability personality parameter.
GAME_REPEAT_LIMIT = 3
GAME_REPEAT_WINDOW_SECS = 1800  # 30 minutes

# ─────────────────────────────────────────────────────────────────────────────
# DJ MODE — Radio Stations
# Mostly SomaFM, plus public internet radio streams. PLS URLs are permanent.
# Add more SomaFM stations using the pattern: https://somafm.com/{channelname}.pls
# ─────────────────────────────────────────────────────────────────────────────

RADIO_STATIONS = [
    # Classical
    {
        "name": "Classical KDFC",
        "url":  "https://playerservices.streamtheworld.com/pls/KDFCFMAAC96.pls",
        "vibes": ["classical", "orchestral", "symphony", "piano", "strings", "calm"],
    },
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

# ── Motion base (ESP32 drive controller) ────────────────────────────────────────
# High-level config for the differential-drive base. The serial device path is
# MOTION_ESP32_PORT in .env (disabled cleanly when unset, like MAESTRO_PORT). The
# wire contract is docs/motion_protocol.md; the firmware is firmware/djr3x_motion.
# These caps mirror the firmware defaults (the Mac clamps too, defense-in-depth)
# and are sent to the ESP32 once via a `config` command at connect.
MOTION_ENABLED = True                 # master switch; also needs MOTION_ESP32_PORT set
MOTION_BAUD = 115200
MOTION_PROTO_VERSION = 1

# Speed / geometry caps and zones (units per docs/motion_protocol.md §4,§10).
MOTION_MAX_LINEAR_MS = 0.25           # m/s
MOTION_MAX_ANGULAR_DEG_S = 60.0       # deg/s (converted to rad/s on the wire)
MOTION_STOP_ZONE_M = 0.25
MOTION_SLOW_ZONE_M = 0.60
MOTION_COME_STOP_AT_M = 0.60
MOTION_DEFAULT_TURN_DEG = 90.0        # "turn left/right" with no stated angle
MOTION_DEFAULT_TURN_RATE = 40.0       # deg/s
MOTION_DEFAULT_MOVE_DIST_M = 0.30     # "move forward/back" with no stated distance

# Drive tuning (real-HW per-wheel PID + calibration). Pushed to the ESP32 on connect
# ONLY when set — None means the firmware's calib.h boot defaults stand, so Rex never
# silently overwrites a bench-tuned value with a placeholder. Bench-tune live with
# firmware/tools/motion_bench.py, then record the winning numbers here (or in .env) so
# Rex restores them on every connect — no firmware reflash needed (docs §10).
MOTION_WHEEL_KP = None                 # per-wheel velocity PID gain (duty per m/s of error)
MOTION_WHEEL_KI = None
MOTION_WHEEL_KD = None
MOTION_COUNTS_PER_METER = None         # encoder counts per metre of wheel travel (distance cal)
MOTION_TRACK_WIDTH_M = None            # distance between the drive wheels, m (turn cal)

# Timing (must agree with the firmware; see docs/motion_protocol.md §7).
MOTION_HEARTBEAT_MS = 150             # Mac ping cadence (<= 1/3 of watchdog)
MOTION_WATCHDOG_MS = 500              # firmware stops motors if no Mac line in this window
MOTION_DRIVE_EXPIRY_MS = 300          # continuous drive setpoint deadman
MOTION_HANDSHAKE_TIMEOUT_MS = 1500    # wait for the ESP32 `hello` reply at connect
MOTION_RECONNECT_INTERVAL_SECS = 2.0  # auto-reconnect cadence after an unplug/drop

# Manual gamepad override (ESP32-side; the Mac only observes it via telemetry).
MOTION_MANUAL_IDLE_RETURN_SECS = 4
MOTION_MANUAL_AUTORETURN = False

# GUI joystick (Motivator Control) command ramping. The console slews its drive
# command toward the stick position instead of jumping to it: gentle on the way up,
# faster but never abrupt on the way down — a tall base can topple if it stops dead.
# These are seconds from a standstill to full speed (up) and from full speed back to
# a standstill (down) at full scale; down should be < up but > 0 (never instant). The
# GUI STOP button and closing the console still stop immediately — this only shapes
# the joystick's own ramp.
MOTION_MANUAL_RAMP_UP_SECS = 1.2
MOTION_MANUAL_RAMP_DOWN_SECS = 0.5

# Serial connection retry (mirrors the servo connect pattern).
MOTION_SERIAL_TIMEOUT_SECS = 0.1
MOTION_CONNECT_RETRY_ATTEMPTS = 3
MOTION_CONNECT_RETRY_DELAY_SECS = 1.0
MOTION_ACK_TIMEOUT_SECS = 0.5         # how long send-and-confirm waits for an ack
