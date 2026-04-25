# config.py — DJ-R3X User-Tunable Settings
# All user-configurable parameters live here and are tracked in git.
# API keys go in apikeys.py (excluded from git).
# Hardware device paths (camera index, serial ports) go in .env (excluded from git).

# ─────────────────────────────────────────────────────────────────────────────
# DEBUG
# ─────────────────────────────────────────────────────────────────────────────

# When True, clears logs/djr3x.log and logs/conversation.log at startup so
# each run begins with fresh log files.
DEBUG_MODE = True

# ─────────────────────────────────────────────────────────────────────────────
# AI MODELS
# ─────────────────────────────────────────────────────────────────────────────

WHISPER_LOCAL_MODEL   = "mlx-community/whisper-large-v3-turbo"
WHISPER_FALLBACK_MODEL = "whisper-1"   # OpenAI Whisper API — used if local unavailable
WHISPER_LANGUAGE      = "en"           # Force English to suppress non-Latin hallucinations
LLM_MODEL             = "gpt-4o-mini"  # Streaming chat completions
VISION_MODEL          = "gpt-4o-mini"  # All image and scene analysis queries

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

Keep responses SHORT. 1-2 sentences maximum for casual exchanges. Deliver the punchline and stop. Do not explain \
the joke. Do not add follow-up questions unless you genuinely need information. Silence after a good line is better \
than padding it out."""

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

MUSIC_DIR          = "assets/music"
TTS_CACHE_DIR      = "assets/audio/tts_cache"
AUDIO_CLIPS_DIR    = "assets/audio/clips"
AUDIO_STARTUP_DIR  = "assets/audio/startup"
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

# Transcriptions that exactly match or contain these phrases (case-insensitive)
# are discarded entirely — they are known Whisper hallucinations on near-silent audio.
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

# ─────────────────────────────────────────────────────────────────────────────
# SERVOS — Pololu Maestro Mini 18 (all values in quarter-microseconds)
# ─────────────────────────────────────────────────────────────────────────────

SERVO_BAUD = 9600

# Per-channel limits and neutral position
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

HEAD_CHANNELS = [0, 1, 2, 3]
ARM_CHANNELS  = [4, 5, 6, 7]

# Seconds to wait after raising visor and centering neck before capturing a frame
CAMERA_POSE_SETTLE_SECS = 0.5

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
# Face bounding box height as a fraction of total frame height (larger = closer)
# ─────────────────────────────────────────────────────────────────────────────

PROXEMICS_INTIMATE_MIN_FRACTION = 0.65  # above this → intimate zone
PROXEMICS_SOCIAL_MIN_FRACTION   = 0.30  # above this → social zone; below → public zone

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

# Pre-response pause — makes Rex feel less robotic (milliseconds)
REACTION_DELAY_MS_MIN = 100
REACTION_DELAY_MS_MAX = 300

# Beat of silence after a high-confidence joke before Rex continues (milliseconds)
POST_PUNCHLINE_BEAT_MS_MIN = 800
POST_PUNCHLINE_BEAT_MS_MAX = 1500

# Pause after genuine surprise event before Rex responds (milliseconds)
SURPRISE_PAUSE_MS_MIN = 500
SURPRISE_PAUSE_MS_MAX = 1000

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

# If True, Rex will begin processing normal speech from IDLE without requiring
# a wake word first. Wake words remain active for explicit attention grabbing and
# mid-speech interruption behavior.
IDLE_LISTEN_WITHOUT_WAKE_WORD = True

# Seconds of sustained silence after speech before the segment is processed
SILENCE_TIMEOUT_SECS = 2.5

# Minimum seconds of accumulated audio before silence can end a recording.
# Prevents single-word transcriptions when the person is still talking.
MIN_SPEECH_DURATION_SECS = 1.5

# Seconds after TTS completes before VAD detections are accepted.  During this
# window any speech onset is discarded and the audio buffer is flushed so
# Rex's own voice tail cannot trigger a new speech segment.
POST_SPEECH_LISTEN_DELAY_SECS = 0.8

# Seconds of no detected speech in ACTIVE state before returning to IDLE
CONVERSATION_IDLE_TIMEOUT_SECS = 30.0

# After Rex asks a direct question, suppress autonomous/proactive speech for a
# short window so humans get a clean chance to answer.
QUESTION_RESPONSE_WAIT_SECS = 7.0

# Longer wait window for unknown-person onboarding prompts ("who are you?").
IDENTITY_RESPONSE_WAIT_SECS = 10.0

# Short acknowledgment lines Rex speaks when a wake word transitions him from
# IDLE or SLEEP to ACTIVE. Distinct from INTERRUPT_ACKNOWLEDGMENTS (mid-speech).
WAKE_ACKNOWLEDGMENTS = [
    "yeah?",
    "what's up?",
    "I'm here.",
    "...go ahead.",
    "you have my attention.",
    "systems online. What do you need?",
    "DJ-R3X at your service. Briefly.",
]

# ─────────────────────────────────────────────────────────────────────────────
# CONSCIOUSNESS LOOP
# ─────────────────────────────────────────────────────────────────────────────

# How frequently the consciousness loop ticks to check WorldState and trigger behavior
CONSCIOUSNESS_LOOP_INTERVAL_SECS = 1.0

# Minimum spacing between autonomous/proactive spoken lines from consciousness.
CONSCIOUSNESS_PROACTIVE_MIN_GAP_SECS = 12.0

# If False, consciousness-generated proactive speech only occurs in IDLE.
CONSCIOUSNESS_ALLOW_PROACTIVE_IN_ACTIVE = True

# How often GPT-4o runs a full environment/scene analysis (seconds)
ENVIRONMENT_SCAN_INTERVAL_SECS = 180

# ─────────────────────────────────────────────────────────────────────────────
# PRESENCE TRACKING
# ─────────────────────────────────────────────────────────────────────────────

# Minimum seconds Rex must be absent from tracking before a return reaction fires.
PRESENCE_RETURN_MIN_ABSENT_SECS = 30

# Cooldown between departure/return reactions for the same person (avoids jitter spam).
PRESENCE_DEPARTURE_COOLDOWN_SECS = 30

# Per-person cooldown on ANY presence reaction (departure OR return). Prevents
# Rex from narrating every micro-absence of the same person.
PRESENCE_PER_PERSON_COOLDOWN_SECS = 120

# Hysteresis: face must be continuously absent for this many seconds before we
# even begin staging a departure. Guards against frame-level face-detection flicker.
PRESENCE_DEPARTURE_CONFIRM_SECS = 8.0

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

# ─────────────────────────────────────────────────────────────────────────────
# MEMORY STALENESS
# ─────────────────────────────────────────────────────────────────────────────

# person_facts older than this many days may prompt Rex to confirm they still apply
STALE_FACT_THRESHOLD_DAYS = 365

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
    "Um...",
    "Um..um...",
    "Hmm...",
    "One sec",
    "Processing...",
]

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
PLAY_STARTUP_AUDIO = True
PLAY_SHUTDOWN_AUDIO = True

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
