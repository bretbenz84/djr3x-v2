"""
user_config.example.py — template for DJ-R3X user-facing overrides.

This committed template is COPIED to user_config.py during setup (setup_macos.sh,
same as apikeys.example.py → apikeys.py and .env.example → .env). user_config.py
is gitignored; edit THAT file, not this one.

Everything here is OPTIONAL. config.py defines every setting with a sensible
default; user_config.py is imported LAST, so any value you uncomment below
overrides the config.py default. If user_config.py is missing or fully commented
out, the robot runs on config.py's defaults exactly as before.

HOW TO USE
  • Each setting is shown commented-out, set to its CURRENT default.
  • Uncomment a line and change the value to override it.
  • Delete / re-comment a line to fall back to the config.py default.
  • config.py remains the source of truth for defaults; values mirrored here as
    of 2026-06-18 (they only take effect once you uncomment them).

NOTE  Settings marked "env:" can also be set via a .env variable. Uncommenting
      them here takes precedence over .env.
NOTE  Per-machine serial ports (MOTION_ESP32_PORT, MAESTRO_PORT, Arduino ports)
      and API keys are NOT here — they live in .env / apikeys.py.
"""

# ═════════════════════════════════════════════════════════════════════════════
# 1. AI MODELS & SERVICES
# ═════════════════════════════════════════════════════════════════════════════

# Main conversational voice — the model that writes what Rex actually says.
# Any OpenAI chat model id. gpt-5.x models route through llm_compat (reasoning
# knobs). This is the one to change to alter Rex's wit/persona quality.
# LLM_CONVERSATION_MODEL = "gpt-5.4-mini"

# Utility model: intent routing, JSON/classifier calls, the action router, vision
# JSON. Cheap + fast matters more than wit here. (The action router follows this.)
# LLM_MODEL = "gpt-4o-mini"

# Image / scene analysis model (room scans, captions).
# VISION_MODEL = "gpt-4o-mini"

# Model for the rolling conversation-arc summary (off the speech path).
# CONVERSATION_ARC_OPENAI_MODEL = "gpt-4o-mini"

# Model used at session end to consolidate memories.
# MEMORY_CONSOLIDATION_MODEL = "gpt-4o-mini"

# Local low-latency sidecar (intent/classifiers). False = OpenAI-only fallback.
# LOCAL_LLM_ENABLED = True
# OLLAMA_MODEL = "qwen2.5:1.5b"           # any model pulled into your Ollama
# OLLAMA_BASE_URL = "http://localhost:11434"

# Local speech-to-text (MLX Whisper) model id. Larger = more accurate, slower.
# WHISPER_LOCAL_MODEL = "mlx-community/whisper-large-v3-turbo"

# ElevenLabs voice — the cloned-voice id Rex speaks with (from your ElevenLabs
# account after cloning). TTS_MODEL_ID is the ElevenLabs engine.
ELEVENLABS_VOICE_ID = "no5jvDWvnx2leN3dFOS7"
# TTS_MODEL_ID = "eleven_multilingual_v2"

# ═════════════════════════════════════════════════════════════════════════════
# 2. PERSONALITY & VOICE CHARACTER
# ═════════════════════════════════════════════════════════════════════════════

# Personality dials, 0–100. Higher humor/sarcasm/roast = more bite (only when the
# moment invites it — they don't force a roast every turn). Low agreeability =
# pushes back instead of complying. Low sentimentality = less mushy. These set the
# first-run baseline; tune up if Rex goes too soft, down if he gets mean.
# PERSONALITY_DEFAULTS = {
#     "humor":           75,
#     "sarcasm":         60,
#     "roast_intensity": 55,
#     "honesty":         90,
#     "talkativeness":   65,
#     "darkness":        40,
#     "sentimentality":  50,
#     "agreeability":    35,
# }

# Hard cap on roast_intensity whenever a child is present (0–100). Lower = safer.
# CHILD_SAFE_ROAST_MAX = 40

# ElevenLabs voice expressiveness applied to every line. stability 0–1 (lower =
# more emotional/variable, higher = monotone); style 0–1 (theatricality);
# similarity_boost 0–1 (adherence to the clone, keep mid-high); use_speaker_boost.
# TTS_VOICE_SETTINGS_BASELINE = {
#     "stability": 0.40,
#     "similarity_boost": 0.80,
#     "style": 0.55,
#     "use_speaker_boost": True,
# }

# Master switch for Rex's persistent "preoccupation" (an opinion he volunteers).
# REX_POV_ENABLED = True

# Base character prompt — Rex's core persona. It drives BOTH voices: the lean brain
# uses it directly as its system persona (via LEAN_BRAIN_PERSONA's fallback), and the
# classic assembled prompt (reply fallback / web-search base) uses it as section 1.
# config.py is actively iterated on this text — overriding it here FREEZES your copy
# and silently opts out of future improvements. To customize anyway, copy the CURRENT
# text from config.py (search REX_CORE_PROMPT) into:
# REX_CORE_PROMPT = """..."""

# Startup "still getting ready" boot lines — the Star Tours-style filler Rex cycles
# through (no repeats between launches) while the heavy models preload at boot. Keep them
# as fixed strings so each caches in the ElevenLabs TTS cache after its first play and
# stays free thereafter. Uncomment + edit the whole list to change what he says while
# loading (the value below mirrors the current config.py default).
# STARTUP_BOOT_TTS_LINES = [
#     "Welcome aboard! This is Captain Rex from the cockpit. Still warming up the old circuits, hang on folks — I know this is probably your first flight, and it's… mine, too!",
#     "Still booting up, folks — I'm not ready yet. Hang tight while my circuits finish waking up.",
#     "Hold please, I'm still loading. The droid you're waiting for is not in the cockpit yet.",
#     "Not ready yet, everybody — running my pre-flight checklist. Thrusters, navi-computer, personality core… still ticking the boxes.",
#     "Hang on, I'm still warming up. Don't talk to me yet — I won't hear a word until I'm loaded.",
#     "Powering up, please wait. My systems are still coming online, and so is my patience.",
#     "Still loading, folks. They told me this droid boots instantly. They lied. Give it a moment.",
#     "One moment — not online yet, still calibrating my photoreceptors. First time flying this thing… and honestly, it's my first time booting it up, too.",
#     "Almost there, but not yet — memory banks still loading. Save your questions for when I'm actually awake.",
#     "Standby, everybody. I'm booting up, not ignoring you. There IS a difference.",
#     "Still spinning up, please hold. They've got me piloting on my first flight — and it's my first boot-up, too. We'll figure it out together.",
#     "Not ready to chat yet, folks — syncing my audio receptors. The second I'm online, you'll know it.",
#     "Loading, loading… still loading. I'd tell you a joke, but I'm not even fully on yet.",
#     "Hang tight, I'm still booting — you can call me Captain Rex. Well, you can once I finish loading. First flight for me too, folks.",
#     "Give the old motivator a second — I'm not ready to talk yet. Showmanship, however, never powers down.",
#     "Please wait while I finish loading. Spend forty years in storage, you forget where everything is.",
#     "Not awake yet, folks — still warming up the circuits. First flight? Same here. They handed me the cockpit and the boot sequence on the same day.",
# ]

# ═════════════════════════════════════════════════════════════════════════════
# 3. LOCATION & WORLD
# ═════════════════════════════════════════════════════════════════════════════

# City used for weather lookups (any "City, Region" string the weather API knows).
# WEATHER_LOCATION = "Sacramento, California"

# Venue name Rex may reference (backstory flavor; he is usually NOT in a cantina).
# VENUE_NAME = "Oga's Cantina"

# ISO 3166-1 country code for the holiday calendar (e.g. "US", "GB", "DE").
# HOLIDAY_COUNTRY_CODE = "US"

# Ambient weather reactions on/off.
# WEATHER_PROACTIVE_REACTIONS_ENABLED = True

# Calendar dates Rex reacts to — keys are (month, day) tuples.
# NOTABLE_DATES = {
#     (5,  4):  "Star Wars Day",
#     (10, 31): "Halloween",
#     (12, 25): "Christmas",
#     (1,  1):  "New Year's Day",
# }

# ═════════════════════════════════════════════════════════════════════════════
# 4. FEATURE SWITCHES  (True / False)
# ═════════════════════════════════════════════════════════════════════════════

# DEBUG_MODE = True                      # timestamped per-run log vs shared file
# LOG_SYSTEM_PROMPT = True               # log the full assembled prompt each turn
# GUI_ENABLED = False                    # optional macOS dashboard
# EMPATHY_ENABLED = True                 # affect classification / emotional intel
# EPISODIC_MEMORY_ENABLED = True         # Rex's own episodic memory capture
# IDLE_BANTER_ENABLED = True             # re-engage when a present person goes quiet
# ONBOARDING_ENABLED = True              # first-meeting question burst for strangers
# WEB_SEARCH_ENABLED = True              # answer current-info questions via web search
# ANIMAL_DETECTION_ENABLED = True        # react to pets / animals
# VISUAL_CURIOSITY_ENABLED = True        # camera-grounded riffs / questions
# SPEAKER_GAZE_ENABLED = True            # env: turn head toward whoever is speaking
# BOREDOM_ENABLED = True                 # env: bored-in-empty-room → doze to sleep
# PLAY_STARTUP_AUDIO = True              # startup sound burst
# PLAY_STARTUP_SPEECH_CLIP = False       # env: spoken startup intro clip
# PLAY_SHUTDOWN_AUDIO = True             # shutdown clip

# ═════════════════════════════════════════════════════════════════════════════
# 5. TIMEOUTS & SENSITIVITY
# ═════════════════════════════════════════════════════════════════════════════

# Seconds of silence before an active conversation lapses back to IDLE.
# CONVERSATION_IDLE_TIMEOUT_SECS = 45.0

# Seconds of silence that close a single spoken turn before it is processed.
# SILENCE_TIMEOUT_SECS = 0.6

# Boredom flow (env-backed): seconds alone before Rex starts bored remarks, and
# total bored seconds before he dozes off to SLEEP.
# BOREDOM_ONSET_SECS = 150.0
# BOREDOM_SLEEP_AFTER_SECS = 600.0

# Wake-word sensitivity 0–1. Raise to cut false triggers, lower for sensitivity.
# Per-model values override the global default.
# WAKE_WORD_THRESHOLD = 0.5
# WAKE_WORD_THRESHOLDS = {
#     "Dee-Jay_Rex": 0.5,
#     "Hey_DJ_Rex":  0.5,
#     "Hey_rex":     0.5,
#     "Yo_robot":    0.5,
#     "wakeuprex":   0.5,
#     "shut_down":   0.6,   # do NOT raise to 0.8 — would reject real shutdowns
# }

# How often (seconds) the GPT vision environment scan runs.
# ENVIRONMENT_SCAN_INTERVAL_SECS = 180

# Days before a stored fact is treated as stale / needs reconfirming.
# STALE_FACT_THRESHOLD_DAYS = 365

# Days away before Rex remarks on a long absence when you return.
# LONG_ABSENCE_THRESHOLD_DAYS = 60

# ═════════════════════════════════════════════════════════════════════════════
# 6. WEB SEARCH  (current-info lookups)
# ═════════════════════════════════════════════════════════════════════════════
# Rex can look things up on the web when a question needs CURRENT info — either
# because you asked out loud ("look that up") or because he decides on his own it
# needs live data. He says a short stall line, runs the search via OpenAI's hosted
# web_search tool, then answers in character. Uses your existing OpenAI key.
# On/off lives in the feature switches above (WEB_SEARCH_ENABLED).

# Model that runs the search + voices the answer. Leave as None to follow your
# conversation model. If your conversation model can't use the web_search tool,
# set a model that can (e.g. "gpt-4o-mini").
# WEB_SEARCH_MODEL = None

# Let Rex trigger a search on his own (vs only when you explicitly ask). The gate
# adds a tiny gpt-4o-mini check that confirms a question really needs live data.
# WEB_SEARCH_AUTONOMOUS_ENABLED = True
# WEB_SEARCH_AUTONOMOUS_GATE_ENABLED = True

# After a search, if you go quiet, have Rex's idle chatter ASK about the topic
# ("what got you asking about that?") instead of piling on more facts/opinions.
# WEB_SEARCH_FOLLOWUP_INQUISITIVE_ENABLED = True

# Spoken phrases that ALWAYS force a search (substring, case-insensitive). Edit freely.
# WEB_SEARCH_TRIGGER_PHRASES = [
#     "look that up", "look it up", "look up", "search the web",
#     "search the internet", "search for", "search online", "google that",
#     "google it", "what's the latest on", "can you look up", "find out for me",
# ]

# What Rex says the instant a search starts (one picked at random). Keep them short
# and in his voice. Edit freely.
# WEB_SEARCH_STALL_LINES = [
#     "Let me check the archives.",
#     "Hold on, pinging the holonet.",
#     "One sec, looking that up.",
#     "Give me a tick, scanning the feeds.",
#     "Patience — consulting the galaxy's databanks.",
# ]
