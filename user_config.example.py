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

# Tool-router Phase 0 shadow (docs/tool_router_scope.md): log the conversation
# model's tool choice next to every shipped routing decision, for the cutover
# report (tools/tool_router_report.py). Costs one small hosted call per routed
# turn — turn on for a collection week, then back off.
# TOOL_ROUTER_SHADOW_ENABLED = False

# Local low-latency sidecar (intent/classifiers). False = OpenAI-only fallback.
# LOCAL_LLM_ENABLED = True
# OLLAMA_MODEL = "qwen2.5:1.5b"           # any model pulled into your Ollama
# OLLAMA_BASE_URL = "http://localhost:11434"

# Local speech-to-text backend: "qwen3" (Qwen3-ASR — ~2x faster than Whisper at
# equal measured accuracy) or "whisper". Either falls back to the other, then to
# the OpenAI API.
# TRANSCRIPTION_BACKEND = "qwen3"
# QWEN_ASR_MODEL_REPO = "mlx-community/Qwen3-ASR-1.7B-8bit"

# Local Whisper (fallback backend, or primary when TRANSCRIPTION_BACKEND="whisper").
# WHISPER_LOCAL_MODEL = "mlx-community/whisper-large-v3-turbo"

# ElevenLabs voice — the cloned-voice id Rex speaks with (from your ElevenLabs
# account after cloning). TTS_MODEL_ID is the ElevenLabs engine.
# ELEVENLABS_VOICE_ID = "no5jvDWvnx2leN3dFOS7"
# TTS_MODEL_ID = "eleven_multilingual_v2"

# On-device TTS (Qwen3-TTS voice clone). ElevenLabs stays Rex's TRUE voice; the
# local engine runs entirely offline. It powers three things: the `--local-tts`
# runtime flag (no ElevenLabs at all this run), automatic fallback when ElevenLabs
# is unreachable / out of credits, and the impersonation feature. Model weights
# (~2.9 GB) are downloaded by setup_assets.py.
#   Keep Rex talking in his local voice if ElevenLabs fails (master switch):
# LOCAL_TTS_FALLBACK_ENABLED = True
#   Which mlx-community Qwen3-TTS variant to run ("1.7B-Base-8bit" = best speed/
#   quality on Apple Silicon; "0.6B-Base-bf16" = lighter):
# LOCAL_TTS_MODEL_VARIANT = "1.7B-Base-8bit"
#   Rex's local reference clip name (assets/voices/rex/<name>.wav + .txt):
# LOCAL_TTS_VOICE = "RX24-pure"
#   Preload the local model at boot even in ElevenLabs mode, so the first fallback
#   line is instant instead of paying a one-time model load:
# LOCAL_TTS_WARM_ON_BOOT = False
#   Cache local takes so a repeated line replays instantly. OFF by default so
#   --local-tts testing always hears freshly synthesized audio; turn on for a
#   production local-only deployment:
# LOCAL_TTS_CACHE_ENABLED = False

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
# IMPERSONATION_ENABLED = True           # "do an impersonation of me/<person>" (needs local TTS)
# ANIMAL_DETECTION_ENABLED = True        # react to pets / animals
# SOUND_AWARENESS_ENABLED = True         # classify non-speech sounds (barks, doorbells, glass…)
# SOUND_AWARENESS_REACTIONS_ENABLED = True  # …and react out loud to the notable ones
# SOUND_EVENT_REACTION_COOLDOWN_SECS = 90.0 # min gap between spoken sound reactions
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
# SILENCE_TIMEOUT_SECS = 0.65

# Four-phase empty-room flow (env-backed): look/comment, get bored, complain he
# was left activated, then resign and sleep. Phase 3 starts 60% of the way from
# boredom onset to sleep. SLEEP wakes only through the wakeuprex ONNX model.
# EMPTY_ROOM_OBSERVATION_ONSET_SECS = 30.0
# BOREDOM_ONSET_SECS = 150.0
# BOREDOM_LEFT_ON_PHASE_FRACTION = 0.60
# BOREDOM_SLEEP_AFTER_SECS = 900.0

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

# Face detection/recognition backend. "insightface" (default): SCRFD + ArcFace
# 512-dim embeddings — much better at distance, odd angles, and the robot's
# upward camera view. "dlib": the legacy stack (also the automatic fallback if
# InsightFace models fail to load). Face enrollments are backend-specific:
# switching requires re-enrolling faces (voice ID is unaffected) — run
# venv/bin/python tools/test_face_id.py --enroll "Name" --replace
# FACE_BACKEND = "insightface"

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

# ── Impersonation ────────────────────────────────────────────────────────────
# On/off lives in the feature switches above (IMPERSONATION_ENABLED). Requires the
# local TTS model. For "impersonate me", Rex asks you to repeat one of these fixed
# lines so he has a clean voice sample with a known transcript. Keep each ~2 short
# sentences (long enough to clone from). Edit freely.
# IMPERSONATION_CAPTURE_LINES = [
#     "Say this exactly like you mean it: the cantina's open, the music's loud, "
#     "and I fly better than I sing. Strap in.",
#     "Repeat after me: I have a very good feeling about this, which historically "
#     "means it is about to go sideways.",
# ]

# What Rex says (in HIS voice) just before an impression — also covers the one-time
# model-load pause. One picked at random.
# IMPERSONATION_INTRO_LINES = [
#     "Okay, okay — clearing my vocal buffers. Ahem.",
#     "Alright, loading the impression module. This is going to be uncanny.",
# ]

# Optional Rex-voice button spoken right after the impression (a cheap laugh).
# IMPERSONATION_OUTRO_ENABLED = True
# IMPERSONATION_OUTRO_LINES = [
#     "...I do not sound like that.",
#     "Tip your droid.",
# ]

# Drop your own famous-person clips (with a matching transcript) in
# assets/voices/famous/<name>.wav + <name>.txt to enable "impersonate <that name>".

# ── Drive base feel (motion) ─────────────────────────────────────────────────
# Dead-stop breakaway punch for the drive wheels (PWM duty, 0..1023). The
# full-weight droid needs a substantial kick to overcome static friction from a
# stop; this floor applies ONLY while a commanded wheel is still stationary and
# drops away the instant it rolls, so slow driving stays controllable. Raise if
# he hums without moving at low stick; lower if starts feel lurchy.
# MOTION_WHEEL_BREAKAWAY_DUTY = 358

# Running duty floor while a wheel is already rolling (keeps creep alive over
# bumps/carpet seams without overshooting slow commands).
# MOTION_WHEEL_MIN_DUTY = 120

# Velocity feedforward: duty per m/s of commanded speed (droid-measured plant
# gain ~640). Only touch alongside a bench retune (firmware/tools/motion_bench.py).
# MOTION_WHEEL_KFF = 640

# ── Parlor games (Jeopardy) ──────────────────────────────────────────────────

# Speak "Remaining categories: …" even with the dashboard up. On by default:
# players sitting around the ROBOT cannot see the laptop board. The read-out
# follows a fatigue curve either way (four full reads, then every third turn),
# so this is not the "stop repeating it" knob. Set False for a table that is
# actually looking at the screen.
# JEOPARDY_READ_CATEGORIES_WITH_GUI = True

# How many players get a missed clue AFTER the one who picked it. 1 = a single
# second chance around the table. 0 = no rebound at all (real Jeopardy rules).
# 3+ = the old lap-the-table behavior, which reads one clue to four people in a
# row.
# JEOPARDY_MAX_REBOUNDS = 1

# Score nothing for a turn that was not an answer attempt — a bare "What is?",
# a turn far past any real answer's length, someone calling the dog, or a
# complaint about the game. Rex stays quiet and the answer clock keeps running.
# JEOPARDY_IGNORE_NON_ANSWERS = True

# Only charge the current player for a wrong answer when it could plausibly be
# theirs (the speaker is unresolved, or resolves to them). A confidently
# recognized OTHER contestant shouting a guess is the room helping out, and
# their miss costs nobody. Correct answers still score for whoever's turn it is.
# JEOPARDY_ONLY_CHARGE_THE_ANSWERER = True
