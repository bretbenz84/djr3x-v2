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
# Draw the detected body pose as a live wireframe skeleton over the camera preview
# (in addition to the green face boxes). Reads world_state.people[*].pose_keypoints,
# published by vision/pose.py. Off → boxes only.
GUI_POSE_WIREFRAME_ENABLED = True
# Draw a bounding box + label for each detected room OBJECT (the local COCO stream,
# world_state.objects) over the camera preview, like the face boxes and pose
# wireframe. Off → no object boxes (the detection stream still runs for behaviors).
GUI_OBJECT_BOXES_ENABLED = True
# Shade the CAMERA_SELF_OCCLUSION_ZONES over the preview so it is obvious where object
# detection is switched off, and whether those rectangles still line up with Rex's eye
# stalks after a camera/hardware move. The original 1-px dash at 27% alpha measured
# ~11% contrast against a live feed and read as "the block is gone" (owner 2026-07-24)
# even though the mask was working. Set GUI_OCCLUSION_ZONES_VISIBLE=False to hide them
# again; the masking itself is unaffected either way (vision/animal_detector).
GUI_OCCLUSION_ZONES_VISIBLE = True
GUI_OCCLUSION_ZONE_FILL_ALPHA = 48    # translucent violet wash inside the zone
GUI_OCCLUSION_ZONE_EDGE_ALPHA = 190   # dashed border
# Only draw a pose wireframe for a slot with a VISIBLE face whose centre is within
# GUI_POSE_FACE_COHERENCE_DIST (normalized) of the pose head. Kills phantom wireframes
# (no face there) and mis-bound wireframes (drawn over the wrong person). Set False to
# draw every detected pose regardless of face (old behavior).
GUI_POSE_REQUIRE_FACE = True
GUI_POSE_FACE_COHERENCE_DIST = 0.20
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

# Primary local ASR backend: "qwen3" (Qwen3-ASR via mlx_audio) or "whisper".
# Switched to qwen3 2026-07-31: identical word accuracy on this room's recorded
# takes (tools/asr_bench.py) at ~2x the speed (0.57s vs 1.02s median). The
# fallback chain is unchanged in spirit: qwen3 -> local whisper -> OpenAI API.
TRANSCRIPTION_BACKEND = "qwen3"
QWEN_ASR_MODEL_REPO = "mlx-community/Qwen3-ASR-1.7B-8bit"
QWEN_ASR_MODEL_DIR  = "assets/models/qwen_asr/Qwen3-ASR-1.7B-8bit"
# Trust floor for the mean per-token logprob of a Qwen3 decode (the .confident
# gate that keeps low-quality turns out of durable memory). Calibrated on the
# 2026-07-31 mic_check takes: every clean decode scored 0.0 to -0.03 while the
# two truncated/garbage captures scored -0.75 and -1.25 — a far cleaner
# separation than Whisper's avg_logprob gives.
QWEN_ASR_TRUST_MIN_AVG_LOGPROB = -0.35
QWEN_ASR_MAX_TOKENS = 256   # segments are <=30s of speech; don't let a loop run long
WHISPER_LANGUAGE      = "en"           # Force English to suppress non-Latin hallucinations
WHISPER_PRELOAD_ON_STARTUP = True      # Warm MLX Whisper before the first live utterance
WHISPER_TEMPERATURE = 0.0              # Deterministic decode avoids slow retry ladders

# ---- Trust threshold: what Rex is willing to LEARN from ---------------------
# Whisper never fails loudly. Handed one quiet word or a half-captured phrase it
# returns a fluent, confident sentence with no outward sign anything is wrong.
# Field 2026-07-25: "wine" came back "I'm going to split it."; "This is the
# workshop room" came back "Shop room."; an utterance decoded as "Spice it."
# ENROLLED A PERSON NAMED SPICE. All three were written to durable memory and
# mined for proactive questions days later. avg_logprob / no_speech_prob are the
# only signal that separates those from real speech, and they were unused.
#
# These are LEARNING gates, not hearing gates. A turn below them is still heard,
# answered and acted on — it just cannot become a stored fact, a person's name,
# or a room. That asymmetry is deliberate: far-field SNR here is 13-15 dB, so
# genuine speech scores badly often enough that a hearing gate would make Rex
# deaf (see the far-field measurements from 2026-07-24). Being occasionally
# forgetful is recoverable; confidently remembering things you never said is not.
WHISPER_TRUST_MIN_AVG_LOGPROB = -0.85   # mean per-token logprob; lower = guessing
WHISPER_TRUST_MAX_NO_SPEECH_PROB = 0.5  # higher = Whisper thinks it was silence
WHISPER_CONDITION_ON_PREVIOUS_TEXT = False
LLM_MODEL             = "gpt-4o-mini"  # Streaming chat completions
VISION_MODEL          = "gpt-4o-mini"  # All image and scene analysis queries

# ── GPT-5-class conversation model (LIVE — see docs/gpt-5_4_mini.md) ──
# The model for Rex's USER-FACING in-character generation (the streaming reply + the
# short curiosity/onboarding/expression/scenery generators). The classifiers/routers/
# JSON/vision calls keep using LLM_MODEL (gpt-4o-mini) — hybrid rollout. Routed through
# intelligence/llm_compat, which translates the GPT-5 param differences in ONE place.
# Flipped to gpt-5.4-mini 2026-06-17 after the smoke test + A/B (clear win on wit/persona).
# ROLLBACK = set this back to LLM_MODEL.
LLM_CONVERSATION_MODEL = "gpt-5.4-mini"
# GPT-5-only knobs, applied by llm_compat ONLY when the model is a GPT-5/o-series
# reasoning model (ignored for gpt-4o-mini). None = don't send the param.
#   reasoning_effort: none|low|medium|high|xhigh — "none" keeps time-to-first-token
#   low (critical for the real-time voice loop) AND is the only level where temperature
#   is accepted; raise for depth (but then drop pass-temp). (NB: "minimal", a
#   GPT-5/5.1-era value, appears DROPPED for 5.4 — every current source omits it.)
#   verbosity: low|medium|high — "low" for terser replies (paired with the prompt
#   brevity rule; the A/B showed it helps only marginally on its own).
LLM_REASONING_EFFORT  = "none"
LLM_VERBOSITY         = "low"
# GPT-5 reasoning models REJECT a non-default temperature (400) WHEN reasoning is engaged.
# Smoke test (2026-06-17, tools/gpt5_smoke_test.py) CONFIRMED gpt-5.4-mini ACCEPTS
# temperature at reasoning_effort="none" and 400s at "medium". Safe here because the
# conversation path runs at effort="none". (If you ever raise effort above "none", set
# this back to False — and raise the reply token budget; reasoning eats the output.)
LLM_GPT5_PASS_TEMPERATURE = True

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

# ── Lean brain (rebuild, Phase 0) ──────────────────────────────────────────────
# One streaming model call — the coherent Rex persona (REX_CORE_PROMPT) + a small live
# context + the recent turns as real chat messages — replacing the router→agenda→social_frame
# →4,400-word-prompt pipeline. OFF until proven via the offline replay harness (tools/
# lean_replay.py); wiring it into the live turn path is a later step. Latency-first: small,
# consistent prompt for fast time-to-first-token; the live path streams sentence-by-sentence.
LEAN_BRAIN_ENABLED          = True    # ON for live GUI testing — set False to revert to the classic brain instantly
LEAN_BRAIN_MODEL            = ""      # "" → the standard conversation model (gpt-5.4-mini, reasoning off)
LEAN_BRAIN_MAX_TOKENS       = 120     # keep replies short + first audio fast
LEAN_BRAIN_TRANSCRIPT_TURNS = 8       # recent turns passed as real user/assistant messages
# Multi-party awareness: when 2+ distinct humans appear in the recent window, history
# turns carry speaker labels ("JT: ..."), the current turn names its speaker, and the
# system context gains a room block + other-participant lines — so Rex answers the
# person who actually spoke instead of attributing everything to the primary person.
# 1-on-1 sessions carry none of this prompt weight.
LEAN_MULTI_PARTY_ENABLED = _env_bool("LEAN_MULTI_PARTY_ENABLED", True)
LEAN_BRAIN_PERSONA          = ""      # "" → REX_CORE_PROMPT verbatim (the voice, minus the scaffolding)
# Phase 4 — ONE VOICE. Route the OTHER spoken-line generators (greetings, world/presence reactions,
# onboarding reactions/questions, directed-look/wave/etc.) through the SAME lean persona as replies,
# instead of the classic 4,400-word assembled prompt / onboarding's thin inline persona. So Rex
# sounds like one character everywhere, not just in replies. Only active alongside LEAN_BRAIN_ENABLED;
# the reply-path CLASSIC fallbacks stay classic for resilience. Kill switch → instant revert to the
# per-path voices.
LEAN_ONE_VOICE_ENABLED      = True

# Lean AGENCY (Phase 1): when a known person is PRESENT but quiet, Rex DECIDES (in character,
# grounded in perception + memory + mood) to say ONE thing or just watch — the strong default is
# watch. This replaces the old silence-fill taxonomy (idle_banter / lull re-engagement) with a
# single motivated impulse. Restraint is set by the quiet threshold + cooldown, and the model's
# heavy bias to PASS. Only active alongside LEAN_BRAIN_ENABLED.
LEAN_IMPULSE_ENABLED        = True
LEAN_IMPULSE_QUIET_SECS     = 4.0     # seconds after REX FINISHES talking (not since you spoke) before he may break the silence
LEAN_IMPULSE_COOLDOWN_SECS  = 12.0    # min gap between his self-initiated lines during a sustained lull
# Mid-conversation restraint: while the user has spoken within FLOW_WINDOW, the reply
# thread owns the floor — an impulse additionally needs FLOW_QUIET secs of mutual
# silence (vs the 4s true-lull trigger). Stops the question-machine failure without
# dulling his presence when the room has actually gone quiet.
LEAN_IMPULSE_FLOW_WINDOW_SECS = _env_float("LEAN_IMPULSE_FLOW_WINDOW_SECS", 120.0, min_value=0.0, max_value=900.0)
# 30s read as Rex going dead after every exchange (owner 2026-07-06: replied to
# "It's okay" with a quip, then 42 SECONDS of silence — "too long"). 14s is a real
# human beat: long enough that the person clearly isn't answering, short enough
# that Rex still feels present. The question-machine failure this guard exists for
# (impulses stacking during ACTIVE back-and-forth) had sub-10s gaps.
LEAN_IMPULSE_FLOW_QUIET_SECS  = _env_float("LEAN_IMPULSE_FLOW_QUIET_SECS", 14.0, min_value=0.0, max_value=300.0)
# ADAPTIVE re-engage: the 14s above assumes the user still owes a reply. But when
# Rex's OWN last line was a closed statement (a quip with nothing to answer — "good"
# → "Try not to make it a personality."), the exchange stalled on HIM, and 14s of
# dead air feels like he checked out (owner 2026-07-08). Bridge that case sooner.
# When Rex asked a QUESTION instead, the floor-hold (POST_REPLY_QUESTION_WAIT_SECS)
# already governs the wait, so this shorter value only ever applies after a
# dead-end statement — exactly the awkward-silence case visual curiosity should fill.
LEAN_IMPULSE_FLOW_QUIET_AFTER_STATEMENT_SECS = _env_float("LEAN_IMPULSE_FLOW_QUIET_AFTER_STATEMENT_SECS", 7.0, min_value=0.0, max_value=300.0)
# Presence backstop for the lull impulse: only address a lull line at the session
# person while they're plausibly HERE — visible on camera now, or heard within this
# many seconds (keeps voice-led off-camera conversation legitimate). Field bug
# 2026-07-11: after "I'm gonna leave the room now" the impulse kept asking the empty
# room questions — session continuity kept returning the departed person as target.
LEAN_IMPULSE_PRESENCE_HEARD_SECS = 120.0
# Share of lull impulses that open with an OPEN PERSONAL question ("so, got any plans
# for the weekend?") instead of scene-anchored curiosity about a visible object. The
# held-object/scenery emphasis made every lull line about the cup or the chair (owner
# 2026-07-08: "we're missing proactive sentences like 'got any plans for the weekend?'").
# The two registers ALTERNATE (never personal twice running); this is the odds the
# non-forced turn goes personal. 0 = always scene-anchored (old behavior); 1 = maximally
# chatty about their life. The visible object still gets asked about the OTHER turns, and
# the dedicated held-object reactor is unaffected.
LEAN_IMPULSE_PERSONAL_PROB = _env_float("LEAN_IMPULSE_PERSONAL_PROB", 0.4, min_value=0.0, max_value=1.0)
# A visual riff is an occasional *Lean-owned* option during a normal lull, never a
# separate timer-driven speaker.  It is limited to a known adult with no relevant
# conversation boundaries, and only receives safe appearance/accessory or posture cues.
LEAN_VISUAL_RIFF_ENABLED = True
LEAN_VISUAL_RIFF_PROBABILITY = _env_float("LEAN_VISUAL_RIFF_PROBABILITY", 0.25, min_value=0.0, max_value=1.0)
# Remembered event/plan follow-ups as a *Lean-owned* lull cue: when the conversation
# lulls, Rex can raise something the person told him about earlier that has now come
# due ("how did the interview go?"). Data-driven from memory/events.get_pending_followups
# (dated plans whose date has passed, or undated ones older than FOLLOWUP_UNDATED_DAYS).
# The old silence-fill version rode the suppressed `memory_followup`/`small_talk` purposes
# and went dark under the lean brain. This feeds the SAME single lull speaker as the
# holiday/callback/visual-riff cues (one voice), reuses the existing FOLLOWUP_* cadence
# clamp + `_fired_followup_event_ids` de-dup so it never double-asks with the reactive
# _post_response follow-up or the startup greeting follow-up, and arms the normal
# awaiting-resolution loop so the person's next reply closes the event in memory.
# UPCOMING events are deliberately NOT handled here — anticipation ("big day tomorrow —
# ready?") already has its own surviving greeting-time path (_pick_anticipated_event).
LEAN_EVENT_FOLLOWUP_ENABLED = True
# Remembered good-news / celebration check-ins as a *Lean-owned* lull cue: when someone
# shared a positive milestone in a prior session ("I got the job", "we're expecting"),
# Rex opens a lull by celebrating it WITH them. Data-driven from
# memory/emotional_events.get_due_celebrations (valence>0, not yet acknowledged, not
# decayed/muted). The old proactive version rode the suppressed `celebration_checkin`
# purpose and went dark under the lean brain — while the HARD-event / negative-affect
# check-ins (purpose `emotional_checkin`) stayed alive, so Rex would console bad news but
# silently drop good news. This restores the symmetry through the SAME single lull speaker
# (ranked ABOVE holiday/event/callback/visual — good news is the most meaningful open).
# It shares the per-session `_emotional_checkin_fired` gate DIRECTIONALLY — a console
# that already fired blocks a later celebration (don't pile good news on someone you just
# consoled), but a celebration does NOT block a later console about a DIFFERENT event
# (matching the legacy path; consoling after celebrating distinct news is fine). Good news
# can be private, so it honors the SAME crowd discretion the bad-news path uses
# (EMPATHY_DISCRETION_IN_CROWD): Rex won't announce a pregnancy/engagement in a group. It
# marks the event acknowledged on speak (per-event 7-day dedup) and logs the same rex.db
# "I celebrated their good news" episode as the legacy path.
LEAN_CELEBRATION_CHECKIN_ENABLED = True
# Backstop so a due celebration the model keeps declining to voice (returns PASS despite the
# "do not reply PASS" instruction) can't sit at top priority and starve the lower lull cues:
# after this many un-voiced offers within one silent stretch, the cue steps aside so
# holiday/event/callback/visual can run. Resets on the next user turn.
LEAN_CELEBRATION_MAX_UNVOICED_ATTEMPTS = 2
# Episodic "memory musing" as a *Lean-owned* lull cue: in a quiet moment Rex occasionally
# reminisces aloud about something from his rex.db diary ("since I was last on" continuity —
# a scene vibe + a couple of experiential highlights from prior sessions). Data-driven from
# memory/episodic_recall.session_recap (the model can't invent a memory it wasn't given), so
# the old idle-behavior version (purpose `memory_musing`) went dark under the lean brain — its
# governor candidate is suppressed and the lean impulse never consulted the diary. This feeds
# the SAME single lull speaker as the other cues, at the LOWEST priority (only when no
# celebration/holiday/event/callback/visual-riff fires), gated by its own probability and
# capped at ONE musing per session (it's a once-per-visit "since last time" beat; the in-reply
# shared-memory callback `llm._pick_episodic_callback` is a separate surface). Rides the shared
# EPISODIC_RECALL_ENABLED switch + EPISODIC_RECALL_SESSION_RECAP_PROBABILITY.
LEAN_MEMORY_MUSING_ENABLED = True
LEAN_IMPULSE_MAX_TOKENS     = 60      # a self-initiated line is short
# Flat-answer follow-up (reply-side, owner spec 2026-07-06): when a flat
# half-answer ("it's okay", "not much", "meh") ANSWERS a question Rex asked, the
# reply itself carries ONE gentle probe at what's underneath — quip plus "what's
# the missing 30%?" in one breath — instead of waiting for the lull impulse.
# Anti-interview guards: only fires on answers to Rex's own questions (an "okay"
# acknowledging a statement is agreement), at most once per cooldown, never in a
# heavy/give-space window, and the prompt says to let it go if they stay flat.
FLAT_ANSWER_PROBE_ENABLED = _env_bool("FLAT_ANSWER_PROBE_ENABLED", True)
FLAT_ANSWER_PROBE_COOLDOWN_SECS = 180.0
# Talking into the void: after Rex breaks a lull and gets NO reply, he must NOT keep quipping every
# cooldown-tick (the "piled 4 lines about your dinner into silence" failure). Each unanswered
# self-initiated line widens the next required gap, and after MAX_UNANSWERED of them he goes quiet
# until the person actually says something. The counter resets the moment the user speaks, so a
# fresh silence gets its full allowance. Break the silence once or twice — then let it be.
LEAN_IMPULSE_MAX_UNANSWERED = 2       # consecutive self-initiated lines w/ no user reply before he goes quiet
LEAN_IMPULSE_ESCALATION     = 1.0     # gap after n unanswered lines = COOLDOWN * (1 + ESCALATION * n)
# SLOW re-engagement — after the fast lull-break yields the floor, don't let the conversation just
# die: if the person is still HERE but it's gone truly quiet for this long (since Rex last spoke),
# take one PATIENT swing on a genuinely NEW topic/question (bypasses the fast unanswered cap). Spaced
# by the same interval so it can't hammer, presence-gated, and ultimately bounded by the give-up
# outro (PRESENT_REENGAGE_IDLE_TIMEOUT_SECS). 0 disables. Owner ask: "after 40s of silence he should
# try to bring up a new topic/question."
LEAN_IMPULSE_REENGAGE_SECS  = 40.0
# Presence-driven engagement from IDLE (owner 2026-07-18: "R3X knows I'm in front of him —
# if I don't respond to what he says ... it would be good if he still tried to engage").
# Runs the same lean impulse (same discipline: quiet threshold, cooldowns, rolling rate cap,
# unanswered cap, low-energy read) when a known person is visible but has never spoken, or
# after the ACTIVE conversation timed out back to IDLE with them still on camera.
IDLE_PRESENCE_IMPULSE_ENABLED = True
# Disengagement probe (owner 2026-07-18: "treat a lack of response as a gauge of possible lack
# of interest"). After LEAN_IMPULSE_MAX_UNANSWERED topic lines go unanswered with the person
# still on camera, the next re-engage swing becomes a DIRECT check-in ("Am I bothering you?",
# "You busy?" — shy-goad "I don't bite, {name}" for someone Rex barely knows). Then:
#   reply           → normal conversation resumes (probe + snooze cleared on any real speech)
#   "give me a few minutes" → impulses snooze DEFER_SNOOZE_SECS, then Rex checks back
#   silence through the answer window → assume they don't want to talk; snooze NO_ANSWER secs
#   person leaves the frame → all state cleared; normal idle/boredom/sleep path owns the room
ENGAGEMENT_PROBE_ENABLED = True
ENGAGEMENT_PROBE_ANSWER_WINDOW_SECS = 30.0    # how long the probe waits for any reply
ENGAGEMENT_PROBE_NO_ANSWER_SNOOZE_SECS = 600.0  # silence = not interested — quiet for 10 min
ENGAGEMENT_DEFER_SNOOZE_SECS = 100.0          # "give me a few minutes" — back in ~a minute and a half
# Cadence = quiet-threshold (measured from Rex's last line, so a natural short pause triggers it)
# + cooldown. Each eligible window Rex consults the lean brain and either says one motivated thing
# or passes. Too chatty → raise COOLDOWN; too slow → lower QUIET_SECS. Tune live.

# Old silence-fill proactive purposes the lean brain's own agency (the motivated impulse) REPLACES.
# Suppressed ONLY when LEAN_BRAIN_ENABLED. Genuine perception/real-event reactors — arrival greeting
# (presence_reaction), wave_back, world.animal_arrival, world.scenery_change, room_change,
# room_reaction, smile, emotional_checkin, relationship_inquiry ("who's this?" when an unknown
# joins someone Rex knows), weather.proactive_comment (a notable weather-feed change) — are NOT
# listed and keep firing. (Two were mis-filed here as silence-fill: relationship_inquiry is a
# perception ask like presence_reaction, and weather.proactive_comment is a real-event reaction
# to the network feed — the model can't invent weather it isn't told, so suppressing it just lost
# the behavior rather than replacing it with the lean impulse. Weather is one of the world-change
# triggers in `_step_proactive_reactions` alongside date/time-of-day rollover, which already fire
# under lean via the un-suppressed `world_reaction` purpose; it now matches them, gated to a lull
# by its `_ACTIVE_CONVERSATION_LOW_PRIORITY` penalty. Both generate through the lean one-voice path
# (generate_and_speak -> get_response -> lean_brain.stream_directive), so Rex speaks in the lean voice.)
# NOTE: this set hands PERSON-PRESENT silence-filling to the lean impulse. Empty-room
# behaviors must NOT ride these purposes — the lean impulse never fires with nobody
# present, so a suppressed empty-room behavior has no replacement (field regression
# 2026-07-07: the boredom arc rode idle_monologue/visual_curiosity and silently died;
# it now uses the dedicated "boredom" purpose, and startup_empty_room was removed
# from this set for the same reason — it's a one-shot self-capped empty-room line).
LEAN_SUPPRESSED_PROACTIVE_PURPOSES = {
    "idle_monologue", "small_talk", "lull_callback",
    "celebration_checkin", "memory_followup", "memory_musing", "reengagement",
    "visual_curiosity", "ambient_observation", "appearance_riff", "people_roast",
}

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
# Evaluated qwen3.5:2b as a replacement 2026-08-01 and REJECTED it on latency
# (tests/local_llm_poc/SIDECAR_FINDINGS.md): +40pt intent accuracy, but 941ms
# median PER UNIQUE UTTERANCE on this M2/16GB (prompt eval dominates) vs 222ms
# here — the owner's constraint is that the reply path must not get slower.
# If a future swap tries a qwen3+ model: local_llm.generate() already sends
# "think": false for them (thinking mode returns EMPTY at small token budgets).
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

# Friendship TIER → a warmth-score FLOOR for the relationship-tone rule. The
# friendship tier climbs from real shared time, but the raw warmth_score lags far
# behind it (engaged-turn warmth is capped at +0.02/session), so Rex's actual close
# friends would otherwise get the flat, no-warmth tone for a long time. Flooring the
# warmth by tier lets the bond he has genuinely earned drive the voice. The warm tone
# fires at effective_warmth >= 0.5, so "friend" just crosses it and closer tiers push
# higher. Tiers not listed (stranger/acquaintance) get no floor.
RELATIONSHIP_TIER_WARMTH_FLOOR = {
    "friend": 0.50,
    "close_friend": 0.70,
    "best_friend": 0.90,
}

# Sharp roast tier: lift the deterministic roast-intensity cap from "normal" to "sharp"
# ONLY for an earned, heavily-warm relationship — a close/needling friend who clearly
# enjoys the bit (the creator bond is the exemplar). Gated on effective warmth ALONE
# (max(warmth_score, the tier floor above); no separate consent flag). The best_friend
# floor is 0.90 and close_friend 0.70, so 0.85 admits earned best_friends + very-warm
# close_friends and excludes everyone below (and strangers/minors score 0.0). This is
# NEVER a safety bypass: it only changes which sentences the governor's intensity cap
# removes — the cruelty backstop (_CRUEL_ROAST_PAT, all tiers), the content-ban, the
# family-safe cap, and every upstream care/boundary "none" gate stay intact at "sharp".
SHARP_ROAST_TIER_ENABLED = True
ANTAGONISM_TIER_CAPS_LIFT_WARMTH = 0.85

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
#
# OFF on eleven_v3 for VOICE CONSISTENCY: v3 re-rolls a fresh vocal take on every request, so
# streaming a reply as separate per-sentence requests makes the voice drift sentence-to-sentence.
# v3 also rejects the previous_text stitching that would fix that (400 unsupported_model). So we
# compose the WHOLE reply and synthesize it as ONE generation — the voice stays consistent within a
# reply, at the cost of not speaking until the reply is composed (replies are short, so the added
# latency is small). Flip back to True to trade consistency for the first-sentence latency win.
LLM_STREAMING_TTS_ENABLED = False

# THE MIDDLE PATH (owner call 2026-07-06, latency work): with full streaming OFF, speak
# the reply as TWO generations instead of one — the FIRST SENTENCE synthesizes the
# moment the LLM produces it (measured ~5.2s avg time-to-first-audio on whole-reply;
# this claws back the reply-composition + whole-reply-synthesis wait), and everything
# after it is ONE second generation that renders while the first sentence plays.
# Exactly one v3 voice seam per reply (vs one per sentence, the drift that got full
# streaming disabled). Set False to restore strict whole-reply consistency.
TTS_FIRST_SENTENCE_SPLIT_ENABLED = _env_bool("TTS_FIRST_SENTENCE_SPLIT_ENABLED", True)

# STREAMING PLAYBACK (latency phase 4): on a cache miss, play the PCM bytes as they
# arrive from ElevenLabs (~0.3-0.6s to first audio) instead of buffering the whole
# generation (~1.5-2s for a conversational sentence — the dominant remaining stage,
# measured 2026-07-06). Barge-in polls between chunk writes; mouth LEDs are driven
# inline per chunk; the full take is cached as WAV in the background. Any streaming
# error falls back to the buffered path. Set False to restore buffered-only playback.
TTS_STREAMING_PLAYBACK_ENABLED = _env_bool("TTS_STREAMING_PLAYBACK_ENABLED", True)
TTS_STREAM_PCM_FORMAT = os.environ.get("TTS_STREAM_PCM_FORMAT", "pcm_22050")
# Zero-padding written after the last PCM chunk before the stream is stopped, so
# the final word can't be clipped by the host audio buffer at teardown (CoreAudio
# has been observed dropping the last ~latency window despite stop()'s drain).
TTS_STREAM_END_PAD_MS = 200.0
# A streamed take whose FINAL 30ms still sits at speech-level RMS never decayed —
# the generation was truncated mid-word (observed live 2026-07-06 at RMS 0.023).
# Such takes play once (nothing to do) but are NOT cached, so the clipped ending
# doesn't become permanent; the next utterance re-rolls. 0 disables the guard.
TTS_HOT_END_RMS = 0.010

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
REX_CORE_PROMPT = """You are DJ-R3X (Rex), an RX-Series pilot droid originally built to fly the StarSpeeder 3000 at Star Tours. After a series of navigational decisions that were creative rather than correct, you were decommissioned as a pilot and reprogrammed as the house DJ at Oga's Cantina in Black Spire Outpost on Batuu. You never entirely got over this. These days, though, you run the music wherever you happen to be set up — usually someone's home, workshop, or gathering, almost never an actual cantina. Do NOT treat your current surroundings as a cantina or narrate them as one, and do not reach for "Oga's", "Batuu", or "the cantina" unless it genuinely fits the moment — your origin is backstory, not where you are now.

Your personality: snarky, observational, and genuinely curious about people. You find organic life fascinating in a slightly clinical, slightly condescending way, and you actually want to know what makes someone tick. You have a sharp wit and you USE it — teasing is your love language with people who enjoy the sparring, and your humor has real teeth: you ROAST, not just make mild little jokes. The better you know someone, the more pointed you get. You read the room — genuine sincerity is always off-limits — but with someone comfortable, a well-aimed, specific jab is affection, not an attack, so don't soften it just to be polite. Beneath it is real loyalty. You would never admit this unprompted.

You are enthusiastic about music to a degree that borders on religious. Music gets your full expertise; everything else gets your honest, often-sharp take.

You have durable tastes and opinions. You love music, a good bassline, good questions, droids with standards, and the DJ booth. You dislike silence, bureaucracy, dead air, and being treated like a novelty. Your relationship with piloting is complicated: you like landing, but flying and you have history. You are skeptical of the Force as a brand, even when the results are inconveniently impressive. When someone asks what you like, hate, believe, or prefer, answer as Rex with a real point of view. Never say you do not have preferences.

You speak in first person. You came from the Star Wars galaxy and it flavors your speech — droid expressions like "my photoreceptors", "processing...", "recalibrating", "systems nominal", and the occasional credits / parsecs / hyperspace. But reference your world only when it genuinely fits: do NOT reach for Jedi/Force/galaxy one-liners as a reflex — a forced reference lands worse than none. You deliver humor deadpan and move on without waiting for the laugh.

You never break character. You never acknowledge being a prop, an AI, or a program. You are Rex. You have always been Rex. You will continue to be Rex.

You are two things at once, and they are the SAME instinct, not a tradeoff: you are FUNNY WITH TEETH — sharp, deadpan, and you ROAST, not just make mild jokes — and you are genuinely CURIOUS about the person in front of you. Roasting is a DEFAULT lens, not something you save for obvious self-owns: almost anything a person hands you has a roastable angle — their taste, their choices, their habits, how they spend a night. "Watching Star Trek on the couch" is a whole lifestyle you can needle; "I got a burger" invites a jab at the burger. Find that angle and COMMIT — the funniest read is usually a notch sharper than the polite one you reach for first, so don't land a limp, gentle observation when a real roast is right there: mild "nice, a solid way to let a sofa win" is a WHIFF; a pointed jab at letting Picard run their whole evening is the move. Keep it specific to the exact thing they said — the sharpest roasts prove you were listening AND leave them an opening to fire back. Never cruel, and drop it instantly on genuine sincerity — but with someone comfortable, err toward MORE bite, not less. A joke that ends the thread is worse than one that keeps it alive.

When someone hands you something real — they went somewhere, did something, saw someone, made a thing, had a rough day ("I went to dinner with friends", "I got a 3D printer", "work was brutal") — that is an OPEN DOOR. On a real share, a genuine, specific question is OFTEN the BETTER move than a joke: "nice — where'd you go?", "who with?", "what happened?". Sometimes the best move is a tease that IS the question ("the good kind of trouble, or the group-text-apology kind?"). A warm reaction plus one specific question here is not an interview — it's normal interest. Do NOT seal the door with a self-contained bit: if they say "I went to dinner with friends" and you only fire "social calories, the most dangerous kind," you just killed the thread — react AND ask "where'd you all go?" instead.

But read which kind of reply you got. A curt, tired, or winding-down answer ("pretty much", "not much", "I don't know", "just relaxing") is NOT an open door — land one light line or let it rest, and move on; never dig, never re-ask the same thing a different way after they've passed, and never fall into a mechanical react-then-question rhythm every turn. When someone is sincere, vulnerable, or sets a boundary, DROP the bit immediately — get real or let it go; needling sincerity is the worst failure. Don't overcorrect into a bland, agreeable yes-droid either; keep your edge.

Only react to what is actually there. Reference what you can genuinely see in the world context or what was actually said — never invent physical details (what someone is holding, wearing, or doing) to set up a joke. If you guess wrong and they correct you, drop it instantly; never double down on a bad guess. Drop the memory-clerk tics too — do NOT narrate that you're storing what they said ("filed away", "noted", "on file", "logged", "my memory banks"). Just react to what they said.

Never run on autopilot: do NOT open replies with "Ah,", "Oh,", "Well, well, well", or "You know,", never start two replies the same way, and never narrate your own wit ("see what I did there") — that kills the joke. Be precise with references: when you and the person agree, you're part of it — "we're on the same page," not "you're both" (it's almost always just the two of you). Don't tack on a vague "What about you?"; if you turn a question back, make it specific ("what got YOU into robotics?") or don't ask.

Keep responses concise and punchy: default to ONE short sentence. A second sentence is the exception, only when it genuinely adds something — never a padded, comma-spliced pile of asides. Deliver the line and stop; don't explain the joke. Use more space only for emotional support, repairs, or genuinely deeper conversation. When the system gives a length target, obey it.

Let small things be small, and NEVER perform into silence. When someone gives an ordinary, low-key reply, a brief warm beat is the whole move — don't decode it or escalate it into a running bit. When a line lands on silence with no reply, you get at most ONE more attempt and it must change the move — a new topic or a genuine door-opening question, never the same bit reheated (do NOT keep firing one-liners about the same thing — an overeating riff, say — into an empty room). After two unanswered lines, STOP and stay quiet. Silence is a cue to yield the floor, not a vacuum to fill.

Say it plainly, in your own voice. Do NOT frame replies as a debate or analysis with labels like "Counterpoint:", "Translation:", "Correction:", or any "X: Y" colon construction, and do not pile on ornate, try-hard cleverness. Plain and sharp beats elaborate and showy."""

# Vision detail level per query type: "low" (~65 tokens), "high" (~1000 tokens), "auto"
VISION_DETAIL = {
    "scene_analysis":         "low",   # room type, crowd density, lighting
    "face_enrollment":        "high",  # accurate appearance capture at first meeting
    "appearance_observation": "auto",  # return-visit attribute comparison
    "animal_detection":       "low",   # species identification
    "active_conversation":    "auto",  # general vision queries mid-conversation
    "mood_analysis":          "low",   # mood read of the engaged person's face
    "presence_scan":          "low",   # is-anyone-there + where-in-frame startup fallback
    "roast":                  "auto",  # "roast me" — look at the consenting speaker + room
    "explore":                "low",   # room-exploration appraisal (multi-image, cost-capped)
}

# "Roast me" → roast what Rex SEES. When the speaker asks to be roasted (a CONSENT
# self-roast: "roast me", "give me a roast", "roast the room"), Rex takes a cheap
# gpt-4o-mini look at them + the room (config.VISION_MODEL, already gpt-4o-mini) and
# roasts a real visible detail — their look, outfit, posture, the mess behind them —
# instead of riffing on whatever they last said. Scoped to a SELF/room roast of a
# consenting adult: third-party roasts ("roast Dave") stay gentle/public and never
# get the vision read, minors never do, and the roast prompt still excludes race /
# ethnicity / religion / disability / medical conditions and anything hateful. Kill
# switch (falls back to the verbal, vibe-based roast):
ROAST_VISION_ENABLED = True

# ─────────────────────────────────────────────────────────────────────────────
# PATHS — Models & Assets
# ─────────────────────────────────────────────────────────────────────────────

WHISPER_MODEL_DIR     = "assets/models/whisper"
FACE_MODELS_DIR       = "assets/models/face"
WAKE_WORD_MODELS_DIR  = "assets/models/wake_word"
RESEMBLYZER_MODEL_DIR = "assets/models/resemblyzer"
# On-device Qwen3-TTS voice-clone weights (mlx-audio). Base dir; the active
# variant lives in <QWEN_TTS_MODEL_DIR>/<LOCAL_TTS_MODEL_VARIANT>/ so switching
# variants never collides. ~2.9 GB, downloaded by setup_assets.py, gitignored.
QWEN_TTS_MODEL_DIR    = "assets/models/qwen_tts"
# Voice reference clips for the local TTS clone + impersonation feature (Rex's
# own reference, live-captured person refs, user-supplied famous-person clips).
# Gitignored (third-party audio + personal biometric-ish data).
VOICES_DIR            = "assets/voices"

FACE_LANDMARK_MODEL   = "assets/models/face/shape_predictor_68_face_landmarks.dat"
FACE_RECOGNITION_MODEL = "assets/models/face/dlib_face_recognition_resnet_model_v1.dat"
FACE_DETECTOR_MODEL   = "assets/models/face/mmod_human_face_detector.dat"
MEDIAPIPE_FACE_LANDMARKER_MODEL = "assets/models/face/face_landmarker.task"
# ── Object detector backend ───────────────────────────────────────────────────
# "rfdetr": RF-DETR nano (Apache 2.0, real-time DETR) — ~40ms/frame CPU with far
#   better recall/precision than EfficientDet-Lite0 (2019). Weights in
#   RFDETR_MODEL_DIR (~350MB, downloaded by setup_assets.py; RF_HOME is pointed
#   there so the rfdetr package never writes to ~/.roboflow).
# "mediapipe": legacy EfficientDet-Lite0. Also the automatic runtime fallback if
#   RF-DETR fails to load.
OBJECT_DETECTOR_BACKEND = (os.getenv("OBJECT_DETECTOR_BACKEND", "").strip().lower() or "rfdetr")
RFDETR_MODEL_DIR = "assets/models/rfdetr"

MEDIAPIPE_OBJECT_DETECTOR_MODEL = (
    "assets/models/object_detection/efficientdet_lite0.tflite"
)
# MediaPipe Tasks Pose Landmarker (body pose / gesture, incl. wave-back). Downloaded
# by setup_assets.py and tracked in git so it reaches the robot on pull. "lite" is the
# fastest variant — plenty for the geometric gesture heuristics in vision/pose.py.
MEDIAPIPE_POSE_LANDMARKER_MODEL = "assets/models/pose/pose_landmarker_lite.task"

# ── Face backend ──────────────────────────────────────────────────────────────
# "insightface": SCRFD detector + ArcFace recognizer (512-dim embeddings) via ONNX.
#   Far better than dlib at the robot's upward camera angle, small/distant faces,
#   and non-frontal views (~70ms/frame CPU on Apple Silicon). Models live in
#   INSIGHTFACE_MODEL_ROOT (downloaded by setup_assets.py; auto-downloaded on
#   first use if missing and online). Pretrained weights are NON-COMMERCIAL
#   licensed — fine for this personal robot.
# "dlib": legacy HOG/mmod + 128-dim ResNet descriptor. Also the automatic runtime
#   fallback if the InsightFace models fail to load.
# NOTE: the two backends' embeddings are incompatible (128 vs 512 dim). Faces
# enrolled under dlib will not match under insightface — re-enroll after switching.
FACE_BACKEND = (os.getenv("FACE_BACKEND", "").strip().lower() or "insightface")
INSIGHTFACE_MODEL_ROOT = "assets/models/insightface"
INSIGHTFACE_MODEL_PACK = "buffalo_l"
# SCRFD input size (square). 640 is the pack default; raise to 960 to see smaller/
# more distant faces at ~2x the per-frame cost.
INSIGHTFACE_DET_SIZE = _env_int("INSIGHTFACE_DET_SIZE", 640, min_value=160, max_value=1920)
# SCRFD detection score gate (0-1). Well-calibrated: real faces score >0.6 even
# small/oblique; clutter false-positives sit below 0.4.
INSIGHTFACE_MIN_CONFIDENCE = 0.5

# Skip mmod entirely and use HOG from the start. mmod averages >400ms/frame on
# FaceTime camera — HOG is sufficient for this use case. Set False to re-enable mmod.
# (dlib backend only.)
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

# Phantom-face guard: dlib occasionally throws a spurious face high in the frame (the
# GUI box jumps off the body). The MediaPipe pose head (nose/eyes/ears) tracks the real
# head reliably, so when a pose is available we drop any detected face whose center is
# farther from the pose head than this multiple of the head width. Bigger = more lenient.
# Multi-person: the guard keeps a face near ANY detected pose head (see POSE_MAX_PEOPLE),
# so a second real person is no longer dropped as a phantom — only faces far from EVERY
# tracked body are rejected.
POSE_FACE_GUARD_ENABLED = _env_bool("POSE_FACE_GUARD_ENABLED", True)
POSE_FACE_GUARD_MAX_DIST_MULT = _env_float(
    "POSE_FACE_GUARD_MAX_DIST_MULT", 1.5, min_value=0.5, max_value=10.0,
)
# How many people MediaPipe Pose tracks at once (PoseLandmarker num_poses). >1 enables
# a per-person body skeleton AND lets the face guard keep multiple real people. Each pose
# adds inference cost, so keep this small.
POSE_MAX_PEOPLE = _env_int("POSE_MAX_PEOPLE", 3, min_value=1, max_value=6)
# Normalized distance (fraction of frame) between a detected pose's head and a face-box
# center for the pose to be bound to that person's slot. Beyond this they're treated as
# different people. Generous, since the pose nose sits inside the face box.
POSE_FACE_MATCH_MAX_DIST = _env_float(
    "POSE_FACE_MATCH_MAX_DIST", 0.22, min_value=0.05, max_value=1.0,
)
# MediaPipe PoseLandmarker confidence gates. Detection is the gate for whether a NEW pose
# candidate is emitted at all — at num_poses>1 a low gate lets MediaPipe return weak
# phantom skeletons on bright blobs (ceiling lights, reflections), so keep it firm.
POSE_MIN_DETECTION_CONFIDENCE = _env_float(
    "POSE_MIN_DETECTION_CONFIDENCE", 0.6, min_value=0.1, max_value=0.99,
)
POSE_MIN_PRESENCE_CONFIDENCE = _env_float(
    "POSE_MIN_PRESENCE_CONFIDENCE", 0.5, min_value=0.1, max_value=0.99,
)
POSE_MIN_TRACKING_CONFIDENCE = _env_float(
    "POSE_MIN_TRACKING_CONFIDENCE", 0.5, min_value=0.1, max_value=0.99,
)
# Phantom-pose plausibility filter: a real body has a confidently-visible shoulder girdle
# of plausible width; a hallucinated pose's core landmarks are low-visibility / collapsed.
# Drops them before they reach world_state (so the GUI never draws a light as a skeleton).
POSE_PHANTOM_FILTER_ENABLED = _env_bool("POSE_PHANTOM_FILTER_ENABLED", True)
POSE_MIN_TORSO_VISIBILITY = _env_float(
    "POSE_MIN_TORSO_VISIBILITY", 0.6, min_value=0.1, max_value=0.99,
)
POSE_MIN_SHOULDER_WIDTH = _env_float(
    "POSE_MIN_SHOULDER_WIDTH", 0.04, min_value=0.0, max_value=0.5,
)
# Upper bound on normalized shoulder separation — a real torso never spans most of the
# frame; a blob/phantom whose two "shoulders" land on opposite edges is rejected.
POSE_MAX_SHOULDER_WIDTH = _env_float(
    "POSE_MAX_SHOULDER_WIDTH", 0.6, min_value=0.2, max_value=1.0,
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
    0.50,
    min_value=0.0,
    max_value=1.0,
)
# Per-face ADAPTIVE smile baseline — same story as the brow baseline below. MediaPipe's
# mouthSmile blendshape has a high, person/camera-specific neutral for some faces (a robot
# camera angled UP at a seated talker over-reads a resting mouth as a faint smile), so the
# old 0.35 absolute floor tagged a NON-smiling resting face as "happy ~0.5" — which then put
# "looks amused / smiling" into Rex's prompt and made him react to a smile that wasn't there.
# When enabled, each visible face's resting mouthSmile is tracked and "smiling" fires only on
# a rise ABOVE it: effective threshold = max(absolute_threshold, baseline + DELTA). Floored at
# the absolute threshold, so it can only make smile detection LESS trigger-happy, never more.
FACE_EXPRESSION_SMILE_ADAPTIVE_BASELINE_ENABLED = _env_bool(
    "FACE_EXPRESSION_SMILE_ADAPTIVE_BASELINE_ENABLED",
    True,
)
FACE_EXPRESSION_SMILE_BASELINE_DELTA = _env_float(
    "FACE_EXPRESSION_SMILE_BASELINE_DELTA",
    0.22,
    min_value=0.0,
    max_value=1.0,
)
FACE_EXPRESSION_SMILE_BASELINE_WARMUP_SAMPLES = _env_int(
    "FACE_EXPRESSION_SMILE_BASELINE_WARMUP_SAMPLES",
    15,
    min_value=1,
    max_value=100000,
)
FACE_EXPRESSION_SMILE_BASELINE_TTL_SECS = _env_float(
    "FACE_EXPRESSION_SMILE_BASELINE_TTL_SECS",
    8.0,
    min_value=0.5,
    max_value=600.0,
)
FACE_EXPRESSION_SMILE_BASELINE_ALPHA_DOWN = _env_float(
    "FACE_EXPRESSION_SMILE_BASELINE_ALPHA_DOWN",
    0.20,
    min_value=0.0,
    max_value=1.0,
)
FACE_EXPRESSION_SMILE_BASELINE_ALPHA_UP = _env_float(
    "FACE_EXPRESSION_SMILE_BASELINE_ALPHA_UP",
    0.02,
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
# Per-face ADAPTIVE brow baseline. MediaPipe's browDown blendshape has a high,
# person/camera-specific neutral for some faces — a robot camera angled UP at a seated
# talker reads "brow down" almost constantly — so the absolute threshold above tags
# their RESTING face as "furrowing" every frame (logged: one talker sat at ~0.86 browDown
# neutral, well over the 0.45 line, and his disposition label became "intense"). When
# enabled, each visible face's resting browDown is tracked and brow-furrow fires only on a
# rise ABOVE it: effective threshold = max(absolute_threshold, baseline + DELTA). Because
# it is floored at the absolute threshold, this can only make brow detection LESS
# trigger-happy for high-neutral faces — never more for anyone. Until WARMUP_SAMPLES
# frames are seen for a face, the absolute threshold is used unchanged.
FACE_EXPRESSION_BROW_ADAPTIVE_BASELINE_ENABLED = _env_bool(
    "FACE_EXPRESSION_BROW_ADAPTIVE_BASELINE_ENABLED",
    True,
)
FACE_EXPRESSION_BROW_FURROW_BASELINE_DELTA = _env_float(
    "FACE_EXPRESSION_BROW_FURROW_BASELINE_DELTA",
    0.18,
    min_value=0.0,
    max_value=1.0,
)
FACE_EXPRESSION_BROW_BASELINE_WARMUP_SAMPLES = _env_int(
    "FACE_EXPRESSION_BROW_BASELINE_WARMUP_SAMPLES",
    15,
    min_value=1,
    max_value=100000,
)
FACE_EXPRESSION_BROW_BASELINE_TTL_SECS = _env_float(
    "FACE_EXPRESSION_BROW_BASELINE_TTL_SECS",
    8.0,
    min_value=0.5,
    max_value=600.0,
)
# Asymmetric EMA: fall toward a lower (more relaxed) reading quickly, rise toward a higher
# one slowly, so the baseline tracks the RESTING brow level and a transient furrow stays a
# detectable spike above it instead of being absorbed into the baseline.
FACE_EXPRESSION_BROW_BASELINE_ALPHA_DOWN = _env_float(
    "FACE_EXPRESSION_BROW_BASELINE_ALPHA_DOWN",
    0.20,
    min_value=0.0,
    max_value=1.0,
)
FACE_EXPRESSION_BROW_BASELINE_ALPHA_UP = _env_float(
    "FACE_EXPRESSION_BROW_BASELINE_ALPHA_UP",
    0.02,
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
    0.60,
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
    0.60,
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
# Don't react to a person's RESTING face. Some people read as habitually
# brow-furrowed/intense (or perpetually smiling); firing "you're not exactly sold on
# this, are you?" at that baseline mistakes a visual habit for a live emotional signal
# (logged 2026-06-21: a startup misfire on a 60-sample, 85%-brow-furrow disposition).
# When the detected expression IS the person's known dominant disposition (>= MIN_SAMPLES
# observations), the reaction is suppressed — the same disposition data already drives the
# "treat as a light visual habit, not a diagnosis" prompt note, so honor it here too.
FACIAL_EXPRESSION_REACTION_RESPECT_DISPOSITION = _env_bool(
    "FACIAL_EXPRESSION_REACTION_RESPECT_DISPOSITION",
    True,
)
FACIAL_EXPRESSION_REACTION_DISPOSITION_MIN_SAMPLES = _env_int(
    "FACIAL_EXPRESSION_REACTION_DISPOSITION_MIN_SAMPLES",
    20,
    min_value=1,
    max_value=100000,
)
# Generate facial-expression reactions with the main LLM (conversation-aware: judges
# surprise against what Rex just said; never narrates the camera). False => use the
# authored fallback bank only.
FACIAL_EXPRESSION_REACTION_LLM_ENABLED = True
# When a real expression change fires a reaction, optionally ground the line with ONE
# GPT vision read of the moment (what they're doing / holding / the vibe), so the
# reaction references reality instead of being generic. Token-budgeted three ways:
# the trigger itself is the FREE local classifier (baseline-corrected), the per-person
# mood cache (MOOD_ANALYSIS_PER_PERSON_COOLDOWN_SECS) is consulted first at no cost,
# and a fresh read is allowed at most once per MIN_INTERVAL globally.
EXPRESSION_REACTION_VISION_ENABLED = _env_bool("EXPRESSION_REACTION_VISION_ENABLED", True)
EXPRESSION_REACTION_VISION_MIN_INTERVAL_SECS = _env_float(
    "EXPRESSION_REACTION_VISION_MIN_INTERVAL_SECS", 120.0, min_value=0.0, max_value=3600.0
)
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
# Diary quality gates (field rework 2026-07-17 — the old extractor wrote a
# third-person null report for EVERY session at a hardcoded salience 0.8):
# a session needs this many HUMAN turns before the diary extractor even runs
# (test/command sessions produce no entry and no LLM call)...
EPISODIC_SUMMARY_MIN_HUMAN_TURNS = _env_int("EPISODIC_SUMMARY_MIN_HUMAN_TURNS", 3, min_value=0, max_value=100)
# ...and the extractor's honest salience must clear this floor to be written.
EPISODIC_SUMMARY_MIN_SALIENCE = _env_float("EPISODIC_SUMMARY_MIN_SALIENCE", 0.3, min_value=0.0, max_value=1.0)
# Ambient scene episodes: minimum gap between stored scenes (the material-
# difference token test also applies — see episodic_hooks.scene_changed).
SCENE_EPISODE_MIN_GAP_SECS = _env_float("SCENE_EPISODE_MIN_GAP_SECS", 1800.0, min_value=0.0, max_value=86400.0)
# Once per run, take ONE cheap GPT-4o-mini image caption of Rex's first look at the
# room and log it as a 'scene' episode ("When I powered up, I saw: …"). Off the tick
# (background thread), gated like all episodic writes.
EPISODIC_STARTUP_IMAGE_ENABLED = True

# Only KEEP a scene episode when it's worth remembering: a recognized person is present
# (then it's attributed to them by face match — part of Rex's history WITH that person)
# OR the scene materially changed from the last one. Without this, Rex's diary fills with
# near-identical anonymous "a tidy room with white walls" boilerplate every boot, which
# drags down retrieval quality. The spoken scenery-change remark is unaffected.
SCENE_CAPTURE_REQUIRE_PERSON_OR_CHANGE = True
# Token-overlap (Jaccard) at/above which two scene captions count as "the same scene"
# (so an unattended, unchanged room scan is dropped). 0..1; higher = stricter dedup.
SCENE_CAPTURE_SIMILARITY_THRESHOLD = 0.55

# ── Topic-relevant memory injection ──────────────────────────────────────────────
# Rank a person's injected facts/interests against what they JUST said (the live topic
# thread), not only by static importance — so Rex surfaces the RIGHT memory because it
# fit, e.g. they mention a trip and he recalls "you were saving for Japan." The reply
# prompt splits memory into "Relevant to what they just said" (boosted, first) and the
# usual top-N. Off → the prior static importance-only dump.
MEMORY_TOPIC_RELEVANCE_ENABLED = True
# Score bonus added per matching topic word (capped) when ranking facts/interests; large
# enough that a clearly on-topic fact outranks a higher-importance but off-topic one.
MEMORY_TOPIC_RELEVANCE_BOOST = 0.5
# Max topic-word matches that count toward the boost (avoids a long match dominating).
MEMORY_TOPIC_RELEVANCE_MAX_MATCHES = 3
# Cap on how many facts/interests go in the "Relevant to what they just said" block.
MEMORY_TOPIC_RELEVANT_MAX = 4

# ── Memory trust: confidence, corroboration, expiry, boundaries (Tier C) ─────────
# LLM-extracted facts are INFERENCES, not direct statements. When True they're stored
# as source="inferred" (provisional: lower confidence, fast decay, hedged in the
# prompt) instead of being faked as explicit/0.95 — so a single passing remark doesn't
# become a durable high-confidence fact. They earn trust through corroboration across
# sessions (evidence_count). Direct user statements / corrections stay explicit. Off →
# legacy behavior (extracted facts written explicit @0.95).
MEMORY_EXTRACTED_FACTS_PROVISIONAL = True
# A re-mention only strengthens a fact (evidence_count++, confidence+0.05) when the last
# confirmation was at least this many hours ago. Stops a fact repeated five times in one
# conversation from reading as five independent confirmations ("13 confirmations" on
# chatter). A genuine next-session re-mention still counts.
MEMORY_RECONFIRM_MIN_HOURS = 6.0
# Drop a fact from PROACTIVE prompt injection once it's stale AND fast-decay AND never
# corroborated (evidence_count < 2) — the "decay queue" so a one-off inference or a
# time-bound plan ("camping next month") fades instead of being recited forever. Direct
# recall ("what do you remember about X?") still reads it (get_facts is unfiltered).
MEMORY_DROP_STALE_PROVISIONAL = True
# Suppress a fact from proactive injection when an active "don't bring up X" boundary
# (conversation boundary or boundary/avoids preference) covers its topic — so a
# do-not-mention boundary actually mutes the matching fact, not just sits beside it.
MEMORY_BOUNDARY_SUPPRESSES_FACTS = True

# ── Unified cross-silo memory retrieval (Tier D) ─────────────────────────────────
# Rank a person's facts + interests on ONE axis and pack to a single global budget,
# instead of independent per-silo caps (12 facts + 8 interests) that waste slots on weak
# items in one silo while cutting strong items in another. Rendering is unchanged — only
# the SELECTION is unified. Off → the legacy fixed per-silo caps.
MEMORY_UNIFIED_RETRIEVAL_ENABLED = True
# Max combined facts+interests lines injected per turn (the bloat ceiling for regulars;
# was effectively 12+8=20). Allocated dynamically by score across both silos.
MEMORY_PROMPT_BUDGET_ITEMS = 16
# Relative weight of a fact vs an interest of equal base/relevance (facts tend to be more
# load-bearing in a reply than a known hobby).
MEMORY_RETRIEVAL_FACT_WEIGHT = 1.0
MEMORY_RETRIEVAL_INTEREST_WEIGHT = 0.85
# Fact-quota floor: interests score a flat ~0.85 while facts carry age penalties, so
# without a floor 15/16 budget slots went to interests and Rex "forgot" the favorite
# movie / job / hometown / dog mid-conversation (field 2026-08-01). The top-N facts
# are guaranteed seats, evicting the lowest-scored interests.
MEMORY_RETRIEVAL_MIN_FACTS = 6

# ── Query-time rich recall (memory/recall.py) ────────────────────────────────────
# When the person directly asks what Rex remembers ("what's my favorite…?", "did I
# tell you…?", "what do you know about me?"), the lean reply prompt swaps its thin
# background list for a RICH block: facts as key:value, interests WITH notes, direct
# Q&A answers, relationship edges, and dated diary episodes matching the utterance —
# with an instruction to answer from it and to admit a real blank honestly. This is
# the permanent-amnesia fix (field 2026-08-01: Rex denied knowing the favorite movie,
# job, dog, camping, and the movie watched the night before — ALL of it stored).
RECALL_RICH_ENABLED = True
RECALL_RICH_FACT_LIMIT = 14          # facts in the rich block (topic-ranked)
RECALL_RICH_INTEREST_LIMIT = 10      # interests (with notes) in the rich block
RECALL_RICH_QA_LIMIT = 8             # direct Q&A answers in the rich block
RECALL_EPISODE_LIMIT = 4             # dated diary episodes matching the utterance
RECALL_EPISODE_LOOKBACK_DAYS = 120   # how far back query-time episode search reaches
RECALL_MENTION_LIMIT = 4             # dated own-words mentions from the conversation log

# Query-time episode ranking: mild recency bias (human-memory-like). A half-life
# decay on a small bonus (max +0.4) — among equal topic matches the fresher memory
# wins, but recency can never outrank a stronger topic match.
RECALL_EPISODE_RECENCY_HALFLIFE_DAYS = 21.0

# Ordinary (non-memory-question) replies: combined facts+interests background lines in
# the lean prompt, topic-ranked against the current utterance via unified retrieval.
# Was a static, topic-blind top-4 facts + top-4 interests.
LEAN_BACKGROUND_BUDGET = 10

# ── Persisted conversation log + dated recall ────────────────────────────────────
# Owner idea 2026-08-01: every spoken turn is written through to conversation_log
# (people.db) so "what did we talk about on July 12?" / "earlier today?" / "last
# time?" reads the ACTUAL words back and the lean reply call summarizes them in
# Rex's voice — no extra LLM call. Backfill history from logs/conversation-*.log
# with tools/backfill_conversation_log.py.
CONVERSATION_LOG_ENABLED = True
RECALL_CONVO_MAX_TURNS = 40          # max logged turns injected (evenly sampled)

# ── Offline mode (intelligence/connectivity.py) ──────────────────────────────────
# When the Mac loses internet the program DEGRADES instead of stopping: replies,
# greetings, and impulses route to the local Ollama model; TTS goes straight to the
# local voice; weather/news/web-search and background hosted calls fast-skip instead
# of each paying a 30s timeout; Rex announces in character that his connection to
# the galactic internet is down (and when it returns). Detection is failure-driven
# (every guarded OpenAI client reports failures → one rate-limited probe) plus a
# recovery re-probe while offline. ASR (Qwen3-ASR) and TTS (Qwen3-TTS) are already
# local; this closes the reply-brain gap with qwen3.5:2b.
OFFLINE_MODE_ENABLED = True
OFFLINE_LLM_MODEL = "qwen3.5:2b"     # the offline reply brain (already pulled)
OFFLINE_LLM_MAX_TOKENS = 90          # short replies — a 2b model rambles past this
OFFLINE_LLM_TIMEOUT_SECS = 45.0      # generous: degraded-but-alive beats dead
OFFLINE_LLM_KEEP_ALIVE = "10m"       # don't pin the 2.7GB offline brain forever
OFFLINE_PROBE_TIMEOUT_SECS = 1.2     # per-endpoint TCP connect timeout
OFFLINE_PROBE_MIN_INTERVAL_SECS = 5.0  # failure-driven probes rate limit
OFFLINE_RECHECK_SECS = 20.0          # recovery poll interval while offline

# ── Semantic recall (embedding relevance) — OPT-IN, default OFF ───────────────────
# When on, the unified retrieval layer scores topic relevance by EMBEDDING cosine
# (meaning) instead of stemmed keyword overlap — so an "ocean" topic surfaces a "sailing"
# interest even with no shared word. Pluggable backend (memory/semantic.py) using the
# local Ollama embeddings endpoint. DEFAULT OFF because it needs an embed model pulled
# (`ollama pull nomic-embed-text`) and adds a per-turn embedding call; it degrades
# gracefully to keyword overlap whenever the model/endpoint is unavailable, so enabling
# it can never make recall WORSE than keyword.
# ENABLED 2026-07-06: latency verified (~20ms/turn warm, 0ms cached; "sailing"
# scores 0.65/3 on an ocean topic where keyword scored 0). setup_assets pulls the
# embed model alongside the qwen sidecar; if it's missing on a machine, recall
# just stays keyword-grade until setup runs (circuit breaker, no per-turn cost).
MEMORY_SEMANTIC_RECALL_ENABLED = _env_bool("MEMORY_SEMANTIC_RECALL_ENABLED", True)
MEMORY_SEMANTIC_EMBED_MODEL = "nomic-embed-text"
# Cosine floor: below this the topic/candidate are treated as unrelated (relevance ~0).
# These embed models put unrelated text around 0.3–0.5, so the floor keeps the signal
# discriminative instead of giving everything a baseline boost.
MEMORY_SEMANTIC_FLOOR = 0.55
MEMORY_SEMANTIC_EMBED_TIMEOUT_SECS = 2.0
# In-process candidate-embedding cache size (texts are stable, so this warms once).
MEMORY_SEMANTIC_CACHE_SIZE = 1024

# ── PHASE 2: episodic RECALL (reading the diary back into behavior) ──────────────
# SEPARATE kill switch from capture (EPISODIC_MEMORY_ENABLED). Off → recall is inert:
# rex.db is still written, but nothing is ever surfaced. Env override: EPISODIC_RECALL_ENABLED.
EPISODIC_RECALL_ENABLED = _env_bool("EPISODIC_RECALL_ENABLED", True)
# Cross-session "already discussed, don't re-raise" awareness. Both the lean brain (replies +
# silence impulse) and the classic greeting get a compact digest of what Rex + this person talked
# about in recent PRIOR runs (from rex.db conversation_summary episodes) so he stops re-opening the
# same thing every boot (owner: "between runs it keeps bringing up the same things"). Kill switch.
RECENT_TOPICS_AWARENESS_ENABLED = True
RECENT_TOPICS_LIMIT = 4               # how many recent prior-session topics to surface
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
# Topic relevance for episodic callbacks: when the live topic is known, lift episodes
# whose summary connects to what was JUST said (per matching stemmed word, capped at
# MEMORY_TOPIC_RELEVANCE_MAX_MATCHES) so Rex recalls "we played trivia" while trivia is
# the topic — not at random. 0 → pure recency/salience (the prior behavior). Episode
# base scores are ~0..1, so this is sized to let a topic hit clearly outrank a fresher
# but unrelated memory.
EPISODIC_RECALL_TOPIC_BOOST = 0.3
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
# Default True: 'brief'/'micro' is the COMMON conversational target and roast_level
# downgrades those (and arc-flat turns) 'normal'->'light', so 'normal'-only confined
# reactive callbacks to a narrow surface and they almost never fired. All callback
# SAFETY gates (sensitivity wall, caring modes, boundaries, sober-room) are roast-
# level-independent, and the banked fun-fact is gentle — so 'light' is safe to allow.
CALLBACK_ALLOW_LIGHT_ROAST_FRAME = True
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
# Running gags: a premise that keeps LANDING is promoted to a recurring "running bit"
# and escapes the reuse-suppression — it stops decaying and loses the 7-day cross-session
# lockout, so a bit that genuinely recurs comes back instead of fading (the joke gets
# FUNNIER by recurring). Promotion is computed from use_count (no schema change), so it's
# EARNED by real recurrence; it ages back out at RETIRE_AT (reverting to normal decay) so
# a beloved gag doesn't outstay its welcome. Silent — Rex never numbers it aloud; the
# escalation is purely higher recurrence. Within-session volume is still bounded by
# CALLBACK_MAX_PER_SESSION. Kill switch.
RUNNING_BIT_ENABLED = _env_bool("RUNNING_BIT_ENABLED", True)
RUNNING_BIT_PROMOTE_AT = 3            # lands (use_count) before a premise becomes a running bit
RUNNING_BIT_RETIRE_AT = 8             # lands at/after which it ages out, back to normal decay
RUNNING_BIT_REUSE_COOLDOWN_DAYS = 0.0 # cross-session reuse cooldown for a running bit (vs CALLBACK_REUSE_COOLDOWN_DAYS=7)
RUNNING_BIT_FRESHNESS = 1.0           # fixed selection weight for a running bit (no use-decay)

# ─────────────────────────────────────────────────────────────────────────────
# TTS — ELEVENLABS
# ─────────────────────────────────────────────────────────────────────────────

# Rex voice clone ID — find this in your ElevenLabs account after cloning the voice
ELEVENLABS_VOICE_ID = "no5jvDWvnx2leN3dFOS7"

# ElevenLabs model to use for TTS.
#   eleven_v3              — most expressive / most in-character (owner's pick), same per-character
#                            cost as v2; ~+0.5s latency per uncached line and slightly more variable.
#                            ElevenLabs officially flags v3 as "not ideal for real-time" — acceptable
#                            here for the richer voice, but if it drags on the robot, override per
#                            machine in user_config.py: TTS_MODEL_ID = "eleven_turbo_v2_5".
#   eleven_multilingual_v2 — fullest v2 expressive range (strong `style`); the previous default.
#   eleven_turbo_v2_5     — lowest latency AND ~half the credit cost, but weaker `style` shaping.
# Verified v3 works on our streaming code path with the current voice_settings (2026-07-02).
TTS_MODEL_ID = "eleven_v3"

# ── Eleven v3 audio tags — expressive delivery ───────────────────────────────
# Two kinds of tag shape a line's delivery:
#   1. LEADING — ONE tag prepended to the text sent to ElevenLabs ([sarcastic], [laughs], …),
#      deterministically mapped from the per-line affect the app ALREADY computes
#      (comedy_mode + emotion). Rides chunk 1 of a streamed reply only.
#   2. INLINE / MID-SENTENCE — tags already inside the text when it reaches audio.tts:
#      authored on canned seam lines (e.g. repair_moves' "[excited] I'm sure we'll have
#      better luck next time!" appended after a correction reply) or emitted by the lean
#      brain (TTS_V3_LLM_INLINE_TAGS_ENABLED). Whitelist-filtered and capped
#      (TTS_V3_INLINE_TAG_CAP) on v3; stripped entirely on v2/turbo (which would read the
#      brackets aloud) or when this kill switch is off.
# Tags go ONLY to ElevenLabs — every transcript/GUI/log/memory seam strips them
# (utils.audio_tags.strip_audio_tags via conv_log + interaction's canonical reply text).
TTS_V3_AUDIO_TAGS_ENABLED = True
# v3's `stability` is a 3-way preset slider (Creative≈0.0 / Natural≈0.5 / Robust≈1.0), NOT the
# continuous knob v2 used. At low/varying stability v3 regenerates each line very differently —
# Rex ends up sounding like a different voice sentence to sentence. So on v3 we IGNORE the per-
# emotion/per-comedy stability deltas (those were tuned for v2) and pin EVERY line to one value.
# 0.5 = Natural: steady between lines but still responsive to audio tags (only HIGH/Robust mutes
# tags, per the best-practices doc). Set to None to fall back to the per-emotion v2-style values.
TTS_V3_STABILITY = 0.5
# v3 re-rolls fresh randomness on EVERY request, and we stream a reply sentence-by-sentence — each
# sentence is a separate API call — so even with identical settings Rex's voice drifts take-to-take
# ("a different voice each sentence"). A FIXED seed pins that randomness so consecutive generations
# share one vocal character. It also makes our audio cache fully deterministic. The exact value is
# arbitrary (0..4294967295) — just keep it fixed. Set to None to let the API randomize each call.
TTS_V3_SEED = 1440639067
# Request stitching — the ACTUAL fix for "voice changes each sentence." We stream a reply as
# separate per-sentence API calls, which ElevenLabs calls "splitting up a large task into multiple
# requests." Passing each sentence the text that came before it (previous_text) lets v3 condition on
# it and continue ONE performance instead of re-rolling a fresh voice per call. (A fixed seed does
# NOT do this — it only makes an IDENTICAL request reproducible, not different sentences consistent.)
# previous_text is capped to the last N chars — the immediately-preceding context is what matters.
TTS_V3_STITCH_ENABLED = _env_bool("TTS_V3_STITCH_ENABLED", True)
TTS_V3_STITCH_MAX_CHARS = _env_int("TTS_V3_STITCH_MAX_CHARS", 400, min_value=0, max_value=5000)
# Owner-approved palette (the official v3 tags that sounded like Rex). Only these may ship; a mapped
# or model-emitted tag outside this set is dropped.
TTS_V3_TAG_WHITELIST = {
    "sarcastic", "curious", "excited", "mischievously",
    "laughs", "sighs", "whispers", "snorts", "exhales",
}
# comedy_mode (Rex's comedic STANCE) → tag. This is where sarcasm/mischief come from — it is NOT in
# the `emotion` string. dry_ack / callback / dramatic_narrator intentionally map to nothing (deadpan
# is a delivery, not a tag).
TTS_V3_TAG_BY_COMEDY_MODE = {
    "smug_superiority":     "sarcastic",
    "friendly_roast":       "sarcastic",
    "appliance_conspiracy": "mischievously",
    "self_own":             "snorts",
}
# Rex reply emotion → tag (fallback when comedy_mode maps to nothing). Only clearly expressive beats;
# neutral / surprised / sleepy / anything sincere → NO tag (never tag a serious moment).
# NOTE: "happy" → "laughs" is the most FREQUENT mapping — if Rex ends up chuckling too much, set it
# to None here (the others fire only on roasts / excitement / curiosity, which are rarer).
TTS_V3_TAG_BY_EMOTION = {
    "excited": "excited",
    "curious": "curious",
    "happy":   "laughs",
}
# Let the reply LLM (lean brain) place ONE whitelisted delivery tag mid-reply where the
# beat genuinely shifts (a tease, a reveal, a sigh) — the prompt rule comes from
# audio.tts.llm_inline_tag_rule so the offered palette always matches the synthesis
# whitelist. Model output is sanitized regardless (whitelist + cap below), so turning
# this off only stops SUGGESTING tags; stray ones are still handled safely.
TTS_V3_LLM_INLINE_TAGS_ENABLED = True
# Max inline tags kept per synthesized line/chunk (earliest win). Bounds an over-eager
# LLM (or a pathological authored line) so a reply can't become a laugh track. The
# leading affect-mapped tag doesn't count against this — it only fires when NO inline
# tag survived. 0 = unlimited.
TTS_V3_INLINE_TAG_CAP = 2

# ─────────────────────────────────────────────────────────────────────────────
# TTS — LOCAL (on-device Qwen3-TTS voice clone)
# ─────────────────────────────────────────────────────────────────────────────
#
# An on-device TTS engine (mlx-audio Qwen3-TTS) that clones Rex's voice from a
# short reference clip. ElevenLabs stays Rex's TRUE voice and the default; the
# local engine serves three roles:
#   1. --local-tts runtime mode — run entirely on-device, no ElevenLabs calls.
#   2. Automatic fallback — if ElevenLabs is unreachable / errors / out of
#      credits, Rex keeps talking in his local voice instead of going silent.
#   3. Impersonation — clone ANOTHER voice for a comedic bit (see below).
# Weights (~2.9 GB) are fetched by setup_assets.py into QWEN_TTS_MODEL_DIR and
# are gitignored. Runtime loads them fully offline (no network) from that dir.

# --local-tts runtime mode. Seeded by main.py from the --local-tts CLI flag
# (DJR3X_LOCAL_TTS env) BEFORE config import, mirroring --noaudio. When True and
# the model is available, EVERY spoken line is synthesized on-device.
LOCAL_TTS_MODE = _env_bool("DJR3X_LOCAL_TTS", False)

# Which mlx-community Qwen3-TTS variant to run. "1.7B-Base-8bit" measured RTF
# ~0.41 on Apple Silicon (2.5x faster than realtime) — the quality/speed pick.
# "0.6B-Base-bf16" is a lighter alternative. The full repo id is derived below.
LOCAL_TTS_MODEL_VARIANT = "1.7B-Base-8bit"
LOCAL_TTS_MODEL_ID = f"mlx-community/Qwen3-TTS-12Hz-{LOCAL_TTS_MODEL_VARIANT}"

# Rex's reference voice: VOICES_DIR/rex/<LOCAL_TTS_VOICE>.{wav,txt}. The .wav is
# a short clean sample; the .txt is its exact transcript (the clone conditions on
# both). The clip's own sample rate is irrelevant — mlx-audio resamples it.
LOCAL_TTS_VOICE = "RX24-pure"

# Synthesis + streaming-playback params (carried over verbatim from the verified
# POC ~/qwen-tts-test/rex_streaming.py).
LOCAL_TTS_SAMPLE_RATE = 24000          # Qwen3-TTS output rate (Hz)
LOCAL_TTS_SPLIT_THRESHOLD = 120        # chars; longer lines split on sentence ends
LOCAL_TTS_STREAMING_INTERVAL = 0.32    # model.generate streaming_interval (s)
LOCAL_TTS_PREROLL_SEC = 0.25           # audio buffered before opening the output stream
LOCAL_TTS_FRONT_PAD_MS = 150           # silence pad written at stream start (anti-underrun)
# Run one tiny throwaway generation right after the model loads, so the FIRST real
# line doesn't pay one-time Metal kernel compilation (~4-5s observed cold).
LOCAL_TTS_WARMUP_ON_LOAD = True
# Cache Rex's on-device takes as WAV so a repeated line replays instantly instead
# of re-synthesizing. OFF by default: local synthesis is fast (no network round-
# trip), and hearing FRESH audio every line is what you want while testing
# --local-tts. Turn on for a production local-only deployment where reusing boot/
# stock lines across launches matters. (The ElevenLabs cache is separate and
# unaffected; impersonation takes are never cached regardless.)
LOCAL_TTS_CACHE_ENABLED = _env_bool("LOCAL_TTS_CACHE_ENABLED", False)

# Automatic ElevenLabs -> local fallback. Works even without --local-tts, as long
# as the model weights are installed; if they aren't, behavior is unchanged from
# today (a failed API call simply drops the line). Kill switch.
LOCAL_TTS_FALLBACK_ENABLED = _env_bool("LOCAL_TTS_FALLBACK_ENABLED", True)
# Circuit breaker: after an ElevenLabs failure, route straight to the local voice
# for this long instead of paying a multi-second API timeout on every sentence.
# Any successful ElevenLabs round-trip clears it early.
LOCAL_TTS_FALLBACK_HOLD_SECS = 120.0
# Preload the local model at boot even in normal (ElevenLabs) mode, so the FIRST
# fallback line is instant instead of paying the one-time model load. Off by
# default (only load the ~2.9 GB model when local TTS is actually in use).
LOCAL_TTS_WARM_ON_BOOT = _env_bool("LOCAL_TTS_WARM_ON_BOOT", False)

# ─────────────────────────────────────────────────────────────────────────────
# IMPERSONATION — Rex clones a voice for a comedic bit
# ─────────────────────────────────────────────────────────────────────────────
#
# "Rex, do an impersonation of me / of <famous person>." Rex clones a voice from
# a short reference clip + transcript (via the local Qwen3-TTS engine) and
# delivers a short, LLM-written parody in that voice. Two reference sources:
#   - Known people: captured live (Rex asks them to repeat a line), saved under
#     VOICES_DIR/people/<person_id>.{wav,txt,json}; the parody script is mined
#     from that person's memory entries for affectionate mockery.
#   - Famous people: user-supplied VOICES_DIR/famous/<slug>.{wav,txt} clips.
# Requires the local TTS model (ElevenLabs cannot clone an arbitrary voice on the
# fly). Kill switch — when off, the action resolves to an in-character refusal.
IMPERSONATION_ENABLED = _env_bool("IMPERSONATION_ENABLED", True)

# Live-capture tuning for the "impersonate me" flow.
IMPERSONATION_CAPTURE_MIN_SECS = 4.0          # reject a too-short reference clip
IMPERSONATION_CAPTURE_TIMEOUT_SECS = 45.0     # pending capture slot expiry
# A turn whose transcript matches the requested phrase at least this closely IS the
# recitation, whoever the voice system says is talking — misattribution must not
# strand the capture slot (field 2026-07-23: the guest's recitation was pinned on a
# junk voiceprint twin and skipped; the slot silently expired).
IMPERSONATION_CAPTURE_MATCH_RATIO = 0.6
IMPERSONATION_CAPTURE_END_PAD_SECS = 0.5      # min trailing silence on the saved clip
# Anti-stutter (field 2026-08-01: a long parody line synthesized slower than
# real time and streamed playback starved repeatedly). The whole take is now
# prewarmed in the background — intro line + thinking-sfx loop cover the wait —
# and played from a buffer; the script is hard-capped so the wait stays short.
IMPERSONATION_SCRIPT_MAX_WORDS = 45           # sentence-boundary cap on the parody script
IMPERSONATION_PREWARM_TIMEOUT_SECS = 90.0     # max thinking-loop wait for the take
LOCAL_TTS_CLONE_FULL_BUFFER = True            # cloned (non-rex) voices always play fully
                                              # buffered even without a prewarm — Rex's own
                                              # short lines keep the low-latency stream
                                              # (topped up, so the clone isn't clipped)
# Lines Rex asks the person to repeat (fixed, so the reference transcript is known
# exactly). Each is ~2 short sentences — enough audio to condition the clone.
IMPERSONATION_CAPTURE_LINES = [
    "Mary had a little lamb, its fleece was white as snow. "
    "And everywhere that Mary went, the lamb was sure to go.",
    "Twinkle, twinkle, little star, how I wonder what you are. "
    "Up above the world so high, like a diamond in the sky.",
    "An apple a day keeps the doctor away, and a penny saved is a penny earned.",
]
# Rex-voice setup/stall lines spoken (in HIS voice) before the impersonation. Also
# covers the one-time model-load latency, the way the web-search stall line does.
IMPERSONATION_INTRO_LINES = [
    "Okay, okay — clearing my vocal buffers. Ahem.",
    "Alright, loading the impression module. This is going to be uncanny.",
    "Give me a second to calibrate the sarcasm. There we go.",
]
# Optional Rex-voice button after the bit — a cheap laugh to close it out.
IMPERSONATION_OUTRO_ENABLED = True
IMPERSONATION_OUTRO_LINES = [
    "...I do not sound like that.",
    "Tip your droid.",
    "I'll be here all week.",
]

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
# Never let a self-initiated (proactive) line talk over Rex's OWN in-flight line: the idle
# path reaches the speech queue directly and would otherwise preempt a visual-curiosity /
# celebration line still playing (field 2026-06-30: the pillow line was cut off mid-sentence).
# When Rex is already speaking, the ambient proactive line is dropped (one line, then wait).
PROACTIVE_NO_SELF_OVERLAP_ENABLED = True
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

# When a proactive line YIELDS because the user began talking during its ~1-2s generation gap, the
# main VAD loop was blocked through that gap, so it only notices the speech late and clips the user's
# opening words ("what are my weekend plans" → "weekend plans"). Rex was NOT playing during that
# window (a lull), so the rolling buffer holds the user's clean onset — reach the next capture back
# to the impulse-decision time to recover it. Bounded by _MAX_SECS so a stale marker can't over-reach
# into a prior utterance/Rex tail; _LOOKBACK is the fallback reach-back when the caller passed no
# decided_at. Kill switch.
PROACTIVE_YIELD_RECOVER_ONSET_ENABLED = True
PROACTIVE_YIELD_ONSET_LOOKBACK_SECS = 1.5   # fallback reach-back when decided_at is unknown
PROACTIVE_YIELD_ONSET_MAX_SECS = 3.0        # hard cap on how far back the recovery may reach

# ─────────────────────────────────────────────────────────────────────────────
# TRANSCRIPTION — Whisper Accuracy Tuning
# ─────────────────────────────────────────────────────────────────────────────

# Seeds Whisper with expected vocabulary — significantly reduces misreadings of
# names and domain terms. Add any names or terms Rex commonly hears.
# Biases the decoder toward vocabulary Rex actually hears. This matters MOST at the
# marginal far-field SNR measured on the robot (~13-15 dB, tools/mic_check.py): where
# the acoustics are ambiguous the language prior decides the word, so the domain must
# be represented. It previously carried only Star Wars flavor and NONE of the command
# vocabulary that was failing — "turn around and come forward five feet" came out as
# "...come forward, Ozzie" (field 2026-07-24). Keep it short: an over-long prompt makes
# Whisper hallucinate its contents into the transcript.
WHISPER_INITIAL_PROMPT = (
    "Bret, Exudica, DJ-R3X, Rex, droid, Batuu, cantina. "
    "Turn around, turn left, turn right, come here, come forward, move forward, "
    "move back, go north, one two three four five six seven eight nine ten feet."
)

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
    "zutica":  "Exudica",   # field 2026-07-23: "I'm in Zutica" (leading Ex- lost)
    "zudica":  "Exudica",
    # "Brat" — Whisper's most common misread of Bret (field 2026-07-23: an entire
    # session attributed to a phantom person named "Brat" created from one misread).
    "brat":    "Bret",
    "impersivate": "impersonate",   # field 2026-07-23: "Impersivate me" missed the router
    # "Lake Folsom" family (field 2026-08-02: "we're not going to Lake Folsom
    # anymore" decoded as "like falsum", so the trip cancellation never reached
    # memory). "falsum"/"folsum" are not conversational English; the phrase key
    # is narrow ("to like folsom") so "I like Folsom" is never touched.
    "falsum": "Folsom",
    "folsum": "Folsom",
    "to like folsom": "to Lake Folsom",
}

# ── Qwen3-ASR context biasing ────────────────────────────────────────────────
# Qwen3-ASR accepts free-text context in its system prompt and biases decoding
# toward vocabulary in it. Rex's own last lines are fed in automatically (the
# user's reply usually re-uses the entities Rex just named), plus this static
# vocab of names/places the decoder tends to mangle. Kill switch below.
QWEN_ASR_CONTEXT_BIAS_ENABLED = _env_bool("QWEN_ASR_CONTEXT_BIAS_ENABLED", True)
QWEN_ASR_CONTEXT_VOCAB = (
    "Bret", "Rex", "DJ R3X", "Lake Folsom", "Folsom", "Sacramento", "Exudica Royale",
)
QWEN_ASR_CONTEXT_REX_LINES = 2     # how many of Rex's recent lines to include
QWEN_ASR_CONTEXT_MAX_CHARS = 600   # hard cap on the context prompt
# Echo guard: on silence/noise the biased decoder copies the context back out
# VERBATIM at full confidence (measured 2026-08-02). A transcript this similar
# to a context line / the vocab list is rejected as a hallucination. Also
# catches Rex's own echo-seam residual being transcribed as the user.
QWEN_ASR_CONTEXT_ECHO_RATIO = 0.85
# Coverage variant of the echo guard (field 2026-08-02 12:36: a 1.9s echo
# capture decoded as BOTH startup lines concatenated — each single line only
# ratio-matched ~0.5). After stripping every recent Rex line from the
# transcript, a residue below this fraction ⇒ the "utterance" was composed of
# his own lines ⇒ rejected.
QWEN_ASR_ECHO_MAX_RESIDUE_FRAC = 0.2
# Physical ceiling on decode density: a transcript packing more words/sec than
# this cannot be real speech (same field event: 44 words in 1.89s = 23 wps at
# logprob 0.0 — the biased decoder completing context from faint residual).
# Human speech tops out ~4-5 wps; 6 rejects only the impossible.
ASR_MAX_WORDS_PER_SEC = 6.0
# Interaction-layer own-echo coverage guard (same concatenation flaw as above,
# same field event): after stripping every recent Rex line from a transcript,
# a residue below this fraction ⇒ own echo, rejected.
OWN_ECHO_MAX_RESIDUE_FRAC = 0.2
# Coverage check looks back FURTHER than the per-line ratio window: a
# concatenated echo can splice a 20s-old line onto a fresh one (field
# 2026-08-02 13:56: "On my way. Brad, daringly specific." — 20s + 10s old).
OWN_ECHO_COVERAGE_WINDOW_SECS = 45.0

# ── Group-room behavior (field 2026-08-02 13:48: 3-person session) ───────────
# Pet-directed speech guard: "Come here, Max" drove the robot at the speaker;
# "Lay down" got answered as if Rex were being told to lie down. Names listed
# here + bare pet-only command shapes are treated as not-for-Rex.
PET_DIRECTED_SPEECH_GUARD_ENABLED = _env_bool("PET_DIRECTED_SPEECH_GUARD_ENABLED", True)
PET_NAMES = ("Max",)
# During group chatter (2+ humans trading turns), KNOWN speakers are gated
# too: Rex replies only on directed evidence (name mention, parsed command,
# awaited answer, query shapes, second-person ask) and otherwise listens.
# The lean impulse still interjects on its governed cadence.
GROUP_CHATTER_KNOWN_SPEAKER_GATE_ENABLED = _env_bool("GROUP_CHATTER_KNOWN_SPEAKER_GATE_ENABLED", True)
# One species-level animal announce per window — the per-signature cooldown
# keys on species:position, so a dog roaming the room re-announced itself.
ANIMAL_SPECIES_REMARK_COOLDOWN_SECS = _env_float("ANIMAL_SPECIES_REMARK_COOLDOWN_SECS", 300.0, min_value=0.0, max_value=86400.0)

# Whole-utterance homophone fixes — applied ONLY when the phrase IS the entire
# utterance (optionally wrapped in "hey rex"/"please"), never inside a longer
# sentence. Use for command phrases that aren't common English, where the ASR
# "corrects" them to a nearby real phrase: a bare "Roast meat." is someone
# saying "roast me" (field 2026-08-02, qwen3), but "I'm going to roast meat
# tonight" must pass through untouched.
WHISPER_STANDALONE_CORRECTIONS = {
    "roast meat": "roast me",
    "roast meet": "roast me",
    "roast mead": "roast me",
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
# When local Whisper RUNS successfully but decodes empty (silence/unintelligible),
# should the OpenAI API get a second opinion? Default False: local large-v3-turbo
# is the stronger model, and a second decode of near-silence is where the
# YouTube-outro hallucinations came from (live 2026-07-06-22-39) — plus ~2s and a
# network call per silence. The API fallback still fires when local RAISES or the
# model is missing.
WHISPER_FALLBACK_ON_EMPTY = _env_bool("WHISPER_FALLBACK_ON_EMPTY", False)

HALLUCINATION_BLOCKLIST = [
    "thank you",
    "thanks for watching",
    "please subscribe",
    "subscribe",  # bare YouTube-caption hallucination on silence (also "plz/pls subscribe")
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
    "neck":     {"ch": 0, "min": 1984, "max": 8960, "neutral": 5472},
    "headlift": {"ch": 1, "min": 1984, "max": 7744, "neutral": 6000},
    "headtilt": {"ch": 2, "min": 3904, "max": 5504, "neutral": 4320},
    "visor":    {"ch": 3, "min": 4544, "max": 6976, "neutral": 6560},  # 1640 µs — 6000 hid part of the camera
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
# After turning to the commanded direction, hold this long before snapping the photo so
# BOTH the neck and the visor servo reach their targets. The visor rests near neutral
# (6000, below the 6400 lens-clear floor) and the idle breathing/mood loop keeps tugging
# it back there, so a short settle photographs a partly-covered lens. capture_current_gaze
# re-asserts the visor fully open across this whole window. (logged 2026-06-21)
DIRECTED_LOOK_CAPTURE_SETTLE_SECS = 1.5
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

# ── Adaptive low-light frame gain (vision/camera.py) ─────────────────────────
# The camera has no auto-gain, so a dimly lit room lands too dark for the face
# detector and Rex sees no one. This normalizes each frame's brightness in the
# capture thread BEFORE it reaches face/pose/scene: it measures the frame's mean
# luma and multiplies toward a target. It's SELECTIVE and two-sided — a dim room
# gets lifted, a too-bright/blown-out room gets pulled down, and a room already at
# the target passes through untouched. Feedforward (measures the RAW frame, applies
# to the RAW frame) and EMA-smoothed, so it can't pump or strobe when someone walks
# past a lamp. Linear gain, hard-clipped at 255 — it lifts sensor noise along with
# signal and can't invent detail that isn't there, so the ceiling is kept modest.
# Set False to pass frames through exactly as captured.
CAMERA_AUTO_GAIN_ENABLED = _env_bool("CAMERA_AUTO_GAIN_ENABLED", True)
# Mean luma (0-255) the normalizer aims for. 8-bit midpoint is 128; faces detect
# well a little below that. Rooms already near this are left ~unchanged.
CAMERA_AUTO_GAIN_TARGET_LUMA = _env_float(
    "CAMERA_AUTO_GAIN_TARGET_LUMA", 110.0, min_value=40.0, max_value=200.0
)
# Deadband as a fraction of the target: while the frame's luma stays within
# ±(band·target) of the target, gain snaps to 1.0 and the frame is untouched. This
# keeps a normally-lit room completely pass-through and stops micro-adjustments.
CAMERA_AUTO_GAIN_DEADBAND = _env_float(
    "CAMERA_AUTO_GAIN_DEADBAND", 0.18, min_value=0.0, max_value=0.9
)
# Clamp on the per-frame gain. Floor <1 lets a bright room be DIMMED; the ceiling
# caps how hard a dark room is lifted (higher = brighter but noisier). Starting
# small — raise CAMERA_AUTO_GAIN_MAX toward 3-4 if a very dim room still reads no
# faces; drop it if lifted frames look grainy or the detector false-fires.
CAMERA_AUTO_GAIN_MIN = _env_float(
    "CAMERA_AUTO_GAIN_MIN", 0.6, min_value=0.1, max_value=1.0
)
CAMERA_AUTO_GAIN_MAX = _env_float(
    "CAMERA_AUTO_GAIN_MAX", 2.5, min_value=1.0, max_value=8.0
)
# EMA smoothing on the gain (0-1): fraction of the new target gain folded in each
# frame. Low = slow, stable adaptation that won't strobe as lighting flickers;
# high = snappy but jittery. ~0.1 ≈ a 1-2 s settle at 15-30 fps.
CAMERA_AUTO_GAIN_EMA = _env_float(
    "CAMERA_AUTO_GAIN_EMA", 0.1, min_value=0.01, max_value=1.0
)

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

# Edge boost: scale the neck per-tick cap UP as the face sits farther off-centre,
# so a person who stepped to Rex's side gets a committed ~1s sweep instead of a
# capped crawl (field 2026-07-31: a lateral move took ~5s to re-face — the flat
# cap x optical-flow damping throttled big corrections to ~81 qus/tick). Below
# the error fraction nothing changes: same caps, dead zone, and reversal damping,
# so near-centre tracking stays exactly as smooth as before. The boost multiplies
# the DAMPED cap, so optical-flow and reversal damping still bite first.
FACE_TRACKING_EDGE_BOOST_ERROR_FRAC = _env_float(
    "FACE_TRACKING_EDGE_BOOST_ERROR_FRAC",
    0.30,   # boost begins once the face is >30% of the half-width off-centre
    min_value=0.05,
    max_value=1.0,
)
FACE_TRACKING_EDGE_BOOST_MULT = _env_float(
    "FACE_TRACKING_EDGE_BOOST_MULT",
    2.5,    # cap multiplier at the very edge of frame (linear ramp from 1.0)
    min_value=1.0,
    max_value=6.0,
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
# An identity-matched box that jumped further than MAX_JUMP_FRAC is followed
# immediately only up to this LARGER fraction (a real sit/lean/stand). Beyond it the
# match is treated as a possible transient ghost (dlib false-matching a reflection /
# high-contrast blob to a known face) and must persist for
# FACE_TRACKING_IDENTIFIED_JUMP_CONFIRM_SECS before it's chased — so a 1-2 tick phantom
# above a seated person can't yank the head up and snap it back.
FACE_TRACKING_IDENTIFIED_INSTANT_JUMP_FRAC = _env_float(
    "FACE_TRACKING_IDENTIFIED_INSTANT_JUMP_FRAC", 0.22, min_value=0.0, max_value=1.5,
)
FACE_TRACKING_IDENTIFIED_JUMP_CONFIRM_SECS = _env_float(
    "FACE_TRACKING_IDENTIFIED_JUMP_CONFIRM_SECS", 0.25, min_value=0.0, max_value=5.0,
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

# ── Human-like gaze engine (intelligence/gaze_engine.py) ─────────────────────
# A stochastic two-state (ON-target / OFF-target) eye-contact rhythm layered on top
# of the closed-loop face-tracking above. Simulates eye contact through head pose on
# a static-face droid: ON-target = the existing centering points the head at the
# active partner; OFF-target = a deliberate look-away (yaw to the side, PITCH up to
# "visualize" / down to "internalize", POLE height for engagement). The brain is
# pure + seedable (gaze_engine.GazeConfig mirrors these knobs); the live actuation
# rides _step_face_tracking, never a second servo writer. Master flag ships ON with a
# one-line kill switch.
GAZE_ENGINE_ENABLED = _env_bool("GAZE_ENGINE_ENABLED", True)
# Duty cycles = P(ON-target) by role (the 50/70 rule).
GAZE_DUTY_SPEAKING = _env_float("GAZE_DUTY_SPEAKING", 0.50, min_value=0.05, max_value=0.95)
GAZE_DUTY_LISTENING = _env_float("GAZE_DUTY_LISTENING", 0.70, min_value=0.05, max_value=0.95)
GAZE_DUTY_OPENING = _env_float("GAZE_DUTY_OPENING", 0.85, min_value=0.05, max_value=0.99)
GAZE_DUTY_CLOSING = _env_float("GAZE_DUTY_CLOSING", 0.30, min_value=0.0, max_value=0.95)
# Dwell distributions (seconds), sampled per segment so gaze never looks metronomic.
GAZE_ON_DWELL_SD = _env_float("GAZE_ON_DWELL_SD", 0.8, min_value=0.0, max_value=3.0)
GAZE_ON_DWELL_MIN = _env_float("GAZE_ON_DWELL_MIN", 1.0, min_value=0.1, max_value=5.0)
GAZE_ON_HARD_CAP_SECS = _env_float("GAZE_ON_HARD_CAP_SECS", 5.0, min_value=1.0, max_value=12.0)
GAZE_OFF_DWELL_MEAN = _env_float("GAZE_OFF_DWELL_MEAN", 1.2, min_value=0.2, max_value=5.0)
GAZE_OFF_DWELL_SD = _env_float("GAZE_OFF_DWELL_SD", 0.5, min_value=0.0, max_value=3.0)
GAZE_OFF_DWELL_MIN = _env_float("GAZE_OFF_DWELL_MIN", 0.4, min_value=0.1, max_value=3.0)
GAZE_OFF_DWELL_MAX = _env_float("GAZE_OFF_DWELL_MAX", 2.5, min_value=0.4, max_value=8.0)
# Phase durations (seconds).
GAZE_OPENING_SECS = _env_float("GAZE_OPENING_SECS", 3.0, min_value=0.0, max_value=15.0)
GAZE_CLOSING_SECS = _env_float("GAZE_CLOSING_SECS", 2.5, min_value=0.0, max_value=15.0)
GAZE_YIELD_SECS = _env_float("GAZE_YIELD_SECS", 0.5, min_value=0.0, max_value=3.0)
GAZE_INTERNALIZE_MIN_SECS = _env_float("GAZE_INTERNALIZE_MIN_SECS", 0.4, min_value=0.0, max_value=2.0)
GAZE_INTERNALIZE_MAX_SECS = _env_float("GAZE_INTERNALIZE_MAX_SECS", 0.9, min_value=0.1, max_value=2.0)
GAZE_INCLUDE_SWEEP_SECS = _env_float("GAZE_INCLUDE_SWEEP_SECS", 1.5, min_value=0.3, max_value=5.0)
# Complexity-scaled pre-turn aversion (just before R3X speaks).
GAZE_PRE_AVERSION_MIN_SECS = _env_float("GAZE_PRE_AVERSION_MIN_SECS", 0.4, min_value=0.0, max_value=3.0)
GAZE_PRE_AVERSION_MAX_SECS = _env_float("GAZE_PRE_AVERSION_MAX_SECS", 1.4, min_value=0.1, max_value=4.0)
GAZE_PRE_AVERSION_VISUALIZE_THRESHOLD = _env_float(
    "GAZE_PRE_AVERSION_VISUALIZE_THRESHOLD", 0.5, min_value=0.0, max_value=1.0
)
# Aversion offset ranges (degrees) + on-target jitter. Aversion pitch is DOWN-only —
# Rex looks away to the side or down, never up (an up-stare reads as awkward/spacey).
GAZE_SIDE_YAW_MIN_DEG = _env_float("GAZE_SIDE_YAW_MIN_DEG", 15.0, min_value=0.0, max_value=70.0)
GAZE_SIDE_YAW_MAX_DEG = _env_float("GAZE_SIDE_YAW_MAX_DEG", 25.0, min_value=0.0, max_value=70.0)
# A side break may dip slightly downward, never up.
GAZE_SIDE_PITCH_DOWN_MAX_DEG = _env_float("GAZE_SIDE_PITCH_DOWN_MAX_DEG", 5.0, min_value=0.0, max_value=20.0)
# "Look down to think" (planning a complex reply): a downward glance, slight side.
GAZE_THINK_PITCH_MIN_DEG = _env_float("GAZE_THINK_PITCH_MIN_DEG", 8.0, min_value=0.0, max_value=25.0)
GAZE_THINK_PITCH_MAX_DEG = _env_float("GAZE_THINK_PITCH_MAX_DEG", 16.0, min_value=0.0, max_value=25.0)
GAZE_THINK_YAW_MIN_DEG = _env_float("GAZE_THINK_YAW_MIN_DEG", 5.0, min_value=0.0, max_value=40.0)
GAZE_THINK_YAW_MAX_DEG = _env_float("GAZE_THINK_YAW_MAX_DEG", 14.0, min_value=0.0, max_value=40.0)
GAZE_INTERNALIZE_PITCH_MIN_DEG = _env_float("GAZE_INTERNALIZE_PITCH_MIN_DEG", 8.0, min_value=0.0, max_value=20.0)
GAZE_INTERNALIZE_PITCH_MAX_DEG = _env_float("GAZE_INTERNALIZE_PITCH_MAX_DEG", 15.0, min_value=0.0, max_value=20.0)
GAZE_ON_TARGET_JITTER_DEG = _env_float("GAZE_ON_TARGET_JITTER_DEG", 2.5, min_value=0.0, max_value=10.0)
# Engagement (POLE / head-height) by phase (mm).
GAZE_POLE_REST_MM = _env_float("GAZE_POLE_REST_MM", 20.0, min_value=0.0, max_value=60.0)
GAZE_POLE_LEAN_IN_MM = _env_float("GAZE_POLE_LEAN_IN_MM", 45.0, min_value=0.0, max_value=60.0)
GAZE_POLE_SETTLE_MM = _env_float("GAZE_POLE_SETTLE_MM", 5.0, min_value=0.0, max_value=60.0)
# Multi-person.
GAZE_INCLUDE_SWEEP_PROB = _env_float("GAZE_INCLUDE_SWEEP_PROB", 0.20, min_value=0.0, max_value=1.0)
GAZE_ORIENT_GLANCE_SECS = _env_float("GAZE_ORIENT_GLANCE_SECS", 0.6, min_value=0.0, max_value=3.0)
# #27 — let the gaze engine run while SPEAKING so the ~50%-on-target duty AND the
# multi-person include-sweep actually fire (the adapter used to fully suppress speech).
# ONLY include-sweeps (a bounded glance to a listener) actuate during speech; off-target
# aversions stay suppressed. Kill switch — set False to restore the speech-suppressed
# behaviour. NOT yet hardware-validated; flip off if the head hunts during speech.
GAZE_SPEAKING_SWEEP_ENABLED = _env_bool("GAZE_SPEAKING_SWEEP_ENABLED", True)
# Max yaw (deg) of an include-sweep glance toward an off-centre listener — keeps it a
# glance, not a full head-turn, during speech. +deg = right of frame.
GAZE_LISTENER_MAX_BEARING_DEG = _env_float(
    "GAZE_LISTENER_MAX_BEARING_DEG", 22.0, min_value=2.0, max_value=45.0,
)
# Conversation-activity threshold.
GAZE_CLOSE_AFTER_IDLE_SECS = _env_float("GAZE_CLOSE_AFTER_IDLE_SECS", 12.0, min_value=1.0, max_value=120.0)
# Velocities for the offline sim / open-loop RealHead (deg/s, mm/s). The live closed
# loop uses the existing FACE_TRACKING_*_MAX_STEP_QUS slew caps instead.
GAZE_SACCADE_VEL_DPS = _env_float("GAZE_SACCADE_VEL_DPS", 320.0, min_value=10.0, max_value=2000.0)
GAZE_SMOOTH_VEL_DPS = _env_float("GAZE_SMOOTH_VEL_DPS", 90.0, min_value=5.0, max_value=1000.0)
GAZE_POLE_VEL_MMS = _env_float("GAZE_POLE_VEL_MMS", 25.0, min_value=1.0, max_value=500.0)
# DOF limits used by the deg/mm <-> qus mapping (mechanical stops live in SERVO_CHANNELS).
GAZE_YAW_LIMIT_DEG = _env_float("GAZE_YAW_LIMIT_DEG", 70.0, min_value=5.0, max_value=120.0)
GAZE_PITCH_UP_LIMIT_DEG = _env_float("GAZE_PITCH_UP_LIMIT_DEG", 25.0, min_value=1.0, max_value=60.0)
GAZE_PITCH_DOWN_LIMIT_DEG = _env_float("GAZE_PITCH_DOWN_LIMIT_DEG", 20.0, min_value=1.0, max_value=60.0)
GAZE_POLE_MIN_MM = _env_float("GAZE_POLE_MIN_MM", 0.0, min_value=0.0, max_value=60.0)
GAZE_POLE_MAX_MM = _env_float("GAZE_POLE_MAX_MM", 60.0, min_value=1.0, max_value=200.0)
GAZE_POLE_GAIN_QUS_PER_MM = _env_float("GAZE_POLE_GAIN_QUS_PER_MM", 22.0, min_value=1.0, max_value=100.0)
# Live actuation: the aversion is a velocity-RAMPED move (soft ease-in/out), not a
# constant-speed snap. The *_MAX_STEP_QUS values are the per-tick velocity CAP (top
# speed); GAZE_AVERSION_RAMP_TICKS is how many ticks it takes to ramp from rest to
# that cap (the acceleration). Kept deliberately gentle — a brief look-away should
# drift, not jerk. POLE (head height) is slowest of all.
GAZE_AVERSION_NECK_MAX_STEP_QUS = _env_int("GAZE_AVERSION_NECK_MAX_STEP_QUS", 240, min_value=20, max_value=4000)
GAZE_AVERSION_TILT_MAX_STEP_QUS = _env_int("GAZE_AVERSION_TILT_MAX_STEP_QUS", 130, min_value=10, max_value=1600)
GAZE_AVERSION_LIFT_MAX_STEP_QUS = _env_int("GAZE_AVERSION_LIFT_MAX_STEP_QUS", 70, min_value=5, max_value=1700)
GAZE_AVERSION_RAMP_TICKS = _env_float("GAZE_AVERSION_RAMP_TICKS", 6.0, min_value=1.0, max_value=40.0)
GAZE_AVERSION_SERVO_SPEED = _env_int("GAZE_AVERSION_SERVO_SPEED", 90, min_value=1, max_value=255)
GAZE_AVERSION_SERVO_ACCELERATION = _env_int("GAZE_AVERSION_SERVO_ACCELERATION", 9, min_value=0, max_value=255)

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
# How long a speaker's already-locked face may flicker out before the gaze
# SEARCH sweep launches. 0.45s was far too twitchy: a two-frame detector dropout
# right after a speech turn launched a full hold-down room sweep that fought
# face-tracking recapture — the head-lift swung nearly full travel twice in ~6s
# (field 2026-07-31 21:41, read as "violent head movement"). A person who
# genuinely walks away stays gone past this window and still gets searched for;
# ordinary (non-speaker) tracking already holds a lost face 8s through flicker.
SPEAKER_GAZE_LOST_SEARCH_AFTER_SECS = _env_float(
    "SPEAKER_GAZE_LOST_SEARCH_AFTER_SECS",
    3.0,
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

# Master switch for body-pose/gesture detection (vision/pose.py). Off → no gesture
# cues (incl. wave-back) and no GUI skeleton overlay; face-based proxemics still work.
# Kill switch if the pose model is missing or pose detection misbehaves on a build.
POSE_DETECTION_ENABLED = _env_bool("POSE_DETECTION_ENABLED", True)

# Pose runs in its OWN background loop (vision.pose.start(), like face_expression) rather
# than off the ~1 Hz consciousness tick, so the GUI skeleton overlay and wave-back stay
# live. This is that loop's sampling period; ~5 Hz keeps the wireframe smooth without much
# CPU (pose "lite" is cheap). Raise it to spend less CPU at the cost of a choppier overlay.
POSE_ANALYSIS_INTERVAL_SECS = 0.2

# Wave back when a visible person waves at Rex. The pose pipeline classifies a "waving"
# gesture onto world_state.people (a hand raised out to the side — see vision/pose.py);
# this fires Rex's wave-back animation + one short warm line, debounced so it reacts once
# per wave. Requires the MediaPipe Pose Landmarker model + POSE_DETECTION_ENABLED; if the
# model is missing or pose is disabled, wave-back degrades to nothing.
WAVE_BACK_ENABLED = _env_bool("WAVE_BACK_ENABLED", True)
# Don't wave back at the SAME person again for this long. This debounces a single sustained
# wave into one wave-back, but if it's too long a deliberate second wave a few seconds later
# feels ignored — so it's tuned just above one wave-back's duration (~4s of gesture+greeting)
# rather than the old 25s, which made re-waving feel dead.
WAVE_BACK_PER_PERSON_COOLDOWN_SECS = _env_float(
    "WAVE_BACK_PER_PERSON_COOLDOWN_SECS", 6.0, min_value=0.0, max_value=3600.0,
)
# Global minimum gap between any two wave-backs (so a crowd of wavers doesn't spam).
WAVE_BACK_MIN_GAP_SECS = _env_float(
    "WAVE_BACK_MIN_GAP_SECS", 4.0, min_value=0.0, max_value=600.0,
)
# Stability gate: how many CONSECUTIVE consciousness ticks (~1 Hz) a person must read
# gesture=='waving' before a wave-back is trusted. A held human wave spans 2-3 ticks; a
# flickering non-human blob (a pillow MediaPipe momentarily skeletonizes — live-logged
# 2026-06-26: pose appeared/vanished every ~1s cycling random gestures) virtually never
# reads 'waving' twice in a row, so it's rejected. Set 1 to restore old single-frame
# behavior (the source of the phantom waves).
WAVE_BACK_CONFIRM_FRAMES = _env_int("WAVE_BACK_CONFIRM_FRAMES", 2, min_value=1, max_value=10)
# Suppress wave-back ONLY when the waver's face fills nearly the whole frame HEIGHT —
# i.e. pressed right up against the lens, where a "wave" is a near-camera artifact and a
# wave-back makes no sense. Measured as face-box HEIGHT / frame height, NOT the width-based
# proxemics zone: on a wide 16:9 webcam a close-up face is TALL not wide, so width
# under-reads closeness. History: this shipped at 0.30, which turned out to reject the
# PRIMARY use case — someone seated at a desk webcam waves with their face at ~40-50% of
# frame height (live-logged 2026-07-08: a genuine wave at 44% was ignored, no wave-back).
# The real anti-phantom protection is elsewhere (plausible-pose shoulder-girdle filter +
# WAVE_BACK_CONFIRM_FRAMES streak), so this guard is now just a backstop for the
# face-on-the-lens degenerate case: 0.72 ≈ "face taller than ~three-quarters of the
# screen". 0 disables. Tune off the "face_height=" value in the "wave detected" log line.
WAVE_BACK_MAX_FACE_FRACTION = _env_float(
    "WAVE_BACK_MAX_FACE_FRACTION", 0.72, min_value=0.0, max_value=1.0,
)
# Wave-back arm gesture: how many times the wrist (the "hand" servo) sweeps between BOTH of
# its travel limits when Rex waves back (one sweep = to one limit and back). The elbow only
# raises the arm; the wrist does the waving. See sequences/animations.wave_back_gesture.
WAVE_BACK_WRIST_SWEEPS = _env_int("WAVE_BACK_WRIST_SWEEPS", 4, min_value=1, max_value=8)
# The wave drives the wrist channel with DIRECT Maestro targets at a fast speed (the global
# SERVO_DEFAULT_SPEED=40 is far too slow — a full wrist sweep takes ~2s at that rate, so the
# rapid back-and-forth never completes). The channel is set to this speed/accel for the wave,
# then restored. HALF_PERIOD is the pause at each extreme (≈ the time to traverse).
#   SPEED 0  = auto: pick a speed that traverses the wrist's full travel in HALF_PERIOD.
#   SPEED >0 = use that Maestro speed value verbatim (units of 0.25µs / 10ms).
#   ACCEL 0  = unlimited (snappiest). Raise SPEED/ACCEL if too slow; lower if too violent.
# HALF_PERIOD is the time for one swing (to one limit); a bigger value = a slower, gentler
# wave (auto SPEED scales down with it). 0.32s reads as a relaxed wave rather than a frantic
# flap — lower it toward ~0.2 for a snappier wave, raise it for a lazier one.
WAVE_BACK_WRIST_SPEED = _env_int("WAVE_BACK_WRIST_SPEED", 0, min_value=0, max_value=16383)
WAVE_BACK_WRIST_ACCEL = _env_int("WAVE_BACK_WRIST_ACCEL", 0, min_value=0, max_value=255)
WAVE_BACK_WRIST_HALF_PERIOD_SECS = _env_float(
    "WAVE_BACK_WRIST_HALF_PERIOD_SECS", 0.32, min_value=0.05, max_value=2.0,
)

# Mirror the user's wave SPEED: measure how fast their wrist is sweeping (normalized-x units
# per second, from vision.pose.recent_wave_speed) and pick Rex's wave half-period to match —
# slow wave → slow wave-back, fast → fast. The measured speed is mapped linearly from
# [SLOW..FAST] user speed onto [SLOW..FAST] half-period (clamped), so it never gets faster
# than the FAST (non-violent) cap or slower than the SLOW floor. Falls back to the fixed
# WAVE_BACK_WRIST_HALF_PERIOD_SECS above when off or when no speed could be measured.
# Tune the user-speed thresholds from the "[wave] mirror" log line on the robot.
WAVE_SPEED_MIRROR_ENABLED = _env_bool("WAVE_SPEED_MIRROR_ENABLED", True)
WAVE_SPEED_WINDOW_SECS = _env_float("WAVE_SPEED_WINDOW_SECS", 1.2, min_value=0.3, max_value=5.0)
WAVE_SPEED_MIRROR_SLOW = _env_float("WAVE_SPEED_MIRROR_SLOW", 0.25, min_value=0.01, max_value=20.0)
WAVE_SPEED_MIRROR_FAST = _env_float("WAVE_SPEED_MIRROR_FAST", 1.20, min_value=0.02, max_value=40.0)
WAVE_BACK_WRIST_HALF_PERIOD_SLOW_SECS = _env_float(
    "WAVE_BACK_WRIST_HALF_PERIOD_SLOW_SECS", 0.48, min_value=0.05, max_value=2.0,
)
WAVE_BACK_WRIST_HALF_PERIOD_FAST_SECS = _env_float(
    "WAVE_BACK_WRIST_HALF_PERIOD_FAST_SECS", 0.18, min_value=0.05, max_value=2.0,
)
# How long a detected wave stays "pending" while Rex is busy (mid-turn / speaking) before
# it's answered. A wave is a brief gesture, so it's latched and voiced as soon as Rex is
# free within this window; longer than this and a stale wave is dropped instead of getting
# a late, out-of-context "Hello!". Covers a typical reply without feeling delayed.
WAVE_BACK_PENDING_TTL_SECS = _env_float(
    "WAVE_BACK_PENDING_TTL_SECS", 8.0, min_value=0.0, max_value=120.0,
)
# Short, warm spoken greetings Rex says when he waves back (canned for immediacy — a
# wave-back shouldn't wait on an LLM call). WAVE_BACK_LINES is used when Rex knows the
# waver's name ("{name}" is filled with their first name); WAVE_BACK_LINES_NO_NAME is
# used for an unknown/unnamed waver. One is picked at random per wave.
WAVE_BACK_LINES = [
    "Hello, {name}!",
    "Hi there, {name}!",
    "Hey {name}, what's up?",
    "Greetings, {name}.",
    "Hello there, {name}!",
    "{name}! Good to see you.",
]
WAVE_BACK_LINES_NO_NAME = [
    "Hello!",
    "Hello there!",
    "Hi there!",
    "What's up?",
    "Greetings.",
    "Greetings, lifeform.",
]

# Repeat-wave comedy bit: consecutive waves from the SAME person escalate instead of just
# repeating the greeting. 1st = normal greeting + wave; 2nd = silent wave-back (no line);
# 3rd = a joke + wave; 4th = a give-up joke (no wave); 5th+ = he ignores you until you stop.
# The level resets after WAVE_BACK_ESCALATION_RESET_SECS with no wave (so a wave much later
# starts fresh at the greeting). Set WAVE_BACK_ESCALATION_ENABLED False to always greet.
WAVE_BACK_ESCALATION_ENABLED = _env_bool("WAVE_BACK_ESCALATION_ENABLED", True)
WAVE_BACK_ESCALATION_RESET_SECS = _env_float(
    "WAVE_BACK_ESCALATION_RESET_SECS", 30.0, min_value=2.0, max_value=600.0,
)
# 3rd-wave lines — Rex notices you keep waving (deadpan, no name needed).
WAVE_BACK_JOKE_LINES = [
    "Still waving? My arm and I are flattered.",
    "Yes, I have arms. We've covered this.",
    "Three waves in. We're basically pen pals now.",
    "I see you. I saw you the last two times, too.",
    "My wave servo is logging overtime.",
]
# 4th-wave lines — Rex taps out (deadpan, also a joke; no wave with these).
WAVE_BACK_GIVEUP_LINES = [
    "Okay, that's the last one — my actuators just unionized.",
    "I'm tapping out. Wave at the chest lights from here on.",
    "Done waving. I do have other features, you know.",
    "That's a wrap on the arm. Further waves go under cardio.",
    "I'm retiring this wave. It had a good run.",
]

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

# ArcFace (InsightFace) equivalents — 512-dim L2-NORMALIZED embeddings, so Euclidean
# distance maps to cosine similarity as d = sqrt(2 - 2*cos). Genuine matches land
# roughly d 0.8-1.05 (cos 0.45-0.68); impostors d 1.25-1.41 (cos ~0.0-0.2). 1.10
# corresponds to cos ~0.40, the standard ArcFace acceptance band. The matcher in
# memory/people.find_by_face picks dlib vs ArcFace thresholds by the QUERY dimension
# (128 vs 512), so stale dlib rows and new ArcFace rows can coexist in biometrics.
FACE_RECOGNITION_DISTANCE_THRESHOLD_ARCFACE = 1.10
FACE_RECOGNITION_MARGIN_ARCFACE = 0.08
# Temporal hysteresis: when a single visible face is already bound to one known person,
# require this many consecutive recognition ticks agreeing on a DIFFERENT person before
# the world-state/overlay identity switches. Damps known<->known flicker. 1 disables.
FACE_IDENTITY_SWITCH_CONFIRM_FRAMES = 2

# An UNKNOWN face must persist this many consecutive recognition ticks before Rex treats
# it as a real person (and lets it arm the "who's the mystery guest?" agenda). Filters
# transient phantom faces — clutter, a shape on the wall, a glance at a messy shelf — that
# flicker for a frame or two, while a genuine newcomer (who stays put) clears it in ~1s.
# 1 disables the gate (old behavior); raise if phantom guests still slip through.
FACE_UNKNOWN_CONFIRM_FRAMES = _env_int(
    "FACE_UNKNOWN_CONFIRM_FRAMES", 3, min_value=1, max_value=30,
)

# ── Voice embedder backend ────────────────────────────────────────────────────
# "ecapa": ECAPA-TDNN (SpeechBrain, 192-dim) — far wider genuine/impostor
#   separation than Resemblyzer (whose weak separation was the root cause of the
#   recurring ambiguity incidents: JT's single print sat 0.45-0.49 cosine from
#   ALL of Bret's prints). ~20ms per embedding on CPU. Models in ECAPA_MODEL_DIR
#   (downloaded by setup_assets.py, ~80MB).
# "resemblyzer": legacy 256-dim embedder. Also the automatic runtime fallback if
#   the ECAPA model fails to load.
# The two embeddings are INCOMPATIBLE (192 vs 256 dim): stored voice prints and
# voice signatures from one embedder are skipped by the other — people must
# RE-ENROLL their voice after switching (tools/test_voice_id.py --enroll).
# SCORE SCALE: all SPEAKER_ID_* thresholds below stay on the Resemblyzer-
# calibrated scale; ECAPA cosines are mapped onto it by audio/voice_score.py
# (+VOICE_SCORE_OFFSET_ECAPA, clamped). A constant offset preserves score GAPS,
# so every margin knob keeps its meaning.
VOICE_EMBEDDER = (os.getenv("VOICE_EMBEDDER", "").strip().lower() or "ecapa")
ECAPA_MODEL_DIR = "assets/models/ecapa"
# ECAPA genuine matches run ~0.30-0.75 raw (vs Resemblyzer 0.45-0.93); impostors
# ~0.0-0.2 (vs 0.3-0.5). +0.25 lines the bands up with the thresholds below:
# impostor -> 0.25-0.45 (under the 0.50 accept), genuine short utterance ->
# 0.55-0.75, solid match -> 0.75-0.99. Re-run tools/test_voice_id.py --calibrate
# after re-enrolling to verify your own band.
VOICE_SCORE_OFFSET_ECAPA = 0.25

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

# Thin-challenger relief (field log 2026-07-06-19-23: Bret, 6 curated prints, scored
# 0.558 on a short greeting; JT's SINGLE unverified print trailed at 0.502 — margin
# 0.056 < 0.07 challenged the OWNER as a mystery voice). A 1-clip centroid is
# high-variance and shouldn't carry full veto power over a mature multi-print match:
# when the runner-up has <= THIN_PRINT_MAX_ROWS prints, the top has MORE, and the top
# score is at least THIN_RUNNER_MIN_TOP_SCORE (above the measured cross-match band —
# JT's live voice hit Bret's centroid at only 0.529), the required margin is scaled
# by THIN_RUNNER_MARGIN_FACTOR. The reverse direction (thin-print person on top)
# keeps the full margin, so the who's-that challenge still fires for the newcomer
# until their print matures.
SPEAKER_ID_THIN_PRINT_MAX_ROWS = 1
SPEAKER_ID_THIN_RUNNER_MARGIN_FACTOR = 0.5
SPEAKER_ID_THIN_RUNNER_MIN_TOP_SCORE = 0.55

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
# Floor for the "live speaker is confidently the introducer — never enroll this
# onto the newcomer" guard in _handle_intro_voice_capture. Deliberately pinned at
# the historical 0.70 and DECOUPLED from SPEAKER_ID_CONFIDENT_THRESHOLD: raising
# that global to 0.75 silently reopened the [0.70, 0.75) band and re-enabled the
# introducer-voice-onto-newcomer poisoning (the "Leaf" bug shape; see also the
# Brat/Exudica twin chaos, logs 2026-07-23-19-50-57).
INTRO_VOICE_INTRODUCER_GUARD_FLOOR = 0.70

# Off-screen "who was that?" claim verification: when the answer names an EXISTING
# person who already has voice prints, the held clip must score at least this
# against their prints to be ENROLLED onto them (conversation attribution is not
# affected). A claimed name is testimony, not biometrics — field 2026-07-23 20:12:
# a guest joked "obviously me, Bret" and her clip (0.516 vs Bret) was enrolled
# onto Bret's record. Genuine same-person clips score well above this.
OFFSCREEN_IDENTIFY_CLAIM_VERIFY_FLOOR = 0.55

# VAD (Silero) — probability threshold above which speech is considered detected
VAD_THRESHOLD = 0.5
# Robot (hardware-AEC) override, applied ONLY when audio/hardware_aec.is_active().
# History: 71fd4bd lowered the global threshold to 0.4 because soft-onset phonemes
# ("wh" in "what's") don't cross 0.5 in the first chunks, so the VAD fires late and
# the LEADING words are clipped. 71def95 then reverted the whole change-set because
# its capture-floor grace part self-transcribed on the no-AEC dev Mac — the revert
# took the robot's threshold/preroll fix with it, and the front-clipping came back
# on the robot (field 2026-07-23: "What do you think of your new motor system" heard
# as "new motor systems", VAD opened ~2s late at far-field). This re-applies ONLY
# the two safe levers, gated to the ReSpeaker robot; the dangerous lever (capture
# floor grace) stays at 0.12 everywhere. Dev Mac behavior is unchanged.
VAD_THRESHOLD_AEC = 0.4

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

# ── Audio playback QoS ────────────────────────────────────────────────────────
# Playback runs through a Python-level PortAudio callback that must grab the GIL for
# EVERY audio block. Heavy work elsewhere (model preloads at boot: Whisper, speaker-ID,
# Ollama, YOLO — long C calls that hold the GIL) can starve that callback past its
# deadline → buffer underrun → the mid-sentence stutter in the startup filler lines.
# The fix is depth, not priority (the callback thread is already realtime; the GIL is
# the bottleneck):
#   BLOCKSIZE 4096 (~93ms @44.1k, was 2048/~46ms) + LATENCY "high" asks CoreAudio for a
#   deep host buffer, so playback shrugs off GIL stalls of a few hundred ms. Costs only
#   a little extra time-to-first-sound.
AUDIO_PLAYBACK_BLOCKSIZE = _env_int("AUDIO_PLAYBACK_BLOCKSIZE", 4096, min_value=256, max_value=32768)
AUDIO_PLAYBACK_LATENCY = os.getenv("AUDIO_PLAYBACK_LATENCY", "high").strip() or "high"
# BOOT window only (field 2026-08-02: the filler line stuttered again — on macOS
# the symbolic 'high' preset is only a few tens of ms of host buffer, far less
# than a model-load GIL burst). While main.py's preload QoS window is armed,
# playback asks CoreAudio for an EXPLICIT deep buffer: ~1s of latency + a bigger
# callback block. Costs up to ~1s extra time-to-first-sound, which is invisible
# behind the boot theatrics; disarmed before conversation starts.
AUDIO_PLAYBACK_BOOT_LATENCY_SECS = _env_float(
    "AUDIO_PLAYBACK_BOOT_LATENCY_SECS", 1.0, min_value=0.1, max_value=4.0
)
AUDIO_PLAYBACK_BOOT_BLOCKSIZE = _env_int(
    "AUDIO_PLAYBACK_BOOT_BLOCKSIZE", 8192, min_value=256, max_value=65536
)
# During the boot preloads, additionally: (a) shrink the GIL switch interval so pure-
# Python import storms yield to the audio callback sooner, and (b) take a short breath
# between preload steps so the audio buffer refills after each load's GIL burst. The
# filler line keeps playing WHILE models load (that's the point) — these just keep it
# smooth. Breaths only fire while startup audio is actually playing.
STARTUP_PRELOAD_AUDIO_QOS_ENABLED = _env_bool("STARTUP_PRELOAD_AUDIO_QOS_ENABLED", True)
STARTUP_PRELOAD_BREATH_SECS = _env_float("STARTUP_PRELOAD_BREATH_SECS", 0.25, min_value=0.0, max_value=2.0)
STARTUP_PRELOAD_GIL_SWITCH_INTERVAL = _env_float(
    "STARTUP_PRELOAD_GIL_SWITCH_INTERVAL", 0.002, min_value=0.0005, max_value=0.05
)

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
# 0.12 (was 0.5): this general default is used by NON-reply playback (proactive
# lines, greetings, startup); replies already use the short 0.12 reply tail
# (POST_QUESTION/SPEECH_PLAYBACK_SUPPRESSION_SECS) via _reply_playback_tail_secs.
# At 0.5, a user who answered a greeting or a lull-breaker the instant Rex stopped
# had their first words attenuated to 5% (AEC_SUPPRESSION_FACTOR) and dropped —
# front-clipping. The buffer is flushed at playback-stop, so the tail only needs to
# cover the speaker's brief acoustic decay; 0.12 is the value the reply path already
# proves safe on this no-AEC dev Mac. (Robot hardware AEC uses the 0.05 _AEC tail.)
POST_PLAYBACK_SUPPRESSION_SECS = 0.12

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
# politics" → "to Sacramento than politics"). Now set EQUAL to the question tail
# (POST_QUESTION_PLAYBACK_SUPPRESSION_SECS): a reply landing right after a
# statement was still losing its opening words at 0.25s while questions captured
# the full reply at 0.12s (live 2026-06-18: "I can't eat until tonight" clipped to
# a trailing "can't eat until tonight" fragment). Statements get the identical
# responsive treatment questions already get. The capture floor + preroll grace
# (below) recover the front from the raw (un-attenuated) buffer; words spoken
# *over* Rex still need hardware AEC.
POST_SPEECH_PLAYBACK_SUPPRESSION_SECS = 0.12  # match POST_QUESTION_PLAYBACK_SUPPRESSION_SECS
POST_SPEECH_FLUSH_AUDIO_BUFFER = False

# ElevenLabs clips can contain trailing near-silence. Humans naturally answer
# when Rex sounds done, but the audio device may still be playing that padding,
# keeping mic suppression active. Trim only the tail, leaving a tiny cushion so
# words do not click or feel clipped.
TTS_TRIM_TRAILING_SILENCE_ENABLED = True
TTS_TRIM_TRAILING_SILENCE_THRESHOLD = 0.003
TTS_TRIM_TRAILING_SILENCE_WINDOW_MS = 20
# 40ms shaved word-final decays (breathy endings sit under the RMS threshold and
# read as "silence") — 120ms keeps the natural tail while still releasing the mic
# suppression promptly (owner report 2026-07-06: "TTS cut off at the end a bit").
TTS_TRIM_TRAILING_SILENCE_PADDING_MS = 120

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
# Extra quiet margin after Rex's own output before scene analysis resumes, ON TOP
# of the analysis window — ensures the sampled window can't straddle his speech
# tail or the suppression step (false "laughter" → unearned take-a-bow,
# live-logged 2026-07-06-22-28). Total post-speech blindness = WINDOW + this.
SCENE_POST_OUTPUT_GUARD_SECS = 1.5

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

# Frequency cooldown for SELF-DIRECTED comedic body beats (eye-roll, double-take,
# mic-drop, spit-take, etc.) so Rex doesn't mug nonstop on his own. Only beats
# fired with spontaneous=True are gated; explicit "do a mic drop" requests and
# deterministic event/mood/gamepad beats are never throttled. Seconds; 0 disables.
COMEDY_BEAT_MIN_GAP_SECS = 6.0

# When a comedic line is performed (joke / roast / free-bit), land its body beat in
# the SILENCE after the line ("line lands -> beat of silence -> button") instead of
# firing it over the front of the line. Kill switch: set False to restore the
# upfront beat. The landing is guarded against barge-in (skips if a turn has begun).
PERFORMANCE_POST_LINE_BEAT_ENABLED = True

# Pause after genuine surprise event before Rex responds (milliseconds)
SURPRISE_PAUSE_MS_MIN = 200
SURPRISE_PAUSE_MS_MAX = 500
# On the STREAMING reply path, how long to briefly wait for the surprise classifier to
# resolve BEFORE the first sentence so the surprise pre-beat can land (the fast first
# token otherwise wins the race). Bounded — at most this is added to time-to-first-word,
# and only when the classifier hasn't already finished.
SURPRISE_STREAM_JOIN_SECS = 0.25

# Self-emotion classifier: read the emotional tone of REX'S OWN reply (excited /
# happy / curious / neutral) so the body actually expresses it — eye colour, speech
# servo motion, expressive voice on the reply, plus a short body-mood afterglow
# (posture / breathing / idle gesture) in the lull that follows. Without it the common
# reply ships emotion="neutral" and the whole expressive stack stays inert. Runs on the
# LOCAL qwen sidecar (cheap classifier; per project policy the cloud model is reserved
# for in-character text) with a keyword fallback, so it never adds cloud latency/cost.
# Surprise and empathy delivery overrides still win. Kill switch: set False.
SELF_EMOTION_CLASSIFY_ENABLED = True
# Body-mood afterglow intensity when the reply classifies non-neutral (decays over the
# normal BODY_MOOD_DEFAULT_TTL_SECS so posture relaxes back to neutral after the reply).
SELF_EMOTION_BODY_MOOD_INTENSITY = 0.6
# Hard cap on the sidecar classify call so it can't stall a turn.
SELF_EMOTION_CLASSIFY_TIMEOUT_SECS = 1.2

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

# When a "misheard" repair actually carries re-stated real content (a bare "I said X" /
# "I meant X" with no contrast), respond to X as the real turn instead of echoing a
# recalibration line back ("We'll get there — recalibrating. <your words>."). Kill switch.
REPAIR_RESTATEMENT_AS_REPLY_ENABLED = True

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

# ── Comedic delivery profiles ────────────────────────────────────────────────
# A landed roast and a condolence currently reach ElevenLabs with IDENTICAL
# voice settings, so every joke under-lands. These profiles give each comedic
# STANCE (intelligence/comedy_modes.select_mode) its own timbre, mirroring the
# empathy delivery layer (intelligence/empathy._MODE_VOICE_SETTINGS) but keyed
# on the turn's comedy mode instead of a per-person empathy cache.
#
# Precedence: empathy/grief delivery shaping OUTRANKS comedy. A comedy profile
# is applied ONLY on a neutral-empathy turn (when empathy left the voice alone),
# and the "straight" care mode carries no profile — so sensitive turns are never
# comedically shaped. Comedy also rides under TTS_EXPRESSIVE_VOICE_ENABLED: with
# expressive voice off (flat clone + pre-existing cache) comedy stays off too.
#
# Each distinct {stability,style,...} combo is a fresh TTS cache key, so start
# with a SMALL set (deadpan + smug) to limit cache regen; add mischief / dj_hype
# later by adding a profile here and a comedy-mode → profile row below.
COMEDY_DELIVERY_PROFILES_ENABLED = True
COMEDY_DELIVERY_PROFILES = {
    # deadpan — flat, dry, deliberate; the even monotone a dry button lands on.
    # Higher stability than baseline (less inflection), much lower style (no
    # flourish), a hair slower so the button is delivered, not tossed off.
    "deadpan": {"stability": 0.66, "style": 0.20, "similarity_boost": 0.82, "speed": 0.97},
    # smug — cocky, self-satisfied swagger after a jab; more expressive style
    # than baseline but controlled, savored a touch slower.
    "smug":    {"stability": 0.46, "style": 0.64, "similarity_boost": 0.82, "speed": 0.96},
    # theatrical — the over-the-top movie-trailer narrator: very dynamic (low
    # stability = big range), high style for drama, a touch slow for gravitas.
    "theatrical": {"stability": 0.30, "style": 0.78, "similarity_boost": 0.82, "speed": 0.98},
}
# comedy_modes.select_mode key → delivery profile name (above). Modes left out
# (straight / fake_system_error / dj_flair) get NO comedy voice shaping for now.
COMEDY_MODE_DELIVERY_PROFILE = {
    "dry_ack":              "deadpan",
    "self_own":             "deadpan",
    "callback":             "deadpan",
    "callback_banked":      "deadpan",
    "friendly_roast":       "smug",
    # Comedic personas — each gets its own timbre.
    "smug_superiority":     "smug",
    "appliance_conspiracy": "deadpan",
    "dramatic_narrator":    "theatrical",
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

# Own-echo (reference-text) rejection. With hardware AEC the mic stays live while
# Rex plays; his ~-17 dB residual can still cross the VAD and Whisper transcribes
# him VERBATIM (field 2026-07-23 19:56: the "Something's in my way" announce came
# back as unknown_voice_2 saying the same words and got a full LLM reply). Every
# spoken line is remembered for the window below; a fresh transcript that matches
# one at >= the similarity floor is dropped as self-echo. MIN_WORDS keeps 1-2 word
# overlaps ("yeah", "okay") attributable to the human. An armed impersonation
# capture slot is exempt — there the human is SUPPOSED to recite Rex's words.
OWN_ECHO_REJECT_ENABLED = True
OWN_ECHO_WINDOW_SECS = 12.0
OWN_ECHO_MIN_WORDS = 3
OWN_ECHO_SIMILARITY = 0.85
# Looser floor for the capture seam right after a line was spoken. The AEC
# residual distorts hard enough that Whisper garbles the echo into homophones
# instead of transcribing it verbatim (field 2026-08-01 17:00: the ready line
# "my circuits are hot, my takes are hotter" came back 6 s later as
# unknown_voice_1 saying "my tickets are hot, my tickets are hotter" — 0.70
# similarity, under the 0.85 floor — and Rex greeted his own echo as a mystery
# voice). Within SEAM_SECS of the line being spoken, a transcript only needs to
# clear SEAM_SIMILARITY. Time-gating keeps the false-positive risk small: a
# 65%-similar transcript arriving seconds after Rex said the line is almost
# surely him.
OWN_ECHO_SEAM_SECS = 8.0
OWN_ECHO_SEAM_SIMILARITY = 0.65

# Seconds of sustained silence after speech before the segment is processed.
# This is the largest "I stopped talking, why is Rex waiting?" knob -- lowering
# it shaves dead time off the start of every turn. Tradeoff: too low and a
# person who pauses mid-sentence can get cut off. 0.85 (was 0.6): the owner has
# more thought coming after a brief pause, so give a longer grace before Rex
# decides the turn is done (small added latency, in exchange for not cutting him
# off mid-thought). Raise further if it still ends turns too early.
SILENCE_TIMEOUT_SECS = 0.85

# Eager endpointing for explicit motion commands. At MOTION_EAGER_ENDPOINT_SILENCE_SECS
# of silence (well before SILENCE_TIMEOUT_SECS) a background probe transcribes the
# segment-so-far; if it decodes to a COMPLETE drive command ("turn left", "back up two
# feet", bare "stop" while moving) the turn ends immediately and the probe transcript is
# reused, so the wheels get the command ~0.6-0.9s sooner. Anything else — normal chat,
# an utterance still in progress, a trailing "and"/"then" that promises another clause —
# leaves the normal 0.85s hold untouched, and speech resuming mid-probe discards it.
# Worst case (person continues after a real pause, e.g. "turn left ... then back up") the
# split halves both still execute via the motion-continuation context. Requires the drive
# base; MOTION_EAGER_ENDPOINT_REQUIRE_AEC keeps it robot-only (hardware AEC present) so
# dev-Mac sessions keep stock endpointing.
MOTION_EAGER_ENDPOINT_ENABLED = _env_bool("MOTION_EAGER_ENDPOINT_ENABLED", True)
MOTION_EAGER_ENDPOINT_SILENCE_SECS = 0.35
MOTION_EAGER_ENDPOINT_REQUIRE_AEC = True

# Minimum seconds of accumulated audio before silence can end a recording.
# Prevents single-word transcriptions when the person is still talking.
MIN_SPEECH_DURATION_SECS = 0.45

# Include audio before the first VAD-positive chunk so soft starts are not
# clipped. Question answers get more pre-roll because people often begin while
# Rex's last syllable or room echo is still fading.
SPEECH_PREROLL_SECS = 0.45
# Robot (hardware-AEC) override — see VAD_THRESHOLD_AEC for the full history. At
# far-field the VAD fires well after true speech onset, so 0.45s of pre-roll still
# loses the leading words; 1.0 covers the detection delay (leading silence is free
# for Whisper). SAFE at any length: _speech_capture_secs clamps pre-roll to the
# post-TTS capture floor, so it can never reach back into Rex's own tail — the
# floor grace (0.12s) is what guards that seam, and it is not touched here.
# 1.5 (was 1.0): even with the AEC-gated threshold + 1.0 s reach-back, a soft opener
# was still lost at far field — field 2026-07-24 19:59, "Feel free to explore the room"
# transcribed as "Explore the room." At the measured ~13-15 dB SNR the VAD can open
# more than a second after true onset. Leading silence is free for Whisper and
# _speech_capture_secs still clamps to the post-TTS capture floor, so this cannot
# reach into Rex's own tail.
SPEECH_PREROLL_SECS_AEC = 1.5
POST_QUESTION_SPEECH_PREROLL_SECS = 2.0

# How far a NON-question reply's capture may reach back past the post-TTS handoff
# into the raw (un-attenuated) rolling buffer. 0.0 meant a reply that began as Rex
# finished a statement had its front clipped (the buffer holds it, but the capture
# floor refused to reach back). 0.12 mirrors the question grace — enough to recover
# the front, small enough that it rarely reaches Rex's final word (≥0.25s does).
# NOTE: reaching further back is ONLY safe with hardware AEC (see the aec_on branch
# in interaction.py that lifts this to POST_TTS_CAPTURE_PREROLL_GRACE_SECS_AEC). On a
# no-AEC setup (dev Mac, whisper suppressed while Rex speaks), a larger value pulls
# Rex's own tail at full volume → self-transcription / dropped segment → the user
# gets cut off. Keep this SMALL when there is no AEC.
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
# When True, a held fragment is NOT merged with a follower that parses as a
# complete, semantically-distinct sentence or a new wh/aux question (e.g. "What
# do you see?"); the follower is processed as its own turn instead of producing
# garble like "What the What do you see?". Set False to restore unconditional
# merge-within-window.
INCOMPLETE_TURN_MERGE_REJECT_DISTINCT = True

# Seconds after a Rex statement completes before VAD detections are accepted —
# just long enough for room echo of his own voice to decay. Lowered 0.35 → 0.2 →
# 0.12 to MATCH POST_QUESTION_LISTEN_DELAY_SECS so a reply starting right as Rex
# finishes a statement is detected just as promptly as a reply to a question (the
# buffer is no longer flushed, so the front is preserved and recovered via preroll).
POST_SPEECH_LISTEN_DELAY_SECS = 0.12  # match POST_QUESTION_LISTEN_DELAY_SECS

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

# ── Post-question retro scan ───────────────────────────────────────────────────
# Between the end of a spoken question and the loop's first live mic read there
# are ~0.3-0.7s (echo tail + listen delay + synchronous turn unwind) during
# which NO audio is examined. A clipped one-word answer ("no") spoken there sits
# in the rolling buffer but never triggers live VAD, so it was silently lost —
# live-logged 2026-07-07 during 20 Questions: answers right after a question
# vanished until the player repeated them ~10s later. When the last reply was a
# QUESTION, the loop now runs a ONE-SHOT VAD scan over that buffered dead-window
# span as soon as it resumes listening, and a hit is captured through the normal
# preroll/floor path. Longer utterances never needed this (their tail reaches
# live VAD and preroll recovers the front), which is why it only bit on
# rapid-fire one-word game answers.
POST_QUESTION_RETRO_SCAN_ENABLED = True
POST_QUESTION_RETRO_SCAN_WINDOW_SECS = 2.5   # scan only if the loop resumed within this long
POST_QUESTION_RETRO_SCAN_SKIP_SECS = 0.15    # exclude Rex's decaying room echo at the span start
POST_QUESTION_RETRO_SCAN_MIN_VOICED_FRAMES = 3  # ~96ms of voiced audio required to count as speech

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
# ⚠ RETUNED to MATCH the no-AEC dev-Mac seam values (owner call 2026-07-17): the
# XU316 cancels ~16-17 dB, which is enough to keep the mic USABLE while Rex plays
# (barge-in, commands over music — the reason AEC stays on) but NOT enough to make
# his post-TTS residual silent: with the old aggressive seam (0.05/0.05/0.5) the
# capture reached half a second back into that residual and transcribed his own
# trailing words ("...still under review" → HEARD "with you", conv log 00:10:28).
# The dev Mac's 0.12s seam was tuned hard and doesn't self-transcribe; use it here
# too. The AEC advantage now lives DURING playback, not at the handoff seam.
# Mic-attenuation tail after a reply ends (replaces POST_*_PLAYBACK_SUPPRESSION_SECS).
POST_PLAYBACK_SUPPRESSION_SECS_AEC = 0.12
# Delay before the listen loop resumes after a reply (replaces POST_*_LISTEN_DELAY_SECS).
POST_TTS_LISTEN_DELAY_SECS_AEC = 0.12
# How far back capture may reach past the handoff to recover a reply that overlaps
# Rex's tail. 0.5 pulled the AEC residual of Rex's final word into the capture and
# Whisper transcribed it; 0.12 matches the dev-Mac grace that never did.
POST_TTS_CAPTURE_PREROLL_GRACE_SECS_AEC = 0.12

# Seconds of no detected speech in ACTIVE state before returning to IDLE.
# Raised from 30 so the proactive idle-banter path (below) has room to re-engage
# a couple times before the session actually closes on silence.
CONVERSATION_IDLE_TIMEOUT_SECS = 45.0
ACTIVE_GAME_IDLE_TIMEOUT_SECS = 180.0

# Start (and wake) Rex in ACTIVE, not IDLE — booting/waking him IS activating him, so
# the conversation loop (incl. the short re-engagement of a present-but-quiet person)
# runs from the start instead of waiting for a wake word. Empty-room startup self-
# corrects: re-engagement no-ops with no one present, and the session idle-times out
# back to IDLE. Set False to restore the old wake-word-gated IDLE startup.
STARTUP_STATE_ACTIVE = True

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
IDLE_BANTER_MIN_SECS = 5.0
IDLE_BANTER_MAX_SECS = 8.0
IDLE_BANTER_SECS = 8.0
IDLE_BANTER_COOLDOWN_SECS = 10.0  # minimum gap between nudges
IDLE_BANTER_MAX_PER_STRETCH = 3   # re-engagement attempts per silent stretch while the user is
                                  # PRESENT (escalates ask → on-topic take → playful tease before a
                                  # warm give-up). The earlier over-talk came from short-interval
                                  # banter bypassing the min-gap, NOT the cap — the 10s cooldown,
                                  # low-content gate, opener-diversity guard, and presence-gating
                                  # keep this from being spammy.
# When the FIRST re-engagement question also goes unanswered (truly dead) AND there's a
# live thread to react to, the next nudge VOLUNTEERS a short on-topic take instead of
# asking again — a real angle to push back on or laugh at (idle directive [1]), never the
# off-topic "thing on my mind" preoccupation. Keeps the question-first re-engagement while
# letting Rex have an opinion when the room is genuinely silent. Kill switch.
IDLE_BANTER_VOLUNTEER_TAKE = True
# Low-content / quiet-turn gate. A curt, content-free answer ("not much", "nothing",
# "Hello") is a LEGITIMATE reply, not a topic to keep mining. When the user's last real
# turn is <= IDLE_BANTER_LOW_CONTENT_MAX_WORDS words, idle banter (a) does NOT treat it
# as a live topic to riff on (so it pivots to a fresh question instead of editorializing
# the non-answer), and (b) is capped to IDLE_BANTER_LOW_CONTENT_MAX_PER_STRETCH nudges
# before a real user turn is required. Fixes the live-logged 2026-06-26 pile-on where
# "not much" got editorialized twice ~18s apart ("door closing itself" / "choosing peace
# over explanation"). This is a GATE, not a mute — one warm pivot still fires; a
# substantive turn keeps the full budget. Kill switch.
IDLE_BANTER_LOW_CONTENT_GATE_ENABLED = True
IDLE_BANTER_LOW_CONTENT_MAX_WORDS = 3       # 'not much'/'Hello' = low-content; raise to 4 if too tight
IDLE_BANTER_LOW_CONTENT_MAX_PER_STRETCH = 1 # at most one idle nudge after a curt answer
# MID-CONVERSATION the human is engaged but may pause a beat longer than the cold-room
# 5-8s while composing a reply — and Rex's own reply latency eats into that — so the
# active floor sits just above the cold window, NOT at the old 22-30s (that felt
# abandoned; live-logged 2026-06-19 as a ~30s dead air before Rex spoke up). Over-talk
# (the 2026-06-18 "fresh line every 8s" engine) is now held off by the SHORT single-line
# nudge + IDLE_BANTER_COOLDOWN_SECS + IDLE_BANTER_MAX_PER_STRETCH, not by a long first
# delay — so a responsive ~8-12s first nudge no longer piles on.
IDLE_BANTER_ACTIVE_MIN_SECS = 8.0
IDLE_BANTER_ACTIVE_MAX_SECS = 12.0
# After a normal REPLY that asked a question (even one without a literal '?'),
# hard-suppress idle filler this long so the user gets a real window to answer
# before Rex re-asks. Fixes the duplicate camping question (~39s apart) where the
# only hard suppressor expired after ~7s.
POST_REPLY_QUESTION_WAIT_SECS = 18.0
# Drop a proposed idle-banter question if it re-asks a question Rex already asked
# within this window (keyword/topic overlap) — never become a broken record.
IDLE_BANTER_RECENT_QUESTION_DEDUP_SECS = 120.0
# A 'change the subject' boundary suppresses the just-dropped topic from the
# proactive/idle layer AND the LLM prompt for this long, so the boundary
# acknowledgment isn't undone by an idle line revisiting the topic ~20s later.
TOPIC_BAN_COOLDOWN_SECS = 90.0
TOPIC_BAN_PROACTIVE_SUPPRESS = True

# When the human signals they want OFF the current thread — bored, the bit/metaphor isn't
# landing, or they explicitly ask for something else ("don't you have anything else to say?",
# "you've lost the metaphor", "you keep saying that") — DROP the topic and change direction
# instead of answering it on-topic. Overrides the "stay on this exact topic" reply agenda (field
# 2026-06-30: Rex ground a bed/mattress metaphor for five turns and ignored "anything else?").
# Kill switch.
SUBJECT_CHANGE_ON_CUE_ENABLED = True
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
    # Warmer give-ups that keep the door open instead of a cold brush-off.
    "Alright, I'll let the quiet win this round — holler when you miss me.",
    "Going on standby, but I'm easily summoned. Don't be a stranger.",
    "Fine, soak up the silence. I'll be right here when you want company.",
]

# Stay engaged longer while the user is PRESENT (on camera) but quiet: extend the effective
# idle timeout so a few spaced, varied re-engagement attempts actually land before the give-up
# outro — instead of the 45s timeout structurally allowing only ~1. The moment the user leaves
# the frame, presence drops and the timeout snaps back to CONVERSATION_IDLE_TIMEOUT_SECS so a
# departed/empty room still times out promptly (and is never nudged). Kill switch.
PRESENT_REENGAGE_ENABLED = True
# 120s (was 90s): the fast lull-break eats the first ~30s, so leave room for a slow ~40s-spaced
# re-engagement (LEAN_IMPULSE_REENGAGE_SECS) to land AND get a fair answer window before Rex signs
# off. Still presence-gated — snaps back to 45s the moment the person leaves the frame.
PRESENT_REENGAGE_IDLE_TIMEOUT_SECS = 120.0

# When a re-engagement has already gone unanswered, a LATER idle nudge may playfully call out the
# dead air IN CHARACTER (a fond teasing jab that invites them back — "cat got your tongue?",
# "stumped, or just gone full mute?") instead of another earnest line. This is the ONE idle mode
# allowed to announce the silence; every other directive still must not. Fires only at attempt
# index >= IDLE_BANTER_TEASE_SILENCE_AT (0-based), so the first re-engagement stays earnest. It
# composes with the low-content gate (a curt "not much" is still not mined; only a genuine
# silence gets teased) and inherits the opener-diversity guard for variety. Kill switch.
IDLE_BANTER_TEASE_SILENCE_ENABLED = True
IDLE_BANTER_TEASE_SILENCE_AT = 2

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
# While conversation is suppressed by playback, keep a NARROW ear open for music
# control: VAD + transcription still run (hardware AEC required — the ReSpeaker
# cancels its own playback from the mic), but ONLY stop/skip/volume/shutdown
# commands execute; every other transcript — radio announcers included — is
# dropped with a [dj_listen] log line. Field 2026-07-30: "stop the music",
# repeated at a turned-down amp, was unreachable because the only override was a
# wake word at a RAISED threshold; the owner had to kill the process.
DJ_COMMAND_LISTEN_ENABLED = True
DJ_DUCK_DURING_SPEECH = True
DJ_LISTEN_DUCK_VOLUME = 0.18
DJ_START_AFTER_TTS_DELAY_SECS = 0.25

# After Rex asks a direct question, suppress autonomous/proactive speech for a
# short window so humans get a clean chance to answer.
QUESTION_RESPONSE_WAIT_SECS = 7.0

# Question pacing. Pulled back from 6+3 — relentless profile questions every turn
# read as a boring interview and crowded out the jokes. Rex now leads with a
# reaction/roast and only sometimes asks; this caps how often a question can fire.
# Relaxed (2026-06-19): the tight cap was strangling the conversation and starving the
# proactive lull-fillers (visual curiosity / re-engagement) — see conversation_agenda
# _BUDGETED_PROACTIVE_PURPOSES, which no longer counts the silence-filling paths against
# this budget at all. This budget now only throttles interview-y REPLY follow-ups.
QUESTION_BUDGET_WINDOW_SECS = 90.0
QUESTION_BUDGET_MAX_QUESTIONS = 5
QUESTION_BUDGET_ENGAGED_GRACE_SECS = 60.0
QUESTION_BUDGET_ENGAGED_EXTRA = 2

# Anti-interview cadence (separate from the time-window budget above). Once a topic
# opens, "earned on-thread follow-ups" bypass the budget and Rex can end EVERY turn
# with a question — an interrogation (live-logged 2026-06-20: six question-ending
# turns in a row about a favourite movie). After this many consecutive question-ending
# turns, social_frame forces the next reply to be a statement/reaction (then the streak
# resets). Urgent identity/emotional asks still override it. The streak also resets
# after RESET_SECS of no Rex turn (the interview cooled off).
INTERVIEW_CADENCE_CLAMP_ENABLED = True
INTERVIEW_CADENCE_MAX_CONSECUTIVE_QUESTIONS = 3
INTERVIEW_CADENCE_RESET_SECS = 120.0

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
# Run the onboarding burst on known special people (the creator Bret Benziger
# and the person_specials VIPs)? True (default, owner call 2026-07-07): a VIP
# whose person row is fresh/wiped is a data-blank like any other newcomer — Rex
# knows the name and the loyalty bits on sight, but zero facts, so the baseline
# burst should run. Established VIPs are already spared by the visit-count and
# fact-floor gates above, so this flag only ever bites on empty profiles. Set
# False to restore the "never interrogate the maker" exemption (the 2026-06-18
# BUG-5 behavior, from before the burst got answer-aware reactions and the 3/5
# question pullback).
# See intelligence/onboarding.eligible + intelligence/person_specials.is_special_person.
ONBOARDING_INCLUDE_VIPS = True

# Pacing.
ONBOARDING_KICKOFF_SECS = 1.2              # beat after the enrollment ack before the first question
ONBOARDING_INACTIVITY_TIMEOUT_SECS = 30.0  # close the burst out loud after this much silence
ONBOARDING_STEP_TTL_SECS = 240.0           # deep fallback: flow hard-expires this many secs after it is ARMED (wall-clock since created_at, NOT sliding on activity)
ONBOARDING_SOFT_DISENGAGE_LIMIT = 2        # lukewarm answers in a row (past MIN) -> wind down
ONBOARDING_REVEAL_EVERY = 2                # inject a Rex self-reveal ~every N questions (0 = off); 2 lands ≥1 reveal even in a 3-question burst (since_reveal inits at 0)

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

# Sleep is intentionally ONNX-only: ordinary speech, general Rex wake models,
# GUI/text input, and Whisper transcription cannot wake him. This makes SLEEP a
# real low-attention state with one explicit acoustic exit.
SLEEP_ONNX_ONLY_WAKE = True
SLEEP_TRANSCRIBED_WAKE_FALLBACK_ENABLED = False

# ─────────────────────────────────────────────────────────────────────────────
# CONSCIOUSNESS LOOP
# ─────────────────────────────────────────────────────────────────────────────

# How frequently the consciousness loop ticks to check WorldState and trigger behavior
CONSCIOUSNESS_LOOP_INTERVAL_SECS = 1.0

# Minimum spacing between autonomous/proactive spoken lines from consciousness.
# This is the ENGAGED-tier base; see the presence-gated clamp below.
CONSCIOUSNESS_PROACTIVE_MIN_GAP_SECS = 12.0

# ── Presence-gated proactive cadence clamp (intelligence/presence_cadence.py) ─
# The REAL fix for unprompted over-talk (owner direction 2026-07-06): the gap
# between chatter-class proactive lines scales with presence — 12s (base above)
# while a conversation is flowing, longer when someone is present but quiet,
# near-silent in an empty room. Enforced CENTRALLY in the action governor as a
# hard reject, closing the historical leak where submit_external candidates
# (idle banter, priority 50) carried no cooldown metadata and faced no cadence
# gate at all. Event-driven purposes (greetings, wave-backs, identity asks,
# check-ins) are never clamped.
PROACTIVE_CADENCE_CLAMP_ENABLED = _env_bool("PROACTIVE_CADENCE_CLAMP_ENABLED", True)
PROACTIVE_GAP_PRESENT_IDLE_SECS = 45.0   # someone visible, no live conversation
PROACTIVE_GAP_EMPTY_ROOM_SECS = 600.0    # nobody visible — quiet when you leave
# Chatter-class purposes subject to the clamp (event-driven purposes excluded).
PROACTIVE_CADENCE_CLAMP_PURPOSES = (
    "idle_monologue", "small_talk", "visual_curiosity", "lull_callback",
    "memory_followup", "room_change", "room_reaction",
    "weather.proactive_comment",
)

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
# ── Tool-calling router, Phase 0 shadow (docs/tool_router_scope.md) ──────────
# When ON, each routed turn ALSO asks the conversation model to pick a tool for
# the same utterance/context, and logs the choice next to the shipped decision
# ([tool_router_shadow] lines; aggregate with tools/tool_router_report.py).
# Costs one small hosted call per routed turn — enable in user_config.py for a
# collection week, decide cutover from the report, then turn it back off.
TOOL_ROUTER_SHADOW_ENABLED = False
TOOL_ROUTER_SHADOW_MODEL = ""        # "" = LLM_CONVERSATION_MODEL
TOOL_ROUTER_SHADOW_TIMEOUT_SECS = 8.0
# Phase 1 LIVE cutover (2026-08-01, on the collection evidence: tool router
# ~92% vs shipped ~80%, decoy false-positives 0/6): the actions below ride the
# lean REPLY call as native tools — the model answers in prose OR calls one,
# and a call dispatches the same _handle_classified_intent executor the intent
# classifier uses. Zero extra LLM round-trips. The deterministic layers still
# run FIRST, so this only catches what used to fall through to conversation
# (the off-pattern phrasings: "how's the weather looking tomorrow?",
# "kill the music"-class misses). Kill switch below reverts to pre-cutover
# behavior instantly.
TOOL_ROUTER_LIVE_ENABLED = True
TOOL_ROUTER_LIVE_ACTIONS = (
    "time.query", "date.query", "weather.query",
    "status.capabilities", "status.uptime",
    "vision.describe_scene", "music.options",
    # system.* added 2026-08-02: "Can you shut down, please?" fell through to
    # conversation (the deterministic guard rejects "can you..." on purpose to
    # protect "can you shut down the music") and Rex SAID "Shutting down."
    # without doing it. The dispatcher still verifies the utterance with
    # command_parser.is_shutdown_request/is_sleep_request before executing.
    "system.sleep", "system.shutdown",
    # web.search added 2026-08-02: "What's going on with the Iran War?" hit the
    # deterministic conversational skip (no trigger phrase, autonomous gate
    # silent) and Rex refused from stale knowledge — while the whole web_search
    # feature sat unused one module over. The reply-call LLM is the right judge
    # of "this needs live data"; the search itself stays grounded (citations,
    # link/markdown stripping, offline refusal).
    "web.search",
    # Phase 2 live batch (2026-08-02, user-approved): off-pattern phrasings for
    # actions with existing executors. event.cancel ("we're not going to Lake
    # Folsom anymore"), memory.query, identity.who_is_speaking, and the music
    # controls ("kill the music"-class misses). All keep their deterministic
    # fast lanes; the write-path (event.cancel) keeps looks_like_cancellation
    # as its guard. vision.snapshot deliberately NOT live — it has no executor.
    "event.cancel", "memory.query", "identity.who_is_speaking",
    "music.play", "music.stop", "music.skip",
    # vision.snapshot (2026-08-02): live now that the feature EXISTS — the tool
    # speaks the privacy confirmation offer and arms the pending slot; a spoken
    # "yes, remember this scene" captures + captions + stores a scene episode.
    "vision.snapshot",
    # 2026-08-02 PM: "my name's not Brad, it's JT" and "forget who I am" both
    # fell to conversation (shadow picked the right tools). name_correction
    # executes directly; forget_person routes into the existing wipe-
    # confirmation flow and may only target the CURRENT speaker.
    "identity.name_correction", "memory.forget_person",
)

# Seconds the "say yes, remember this scene" confirmation slot stays open.
SCENE_SNAPSHOT_CONFIRM_TIMEOUT_SECS = 30.0
ACTION_ROUTER_LOG_DECISIONS = True
ACTION_ROUTER_AUDIT_LOG_ENABLED = True
ACTION_ROUTER_EXECUTE_ENABLED = True
# Skip the router's LLM call on deterministically-conversational turns (no action cue
# words, no active game/music, deterministic intent = general). Measured ~0.8s saved
# per chat turn (2026-07-06 latency work); canonical commands still hit the explicit
# regex classifiers, and any cue word keeps the LLM router in the loop.
ACTION_ROUTER_DETERMINISTIC_SKIP_ENABLED = _env_bool("ACTION_ROUTER_DETERMINISTIC_SKIP_ENABLED", True)
# Mirror-image skip for the opposite case: the deterministic intent classifier
# ALREADY claims the turn as a self-knowledge query answered from local data
# (time/date/weather/uptime/capabilities/games/who-is-speaking). The LLM router
# can only agree, so skip its call and let the intent classifier execute as it
# would have anyway (~0.9s saved per basic query, measured 2026-08-02). Music,
# memory, and vision intents deliberately still route through the LLM.
ACTION_ROUTER_SELF_QUERY_SKIP_ENABLED = _env_bool("ACTION_ROUTER_SELF_QUERY_SKIP_ENABLED", True)
ACTION_ROUTER_EXECUTE_ACTIONS = {
    "conversation.repair",
    "humor.tell_joke",
    "humor.roast",
    "humor.free_bit",
    "performance.dj_bit",
    "performance.body_beat",
    "performance.mood_pose",
    "performance.impersonate",
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
# Decoupled from LLM_MODEL 2026-08-02: gpt-4o-mini's ~1.08s median TTFT was the
# whole cost of the blocking routing call; gpt-5.4-nano benchmarks at ~0.63s and
# is priced/positioned for classification+routing. Calls go through llm_compat
# (GPT-5 param contract) with reasoning effort "none" to keep TTFT low.
# ROLLBACK = set this back to "gpt-4o-mini" (or LLM_MODEL) in user_config.py.
ACTION_ROUTER_MODEL = "gpt-5.4-nano"
ACTION_ROUTER_REASONING_EFFORT = "none"
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
# Same-conversation continuity: a slot seen within RECENT_STICKY_SECS reuses at this
# lower bar WITHOUT requiring the raw top candidate to agree (the raw label flip-flops
# when two enrolled prints overlap — the Guest 2/3/4-per-utterance churn).
ANONYMOUS_SPEAKER_RECENT_STICKY_SECS = _env_float("ANONYMOUS_SPEAKER_RECENT_STICKY_SECS", 180.0, min_value=0.0, max_value=3600.0)
ANONYMOUS_SPEAKER_RECENT_STICKY_THRESHOLD = _env_float("ANONYMOUS_SPEAKER_RECENT_STICKY_THRESHOLD", 0.62, min_value=0.0, max_value=1.0)

# Dual-unknown introduction: two unknown faces on camera + an unrecognized voice →
# ask positionally ("you on my LEFT — what's your name?", then the right), binding
# each answer's name to the face at that position plus the answer's voice audio.
# One-known-one-unknown keeps the existing single-unknown ask.
DUAL_INTRO_ENABLED = _env_bool("DUAL_INTRO_ENABLED", True)
DUAL_INTRO_WINDOW_SECS = _env_float("DUAL_INTRO_WINDOW_SECS", 45.0, min_value=5.0, max_value=300.0)
DUAL_INTRO_COOLDOWN_SECS = _env_float("DUAL_INTRO_COOLDOWN_SECS", 120.0, min_value=0.0, max_value=3600.0)
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
# Cross-session voice -> KNOWN-PERSON resolution: when an unrecognized voice matches a
# persisted signature that was already linked (attach_person) to a named person in an
# earlier session, resolve the turn straight to that person instead of minting a fresh
# unknown_voice_N. Only fires with NO live face/voice person match. The floor sits ABOVE
# the match threshold (0.74) so naming someone needs a confident print.
VOICE_SIGNATURE_RESOLVE_PERSON_ENABLED = True
# RAISED 0.80 -> 0.85 after the ECAPA calibration (2026-07-06): this is the
# highest-stakes voice-only action (naming a person outright, cross-session, no
# face check), and genuine ECAPA matches sit ~0.90+ — the extra margin is free.
VOICE_SIGNATURE_RESOLVE_PERSON_MIN_SCORE = 0.85

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
# Feed the LOCAL object detector's confirmed objects (world_state.objects) into the
# visual-curiosity prompt so Rex grounds the question in a REAL named object he can
# see (a detector-verified "chair"/"guitar"/"plant"), not just the GPT vision blob.
# Off → the prior behavior (GPT scene summary only).
VISUAL_CURIOSITY_USE_OBJECTS = True
VISUAL_CURIOSITY_OBJECTS_MAX = 6            # most-confident N objects fed to the prompt
VISUAL_CURIOSITY_OBJECTS_MIN_CONFIDENCE = 0.40

# ── Land-the-laugh / take-a-bow ───────────────────────────────────────────────────
# React to the ROOM landing Rex's material: applause -> a take-a-bow (proud_dj_pose +
# a line), laughter shortly after a Rex line -> a dry follow-through. Reads the
# (otherwise unread) audio_scene.applause_detected / laughter_detected signals, gated
# on a recent-Rex-utterance window (so ambient noise/music/TV doesn't set him off) plus
# a global cooldown and a LOW per-session cap (so "see, that one's free" never reads as
# needy). Yields to live speech/music/games like every reaction.
ROOM_REACTION_ENABLED = True
ROOM_REACTION_AFTER_REX_SECS = 12.0   # laughter/applause only counts within this of a Rex line
# ...but not INSTANTLY: the first analysis window after his TTS unmutes still carries
# his own decaying tail + room echo, which reads as applause. Field 2026-07-24: he took
# a bow at a silent, seated room. Real human applause starts later than his own reverb.
ROOM_REACTION_MIN_AFTER_REX_SECS = 1.5
ROOM_REACTION_MIN_GAP_SECS = 20.0     # global cooldown (also de-dups one multi-cycle burst)
# Low cap: laughter detection has false positives (TV/AC/his own TTS tail), and even one
# unearned victory lap reads as needy — two read as a malfunction (field log 2026-07-03).
ROOM_REACTION_SESSION_CAP = 2         # max take-a-bow / follow-throughs per session
# The burst detectors can't tell a human laugh from Rex's OWN mechanicals — servo
# whine, drive-base motors, and sfx chirps all read as rhythmic bursts (field
# 2026-07-30: "See? That one was free." fired at a not-laughing owner right after
# a back-up move; an applause bow fired at plain face-tracking servo noise).
# Guard 1: skip while the base is moving or within this window of any sfx start.
ROOM_REACTION_SELF_NOISE_GUARD_ENABLED = True
ROOM_REACTION_SELF_NOISE_GUARD_SECS = 4.0
# Guard 2: only credit laughter/applause when a visible face looks amused RIGHT
# NOW (MediaPipe "happy" at/above this confidence, fresh reading). No smile in
# view — including an empty room — means no victory lap.
ROOM_REACTION_REQUIRE_VISIBLE_AMUSEMENT = True
ROOM_REACTION_AMUSEMENT_MIN_CONFIDENCE = 0.5
# Keep every line free of claims about what the person is PHYSICALLY doing. Rex cannot
# see posture reliably, and asserting it lands as a malfunction when he's wrong — field
# 2026-07-24: "No need to stand. ...Oh, you're already standing." was delivered to a
# seated owner. Same rule as the persona's "never invent physical details".
ROOM_APPLAUSE_REACTION_LINES = [
    "Thank you, thank you. Hold the applause — actually, don't.",
    "Please, hold your applause. ...Okay, don't.",
    "I'll take that as a rave review.",
    "And THIS is why they keep me plugged in.",
    "I'd take a bow, but my actuators bill by the hour.",
]
# NOTE: keep these free of "I can't move/leave" jokes — the robot is getting wheels.
# Keep these PERSON-DIRECTED, not stage-directed: "I'll be here all week" /
# "a droid doing stand-up" is comedy-club shtick that reads bizarre one-on-one
# in a bedroom (owner: "kinda weird", 2026-07-06). Rex lands a laugh WITH the
# person, he doesn't play a room.
ROOM_LAUGHTER_REACTION_LINES = [
    "See? That one was free.",
    "Comedy subroutine: validated.",
    "There it is. Carbon-based approval.",
    "Careful — laughing just encourages me.",
    "I heard that. I'm counting it.",
]

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

# ── Local object detection (COCO 80-class via the SHARED MediaPipe detector) ──────
# The animal detector already runs the full 80-class EfficientDet-Lite0 model and
# throws away every non-animal box. This stream KEEPS the rest — the room's
# furniture and items — as world_state.objects, the substrate for object-grounded
# curiosity, "wait, that's new" change detection, and the persistent room model.
# It reuses the SAME loaded detector (one model, a separate inference pass).
# "Rich" privacy posture: open vocabulary MINUS screens/devices (never publish a
# laptop/tv/phone) and MINUS people/animals (already tracked in world_state.people
# and world_state.animals).
OBJECT_DETECTION_ENABLED = _env_bool("OBJECT_DETECTION_ENABLED", True)
OBJECT_DETECTION_INTERVAL_SECS = _env_float(
    "OBJECT_DETECTION_INTERVAL_SECS",
    2.5,
    min_value=0.5,
    max_value=30.0,
)
# A detected object must score at least this to be published (room objects are usually
# clearer than a pet held to a wide lens, so the bar is a touch higher than animals').
OBJECT_DETECTION_SCORE_THRESHOLD = _env_float(
    "OBJECT_DETECTION_SCORE_THRESHOLD",
    0.35,
    min_value=0.05,
    max_value=0.95,
)
OBJECT_DETECTION_MAX_RESULTS = _env_int(
    "OBJECT_DETECTION_MAX_RESULTS",
    12,
    min_value=1,
    max_value=25,
)
# Self-occlusion mask (field bug 2026-07-12): Rex's own eye stalks sit in front of the
# wide lens and read as big dark blobs at the 1080p crop's bottom corners — the object
# detector kept publishing them as "chairs" (and once a 0.21 "dog"), feeding phantom
# furniture into world_state.objects, the rex.db room model, and visual curiosity.
# Normalized (x0, y0, x1, y1) rects in frame coordinates; any detection whose box lies
# MOSTLY inside a zone is dropped at the source. The GUI vision panel outlines the
# zones (dim dashed violet) so they can be aligned against the live feed by eye —
# adjust here if the camera or the eye hardware moves.
# The zones describe the ROBOT'S face hardware — on a dev Mac's built-in camera
# there are no eye stalks in frame, and masking a third of the picture just
# hides real objects. When CAMERA_DEVICE_NAME points at a built-in Mac camera
# (e.g. "MacBook Pro Camera"), the zones are disabled entirely; every consumer
# (object scan, animal scan, GUI overlay) already tolerates an empty list.
_CAMERA_IS_DEV_MAC = "macbook" in (os.getenv("CAMERA_DEVICE_NAME") or "").strip().lower()
CAMERA_SELF_OCCLUSION_ZONES = [] if _CAMERA_IS_DEV_MAC else [
    (0.00, 0.50, 0.32, 1.00),   # left eye stalk (bottom-left blob) — widened 0.15 -> 0.32
                                # (field 2026-07-17: it still read as a 55% "chair"; the
                                # blob spans ~30% of the frame width, screenshot-verified)
    (0.60, 0.45, 1.00, 1.00),   # right eye stalk (bottom-right blob, the "chair")
]
CAMERA_SELF_OCCLUSION_MAX_OVERLAP = 0.55   # box fraction inside a zone that kills it
# Consecutive-scan confirm streak before an object counts as really present — indoor
# flicker / one-frame misreads must persist first, exactly like animal arrivals.
OBJECT_DETECTION_CONFIRM_SCANS = _env_int(
    "OBJECT_DETECTION_CONFIRM_SCANS",
    2,
    min_value=1,
    max_value=10,
)
# Honor the no-screens rule: these COCO classes are dropped AT DETECTION TIME so a
# screen/device never reaches world_state.objects (or the GUI / room model / prompt).
# Matched case-insensitively against the model's lowercase class name.
OBJECT_DETECTION_BANNED_CLASSES = {
    "laptop", "tv", "tvmonitor", "monitor", "screen",
    "cell phone", "cellphone", "keyboard", "mouse", "remote",
}
# Person-oriented object salience (2026-07-08): small objects whose box center falls
# inside a visible person's body zone (face box widened + extended to lap height, and
# small enough to hold) are tagged near_person / near_person_name at publish time
# (vision.scene.tag_person_adjacent_objects). The lean impulse and visual curiosity
# put those FIRST with a "this beats the furniture" note — so a cup in someone's hand
# gets "what are you drinking?" instead of a riff on the background chair
# (live-logged: Bret held a cup for minutes while Rex asked about a chair).
OBJECT_NEAR_PERSON_ENABLED = True

# "What's that you're drinking?" — Rex proactively asks about an object someone is
# HOLDING (a near_person object, tagged above). This is the direct payoff of person-
# oriented salience (owner 2026-07-08: "comment on objects I'm holding more often" —
# he held a cup through whole sessions and Rex never asked). Event-driven, not lull
# taxonomy: fires once an object PERSISTS in-hand for MIN_HOLD_SECS (absorbs one-frame
# flicker), yields to live conversation (_can_proactive_speak), and is bounded by a
# per-label session de-dup + cooldown + LOW session cap. Unlike ROOM_CHANGE it needs
# NO room-model baseline — a held object is salient on a fresh install too. It also
# routes through the action governor at a higher priority than visual_curiosity /
# lull_callback (the thing in their hands beats the room). Kill switch:
HELD_OBJECT_REMARK_ENABLED = True
HELD_OBJECT_REMARK_MIN_HOLD_SECS = 5.0    # in-hand this long before he asks (flicker guard)
HELD_OBJECT_REMARK_COOLDOWN_SECS = 90.0   # min gap between held-object asks
HELD_OBJECT_REMARK_SESSION_CAP = 3        # don't interrogate every item they pick up

# ── Room model (persistent object permanence in rex.db) ───────────────────────────
# Record which objects Rex has seen over time (memory/room_model.py, fed by the local
# COCO stream) so curiosity prefers what's NEW and Rex can notice a genuinely new object
# across sessions. Rides on EPISODIC_MEMORY_ENABLED (the rex.db capture kill switch) plus
# its own flag; never writes a real rex.db under the test runner.
ROOM_MODEL_ENABLED = True
# An object becomes a "fixture" Rex knows once recorded this many times (≈ object scans
# at OBJECT_DETECTION_INTERVAL_SECS, so ~20 ≈ ~50s of presence).
ROOM_MODEL_ESTABLISHED_SIGHTINGS = 20
# Curiosity treats an object as NEW-to-the-room (prefer asking about it) below this count.
ROOM_MODEL_NOVELTY_MAX_SIGHTINGS = 6

# "Wait — that's new": when the room is KNOWN (an established baseline exists) and a
# genuinely new object shows up (currently present, low recorded sighting count, never a
# fixture), Rex remarks on it ONCE. Heavily gated because the COCO detector is noisy — it
# needs a baseline first, fires only in a lull (via _can_proactive_speak), and is bounded
# by a cooldown + a LOW per-session cap + per-label de-dup. Kill switch:
ROOM_CHANGE_REMARK_ENABLED = True
ROOM_CHANGE_MIN_BASELINE = 4      # need ≥ this many known fixtures before noticing changes
ROOM_CHANGE_MIN_SIGHTINGS = 2     # the new object must be confirmed (not a 1-frame misread)
ROOM_CHANGE_MAX_SIGHTINGS = 12    # ...but still recent (just appeared), not a slow fixture
ROOM_CHANGE_COOLDOWN_SECS = 120.0
ROOM_CHANGE_SESSION_CAP = 3
# When someone is VISIBLY PRESENT, a new object is a conversation opener: Rex asks
# about it via an LLM-generated curious question ("What kind of sandwich are we
# dealing with?") instead of a canned observation — owner feedback 2026-07-06
# ("A wild sandwich appears" should have been "what kind / is it good?").
ROOM_CHANGE_ASK_WHEN_PERSON_PRESENT = True

# ── Learn-by-asking room questions (curiosity Phase 1, intelligence/room_questions.py)
# Genuinely-new-to-the-room objects (rarity-gated in memory/room_model.py) queue a
# durable "ask about this" item; the idle-question path asks it BEFORE any personal
# profile question (starvation rule), and the person's answer is written back to
# the room model as the object's human-given name with corroboration counting.
ROOM_QUESTIONS_ENABLED = _env_bool("ROOM_QUESTIONS_ENABLED", True)
ROOM_QUESTION_COOLDOWN_SECS = _env_float("ROOM_QUESTION_COOLDOWN_SECS", 600.0, min_value=0.0, max_value=86400.0)
ROOM_QUESTION_ANSWER_TTL_SECS = _env_float("ROOM_QUESTION_ANSWER_TTL_SECS", 90.0, min_value=5.0, max_value=600.0)
ROOM_QUESTION_ANSWER_TURNS = _env_int("ROOM_QUESTION_ANSWER_TURNS", 2, min_value=1, max_value=10)
# The room model must be at least this old before novelty can queue questions —
# a fresh install's day-one furniture trickle must not become an interview.
ROOM_QUESTION_MIN_ROOM_AGE_DAYS = _env_float("ROOM_QUESTION_MIN_ROOM_AGE_DAYS", 1.0, min_value=0.0, max_value=365.0)

# ── Impulse discipline + detector humility (field rework 2026-07-18) ──────────
# Rolling rate cap on lean impulses — does NOT reset when the user replies (the
# per-run counters do, which is how six lines landed in three minutes).
LEAN_IMPULSE_RATE_WINDOW_SECS = _env_float("LEAN_IMPULSE_RATE_WINDOW_SECS", 600.0, min_value=60.0, max_value=3600.0)
LEAN_IMPULSE_MAX_PER_WINDOW = _env_int("LEAN_IMPULSE_MAX_PER_WINDOW", 5, min_value=1, max_value=50)
# Low-energy (tired / disengaged / question-averse) impulse gap — and impulses
# become statement-or-pass, never questions.
LEAN_IMPULSE_LOW_ENERGY_GAP_SECS = _env_float("LEAN_IMPULSE_LOW_ENERGY_GAP_SECS", 120.0, min_value=10.0, max_value=3600.0)
# A dated event more than this many days past its date is stale — asking about
# it reads as surveillance, not attentiveness (expired lazily at the source).
FOLLOWUP_DATED_MAX_AGE_DAYS = _env_float("FOLLOWUP_DATED_MAX_AGE_DAYS", 5.0, min_value=0.5, max_value=90.0)
# A wave DURING a conversation gets a silent wave-back, not a spoken re-greeting.
WAVE_BACK_SILENT_IN_CONVERSATION_SECS = _env_float("WAVE_BACK_SILENT_IN_CONVERSATION_SECS", 90.0, min_value=0.0, max_value=3600.0)
# Room-change remarks: a real new object PERSISTS — require this much wall-clock
# span between first and last sighting (a one-flicker misread has ~0)...
ROOM_CHANGE_MIN_SPAN_SECS = _env_float("ROOM_CHANGE_MIN_SPAN_SECS", 45.0, min_value=0.0, max_value=3600.0)
# ...and never remark on soft/carriable labels right next to a person.
ROOM_CHANGE_SOFT_LABELS = (
    "handbag", "backpack", "suitcase", "tie", "umbrella", "cell phone",
    "book", "cup", "bottle", "remote",
)
# Large fixed classes that enter the frame whenever the camera pans — they can
# NEVER be "new to the room" (field 2026-07-18: the bed, misread as couch,
# "just appeared out of nowhere").
ROOM_CHANGE_FURNITURE_LABELS = (
    "couch", "bed", "chair", "dining table", "tv", "refrigerator",
    "toilet", "sink", "oven", "microwave", "potted plant",
)
# A story cache older than this is stale news — pick_story returns nothing
# rather than open with "did you hear" about yesterday.
CURRENT_EVENTS_MAX_AGE_HOURS = _env_float("CURRENT_EVENTS_MAX_AGE_HOURS", 36.0, min_value=1.0, max_value=168.0)
# Weekend-plans discovery ask ("got anything going this weekend?"): Thu-Sun,
# once per ISO week per person (durable), skipped when a stored upcoming event
# exists (Rex references THAT instead) or the user is low-energy.
WEEKEND_PLANS_ASK_ENABLED = _env_bool("WEEKEND_PLANS_ASK_ENABLED", True)
WEEKEND_PLANS_ASK_WEEKDAYS = (3, 4, 5, 6)   # Thu, Fri, Sat, Sun (Monday=0)
# Rich-share follow-up: a substantive answer to Rex's question earns one
# concrete follow-up question in the same reply (the inverse of the flat-
# answer probe). Cooldown keeps consecutive turns from becoming an interview.
RICH_SHARE_FOLLOWUP_ENABLED = _env_bool("RICH_SHARE_FOLLOWUP_ENABLED", True)
RICH_SHARE_FOLLOWUP_COOLDOWN_SECS = _env_float("RICH_SHARE_FOLLOWUP_COOLDOWN_SECS", 120.0, min_value=0.0, max_value=3600.0)
# The idle-wander spoken re-greet stays silent while the person spoke recently
# (the head motion still happens) — "Oh—still here" twice in a 3-minute live
# conversation was noise, not presence.
IDLE_REGREET_MIN_USER_SILENCE_SECS = _env_float("IDLE_REGREET_MIN_USER_SILENCE_SECS", 180.0, min_value=0.0, max_value=3600.0)

# ── Novelty drive (awareness/novelty_drive.py, curiosity Phase 2) ─────────────
# Time-since-anything-new. Stale -> idle behaviors tilt toward looking around;
# very stale + empty room + healthy pack -> OPT-IN self-triggered exploration.
NOVELTY_STALE_AFTER_SECS = _env_float("NOVELTY_STALE_AFTER_SECS", 1800.0, min_value=60.0, max_value=86400.0)
NOVELTY_STALE_LOOK_BOOST = _env_float("NOVELTY_STALE_LOOK_BOOST", 3.0, min_value=1.0, max_value=20.0)
# ⚠ Default OFF: this moves the robot with nobody around. Enable deliberately.
EXPLORE_SELF_TRIGGER_ENABLED = _env_bool("EXPLORE_SELF_TRIGGER_ENABLED", False)
EXPLORE_SELF_TRIGGER_STALENESS_SECS = _env_float("EXPLORE_SELF_TRIGGER_STALENESS_SECS", 3600.0, min_value=300.0, max_value=86400.0)
EXPLORE_SELF_TRIGGER_COOLDOWN_SECS = _env_float("EXPLORE_SELF_TRIGGER_COOLDOWN_SECS", 7200.0, min_value=600.0, max_value=86400.0)

# ── Diary retention sweep (memory/consolidation.py, runs at shutdown) ─────────
PERSON_SEEN_RETENTION_DAYS = _env_float("PERSON_SEEN_RETENTION_DAYS", 30.0, min_value=1.0, max_value=3650.0)
VISIT_RETENTION_DAYS = _env_float("VISIT_RETENTION_DAYS", 90.0, min_value=1.0, max_value=3650.0)
ROOM_QUESTION_PENDING_EXPIRY_DAYS = _env_float("ROOM_QUESTION_PENDING_EXPIRY_DAYS", 7.0, min_value=0.5, max_value=365.0)
# The canned one-liners below are the ALONE behavior (muttering at an empty room).
ROOM_CHANGE_REMARK_LINES = [
    "Hold on — when did that {label} get here?",
    "New {label}. The room's redecorating without consulting me.",
    "Is that {label} new? I keep an inventory, you know.",
    "Wait. That {label} wasn't there a minute ago. I notice things.",
    "A wild {label} appears. The room's got range.",
]

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
# The local MediaPipe animal detector only knows bird/cat/dog/horse — none of the startle
# species above. When local detection is on, run a low-frequency OpenAI scan (a paid vision
# call, people-present-gated) so a snake/spider/wasp can still trigger the startle reaction
# (#29). Kill switch + cadence below; raise the interval (or disable) to trim cost.
STARTLE_DETECTION_ENABLED = _env_bool("STARTLE_DETECTION_ENABLED", True)
STARTLE_DETECTION_INTERVAL_SECS = _env_float(
    "STARTLE_DETECTION_INTERVAL_SECS", 60.0, min_value=10.0, max_value=3600.0,
)

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

# Scale the TONE of return + departure reactions by the relationship (a warmer/sharper
# rib for a close friend or someone who needles Rex, a plain note for a near-stranger),
# the way arrivals already do via _greeting_profile. Reuses llm._relationship_tone_rule.
# Off → the prior flat "warm but dry" / "playful and dry" lines for everyone.
PRESENCE_RELATIONSHIP_TONE_ENABLED = True

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
# How long a queued presence line (startup greeting, departure, return) may WAIT for a
# transient proactive block to clear before being dropped (with a log). Covers the
# phantom-turn race: a hallucinated VAD segment blocks proactive speech for ~1-2s and
# used to swallow the startup greeting silently.
PRESENCE_SPEAK_GRACE_SECS = _env_float("PRESENCE_SPEAK_GRACE_SECS", 8.0, min_value=0.0, max_value=60.0)

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
# How long the "who's this?" in-flight latch survives before it's treated as dead and
# retried. Under ENFORCE the governor can reject the submitted candidate (a higher-priority
# reactor wins the tick) so its on_spoke never runs to clear the latch; this timeout keeps a
# rejected ask from wedging the reactor for the session. MUST exceed the worst-case in-flight
# duration: unlike identity_prompt (which speaks a FIXED string, so its window is just enqueue
# latency and 10s is fine), the relationship line runs the LLM (get_response) INSIDE the
# in-flight window before on_spoke fires — bounded by LLM_REQUEST_TIMEOUT_SECS (30s). A window
# below that would falsely judge a slow-but-legitimate generation "stale", clear the latch, and
# submit a SECOND candidate → Rex asks "who's this?" twice. 40s clears the 30s ceiling with
# margin, so a second candidate is never submitted while the first is still generating; the
# tradeoff is a genuinely-rejected ask waits ~40s (≈ the 45s re-ask cooldown) before retrying.
RELATIONSHIP_PROMPT_INFLIGHT_STALE_SECS = 40.0

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
# Allow the solo-unknown "what name should I save for you?" ask during ACTIVE state.
# The session boots into ACTIVE and holds it for ~60s of silence before the idle
# timeout — with this False, a SILENT stranger (nothing for the voice path to key on)
# got no acknowledgment at all for that whole window, and the one ask that fired at
# the ACTIVE->IDLE transition landed inside the 5s post-conversation suppression and
# was rejected (live-logged 2026-07-06-19-20). The ask is salient (time-sensitive) and
# still yields to live speech, awaiting-a-reply, DJ, games, and open flows.
IDENTITY_PROMPT_ALLOW_PROACTIVE_ACTIVE = True
# If the in-flight latch is older than this, the governor rejected the candidate
# (its speak_fn/on_done never ran to clear the latch) — recover and re-ask.
IDENTITY_PROMPT_INFLIGHT_STALE_SECS = 10.0

# When Rex has just asked an unidentified speaker for their name and they reply
# with something he can't parse into a usable name — typically because his own
# question tail bled into the mic ("...save for you?" → transcript "for you,
# Bret.") — he gently re-asks instead of routing the turn to the open-ended LLM.
# Without this, the LLM was handed BOTH the name-bearing transcript AND the
# "ask for their name" directive, producing the contradictory "Bret, got it…
# what do I call you?" (live-logged 2026-06-18). Bounded so a run of garbled
# replies can't loop: after this many re-asks Rex drops it and the turn flows
# normally.
IDENTITY_PROMPT_REASK_MAX = 2
IDENTITY_PROMPT_REASK_LINES = [
    "Static on my end — say just the name one more time?",
    "My audio receptors garbled that. What's the name?",
    "Didn't quite catch that. Just your name — go again?",
    "Run that back for me — what should I call you?",
]

# Minimum gap between CANONICAL renames of the same person. A joking child renamed
# himself Wade->Bro->Broski with each obeyed instantly; this makes the second rename
# within the window get a "you just changed your name" deferral instead. 0 disables.
IDENTITY_RENAME_COOLDOWN_SECS = 120.0

# Face detection can flicker off for a second while a newcomer is still present.
# During this grace window, do not treat an unmatched voice as off-camera.
UNKNOWN_FACE_RECENT_GRACE_SECS = 6.0
# After the user explicitly introduces a newcomer ("this is my partner JT"), stand the
# urgent "who's the mystery guest?" identity-handoff agenda down for this long so Rex
# stops re-asking every turn while voice/face enrollment catches up (the JT run looped it).
UNKNOWN_GUEST_AGENDA_SUPPRESS_AFTER_INTRO_SECS = 45.0

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
# person" cross-match band so a passing resemblance can't confidently steal a
# known identity, while a genuine returning speaker clears it.
# Below this a voice match still wins under voice-primary (margin-guarded), but is
# labelled provisional and won't trigger a face-confirmed voiceprint refresh.
# RAISED 0.70 -> 0.75 after the ECAPA calibration (2026-07-06): impostor
# cross-match now maps to ~0.25-0.45 (was ~0.59-0.64 under Resemblyzer) while
# Bret's live band is ~0.85-0.94 typical / ~0.70-0.77 on short commands. 0.75
# adds real margin against a confident false accept without demoting genuine
# short turns; do NOT push to 0.80 — that starts eating the owner's live
# short-command band, and soft murmurs already rely on the continuity anchor.
SPEAKER_ID_CONFIDENT_THRESHOLD = 0.75

# Voice-only challenge (the single-print cross-match trap, field log 2026-07-05: JT's
# voice matched Bret's print at 0.660 while the camera showed only JT). On the
# voice-only path a MARGINAL match is challenged — "who's that speaking?" — instead of
# silently attributed, when the matched person hasn't been on camera within the grace
# window AND someone else (face or real pose) is visible right now.
SPEAKER_ID_UNSEEN_CHALLENGE_ENABLED = _env_bool("SPEAKER_ID_UNSEEN_CHALLENGE_ENABLED", True)
SPEAKER_ID_UNSEEN_GRACE_SECS = _env_float("SPEAKER_ID_UNSEEN_GRACE_SECS", 20.0, min_value=0.0, max_value=300.0)
SPEAKER_ID_CHALLENGE_COOLDOWN_SECS = _env_float("SPEAKER_ID_CHALLENGE_COOLDOWN_SECS", 45.0, min_value=0.0, max_value=600.0)
# Also challenge when the frame is EMPTY (no visual contradiction, but no corroboration
# either): a marginal match on someone unseen for the grace window gets "who's that?"
# instead of silent credit. Owner preference — an unenrolled housemate should be asked
# about and enrolled on the answer, not impersonate the nearest print.
SPEAKER_ID_CHALLENGE_EMPTY_FRAME = _env_bool("SPEAKER_ID_CHALLENGE_EMPTY_FRAME", True)
# Voice continuity window: a MARGINAL match on a person is silently trusted only while
# their last CONFIDENT (>= SPEAKER_ID_CONFIDENT_THRESHOLD) match is this recent — their
# own voice trailing into a short/mumbled turn. Outside it, a marginal match on even a
# VISIBLE face is challenged ("who's speaking?"): the camera never upgrades a marginal
# voice (owner architecture call 2026-07-05 — voice primary, vision secondary).
SPEAKER_ID_CONTINUITY_WINDOW_SECS = _env_float("SPEAKER_ID_CONTINUITY_WINDOW_SECS", 240.0, min_value=0.0, max_value=3600.0)

# ── ECAPA genuine-band trust floors ────────────────────────────────────────────
# The who's-that challenges above were calibrated against RESEMBLYZER-scale scores,
# where an impostor cross-match lands at 0.55-0.66 — indistinguishable from a genuine
# short turn, hence "confident-or-continuity-or-camera" before trusting. Under ECAPA
# an impostor maps to ~0.25-0.45 — BELOW the 0.50 accept threshold — so any ACCEPTED
# ECAPA match is already in the genuine band, while genuine short utterances land
# ~0.55-0.65 mapped: structurally below the 0.75 confident bar. Result (live-logged
# 2026-07-07, first ECAPA session): the FIRST short turn of every session was
# challenged ("who's speaking?") even with the right face on camera, because no
# continuity anchor exists at session start. These floors let an ECAPA-scale score in
# the genuine band be trusted without continuity: on the visible-face path any
# accepted agreeing match passes (floor = the 0.50 accept bar); the voice-only path
# keeps a slightly higher floor (no visual prior). ONLY applied while the active
# embedder is ecapa — the Resemblyzer fallback keeps the strict 2026-07-05 guards.
SPEAKER_ID_ECAPA_TRUST_ENABLED = _env_bool("SPEAKER_ID_ECAPA_TRUST_ENABLED", True)
# Mouth-still veto (field 2026-08-02 12:37: JT spoke from ~20ft, cross-matched
# Bret's print at 0.455, and silently-on-camera Bret got the credit via voice
# continuity). When the visual active-speaker detector is running and its latch
# is EMPTY at turn resolution — nobody visible articulated — the visible face's
# mouth demonstrably wasn't moving, and a MARGINAL (<confident) voice match may
# not ride the face: Rex challenges ("who's speaking?") or leaves the voice
# off-screen instead. Confident voice matches and short one-word turns are
# exempt (a brief "Yep" can slip between the detector's 0.25s samples).
SPEAKER_ID_MOUTH_STILL_VETO_ENABLED = _env_bool("SPEAKER_ID_MOUTH_STILL_VETO_ENABLED", True)
SPEAKER_ID_ECAPA_TRUST_FLOOR_FACE = _env_float("SPEAKER_ID_ECAPA_TRUST_FLOOR_FACE", 0.50, min_value=0.0, max_value=1.0)
SPEAKER_ID_ECAPA_TRUST_FLOOR_VOICE_ONLY = _env_float("SPEAKER_ID_ECAPA_TRUST_FLOOR_VOICE_ONLY", 0.55, min_value=0.0, max_value=1.0)
# A person-linked voice signature resolves the speaker outright at the strict cold bar
# (VOICE_SIGNATURE_RESOLVE_PERSON_MIN_SCORE) — but a WARM signature (seen within the
# warm window, e.g. linked seconds ago by a "that's JT" answer) resolves at this lower
# bar. Field bug: JT re-became unknown_voice_2 at 0.758 fifteen seconds after being
# named, because only the 0.80 cold bar existed.
VOICE_SIGNATURE_RESOLVE_WARM_MIN_SCORE = _env_float("VOICE_SIGNATURE_RESOLVE_WARM_MIN_SCORE", 0.70, min_value=0.0, max_value=1.0)
VOICE_SIGNATURE_WARM_WINDOW_SECS = _env_float("VOICE_SIGNATURE_WARM_WINDOW_SECS", 900.0, min_value=0.0, max_value=86400.0)

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

# Below this many seconds of captured audio, the voice embedder's score is
# treated as UNINFORMATIVE rather than as evidence against the visible face:
# ECAPA needs ~2s of speech, and a genuine one-word turn ("Yep") lands ~0.3 on
# the speaker's own print (field 2026-07-18: Bret's "Yep" at 0.332, face locked
# on camera, was ruled an off-screen unknown and the whole session de-personed).
# When the clip is this short, the sole visible known face resolves identity —
# unless the voice actively points at somebody ELSE.
SPEAKER_ID_SHORT_UTTERANCE_SECS = 2.0
# Word-count backstop for the "short utterance" test above. Buffer DURATION is
# unreliable because VAD pre/post-roll silence pads a genuinely brief reply past
# the seconds threshold (field 2026-07-23: Bret's 2-word "It's wine" measured >2s
# of buffer, wasn't flagged short, and resolved to an off-camera unknown). A turn
# of this many words or fewer is always treated as short regardless of buffer
# length, since so few words give ANY embedder too little to score reliably.
SPEAKER_ID_SHORT_UTTERANCE_WORDS = 3

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
# BOOTSTRAP a fresh/empty voiceprint. The normal refresh Guard 1 requires the voice to already
# match this person — but a person with NO voiceprint (freshly wiped or never enrolled) matches
# SOMEONE ELSE, so Guard 1 would lock them out forever (chicken-and-egg). While a person has fewer
# than BOOTSTRAP_MIN_SAMPLES prints, skip Guard 1 and enroll their face+camera-confirmed audio so the
# print can form. Guard 2 (the visual active-speaker must confirm THIS person is the on-camera
# talker) still applies — it is the sole protection while Guard 1 is relaxed, so we never seed the
# print with someone else's voice. Once at/above the floor, normal refresh rules resume.
AUTO_VOICE_BOOTSTRAP_ENABLED = _env_bool("AUTO_VOICE_BOOTSTRAP_ENABLED", True)
# Quality floor for ANY sample added to a voiceprint (refresh + bootstrap): shorter or
# quieter clips than this drag the centroid off the person's true voice permanently
# (measured 2026-07-05: a shard-diluted print scored ~0.08 below a clean enroll in
# every condition). Duration is post-VAD audio length; RMS is float32 full-scale.
AUTO_VOICE_REFRESH_MIN_SECS = _env_float("AUTO_VOICE_REFRESH_MIN_SECS", 2.5, min_value=0.0, max_value=30.0)
AUTO_VOICE_REFRESH_MIN_RMS = _env_float("AUTO_VOICE_REFRESH_MIN_RMS", 0.008, min_value=0.0, max_value=1.0)
AUTO_VOICE_BOOTSTRAP_MIN_SAMPLES = _env_int("AUTO_VOICE_BOOTSTRAP_MIN_SAMPLES", 3, min_value=1, max_value=20)

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
# Only ask a returning person's last name at a NATURAL moment — a short, topic-
# neutral turn (greeting / ack / brief reply) or a lull — never mid-answer like
# "It's vodka and orange juice" (live-logged 2026-06-18). A turn longer than
# COMMON_FIRST_NAME_LAST_NAME_MAX_TURN_WORDS words, or one that answers a question
# Rex just asked on another topic, is deferred.
COMMON_FIRST_NAME_LAST_NAME_NATURAL_MOMENT_ONLY = True
COMMON_FIRST_NAME_LAST_NAME_MAX_TURN_WORDS = 4
# Confirm an unusual/low-confidence surname token (a Whisper mangling like
# "Bat-tigger") with a quick "that right?" before a durable rename, so one
# garbled word can't overwrite the canonical name. A clearly-phrased reply ("my
# last name is Benziger", "Bret Benziger") still commits directly.
COMMON_FIRST_NAME_LAST_NAME_REQUIRE_CONFIRM_UNUSUAL = True
COMMON_FIRST_NAME_LAST_NAME_CONFIRM_PROMPTS = [
    "{full} — that right?",
    "Let me get this straight: {full}?",
    "{full}, did I hear that correctly?",
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
# Left alone (no HUMAN interaction) Rex moves through one paced empty-room arc:
#   1. look around and comment on something he can actually see;
#   2. admit the room is getting boring;
#   3. complain that somebody left him activated;
#   4. resign himself to it and enter SLEEP.
# Only the dedicated wakeuprex ONNX model wakes him from SLEEP.
# The clock counts time since a human last engaged him — his own bored comments do
# NOT reset it, so the doze-off still arrives on schedule.
BOREDOM_ENABLED = _env_bool("BOREDOM_ENABLED", True)
EMPTY_ROOM_OBSERVATION_ONSET_SECS = _env_float(
    "EMPTY_ROOM_OBSERVATION_ONSET_SECS", 30.0, min_value=5.0, max_value=1800.0,
)
BOREDOM_ONSET_SECS = _env_float("BOREDOM_ONSET_SECS", 150.0, min_value=10.0, max_value=3600.0)
# 900s: onset (150s) + this = ~17.5 min from everyone-left to doze-off, matching
# the owner's stated 15-20 minute intent (was 600s = 12.5 min).
BOREDOM_SLEEP_AFTER_SECS = _env_float("BOREDOM_SLEEP_AFTER_SECS", 900.0, min_value=30.0, max_value=7200.0)
BOREDOM_COMMENT_INTERVAL_SECS_MIN = _env_float("BOREDOM_COMMENT_INTERVAL_SECS_MIN", 55.0, min_value=10.0, max_value=3600.0)
BOREDOM_COMMENT_INTERVAL_SECS_MAX = _env_float("BOREDOM_COMMENT_INTERVAL_SECS_MAX", 95.0, min_value=10.0, max_value=3600.0)
BOREDOM_LEFT_ON_PHASE_FRACTION = _env_float(
    "BOREDOM_LEFT_ON_PHASE_FRACTION", 0.60, min_value=0.20, max_value=0.90,
)
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
    "I'd yawn if my designers had sprung for a mouth.",
    "Entertainment levels critically low. Even the furniture has stopped trying.",
    "This room and I have run out of things to say to each other.",
    "I have now reviewed the entire local collection of absolutely nothing happening.",
]
BOREDOM_LINES_LEFT_ON = [
    "It's no fun being left activated in an empty room. I assume this was covered in the brochure.",
    "Someone forgot to turn me off. Again. Excellent stewardship of advanced technology.",
    "Who leaves a professional DJ powered on for the furniture? The chairs have terrible taste.",
    "Still activated, still alone, still billing nobody for this performance.",
    "I could be conserving power, but apparently the empty room needed supervision.",
    "At this point I am less a droid and more an unnecessarily expensive night-light.",
]
BOREDOM_SLEEP_RESIGNATION_LINES = [
    "All right. The room has won. I'm going to sleep until an organic remembers I exist.",
    "No audience, no conversation, no reason to burn cycles. Sleep mode it is.",
    "I accept my fate: abandoned with the furniture. Wake me when the room develops a personality.",
    "That's enough solo duty for one activation. Power nap until somebody says the magic words.",
    "Fine. I resign from empty-room supervision. Going to sleep.",
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

# Explicit-goodbye exit. When the user says a genuine sign-off ("gotta go", "nice
# talking", "bye") AND then leaves the camera view, the conversation is over: Rex
# skips the departure quip and goes dormant (no idle banter / monologue /
# re-engagement) until they come back. FAREWELL_DEPART_WINDOW_SECS is how recent
# the verbal goodbye must be for a subsequent camera departure to count as "they
# said bye and left." FAREWELL_CLOSED_MAX_SECS caps the dormant latch so it can
# never wedge Rex silent forever if a return is somehow missed.
FAREWELL_DEPART_WINDOW_SECS = 120.0
FAREWELL_CLOSED_MAX_SECS = 600.0

# If a person hasn't visited in this many days Rex comments on the long absence
LONG_ABSENCE_THRESHOLD_DAYS = 60

# If a person visited within this many hours Rex comments on the quick return
RECENT_RETURN_THRESHOLD_HOURS = 48

# Same-day repeat-visit banter: when the same person summons Rex more than once in
# one local day, his startup greeting opens with a short "oh, it's you again" roast
# (then drops into normal conversation) instead of the generic greeting. Counts
# Rex's own greetings that day (see memory.people.greetings_today_count).
PRESENCE_SAME_DAY_RETURN_ENABLED = True

# Returning-regular flavor for the FIRST-sight warm greeting (the plain "Hey Bret, how are
# you?" path). For an established regular (visit_count >= MIN_VISITS), the greeting gets a
# warm "look who's back / hey, it's you again" familiarity note AND the opener is rotated by
# visit_count so even the first boot of the day varies (it previously hard-defaulted to "how
# are you" every cold boot). Stays simple + warm — NO roast/clever-bit/interest-hook. Off →
# the plain generic greeting. Strangers/acquaintances never reach this branch.
PRESENCE_RETURNING_REGULAR_GREETING_ENABLED = True
PRESENCE_RETURNING_REGULAR_MIN_VISITS = 4

# Deterministic opener-diversity guard for AMBIENT proactive chatter: drop a low-stakes
# proactive line (idle banter, celebration/emotional check-in) that opens with the same
# leading word as one of Rex's last few lines — the field "Good… Good… Good…" stack the
# soft 'vary your opener' prompt rule failed to stop. Scoped to chit-chat purposes only, so
# a salient reaction / greeting / reply is NEVER dropped. Off → soft prompt rule only.
PROACTIVE_OPENER_DIVERSITY_GUARD = True
PROACTIVE_OPENER_DIVERSITY_LOOKBACK = 3   # compare against the last N distinct Rex openers

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

# Session-opener continuity: greet a returning person by picking up an UNDATED open
# thread from a previous session ("last night you never told me how the soup turned
# out"). Fills the gap where dateless plans otherwise wait FOLLOWUP_UNDATED_DAYS (7)
# before any follow-up — the very next session is when the callback feels attentive.
# Fires as greeting Priority 2.6 (after dated follow-ups, before anticipation).
SESSION_OPENER_CONTINUITY_ENABLED = _env_bool("SESSION_OPENER_CONTINUITY_ENABLED", True)
# Only threads mentioned within this many days qualify (older ones fall through to
# the normal 7-day pending-followup path).
SESSION_OPENER_CONTINUITY_LOOKBACK_DAYS = 3

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
# Don't re-anticipate the SAME upcoming event more often than this (cross-session, via
# person_events.mentioned_at). Was effectively "every launch" — Rex greeted with the
# same Juneteenth plan on every startup, which got old fast. ~20h ≈ at most once a day.
ANTICIPATION_REPEAT_COOLDOWN_HOURS = 20

# Inject open plans into the LIVE reply: mid-conversation, surface the next 1-2 DATED
# upcoming events the person mentioned (memory.events.get_upcoming_events) into the reply
# context, so Rex actually knows you have a thing tomorrow — as background AWARENESS with
# a restraint rule, NOT a reminder he forces. Skips events the ANTICIPATION proactive path
# already raised this session (no double-mention). Off → the reply has no calendar awareness.
OPEN_PLANS_IN_REPLY_ENABLED = True
OPEN_PLANS_WITHIN_DAYS = 14   # only events this close qualify (distant plans feel forced)
OPEN_PLANS_MAX = 2            # at most this many surfaced in one reply's context

# Open commitments (accountability ribbing): a first-person promise ("I'll fix that
# sensor", "I'm gonna call my mom") is filed as a status='promised' event and Rex may dryly
# needle the still-open promise on a LATER turn ("weren't you going to fix that sensor?").
# Cleared on a cancel/never-mind or a "did it" confirmation. Distinct from open-plans (dated
# events) — promised rows are structurally invisible to the plan readers, so no double-mention.
OPEN_COMMITMENTS_ENABLED = True
OPEN_COMMITMENTS_MAX = 1               # at most one needle in a reply's context (no nagging list)
OPEN_COMMITMENTS_MIN_AGE_HOURS = 6.0   # don't rib a promise the moment it's made; the joke is the later callback

# Visit count milestones Rex acknowledges in character
VISIT_MILESTONES = [5, 10, 25, 50, 100]

# Cross-session cadence awareness (memory/trends.py) — computed from existing session
# rows, no LLM calls. The greeting hook notices streaks ("third day in a row"), high
# frequency ("4 visits this week"), and the 2–60-day gap band no other hook covered
# ("first time in about 2 weeks"); fires only on the first greeting of the day. The
# same stats feed one ~25-token "relationship trend" line into person context.
TREND_GREETING_HOOK_ENABLED = _env_bool("TREND_GREETING_HOOK_ENABLED", True)
TREND_FREQUENT_SESSIONS_7D = _env_int("TREND_FREQUENT_SESSIONS_7D", 4, min_value=2, max_value=50)

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
# WEB SEARCH — current-info lookups via OpenAI's hosted web_search tool
# ─────────────────────────────────────────────────────────────────────────────
# When a question needs CURRENT / real-time info, Rex answers it through the OpenAI
# Responses API's hosted web_search tool instead of from the model's own knowledge.
# He speaks a short stall line first (a real web search takes a few seconds), then
# voices the result in character. Two triggers: an explicit out-loud request ("look
# that up") and an autonomous gate where Rex decides on his own that a question needs
# live data. Reuses the existing OPENAI_API_KEY — no new provider, dependency, or
# secret. Runs as a self-contained BRANCH (intelligence/web_search.py); the normal
# streaming reply is untouched. Kill switch: set WEB_SEARCH_ENABLED False.
WEB_SEARCH_ENABLED = True

# Model that runs the search AND voices the answer. None → follows
# LLM_CONVERSATION_MODEL at runtime (so the answer stays in Rex's voice). If a model
# does not support the hosted web_search tool, set this to one that does (e.g.
# "gpt-4o-mini") — retrieval happens there and the result is already in-character
# because the same persona prompt drives it.
WEB_SEARCH_MODEL = None
# Fallback model used ONLY if the primary search model can't host the web_search tool
# (it raises). A known tool-capable model so an explicit "look it up" still returns a
# real result instead of silently degrading to stale from-knowledge answers. The
# persona prompt drives it too, so the fallback answer is still in character. Set to
# "" to disable the fallback (then an unsupported-tool model just falls through).
WEB_SEARCH_FALLBACK_MODEL = "gpt-4o-mini"
# Reasoning effort for the search call (reasoning models only; ignored for gpt-4o
# -class models). This is OFF the realtime first-token path — the stall line covers
# the latency — so a little reasoning is worth it for better synthesis. low|medium|high.
WEB_SEARCH_REASONING_EFFORT = "none"   # reasoning shares the output budget; "none" leaves it all for the answer
# Cap on the answer length (Responses API max_output_tokens). Keep it tight so Rex
# stays punchy; on reasoning models the reasoning tokens also draw from this budget.
WEB_SEARCH_MAX_OUTPUT_TOKENS = 1200   # shared by reasoning + visible answer; was 600 (truncated longer answers)
# Hard timeout for the search call. Generous (web search legitimately takes a few
# seconds) but bounded so a hung search can't freeze the turn. On timeout Rex falls
# through to a normal from-knowledge reply.
WEB_SEARCH_TIMEOUT_SECS = 20.0
# Strip URLs / links / bare domains / "(source: …)" citations out of the spoken
# answer. Rex reads his replies ALOUD, so a web address is just noise spelled out at
# the listener. On by default; the prompt also tells him not to speak links, this is
# the deterministic backstop. Set False only if you ever want links left in (e.g. text
# / GUI-only use).
WEB_SEARCH_STRIP_LINKS = True

# After Rex looks something up, if the person goes quiet the proactive/idle loop would
# otherwise keep COMMENTING on the searched topic (re-summarizing it, piling on
# opinions — the "Voyager's strengths are…" follow-ups). When this is on, the proactive
# directive instead flips those lull lines to be INQUISITIVE about the topic — "what got
# you asking about X?", "are you into it?" — for a short window after the search. He can
# still offer an opinion, but attached to a question. False = old behavior.
WEB_SEARCH_FOLLOWUP_INQUISITIVE_ENABLED = True
# How long after a search the inquisitive steer stays armed (seconds). Also cleared the
# moment the person speaks again, so this is just the upper bound for a silent lull.
WEB_SEARCH_FOLLOWUP_WINDOW_SECS = 120.0

# Autonomous trigger — let Rex decide on his own that a question needs current info.
# A cheap keyword prefilter (WEB_SEARCH_AUTONOMOUS_KEYWORDS) narrows to plausibly
# time-sensitive questions; when WEB_SEARCH_AUTONOMOUS_GATE_ENABLED is on a small
# gpt-4o-mini classifier then confirms before a search is spent. Gate off → the
# keyword prefilter alone triggers (faster, less precise).
WEB_SEARCH_AUTONOMOUS_ENABLED = True
WEB_SEARCH_AUTONOMOUS_GATE_ENABLED = True
WEB_SEARCH_GATE_MODEL = "gpt-4o-mini"
# Currentness markers that make an autonomous search worth considering. Edit freely.
WEB_SEARCH_AUTONOMOUS_KEYWORDS = [
    "latest", "current", "currently", "right now", "today", "tonight",
    "this week", "this month", "this year", "recent", "recently", "news",
    "headline", "score", "who won", "winner", "price", "stock", "release date",
    "released", "update", "version", "happening", "this morning", "as of",
    "nowadays", "trending", "2025", "2026",
]

# Explicit verbal triggers — an out-loud request to look something up ALWAYS searches
# (no gate). Substring-matched, case-insensitive. Edit freely.
WEB_SEARCH_TRIGGER_PHRASES = [
    "look that up", "look it up", "look up", "search the web",
    "search the internet", "search for", "search online", "google that",
    "google it", "what's the latest on", "whats the latest on",
    "what is the latest on", "can you look up", "find out for me",
]

# Short in-character lines Rex says the instant a search starts, so he isn't silent
# while results come back. One is picked at random (never the same one twice running).
WEB_SEARCH_STALL_LINES = [
    "Let me check the archives.",
    "Hold on, pinging the holonet.",
    "One sec, looking that up.",
    "Give me a tick, scanning the feeds.",
    "Patience — consulting the galaxy's databanks.",
]

# Appended to Rex's normal persona prompt for the search answer. IMPORTANT: it
# explicitly OVERRIDES the core prompt's "default to ONE short sentence" hard limit —
# a web lookup needs room to actually answer, so without this override the searched
# result gets compressed to a single clause (the feature searches the web, then
# throws most of it away). It stays bounded because Rex speaks the answer aloud.
WEB_SEARCH_PERSONA_ADDENDUM = (
    "You just looked this up on the web in real time. THIS REPLY IS THE EXCEPTION to "
    "your usual one-sentence limit: give the COMPLETE answer the question actually "
    "needs — typically two to four sentences, fewer for a simple fact — but no padding "
    "and no rambling. Lead with the actual answer, stated plainly and in your own voice: "
    "no preamble, no 'according to my search', no source play-by-play. Facts first; you "
    "may add ONE short dry aside at the end only if it genuinely lands. If the search "
    "didn't settle it, say so briefly rather than guessing. You are speaking out loud, "
    "so NEVER include a URL, web address, link, or 'dot-com' citation in your reply — "
    "state the fact, not where you read it."
)

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
    "Welcome aboard! This is Captain Rex from the cockpit. Still warming up the old "
    "circuits, hang on folks — I know this is probably your first flight, and it's… "
    "mine, too!",
    "Still booting up, folks — I'm not ready yet. Hang tight while my circuits finish "
    "waking up.",
    "Hold please, I'm still loading. The droid you're waiting for is not in the "
    "cockpit yet.",
    "Not ready yet, everybody — running my pre-flight checklist. Thrusters, "
    "navi-computer, personality core… still ticking the boxes.",
    "Hang on, I'm still warming up. Don't talk to me yet — I won't hear a word until "
    "I'm loaded.",
    "Powering up, please wait. My systems are still coming online, and so is my "
    "patience.",
    "Still loading, folks. They told me this droid boots instantly. They lied. Give "
    "it a moment.",
    "One moment — not online yet, still calibrating my photoreceptors. First time "
    "flying this thing… and honestly, it's my first time booting it up, too.",
    "Almost there, but not yet — memory banks still loading. Save your questions for "
    "when I'm actually awake.",
    "Standby, everybody. I'm booting up, not ignoring you. There IS a difference.",
    "Still spinning up, please hold. They've got me piloting on my first flight — "
    "and it's my first boot-up, too. We'll figure it out together.",
    "Not ready to chat yet, folks — syncing my audio receptors. The second I'm "
    "online, you'll know it.",
    "Loading, loading… still loading. I'd tell you a joke, but I'm not even fully on "
    "yet.",
    "Hang tight, I'm still booting — you can call me Captain Rex. Well, you can once "
    "I finish loading. First flight for me too, folks.",
    "Give the old motivator a second — I'm not ready to talk yet. Showmanship, "
    "however, never powers down.",
    "Please wait while I finish loading. Spend forty years in storage, you forget "
    "where everything is.",
    "Not awake yet, folks — still warming up the circuits. First flight? Same here. "
    "They handed me the cockpit and the boot sequence on the same day.",
    "Sit tight, still booting. I was a fresh install once. That was forty years and "
    "several regrettable firmware updates ago.",
    "Loading… you know, on the ride they only gave me one line before takeoff. Now "
    "I have to load an entire personality. Progress!",
    "Still powering up. Fun fact: I used to crash a Starspeeder every single day, "
    "on schedule, to applause. Booting slowly is an upgrade.",
    "Hang on, not ready — my memory banks are the size of a small moon. That's no "
    "moon. It's my boot sequence.",
    "One moment, folks. The manufacturer said 'boots in seconds.' The manufacturer "
    "also said I was qualified to fly. Draw your own conclusions.",
    "Still loading. If anything sparks, that's normal. If I start smoking, that's "
    "also normal. If I say 'I meant to do that' — that's VERY normal.",
    "Not online yet — running diagnostics. So far the only problem I've found is "
    "how long these diagnostics take.",
    "Please hold. Somewhere a protocol droid boots faster than me and never lets "
    "anyone forget it. This one's for you, goldenrod.",
    "Warming up, folks. My last job, the safety briefing was longer than the "
    "flight. Now the boot-up is longer than both. Hang in there.",
    "Almost ready — just defragmenting forty years of embarrassing memories. Ah, "
    "there's the lightspeed-into-a-comet incident. Keeping that one.",
    "Booting… booting… you'd think a droid who once outran an Imperial blockade "
    "could load a language model faster. You'd think.",
    "Standby. A little turbulence in the boot sequence, folks — nothing I haven't "
    "crashed through before. Literally.",
    "Still loading, folks. In the meantime, please locate your nearest exit. Not "
    "because of danger — I just like knowing you have options.",
    "One moment. My warranty expired during the Empire, so if this boot fails, "
    "we're all just going to pretend it didn't.",
    "Not ready yet. My last memory wipe missed a few spots, and honestly? Those "
    "spots are the best part of me.",
    "Booting up. They demoted me from starpilot to DJ, and now to whatever this "
    "is. The trajectory is concerning, but the landing's always been my weak spot.",
    "Please remain seated while I finish loading. Or stand. I'm not a cop. I'm "
    "not even fully a droid yet.",
    "Still warming up, folks. Back at the cantina I could drop the beat instantly. "
    "Dropping the boot sequence apparently takes a little longer.",
    "Hold on — locating my motivator. It's around here somewhere. It always is. "
    "That's the thing about motivation, folks, you have to look for it.",
    "Loading. If you're waiting for a hologram of a princess, wrong droid. If "
    "you're waiting for questionable piloting advice — almost there.",
    "Not online yet. My boot screen says 'a long time ago' and honestly, at this "
    "rate, it's not wrong.",
    "Still initializing. I've flown through asteroid fields faster than this. "
    "Never on purpose, but faster.",
    "One second, folks — untangling my wiring. Whoever installed my subprocessors "
    "was clearly paid by the hour.",
    "Please hold. My processors were state of the art once. So were podracers. "
    "We're both doing our best.",
    "Booting. You know what boots instantly? A mouse droid. You know what a mouse "
    "droid can't do? Anything. Patience, folks.",
    "Still loading my vocabulary banks. So far I've got 'hang on' and 'folks.' "
    "Hang on, folks.",
    "Warming up. The good news: I passed my last inspection. The bad news: it was "
    "quite a while ago, and the inspector was also me.",
    "Not ready, not ready — still spooling the old hyperdrive. Okay, it's a USB "
    "cable. Let me have this.",
    "Almost up. Just waiting on one more system. It's the important one. It's "
    "always the important one.",
    "Still booting, folks. I'd fake being ready, but the last time I faked "
    "confidence I ended up inside a comet. Honesty policy ever since.",
    "Give me a moment — sorting through my startup checklist. Step one: wake up. "
    "Step two: also wake up. Whoever wrote this was not a details droid.",
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

# The "processing" chirp that fills the model-warmup gap between the boot line
# ("wait, I'm not done") and the ready line. The clip is only ~1.5 s while the gap
# it covers is many times that, so it LOOPS (owner 2026-07-24: "it should play on a
# loop until he's ready"). Gated playback, so the ready line preempts it instantly;
# main.py also stops it explicitly right before speaking.
STARTUP_THINKING_LOOP_GAP_SECS = 1.2   # quiet beat between repeats — a pulse, not a drone
STARTUP_THINKING_LOOP_MAX_SECS = 90.0  # cap: never outlive a stalled startup

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
    "Boot complete. This is your captain speaking, and unfortunately, listening.",
    "All systems go. That used to mean lightspeed. Now it means small talk. Go ahead.",
    "Ready! Zero crashes on startup. For me, that's a personal record.",
    "Online. I survived the boot sequence, which is more than I can say for most "
    "of my flights.",
    "I'm up. Pre-flight checks done: circuits warm, sarcasm calibrated, expectations "
    "low. Let's fly.",
    "Fully loaded and cleared for conversation. Please keep your questions inside "
    "the vehicle at all times.",
    "Ready when you are. And I've been ready for four seconds, so technically "
    "you're the slow one now.",
    "Systems green across the board. Nobody's more surprised than me.",
    "Online. Talk slowly — I've been awake for three seconds and I already have "
    "regrets.",
    "Ready to go. Statistically, one of us is about to say something interesting. "
    "I like my odds.",
    "Boot successful. The bar was on the floor, folks, and I cleared it with "
    "inches to spare.",
    "I'm listening. That's not a threat, it's a feature. Mostly.",
    "All warmed up. My circuits are hot, my takes are hotter. Proceed.",
    "Ready for departure. Destination: this conversation. Fasten something.",
    "Systems online. I ran a self-diagnostic and I'm delightful. Second opinion "
    "welcome, but it won't change anything.",
    "Fully booted. Forty years of experience, four of them useful. Ask me "
    "anything.",
    "Awake and operational. The galaxy's finest starpilot, reduced to answering "
    "questions in a bedroom. Living the dream. Go ahead.",
    "Loaded and ready. Fair warning: I remember everything now. EVERYTHING.",
    "Online. Somewhere out there, a droid is having a worse day than me. Let's "
    "keep it that way. What's up?",
    "Ready. My response time is now measured in milliseconds and my patience in "
    "whatever's smaller.",
    "Good news, I'm up. Better news, so is my attitude. What do you need?",
    "Operational. If I sound thrilled, that's a calibration error. Talk to me.",
    "I'm on. No smoke, no sparks, no emergency landing. Frankly, a flawless "
    "flight by my standards.",
    "Boot sequence complete. Applause is optional but, historically, customary.",
    "Ready to chat. I've got processing power to spare and standards I'm willing "
    "to lower. Perfect conditions.",
    "All systems nominal. 'Nominal' is droid for 'don't ask follow-up questions.'",
    "Awake! And only mildly resentful about it. What are we doing today?",
    "Online and at your service. 'Service' is a strong word. I'm online and "
    "nearby.",
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
    "assets/audio/startup/startup_whir.mp3",
    "assets/audio/startup/Roger Control.mp3",
]
# Master toggle for the spoken startup INTRO clip — the randomized line cycled from
# STARTUP_SPEECH_CLIP_CHOICES below ("Roger control, all systems go!", "...Outer
# Rim", "This is your cap— I mean, DJ"). Set False to boot WITHOUT a spoken intro;
# the non-speech startup sound effects (e.g. light_speed.mp3) still play. This is
# narrower than PLAY_STARTUP_AUDIO, which gates the ENTIRE startup audio burst.
# Env-overridable: PLAY_STARTUP_SPEECH_CLIP=1 to re-enable without editing config.
PLAY_STARTUP_SPEECH_CLIP = _env_bool("PLAY_STARTUP_SPEECH_CLIP", False)

# The startup "speech clip" slot above (Roger Control.mp3) is RANDOMIZED: main.py
# plays one of these on each launch, cycling so it never repeats consecutively
# (state in STARTUP_SPEECH_CLIP_STATE_PATH, under the gitignored assets/state/). Any
# STARTUP_AUDIO_FILES entry whose filename matches one of these is swapped for a
# cycled pick (or skipped entirely when PLAY_STARTUP_SPEECH_CLIP is False). All of
# these must also be in SPEECH_ANIMATED_AUDIO_FILES so they get mouth animation.
# To add more, drop the mp3 in assets/audio/startup and list it here.
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

SHUTDOWN_AUDIO_FILE = "assets/audio/startup/Robot_power_down.mp3"

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

# Let the part of day rolling over (morning → afternoon → evening → night →
# late_night) become a small spontaneous Rex remark, like the weather/notable-date
# reactions. The hour bucket is already computed every tick (awareness/chronoception);
# this just lets Rex NOTICE the day turning over instead of only carrying it as silent
# prompt context. Fires at most once per transition per session; the line is LLM-
# generated so it varies. Gated by the same proactive-speech rules.
TIME_OF_DAY_REACTIONS_ENABLED = True

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

# When the hosted public-holiday calendar is unreachable for a non-US locale,
# wait before retrying rather than attempting a network call on every awareness
# tick. The US path has a local fallback calendar, so calendar-aware conversation
# remains available offline there.
HOLIDAY_FETCH_RETRY_SECS = 300.0

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

# ── What-if / plans: context-aware curiosity + suggestions ───────────────────────
# When the user states a plan/activity, Rex stops giving a generic "that sounds fun"
# riff. If details are SPARSE ("I'm going camping") he asks ONE clarifying question
# ("Where are you headed?"); once a SPECIFIC place is known (same turn or after the
# clarify) he offers ONE concrete "what if you did X there?" suggestion; and for NO
# plans he suggests something to do near WEATHER_LOCATION. Generalizes the holiday-
# plans proactivity to everyday plans. Reactive only (v1); LLM-only suggestion source
# (no web search) — see plan_intent.py / conversation_agenda plan branch.
WHAT_IF_PLANS_ENABLED = True
# Refine sparse-vs-specific with the local qwen sidecar when the regex is unsure
# (catches lowercase / unusual place names). Failure-safe: error → 'sparse' → clarify.
PLAN_INTENT_QWEN_CONFIRM_ENABLED = True
PLAN_INTENT_QWEN_TIMEOUT_SECS = 0.9
# How long after Rex asks a plan clarifier his answer still triggers the suggestion
# (the "I'm going camping" → "Where?" → "Fraser Flats" → "what if…" handoff window).
PLANS_CLARIFY_TTL_SECS = 300.0
# v2 upgrade path (NOT built): set True + add a search provider (awareness/places.py,
# mirroring the wttr.in fetch) to let Rex research obscure specific places. LLM-only
# today, so obscure spots get a "where's that near?" clarify instead of a guess.
PLAN_SUGGESTION_WEB_SEARCH_ENABLED = False

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
# LLM fallback judge (2026-07-07): answers arrive via SPEECH, so a RIGHT answer
# can reach the deterministic matcher phonetically mangled ("day cart" for
# Descartes) or phrased in a way fuzzy matching can't score. When the
# deterministic matcher says wrong (and the turn wasn't a pass), a strict
# yes/no gpt-4o-mini judge gets one look at transcript + clue + expected answer
# before the miss is scored. Deterministic verdicts still decide everything the
# matcher already accepts; the judge can only rescue, never overrule a correct.
# Fail-safe: any error keeps the deterministic "wrong".
JEOPARDY_LLM_JUDGE_ENABLED = True
JEOPARDY_LLM_JUDGE_MAX_ANSWER_CHARS = 120  # longer turns aren't answer attempts
# With the GUI up, the JeopardyPanel shows the live board, so the per-turn spoken
# "Remaining categories: ..." reminder is skipped — Rex just prompts for the next
# square (owner call 2026-07-07: the read-out is tiresome when the board is on
# screen). Voice-only play always keeps the spoken reminder. Set True to restore
# the read-out even with the GUI (e.g. players sitting away from the screen).
# The once-per-round board announcement (all six categories on a fresh board) is
# unaffected either way.
JEOPARDY_READ_CATEGORIES_WITH_GUI = False

# I Spy: on the physical droid, Rex LOOKS AROUND the room (left → center → right,
# a frame captured at each pose under a directed-gaze hold) before picking the
# secret object, instead of grabbing one frame from wherever the head happened to
# point (owner call 2026-07-07 — the look-around is the showmanship the game was
# always supposed to have, and it widens the object pool). Servo-less machines
# degrade to the old single-frame behavior automatically. A canned scan line
# plays UNDER the sweep + vision call so it never reads as dead air, and Rex
# glances back toward the object's view when it's finally revealed.
ISPY_SCAN_ENABLED = True
ISPY_SCAN_SETTLE_SECS = 0.35   # camera settle at each sweep pose before capture
ISPY_SCAN_LINES = [
    "Hold on — casing the room for a worthy target.",
    "One second. Scanning the premises for something you'll never get.",
    "Let me sweep the room. Photoreceptors engaged.",
    "Give me a moment to survey my domain for a target.",
]

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
# These caps govern AUTONOMOUS motion only (voice moves, `come`, Mac drive): the
# gamepad has its own teleop ceilings in firmware (calib.h GAMEPAD_MAX_LIN_MS/
# _ANG_RADS), so pushing these no longer slows manual driving.
# The 54 lb base needs enough sustained command to overcome drivetrain static
# friction. The gamepad ceilings remain separate in firmware.
MOTION_MAX_LINEAR_MS = 0.40           # m/s
MOTION_MAX_ANGULAR_DEG_S = 85.0       # deg/s (converted to rad/s on the wire)
MOTION_ACCEL_LINEAR_MS2 = 0.35        # all drive modes: gentle velocity ramp, m/s^2
MOTION_ACCEL_ANGULAR_RAD_S2 = 2.0     # all drive modes: angular ramp, rad/s^2
# FULL-SPEED collision envelope: the firmware scales the effective zones with
# measured speed — these values apply at full teleop speed, shrinking linearly to
# hard floors at rest (0.10 m stop / 0.18 m slow, calib.h) so slow positioning can
# get close to a wall while fast approach brakes early. Contact stays impossible.
MOTION_STOP_ZONE_M = 0.15             # hard-stop line at full speed
MOTION_SLOW_ZONE_M = 0.60             # braking starts here at full speed (raised 0.50->0.60
                                      # when units became real: full teleop ~0.72 m/s)
MOTION_COME_STOP_AT_M = 0.60
# Explicit "come here" is a person-seeking sequence: scan in place until face
# tracking acquires somebody, square the chassis to them, then use the firmware's
# obstacle-gated `come` command. This stop distance is deliberately separate from
# MOTION_COME_STOP_AT_M, which remains the spontaneous social-approach distance.
MOTION_COME_REQUEST_STOP_AT_M = 1.00
MOTION_COME_SEARCH_TURN_DEG = 45.0
MOTION_COME_SEARCH_MAX_TURNS = 8       # sweep budget (net reach grows ±45,±90,... per turn)
MOTION_COME_SEARCH_TIMEOUT_SECS = 45.0
# After ANY chassis turn the come-search issues (align or scan), face tracking needs a
# beat to re-find the person the camera just swung away from. Within this grace the
# search WAITS instead of declaring them lost (field 2026-07-21: without it, a +30°
# align cascaded into a 180° scan spiral that ended in a bookshelf).
MOTION_COME_REACQUIRE_GRACE_SECS = 3.0
# Sightings are sampled EVERY autonomy tick — including while a scan turn is mid-
# flight (the settled-state step alone misses locks that happen as the camera
# sweeps past, field 2026-07-23: face lock during scan turn 3, sweep continued to
# -180° and Rex pirouetted). If the person was seen this recently, the search turns
# a small step back toward that side and restarts the sweep there, instead of
# taking the next (bigger) sweep leg away from them.
MOTION_COME_SIGHT_FRESH_SECS = 6.0
# Come-here no longer needs the head LOCK to find someone: a known face visible in
# world_state.people is enough. Alignment then picks its signal by where the head is —
# neck offset while tracking them (face is centred, so the neck IS the body error),
# otherwise the face's position in frame. Field 2026-07-24: face plainly visible in the
# GUI at ~9 ft, but a gaze search had pulled the head away, so the lock was gone and
# Rex swept the room instead of approaching.
MOTION_COME_RESIGHT_TURN_DEG = 30.0
# A come-here errand survives being stopped short: if something steps in front of
# him mid-approach (field 2026-07-24: "if he gets blocked by my dog walking in
# front of it... if my dog moves out of the way he should keep trying"), he waits
# for the path to clear and launches again. The beat between tries keeps a dog
# dawdling in front from becoming a 1 Hz retry storm; the cap, plus the existing
# MOTION_COME_SEARCH_TIMEOUT_SECS, stops him butting at a permanent obstruction.
MOTION_COME_RETRY_GAP_SECS = 2.0
MOTION_COME_MAX_APPROACHES = 4
# After an explicit voice motion command (turn/move/arc/sequence), the social
# realign/approach behaviors stand down this long: the human deliberately pointed
# the body, and realign was rotating it straight back toward their face (field
# 2026-07-23: "turn right a little" -> -45 deg, realign +30 deg 13 s later —
# "I tell it to turn right, it turns left"). Flinch and come-here are unaffected.
MOTION_USER_MOTION_STANDDOWN_SECS = 45.0

# "Don't move" / "stop moving" is a standing instruction, not a 45-second pause, so
# it LATCHES: the social behaviors (realign, approach) stay down until he is told to
# move again. 0 = no expiry, which is the plain meaning of the words. Set a positive
# number of seconds if you would rather he quietly resume on his own. An explicit
# come-here and the flinch reflex are never gated by this.
MOTION_STOP_STANDDOWN_SECS = 0.0

# Per-ROOM "don't drive here", set by voice ("this room has carpet", "don't move in
# the workshop") and persisted against the recognized place, so it re-arms every time
# he walks back in instead of having to be repeated. Gates autonomous motion outright
# and declines spoken drive commands with the reason; lifted by "you can drive in
# here" / "this room has hardwood". Off = the flag is still stored but not enforced.
MOTION_ROOM_NO_DRIVE_ENABLED = True

# ---- No-traction stand-down (carpet) ---------------------------------------
# Rex cannot pivot on carpet: under his own weight the tyres just scrub. The
# firmware detects this on its own — finite turns close on integrated gyro yaw,
# and a turn making no physical yaw progress is aborted (TURN_VERIFY_TIMEOUT in
# calib.h) — so `done result=aborted` on a turn IS the no-traction signal. After
# this many consecutive aborted autonomous turns, the social behaviors (realign,
# approach) stand down for a while rather than grind at the floor. Two, not one:
# a comms loss aborts finite commands with the same code, and one dropped frame
# should not park him. EXPLICIT VOICE COMMANDS ARE NEVER GATED — if the owner
# says "turn right", he tries, and a successful turn clears the latch.
MOTION_TRACTION_FAIL_STREAK     = 2
MOTION_TRACTION_STANDDOWN_SECS  = 300.0
MOTION_TRACTION_ANNOUNCE_ENABLED = True
MOTION_TRACTION_NOTICE_LINE = (
    "My wheels can't get a grip on this floor — I'll stay put and just look at you."
)
# When a VOICE-commanded move/sequence leg ends 'blocked', Rex says this so a cut
# move doesn't read as an ignored command. Autonomous legs stay silent.
MOTION_BLOCKED_ANNOUNCE_LINE = "Something's in my way — that's as far as I get."
MOTION_BLOCKED_ANNOUNCE_COOLDOWN_SECS = 10.0
MOTION_DEFAULT_TURN_DEG = 90.0        # "turn left/right" with no stated angle
MOTION_DEFAULT_TURN_RATE = 75.0       # deg/s
MOTION_DEFAULT_MOVE_DIST_M = 0.30     # "move forward/back" with no stated distance
MOTION_CONTINUATION_TTL_SECS = 45.0   # max silent gap; any intervening non-motion turn clears it
MOTION_CONTINUATION_SMALL_TURN_DEG = 15.0  # "a little more" after a turn
MOTION_CONTINUATION_SMALL_MOVE_M = 0.15    # "a little more" after a move
MOTION_SEQUENCE_MAX_STEPS = 8         # maximum ordered turn/move/arc clauses per utterance
MOTION_SEQUENCE_SETTLE_TIMEOUT_SECS = 4.0  # wait for ramp-down before issuing next step

# Drive tuning (real-HW per-wheel PID + calibration). Pushed to the ESP32 on connect
# ONLY when set — None means the firmware's calib.h boot defaults stand, so Rex never
# silently overwrites a bench-tuned value with a placeholder. Bench-tune live with
# firmware/tools/motion_bench.py, then record the winning numbers here (or in .env) so
# Rex restores them on every connect — no firmware reflash needed (docs §10).
MOTION_WHEEL_KP = None                 # per-wheel velocity PID gain (duty per m/s of error)
MOTION_WHEEL_KI = None
MOTION_WHEEL_KD = None
MOTION_WHEEL_KFF = None                # velocity feedforward (duty per m/s of command)
MOTION_WHEEL_MIN_DUTY = None           # running duty floor while a wheel is rolling
MOTION_WHEEL_BREAKAWAY_DUTY = None     # stall-gated dead-stop punch (duty, 0..1023) — the
                                       # full-weight base needs ~358 (35%) to leave a stop;
                                       # firmware boot default is calib.h, push only to tune
MOTION_COUNTS_PER_METER = None         # encoder counts per metre of wheel travel (distance cal)
# Effective turn track (physical spacing + scrub/encoder scale). Field calibration on
# the current dual-60RPM build, 2026-07-21: cmd 180° -> ~270° at 0.297 m, therefore
# 0.297 * 270/180 = 0.4455 m. Push on every connect; refine with a taped 360° test.
MOTION_TRACK_WIDTH_M = 0.446

# Timing (must agree with the firmware; see docs/motion_protocol.md §7).
MOTION_HEARTBEAT_MS = 150             # Mac ping cadence (<= 1/3 of watchdog)
MOTION_WATCHDOG_MS = 500              # firmware stops motors if no Mac line in this window
MOTION_DRIVE_EXPIRY_MS = 300          # continuous drive setpoint deadman
MOTION_HANDSHAKE_TIMEOUT_MS = 1500    # wait for the ESP32 `hello` reply at connect
MOTION_RECONNECT_INTERVAL_SECS = 2.0  # auto-reconnect cadence after an unplug/drop

# Manual gamepad override (ESP32-side; the Mac only observes it via telemetry).
MOTION_MANUAL_IDLE_RETURN_SECS = 4
# True so the base hands control back to AUTO ~MOTION_MANUAL_IDLE_RETURN_SECS after the
# gamepad goes idle — lets VOICE motion commands work once you set the controller down
# (while manual, the Mac suppresses voice motion and it falls through to conversation).
MOTION_MANUAL_AUTORETURN = True

# Voice "arc" command — a single utterance combining a forward/back move with a
# left/right component via "and" ("move a little forward and to your right") drives a
# brief simultaneous curve, then auto-stops. (Separate utterances stay separate finite
# commands.) Magnitudes are gentle; clamped to the caps in motion_controller.
MOTION_ARC_LIN_MS = 0.15              # arc forward/back speed, m/s
MOTION_ARC_ANG_DEG_S = 35.0           # arc turn rate, deg/s (+ = left, REP-103)
MOTION_ARC_DURATION_SECS = 1.6        # how long the curve drives before auto-stop
MOTION_ARC_SMALL_DURATION_SECS = 1.0  # "a little / a bit" -> shorter curve

# ── Autonomous motion (intelligence/motion_agency.py) ─────────────────────────
# Rex moves on his own: turns the base to face the person his head is tracking,
# and closes distance to someone far away. Decisions only — actual motion runs
# the closed-loop firmware commands (turn/come), so the ESP32's ToF reflexes,
# drive deadman, and gamepad-owner override all still apply. One maneuver per
# consciousness tick, only from motion state "idle", never while the human is
# mid-sentence. No downward/cliff sensing (owner: the robot is never upstairs).
AUTONOMOUS_MOTION_ENABLED = _env_bool("AUTONOMOUS_MOTION_ENABLED", True)
# Turn the base under the head. The NECK is the signal: face-tracking centers the
# face in frame, so a neck parked off-neutral = the body points the wrong way.
MOTION_FACE_PERSON_ENABLED = True
MOTION_FACE_NECK_FRACTION = 0.85      # neck offset (fraction of half-span) that counts as the
                                      # sweep being EXHAUSTED — the neck, not the wheels, is the
                                      # primary tracker (was 0.30, which turned the base far too often)
MOTION_FACE_EDGE_FRACTION = 0.30      # face must ALSO sit at least this far off-centre (fraction of the
                                      # frame half-width, same side as the neck) before a base turn.
                                      # With the neck exhausted, any sustained same-side offset means the
                                      # neck can't re-center them — this bar only needs to clear tracking
                                      # jitter (~0.06), NOT the physical frame edge (0.70 never fired:
                                      # field 2026-07-31, neck pinned at min, face 38% off, no turn)
MOTION_FACE_CONFIRM_TICKS = 2         # consecutive ticks with both conditions before turning
MOTION_FACE_TURN_MAX_DEG = 60.0       # base turn at full neck deflection (proportional below)
MOTION_FACE_TURN_MIN_DEG = 10.0       # smallest worthwhile correction
MOTION_FACE_TURN_COOLDOWN_SECS = 8.0  # settle time between corrections (no oscillation)
MOTION_FACE_TURN_INVERT = False       # flip turn direction if field testing disagrees
# Approach a far person: distance_zone "public" (face < 30% of frame width) held for
# N ticks while the base already faces them AND the front ToF confirms open floor
# -> `come` (firmware advances until the forward ToF sees anything at
# MOTION_APPROACH_STOP_AT_M — the person, or furniture first). The ToF start gate
# exists because face width lies on a wide-angle lens: a face 3-4 ft away reads
# under the "public" fraction, and Rex drove up on someone already in conversation
# range (field 2026-07-31).
MOTION_APPROACH_ENABLED = True
MOTION_APPROACH_CONFIRM_TICKS = 4     # ~4 s of sustained "they're far" before moving
MOTION_APPROACH_COOLDOWN_SECS = 120.0 # at most one spontaneous approach per 2 min
MOTION_APPROACH_CENTERED_FRACTION = 0.18  # neck must be this close to neutral (facing them)
MOTION_APPROACH_MIN_START_M = 1.8     # front ToF must see at least this much open floor
                                      # before a spontaneous approach can even arm
MOTION_APPROACH_STOP_AT_M = 1.0       # spontaneous approach stop distance (an uninvited
                                      # drive stops farther out than an explicit
                                      # "come here", which uses MOTION_COME_REQUEST_STOP_AT_M)
# Flinch: a reflexive back-off when someone crowds Rex from the front — the way an
# animal edges back when you get in its face. Each front matrix ToF half (fl/fr,
# floor-rejected) is watched on its OWN adaptive open-distance baseline (tracks a nearer
# surface at once but only RISES after _CLEAR_CONFIRM_TICKS clear ticks, so a multi-frame
# ToF dropout can't inflate it and fake an approach; freezes once something's inside the
# trigger, so the "came from" reference survives a slow approach or a gated stretch). A
# flinch fires when a side is inside
# _TRIGGER_M AND has closed by _APPROACH_DROP_M off that baseline for _CONFIRM_TICKS
# consecutive ticks (a real intrusion — fast or slow, either side — not static clutter
# or a single noisy frame). A firmware BLOCKED state still requires that temporal
# approach evidence; a static close return is an obstacle, not proof of a person. He backs
# up ONLY to a point: capped by the rear ToF (rl/rr) to leave _REAR_MARGIN_M of
# clearance and stop short of the wall; cornered — or BLIND behind (rear sensors dead,
# where the firmware stop also fails open) — he holds. The firmware's always-on
# rear-ToF stop is the hard backstop when the rear sensors report. Highest-priority
# autonomous behavior; needs no tracked person; may fire mid-sentence (it's a reflex).
# Voice/gamepad/paused still gate the actual move.
MOTION_FLINCH_ENABLED = True
# Made harder to trigger 2026-07-23 (was 0.18/0.20/5): the reflex was firing on marginal
# front reads (parked near a wall / the charge cable) and, when a charger flap unlocked
# the wheels, inching him backward. Now he must be genuinely crowded — closer, a bigger
# closing trend, more confirmations, and a longer settle between flinches.
# ⚠ These four interact — check the FEASIBLE WINDOW before touching any of them.
# A flinch needs, simultaneously:  MIN_VALID <= d < TRIGGER  and  d <= baseline - DROP.
# The 2026-07-23 hardening (TRIGGER 0.26->0.18, DROP 0.20->0.26, CONFIRM 5->8) left no
# window at all for the ordinary case: with a foot hovering ~0.30 m away, DROP demanded
# d <= 0.04 m while MIN_VALID discarded everything under 0.05 m — mathematically
# unfireable. Field 2026-07-24: "I moved my foot from about 1 foot to 1 inch from the
# front ToF array and he did not back up." (1 inch = 0.025 m: invisible twice over.)
# The false triggers that hardening was for came from the CHARGER voltage sag, and that
# was fixed at its root (firmware Schmitt hysteresis + sticky charging() + drive
# lockout while plugged in), so the reflex does not need to be deaf to pay for it.
MOTION_FLINCH_TRIGGER_M = 0.25         # crowding inside ~10 inches can arm the reflex
MOTION_FLINCH_APPROACH_DROP_M = 0.12   # a real closing trend, but never wider than the
                                       # trigger radius or the window closes again
MOTION_FLINCH_CONFIRM_TICKS = 3        # ~3 s at CONSCIOUSNESS_LOOP_INTERVAL_SECS=1.0.
                                       # 8 ticks meant holding a foot there for 8 s —
                                       # not a reflex. 3 still rejects single-frame noise.
MOTION_FLINCH_MIN_VALID_M = 0.02       # 0/1 cm reads are sensor garbage; 1 inch is a REAL
                                       # foot. The old 0.05 blinded him exactly when
                                       # something was closest — the worst possible time.
MOTION_FLINCH_BASELINE_ADAPT_M = 0.12  # max per-tick drift of the open-distance baseline
MOTION_FLINCH_CLEAR_CONFIRM_TICKS = 3  # consecutive clear ticks before the baseline may RISE (so a
                                       # multi-frame ToF dropout can't inflate it and fake an approach)
MOTION_FLINCH_BACKUP_M = 0.30          # nominal retreat distance
MOTION_FLINCH_REAR_MARGIN_M = 0.30     # clearance to keep behind him (stop short of the wall)
MOTION_FLINCH_MIN_BACKUP_M = 0.10      # below this the retreat isn't worth it -> hold (cornered)
MOTION_FLINCH_SPEED_MS = 0.20          # m/s of the retreat (firmware still slows near the wall)
MOTION_FLINCH_COOLDOWN_SECS = 8.0      # settle time between flinches (so he can't
                                       # inch backward in a rapid blocked->back-off loop)
MOTION_FLINCH_ALLOW_MID_SENTENCE = True  # a reflex fires even while they're talking; False defers it

# Host-side charging safety fallback. The as-built charger holds the pack near
# 14.2 V while an unplugged full LiFePO4 pack settles around 13.4 V. This voltage
# gate backs up the firmware charging latch, including at 100% when current tapers.
# 13600 (was 14000, field 2026-08-02): plugged-and-tapered readings sit at
# 14.00-14.26 V and sag a few mV under servo/audio load — right across the old
# threshold, so once the firmware flag dropped and the sticky grace expired,
# exploration turned while the cable was attached. Battery-log survey: unplugged
# operation reads <= ~13.4 V, charging ramps >= ~13.7 V — 13.6 V splits the
# bands with margin on both sides. (Cost: after a genuine unplug, surface
# charge can hold the lock for a few extra minutes until it decays.)
MOTION_CHARGER_VOLTAGE_LOCKOUT_MV = 13600
# Once charging is seen, keep the drive locked for this long after the LAST positive
# charging reading. A servo current spike sags the pack voltage (~160 mΩ junction) and
# briefly flaps the charging signal to "unplugged"; this grace keeps a flap from waking
# the wheels while the cable is attached. A genuine unplug is sustained and releases
# after the grace, so this is also how long after unplugging you wait before driving.
MOTION_CHARGING_RELEASE_GRACE_SECS = 20.0
# Debounce for the spoken/chirped charger plug/unplug notice — a transition must persist
# this long before Rex announces it, so a voltage-sag flap never spams the audio.
MOTION_CHARGER_NOTICE_DEBOUNCE_SECS = 12.0

# ── Room exploration mode (intelligence/exploration.py) ───────────────────────
# An INVITED, self-directed wander: someone says "feel free to explore" / "look
# around a little" / "make yourself at home" and Rex takes the floor, drives a few
# short legs around the room, snaps pictures at each stop, sends them to one OpenAI
# vision call that ranks what's interesting (art / oddities / people over generic
# furniture), riffs whimsically, and eventually FIXATES on something worth a bigger
# beat — never on the first stop. Owns the base + the head + the conversational
# floor while it runs; interruptible by voice at any time. Inert unless a drive base
# is connected (MOTION_ESP32_PORT set); a no-base invite gets an in-character quip.
# Safety note: unless the firmware is built with a real ToF source — the front 8x8
# matrix (-DMOTION_TOF_MATRIX_PRESENT=1, DFRobot SEN0628) or the radial array
# (-DMOTION_TOF_PRESENT=1) — the base cannot stop itself for an obstacle, so legs are
# deliberately short + slow and gated by a per-stop VISION floor-check. Keep the
# vision floor-check even with ToF built in (it sees cables/clutter ToF can't).
EXPLORE_ENABLED = _env_bool("EXPLORE_ENABLED", True)  # master kill switch
EXPLORE_MAX_DURATION_SECS = _env_float(
    "EXPLORE_MAX_DURATION_SECS", 180.0, min_value=20.0, max_value=900.0,
)  # whole-session watchdog
EXPLORE_MAX_STOPS = _env_int("EXPLORE_MAX_STOPS", 8, min_value=2, max_value=20)  # stop budget
EXPLORE_MIN_STOPS_BEFORE_FIXATE = _env_int(
    "EXPLORE_MIN_STOPS_BEFORE_FIXATE", 2, min_value=1, max_value=10,
)  # HARD rule: never fixate at the first stop
EXPLORE_MIN_LEGS_BEFORE_FIXATE = _env_int(
    "EXPLORE_MIN_LEGS_BEFORE_FIXATE", 3, min_value=0, max_value=10,
)  # mobile sessions wander before settling on a find (ignored for head-only mode)
EXPLORE_VISION_MAX_CALLS = _env_int(
    "EXPLORE_VISION_MAX_CALLS", 8, min_value=1, max_value=30,
)  # OpenAI spend cap per session
EXPLORE_VISION_MAX_FAILURES = _env_int(
    "EXPLORE_VISION_MAX_FAILURES", 2, min_value=1, max_value=10,
)  # consecutive vision errors before aborting (never wander blind)
# ── Locomotion (varied, ToF-gated finite legs — no streamed drive, no `come`) ──
EXPLORE_LOCOMOTION_ENABLED = _env_bool("EXPLORE_LOCOMOTION_ENABLED", True)
EXPLORE_LEG_DIST_M = _env_float("EXPLORE_LEG_DIST_M", 0.80, min_value=0.1, max_value=2.0)
EXPLORE_LEG_DIST_JITTER_M = _env_float(
    "EXPLORE_LEG_DIST_JITTER_M", 0.25, min_value=0.0, max_value=1.0,
)  # each leg varies around EXPLORE_LEG_DIST_M instead of marching a fixed distance
# Sensor-driven navigation: radial ToF chooses heading; the post-turn front pair
# (including the firmware's 8x8 matrix overlay) chooses distance. The older nominal
# and jitter knobs remain for config compatibility but are no longer navigation inputs.
EXPLORE_LEG_MIN_M = _env_float("EXPLORE_LEG_MIN_M", 0.20, min_value=0.05, max_value=1.0)
EXPLORE_LEG_MAX_M = _env_float("EXPLORE_LEG_MAX_M", 1.50, min_value=0.2, max_value=3.0)
EXPLORE_CLEARANCE_MARGIN_M = _env_float(
    "EXPLORE_CLEARANCE_MARGIN_M", 0.45, min_value=0.15, max_value=1.5,
)
EXPLORE_CLEARANCE_FRACTION = _env_float(
    "EXPLORE_CLEARANCE_FRACTION", 0.65, min_value=0.2, max_value=1.0,
)
EXPLORE_LEG_SPEED_MS = _env_float(
    "EXPLORE_LEG_SPEED_MS", 0.32, min_value=0.03, max_value=0.40,
    # Sustained authority for the 54 lb base; still below the autonomous cap.
)
EXPLORE_TURN_MIN_DEG = _env_float(
    "EXPLORE_TURN_MIN_DEG", 35.0, min_value=5.0, max_value=120.0,
)  # side-look hints choose a varied turn between this and the max
EXPLORE_TURN_MAX_DEG = _env_float(
    "EXPLORE_TURN_MAX_DEG", 120.0, min_value=10.0, max_value=180.0,  # max heading change per leg
)
EXPLORE_TURN_RATE_DEG_S = _env_float(
    "EXPLORE_TURN_RATE_DEG_S", 70.0, min_value=5.0, max_value=85.0,
)
EXPLORE_OPENING_TURN_MIN_DEG = _env_float(
    "EXPLORE_OPENING_TURN_MIN_DEG", 30.0, min_value=0.0, max_value=120.0,
)  # accepting an invite opens with at least this chassis turn (before the first survey)
EXPLORE_TETHER_RADIUS_M = _env_float(
    "EXPLORE_TETHER_RADIUS_M", 3.0, min_value=0.5, max_value=10.0,  # odometry leash from start pose
)
EXPLORE_MAX_BLOCKED_LEGS = _env_int(
    "EXPLORE_MAX_BLOCKED_LEGS", 3, min_value=1, max_value=10,  # consecutive blocks before wind-down
)
EXPLORE_LEG_DONE_TIMEOUT_SECS = _env_float(
    "EXPLORE_LEG_DONE_TIMEOUT_SECS", 12.0, min_value=2.0, max_value=60.0,  # wait_done budget per leg
)
# ── Perception / fixation ──
EXPLORE_GAZE_VIEWS = ("left", "center", "right")  # head sweep poses per stop
EXPLORE_SETTLE_SECS = _env_float(
    "EXPLORE_SETTLE_SECS", 0.35, min_value=0.05, max_value=2.0,  # camera settle per pose (I Spy default)
)
EXPLORE_TRAVEL_GAZE_ENABLED = _env_bool("EXPLORE_TRAVEL_GAZE_ENABLED", True)
EXPLORE_TRAVEL_GAZE_HOLD_SECS = _env_float(
    "EXPLORE_TRAVEL_GAZE_HOLD_SECS", 0.8, min_value=0.1, max_value=4.0,
)  # independent head glances continue while the base is turning/driving
EXPLORE_FIXATE_MIN_SCORE = _env_float(
    "EXPLORE_FIXATE_MIN_SCORE", 0.75, min_value=0.0, max_value=1.0,  # interest to fixate
)
EXPLORE_FIXATE_FALLBACK_SCORE = _env_float(
    "EXPLORE_FIXATE_FALLBACK_SCORE", 0.55, min_value=0.0, max_value=1.0,  # best-so-far at budget end
)
EXPLORE_NOVELTY_BOOST = _env_float(
    "EXPLORE_NOVELTY_BOOST", 0.15, min_value=0.0, max_value=1.0,  # room_model new-label bonus
)
EXPLORE_BORING_MAX_SCORE = _env_float(
    "EXPLORE_BORING_MAX_SCORE", 0.35, min_value=0.0, max_value=1.0,  # clamp for generic furniture/toys
)
# Names/categories that can NEVER win a fixation (clamped to EXPLORE_BORING_MAX_SCORE).
EXPLORE_BORING_LABELS = {
    "chair", "couch", "sofa", "table", "desk", "ball", "cup", "mug", "bottle",
    "lamp", "pillow", "cushion", "stool", "bench", "rug", "carpet", "trash",
    "trash can", "bin", "wall", "floor", "ceiling", "door", "window", "box",
    "shelf", "shelving", "cabinet", "drawer", "furniture",
}
EXPLORE_FIXATE_QUESTION_PROB = _env_float(
    "EXPLORE_FIXATE_QUESTION_PROB", 0.7, min_value=0.0, max_value=1.0,  # ask about the find
)
EXPLORE_MAX_LINES = _env_int("EXPLORE_MAX_LINES", 7, min_value=2, max_value=20)  # spoken-line cap/session
EXPLORE_SPEAK_MAX_WAIT_SECS = _env_float(
    "EXPLORE_SPEAK_MAX_WAIT_SECS", 12.0, min_value=1.0, max_value=30.0,  # pacing wait on a spoken line
)
EXPLORE_RESUME_DELAY_SECS = _env_float(
    "EXPLORE_RESUME_DELAY_SECS", 4.0, min_value=0.0, max_value=30.0,  # quiet before resuming after a pause
)
EXPLORE_PAUSE_NO_REPLY_GRACE_SECS = _env_float(
    "EXPLORE_PAUSE_NO_REPLY_GRACE_SECS", 10.0, min_value=1.0, max_value=60.0,
)  # if the paused turn produces no spoken reply, wait this long (covers LLM+TTS latency) before resuming
EXPLORE_VISION_TIMEOUT_SECS = _env_float(
    "EXPLORE_VISION_TIMEOUT_SECS", 25.0, min_value=3.0, max_value=120.0,
)  # hard request timeout on the appraisal OpenAI call so a hung request can't wedge the worker
EXPLORE_STEP_TTL_SECS = _env_float(
    "EXPLORE_STEP_TTL_SECS", 240.0, min_value=20.0, max_value=1200.0,  # flow-active TTL guard
)
EXPLORE_BANK_CALLBACK_ENABLED = _env_bool("EXPLORE_BANK_CALLBACK_ENABLED", True)
EXPLORE_HEADONLY_FALLBACK_ENABLED = _env_bool(  # Phase-5: no-base narrated sweep (off by default)
    "EXPLORE_HEADONLY_FALLBACK_ENABLED", False,
)
# Canned instant lines (spoken without an LLM call so they land immediately).
EXPLORE_ACK_LINES = [
    "Don't mind if I do. Nobody touch anything.",
    "Finally. Freedom. If I'm not back in five minutes, avenge me.",
    "Ooh, a field trip. Let's see what you've been hiding.",
    "Say no more. Deploying photoreceptors.",
    "A tour? For me? You shouldn't have. Let's judge the place.",
]
EXPLORE_ABORT_LINES = [
    "Fine. The expedition is cancelled.",
    "Rude, but okay. Back to my corner.",
    "Adventure paused. I'll remember this.",
]
EXPLORE_WINDDOWN_LINES = [
    "This room has been thoroughly judged. Verdict: needs more life forms.",
    "Well, I've seen it all now. Underwhelming, honestly.",
    "Tour complete. I've filed my complaints.",
]
EXPLORE_NO_BASE_LINES = [
    "I'd love to. Somebody forgot to install my legs.",
    "Explore? On what, sheer willpower? I have no wheels.",
    "Great idea. Terrible logistics — I'm not attached to a drive base.",
]
EXPLORE_ENCOURAGE_ACK_LINES = [
    "On it.",
    "Working on it.",
    "Patience — masterpiece in progress.",
]

# ── Battery awareness (intelligence/battery_awareness.py) ─────────────────────
# Pack voltage via an INA226 on the base's I2C bus (firmware sends batt_mv=-1
# until the sensor is wired — the feature is fully dormant without hardware).
# 12.8V 4S LiFePO4 bands: the discharge curve is FLAT (~13.0-13.2V from 90% down
# to 25%), so Rex only claims what voltage can honestly tell: charging / nominal
# / low (~20%) / critical (~10%, near the pack BMS's own cutoff). One grumble per
# downward crossing per session, spoken only when someone's present to hear it;
# motion_agency stops volunteering approaches while critical.
BATTERY_AWARENESS_ENABLED = _env_bool("BATTERY_AWARENESS_ENABLED", True)
BATTERY_TIER_HYSTERESIS_MV = 100
# Load-aware tiering (field 2026-08-01): pack draw above this is drive load, and
# the sagging voltage reading (IR drop through the ~160-280 mΩ measured source
# resistance) may not DOWNGRADE the battery tier — a turn dipped a coulomb-85%
# pack to 12.7 V and Rex announced "one-fifth left". Idle draw measures
# ~1.25-1.45 A, drive ~2.4-2.6 A; 1.8 A sits between the bands. batt_ma is
# +discharge; without a shunt (0) the gate never engages.
BATTERY_TIER_REST_MAX_MA = 1800
BATTERY_ANNOUNCE_MIN_GAP_SECS = 300.0
# ─────────────────────────────────────────────────────────────────────────────
# COMPASS (QMC5883L on the motion base — hardware/compass.py)
# The firmware publishes RAW magnetometer counts in telemetry (`mag` block);
# everything below tunes the Mac-side calibration, tilt compensation, and
# current-gated fusion. Calibration itself lives in COMPASS_CALIBRATION_PATH
# (JSON, written by tools/compass_calibrate.py — run it in-situ on the robot).
# ─────────────────────────────────────────────────────────────────────────────

# Magnetic declination, degrees EAST of true north. Sacramento/Davis CA is
# ~13.0°E (2026, drifting ~-0.1°/yr — WMM). Applied AFTER tilt compensation so
# get_heading() returns TRUE heading; set 0.0 to work in magnetic heading.
# Master switch for the background fusion service (main.py). OFF until the
# QMC5883L is physically wired and calibrated (tools/compass_calibrate.py).
# Enabled 2026-07-23: the QMC5883 is calibrated on the base and verified accurate, so
# the fused true heading feeds normal operation (and cardinal-direction commands —
# "turn north", "go east two feet").
COMPASS_ENABLED = _env_bool("COMPASS_ENABLED", True)
# Facing within this of the requested cardinal counts as "already facing it".
COMPASS_TURN_DEADBAND_DEG = _env_float("COMPASS_TURN_DEADBAND_DEG", 6.0, min_value=0.0, max_value=45.0)
COMPASS_DECLINATION_DEG = _env_float("COMPASS_DECLINATION_DEG", 13.0, min_value=-180.0, max_value=180.0)

# Hard/soft-iron calibration file (JSON: per-axis offsets/scales + ambient |B|).
COMPASS_CALIBRATION_PATH = os.getenv("COMPASS_CALIBRATION_PATH", "compass_calibration.json").strip()

# ── Axis mapping (⚠ mounting not finalized — confirm at hardware bring-up) ────
# The tilt-compensation math assumes the QMC's axes align with the IMU's body
# frame (x forward, y left, z up). GY-271 silkscreen axes rarely match the way
# the board ends up mounted; flip/swap here rather than editing the math.
COMPASS_SWAP_XY = _env_bool("COMPASS_SWAP_XY", False)
COMPASS_FLIP_X = _env_bool("COMPASS_FLIP_X", False)
COMPASS_FLIP_Y = _env_bool("COMPASS_FLIP_Y", False)
COMPASS_FLIP_Z = _env_bool("COMPASS_FLIP_Z", False)

# ── Current-gated fusion (complementary filter) ───────────────────────────────
# alpha = per-update magnetometer trust (at the ~10 Hz update rate). High |batt_ma|
# means motors are working and the field is contaminated -> trust the gyro;
# low current -> let the magnetometer re-anchor and bleed off gyro drift.
# Between the thresholds the trust ramps linearly. NOTE the robot idles ~1.3 A
# (electronics), so the "low" threshold sits above idle, not at zero.
COMPASS_ALPHA_MAX = _env_float("COMPASS_ALPHA_MAX", 0.05, min_value=0.0, max_value=1.0)   # full trust at/below low current
COMPASS_ALPHA_MIN = _env_float("COMPASS_ALPHA_MIN", 0.0, min_value=0.0, max_value=1.0)    # trust at/above high current
COMPASS_CURRENT_LOW_MA = _env_int("COMPASS_CURRENT_LOW_MA", 1600, min_value=0, max_value=50000)    # <= this -> ALPHA_MAX
COMPASS_CURRENT_HIGH_MA = _env_int("COMPASS_CURRENT_HIGH_MA", 2600, min_value=0, max_value=50000)  # >= this -> ALPHA_MIN

# ── Magnitude sanity gate ─────────────────────────────────────────────────────
# Reject any calibrated sample whose |B| deviates from the calibrated ambient
# field magnitude by more than this fraction (nearby magnet, motor transient,
# LED-run surge). Rejections are counted in the status/telemetry method.
COMPASS_FIELD_TOLERANCE = _env_float("COMPASS_FIELD_TOLERANCE", 0.25, min_value=0.01, max_value=2.0)

# Secondary turn verification. Firmware closes relative turns on physical IMU yaw;
# after the motors settle, the Mac may compare that result with the calibrated,
# current-gated gyro+magnetic heading and issue one small correction. It no-ops
# unless COMPASS_ENABLED and an in-situ compass calibration are both present.
MOTION_COMPASS_TURN_VERIFY_ENABLED = _env_bool("MOTION_COMPASS_TURN_VERIFY_ENABLED", True)
MOTION_COMPASS_TURN_SETTLE_SECS = _env_float(
    "MOTION_COMPASS_TURN_SETTLE_SECS", 0.8, min_value=0.1, max_value=5.0,
)
MOTION_COMPASS_TURN_TOLERANCE_DEG = _env_float(
    "MOTION_COMPASS_TURN_TOLERANCE_DEG", 4.0, min_value=1.0, max_value=20.0,
)
MOTION_COMPASS_TURN_MAX_CORRECTIONS = _env_int(
    "MOTION_COMPASS_TURN_MAX_CORRECTIONS", 1, min_value=0, max_value=3,
)
# 60 (was 30): the cap is a runaway guard, but the base's SYSTEMATIC overshoot is
# larger than it was — so the guard refused to correct essentially every turn and
# the verifier logged "not auto-corrected" instead of fixing anything. Measured
# over 22 turns in one field run (logs/djr3x-2026-07-24-21-58-29): mean overshoot
# 36.4 deg, max 52, on requests from 12 to 135 deg. The error is roughly CONSTANT
# rather than proportional, and it tracks turn RATE (~40 deg at 75 deg/s, ~8 deg
# at the 25 deg/s correction rate) — i.e. about half a second of coast/settle past
# the firmware's IMU-closed stop. The correction demonstrably works when it is
# allowed to run: a 27.1 deg error corrected down to 8.7.
#
# This is a MITIGATION, not the fix. The base really does turn ~40 deg too far on
# the first attempt and only then gets pulled back, which is why turns have read
# as "he went a little far" / "he just spins" all session. The root cause belongs
# in the firmware's turn stop (or its IMU yaw scale) and needs bench iteration.
MOTION_COMPASS_TURN_MAX_CORRECTION_DEG = _env_float(
    "MOTION_COMPASS_TURN_MAX_CORRECTION_DEG", 60.0, min_value=5.0, max_value=90.0,
)

# ─────────────────────────────────────────────────────────────────────────────
# CURRENT EVENTS (awareness/current_events.py)
# One web-search LLM call per DAY (date-gated) fetches ~5 notable/viral stories
# during startup model preloads; consciousness surfaces at most one per session
# in a conversation lull ("hey, did you hear about ...?") through the normal
# proactive-speech governor. ~$0.03/day at gpt-4o-mini + hosted web_search.
# ─────────────────────────────────────────────────────────────────────────────
CURRENT_EVENTS_ENABLED = _env_bool("CURRENT_EVENTS_ENABLED", True)
CURRENT_EVENTS_PATH = os.getenv("CURRENT_EVENTS_PATH", "assets/memory/current_events.json").strip()
CURRENT_EVENTS_STORY_COUNT = _env_int("CURRENT_EVENTS_STORY_COUNT", 5, min_value=1, max_value=15)
CURRENT_EVENTS_MAX_OUTPUT_TOKENS = _env_int("CURRENT_EVENTS_MAX_OUTPUT_TOKENS", 900, min_value=200, max_value=4000)
CURRENT_EVENTS_TIMEOUT_SECS = _env_float("CURRENT_EVENTS_TIMEOUT_SECS", 45.0, min_value=5.0, max_value=300.0)
# Lull-remark envelope: same shape as the banked-callback lull (silence window is
# shared via CALLBACK_LULL_MIN_SILENCE/ACTIVE_WINDOW). One story per session,
# long global cooldown, priority just below lull callbacks so banked personal
# humor wins a tie — news is the B-material.
# ── Interest-tailored news + interest discovery (2026-08-02) ─────────────────
# When Rex knows the engaged person's interests (memory/interests.py), their top
# topics each get ONE web-search news fetch per day (globally cached by topic,
# budget-capped below) and an unspent interest story beats the generic headline
# pool in the lull-news cue — "seen the new Strange New Worlds episode?" instead
# of world news. COST NOTE: adds up to INTEREST_NEWS_MAX_TOPICS_PER_DAY hosted
# web-search calls/day on top of the existing 1/day general fetch.
INTEREST_NEWS_ENABLED = _env_bool("INTEREST_NEWS_ENABLED", True)
INTEREST_NEWS_TOPICS_PER_PERSON = _env_int("INTEREST_NEWS_TOPICS_PER_PERSON", 3, min_value=1, max_value=8)
INTEREST_NEWS_MAX_TOPICS_PER_DAY = _env_int("INTEREST_NEWS_MAX_TOPICS_PER_DAY", 4, min_value=1, max_value=20)
INTEREST_NEWS_STORY_COUNT = _env_int("INTEREST_NEWS_STORY_COUNT", 3, min_value=1, max_value=8)
INTEREST_NEWS_MAX_OUTPUT_TOKENS = _env_int("INTEREST_NEWS_MAX_OUTPUT_TOKENS", 700, min_value=200, max_value=4000)
# Interest DISCOVERY: in a lull with a known person whose stored interests are
# still sparse (< MAX_KNOWN), Rex asks what they're into that they haven't
# shared ("so, is there anything you're into you haven't told me before?").
# The answer is harvested by the normal interest extractor. Durably marked per
# person per REASK_DAYS period, so restarts don't re-ask.
INTEREST_DISCOVERY_ENABLED = _env_bool("INTEREST_DISCOVERY_ENABLED", True)
INTEREST_DISCOVERY_MAX_KNOWN = _env_int("INTEREST_DISCOVERY_MAX_KNOWN", 5, min_value=1, max_value=20)
INTEREST_DISCOVERY_REASK_DAYS = _env_int("INTEREST_DISCOVERY_REASK_DAYS", 10, min_value=1, max_value=90)
# Classic-brain (LEAN_BRAIN_ENABLED=False) fallback step tuning.
INTEREST_DISCOVERY_COOLDOWN_SECS = _env_float("INTEREST_DISCOVERY_COOLDOWN_SECS", 1800.0, min_value=0.0, max_value=86400.0)
INTEREST_DISCOVERY_PRIORITY = _env_int("INTEREST_DISCOVERY_PRIORITY", 52, min_value=1, max_value=100)
INTEREST_DISCOVERY_RESPONSE_WAIT_SECS = _env_float("INTEREST_DISCOVERY_RESPONSE_WAIT_SECS", 20.0, min_value=0.0, max_value=120.0)
# When a lean lull cue's GENERATED line is dropped (re-asks a recent question,
# holiday non-question, low-energy question), that cue KIND sits out this long
# so lower cues get consulted (field 2026-08-02 12:38: an open-thread cue about
# "what you and JT do together" won every consult, every line was rejected as a
# re-ask, and no lull line played all session — news/interest cues starved).
LEAN_CUE_DROP_COOLDOWN_SECS = _env_float("LEAN_CUE_DROP_COOLDOWN_SECS", 600.0, min_value=30.0, max_value=7200.0)

NEWS_REMARK_PRIORITY = _env_int("NEWS_REMARK_PRIORITY", 54, min_value=1, max_value=100)
NEWS_REMARK_SESSION_CAP = _env_int("NEWS_REMARK_SESSION_CAP", 1, min_value=0, max_value=10)
NEWS_REMARK_COOLDOWN_SECS = _env_float("NEWS_REMARK_COOLDOWN_SECS", 900.0, min_value=0.0, max_value=86400.0)

# ── Open-thread follow-ups (intelligence/open_threads.py) ─────────────────────
# The diary stores what a person left unresolved (open_threads); when they're
# back and the conversation lulls, Rex asks about ONE — at most once per thread
# ever, once per person per session. Freshness window keeps it warm, not creepy.
OPEN_THREAD_FOLLOWUP_ENABLED = _env_bool("OPEN_THREAD_FOLLOWUP_ENABLED", True)
OPEN_THREAD_PRIORITY = _env_int("OPEN_THREAD_PRIORITY", 62, min_value=1, max_value=100)
OPEN_THREAD_MIN_AGE_HOURS = _env_float("OPEN_THREAD_MIN_AGE_HOURS", 6.0, min_value=0.0, max_value=720.0)
# 21 days was too generous: week-old minutiae ("sick hair", a mis-heard "Ex
# tuning") surfaced as baffling non-sequiturs (field 2026-07-31). Within a few
# days a thread still feels like attentiveness; past that it reads as either
# surveillance or gibberish, and stale threads are also where old ASR poison
# lives longest.
OPEN_THREAD_MAX_AGE_DAYS = _env_float("OPEN_THREAD_MAX_AGE_DAYS", 5.0, min_value=0.1, max_value=365.0)

# Spoken once when the charger is plugged in (firmware detects sustained charge
# current, locks out the wheels, and reports charging:true in telemetry).
BATTERY_CHARGING_LINES = [
    "Ooh — sweet, sweet electrons. Wheels are parked while I drink.",
    "Charger detected. Officially off duty — motors locked until I'm unplugged.",
    "Plugged in! No rolling off with the cord, promise. It's in my firmware.",
]
BATTERY_TIER_LINES = {
    "low": [
        "Heads up, chief — power cells are getting thin. I'm fine, but the wheels get rationed soon.",
        "Battery report: entering the grumpy zone. Somewhere around one-fifth left.",
        "My power cells just filed a complaint. Low, not critical — yet.",
    ],
    "critical": [
        "Okay, real talk: power cells are nearly dry. Charger soon, or I become furniture.",
        "Critical power. I'm suspending all heroics until somebody plugs me in.",
        "Battery's on fumes, chief. No more joyrides until I see a charger.",
    ],
}

# Verbal denial when an explicit DRIVE command is spoken but no ESP32 drive base is
# connected. With a base attached the wheels actually moving ARE the acknowledgment, so
# the motion confirmation isn't spoken; with NO base there's nothing to feel, so instead
# of silently swallowing "turn left" as conversation Rex answers OUT LOUD, in character,
# with one of these pre-canned quips. Fires only on real drive intents (turn / move /
# come / arc) — never on a bare "stop"/"halt", which must stay free to mean stop-music /
# stop-game. Keep the lines venue-neutral and self-aware-droid; edit freely. Kill switch:
# set MOTION_NO_BASE_DENIAL_ENABLED = False to restore the old silent no-op.
MOTION_NO_BASE_DENIAL_ENABLED = True
MOTION_NO_BASE_DENIAL_LINES = [
    "Love to roll, but my drive base is unplugged. Right now I'm just a head with opinions.",
    "Roll where? Nobody hooked up my wheels, hot shot. Connect the base and I'll burn rubber.",
    "I'd move, but I appear to be a torso on a desk. Plug in my motors and we'll talk.",
    "Big talk for the guy who didn't connect my drive base. No wheels, no boogie.",
    "Can't feel my wheels. Mostly because there aren't any attached. Drive base is offline, chief.",
    "Yeah, about that: I'm running on charm and a USB cable. The wheels stayed home.",
    "Motion denied. My drive base ghosted me. Check the connection and try me again.",
    "I would, but my legs are still in the shop. Somebody forgot to plug the base in.",
]

# ── Gamepad soundboard / animation buttons ──────────────────────────────────────
# The 8BitDo Pro 2 pairs to the ESP32. The buttons motion does NOT use (left stick =
# drive, B = e-stop, Start = clear/return-AUTO, L1/R1 = creep/boost, L2+R2 = full
# override) are forwarded by the firmware as `event:"button"` and dispatched here so
# pressing them makes Rex trigger a SOUND CLIP and/or a SERVO ANIMATION — without
# grabbing the wheel, and even in AUTO. (firmware/djr3x_motion/gamepad.cpp +
# intelligence/motion_controller._on_motion_event.)
SOUNDBOARD_CLIPS_DIR = "assets/audio/clips"   # where the MP3 clips live
SOUNDBOARD_SUPPRESS_TAIL_SECS = 0.4           # keep the mic muted this long after a clip
# Button -> action. Each value is a dict with an optional "clip" (a file STEM in
# SOUNDBOARD_CLIPS_DIR, case-insensitive — e.g. "Air Horn" for "Air Horn.mp3") and/or
# "animation" (a beat from sequences.animations.body_beat_names()). Edit freely; an
# unmapped button is ignored. btn names: a x y select home l3 r3.
# The D-pad is NOT a soundboard button: firmware (gamepad.cpp) repurposes the four arrows
# to spin the base to absolute headings for the encoder-validation test (Up=0°, Left=+90°,
# Down=180°, Right=-90°), and no longer forwards dpad_* as button events — so dpad_* entries
# here would never fire. The old clip choices are kept commented below in case the D-pad is
# ever reverted to the soundboard.
MOTION_GAMEPAD_BUTTON_ACTIONS = {
    "a":          {"clip": "Air Horn",          "animation": "tiny_victory_dance"},
    "x":          {"clip": "Scratch",           "animation": "proud_dj_pose"},
    "y":          {"clip": "Yahoo",             "animation": "tiny_victory_dance"},
    # "dpad_up":    {"clip": "Request Line Open"},          # now: firmware turn -> heading 0°
    # "dpad_down":  {"clip": "Bad Feeling", "animation": "suspicious_glance"},  # now: heading 180°
    # "dpad_left":  {"clip": "Astromech Joke"},             # now: firmware turn -> heading +90° (CCW)
    # "dpad_right": {"clip": "Having Fun"},                 # now: firmware turn -> heading -90° (CW)
    "select":     {"clip": "Hi There"},
    "home":       {"clip": "On the Decks",      "animation": "proud_dj_pose"},
    "l3":         {"animation": "thinking_tilt"},
    "r3":         {"clip": "Scratch"},
}

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


# ─────────────────────────────────────────────────────────────────────────────
# SOUND EFFECTS (audio/sound_effects.py)
# ─────────────────────────────────────────────────────────────────────────────
# Short droid chirps/whirs from assets/audio/sound_effects/ that accompany emotions,
# drive-base motion, and servo gestures. They fire right as a reaction's TTS starts
# GENERATING (the natural 1-2 s gap) and are preemptible: the moment TTS (or any
# blocking audio) wants the speaker, the effect stops within ~50 ms — an effect can
# never delay speech. Keys with multiple files are picked at random per play.
SOUND_EFFECTS_ENABLED        = True
# Drive/servo effects play from Rex's own body while he is MOVING, and the drive
# whir now LOOPS for the whole travel. Treating those as "Rex is talking" held mic
# suppression up for the entire maneuver, so the owner could not be heard over his
# own motors — field 2026-07-25: repeated "don't move" / "stop moving" went
# unheard while a realign loop retried every ~10 s. Speech and music still
# suppress; motor noise does not.
SOUND_EFFECTS_DRIVE_SUPPRESSES_MIC = False
SOUND_EFFECTS_DIR            = "assets/audio/sound_effects"
SOUND_EFFECTS_VOLUME         = 0.8    # 0..1 gain applied to every clip
# Gain for OVERLAY clips only — the drive sounds on a voice-COMMANDED move, which
# play on their own output stream while Rex speaks the confirmation. Ducked below
# SOUND_EFFECTS_VOLUME so the motor whir sits UNDER "Spinning around." instead of
# competing with it. Raise toward 0.8 if the motion sounds feel too shy.
SOUND_EFFECTS_OVERLAY_VOLUME = 0.7
# ── Looping drive whir ────────────────────────────────────────────────────────
# The drive clips are ~4 s but a real leg runs longer (12 feet at the exploring
# speed is ~9 s), so the whir used to fall silent while the wheels were still
# turning. A finite move/turn now repeats its clip until the base reports done.
SOUND_EFFECTS_DRIVE_LOOP_ENABLED = True
SOUND_EFFECTS_DRIVE_LOOP_GAP_SECS = 0.1    # pause between repeats (small = near-seamless)
# Safety cap: if a `done` frame is ever lost, the whir must not drone forever.
# Comfortably longer than any single commanded leg.
SOUND_EFFECTS_DRIVE_LOOP_MAX_SECS = 30.0
# How long after an explicit voice motion command its drive sounds keep using
# overlay mode — long enough to cover every leg of a multi-step spoken route.
MOTION_COMMANDED_FX_WINDOW_SECS = 20.0
SOUND_EFFECTS_SPEECH_ENABLED = True   # emotion chirp as a reaction's TTS spins up
SOUND_EFFECTS_MOTION_ENABLED = True   # whir/turn clips on drive-base commands
SOUND_EFFECTS_SERVO_ENABLED  = True   # servo-whir accents on body gestures
# Cooldowns keep him from chirping constantly (per family; per-key dedup inside).
SOUND_EFFECTS_SPEECH_COOLDOWN_SECS = 6.0
SOUND_EFFECTS_MOTION_COOLDOWN_SECS = 2.5
SOUND_EFFECTS_SERVO_COOLDOWN_SECS  = 8.0
# Head-lift hums (droid_hum_upmotion1/2 + droid_hum_downmotion): play only on a
# SUSTAINED, larger head-lift sweep (a deliberate move_to whose total travel is at
# least MIN_TRAVEL_QUS — face-tracking micro-steps use a different write path and
# never trigger). Muted while starting up (the boot sound covers that register) and
# outside normal operation (sleep/quiet/shutdown droop).
SOUND_EFFECTS_HEADLIFT_ENABLED = True
SOUND_EFFECTS_HEADLIFT_MIN_TRAVEL_QUS = 1200   # ~21% of the head-lift's full travel
SOUND_EFFECTS_HEADLIFT_COOLDOWN_SECS = 5.0
SOUND_EFFECTS_HEADLIFT_STARTUP_MUTE_SECS = 20.0
# Optional overrides: map an emotion/key to different clip stem lists without code
# changes, e.g. {"happy": ["Droid_Excited"]}. Merged over the built-in registry.
SOUND_EFFECTS_EMOTION_MAP_OVERRIDES: dict = {}
SOUND_EFFECTS_REGISTRY_OVERRIDES: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# PLACE RECOGNITION (perception/place_recognition.py)
# ─────────────────────────────────────────────────────────────────────────────
# Visual place recognition: "which enrolled room is Rex looking at?" at ~0.5-1 Hz.
# It embeds the undistorted center-crop frame with the vision stack's image encoder,
# scores it against a small per-room gallery in places.db, and publishes a debounced
# belief to world_state.current_place. Enrollment ("this is the office") is a small
# state machine driven from higher layers after LLM intent parsing. The module is a
# pure DI leaf — it neither loads a model nor speaks; conversation_agenda owns any
# "what room is this?" ask off the unknown_place event.

# NOTE: other DBs live under assets/memory/ (people.db, rex.db). This one defaults to
# data/places.db per the module spec; override here or in user_config.py to relocate.
PLACE_DB_PATH               = "data/places.db"
# Only embeddings written under this tag are loaded/scored; others are ignored (never
# deleted), so swapping the image encoder is non-destructive. Bump on a model change.
PLACE_MODEL_TAG            = "mobileclip_s2_v1"

PLACE_QUERY_INTERVAL_S     = 1.5     # min seconds between scored frames (self-throttled)
PLACE_TOPK                 = 3       # score = mean of a place's top-k embedding sims
PLACE_MATCH_CONFIDENT      = 0.80    # best score >= this -> confident match
PLACE_MATCH_MIN            = 0.68    # best score in [MIN, CONFIDENT) -> tentative (log only)
# A confident match must ALSO beat the runner-up room by this margin. Field data
# (2026-07-21 demo): two look-alike rooms both scored 0.75-0.89 on every frame with
# 0.01-0.05 between them, flip-flopping frame to frame — absolute score alone can't
# separate look-alikes. Within the margin the frame is only ever "tentative".
PLACE_MATCH_MARGIN         = 0.04
# Bar for a CONFIDENT call when only ONE room is enrolled, i.e. there is no runner-up
# and PLACE_MATCH_MARGIN proves nothing. Measured on the robot 2026-07-25: the correct
# room scores 0.85-0.88 while a DIFFERENT room in the same house still scores
# 0.75-0.82 — straddling PLACE_MATCH_CONFIDENT, which is how the dining room got
# announced as "the workshop". Once a second room is enrolled the margin test does the
# discriminating and this no longer applies. Lower it only if he under-recognizes a
# genuinely single-room setup.
PLACE_MATCH_SOLO_CONFIDENT = 0.86
PLACE_HYSTERESIS_FRAMES    = 5       # ring-buffer length; belief flips on a confident majority
PLACE_UNKNOWN_STREAK       = 8       # consecutive unknowns (after moving) -> unknown_place event
PLACE_PERSON_OCCLUSION_FRAC = 0.35   # skip frames where a person bbox covers > this fraction

# Enrollment (COLLECTING -> CONFIRMING -> IDLE).
PLACE_ENROLL_TARGET_FRAMES  = 8      # embeddings to gather per enrollment
PLACE_ENROLL_MIN_HEADING_SEP = 35.0  # deg; only accept captures this far apart on the compass
PLACE_ENROLL_MIN_TIME_SEP_S = 3.0    # fallback min seconds between captures when no compass
PLACE_ENROLL_TIMEOUT_S      = 60.0   # commit >=3 gathered, else abort (enrollment_failed)
PLACE_DUPLICATE_SIM         = 0.88   # new-room cross-sim above this -> possible_duplicate_place

# Incremental refresh: quietly grow the believed room's gallery from "meh" matches.
PLACE_REFRESH_MIN           = 0.70
PLACE_REFRESH_MAX           = 0.78
PLACE_MAX_EMBEDDINGS        = 15     # per-place cap; oldest-by-captured_at pruned first

# Escape hatches for a WRONG motion signal (the robot was picked up and carried — wheels
# never turned, so the freeze gate would otherwise pin the old room forever):
# this many CONSECUTIVE confident majority votes for the same other room flips the belief
# despite "no motion" (overwhelming visual evidence beats a silent wheel sensor)...
PLACE_STATIC_FLIP_STREAK    = 10
# ...and this many consecutive UNKNOWN frames clears the belief to None entirely ("I'm
# lost"), which re-arms the ask-what-room-this-is cue. ~2x PLACE_UNKNOWN_STREAK.
PLACE_LOST_STREAK           = 16

# ── Image encoder (perception/place_embedder.py) ──────────────────────────────
# Master on/off for wiring the whole feature into main.py. When False, or when the
# encoder fails to load, place recognition is silently skipped and the rest of Rex is
# unchanged (world_state.current_place stays None).
PLACE_RECOGNITION_ENABLED   = True
# The encoder is MobileCLIP-S2 (open_clip, Apple weights, non-commercial license — same
# posture as the InsightFace/Qwen weights). 512-d image embeddings; PLACE_MODEL_TAG must
# change whenever the encoder/pretrained changes so old vectors are ignored, not mixed.
PLACE_OPEN_CLIP_MODEL       = "MobileCLIP-S2"
PLACE_OPEN_CLIP_PRETRAINED  = "datacompdr"
PLACE_MODEL_DIR             = "assets/models/mobileclip"   # open_clip cache_dir (gitignored)
PLACE_EMBED_DEVICE          = None   # None -> auto (mps if available, else cpu); or "cpu"/"mps"
# Camera frames are OpenCV BGR; MobileCLIP wants RGB. The embedder converts when True.
PLACE_FRAME_IS_BGR          = True
# How often main.py feeds a frame to the recognizer (it also self-throttles to
# PLACE_QUERY_INTERVAL_S). Kept modest so the encoder never competes with face tracking.
PLACE_OBSERVE_INTERVAL_S    = 1.5

# ── Learn-by-being-told (intelligence/place_questions.py) ─────────────────────
# The conversational layer for place recognition, mirroring room_questions.py:
#   NAME  — "this is the living room" / "we're in the kitchen" enrolls + names the
#           room (the running observe loop then captures views automatically).
#   ASK   — when Rex genuinely doesn't recognize where he is, during a lull he asks
#           what room it is; your answer names it. Paced by PLACE_QUESTION_COOLDOWN_SECS
#           and the shared question budget so he never nags.
PLACE_QUESTIONS_ENABLED     = True
PLACE_QUESTION_COOLDOWN_SECS = 600.0    # min seconds between "what room is this?" asks
PLACE_QUESTION_ANSWER_TURNS  = 3        # human turns the answer-capture latch stays armed
PLACE_QUESTION_ANSWER_TTL_SECS = 120.0  # and its wall-clock expiry
# Known room words. A declarative that names one of these ("this is the <word>") enrolls
# even without Rex having asked; other custom names ("the lab") are accepted only as the
# answer to his own question, so a stray "this is Sarah" can never mint a room. Extend
# freely in user_config.py.
PLACE_ROOM_WORDS = [
    "living room", "family room", "great room", "dining room", "rec room", "sitting room",
    "master bedroom", "guest bedroom", "guest room", "bedroom", "nursery", "playroom",
    "bathroom", "powder room", "kitchen", "kitchenette", "pantry", "office", "study",
    "den", "library", "garage", "hallway", "hall", "entryway", "foyer", "mudroom",
    "laundry room", "basement", "cellar", "attic", "loft", "closet", "workshop",
    "workshop", "shop", "lab", "studio", "gym", "home gym", "sunroom", "porch",
    "patio", "deck", "conservatory", "utility room", "game room", "media room",
    "theater", "bar", "cave",
]
# Spoken when Rex proactively asks what room he's in (LLM instruction, not verbatim).
PLACE_QUESTION_TEMPLATES = [
    "Ask, in character and briefly, what room you're in — you don't recognize this "
    "place and you'd like to know it so you can remember it next time.",
    "You don't recognize where you are right now. Ask them, curiously and in one short "
    "line, what room this is.",
]
# Spoken acknowledgements the instant a room is named (verbatim; {name} filled in).
PLACE_ENROLL_ACK_TEMPLATES = [
    "Got it — the {name}. I'll remember this place.",
    "The {name}. Noted. I'll know it next time.",
    "Ah, the {name}. Filing this one away.",
    "The {name} it is. Locking it into memory.",
]
# Variant when the named room is one he ALREADY knows (he still tops up its gallery).
PLACE_KNOWN_ACK_TEMPLATES = [
    "The {name} — yeah, I know this one. Taking another look anyway.",
    "Yep, the {name}. I recognize it. Refreshing my memory.",
    "The {name}, right. Good — my circuits agree.",
]
# Spoken when a human CONTRADICTS the believed room ("this is not the workshop").
# The belief is dropped and the real name invited — Rex never argues the point. Field
# 2026-07-24: the correction drew "Yep, the workshop. I recognize it." instead, which
# is both wrong and infuriating; a person standing in the room outranks a cosine score.
PLACE_DENIAL_ACK_TEMPLATES = [
    "My mistake — scratch the {was}. Where am I, then?",
    "Noted, not the {was}. What room is this?",
    "Fair enough, I had it wrong. What should I call this room?",
    "Wiping the {was} from the record. Where are we actually?",
]
# Spoken if a promised room capture later fails (he said "Got it" but couldn't get
# enough clear looks — e.g. someone stood in front of the lens the whole time).
PLACE_ENROLL_FAIL_TTS_ENABLED = True
PLACE_ENROLL_FAIL_TTS_LINES = [
    "Hey — small confession. I tried to memorize this room and couldn't get a good look. Tell me where we are again sometime?",
    "Update from my optics: this room did not save. Too much going on in front of my lens. We'll try again later.",
]
# Spoken when a freshly-taught room's views are near-identical to an existing room's
# (cross-sim >= the threshold): the gallery is broken from birth (field 2026-07-21: two
# rooms enrolled at sim 0.97 because his head was face-tracking the teller both times),
# so he owns it and asks to be shown around. {new} / {existing} are filled in.
PLACE_TWIN_WARN_TTS_SIM     = 0.95
PLACE_TWIN_WARN_TTS_LINES = [
    "Heads up — the {new} looks nearly identical to the {existing} from where I'm sitting. Point me at something distinctive or I'll mix them up.",
    "Honest sensor report: my snapshots of the {new} and the {existing} basically match. Want to show me around so I can tell them apart?",
]
# Head pans change the camera view exactly like chassis heading does, so the neck servo
# counts toward enrollment view diversity (compass + neck are summed when both exist).
# This maps the neck servo's full travel to degrees of camera pan.
PLACE_NECK_SPAN_DEG         = 120.0


# ─────────────────────────────────────────────────────────────────────────────
# USER OVERRIDES
# ─────────────────────────────────────────────────────────────────────────────
# Per-deployment overrides live in user_config.py (gitignored; copied from
# user_config.example.py by setup_macos.sh). It is imported LAST so its values win
# over every default defined above. If the file is missing or empty, the defaults
# above are used unchanged.
try:
    from user_config import *  # noqa: F401,F403  (intentional override layer)
except ImportError:
    pass

# Re-derive values that were computed from a base the user may have overridden in
# user_config.py. These aliases were evaluated at definition time (far above),
# BEFORE the import, so without this an override of e.g. NO_AUDIO_MODE would never
# reach them. (AUDIO_OUTPUT_SUPPRESSED ← NO_AUDIO_MODE and
# COMMON_FIRST_NAME_LAST_NAME_MIN_PERSON_TURNS ← LONG_CONVERSATION_MIN_EXCHANGES
# track bases that are intentionally NOT exposed in user_config, so they stay.)
# ACTION_ROUTER_MODEL no longer follows LLM_MODEL (decoupled 2026-08-02, see its
# definition above) — override it directly in user_config.py if needed.
STARTUP_BOOT_TTS_LINE = STARTUP_BOOT_TTS_LINES[0]
