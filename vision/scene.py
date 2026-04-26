"""
vision/scene.py — GPT-4o environment analysis, animal detection, and crowd counting.

All three analysis functions encode the frame as JPEG, send it to GPT-4o vision
with a structured JSON prompt, and return the parsed result. Each prompt ends with
an explicit instruction to return only the JSON object — no markdown, no preamble.
_parse_json() handles the cases where GPT-4o wraps the response in code fences anyway.

Environment analysis is cached: the cached result is returned when the crowd count
is stable (within _CROWD_CHANGE_DELTA people) AND less than
config.ENVIRONMENT_SCAN_INTERVAL_SECS has elapsed since the last real query.

start_periodic_scan() runs analyze_environment in a background thread. The thread
fires immediately on start, then re-fires every interval_secs OR whenever
world_state.crowd.count changes by _CROWD_CHANGE_DELTA or more.
"""

import json
import logging
import threading
import time
from typing import Optional

import config
from vision.image_utils import encode_jpeg_base64
from world_state import world_state

_log = logging.getLogger(__name__)

# ── Environment analysis cache ────────────────────────────────────────────────

_env_cache: Optional[dict] = None
_env_cache_time: float = 0.0
_env_cache_crowd: int = -1   # crowd count recorded at the time of the last analysis

# Re-analyze if crowd count has shifted by at least this many people
_CROWD_CHANGE_DELTA = 2

# ── Periodic scan state ───────────────────────────────────────────────────────

_scan_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _get_client():
    """Return an OpenAI client. Raises ImportError when apikeys or openai are missing."""
    try:
        import apikeys
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(f"vision/scene.py requires apikeys and openai: {exc}") from exc
    return OpenAI(api_key=apikeys.OPENAI_API_KEY)


def _encode_frame(frame) -> Optional[str]:
    """JPEG-encode a BGR frame and return base64, or None on failure."""
    encoded = encode_jpeg_base64(frame, quality=85)
    if encoded is None:
        _log.error("_encode_frame: JPEG encode failed")
        return None
    return encoded


def _parse_json(text: str):
    """
    Parse JSON from a GPT-4o response, tolerating markdown code-fence wrapping.

    GPT-4o occasionally returns valid JSON wrapped in code fences despite being
    instructed not to. Three strategies are tried in order:

    1. Direct json.loads() — succeeds for clean responses.
    2. Strip the opening fence line (```json or ```) and closing ```, then retry.
       Handles both ```json\\n{...}\\n``` and ```\\n{...}\\n```.
    3. Brace/bracket extraction — scan for the first { or [ and the last } or ],
       parse that substring. Handles responses with stray leading/trailing text.

    Returns the parsed object (dict or list) or None if all strategies fail.
    """
    stripped = text.strip()

    # Strategy 1: direct parse
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown code fences
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1:]
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    # Strategy 3: brace / bracket extraction
    for open_c, close_c in [("{", "}"), ("[", "]")]:
        start = stripped.find(open_c)
        end   = stripped.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                pass

    _log.error("_parse_json: all strategies failed on: %.120s", text)
    return None


def _call_gpt4o(
    frame,
    prompt: str,
    detail_key: str,
    *,
    max_tokens: int = 400,
) -> Optional[str]:
    """
    Send frame + prompt to GPT-4o vision. Returns the raw response string or None.
    detail_key is looked up in config.VISION_DETAIL for the image detail level.
    """
    b64 = _encode_frame(frame)
    if b64 is None:
        return None

    detail = config.VISION_DETAIL.get(detail_key, "low")

    try:
        client = _get_client()
    except ImportError as exc:
        _log.error("_call_gpt4o: %s", exc)
        return None

    try:
        response = client.chat.completions.create(
            model=config.VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url":    f"data:image/jpeg;base64,{b64}",
                            "detail": detail,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=max_tokens,
        )
    except Exception as exc:
        _log.error("_call_gpt4o [%s]: API error: %s", detail_key, exc)
        return None

    return response.choices[0].message.content.strip()


def _count_label(count: int) -> str:
    """Map a capped integer person count to a crowd label."""
    if count <= 1:
        return "alone"
    if count == 2:
        return "pair"
    if count <= 4:
        return "small_group"
    return "crowd"   # 5 means "5 or more" — the integer cap


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_environment(frame, force: bool = False) -> dict:
    """
    Analyze the scene in frame using GPT-4o vision (config detail: "scene_analysis").

    Returns a dict with keys: scene_type, indoor_outdoor, lighting, crowd_density,
    time_of_day, description, last_updated. Updates world_state.environment.

    Returns the cached result if the scene is unlikely to have changed:
      - crowd count has not shifted by _CROWD_CHANGE_DELTA or more, AND
      - less than config.ENVIRONMENT_SCAN_INTERVAL_SECS has elapsed.

    Set force=True to bypass the cache and always make a fresh API call (e.g. when
    the user explicitly asks "what do you see?").

    Returns the cached dict (or {}) without an API call on frame=None or on failure.
    """
    global _env_cache, _env_cache_time, _env_cache_crowd

    if frame is None:
        return _env_cache or {}

    # Cache check — skip the API call if the scene is likely unchanged
    now           = time.monotonic()
    current_crowd = world_state.get("crowd").get("count", 0)
    cache_age     = now - _env_cache_time
    crowd_stable  = abs(current_crowd - _env_cache_crowd) < _CROWD_CHANGE_DELTA

    if (not force
            and _env_cache is not None
            and cache_age < config.ENVIRONMENT_SCAN_INTERVAL_SECS
            and crowd_stable):
        _log.debug("analyze_environment: cache hit (age=%.0fs)", cache_age)
        return _env_cache

    prompt = (
        "Analyze the scene in this image and return a JSON object with exactly "
        "these keys:\n"
        '  "scene_type": a short label for the setting, e.g. "convention_floor", '
        '"office", "park", "restaurant", "home",\n'
        '  "indoor_outdoor": "indoor" or "outdoor",\n'
        '  "lighting": "bright", "moderate", or "dim",\n'
        '  "crowd_density": "empty", "sparse", "moderate", or "dense",\n'
        '  "time_of_day": "morning", "afternoon", "evening", "night", or "unknown",\n'
        '  "description": one concise sentence describing the scene.\n'
        "Return ONLY the JSON object — no preamble, no explanation, no markdown fences."
    )

    raw = _call_gpt4o(frame, prompt, "scene_analysis")
    if raw is None:
        return _env_cache or {}

    data = _parse_json(raw)
    if not isinstance(data, dict):
        _log.error("analyze_environment: expected dict, got: %.120s", raw)
        return _env_cache or {}

    result = {
        "scene_type":     data.get("scene_type"),
        "indoor_outdoor": data.get("indoor_outdoor"),
        "lighting":       data.get("lighting"),
        "crowd_density":  data.get("crowd_density"),
        "time_of_day":    data.get("time_of_day"),
        "description":    data.get("description"),
        "last_updated":   time.time(),
    }

    _env_cache       = result
    _env_cache_time  = now
    _env_cache_crowd = current_crowd

    world_state.update("environment", result)
    _log.info(
        "analyze_environment: %s / %s / %s / crowd=%s",
        result.get("scene_type"),
        result.get("indoor_outdoor"),
        result.get("lighting"),
        result.get("crowd_density"),
    )
    return result


def detect_animals(frame) -> list[dict]:
    """
    Detect animals in frame using GPT-4o vision (config detail: "animal_detection").

    Returns a list of dicts, each containing:
        id         str  — "animal_1", "animal_2", ...
        species    str  — common name, e.g. "dog", "cat", "parrot"
        position   str  — rough location in frame, e.g. "left side", "background right"
        last_seen  float — time.time() timestamp

    Updates world_state.animals. Returns [] if no animals are present or on failure.
    """
    if frame is None:
        return []

    prompt = (
        "Examine this image for any animals. "
        "Return a JSON array — one object per animal detected. "
        "If no animals are visible, return an empty array: []\n"
        "Each object must have exactly two keys:\n"
        '  "species": common name of the animal, e.g. "dog", "cat", "parrot",\n'
        '  "position": brief location in frame, e.g. "left side", "center", '
        '"background right", "foreground".\n'
        "Return ONLY the JSON array — no preamble, no explanation, no markdown fences."
    )

    raw = _call_gpt4o(frame, prompt, "animal_detection")
    if raw is None:
        return []

    data = _parse_json(raw)
    if not isinstance(data, list):
        _log.error("detect_animals: expected list, got: %.120s", raw)
        return []

    animals = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            continue
        species = entry.get("species")
        if not species:
            continue
        animals.append({
            "id":        f"animal_{i + 1}",
            "species":   species,
            "position":  entry.get("position", "unknown"),
            "last_seen": time.time(),
        })

    world_state.update("animals", animals)
    if animals:
        _log.info(
            "detect_animals: %d detected — %s",
            len(animals),
            [a["species"] for a in animals],
        )
    return animals


def count_crowd(frame) -> dict:
    """
    Count people in frame using GPT-4o vision (config detail: "scene_analysis").

    Returns a dict with:
        count        int  — people detected, capped at 5 (5 means "5 or more")
        count_label  str  — "alone" (0–1), "pair" (2), "small_group" (3–4), "crowd" (5+)

    Updates world_state.crowd, preserving the existing dominant_speaker value.
    Returns {"count": 0, "count_label": "alone"} on frame=None or failure.
    """
    _fallback = {"count": 0, "count_label": "alone"}

    if frame is None:
        return _fallback

    prompt = (
        "Count the number of people visible in this image. "
        "Return a JSON object with exactly two keys:\n"
        '  "count": integer — number of people visible. Use 5 to mean "5 or more".\n'
        '  "count_label": "alone" for 0–1 people, "pair" for 2, '
        '"small_group" for 3–4, "crowd" for 5 or more.\n'
        "Return ONLY the JSON object — no preamble, no explanation, no markdown fences."
    )

    raw = _call_gpt4o(frame, prompt, "scene_analysis")
    if raw is None:
        return _fallback

    data = _parse_json(raw)
    if not isinstance(data, dict):
        _log.error("count_crowd: expected dict, got: %.120s", raw)
        return _fallback

    try:
        count = min(int(data.get("count", 0)), 5)
    except (TypeError, ValueError):
        _log.warning("count_crowd: non-integer count in response — defaulting to 0")
        count = 0

    label = data.get("count_label") or _count_label(count)

    # Preserve dominant_speaker set by speaker-id pipeline — do not clobber it
    existing = world_state.get("crowd")
    result = {
        "count":             count,
        "count_label":       label,
        "dominant_speaker":  existing.get("dominant_speaker"),
        "last_updated":      time.time(),
    }

    world_state.update("crowd", result)
    _log.debug("count_crowd: %d people (%s)", count, label)
    return {"count": count, "count_label": label}


def describe_scene() -> str:
    """
    Return a short natural-language scene summary using the latest WorldState data.

    If a current camera frame is available, refresh the environment cache first.
    """
    try:
        from vision import camera
        frame = camera.get_frame()
        if frame is not None:
            analyze_environment(frame, force=True)
    except Exception as exc:
        _log.debug("describe_scene: camera refresh skipped: %s", exc)

    env = world_state.get("environment")
    crowd = world_state.get("crowd")
    animals = world_state.get("animals")

    parts = []

    description = env.get("description") or env.get("scene_type")
    if description:
        parts.append(description)

    count = crowd.get("count", 0) or 0
    if count == 0:
        parts.append("No people are visible")
    else:
        noun = "person" if count == 1 else "people"
        parts.append(f"{count} {noun} visible")

    if animals:
        animal_list = ", ".join(a.get("species", "unknown") for a in animals)
        parts.append(f"Animals spotted: {animal_list}")

    return ". ".join(parts) + "." if parts else "Nothing notable right now."


def describe_scene_detailed(frame) -> dict:
    """
    Return a detailed, safety-filtered visual summary for conversation hooks.

    This is intentionally separate from analyze_environment(): idle conversation
    needs concrete details such as clothing, objects, activities, and visible
    setup, while the environment scanner only needs a cheap room-level label.
    """
    if frame is None:
        return {}

    prompt = (
        "Analyze this image as visual context for a conversational robot. "
        "Return a JSON object with exactly these keys:\n"
        '  "overall_summary": one or two concise sentences about the scene,\n'
        '  "people": an array of objects with "position", "visible_clothing", '
        '"accessories", and "activity" fields; use empty strings when unclear,\n'
        '  "notable_details": an array of concrete visible details such as '
        "objects, decorations, screens, tools, furniture, logos, colors, or "
        "interesting layout details,\n"
        '  "conversation_hooks": an array of 3 to 6 short question ideas based '
        "only on visible, non-sensitive details.\n"
        "Safety rules: do not identify anyone. Do not infer or mention race, "
        "ethnicity, religion, politics, disability, health, attractiveness, body "
        "size, socioeconomic status, or other sensitive traits. Avoid reading "
        "private text on screens or documents. Focus on clothing, accessories, "
        "objects, activities, and environment. Return ONLY the JSON object — no "
        "markdown, no preamble."
    )

    raw = _call_gpt4o(
        frame,
        prompt,
        "active_conversation",
        max_tokens=700,
    )
    if raw is None:
        return {}

    data = _parse_json(raw)
    if not isinstance(data, dict):
        _log.error("describe_scene_detailed: expected dict, got: %.120s", raw)
        return {}

    return {
        "overall_summary": data.get("overall_summary") or "",
        "people": data.get("people") if isinstance(data.get("people"), list) else [],
        "notable_details": (
            data.get("notable_details")
            if isinstance(data.get("notable_details"), list)
            else []
        ),
        "conversation_hooks": (
            data.get("conversation_hooks")
            if isinstance(data.get("conversation_hooks"), list)
            else []
        ),
    }


# ── Periodic scan ─────────────────────────────────────────────────────────────

def start_periodic_scan(interval_secs: float) -> None:
    """
    Start a background thread that calls analyze_environment at regular intervals.

    The first scan fires immediately on start. Subsequent scans fire when:
      - interval_secs has elapsed since the last scan, OR
      - world_state.crowd.count has changed by _CROWD_CHANGE_DELTA or more.

    If a scan is already running it is stopped cleanly before the new one starts.
    """
    global _scan_thread
    stop()

    _stop_event.clear()
    _scan_thread = threading.Thread(
        target=_scan_loop,
        args=(interval_secs,),
        daemon=True,
        name="scene-scan",
    )
    _scan_thread.start()
    _log.info("Periodic scene scan started (interval=%.0fs)", interval_secs)


def stop() -> None:
    """Stop the periodic scan background thread if running."""
    global _scan_thread
    if _scan_thread is not None and _scan_thread.is_alive():
        _stop_event.set()
        _scan_thread.join(timeout=5.0)
    _scan_thread = None
    _stop_event.clear()


def _scan_loop(interval_secs: float) -> None:
    """
    Periodic scan worker. camera is imported lazily inside the thread to avoid
    a circular import at module load time (camera → scene is not needed; scene → camera
    is only needed at runtime inside this thread).
    """
    from vision import camera

    last_scan_time   = 0.0   # 0.0 ensures the first iteration fires immediately
    last_crowd_count = -1    # -1 sentinel means "never observed"

    while not _stop_event.is_set():
        now           = time.monotonic()
        current_crowd = world_state.get("crowd").get("count", 0)

        time_elapsed = (now - last_scan_time) >= interval_secs
        crowd_jumped = (last_crowd_count >= 0 and
                        abs(current_crowd - last_crowd_count) >= _CROWD_CHANGE_DELTA)

        if time_elapsed or crowd_jumped:
            if crowd_jumped:
                _log.debug(
                    "_scan_loop: crowd %d → %d — triggering rescan",
                    last_crowd_count, current_crowd,
                )
            frame = camera.get_frame()
            if frame is not None:
                analyze_environment(frame)
                if getattr(config, "ANIMAL_DETECTION_ENABLED", True):
                    detect_animals(frame)
            else:
                _log.debug("_scan_loop: no frame available — skipping scan")

            last_scan_time   = now
            last_crowd_count = current_crowd

        _stop_event.wait(1.0)

    _log.info("Periodic scene scan stopped")
