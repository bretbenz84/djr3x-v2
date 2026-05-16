"""
vision/scene.py — scene analysis, local animal detection, and crowd counting.

OpenAI-backed scene helpers encode the frame as JPEG, send it to GPT-4o vision
with a structured JSON prompt, and return the parsed result. The live animal
monitor is separate: it uses a local MediaPipe object detector against the same
camera frame buffer and spends no OpenAI credits.

Environment analysis is cached: the cached result is returned when the crowd count
is stable (within _CROWD_CHANGE_DELTA people) AND less than
config.ENVIRONMENT_SCAN_INTERVAL_SECS has elapsed since the last real query.

start_periodic_scan() runs local animal detection frequently and full environment
analysis in a background thread. Full scene analysis fires immediately on start,
then re-fires every interval_secs OR whenever world_state.crowd.count changes by
_CROWD_CHANGE_DELTA or more.
"""

import json
import logging
import threading
import time
from typing import Optional

import config
from vision import animal_detector as local_animal_detector
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


def _confidence_allows(value) -> bool:
    if value is None:
        return True
    if isinstance(value, (int, float)):
        return float(value) >= 0.55
    text = str(value or "").strip().lower()
    if not text:
        return True
    try:
        return float(text) >= 0.55
    except ValueError:
        pass
    return text not in {"low", "uncertain", "maybe", "unknown", "none"}


def _animal_records_from_response(data, *, now: Optional[float] = None) -> list[dict]:
    if not isinstance(data, list):
        return []
    seen: set[tuple[str, str]] = set()
    animals = []
    timestamp = time.time() if now is None else now
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            continue
        species = str(entry.get("species") or "").strip()
        if not species:
            continue
        if not _confidence_allows(entry.get("confidence")):
            continue
        position = str(entry.get("position") or "unknown").strip() or "unknown"
        key = (species.lower(), position.lower())
        if key in seen:
            continue
        seen.add(key)
        animal = {
            "id": f"animal_{len(animals) + 1}",
            "species": species,
            "position": position,
            "last_seen": timestamp,
        }
        if "furred" in entry:
            animal["furred"] = bool(entry.get("furred"))
        if entry.get("confidence") is not None:
            animal["confidence"] = entry.get("confidence")
        animals.append(animal)
    return animals


def _update_crowd_count(count: int) -> dict:
    count = min(max(int(count), 0), 5)
    existing = world_state.get("crowd")
    result = {
        "count": count,
        "count_label": _count_label(count),
        "dominant_speaker": existing.get("dominant_speaker"),
        "last_updated": time.time(),
    }
    world_state.update("crowd", result)
    return result


def _visible_people_count_from_world_state() -> int:
    try:
        people = world_state.get("people") or []
    except Exception:
        return 0
    count = 0
    for person in people:
        if not isinstance(person, dict):
            continue
        if person.get("face_visible") is False or person.get("face_missing"):
            continue
        count += 1
    return min(max(count, 0), 5)


def _crowd_density_for_count(count: int) -> str:
    if count <= 0:
        return "empty"
    if count <= 2:
        return "sparse"
    if count <= 4:
        return "moderate"
    return "dense"


def _recent_animals(max_age_secs: float = 10.0) -> list[dict]:
    try:
        animals = world_state.get("animals") or []
    except Exception:
        return []
    now = time.time()
    recent: list[dict] = []
    for animal in animals:
        if not isinstance(animal, dict):
            continue
        last_seen = animal.get("last_seen")
        try:
            if last_seen is not None and (now - float(last_seen)) > max_age_secs:
                continue
        except (TypeError, ValueError):
            pass
        recent.append(animal)
    return recent


def _ground_environment_with_local_telemetry(result: dict) -> dict:
    grounded = dict(result or {})
    local_people = _visible_people_count_from_world_state()
    if local_people > 0:
        crowd = world_state.get("crowd") or {}
        if int(crowd.get("count") or 0) < local_people:
            crowd = _update_crowd_count(local_people)
        grounded["crowd_density"] = _crowd_density_for_count(local_people)
        grounded["local_people_count"] = local_people
        grounded["local_crowd_label"] = crowd.get("count_label") or _count_label(local_people)
    animals = _recent_animals()
    if animals:
        grounded["animals_visible"] = [
            a.get("species", "unknown") for a in animals if isinstance(a, dict)
        ]
    return grounded


def _world_state_has_visible_people() -> bool:
    try:
        people = world_state.get("people") or []
        if any(
            isinstance(person, dict)
            and person.get("face_visible") is not False
            and not person.get("face_missing")
            for person in people
        ):
            return True
        return int((world_state.get("crowd") or {}).get("count", 0) or 0) > 0
    except Exception:
        return False


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
    current_crowd = max(
        int((world_state.get("crowd") or {}).get("count", 0) or 0),
        _visible_people_count_from_world_state(),
    )
    cache_age     = now - _env_cache_time
    crowd_stable  = abs(current_crowd - _env_cache_crowd) < _CROWD_CHANGE_DELTA

    if (not force
            and _env_cache is not None
            and cache_age < config.ENVIRONMENT_SCAN_INTERVAL_SECS
            and crowd_stable):
        _log.debug("analyze_environment: cache hit (age=%.0fs)", cache_age)
        grounded_cache = _ground_environment_with_local_telemetry(_env_cache)
        if grounded_cache != _env_cache:
            _env_cache = grounded_cache
            world_state.update("environment", grounded_cache)
        return grounded_cache

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
    result = _ground_environment_with_local_telemetry(result)

    _env_cache       = result
    _env_cache_time  = now
    _env_cache_crowd = max(current_crowd, _visible_people_count_from_world_state())

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

    animals = _animal_records_from_response(data)

    world_state.update("animals", animals)
    if animals:
        _log.info(
            "detect_animals: %d detected — %s",
            len(animals),
            [a["species"] for a in animals],
        )
    return animals


def detect_animals_local(frame) -> list[dict]:
    """
    Detect animals in frame using the local MediaPipe object detector.

    This is the live, no-OpenAI-credits path. It updates world_state.animals
    when the detector is available. If the local model is missing/unavailable,
    the existing animal state is preserved and returned.
    """
    if frame is None:
        return world_state.get("animals") or []

    animals = local_animal_detector.detect_animals(frame)
    if animals is None:
        return world_state.get("animals") or []

    world_state.update("animals", animals)
    if animals:
        _log.info(
            "detect_animals_local: %d detected — %s",
            len(animals),
            [a["species"] for a in animals],
        )
    else:
        _log.debug("detect_animals_local: no animals detected")
    return animals


def detect_lifeforms(frame) -> dict:
    """
    Low-token visual change scan for people count and animal arrivals.

    This is cheaper and more frequent than analyze_environment(). It updates
    world_state.crowd and world_state.animals, then returns the normalized result.
    """
    fallback = {
        "people_count": int((world_state.get("crowd") or {}).get("count", 0) or 0),
        "animals": world_state.get("animals") or [],
    }
    if frame is None:
        return fallback

    prompt = (
        "Do a low-cost visual change scan for a social robot. "
        "Return a JSON object with exactly these keys:\n"
        '  "people_count": integer number of clearly visible people, capped at 5,\n'
        '  "animals": array of visible real animals, not toys/logos/screens. '
        "Each animal object must have exactly: "
        '"species" (common name), "position" (brief location), '
        '"furred" (true if it appears furry/hairy), and '
        '"confidence" ("low", "medium", or "high").\n'
        "Only include animals you can actually see; if no animal is visible use []. "
        "Return ONLY the JSON object — no preamble, no explanation, no markdown fences."
    )

    raw = _call_gpt4o(
        frame,
        prompt,
        "animal_detection",
        max_tokens=int(getattr(config, "SCENE_CHANGE_MONITOR_MAX_TOKENS", 260) or 260),
    )
    if raw is None:
        return fallback

    data = _parse_json(raw)
    if not isinstance(data, dict):
        _log.error("detect_lifeforms: expected dict, got: %.120s", raw)
        return fallback

    try:
        people_count = min(max(int(data.get("people_count", 0)), 0), 5)
    except (TypeError, ValueError):
        people_count = fallback["people_count"]
    crowd = _update_crowd_count(people_count)

    animals = _animal_records_from_response(data.get("animals") or [])
    if not animals:
        animals = _recent_animals()
    world_state.update("animals", animals)

    if animals:
        _log.info(
            "detect_lifeforms: people=%d animals=%s",
            people_count,
            [a["species"] for a in animals],
        )
    else:
        _log.debug("detect_lifeforms: people=%d animals=0", people_count)
    return {
        "people_count": people_count,
        "count_label": crowd.get("count_label"),
        "animals": animals,
    }


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


def analyze_directed_attention(
    frame,
    *,
    direction: str = "current",
    utterance: str = "",
    target_hint: str = "",
) -> dict:
    """
    Analyze the view Rex was explicitly asked to look at.

    Used after the head/neck servos move for commands such as "look left" or
    "look at this". The output is structured so the dialogue layer can turn it
    into a short roast-style observation without inventing visual details.
    """
    if frame is None:
        return {}

    direction = (direction or "current").strip().lower()
    direction_note = {
        "left": "Rex has turned his head toward his own left.",
        "right": "Rex has turned his head toward his own right.",
        "up": "Rex has tilted his camera upward.",
        "down": "Rex has tilted his camera downward.",
        "center": "Rex has centered his gaze.",
        "current": "Rex is inspecting the current view; the user may be pointing or showing something nearby.",
    }.get(direction, "Rex is inspecting the current view.")

    prompt = (
        "You are analyzing an image from DJ-R3X's camera after a person told him "
        f"to look somewhere. {direction_note}\n"
        f"Original spoken request: {utterance!r}\n"
        f"Target hint extracted from the request: {target_hint!r}\n\n"
        "Decide what the person most likely wants Rex to notice. Prioritize "
        "salient objects, room features, people at the edge of view, children "
        "low in the frame, pets, or something being held/shown to the camera. Return a "
        "JSON object with exactly these keys:\n"
        '  "target_summary": one concise sentence describing the likely target,\n'
        '  "target_visible": boolean — true if the requested target/hint is '
        "actually visible; false if it is missing or unclear,\n"
        '  "subject_type": one of "person", "animal", "object", "room_feature", '
        '"screen", "unknown",\n'
        '  "visible_people_count": integer count of visible people, capped at 5,\n'
        '  "animals": array of objects with "species" and "position", or [],\n'
        '  "notable_details": array of up to 5 concrete visible details,\n'
        '  "roast_angle": one friendly roast/opinion based only on visible '
        "non-sensitive details,\n"
        '  "confidence": "low", "medium", or "high".\n'
        "Safety rules: do not identify anyone. Do not infer or mention race, "
        "ethnicity, religion, politics, disability, health, attractiveness, body "
        "size, or socioeconomic status. Do not read private text on documents or "
        "screens. Keep the roast about objects, decor, staging, droid-level taste, "
        "or general harmless chaos. Return ONLY the JSON object — no markdown, "
        "no preamble."
    )

    raw = _call_gpt4o(
        frame,
        prompt,
        "active_conversation",
        max_tokens=650,
    )
    if raw is None:
        return {}

    data = _parse_json(raw)
    if not isinstance(data, dict):
        _log.error("analyze_directed_attention: expected dict, got: %.120s", raw)
        return {}

    animals_raw = data.get("animals") if isinstance(data.get("animals"), list) else []
    animals = []
    for entry in animals_raw:
        if not isinstance(entry, dict):
            continue
        species = str(entry.get("species") or "").strip()
        if not species:
            continue
        animals.append({
            "species": species,
            "position": str(entry.get("position") or "unknown").strip() or "unknown",
        })

    try:
        visible_people_count = min(max(int(data.get("visible_people_count", 0)), 0), 5)
    except (TypeError, ValueError):
        visible_people_count = 0

    result = {
        "target_summary": str(data.get("target_summary") or "").strip(),
        "target_visible": bool(data.get("target_visible", False)),
        "subject_type": str(data.get("subject_type") or "unknown").strip() or "unknown",
        "visible_people_count": visible_people_count,
        "animals": animals,
        "notable_details": (
            data.get("notable_details")
            if isinstance(data.get("notable_details"), list)
            else []
        )[:5],
        "roast_angle": str(data.get("roast_angle") or "").strip(),
        "confidence": str(data.get("confidence") or "low").strip() or "low",
    }

    if animals:
        now = time.time()
        world_state.update("animals", [
            {
                "id": f"animal_{i + 1}",
                "species": item["species"],
                "position": item["position"],
                "last_seen": now,
            }
            for i, item in enumerate(animals)
        ])

    _log.info(
        "analyze_directed_attention: direction=%s subject=%s confidence=%s summary=%r",
        direction,
        result.get("subject_type"),
        result.get("confidence"),
        result.get("target_summary"),
    )
    return result


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
    monitor_interval = float(getattr(config, "SCENE_CHANGE_MONITOR_INTERVAL_SECS", 20.0) or 20.0)
    local_animal_interval = float(
        getattr(config, "LOCAL_ANIMAL_DETECTION_INTERVAL_SECS", 2.0) or 2.0
    )
    _log.info(
        "Periodic scene scan started (interval=%.0fs, change_monitor=%.0fs, local_animals=%.1fs)",
        interval_secs,
        monitor_interval if getattr(config, "SCENE_CHANGE_MONITOR_ENABLED", True) else 0.0,
        local_animal_interval if getattr(config, "LOCAL_ANIMAL_DETECTION_ENABLED", True) else 0.0,
    )


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
    last_monitor_time = 0.0
    last_local_animal_time = 0.0
    last_crowd_count = -1    # -1 sentinel means "never observed"

    while not _stop_event.is_set():
        now           = time.monotonic()
        current_crowd = world_state.get("crowd").get("count", 0)
        monitor_interval = max(
            5.0,
            float(getattr(config, "SCENE_CHANGE_MONITOR_INTERVAL_SECS", 20.0) or 20.0),
        )
        local_animal_interval = max(
            0.5,
            float(getattr(config, "LOCAL_ANIMAL_DETECTION_INTERVAL_SECS", 2.0) or 2.0),
        )

        time_elapsed = (now - last_scan_time) >= interval_secs
        monitor_elapsed = (now - last_monitor_time) >= monitor_interval
        local_animal_elapsed = (now - last_local_animal_time) >= local_animal_interval
        crowd_jumped = (last_crowd_count >= 0 and
                        abs(current_crowd - last_crowd_count) >= _CROWD_CHANGE_DELTA)

        if (
            getattr(config, "LOCAL_ANIMAL_DETECTION_ENABLED", True)
            and getattr(config, "ANIMAL_DETECTION_ENABLED", True)
            and local_animal_elapsed
        ):
            frame = camera.get_frame()
            if frame is not None:
                detect_animals_local(frame)
                last_local_animal_time = now
            else:
                _log.debug("_scan_loop: no frame available — skipping local animal detector")

        if time_elapsed or crowd_jumped:
            if crowd_jumped:
                _log.debug(
                    "_scan_loop: crowd %d → %d — triggering rescan",
                    last_crowd_count, current_crowd,
                )
            frame = camera.get_frame()
            if frame is not None:
                analyze_environment(frame)
                if (
                    getattr(config, "ANIMAL_DETECTION_ENABLED", True)
                    and not getattr(config, "LOCAL_ANIMAL_DETECTION_ENABLED", True)
                ):
                    detect_lifeforms(frame)
                    last_monitor_time = now
            else:
                _log.debug("_scan_loop: no frame available — skipping scan")

            last_scan_time   = now
            last_crowd_count = world_state.get("crowd").get("count", current_crowd)
        elif (
            getattr(config, "SCENE_CHANGE_MONITOR_ENABLED", True)
            and getattr(config, "ANIMAL_DETECTION_ENABLED", True)
            and not getattr(config, "LOCAL_ANIMAL_DETECTION_ENABLED", True)
            and monitor_elapsed
            and (
                not bool(getattr(config, "SCENE_CHANGE_MONITOR_ONLY_WITH_PEOPLE", True))
                or _world_state_has_visible_people()
            )
        ):
            frame = camera.get_frame()
            if frame is not None:
                detect_lifeforms(frame)
                last_monitor_time = now
                last_crowd_count = world_state.get("crowd").get("count", current_crowd)
            else:
                _log.debug("_scan_loop: no frame available — skipping change monitor")

        _stop_event.wait(1.0)

    _log.info("Periodic scene scan stopped")
