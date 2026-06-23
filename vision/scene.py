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


def _resolve_known_names(known_names) -> list[str]:
    """Names of recognized visible people to fold into a vision prompt.

    Pass an explicit list to override; None auto-resolves from world_state via
    dlib face recognition (`vision.face.visible_known_names`). This is what lets
    the GPT-4o descriptions say "Bret" instead of "a man" — the identity comes
    from local face recognition, not from the vision model guessing.
    """
    if known_names is not None:
        return [str(n).strip() for n in known_names if str(n).strip()]
    try:
        from vision import face
        return face.visible_known_names()
    except Exception:
        return []


def _known_people_clause(names: list[str]) -> str:
    """Prompt fragment instructing the vision model to use recognized names."""
    if not names:
        return ""
    return (
        " The following people in view are KNOWN to you by name: "
        + ", ".join(names)
        + '. When you describe a visible person who is clearly one of them, refer '
        'to them BY NAME instead of "a man", "a woman", "a person", or "someone". '
        "If several people are visible and you cannot tell which name belongs to "
        "which, describe them generically rather than guessing."
    )


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

def analyze_environment(frame, force: bool = False, known_names=None) -> dict:
    """
    Analyze the scene in frame using GPT-4o vision (config detail: "scene_analysis").

    Returns a dict with keys: scene_type, indoor_outdoor, lighting, crowd_density,
    time_of_day, description, last_updated. Updates world_state.environment.

    Returns the cached result if the scene is unlikely to have changed:
      - crowd count has not shifted by _CROWD_CHANGE_DELTA or more, AND
      - less than config.ENVIRONMENT_SCAN_INTERVAL_SECS has elapsed.

    Set force=True to bypass the cache and always make a fresh API call (e.g. when
    the user explicitly asks "what do you see?").

    ``known_names`` folds dlib face-recognition identity into the description so a
    recognized person is named ("Bret is at his desk") instead of anonymized
    ("a man at a desk"). None auto-resolves the currently-visible recognized
    people from world_state; pass an explicit list to override. Naming costs no
    extra vision spend — it's the same image with a slightly longer text prompt.

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
    prompt += _known_people_clause(_resolve_known_names(known_names))

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

    animals = _confirm_persistent_animals(animals)
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


# Per-species consecutive-scan streak so a flickering misdetection (a lamp the model
# waffles on as a "bird") must persist before it's treated as really present.
_animal_confirm_streak: dict[str, int] = {}


def _confirm_persistent_animals(animals: list[dict]) -> list[dict]:
    """Keep only animals seen in ANIMAL_ARRIVAL_CONFIRM_SCANS consecutive scans. A real
    pet stays detected; a single-scan/oscillating misdetection never confirms (and so
    can't fire an arrival or churn the governor)."""
    need = int(getattr(config, "ANIMAL_ARRIVAL_CONFIRM_SCANS", 1))
    if need <= 1:
        return animals
    seen = {str(a.get("species") or "").strip().lower() for a in animals if a.get("species")}
    for sp in list(_animal_confirm_streak):
        if sp not in seen:
            del _animal_confirm_streak[sp]  # missed this scan → streak broken
    for sp in seen:
        _animal_confirm_streak[sp] = _animal_confirm_streak.get(sp, 0) + 1
    return [
        a for a in animals
        if _animal_confirm_streak.get(str(a.get("species") or "").strip().lower(), 0) >= need
    ]


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


def _scan_for_startle_species(frame) -> list[dict]:
    """Low-frequency OpenAI scan for STARTLE species (snakes/spiders/wasps/...) that the
    local MediaPipe detector can't see (it only knows bird/cat/dog/horse). ADDITIVE: merges
    any startle-species sighting into world_state.animals (dedup by species+position)
    WITHOUT clobbering locally-detected animals or touching the crowd count — unlike
    detect_lifeforms, which overwrites both. Returns the newly-added records (#29)."""
    if frame is None:
        return []
    startle = {
        str(s).strip().lower()
        for s in (getattr(config, "STARTLE_ANIMAL_SPECIES", set()) or set())
    }
    if not startle:
        return []
    prompt = (
        "Scan this image for any SMALL or DANGEROUS real creature a person would flinch "
        "at — snakes, spiders, scorpions, wasps, hornets, bees, rats, mice, bats, lizards, "
        "roaches. Ignore people, pet cats/dogs, toys, screens, and logos. "
        "Return ONLY a JSON object: "
        '{"animals": [{"species": "<common name>", "position": "<brief location>", '
        '"furred": <true|false>, "confidence": "<low|medium|high>"}]}. '
        "Use [] if you see none. No preamble, no markdown."
    )
    raw = _call_gpt4o(
        frame, prompt, "animal_detection",
        max_tokens=int(getattr(config, "SCENE_CHANGE_MONITOR_MAX_TOKENS", 260) or 260),
    )
    if raw is None:
        return []
    data = _parse_json(raw)
    if not isinstance(data, dict):
        return []
    records = _animal_records_from_response(data.get("animals") or [])
    fresh = [
        r for r in records
        if str(r.get("species", "")).strip().lower() in startle
    ]
    if not fresh:
        return []
    current = list(world_state.get("animals") or [])
    sigs = {
        (str(a.get("species", "")).strip().lower(), str(a.get("position", "")).strip().lower())
        for a in current
    }
    added = []
    for r in fresh:
        sig = (str(r.get("species", "")).strip().lower(), str(r.get("position", "")).strip().lower())
        if sig not in sigs:
            current.append(r)
            sigs.add(sig)
            added.append(r)
    if added:
        world_state.update("animals", current)
        _log.info("startle scan: merged %s", [a["species"] for a in added])
    return added


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


# Allowed normalized values for locate_people, used to coerce model output.
_PRESENCE_VERTICALS = ("low", "center", "high")
_PRESENCE_POSTURES = ("seated", "standing", "lying", "unknown")
_PRESENCE_CONFIDENCES = ("low", "medium", "high")


def _locate_people_fallback() -> dict:
    return {
        "present": False,
        "count": 0,
        "vertical": "center",
        "posture": "unknown",
        "confidence": "low",
    }


def locate_people(frame) -> dict:
    """Verify whether a person is present and read WHERE they are in frame.

    A startup fallback for when local dlib face detection finds nobody (wide-angle
    camera makes distant people tiny; dlib misses turned-away faces). Asks the
    vision model only for presence + frame position + posture so Rex can decide if
    the room is truly empty and, if not, greet the person at their height.

    Returns a dict:
        present     bool  — at least one real person visible (not a photo/screen)
        count       int   — people visible, 0..5 (5 means "5 or more")
        vertical    str   — "low" | "center" | "high" (where in the frame, not who)
        posture     str   — "seated" | "standing" | "lying" | "unknown"
        confidence  str   — "low" | "medium" | "high"

    Degrades to a safe "nobody, low confidence" fallback on frame=None, a missing
    API key, or a malformed response — never a false positive.
    """
    if frame is None:
        return _locate_people_fallback()

    prompt = (
        "You are the vision system of a social robot scanning a room for someone "
        "to greet. Report ONLY what is clearly visible. Return a JSON object with "
        "exactly these keys:\n"
        '  "present": boolean — true only if at least one real, live person is '
        "visible (NOT a photo, poster, screen, statue, or reflection),\n"
        '  "count": integer 0..5 (use 5 for "5 or more"),\n'
        '  "vertical": "low" if the nearest/most prominent person occupies the '
        "LOWER part of the frame (e.g. seated, crouched, reclining, or small/far "
        'and low), "high" if the UPPER part (e.g. standing and close/tall), '
        '"center" otherwise,\n'
        '  "posture": "seated", "standing", "lying", or "unknown",\n'
        '  "confidence": "low", "medium", or "high".\n'
        "Safety: do NOT identify anyone; do NOT infer or state age, whether someone "
        "is a child, race, health, body size, or any other sensitive trait. Judge "
        '"vertical" purely from WHERE IN THE FRAME the person appears, never from a '
        "guess about who they are.\n"
        "Return ONLY the JSON object — no preamble, no explanation, no markdown fences."
    )

    raw = _call_gpt4o(frame, prompt, "presence_scan", max_tokens=200)
    if raw is None:
        return _locate_people_fallback()

    data = _parse_json(raw)
    if not isinstance(data, dict):
        _log.error("locate_people: expected dict, got: %.120s", raw)
        return _locate_people_fallback()

    try:
        count = min(max(int(data.get("count", 0)), 0), 5)
    except (TypeError, ValueError):
        count = 0

    vertical = str(data.get("vertical") or "center").strip().lower()
    if vertical not in _PRESENCE_VERTICALS:
        vertical = "center"
    posture = str(data.get("posture") or "unknown").strip().lower()
    if posture not in _PRESENCE_POSTURES:
        posture = "unknown"
    confidence = str(data.get("confidence") or "low").strip().lower()
    if confidence not in _PRESENCE_CONFIDENCES:
        confidence = "low"

    present = bool(data.get("present")) or count > 0

    result = {
        "present": present,
        "count": count,
        "vertical": vertical,
        "posture": posture,
        "confidence": confidence,
    }
    _log.info(
        "locate_people: present=%s count=%d vertical=%s posture=%s confidence=%s",
        present, count, vertical, posture, confidence,
    )
    return result


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


def quick_caption(frame=None, known_people=None) -> str:
    """ONE cheap GPT-4o-mini vision call: a short plain caption of what's in front of
    Rex right now ("a cluttered workshop, dim light, one person at a desk"). Used for
    the once-per-run startup snapshot logged to episodic memory. Low detail + tiny
    token budget = minimal cost. Returns "" on any failure (no frame, no camera, API
    error) — never raises.

    ``known_people`` is a list of names of recognized people in view; when given, the
    caption refers to them BY NAME instead of "a man / a person", so Rex's first-person
    memory records WHO was there ("Bret at his desk") rather than a faceless stranger."""
    try:
        if frame is None:
            from vision import camera
            frame = camera.get_frame()
        if frame is None:
            return ""
        prompt = (
            "In ONE short sentence, plainly describe this room/scene — what kind of "
            "space it is, the lighting, how cluttered or tidy it looks, and roughly "
            "how many people (if any) are visible. Just the description, no preamble."
        )
        names = [str(n).strip() for n in (known_people or []) if str(n).strip()]
        if names:
            prompt += (
                " The following people in view are KNOWN to you — refer to each of them "
                "BY NAME, never as 'a man', 'a woman', 'a person', or 'someone': "
                + ", ".join(names) + "."
            )
        raw = _call_gpt4o(frame, prompt, "scene_analysis", max_tokens=120)
        return (raw or "").strip()
    except Exception as exc:
        _log.debug("quick_caption failed: %s", exc)
        return ""


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
    known_names=None,
) -> dict:
    """
    Analyze the view Rex was explicitly asked to look at.

    Used after the head/neck servos move for commands such as "look left" or
    "look at this". The output is structured so the dialogue layer can turn it
    into a short roast-style observation without inventing visual details.

    ``known_names`` lets Rex name people he already recognizes via dlib face
    recognition (so "what do you see?" answers "you, at your desk" instead of "a
    person at a desk"). None auto-resolves the currently-visible recognized
    people; pass a list to override. Other identity guessing stays disabled.
    """
    if frame is None:
        return {}

    names = _resolve_known_names(known_names)
    identity_rule = (
        "You MAY name these specific people, whom you already recognize: "
        + ", ".join(names)
        + " — refer to a visible person who is clearly one of them by name. Do "
        "NOT guess the identity of anyone else. "
        if names
        else "Do not identify anyone. "
    )

    direction = (direction or "current").strip().lower()
    direction_note = {
        "left": "Rex has turned his head toward his own left.",
        "right": "Rex has turned his head toward his own right.",
        "up": "Rex has tilted his camera upward.",
        "down": "Rex has tilted his camera downward.",
        "center": "Rex has centered his gaze.",
        "current": "Rex is inspecting the current view; the user may be pointing or showing something nearby.",
    }.get(direction, "Rex is inspecting the current view.")

    name_directive = ""
    if names:
        name_directive = (
            " If a visible person is clearly one of these people you recognize — "
            + ", ".join(names)
            + ' — refer to them BY NAME in "target_summary" and "notable_details" '
            'instead of "a person", "a man", or "a woman".'
        )

    prompt = (
        "You are analyzing an image from DJ-R3X's camera after a person told him "
        f"to look somewhere. {direction_note}\n"
        f"Original spoken request: {utterance!r}\n"
        f"Target hint extracted from the request: {target_hint!r}\n\n"
        "Decide what the person most likely wants Rex to notice. Prioritize "
        "salient objects, room features, people at the edge of view, children "
        "low in the frame, pets, or something being held/shown to the camera."
        f"{name_directive} Return a "
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
        f"Safety rules: {identity_rule}Do not infer or mention race, "
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
    last_startle_time = 0.0   # periodic startle-species scan (gap-fill when local is on)
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

        startle_interval = max(
            10.0,
            float(getattr(config, "STARTLE_DETECTION_INTERVAL_SECS", 60.0) or 60.0),
        )
        time_elapsed = (now - last_scan_time) >= interval_secs
        monitor_elapsed = (now - last_monitor_time) >= monitor_interval
        local_animal_elapsed = (now - last_local_animal_time) >= local_animal_interval
        startle_elapsed = (now - last_startle_time) >= startle_interval
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

        # Gap-fill: the local detector can't see snakes/spiders/wasps, so when it's the
        # active detector run a low-frequency OpenAI startle scan (people-present, paid —
        # so cheap cadence) and ADD any startle species so the startle reaction can fire (#29).
        if (
            getattr(config, "STARTLE_DETECTION_ENABLED", True)
            and getattr(config, "LOCAL_ANIMAL_DETECTION_ENABLED", True)
            and getattr(config, "ANIMAL_DETECTION_ENABLED", True)
            and startle_elapsed
            and _world_state_has_visible_people()
        ):
            frame = camera.get_frame()
            if frame is not None:
                try:
                    _scan_for_startle_species(frame)
                except Exception as exc:
                    _log.warning("_scan_loop: startle scan error: %s", exc)
                last_startle_time = now
            else:
                _log.debug("_scan_loop: no frame available — skipping startle scan")

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
