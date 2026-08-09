import copy
import threading

_DEFAULTS = {
    "people": [],
    "crowd": {
        "count": 0,
        "count_label": "alone",
        "dominant_speaker": None,
        "last_updated": None,
    },
    "animals": [],
    # Non-animal room objects from the local COCO detector (vision/scene.detect_objects_local).
    # Each: {id, label, position, last_seen, confidence, source, box?}. Substrate for §2
    # object-grounded curiosity / change detection / room model. Screens/devices, people,
    # and animals are filtered out at detection time (tracked elsewhere or never logged).
    "objects": [],
    "environment": {
        "scene_type": None,
        "indoor_outdoor": None,
        "lighting": None,
        "crowd_density": None,
        "time_of_day": None,
        "description": None,
        "last_updated": None,
    },
    "audio_scene": {
        "ambient_level": "moderate",
        "music_detected": False,
        "music_tempo": None,
        "laughter_detected": False,
        "applause_detected": False,
        "scream_detected": False,
        "sudden_loud_sound_detected": False,
        "group_chatter_detected": False,
        "group_chatter_until": None,
        "group_chatter_reason": None,
        "last_sound_event": None,
        "last_sound_event_seq": 0,
        "sound_events": [],
        "last_updated": None,
    },
    "self_state": {
        "servo_positions": {
            "neck": 6000,
            "headlift": 6000,
            "headtilt": 4320,
            "visor": 6000,
            "elbow": 6720,
            "hand": 6000,
            "pokerarm": 6000,
            "heroarm": 6000,
        },
        "manual_servo_override": False,
        "body_state": "neutral",
        "face_tracking": {
            "locked": False,
            "visible": False,
            "holding_lost_lock": False,
            "searching": False,
            "search_reason": None,
            "search_pose": None,
            "lock_key": None,
            "person_id": None,
            "last_seen_at": None,
            "lost_age_secs": None,
        },
        "last_directed_look": None,
        "last_directed_look_at": None,
        "last_look_target": None,
        "emotion": "neutral",
        "anger_level": 0,
        "cpu_temp": None,
        "cpu_load": None,
        "uptime_seconds": 0,
        "session_interaction_count": 0,
        "last_interaction_ago": None,
    },
    "time": {
        "time_of_day": None,
        "hour": None,
        "day_of_week": None,
        "is_weekend": None,
        "notable_date": None,
    },
    "weather": {
        "location": None,
        "condition": "unknown",
        "temp_f": None,
        "feels_like_f": None,
        "humidity": None,
        "wind_mph": None,
        "description": "unknown",
        "available": False,
        "source": None,
        "fetched_at": None,
        "updated_at": None,
        "mood_bias": "unknown",
        "tone_hint": None,
    },
    # Visual place recognition (perception/place_recognition.py). None until a room
    # belief is confirmed via hysteresis; otherwise a dict:
    #   {"name": str, "place_id": int, "score": float, "since_ts": float}.
    "current_place": None,
    "social": {
        # Set when Rex hears a referential / instructional mention of himself
        # (someone talking ABOUT him, not TO him). Consciousness reads this to
        # decide whether to chime in. Active flag is computed from
        # last_mention_at by the situation assessor / consciousness step.
        "being_discussed": {
            "last_mention_at": None,    # epoch seconds (time.time())
            "last_snippet": None,       # the transcribed utterance
            "speaker_id": None,         # person_db_id of who said it (or None)
            "speaker_name": None,
            "addressee_id": None,       # who they were talking to (or None)
            "label": None,              # "referential" or "instructional"
            "sentiment": None,          # "positive" / "neutral" / "negative"
            "mentions_in_window": 0,    # rolling count in last 60 s
            "chimed_in": False,         # set by consciousness after a chime-in
        },
    },
}


class WorldState:
    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        with cls._instance_lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._lock = threading.Lock()
                instance._state = copy.deepcopy(_DEFAULTS)
                cls._instance = instance
        return cls._instance

    def update(self, field: str, value) -> None:
        with self._lock:
            if field not in self._state:
                raise KeyError(f"Unknown WorldState field: {field!r}")
            self._state[field] = value

    def mutate(self, field: str, fn):
        """Atomically read-modify-write a field while holding the lock.

        `fn` receives a deep copy of the current value and returns the new value
        to store, or None to leave the field unchanged. It runs with the lock
        held, so it must be fast (pure in-memory work) and must NOT call back
        into WorldState — get/update/mutate/snapshot would deadlock.

        Use this instead of get()+update() whenever the new value depends on the
        current one, so concurrent writers cannot silently overwrite each other
        (the lost-update race). Returns a deep copy of the resulting value.
        """
        with self._lock:
            if field not in self._state:
                raise KeyError(f"Unknown WorldState field: {field!r}")
            current = copy.deepcopy(self._state[field])
            updated = fn(current)
            if updated is None:
                return current
            self._state[field] = updated
            return copy.deepcopy(updated)

    def get(self, field: str):
        with self._lock:
            if field not in self._state:
                raise KeyError(f"Unknown WorldState field: {field!r}")
            return copy.deepcopy(self._state[field])

    def snapshot(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._state)


world_state = WorldState()
