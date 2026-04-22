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
        "last_sound_event": None,
        "last_updated": None,
    },
    "self_state": {
        "servo_positions": {
            "neck": 6000,
            "headlift": 6000,
            "headtilt": 4320,
            "visor": 6000,
        },
        "body_state": "neutral",
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

    def get(self, field: str):
        with self._lock:
            if field not in self._state:
                raise KeyError(f"Unknown WorldState field: {field!r}")
            return copy.deepcopy(self._state[field])

    def snapshot(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._state)


world_state = WorldState()
