import logging
import threading
from enum import Enum

logger = logging.getLogger(__name__)


class State(Enum):
    IDLE = "IDLE"
    QUIET = "QUIET"
    ACTIVE = "ACTIVE"
    SLEEP = "SLEEP"
    SHUTDOWN = "SHUTDOWN"


_lock = threading.Lock()
_state = State.IDLE


def get_state() -> State:
    with _lock:
        return _state


def set_state(new_state: State) -> None:
    global _state
    with _lock:
        old = _state
        if old == new_state:
            return
        _state = new_state
    logger.info("State transition: %s → %s", old.value, new_state.value)


def is_state(state: State) -> bool:
    with _lock:
        return _state == state
