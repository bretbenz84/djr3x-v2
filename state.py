import logging
import threading
from enum import Enum
from typing import Callable, List

logger = logging.getLogger(__name__)


class State(Enum):
    IDLE = "IDLE"
    QUIET = "QUIET"
    ACTIVE = "ACTIVE"
    SLEEP = "SLEEP"
    SHUTDOWN = "SHUTDOWN"


_lock = threading.Lock()
_state = State.IDLE
_callbacks: List[Callable[["State", "State"], None]] = []


def add_state_change_callback(cb: Callable[["State", "State"], None]) -> None:
    """Register a callback invoked as cb(old_state, new_state) on every transition."""
    _callbacks.append(cb)


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
    for cb in _callbacks:
        try:
            cb(old, new_state)
        except Exception as exc:
            logger.warning("State callback error: %s", exc)


def is_state(state: State) -> bool:
    with _lock:
        return _state == state
