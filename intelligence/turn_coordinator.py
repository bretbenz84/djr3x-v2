"""Bounded pending input for the serial reply owner.

Keep completed captures in arrival order while Rex finishes his current reply.
This first adapter preserves the existing VAD/AEC capture rules; it does not
start a second response worker or speculate on ASR partials.
"""
from collections import deque
from dataclasses import dataclass
import threading
import time
import numpy as np


@dataclass(frozen=True)
class CapturedTurn:
    audio: np.ndarray
    started_at: float
    ended_at: float
    session: int
    require_trusted: bool = False


class PendingTurns:
    def __init__(self, capacity=4, max_age=60.0):
        self.capacity = capacity
        self.max_age = max_age
        self._items = deque()
        self._lock = threading.Lock()

    def put(self, turn: CapturedTurn) -> bool:
        with self._lock:
            if len(self._items) >= self.capacity:
                return False  # preserve older accepted input; caller reports overflow
            self._items.append(turn)
            return True

    def pop(self, session: int):
        with self._lock:
            while self._items:
                turn = self._items.popleft()
                if turn.session == session and time.monotonic() - turn.ended_at <= self.max_age:
                    return turn
        return None

    def clear(self):
        with self._lock:
            self._items.clear()


pending = PendingTurns()
