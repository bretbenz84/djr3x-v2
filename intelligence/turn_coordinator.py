"""Bounded pending input for the serial reply owner.

Keep completed captures in arrival order while Rex finishes his current reply.
This first adapter preserves the existing VAD/AEC capture rules; it does not
start a second response worker or speculate on ASR partials.
"""
from collections import deque
from dataclasses import dataclass
import threading
import time
import logging
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
            # A split of an earlier capture must precede newer queued captures.
            self._items = deque(sorted(self._items, key=lambda item: item.started_at))
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


class CaptureDuringReply:
    """One input producer alongside one serial response owner.

    scan(cursor) returns completed captures and the last consumed sample time.
    It does no ASR, identity writes or generation. Closing joins the producer
    before returning capture ownership to live VAD; unfinished audio remains in
    the existing recovery window. Accepted audio is copied before ring expiry.
    """
    def __init__(self, scan, cursor, queue=pending, interval=0.5):
        self.scan, self.cursor, self.queue = scan, cursor, queue
        self.interval = interval
        self.stopping = threading.Event()
        self.thread = None
        self.error = None

    def __enter__(self):
        self.thread = threading.Thread(target=self._run, name="reply-input", daemon=True)
        self.thread.start()
        return self

    def _run(self):
        try:
            while not self.stopping.wait(self.interval):
                turns, cursor = self.scan(self.cursor)
                for turn in turns:
                    if not self.queue.put(turn):
                        logging.getLogger(__name__).warning("pending input overflow; capture dropped")
                self.cursor = max(self.cursor, cursor)
        except Exception as exc:
            self.error = exc
            logging.getLogger(__name__).exception("input producer failed; recovery retains ownership")

    def __exit__(self, *_):
        self.stopping.set()
        if self.thread is not None:
            self.thread.join()  # local bounded VAD only, never a provider request
