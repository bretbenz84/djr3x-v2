"""Admission for optional local inference; foreground audio never waits here.

An admitted HTTP/GPU request cannot be preempted. Foreground work prevents new
optional requests until it finishes, while already running work has a timeout.
"""
import threading
from contextlib import contextmanager, ContextDecorator

_lock = threading.Lock()
_foreground = 0
_optional = False


class foreground(ContextDecorator):
    def __enter__(self):
        global _foreground
        with _lock:
            _foreground += 1
        return self

    def __exit__(self, *exc):
        global _foreground
        with _lock:
            _foreground -= 1
        return False


@contextmanager
def optional():
    global _optional
    with _lock:
        admitted = not _foreground and not _optional
        if admitted:
            _optional = True
    try:
        yield admitted
    finally:
        if admitted:
            with _lock:
                _optional = False
