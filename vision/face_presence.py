"""Small capture-interval face history, independent of mouth/pose models."""
from collections import deque
import copy
import threading
import time

_lock = threading.Lock()
_history = deque(maxlen=1024)
_keys = ('id', 'track_id', 'person_db_id', 'face_id', 'face_box', 'bounding_box',
         'bbox', 'face_visible', 'face_missing')


def record(people):
    row = {'monotonic_at': time.monotonic(),
           'faces': [{key: copy.deepcopy(p[key]) for key in _keys if key in p} for p in people]}
    with _lock:
        _history.append(row)


def between(start, end):
    with _lock:
        return [copy.deepcopy(row) for row in _history if start <= row['monotonic_at'] <= end]
