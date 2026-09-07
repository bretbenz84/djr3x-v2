"""Choose an existing person for first CAM++ enrollment from independent evidence.

Never guess from engagement, a camera merely seeing a face, or the closest
voice profile. Callers supply actual interval mouth-motion observations or an
explicit self-identification matched to a person. Short introductions can be
continued using a temporary acoustic reference and the same sole named face.
"""

import time
import threading
import numpy as np


def target(*, observations, windows, explicit_person_id=None):
    visual_ids = {r.get('person_db_id') for r in observations if r.get('person_db_id') is not None}
    window_ids = {r.get('person_id') for r in windows if r.get('person_id') is not None}
    if len(visual_ids)>1 or len(window_ids)>1 or any(r.get('change_suspected') for r in windows):
        return None
    visual = next(iter(visual_ids), None)
    if explicit_person_id is not None:
        if ((visual is not None and visual != explicit_person_id)
                or any(pid != explicit_person_id for pid in window_ids)):
            return None
        return explicit_person_id
    rows = [r for r in observations if r.get('person_db_id') == visual
            and float(r.get('confidence') or 0) >= .5]
    if visual is not None and len(rows)>=3:
        return visual
    return None


def visible_identity(people):
    """One actually visible, named face; voice-only/cached slots do not qualify."""
    faces = [p for p in people if not p.get('face_missing')
             and p.get('face_visible') is not False
             and (p.get('face_visible') or p.get('face_box'))]
    if len(faces) != 1 or not faces[0].get('face_id'):
        return None
    return faces[0].get('person_db_id')

# A name identifies a person even when that clip is too short to train a model.
# Keep a temporary acoustic reference, never a durable print, until a suitable
# follow-up agrees. No camera mouth-motion signal is required.

_pending_lock = threading.RLock()
_pending = None


def clear_pending():
    global _pending
    with _pending_lock:
        _pending = None


def _unit(vector):
    if vector is None:
        return None
    v = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(v)
    return v / norm if v.size and np.isfinite(v).all() and norm > 1e-8 else None


def remember_introduction(person_id, embedding, session_id, *, now=None):
    """Remember confirmed identity for 60s; the short sample itself is not enrolled."""
    global _pending
    vector = _unit(embedding)
    if person_id is None or vector is None:
        return False
    with _pending_lock:
        _pending = dict(person_id=int(person_id), vector=vector, session_id=session_id,
                        at=time.monotonic() if now is None else now)
    return True


def pending_person(session_id, *, now=None):
    now = time.monotonic() if now is None else now
    with _pending_lock:
        if _pending is None:
            return None
        if _pending['session_id'] != session_id or not 0 <= now - _pending['at'] <= 60:
            clear_pending()
            return None
        return _pending['person_id']


def followup_target(embedding, session_id, *, visible_person_id, observations, windows,
                    now=None):
    """Require the introduced person's sole face plus acoustic agreement.

    Camera absence/another face or a conflicting voice retires the proposal.
    Failed similarity never moves or renews the original reference.
    """
    pid = pending_person(session_id, now=now)
    if pid is None:
        return None
    if visible_person_id != pid:
        clear_pending()
        return None
    if (any(r.get('change_suspected') or r.get('person_id') not in (None, pid) for r in windows)
            or any(r.get('person_db_id') not in (None, pid) for r in observations)):
        clear_pending()
        return None
    for row in observations:
        faces = row.get('faces') or []
        if visible_identity(faces) != pid:
            clear_pending()
            return None
    vector = _unit(embedding)
    with _pending_lock:
        if _pending is None:
            return None
        if vector is None or vector.shape != _pending['vector'].shape:
            clear_pending()
            return None
        # Similarity only corroborates an explicit introduction + sole known face.
        # It does not lower the ordinary recognition or enrollment-quality floors.
        if float(np.dot(vector, _pending['vector'])) < .35:
            clear_pending()
            return None
    return pid
