"""Conversational voice learning. No mouth-motion or camera-angle inference.

A spoken identity answer starts a hypothesis, not a biometric. Every accepted
turn must agree with the original anchor AND the accumulated cluster. The caller
supplies utterance-bound evidence after transcript/echo filtering, and commits a
completed batch atomically. Rejected room voices never move the anchor.
"""
from dataclasses import dataclass, field
import hashlib
import re
import threading
import time

import numpy as np

_last_mic_motion_at = 0.0


def mic_moved(reason='mic_or_body_moved'):
    """Invalidate both the proposal and in-flight audio before a physical action."""
    global _last_mic_motion_at
    _last_mic_motion_at = time.monotonic()
    import sys
    interaction = sys.modules.get('intelligence.interaction')
    learner = getattr(interaction, '_voice_learner', None)
    if learner:
        learner.cancel(reason)


def unit(value):
    v = np.asarray(value, dtype=np.float32).reshape(-1) if value is not None else np.array([])
    n = np.linalg.norm(v)
    return v / n if v.size and np.isfinite(v).all() and n > 1e-8 else None


def angular_distance(a, b):
    return abs((float(a) - float(b) + 180) % 360 - 180)


def visible_faces(people):
    return [p for p in people or [] if isinstance(p, dict) and not p.get('face_missing')
            and p.get('face_visible') is not False and (p.get('face_visible') or p.get('face_box'))]


def face_token(p):
    # A tracker id, when available, is more stable than a small face's name guess.
    return p.get('track_id') or p.get('id') or p.get('face_id') or p.get('person_db_id')


def position_for(pid, people):
    faces = visible_faces(people)
    def x(p):
        box = p.get('face_box') or p.get('bounding_box')
        return float(box[0]) + float(box[2]) / 2 if box else None
    if len(faces) <= 1 or any(x(p) is None for p in faces):
        return ''
    faces.sort(key=x)
    indexes = [i for i, p in enumerate(faces) if p.get('person_db_id') == pid]
    if len(indexes) != 1:
        return ''
    i = indexes[0]
    return 'on my left' if i == 0 else 'on my right' if i == len(faces) - 1 else 'in the middle'


@dataclass
class Policy:
    min_turns: int = 3
    min_voiced: float = 8.0
    sample_min_voiced: float = 2.0
    sample_min_words: int = 4
    max_clip_secs: float = 30.0
    anchor_cosine: float = .50
    short_anchor_cosine: float = .35
    cluster_cosine: float = .55
    foreign_cosine: float = .65
    foreign_margin: float = .07
    bearing_tolerance: float = 35.0
    window_secs: float = 60.0
    resume_secs: float = 300.0


@dataclass
class Sample:
    audio: np.ndarray
    embedding: np.ndarray
    rate: int
    voiced: float
    text: str
    started_at: float
    ended_at: float
    faces: list = field(default_factory=list)
    bearing: float | None = None
    direction_conflict: bool = False
    trusted: bool = True
    mixed: bool = False
    ranked: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def key(self):
        return hashlib.sha256(np.asarray(self.audio, dtype=np.float32).tobytes()).hexdigest()


@dataclass
class Proposal:
    person_id: int
    name: str
    session_id: object
    created_at: float
    token: object = None
    position: str = ''
    asked_at: float | None = None
    anchor: np.ndarray | None = None
    short_anchor: bool = True
    bearing: float | None = None
    samples: list = field(default_factory=list)
    seen: set = field(default_factory=set)
    ready_at: float = 0.0
    deadline: float = 0.0
    paused_at: float | None = None
    confirmed: bool = False
    peers: set = field(default_factory=set)


class Learner:
    def __init__(self, policy=None, *, clock=time.monotonic, hold=None, release=None):
        self.policy = policy or Policy()
        self.clock = clock
        self.hold = hold or (lambda duration: self.clock())
        self.release = release or (lambda: None)
        self.lock = threading.RLock()
        self.pending = None
        self.cooldowns = {}
        self.last_reason = 'idle'

    def reset(self):
        with self.lock:
            self.release()
            self.pending = None
            self.cooldowns.clear()
            self.last_reason = 'idle'

    def cancel(self, reason='cancelled'):
        with self.lock:
            p = self.pending
            if p:
                self.cooldowns[p.person_id] = self.clock() + self.policy.resume_secs
            self.release()
            self.pending = None
            self.last_reason = reason

    def tick(self, session_id):
        with self.lock:
            p = self.pending
            if not p:
                return
            now = self.clock()
            if p.session_id != session_id:
                self.reset()
            elif p.paused_at is not None and now - p.paused_at >= self.policy.resume_secs:
                self.cancel('resume_expired')
            elif p.deadline and now >= p.deadline:
                self.release()
                p.paused_at = now
                p.deadline = 0
                self.last_reason = 'paused'
            elif not p.deadline and p.paused_at is None and now - p.created_at >= self.policy.window_secs:
                self.cancel('unspoken_request_expired')

    def request(self, pid, name, session_id, faces):
        with self.lock:
            self.tick(session_id)
            if self.pending or self.clock() < self.cooldowns.get(pid, 0):
                return False
            targets = [p for p in visible_faces(faces) if p.get('person_db_id') == pid]
            if len(targets) != 1:
                return False
            self.pending = Proposal(pid, name, session_id, self.clock(),
                                    token=face_token(targets[0]), position=position_for(pid, faces))
            self.pending.peers = {face_token(f) for f in visible_faces(faces)}
            return True

    def question(self):
        with self.lock:
            p = self.pending
            if not p or p.asked_at is not None or p.confirmed:
                return None
            first = p.name.split()[0]
            where = f'You {p.position} — ' if p.position else ''
            return f'{where}{first}, is that you speaking? I need to recalibrate my audio receptors.'

    def prepare_question(self):
        """Hold before playback; the answer window opens only on completion."""
        with self.lock:
            p = self.pending
            if p:
                p.ready_at = self.hold(self.policy.window_secs)
                p.deadline = self.clock() + self.policy.window_secs

    def question_spoken(self, completed):
        with self.lock:
            if not completed:
                self.cancel('question_interrupted')
            elif self.pending:
                self.pending.asked_at = self.clock()

    def _reject(self, reason):
        self.last_reason = reason
        return False

    def _valid(self, s, pid, *, seed=False):
        if (not s.trusted or s.mixed or s.direction_conflict or unit(s.embedding) is None
                or not np.isfinite(s.audio).all()):
            return self._reject('untrusted_mixed_or_invalid')
        if not 0 < s.rate or not 0 <= s.started_at < s.ended_at <= self.clock():
            return self._reject('invalid_capture_interval')
        if len(s.audio) / s.rate > self.policy.max_clip_secs or s.voiced < .2:
            return self._reject('capture_quality')
        # Confidence is relative. A strong other voice blocks a seed; once a
        # cluster exists it must also beat that other voice by a clear margin.
        for row in s.ranked:
            if row[0] != pid and float(row[2]) >= self.policy.foreign_cosine:
                return self._reject('foreign_enrolled_voice')
        p = self.pending
        if p and p.person_id == pid:
            if p.asked_at is not None and s.started_at < max(p.asked_at, p.ready_at):
                return self._reject('before_question_or_settling')
            if not seed and s.started_at < p.ready_at:
                return self._reject('servo_settling')
            # Capture snapshots contain faces independently of the unreliable
            # mouth identity field. Several faces are allowed throughout.
            for faces in s.faces:
                if not visible_faces(faces) and p.token is None:
                    continue  # explicit off-camera introduction, acoustic chain only
                targets = [f for f in visible_faces(faces)
                           if (face_token(f) == p.token if p.confirmed and p.token is not None
                               else f.get('person_db_id') == pid)]
                if seed and p.token is None and not targets:
                    # A new name has not been attached to its face yet. The
                    # spoken answer may anchor acoustics, but may not choose
                    # between two unnamed faces. Later turns must show the name.
                    if len([f for f in visible_faces(faces) if f.get('person_db_id') is None]) <= 1:
                        continue
                if len(targets) != 1 or (p.token is not None and face_token(targets[0]) != p.token):
                    return self._reject('face_track_changed')
                peers = {face_token(f) for f in visible_faces(faces)}
                if (p.position and peers == p.peers and len(peers) > 1
                        and position_for(targets[0].get('person_db_id'), faces) != p.position):
                    return self._reject('people_changed_positions')
        return True

    def seed(self, pid, name, session_id, s, *, explicit=False):
        with self.lock:
            self.tick(session_id)
            p = self.pending
            if p and p.person_id != pid:
                return self._reject('another_enrollment_active')
            if not p:
                if not explicit or self.clock() < self.cooldowns.get(pid, 0):
                    return False
                self.pending = p = Proposal(pid, name, session_id, self.clock())
            if p.confirmed:
                return False
            if not self._valid(s, pid, seed=True):
                return False
            if not explicit and p.asked_at is None:
                return self._reject('confirmation_not_requested')
            if p.token is None and s.faces:
                faces = visible_faces(s.faces[-1])
                targets = [f for f in faces if f.get('person_db_id') == pid]
                if not targets:
                    targets = [f for f in faces if f.get('person_db_id') is None]
                if len(targets) == 1:
                    p.token = face_token(targets[0])
                    p.position = position_for(targets[0].get('person_db_id'), faces)
                    p.peers = {face_token(f) for f in faces}
            p.anchor = unit(s.embedding)
            p.short_anchor = s.voiced < self.policy.sample_min_voiced
            p.confirmed = True
            # If centering happens AFTER the name reply its direction is in the
            # old mic frame. Use the first later settled utterance as the bearing.
            already_held = p.ready_at > 0 and s.started_at >= p.ready_at
            p.bearing = s.bearing if already_held else None
            p.ready_at = self.hold(self.policy.window_secs)
            if not p.deadline:
                p.deadline = self.clock() + self.policy.window_secs
            p.paused_at = None
            # A name/yes can anchor the chain but never inflate the sample count.
            p.seen.add(s.key)
            if not p.short_anchor and len(s.text.split()) >= self.policy.sample_min_words and s.started_at >= p.ready_at:
                p.samples.append(s)
            self.last_reason = 'identity_confirmed_collecting'
            return True

    def confirm(self, s):
        with self.lock:
            p = self.pending
            if not p or p.confirmed or p.asked_at is None:
                return False
            clean = re.sub(r'[^\w\s]', '', s.text.casefold()).strip()
            if re.match(r'^(no|nope|not me|not now|later|wait)\b', clean):
                self.cancel('confirmation_declined')
                return False
            names = {re.sub(r'[^\w\s]', '', n.casefold()) for n in (p.name, p.name.split()[0])}
            clean = re.sub(r'\s+rex$', '', clean)
            answer = re.sub(r'^(?:yes|yeah|yep|yup)\s+', '', clean)
            accepted = answer in {'yes', 'yeah', 'yep', 'yup', 'it is', 'thats me', 'its me',
                                  'it is me', 'correct', 'right', 'thats right', 'thats correct',
                                  'you got it', 'i am'}
            accepted |= any(answer in {n, 'im ' + n, 'i am ' + n, 'my name is ' + n,
                                      'its ' + n, 'this is ' + n} for n in names)
            if not accepted:
                return self._reject('not_a_confirmation')
            return self.seed(p.person_id, p.name, p.session_id, s)

    def observe(self, s, session_id):
        """Return whether this utterance agrees; ready() separately gates storage."""
        with self.lock:
            self.tick(session_id)
            p = self.pending
            if not p or not p.confirmed:
                return False
            if not self._valid(s, p.person_id):
                return False
            if s.key in p.seen:
                return self._reject('duplicate_capture')
            v = unit(s.embedding)
            if v.shape != p.anchor.shape:
                self.cancel('backend_changed')
                return False
            bar = self.policy.short_anchor_cosine if p.short_anchor else self.policy.anchor_cosine
            anchor_score = float(v @ p.anchor)
            if anchor_score < bar:
                return self._reject('anchor_disagrees')
            if p.samples:
                centroid = unit(np.mean([x.embedding for x in p.samples], axis=0))
                score = float(v @ centroid)
                if score < self.policy.cluster_cosine:
                    return self._reject('cluster_disagrees')
                if any(row[0] != p.person_id and score - float(row[2]) < self.policy.foreign_margin
                       for row in s.ranked):
                    return self._reject('cluster_not_distinct_from_known_voice')
                s.metadata['cluster_cosine'] = score
            if p.paused_at is None and p.bearing is not None and s.bearing is not None:
                if angular_distance(p.bearing, s.bearing) > self.policy.bearing_tolerance:
                    return self._reject('bearing_changed')
            if p.paused_at is not None:
                # Reacquire a bounded hold; never bank the turn captured while
                # the mic was free to move. Keep the acoustic evidence in RAM.
                p.ready_at = self.hold(self.policy.window_secs)
                p.deadline = self.clock() + self.policy.window_secs
                p.paused_at = None
                p.bearing = None
                return self._reject('resuming_after_pause')
            if p.bearing is None and s.bearing is not None:
                p.bearing = s.bearing
            p.seen.add(s.key)
            s.metadata.update(anchor_cosine=anchor_score, anchor_min_cosine=bar,
                              cluster_min_cosine=self.policy.cluster_cosine,
                              identity_confirmation='spoken_answer', person_id=p.person_id,
                              confirmed_mic_bearing_deg=p.bearing)
            if s.voiced >= self.policy.sample_min_voiced and len(s.text.split()) >= self.policy.sample_min_words:
                p.samples.append(s)
            self.last_reason = 'consistent_turn'
            return True

    def ready(self):
        p = self.pending
        return bool(p and len(p.samples) >= self.policy.min_turns
                    and sum(s.voiced for s in p.samples) >= self.policy.min_voiced)

    def context(self):
        p = self.pending
        if not p:
            return None
        return (f'Voice learning for {p.name}: {len(p.samples)} accepted conversational samples. '
                'This is a provisional association, not a recognized voice. Continue ordinary conversation; '
                'never ask for recitation or claim the voice is saved. Do not infer who spoke from this note.')


def visible_identity(people):
    faces = visible_faces(people)
    return faces[0].get('person_db_id') if len(faces) == 1 and faces[0].get('face_id') else None
