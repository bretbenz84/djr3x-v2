"""
perception/place_recognition.py — Visual Place Recognition (VPR) for DJ-R3X.

WHAT THIS IS
    Answers a single question at ~0.5–1 Hz: *which enrolled room is Rex looking at
    right now?* It embeds the live camera frame with an image encoder (MobileCLIP by
    convention, but see "Dependency injection" below — the module is model-agnostic),
    compares it against a small gallery of per-room embeddings, and publishes a
    debounced belief. It also runs a small enrollment state machine so a room can be
    taught by name ("this is the office").

    This is a PURE LEAF. It imports no TTS/LLM/conversation modules. It talks to the
    rest of Rex only two ways:
      * writes ``world_state.current_place`` (the debounced belief), and
      * emits named events (below) for higher layers to react to.
    Deciding what to *say* about a place — or asking "what room is this?" — belongs to
    ``intelligence.conversation_agenda``, NOT here.

    Explicitly out of scope: doorway/edge detection, room graphs, position-within-room,
    navigation, depth, and the camera undistortion pipeline (frames arrive undistorted).

WORLD_STATE CONTRACT
    ``world_state.current_place`` is either ``None`` (no confirmed belief) or::

        {"name": str, "place_id": int, "score": float, "since_ts": float}

    Written through the standard ``world_state.update("current_place", value)`` API, so
    the field must exist in ``world_state._DEFAULTS`` (it does). It flips only through
    temporal hysteresis + the motion gate — never on a single frame.

EVENTS EMITTED  (via the injected ``emit_event(name, payload)``; consumed elsewhere)
    "unknown_place"           -> {"streak": int, "last_score": float|None}
        Rex has moved and then failed to recognize the room for a sustained streak.
        Fired at most once per "lost" episode (re-armed by the next confident match or
        a completed enrollment). conversation_agenda owns any "what room is this?" ask.
    "possible_duplicate_place"-> {"new_place", "new_place_id", "existing_place",
                                  "existing_place_id", "similarity"}
        A room being enrolled looks like one already known. Await ``confirm_duplicate``.
    "place_enrolled"          -> {"name", "place_id", "embeddings"}
        A room was successfully learned/updated.
    "enrollment_failed"       -> {"name", "place_id", "reason", "collected"}
        Enrollment aborted (timeout with too few usable frames).

THREADING
    Queries (camera thread) and the enrollment API (conversation thread) may call in
    concurrently. All shared state — the in-memory gallery, the enrollment session, the
    belief, and every SQLite write — is guarded by a single re-entrant lock
    (``self._lock``). The SQLite connection is opened ``check_same_thread=False`` and is
    only ever touched under that lock, so writes are serialized regardless of which
    thread calls. Slow work (the injected ``embed_fn``, i.e. model inference) runs
    OUTSIDE the lock; only the fast in-memory/DB bookkeeping is inside it.

DEPENDENCY INJECTION
    Everything the module needs from the outside is a constructor argument, which keeps
    it a testable leaf and lets the offline harness stub the whole world:
        embed_fn(frame) -> np.ndarray        raw image embedding (module L2-normalizes)
        get_heading() -> float | None        compass heading in degrees, or None
        get_motion_state() -> MotionState    wheels_moving / accel_active
        get_person_occlusion() -> float      largest person bbox as frac of frame [0,1]
        world_state                          object with .update(field, value)
        emit_event(name, payload)            event sink

STORAGE  (places.db — mirrors people.db: SQLite is the durable write-through store)
    On startup all embeddings matching the configured ``model_tag`` are loaded into an
    in-memory ``(N, dim)`` float32 matrix; queries are brute-force dot products (scale is
    ~10 rooms x <=15 embeddings — no index needed). Embeddings under other model tags are
    ignored, never deleted, so a model swap is non-destructive. Every vector is
    L2-normalized before it is stored, so similarity is a plain dot product.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional, Sequence

import numpy as np

import config

_log = logging.getLogger(__name__)


# ── Enrollment states ───────────────────────────────────────────────────────────
IDLE = "idle"
COLLECTING = "collecting"
CONFIRMING = "confirming"

# ── Query classifications ───────────────────────────────────────────────────────
CONFIDENT = "confident"
TENTATIVE = "tentative"
UNKNOWN = "unknown"

# ── Event names ─────────────────────────────────────────────────────────────────
EVENT_UNKNOWN_PLACE = "unknown_place"
EVENT_POSSIBLE_DUPLICATE = "possible_duplicate_place"
EVENT_PLACE_ENROLLED = "place_enrolled"
EVENT_ENROLLMENT_FAILED = "enrollment_failed"

WORLD_STATE_FIELD = "current_place"


def _cfg(name: str, default):
    """Read a PLACE_* tunable from config, tolerating a config.py without the block."""
    return getattr(config, name, default)


# ── Public value types ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MotionState:
    """Snapshot of whether Rex is physically moving. ``get_motion_state`` returns one."""
    wheels_moving: bool = False
    accel_active: bool = False  # IMU/accelerometer activity above rest

    @property
    def moving(self) -> bool:
        return bool(self.wheels_moving or self.accel_active)


@dataclass(frozen=True)
class PlaceScore:
    place_id: int
    name: str
    score: float          # mean of this place's top-k embedding similarities
    support: int          # how many embeddings actually contributed (min(k, n))


@dataclass(frozen=True)
class QueryResult:
    best: Optional[PlaceScore]          # highest-scoring place, or None (empty gallery)
    classification: str                 # CONFIDENT | TENTATIVE | UNKNOWN
    scores: tuple                       # all PlaceScore, sorted by score desc
    skipped: bool = False               # frame not scored (person occlusion / throttle)
    skip_reason: Optional[str] = None

    @property
    def is_confident(self) -> bool:
        return self.classification == CONFIDENT


@dataclass(frozen=True)
class EnrollResult:
    place_id: int
    name: str
    committed: int          # embeddings actually stored (after the per-place cap)
    provided: int


# ── Internal records ────────────────────────────────────────────────────────────

@dataclass
class _Embedding:
    embedding_id: int
    place_id: int
    vector: np.ndarray      # float32, L2-normalized
    heading_deg: Optional[float]
    captured_at: str        # ISO-8601, sortable


@dataclass
class _EnrollSession:
    name: str
    place_id: int
    created_new: bool                       # did enroll() create the place row?
    started_at: float
    last_capture_at: Optional[float] = None
    vectors: list = field(default_factory=list)     # list[np.ndarray] (normalized)
    headings: list = field(default_factory=list)    # list[float|None], parallel


def _circular_sep(a: float, b: float) -> float:
    """Smallest angular separation between two headings in degrees (350 vs 10 == 20)."""
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


class PlaceRecognizer:
    """Visual place recognition + enrollment. See the module docstring for the full
    architecture, event names, world_state contract, and thread-safety model."""

    def __init__(
        self,
        embed_fn: Callable[[np.ndarray], np.ndarray],
        *,
        get_heading: Callable[[], Optional[float]] = lambda: None,
        get_motion_state: Callable[[], MotionState] = lambda: MotionState(),
        get_person_occlusion: Callable[[], float] = lambda: 0.0,
        world_state=None,
        emit_event: Callable[[str, dict], None] = lambda name, payload: None,
        db_path: Optional[str] = None,
        model_tag: Optional[str] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._embed_fn = embed_fn
        self._get_heading = get_heading
        self._get_motion_state = get_motion_state
        self._get_person_occlusion = get_person_occlusion
        self._world_state = world_state
        self._emit_event = emit_event
        self._clock = clock

        # ── Tunables (getattr defaults keep the module usable without the config block) ──
        self._model_tag = model_tag or _cfg("PLACE_MODEL_TAG", "mobileclip_s2_v1")
        self._query_interval = float(_cfg("PLACE_QUERY_INTERVAL_S", 1.5))
        self._topk = max(1, int(_cfg("PLACE_TOPK", 3)))
        self._match_confident = float(_cfg("PLACE_MATCH_CONFIDENT", 0.80))
        self._match_min = float(_cfg("PLACE_MATCH_MIN", 0.68))
        self._hysteresis_frames = max(1, int(_cfg("PLACE_HYSTERESIS_FRAMES", 5)))
        self._majority = (self._hysteresis_frames // 2) + 1
        self._unknown_streak_max = max(1, int(_cfg("PLACE_UNKNOWN_STREAK", 8)))
        self._occlusion_frac = float(_cfg("PLACE_PERSON_OCCLUSION_FRAC", 0.35))
        self._enroll_target = max(1, int(_cfg("PLACE_ENROLL_TARGET_FRAMES", 8)))
        self._min_heading_sep = float(_cfg("PLACE_ENROLL_MIN_HEADING_SEP", 35.0))
        self._min_time_sep = float(_cfg("PLACE_ENROLL_MIN_TIME_SEP_S", 3.0))
        self._enroll_timeout = float(_cfg("PLACE_ENROLL_TIMEOUT_S", 60.0))
        self._dup_sim = float(_cfg("PLACE_DUPLICATE_SIM", 0.88))
        self._refresh_min = float(_cfg("PLACE_REFRESH_MIN", 0.70))
        self._refresh_max = float(_cfg("PLACE_REFRESH_MAX", 0.78))
        self._max_embeddings = max(1, int(_cfg("PLACE_MAX_EMBEDDINGS", 15)))
        # Minimum usable frames to commit an enrollment that timed out short of target.
        self._enroll_min_frames = min(3, self._enroll_target)

        # ── Shared state (all guarded by _lock) ──
        self._lock = threading.RLock()
        self._dim: Optional[int] = None
        self._places: dict = {}          # place_id -> name
        self._name_to_id: dict = {}      # name -> place_id
        self._embeddings: list = []      # list[_Embedding]
        self._matrix = np.zeros((0, 1), dtype=np.float32)   # (N, dim)
        self._emb_pids = np.zeros((0,), dtype=np.int64)     # (N,)

        self._state = IDLE
        self._enroll: Optional[_EnrollSession] = None
        self._pending_dup: Optional[dict] = None

        self._last_query_at = 0.0
        self._history = deque(maxlen=self._hysteresis_frames)  # (place_id|None, is_confident)
        self._current_place: Optional[dict] = None
        # Has any motion been seen since the last confirmed place? Gates BOTH the freeze
        # (belief can't flip while stationary) and the unknown_place event (only fires
        # after he's moved). Starts False so a stationary cold boot never emits
        # unknown_place; the very first belief still forms because the freeze gate is
        # additionally guarded by ``current_place is not None``.
        self._moved_since_confirm = False
        self._unknown_streak = 0
        self._unknown_armed = True
        # Carried-robot escape hatches (see _update_belief): the wheels never turn when
        # he's picked up and carried, so the motion gate would otherwise pin a stale
        # belief forever. Sustained contrary evidence overrides the silent sensor.
        self._static_flip_max = max(1, int(_cfg("PLACE_STATIC_FLIP_STREAK", 10)))
        self._lost_streak_max = max(1, int(_cfg("PLACE_LOST_STREAK", 16)))
        self._static_flip_pid: Optional[int] = None
        self._static_flip_streak = 0

        # ── Durable store ──
        self._db_path = self._resolve_db_path(db_path)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        try:
            self._init_schema()
            self._load_store()
        except Exception:
            self._conn.close()          # don't leak the connection on a bad store
            raise

    # ── Setup ────────────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_db_path(db_path: Optional[str]) -> str:
        db_path = db_path or _cfg("PLACE_DB_PATH", "data/places.db")
        if db_path == ":memory:":
            return db_path
        if not os.path.isabs(db_path):
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(root, db_path)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        return db_path

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS places (
                    place_id     INTEGER PRIMARY KEY,
                    name         TEXT UNIQUE NOT NULL,
                    created_at   TEXT NOT NULL,
                    notes        TEXT
                );
                CREATE TABLE IF NOT EXISTS place_embeddings (
                    embedding_id INTEGER PRIMARY KEY,
                    place_id     INTEGER NOT NULL REFERENCES places(place_id),
                    vector       BLOB NOT NULL,
                    dim          INTEGER NOT NULL,
                    heading_deg  REAL,
                    captured_at  TEXT NOT NULL,
                    model_tag    TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS place_observations (
                    observed_at  TEXT NOT NULL,
                    place_id     INTEGER,
                    score        REAL
                );
                CREATE INDEX IF NOT EXISTS idx_place_emb_place ON place_embeddings(place_id);
                CREATE INDEX IF NOT EXISTS idx_place_emb_tag   ON place_embeddings(model_tag);
                """
            )
            self._conn.commit()

    def _load_store(self) -> None:
        with self._lock:
            self._places.clear()
            self._name_to_id.clear()
            self._embeddings = []
            for pid, name in self._conn.execute("SELECT place_id, name FROM places"):
                self._places[pid] = name
                self._name_to_id[name] = pid
            loaded: list = []
            dim_counts: Counter = Counter()
            for eid, pid, blob, dim, heading, captured in self._conn.execute(
                "SELECT embedding_id, place_id, vector, dim, heading_deg, captured_at "
                "FROM place_embeddings WHERE model_tag = ?",
                (self._model_tag,),
            ):
                v = np.frombuffer(blob, dtype=np.float32)
                if v.shape[0] != dim:            # defensive: honor the stored dim
                    v = v[: int(dim)]
                dim_counts[int(v.shape[0])] += 1
                loaded.append(_Embedding(eid, pid, v.copy(), heading, captured))
            if len(dim_counts) > 1:
                # A model swap that reused the same model_tag left ragged rows behind.
                # np.stack would crash on unequal lengths, so DEGRADE (as the log has
                # always promised): keep the most common dim, ignore the rest. Nothing is
                # deleted from disk — fixing the tag or re-enrolling restores everything.
                self._dim = dim_counts.most_common(1)[0][0]
                kept = [e for e in loaded if e.vector.shape[0] == self._dim]
                _log.warning(
                    "places.db has mixed embedding dims %s under tag %s; keeping the %d "
                    "row(s) of dim %d and ignoring %d other(s) (nothing deleted)",
                    dict(dim_counts), self._model_tag, len(kept), self._dim,
                    len(loaded) - len(kept),
                )
                self._embeddings = kept
            else:
                self._embeddings = loaded
                if dim_counts:
                    self._dim = next(iter(dim_counts))
            self._rebuild_matrix()

    def _rebuild_matrix(self) -> None:
        """Rebuild the cached (N, dim) matrix + place-id vector from ``_embeddings``.
        Cheap at this scale (~150 rows); called after any gallery mutation."""
        if self._embeddings:
            self._matrix = np.stack([e.vector for e in self._embeddings]).astype(np.float32)
            self._emb_pids = np.array([e.place_id for e in self._embeddings], dtype=np.int64)
        else:
            self._matrix = np.zeros((0, self._dim or 1), dtype=np.float32)
            self._emb_pids = np.zeros((0,), dtype=np.int64)

    # ── Embedding helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n < 1e-12:
            return v
        return v / n

    def _embed(self, frame) -> np.ndarray:
        """Run the injected encoder and L2-normalize. Enforces a stable dimensionality
        across the whole store. Runs the (slow) model OUTSIDE the lock — callers must
        not hold ``_lock`` here."""
        raw = self._embed_fn(frame)
        v = np.asarray(raw, dtype=np.float32).reshape(-1)
        if v.size == 0:
            raise ValueError("embed_fn returned an empty vector")
        if self._dim is None:
            self._dim = int(v.size)
        elif v.size != self._dim:
            raise ValueError(
                f"embedding dim {v.size} != store dim {self._dim} (model/tag mismatch?)"
            )
        return self._normalize(v)

    # ── Scoring ──────────────────────────────────────────────────────────────────

    def _score_vector(self, q: np.ndarray):
        """Score every enrolled place as the mean of its top-k similarities to ``q``.
        Returns ``(sorted_scores, best_or_None, classification)``. Assumes _lock held."""
        if self._matrix.shape[0] == 0:
            return [], None, UNKNOWN
        sims = self._matrix @ q                         # (N,)
        scores: list = []
        for pid, name in self._places.items():
            mask = self._emb_pids == pid
            m = int(mask.sum())
            if m == 0:
                continue
            place_sims = sims[mask]
            k = min(self._topk, m)
            # top-k largest similarities, mean of them
            topk = np.partition(place_sims, m - k)[m - k:]
            scores.append(PlaceScore(pid, name, float(topk.mean()), k))
        scores.sort(key=lambda p: p.score, reverse=True)
        best = scores[0] if scores else None
        return scores, best, self._classify(best)

    def _classify(self, best: Optional[PlaceScore]) -> str:
        if best is None:
            return UNKNOWN
        if best.score >= self._match_confident:
            return CONFIDENT
        if best.score >= self._match_min:
            return TENTATIVE
        return UNKNOWN

    def score_frame(self, frame) -> QueryResult:
        """Pure scoring path used by the offline harness: embed a single frame and score
        it against the gallery. No throttle, no hysteresis, no world_state, no logging."""
        q = self._embed(frame)                          # outside the lock (slow)
        with self._lock:
            scores, best, cls = self._score_vector(q)
        return QueryResult(best=best, classification=cls, scores=tuple(scores))

    # ── Live query loop ──────────────────────────────────────────────────────────

    def observe(self, frame) -> Optional[QueryResult]:
        """Feed one camera frame. Call at camera rate; the query pipeline self-throttles
        to ``PLACE_QUERY_INTERVAL_S``. During enrollment the frame feeds COLLECTING
        instead of the query. Returns the QueryResult when a query ran, else None."""
        now = self._clock()
        with self._lock:
            self._check_enroll_timeout(now)
            state = self._state
        if state == COLLECTING:
            self._collect(frame, now)
            return None

        # Query path: self-throttle.
        with self._lock:
            if (now - self._last_query_at) < self._query_interval:
                return None
            self._last_query_at = now

        occ = self._get_person_occlusion()
        if occ is not None and occ > self._occlusion_frac:
            return QueryResult(None, UNKNOWN, (), skipped=True, skip_reason="person_occlusion")

        q = self._embed(frame)                          # outside the lock (slow)
        with self._lock:
            return self._run_query_locked(q, now)

    def tick(self) -> None:
        """Advance time-based state (enrollment timeout) without a frame. Optional for
        callers that want timeouts to fire during a lull in the camera stream."""
        with self._lock:
            self._check_enroll_timeout(self._clock())

    def _run_query_locked(self, q: np.ndarray, now: float) -> QueryResult:
        # Track motion since the last confirmed place (the freeze/unknown gate).
        # A source may return None: "no motion signal available" (no drive base
        # attached, telemetry down). Without a trustworthy signal the freeze gate is
        # DISABLED — treating silence as "not moving" would pin the belief forever
        # the first time someone carries the robot to another room.
        try:
            ms = self._get_motion_state()
            if ms is None or getattr(ms, "moving", False) \
                    or getattr(ms, "wheels_moving", False) \
                    or getattr(ms, "accel_active", False):
                self._moved_since_confirm = True
        except Exception as exc:  # a flaky motion source must not kill recognition
            _log.debug("get_motion_state failed: %s", exc)

        scores, best, cls = self._score_vector(q)
        result = QueryResult(best=best, classification=cls, scores=tuple(scores))
        self._log_observation(best, cls, now)

        # Incremental refresh: quietly grow the gallery of the believed room when a
        # query lands in the "recognizable but not great" band FOR THAT ROOM — keyed on
        # the believed place's own score, not the top match, so an ambiguous frame where
        # another room edges it out still counts (the believed place is what we refresh).
        if self._current_place:
            believed_pid = self._current_place["place_id"]
            believed = next((p for p in scores if p.place_id == believed_pid), None)
            if believed and self._refresh_min <= believed.score <= self._refresh_max:
                self._append_embedding(believed_pid, q, now)

        self._update_belief(best, cls, scores, now)
        return result

    def _update_belief(self, best, cls, scores, now: float) -> None:
        """Ring-buffer majority vote + motion gate → maybe flip ``current_place``; also
        drive the sustained-unknown event. Assumes _lock held."""
        conf = cls == CONFIDENT
        vote_pid = best.place_id if (best and conf) else None
        self._history.append((vote_pid, conf))

        votes = [p for (p, c) in self._history if c and p is not None]
        flip_attempted = False
        if votes:
            cand, n = Counter(votes).most_common(1)[0]
            cur_pid = self._current_place["place_id"] if self._current_place else None
            if n >= self._majority and cand != cur_pid:
                # Motion gate: he cannot change rooms without moving. Acquiring a FIRST
                # belief (no current place) is always allowed.
                frozen = self._current_place is not None and not self._moved_since_confirm
                s = next((p.score for p in scores if p.place_id == cand),
                         best.score if best else 0.0)
                if not frozen:
                    self._confirm_place_locked(cand, s, now)
                else:
                    # Carried-robot escape hatch: the motion source says "still", yet the
                    # camera keeps insisting on another room. Sustained unanimous evidence
                    # means the sensor missed the move (picked up and carried) — flip.
                    flip_attempted = True
                    if self._static_flip_pid == cand:
                        self._static_flip_streak += 1
                    else:
                        self._static_flip_pid, self._static_flip_streak = cand, 1
                    if self._static_flip_streak >= self._static_flip_max:
                        _log.info(
                            "belief flip despite no motion signal — %d consecutive "
                            "confident votes for place_id=%s (carried?)",
                            self._static_flip_streak, cand,
                        )
                        self._moved_since_confirm = True
                        self._confirm_place_locked(cand, s, now)
        if not flip_attempted:
            self._static_flip_pid, self._static_flip_streak = None, 0

        # Sustained-unknown event (only meaningful once he has actually moved).
        if cls == UNKNOWN:
            self._unknown_streak += 1
            if (self._current_place is not None
                    and self._unknown_streak >= self._lost_streak_max):
                # Sustained unfamiliarity while the belief claims a known room: either
                # the motion sensor missed a move or the room changed around him. Admit
                # being lost — drop the belief (publishes None, which re-arms the
                # ask-what-room-is-this cue) rather than keep asserting a stale room.
                _log.info(
                    "belief cleared — %d consecutive unknown frames while believing "
                    "place_id=%s (lost)",
                    self._unknown_streak, self._current_place.get("place_id"),
                )
                self._current_place = None
                self._moved_since_confirm = True
                self._history.clear()
                self._publish_place(None)
        else:
            self._unknown_streak = 0
            if conf:
                self._unknown_armed = True
        if (self._unknown_streak >= self._unknown_streak_max
                and self._moved_since_confirm and self._unknown_armed):
            self._emit(EVENT_UNKNOWN_PLACE, {
                "streak": self._unknown_streak,
                "last_score": (best.score if best else None),
            })
            self._unknown_armed = False

    def _confirm_place_locked(self, pid: int, score: float, now: float) -> None:
        self._current_place = {
            "name": self._places.get(pid),
            "place_id": pid,
            "score": float(score),
            "since_ts": now,
        }
        self._moved_since_confirm = False
        self._unknown_streak = 0
        self._unknown_armed = True
        self._static_flip_pid, self._static_flip_streak = None, 0
        self._history.clear()               # fresh belief; don't let stale votes re-flip
        self._publish_place(dict(self._current_place))

    # ── Enrollment API ───────────────────────────────────────────────────────────

    def enroll(self, name: str) -> int:
        """Begin (or restart) collecting embeddings for ``name``; creates the place row
        if new. Returns the place_id. Subsequent ``observe`` frames feed COLLECTING."""
        name = (name or "").strip()
        if not name:
            raise ValueError("enroll() requires a non-empty room name")
        with self._lock:
            place_id, created_new = self._ensure_place_locked(name)
            self._enroll = _EnrollSession(
                name=name, place_id=place_id, created_new=created_new,
                started_at=self._clock(),
            )
            self._pending_dup = None
            self._state = COLLECTING
            return place_id

    def cancel_enrollment(self) -> None:
        with self._lock:
            if self._state == IDLE:
                return
            self._drop_created_empty_locked(self._enroll)
            self._enroll = None
            self._pending_dup = None
            self._state = IDLE

    def confirm_duplicate(self, is_same: bool) -> bool:
        """Answer a ``possible_duplicate_place`` prompt. ``is_same`` True merges the new
        session's embeddings into the existing place; False commits them as the new
        place. Returns False if not currently awaiting a duplicate confirmation."""
        with self._lock:
            if self._state != CONFIRMING or self._pending_dup is None or self._enroll is None:
                return False
            sess = self._enroll
            dup = self._pending_dup
            now = self._clock()
            target = dup["place_id"] if is_same else sess.place_id
            self._commit_session_locked(target, now)
            if is_same and sess.created_new and sess.place_id != target:
                # The just-created row is a confirmed duplicate; drop it if it stayed empty.
                if not self._place_has_embeddings_locked(sess.place_id):
                    self._delete_place_locked(sess.place_id)
            return True

    def enroll_from_frames(
        self, name: str, frames: Sequence, *, run_duplicate_check: bool = False,
    ) -> EnrollResult:
        """Offline / bulk enrollment used by the harness: embed and commit a whole list
        of frames at once, bypassing the heading/time diversity gates (irrelevant when
        replaying a directory). Uses the same embed → normalize → store → cap path as
        live enrollment. Returns what was actually committed after the per-place cap."""
        name = (name or "").strip()
        if not name:
            raise ValueError("enroll_from_frames() requires a non-empty room name")
        vectors = [self._embed(f) for f in frames]      # outside the lock (slow)
        with self._lock:
            place_id, _ = self._ensure_place_locked(name)
            if run_duplicate_check and vectors:
                dup = self._find_duplicate_locked(place_id, vectors)
                if dup is not None:
                    _log.info("enroll_from_frames: %r resembles existing %r (sim=%.3f)",
                              name, dup["name"], dup["similarity"])
            now_iso = self._now_iso()
            for v in vectors:
                self._insert_embedding_locked(place_id, v, None, now_iso)
            self._enforce_cap_locked(place_id)
            self._rebuild_matrix()
            committed = int((self._emb_pids == place_id).sum())
        return EnrollResult(place_id, name, committed=committed, provided=len(vectors))

    # ── Collection / commit internals ────────────────────────────────────────────

    def _collect(self, frame, now: float) -> None:
        """Try to add one diverse frame to the active enrollment session. Embedding runs
        outside the lock; the diversity gate is re-checked under the lock before append."""
        with self._lock:
            if self._state != COLLECTING or self._enroll is None:
                return
            session = self._enroll            # identity token: the frame is FOR this session
            occ_frac = self._occlusion_frac
            min_hsep, min_tsep = self._min_heading_sep, self._min_time_sep
            headings_snapshot = list(session.headings)
            last_cap = session.last_capture_at

        occ = self._get_person_occlusion()
        if occ is not None and occ > occ_frac:
            return
        heading = self._get_heading()
        if not self._passes_diversity(heading, now, headings_snapshot, last_cap, min_hsep, min_tsep):
            return

        q = self._embed(frame)                          # outside the lock (slow)
        with self._lock:
            sess = self._enroll
            # Drop the embedding if enrollment ended OR was replaced (cancel+enroll, or
            # enroll of a different room) while we were embedding — otherwise this frame's
            # vector would pollute whatever session now occupies self._enroll.
            if sess is None or self._state != COLLECTING or sess is not session:
                return
            # Re-validate against the (possibly grown) session before committing.
            if not self._passes_diversity(
                heading, now, sess.headings, sess.last_capture_at, min_hsep, min_tsep
            ):
                return
            sess.vectors.append(q)
            sess.headings.append(heading)
            sess.last_capture_at = now
            if len(sess.vectors) >= self._enroll_target:
                self._finish_collection_locked("target", now)

    @staticmethod
    def _passes_diversity(heading, now, headings, last_cap, min_hsep, min_tsep) -> bool:
        """Capture-diversity gate. A genuinely new heading is accepted immediately; the
        time-separation gate is the universal fallback — for no heading source at all,
        but ALSO for a stuck one (head parked on a face for the whole enrollment), which
        must slow captures down, never starve them to an enrollment_failed."""
        if heading is not None and not any(
            _circular_sep(heading, ph) < min_hsep for ph in headings if ph is not None
        ):
            return True
        if last_cap is None:
            return True
        return (now - last_cap) >= min_tsep

    def _check_enroll_timeout(self, now: float) -> None:
        if self._state != COLLECTING or self._enroll is None:
            return
        if (now - self._enroll.started_at) < self._enroll_timeout:
            return
        if len(self._enroll.vectors) >= self._enroll_min_frames:
            self._finish_collection_locked("timeout", now)
        else:
            self._abort_enrollment_locked("timeout", now)

    def _finish_collection_locked(self, reason: str, now: float) -> None:
        sess = self._enroll
        if sess is None:
            return
        if len(sess.vectors) < self._enroll_min_frames:
            self._abort_enrollment_locked(reason, now)
            return
        dup = self._find_duplicate_locked(sess.place_id, sess.vectors)
        if dup is not None:
            self._state = CONFIRMING
            self._pending_dup = dup
            self._emit(EVENT_POSSIBLE_DUPLICATE, {
                "new_place": sess.name,
                "new_place_id": sess.place_id,
                "existing_place": dup["name"],
                "existing_place_id": dup["place_id"],
                "similarity": dup["similarity"],
            })
            return
        self._commit_session_locked(sess.place_id, now)

    def _commit_session_locked(self, target_pid: int, now: float) -> None:
        sess = self._enroll
        if sess is None:
            return
        now_iso = self._now_iso(now)
        for v, h in zip(sess.vectors, sess.headings):
            self._insert_embedding_locked(target_pid, v, h, now_iso)
        self._enforce_cap_locked(target_pid)
        self._rebuild_matrix()
        committed = int((self._emb_pids == target_pid).sum())
        self._enroll = None
        self._pending_dup = None
        self._state = IDLE
        self._emit(EVENT_PLACE_ENROLLED, {
            "name": self._places.get(target_pid),
            "place_id": target_pid,
            "embeddings": committed,
        })
        # He was just told where he is: adopt it as the belief with high confidence.
        self._confirm_place_locked(target_pid, self._self_score_locked(target_pid), now)

    def _abort_enrollment_locked(self, reason: str, now: float) -> None:
        sess = self._enroll
        if sess is not None:
            self._emit(EVENT_ENROLLMENT_FAILED, {
                "name": sess.name,
                "place_id": sess.place_id,
                "reason": reason,
                "collected": len(sess.vectors),
            })
            self._drop_created_empty_locked(sess)
        self._enroll = None
        self._pending_dup = None
        self._state = IDLE

    # ── Duplicate detection ──────────────────────────────────────────────────────

    def _find_duplicate_locked(self, exclude_pid: int, vectors: Sequence[np.ndarray]):
        """Max cross-similarity of ``vectors`` against every OTHER place. Returns the
        strongest place above ``PLACE_DUPLICATE_SIM``, else None."""
        if not vectors:
            return None
        S = np.stack(vectors).astype(np.float32)        # (s, dim)
        best = None
        for pid, name in self._places.items():
            if pid == exclude_pid:
                continue
            mask = self._emb_pids == pid
            if not mask.any():
                continue
            P = self._matrix[mask]                      # (m, dim)
            maxsim = float((S @ P.T).max())
            if maxsim > self._dup_sim and (best is None or maxsim > best["similarity"]):
                best = {"place_id": pid, "name": name, "similarity": maxsim}
        return best

    # ── Gallery / DB write helpers (all assume _lock held) ───────────────────────

    def _ensure_place_locked(self, name: str):
        pid = self._name_to_id.get(name)
        if pid is not None:
            return pid, False
        cur = self._conn.execute(
            "INSERT INTO places (name, created_at, notes) VALUES (?, ?, NULL)",
            (name, self._now_iso()),
        )
        self._conn.commit()
        pid = int(cur.lastrowid)
        self._places[pid] = name
        self._name_to_id[name] = pid
        return pid, True

    def _insert_embedding_locked(self, place_id: int, vector: np.ndarray,
                                 heading: Optional[float], captured_at: str) -> None:
        vec = vector.astype(np.float32)
        cur = self._conn.execute(
            "INSERT INTO place_embeddings (place_id, vector, dim, heading_deg, captured_at, model_tag) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (place_id, vec.tobytes(), int(vec.shape[0]),
             (float(heading) if heading is not None else None), captured_at, self._model_tag),
        )
        self._conn.commit()
        self._embeddings.append(
            _Embedding(int(cur.lastrowid), place_id, vec, heading, captured_at)
        )

    def _append_embedding(self, place_id: int, vector: np.ndarray, now: float) -> None:
        """Single write-through append used by incremental refresh (rebuilds the matrix)."""
        heading = None
        try:
            heading = self._get_heading()
        except Exception:
            heading = None
        self._insert_embedding_locked(place_id, vector, heading, self._now_iso(now))
        self._enforce_cap_locked(place_id)
        self._rebuild_matrix()

    def _enforce_cap_locked(self, place_id: int) -> None:
        rows = [e for e in self._embeddings if e.place_id == place_id]
        if len(rows) <= self._max_embeddings:
            return
        rows.sort(key=lambda e: (e.captured_at, e.embedding_id))    # oldest first
        for e in rows[: len(rows) - self._max_embeddings]:
            self._conn.execute(
                "DELETE FROM place_embeddings WHERE embedding_id = ?", (e.embedding_id,)
            )
            self._embeddings.remove(e)
        self._conn.commit()

    def _drop_created_empty_locked(self, sess: Optional[_EnrollSession]) -> None:
        """Remove a place row that this enroll session created but never populated."""
        if sess is None or not sess.created_new:
            return
        if not self._place_has_embeddings_locked(sess.place_id):
            self._delete_place_locked(sess.place_id)

    def _delete_place_locked(self, place_id: int) -> None:
        self._conn.execute("DELETE FROM place_embeddings WHERE place_id = ?", (place_id,))
        self._conn.execute("DELETE FROM places WHERE place_id = ?", (place_id,))
        self._conn.commit()
        name = self._places.pop(place_id, None)
        if name is not None:
            self._name_to_id.pop(name, None)
        self._embeddings = [e for e in self._embeddings if e.place_id != place_id]
        self._rebuild_matrix()

    def _place_has_embeddings_locked(self, place_id: int) -> bool:
        return any(e.place_id == place_id for e in self._embeddings)

    def _self_score_locked(self, place_id: int) -> float:
        """A representative top-k score for a place against its own gallery — used as the
        confidence stamped on the belief right after enrollment."""
        mask = self._emb_pids == place_id
        if not mask.any():
            return 0.0
        rep = self._matrix[mask][-1]
        scores, _, _ = self._score_vector(rep)
        return next((p.score for p in scores if p.place_id == place_id), 1.0)

    def _log_observation(self, best: Optional[PlaceScore], cls: str, now: float) -> None:
        pid = best.place_id if (best and cls in (CONFIDENT, TENTATIVE)) else None
        score = best.score if best else None
        try:
            self._conn.execute(
                "INSERT INTO place_observations (observed_at, place_id, score) VALUES (?, ?, ?)",
                (self._now_iso(now), pid, score),
            )
            self._conn.commit()
        except Exception as exc:
            _log.debug("place_observations write failed: %s", exc)

    # ── Upward interface ─────────────────────────────────────────────────────────

    def _publish_place(self, value: Optional[dict]) -> None:
        if self._world_state is None:
            return
        try:
            self._world_state.update(WORLD_STATE_FIELD, value)
        except Exception as exc:
            _log.debug("world_state.%s write failed: %s", WORLD_STATE_FIELD, exc)

    def _emit(self, name: str, payload: dict) -> None:
        try:
            self._emit_event(name, payload)
        except Exception as exc:
            _log.debug("emit_event(%s) failed: %s", name, exc)

    # ── Introspection / lifecycle ────────────────────────────────────────────────

    @property
    def state(self) -> str:
        return self._state

    def enrolling_name(self) -> Optional[str]:
        """The room name of the active enrollment session, or None when idle."""
        with self._lock:
            return self._enroll.name if self._enroll is not None else None

    def current_place(self) -> Optional[dict]:
        with self._lock:
            return dict(self._current_place) if self._current_place else None

    def place_names(self) -> list:
        with self._lock:
            return sorted(self._places.values())

    def _now_iso(self, now: Optional[float] = None) -> str:
        ts = now if now is not None else self._clock()
        return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="microseconds")

    def reset_belief(self) -> None:
        """Drop the in-memory belief/hysteresis state (durable store untouched). Tests."""
        with self._lock:
            self._history.clear()
            self._current_place = None
            self._moved_since_confirm = False
            self._unknown_streak = 0
            self._unknown_armed = True
            self._static_flip_pid, self._static_flip_streak = None, 0
            self._publish_place(None)

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass
