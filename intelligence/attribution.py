"""Utterance-bound identity evidence and one authoritative speaker verdict.

resolve_authoritative owns production attribution and learning permission.
The legacy resolve adapter remains for comparison tests. Voice scores remain
similarities, not identity probabilities; direction, identity and addressee
are separate decisions. Suspect mixed captures abstain from personal attribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class UtteranceEvidence:
    turn_id: Optional[int] = None
    session_id: Optional[int] = None
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    visual_observations: list = field(default_factory=list)
    mixed_speakers: bool = False
    text: str = ""
    text_input: bool = False
    words: int = 0
    voiced_secs: float = 0.0
    # Voice (speaker ID): raw argmax over enrolled prints, its runner-up gap, and
    # the bar the scoreboard required for an unambiguous pick.
    raw_best_id: Optional[int] = None
    raw_best_name: Optional[str] = None
    raw_best_score: float = 0.0
    margin: float = 0.0
    required_margin: float = 0.0
    hard_threshold: float = 0.75
    soft_threshold: float = 0.60
    known_floor: float = 0.45
    scoreboard: list = field(default_factory=list)   # [(pid, name, score, n_prints)]
    # What the ladder did.
    accept_tier: Optional[str] = None                # hard / known_floor / sticky / roster / None
    identity_resolution: Optional[str] = None        # override label (visible face, roster, ...)
    final_person_id: Optional[int] = None
    final_name: Optional[str] = None
    anonymous_label: Optional[str] = None
    off_camera_unknown: bool = False
    # Room / camera / array.
    visible_known_ids: list = field(default_factory=list)
    visual_latch_pid: Optional[int] = None           # active_speaker's recent winner
    bearing_selected_pid: Optional[int] = None       # DoA-picked face (None = no pick)
    bearing_contradiction: bool = False              # voice came from off-camera
    engaged_pid: Optional[int] = None
    previous_speaker_pid: Optional[int] = None
    continuity_age_secs: Optional[float] = None
    allow_short_continuity: bool = False

    def as_dict(self) -> dict:
        d = asdict(self)
        d["scoreboard"] = [list(x) for x in (self.scoreboard or [])][:5]
        return d


@dataclass
class Resolution:
    status: str                       # known | unknown | ambiguous
    person_id: Optional[int]
    name: Optional[str]
    basis: str                        # one-line reason for the status
    conflicts: list = field(default_factory=list)
    learning_allowed: bool = True

    def as_dict(self) -> dict:
        return {"status": self.status, "person_id": self.person_id, "name": self.name,
                "basis": self.basis, "conflicts": list(self.conflicts),
                "learning_allowed": self.status == "known" and self.learning_allowed}


def _first(name: Optional[str]) -> str:
    return str(name or "").split()[0] if name else "someone"


def resolve(ev: UtteranceEvidence) -> Resolution:
    """Shadow verdict on the ladder's decision. Conservative: a decision backed by
    one strong signal with nothing contradicting it is `known`; a person picked
    without a confident voice AND with any independent signal pointing elsewhere is
    `ambiguous`; no person is `unknown`."""
    pid = ev.final_person_id
    conflicts: list[str] = []

    if ev.text_input:
        return Resolution("known" if pid is not None else "unknown", pid, ev.final_name,
                          "typed input — attribution is the GUI's", [])
    if ev.mixed_speakers:
        return Resolution("ambiguous", None, None, "multiple speakers within this capture",
                          ["whole-buffer voice scores cannot assign these words to one person"])
    if pid is None:
        basis = "no enrolled voice matched and no single visible face"
        if ev.off_camera_unknown:
            basis = "voice from off camera, no enrolled match"
        return Resolution("unknown", None, None, basis, [])

    voice_agrees = ev.raw_best_id is not None and int(ev.raw_best_id) == int(pid)
    voice_strong = voice_agrees and ev.raw_best_score >= ev.hard_threshold \
        and ev.margin >= ev.required_margin
    voice_soft = voice_agrees and ev.raw_best_score >= ev.soft_threshold

    # Independent signals that point at someone else.
    if (ev.raw_best_id is not None and int(ev.raw_best_id) != int(pid)
            and ev.raw_best_score >= ev.soft_threshold):
        conflicts.append(
            f"the voice scored {_first(ev.raw_best_name)} at {ev.raw_best_score:.2f}, "
            f"not {_first(ev.final_name)}")
    if ev.visual_latch_pid is not None and int(ev.visual_latch_pid) != int(pid):
        conflicts.append("the camera saw a different person's mouth moving")
    if ev.bearing_selected_pid is not None and int(ev.bearing_selected_pid) != int(pid):
        conflicts.append("the voice came from a different visible face's direction")
    if ev.bearing_contradiction and pid in (ev.visible_known_ids or []):
        conflicts.append("the voice came from off camera while that face is on camera")
    if voice_agrees and not voice_soft and ev.margin < ev.required_margin and len(ev.scoreboard) > 1:
        conflicts.append(
            f"the voice barely separated {_first(ev.final_name)} from the runner-up "
            f"(margin {ev.margin:.2f} < {ev.required_margin:.2f})")

    if voice_strong and not conflicts:
        return Resolution("known", pid, ev.final_name, "confident voice match, nothing contradicting", [])
    if conflicts:
        return Resolution("ambiguous", pid, ev.final_name,
                          f"picked by {ev.accept_tier or ev.identity_resolution or 'context'} "
                          "but another signal disagrees", conflicts)
    # No conflict, but the pick rests on context rather than a confident voice.
    tier = str(ev.accept_tier or "")
    if voice_soft or tier in ("hard", "known_floor"):
        return Resolution("known", pid, ev.final_name,
                          f"voice match ({tier or 'soft'}) with corroborating context", [])
    if tier in ("sticky", "roster") or ev.identity_resolution:
        # Continuity / roster / visible-face picks with no voice support: fine
        # to answer, not fine to learn from or to name with confidence when the
        # words were short.
        weak = ev.words <= 3 or ev.voiced_secs < 1.0
        if weak:
            return Resolution("ambiguous", pid, ev.final_name,
                              f"{tier or ev.identity_resolution} pick on a very short clip, no voice support",
                              ["too little speech to confirm the voice"])
        return Resolution("known", pid, ev.final_name,
                          f"{tier or ev.identity_resolution} pick, no contradicting signal", [])
    return Resolution("known", pid, ev.final_name, "ladder decision, no contradicting signal", [])


def resolve_authoritative(ev: UtteranceEvidence) -> Resolution:
    """Resolve raw voice and interval evidence, treating context as a proposal.

    The old candidate ladder may suggest a name, but engagement, roster and a
    last-speaker guess cannot independently authorize identity or learning.
    Existing raw voice thresholds retain their units and meaning.
    """
    if ev.text_input or ev.mixed_speakers:
        return resolve(ev)
    visual_ids = {row.get("person_db_id") for row in ev.visual_observations
                  if row.get("person_db_id") is not None}
    if len(visual_ids) > 1:
        return Resolution("ambiguous", None, None, "speaker transition in utterance interval",
                          ["sequential or overlapping speakers; no word-level attribution"])
    # Only observations from the captured interval can corroborate the voice.
    visual = next(iter(visual_ids), None)
    candidate = ev.raw_best_id
    enough_margin = ev.margin >= ev.required_margin
    strong = candidate is not None and ev.raw_best_score >= ev.hard_threshold and enough_margin
    supported_voice = (candidate is not None and ev.raw_best_score >= ev.soft_threshold
                       and enough_margin and ev.accept_tier in {"hard", "known_floor", "roster"})
    corroborated = (candidate is not None and enough_margin
                    and ev.raw_best_score >= ev.known_floor and visual == candidate)
    if strong or corroborated or supported_voice:
        if (ev.bearing_selected_pid is not None and ev.bearing_selected_pid != candidate
                or ev.bearing_contradiction and candidate in ev.visible_known_ids):
            return Resolution("ambiguous", None, None, "voice and utterance direction disagree",
                              ["conflicting identity evidence"])
        if visual is not None and visual != candidate:
            return Resolution("ambiguous", None, None, "voice and interval mouth motion disagree",
                              ["conflicting identity evidence"])
        return Resolution("known", candidate, ev.raw_best_name,
                          "strong voice" if strong else ("voice with interval visual corroboration"
                          if corroborated else "accepted voice score and margin"))
    if _guarded_short_continuity(ev):
        return Resolution("known", candidate, ev.raw_best_name,
                          "short reply with recent verified speaker and continuous sole face",
                          learning_allowed=False)
    voiced_rows = [row for row in ev.visual_observations if row.get("person_db_id") == visual]
    if (visual is not None and len(voiced_rows) >= 3 and ev.voiced_secs >= 1.0
            and ev.words >= 4 and ev.raw_best_score < ev.soft_threshold
            and not ev.bearing_contradiction
            and ev.bearing_selected_pid in (None, visual)):
        names = [face.get("face_id") for row in voiced_rows for face in row.get("faces", [])
                 if face.get("person_db_id") == visual and face.get("face_id")]
        return Resolution("known", visual, names[-1] if names else None,
                          "sustained mouth motion during this utterance; weak voice evidence")
    return Resolution("ambiguous" if candidate is not None else "unknown", None, None,
                      "insufficient utterance-bound identity evidence")


def _guarded_short_continuity(ev):
    """Conversational attribution only; never a new biometric/learning anchor."""
    import config
    pid = ev.raw_best_id
    if (not ev.allow_short_continuity or pid is None or pid != ev.previous_speaker_pid
            or not 0 < ev.voiced_secs <= 1.5 or not 0 < ev.words <= 4
            or ev.continuity_age_secs is None
            or not 0 <= ev.continuity_age_secs <= float(getattr(config, "SHORT_CLIP_LAST_SPEAKER_SECS", 90))
            or ev.raw_best_score < float(getattr(config, "CAMPPLUS_SHORT_REPLY_MIN_COSINE", .20))
            or ev.margin < ev.required_margin or ev.bearing_contradiction
            or ev.bearing_selected_pid not in (None, pid)
            or set(ev.visible_known_ids) != {pid} or len(ev.visual_observations) < 3):
        return False
    for row in ev.visual_observations:
        if row.get("person_db_id") not in (None, pid):
            return False
        faces = [f for f in row.get("faces", []) if not f.get("face_missing")
                 and f.get("face_visible") is not False
                 and (f.get("face_visible") or f.get("face_box"))]
        if len(faces) != 1 or faces[0].get("person_db_id") != pid:
            return False
    return True


def sequential_boundaries(voiced_runs: list[tuple[float, float]], observations: list[dict]) -> list[float]:
    """Split only at a real silent gap between sustained, different visual speakers.

    Inputs use monotonic seconds. Multiple identities inside a voiced run are
    suspect overlap; no boundary is invented inside speech. No word diarization
    or extra inference model is implied by this conservative segmentation.
    """
    labeled = []
    for start, end in voiced_runs:
        rows = [r for r in observations if start <= r.get("monotonic_at", -1) <= end
                and r.get("person_db_id") is not None]
        identities = {r["person_db_id"] for r in rows}
        pid = next(iter(identities)) if len(identities) == 1 and len(rows) >= 2 else None
        labeled.append((start, end, pid))
    boundaries = []
    for previous, following in zip(labeled, labeled[1:]):
        if (previous[2] is not None and following[2] is not None
                and previous[2] != following[2] and following[0] - previous[1] >= .12):
            boundaries.append((previous[1] + following[0]) / 2)
    return boundaries


def voice_boundaries(voiced_runs: list[tuple[float, float]], windows: list[dict]) -> list[float]:
    """Audio-only boundaries: a silent gap between confidently different windows.

    All times are relative to the same captured buffer. Overlapping speech has
    no supported gap and stays mixed; it is never assigned a fabricated split.
    """
    gaps = [(a[1] + b[0]) / 2 for a, b in zip(voiced_runs, voiced_runs[1:])
            if b[0] - a[1] >= .12]
    cuts = []
    for left, right in zip(windows, windows[1:]):
        known_switch = (left.get("person_id") is not None and right.get("person_id") is not None
                        and left["person_id"] != right["person_id"])
        if not known_switch and not right.get("change_suspected"):
            continue
        lo, hi = (left["start"]+left["end"])/2, (right["start"]+right["end"])/2
        candidates = [gap for gap in gaps if lo < gap < hi]
        if candidates:
            cuts.append(min(candidates, key=lambda gap: abs(gap-(lo+hi)/2)))
    return sorted(set(cuts))
