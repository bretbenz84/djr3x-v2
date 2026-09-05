"""
intelligence/attribution.py — one utterance's speaker evidence and one verdict.

Lean Brain restructuring, phase 2B. `interaction._handle_speech_segment` decides
WHO spoke through a long override ladder (hard voice match, known-floor, session
stickiness, game roster, visible-face corroboration, bearing match, GUI text,
pending-question attribution, short-clip continuity...). That ladder stays
authoritative — it carries months of field fixes. What was missing is a single
place that (a) holds every piece of evidence the turn had, with its own units
left alone (a cosine similarity is not a probability), and (b) says how SURE the
result is, so the reply can stay name-free when the room is ambiguous and the
learning paths can stand down.

`resolve()` runs in SHADOW: it never changes the ladder's person_id. It returns
known / unknown / ambiguous plus the conflicts it saw, which interaction then
(1) logs in [identity_decision], (2) hands to Lean via conversation_state, (3)
marks on the transcript entry, and (4) uses to gate passive voiceprint growth
and per-turn memory learning. Three decisions are kept apart: where the speech
came from (bearing), who spoke (voice/face), whom they addressed (not judged here).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class UtteranceEvidence:
    turn_id: Optional[int] = None
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

    def as_dict(self) -> dict:
        return {"status": self.status, "person_id": self.person_id, "name": self.name,
                "basis": self.basis, "conflicts": list(self.conflicts)}


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
