"""
intelligence/addressee.py — was that line said TO Rex, or to someone else?

Lean Brain restructuring, phase 2B's third decision: where speech came from, who
spoke, and WHOM THEY ADDRESSED are separate questions. The first two have owners;
this is the third. Field 2026-09-05 00:41: Rex asked Bret "Which room is this?",
JT asked Bret "Are you gonna watch both movies?", the dialogue act bound JT's
question as the answer to Rex's, and Rex answered it as if it were his.

Two layers, on purpose:

1. **Deterministic hint** (`assess`): cheap signals decide whether the question
   is even open. A name mention or a parsed command is to Rex. A one-on-one room
   is to Rex — the stay-quiet option is never offered there, so this can never
   make Rex ignore a lone human. Two humans in the recent window, an unknown
   voice while a known person is engaged, an ambiguous speaker verdict, or a
   question from someone other than the person Rex just asked something — those
   make it `uncertain` or `likely_side`.
2. **Model judgment**, only when the hint is not `to_rex`: the ordinary Lean
   reply call gets one extra line describing the doubt and one extra tool,
   `conversation.stay_quiet`. The model then either stays quiet (keeps
   listening, says nothing), chimes in with a short aside that shows it knows
   it is jumping in, or simply answers if the line was plausibly for it. One
   call, no added latency — the same seam every other live tool uses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Optional

import config

_REX_NAME_RE = re.compile(r"\b(?:rex|r3x|r3\s*-?\s*x|dj\s*r3x)\b", re.IGNORECASE)
_QUESTION_RE = re.compile(r"\?\s*$|^\s*(?:are|is|do|does|did|will|would|can|could|should|"
                          r"have|has|were|was|what|where|when|who|why|how|which)\b", re.IGNORECASE)
_SECOND_PERSON_RE = re.compile(r"\byou(?:'re|'ve|'ll|'d| are| were| gonna| going)?\b|\byour\b",
                               re.IGNORECASE)


@dataclass
class AddresseeHint:
    status: str = "to_rex"            # to_rex | uncertain | likely_side
    reasons: list = field(default_factory=list)
    target_name: Optional[str] = None  # who Rex's last question was aimed at, if anyone

    @property
    def offer_stay_quiet(self) -> bool:
        return self.status != "to_rex"

    def as_dict(self) -> dict:
        d = asdict(self)
        d["offer_stay_quiet"] = self.offer_stay_quiet
        return d

    def prompt_line(self) -> str:
        """The one system line Lean gets when the question is open."""
        if not self.offer_stay_quiet:
            return ""
        why = "; ".join(self.reasons[:3])
        lead = ("This line was PROBABLY not said to you" if self.status == "likely_side"
                else "This line may not have been said to you")
        target = (f" You had just asked {self.target_name} something, so a question from "
                  f"someone else is most likely aimed at {self.target_name}, not you."
                  if self.target_name else "")
        return (
            f"ADDRESSEE CHECK: {lead} — {why}.{target} Decide before you speak: if the "
            "humans are talking to EACH OTHER, call the conversation_stay_quiet tool (you "
            "keep listening, nothing is said — that is the normal, polite move). Chime in "
            "ONLY if you have something genuinely worth adding — a good joke, useful "
            "information, or they clearly want you in — and then as ONE short aside that "
            "shows you know you're jumping into their conversation; never answer it as if "
            "it were asked of you. If it plausibly WAS to you, just answer normally."
        )


def _cfg(name: str, default):
    try:
        return getattr(config, name, default)
    except Exception:
        return default


def assess(
    text: str,
    *,
    speaker_pid: Optional[int],
    speaker_known: bool,
    speaker_uncertain: bool,
    humans_in_window: int,
    engaged_pid: Optional[int],
    last_frame_target_pid: Optional[int],
    last_frame_target_name: Optional[str],
    last_frame_is_question: bool,
    command_parsed: bool = False,
) -> AddresseeHint:
    """Cheap, deterministic read of whom `text` was aimed at. See module doc."""
    cleaned = " ".join((text or "").split())
    if not cleaned or not bool(_cfg("ADDRESSEE_JUDGMENT_ENABLED", True)):
        return AddresseeHint("to_rex", ["judgment disabled or empty"])
    if _REX_NAME_RE.search(cleaned):
        return AddresseeHint("to_rex", ["they said your name"])
    if command_parsed:
        return AddresseeHint("to_rex", ["it parses as a command to you"])

    multi_party = humans_in_window >= 2
    stranger_with_known = (not speaker_known) and engaged_pid is not None
    if not (multi_party or stranger_with_known or speaker_uncertain):
        return AddresseeHint("to_rex", ["one-on-one conversation"])

    reasons: list[str] = []
    if multi_party:
        reasons.append(f"{humans_in_window} people have spoken recently")
    if stranger_with_known:
        reasons.append("this voice is not one you know, while someone you know is engaged")
    if speaker_uncertain:
        reasons.append("you are not sure who is speaking")

    other_speaker = (
        last_frame_target_pid is not None
        and (speaker_pid is None or int(speaker_pid) != int(last_frame_target_pid))
    )
    is_question = bool(_QUESTION_RE.search(cleaned))
    second_person = bool(_SECOND_PERSON_RE.search(cleaned))
    if other_speaker and (is_question or second_person):
        who = last_frame_target_name or "the person you were talking to"
        reasons.append(f"someone other than {who} asked a question right after you asked {who} something")
        return AddresseeHint("likely_side", reasons, last_frame_target_name)
    if other_speaker:
        reasons.append(f"a different person spoke than the one you were just talking to")
    return AddresseeHint("uncertain", reasons, last_frame_target_name if other_speaker else None)
