"""
intelligence/turn_plan.py — typed handoff between the conversation governors (Bet 2).

The pipeline used to communicate by PROSE: conversation_agenda emitted a directive
paragraph, and social_frame regex-reparsed it (`_purpose_from` / `_EXPLICIT_FOLLOWUP_PAT`
/ `_ASK_ALLOWED_PAT`) to re-derive decisions the agenda had ALREADY made — so every
wording tweak risked breaking the next layer's regex (the rework's "core problem #1").

TurnPlan carries those decisions losslessly. `conversation_agenda.build_turn_plan()`
populates the structured fields AND renders `directive` (the prompt text);
`social_frame.build_frame()` reads the fields instead of regex-matching the string. The
string still drives the LLM prompt — TurnPlan just ends the guessing between layers.

Migration is incremental and additive: fields default to "the agenda said nothing", and
social_frame falls back to its regex when a field is unset OR no plan is passed, so the
prose path keeps working (and every existing string-only test stays green) until each
signal is migrated and its regex deleted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TurnPlan:
    # The rendered agenda directive (the actual prompt text). Always set.
    directive: str = ""
    # The turn's purpose when the agenda's branch determines it (closure / interest /
    # answer_ack / answer / …). None = "let social_frame derive it" (generic turns
    # where purpose comes from energy/length, exactly as before).
    purpose: Optional[str] = None

    # ── Question-allowance signals. Tri-state like `purpose`: None = "the agenda did
    # not decide → social_frame falls back to its regex"; True/False = the agenda's
    # decision. Migrated incrementally (explicit_followup is live; the rest are still
    # None everywhere → regex, pending a later slice that then deletes the patterns). ──
    ask_allowed: Optional[bool] = None              # the directive invites a question at all
    explicit_followup: Optional[bool] = None        # an earned on-thread follow-up was offered
    fresh_interest_followup: Optional[bool] = None  # human volunteered an interest + follow-up offered
    urgent_identity: Optional[bool] = None          # urgent group-identity handoff (may bypass budget)
    hard_no_question: Optional[bool] = None          # the directive forbids any new question

    # Context the agenda already knows (optional; for future consumers).
    topic: str = ""
    mode: str = ""
