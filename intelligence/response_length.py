"""
intelligence/response_length.py - turn-local response length planning.

The core Rex prompt keeps him concise, but "1-2 sentences" can accidentally
produce two packed, run-on sentences. This layer chooses a concrete response
budget for the current turn so short acknowledgements can actually be short,
while deeper turns still get room.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Optional


_QUESTION_START = re.compile(
    r"^\s*(who|what|when|where|why|how|can|could|would|will|do|does|did|"
    r"is|are|am|should|tell me|explain)\b",
    re.IGNORECASE,
)
_CHECK_ALIVE_PAT = re.compile(
    r"\b(hello|hey|you there|are you there|can you hear me|testing|wake up|rex)\b",
    re.IGNORECASE,
)
_CLOSURE_PAT = re.compile(
    r"\b(that'?s all|that is all|that'?s it|that is it|all good|i'?m good|"
    r"thanks|thank you|got it|sounds good|fair enough|never mind)\b",
    re.IGNORECASE,
)
_DEPTH_QUESTION_PAT = re.compile(r"^\s*(why|how|explain|tell me)\b", re.IGNORECASE)
_TOPIC_KNOWLEDGE_QUESTION_PAT = re.compile(
    r"\b(?:what\s+do\s+you\s+know|do\s+you\s+know\s+anything|"
    r"tell\s+me|explain)\s+(?:about\s+)?[^?.,!;]{3,100}",
    re.IGNORECASE,
)
_DEPTH_STATEMENT_PAT = re.compile(
    r"\b(because|actually|honestly|i think|i feel|it feels|the thing is|"
    r"what happened was|i'm worried|i am worried|i've been|i have been)\b",
    re.IGNORECASE,
)


@dataclass
class ResponseLengthPlan:
    target: str
    max_words: int
    max_sentences: int
    reason: str
    instruction: str


def snapshot(
    user_text: str,
    *,
    answered_question: Optional[dict] = None,
) -> dict:
    return asdict(classify(user_text, answered_question=answered_question))


def build_directive(
    user_text: str,
    *,
    answered_question: Optional[dict] = None,
) -> str:
    plan = classify(user_text, answered_question=answered_question)
    return (
        "Response length control:\n"
        f"- Target: {plan.target}; max_words={plan.max_words}; "
        f"max_sentences={plan.max_sentences}; reason={plan.reason}.\n"
        f"- Shape: {plan.instruction}"
    )


def classify(
    user_text: str,
    *,
    answered_question: Optional[dict] = None,
) -> ResponseLengthPlan:
    cleaned = (user_text or "").strip()
    words = re.findall(r"[A-Za-z0-9']+", cleaned)
    word_count = len(words)
    is_question = "?" in cleaned or bool(_QUESTION_START.search(cleaned))

    # Closure and short answers after Rex's own question should land and stop.
    if _CLOSURE_PAT.search(cleaned):
        return _plan(
            "micro",
            12,
            1,
            "user is closing the thread",
            "Use a tiny landing acknowledgement. A fragment is allowed. No question.",
        )

    if (
        answered_question
        and answered_question.get("question_key") == "startup_conversation_steering"
    ):
        return _plan(
            "short",
            55,
            3,
            "answer to Rex's startup steering question",
            "Acknowledge the chosen topic, add one compact subject-aware beat, "
            "then end with one short natural follow-up question.",
        )

    if answered_question and word_count <= 5:
        return _plan(
            "micro",
            12,
            1,
            "short answer to Rex's question",
            "Acknowledge in one small beat, then stop. Do not squeeze in a new prompt.",
        )

    try:
        from intelligence import conversation_steering
        volunteered_interest = conversation_steering.detect_interest(cleaned)
    except Exception:
        volunteered_interest = None
    if volunteered_interest:
        return _plan(
            "short",
            45,
            2,
            "user volunteered a topic interest",
            "Acknowledge the interest, add one compact subject-aware beat, "
            "then ask one short follow-up about their angle, taste, or favorite part.",
        )

    if word_count <= 3 and _CHECK_ALIVE_PAT.search(cleaned):
        return _plan(
            "micro",
            10,
            1,
            "presence check",
            "One small acknowledgement is enough. Do not add a topic unless asked.",
        )

    # Direct questions may need space, but simple ones still should not sprawl.
    if is_question:
        if _TOPIC_KNOWLEDGE_QUESTION_PAT.search(cleaned):
            return _plan(
                "long",
                125,
                7,
                "general topic knowledge question",
                "Use the main LLM's general knowledge to give a more substantive "
                "answer about the topic. Be concrete, accurate, and in character. "
                "End with at most one short question asking whether this is a "
                "subject they are into or what angle they want next.",
            )
        if _DEPTH_QUESTION_PAT.search(cleaned) or word_count >= 10:
            return _plan(
                "medium",
                65,
                4,
                "substantive user question",
                "Answer directly with substance, using short sentences. Stop when answered.",
            )
        return _plan(
            "brief",
            22,
            2,
            "simple user question",
            "Answer directly in one short sentence; use a tiny second sentence only for flavor.",
        )

    # Let the existing energy layer steer normal conversation, but make the
    # length budget concrete enough that "short" cannot become a long sentence.
    try:
        from intelligence import user_energy
        energy = user_energy.snapshot() or {}
    except Exception:
        energy = {}

    mode = (energy.get("mode") or "").lower()
    response_length = (energy.get("response_length") or "").lower()

    if response_length == "brief" or mode in {"task", "check_alive"}:
        return _plan(
            "brief",
            18,
            1,
            f"user energy: {mode or response_length}",
            "One short sentence. No padding, no extra setup.",
        )

    if mode == "quiet" or response_length == "short":
        return _plan(
            "short",
            24,
            1,
            f"user energy: {mode or response_length}",
            "One short sentence or clause. If you roast, make it a light tap, not a monologue.",
        )

    if mode == "depth" or response_length == "medium" or word_count >= 12 or _DEPTH_STATEMENT_PAT.search(cleaned):
        return _plan(
            "medium",
            70,
            4,
            "user is offering depth",
            "Use 2-4 short sentences only if needed. No run-ons; leave breathing room.",
        )

    if mode == "banter":
        return _plan(
            "short",
            28,
            2,
            "playful banter",
            "A quick jab or comeback. Two sentences only if the second is a tiny button.",
        )

    return _plan(
        "brief",
        22,
        1,
        "default conversational turn",
        "Default to one short line. Do not pad to reach two sentences.",
    )


def _plan(
    target: str,
    max_words: int,
    max_sentences: int,
    reason: str,
    instruction: str,
) -> ResponseLengthPlan:
    return ResponseLengthPlan(
        target=target,
        max_words=int(max_words),
        max_sentences=int(max_sentences),
        reason=reason,
        instruction=instruction,
    )
