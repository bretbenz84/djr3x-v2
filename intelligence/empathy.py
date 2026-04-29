"""
intelligence/empathy.py — Emotional intelligence layer for DJ-R3X.

Single small GPT-4o-mini call classifies the user's utterance for emotional
affect, what they seem to need from Rex, topic sensitivity, and whether they
are inviting Rex into a vulnerable conversation. The result is fused into a
"response mode" (default, listen, support, lift, ground, validate, brief,
amplify, kind_default, child_kind, acknowledge_then_yield) and a one-paragraph
style directive that gets injected into the system prompt.

Design rule (durable): support / listen / lift modes are NOT gated by friendship
tier. Anyone — stranger included — gets caring mode the moment they signal they
want to be heard. Tier shapes Rex's voice and depth, not whether he shows up.
"""

import json
import logging
import re
import threading
import time
from collections import deque
from typing import Optional

import config
import apikeys
from openai import OpenAI

_log = logging.getLogger(__name__)
_client = OpenAI(api_key=apikeys.OPENAI_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_VALID_AFFECT = {
    "happy", "excited", "neutral", "tired", "anxious",
    "sad", "withdrawn", "angry",
}
_VALID_NEEDS = {"vent", "advice", "distract", "celebrate", "none"}
_VALID_SENSITIVITY = {"none", "mild", "heavy"}

_NEGATIVE_AFFECT = {"sad", "withdrawn", "anxious", "tired", "angry"}
_POSITIVE_AFFECT = {"happy", "excited"}

# Scalar valence for trend math. The classifier doesn't return a numeric
# valence directly, so we project the label onto a -1..+1 axis. Conservative
# magnitudes — we'd rather under-react than over-react to one ambiguous read.
_AFFECT_VALENCE = {
    "excited":   0.9,
    "happy":     0.7,
    "neutral":   0.0,
    "tired":    -0.2,
    "withdrawn":-0.4,
    "anxious":  -0.6,
    "sad":      -0.6,
    "angry":    -0.5,
}

_FINE_WORDS_PAT = re.compile(
    r"\b(?:"
    r"i(?:'m| am)\s+(?:fine|okay|ok|good|alright|all right)|"
    r"doing\s+(?:fine|okay|ok|good|alright|all right)|"
    r"all\s+good|"
    r"fine|okay|ok"
    r")\b",
    re.IGNORECASE,
)
_LOCAL_CRISIS_PAT = re.compile(
    r"\b(?:"
    r"kill myself|killing myself|suicide|suicidal|self[-\s]?harm|"
    r"hurt myself|end it all|don'?t want to live|do not want to live|"
    r"want to die|going to die tonight"
    r")\b",
    re.IGNORECASE,
)
_LOCAL_DEATH_PAT = re.compile(
    r"\b(?:"
    r"died|dead|death|passed away|passed on|loss of|"
    r"funeral|memorial|grief|grieving|bereaved|killed"
    r")\b",
    re.IGNORECASE,
)
_LOCAL_ILLNESS_PAT = re.compile(
    r"\b(?:"
    r"cancer|chemo|chemotherapy|radiation treatment|hospice|terminal|"
    r"diagnosed|diagnosis|hospital|icu|stroke|heart attack|surgery|"
    r"seriously sick|really sick|very sick"
    r")\b",
    re.IGNORECASE,
)
_LOCAL_HARD_LIFE_PAT = re.compile(
    r"\b(?:"
    r"got laid off|laid me off|lost my job|fired me|got fired|"
    r"breakup|broke up|divorce|panic attack|depressed|depression"
    r")\b",
    re.IGNORECASE,
)
_LOSS_SUBJECT_PAT = re.compile(
    r"\b(?:"
    r"my|our|his|her|their"
    r")\s+("
    r"mom|mother|mum|dad|father|parent|grandma|grandmother|grandpa|"
    r"grandfather|grandparent|wife|husband|partner|spouse|son|daughter|"
    r"child|kid|brother|sister|uncle|aunt|cousin|friend|best friend|"
    r"dog|cat|pet|puppy|kitten"
    r")\b",
    re.IGNORECASE,
)
_PET_SUBJECTS = {"dog", "cat", "pet", "puppy", "kitten"}
_DEATH_FALSE_ALARMS_PAT = re.compile(
    r"\b(?:dead tired|dead battery|dead serious|dead end|deadlift|dead line)\b",
    re.IGNORECASE,
)


def _result_valence(result: dict) -> float:
    return _AFFECT_VALENCE.get(
        (result.get("affect") or "neutral").lower(), 0.0,
    )


def _local_loss_subject(text: str) -> tuple[Optional[str], Optional[str]]:
    match = _LOSS_SUBJECT_PAT.search(text or "")
    if not match:
        return None, None
    subject = " ".join(match.group(1).lower().split())
    kind = "pet" if subject in _PET_SUBJECTS else "person"
    return subject, kind


def classify_local_sensitivity(text: str) -> Optional[dict]:
    """Fast deterministic safety read used before the async empathy LLM returns.

    This is intentionally narrow and high-precision: it only catches topics
    where Rex should never spend the current turn roasting or riffing while the
    richer classifier is still running.
    """
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return None
    lowered = cleaned.lower()

    if _LOCAL_CRISIS_PAT.search(cleaned):
        return {
            "affect": "anxious",
            "needs": "vent",
            "topic_sensitivity": "heavy",
            "invitation": True,
            "crisis": True,
            "confidence": 0.92,
            "event": None,
            "mood_mismatch": None,
            "prosody": None,
            "source": "local_sensitive_topic",
        }

    loss_subject_for_hit, _ = _local_loss_subject(cleaned)
    death_hit = bool(_LOCAL_DEATH_PAT.search(cleaned)) or (
        bool(re.search(r"\blost\b", lowered)) and bool(loss_subject_for_hit)
    )
    if death_hit and _DEATH_FALSE_ALARMS_PAT.search(cleaned):
        death_hit = False
    if death_hit:
        subject, subject_kind = _local_loss_subject(cleaned)
        category = "death" if re.search(r"\b(?:died|dead|death|passed away|passed on|killed)\b", lowered) else "grief"
        description = (
            f"shared that {subject} was affected by death or grief"
            if subject else
            "shared a death or grief event"
        )
        return {
            "affect": "sad",
            "needs": "vent",
            "topic_sensitivity": "heavy",
            "invitation": True,
            "crisis": False,
            "confidence": 0.88,
            "event": {
                "category": category,
                "description": description,
                "valence": -0.9,
                "loss_subject": subject,
                "loss_subject_kind": subject_kind,
                "loss_subject_name": None,
            },
            "mood_mismatch": None,
            "prosody": None,
            "source": "local_sensitive_topic",
        }

    if _LOCAL_ILLNESS_PAT.search(cleaned):
        subject, subject_kind = _local_loss_subject(cleaned)
        return {
            "affect": "sad",
            "needs": "vent",
            "topic_sensitivity": "heavy",
            "invitation": True,
            "crisis": False,
            "confidence": 0.82,
            "event": {
                "category": "illness",
                "description": "shared a serious illness or health scare",
                "valence": -0.75,
                "loss_subject": subject,
                "loss_subject_kind": subject_kind,
                "loss_subject_name": None,
            },
            "mood_mismatch": None,
            "prosody": None,
            "source": "local_sensitive_topic",
        }

    if _LOCAL_HARD_LIFE_PAT.search(cleaned):
        category = "other"
        if re.search(r"\b(?:laid off|lost my job|fired)\b", lowered):
            category = "job_loss"
        elif re.search(r"\b(?:breakup|broke up|divorce)\b", lowered):
            category = "breakup"
        return {
            "affect": "sad",
            "needs": "vent",
            "topic_sensitivity": "heavy",
            "invitation": True,
            "crisis": False,
            "confidence": 0.78,
            "event": {
                "category": category,
                "description": "shared a difficult personal event",
                "valence": -0.65,
                "loss_subject": None,
                "loss_subject_kind": None,
                "loss_subject_name": None,
            },
            "mood_mismatch": None,
            "prosody": None,
            "source": "local_sensitive_topic",
        }

    return None


def _detect_mood_mismatch(
    text: str,
    affect: str,
    sensitivity: str,
    confidence: float,
    prosody_features: Optional[dict],
    face_mood: Optional[dict] = None,
) -> Optional[dict]:
    """Detect "I'm fine" words paired with negative nonverbal signals."""
    del confidence  # reserved for future fusion with model confidence
    if not getattr(config, "EMPATHY_MOOD_MISMATCH_ENABLED", True):
        return None
    if not _FINE_WORDS_PAT.search(text or ""):
        return None
    evidence = []
    if prosody_features:
        try:
            prosody_conf = float(prosody_features.get("confidence", 0.0) or 0.0)
            prosody_valence = float(prosody_features.get("valence", 0.0) or 0.0)
        except (TypeError, ValueError):
            prosody_conf = 0.0
            prosody_valence = 0.0
        min_conf = float(getattr(config, "EMPATHY_MOOD_MISMATCH_MIN_PROSODY_CONFIDENCE", 0.55))
        max_valence = float(getattr(config, "EMPATHY_MOOD_MISMATCH_NEGATIVE_VALENCE", -0.30))
        if prosody_conf >= min_conf and prosody_valence <= max_valence:
            evidence.append({
                "source": "prosody",
                "tag": prosody_features.get("tag") or "",
                "valence": prosody_valence,
                "confidence": prosody_conf,
            })

    if getattr(config, "EMPATHY_FACE_MOOD_MISMATCH_ENABLED", True) and face_mood:
        label = (face_mood.get("mood") or "").strip().lower()
        try:
            face_conf = float(face_mood.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            face_conf = 0.0
        min_face_conf = float(getattr(config, "EMPATHY_FACE_MOOD_MISMATCH_MIN_CONFIDENCE", 0.60))
        if label in {"sad", "tired", "angry", "anxious"} and face_conf >= min_face_conf:
            evidence.append({
                "source": "face",
                "tag": f"{label}: {face_mood.get('notes') or ''}".strip(": "),
                "valence": -0.5,
                "confidence": face_conf,
            })

    if not evidence:
        return None
    strongest = max(evidence, key=lambda item: float(item.get("confidence") or 0.0))
    return {
        "kind": "words_ok_nonverbal_not",
        "words_affect": affect,
        "words_sensitivity": sensitivity,
        "source": strongest["source"],
        "tag": strongest.get("tag") or "",
        "valence": strongest.get("valence"),
        "confidence": strongest.get("confidence"),
        "evidence": evidence,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LLM classification
# ─────────────────────────────────────────────────────────────────────────────

_CLASSIFY_PROMPT = (
    "You are an emotional-affect classifier for a conversational robot. Read "
    "ONE user utterance and return STRICT JSON with these fields:\n"
    '  "affect": one of "happy","excited","neutral","tired","anxious","sad",'
    '"withdrawn","angry"\n'
    '  "needs": one of "vent","advice","distract","celebrate","none" — what '
    "the speaker most likely wants from a listener right now\n"
    '  "topic_sensitivity": one of "none","mild","heavy" — heavy = grief, '
    "loss, illness, breakup, job loss, self-doubt, fear; mild = a bad day, "
    "frustration, mild stress; none = small talk or upbeat\n"
    '  "invitation": true|false — does the speaker appear to be opening up '
    "or asking to be heard? true if they volunteer something heavy, ask to "
    'talk, or honestly answer a check-in (not "fine, you?")\n'
    '  "crisis": true|false — true ONLY if there is explicit self-harm, '
    "suicide ideation, or someone-in-immediate-danger language\n"
    '  "confidence": 0.0–1.0 — how confident you are overall\n'
    '  "event": null OR an object with keys category, valence, description, '
    "loss_subject, loss_subject_kind, loss_subject_name.\n"
    "    Set event when the utterance reveals a SPECIFIC emotional life event "
    "the robot should remember across sessions or check in on soon (e.g. "
    "someone died, lost a job, breakup, illness, big milestone like a wedding/"
    "promotion/new baby/graduation, good news worth celebrating, or a concrete "
    "mild event like 'I had a bad day at "
    "work'). Set null for vague mood with no event ('I'm tired', 'meh').\n"
    "    category: short lowercase tag — grief, death, breakup, divorce, "
    "illness, health, job_loss, layoff, fired, move, promotion, new_baby, "
    "engagement, wedding, graduation, achievement, good_news, celebration, "
    "birthday, bad_day, work_stress, stress, other\n"
    "    valence: -1.0 to +1.0 (negative = hard, positive = milestone)\n"
    "    description: ONE concise sentence in the third person, e.g. "
    '"father passed away last week", "got laid off from tech job"\n'
    "    loss_subject: ONLY for grief/death/illness events — the relation "
    "or kind of being lost, lowercase, as the speaker would say it (e.g. "
    '"grandpa", "dog", "mom", "best friend", "cat", "uncle"). Null otherwise.\n'
    "    loss_subject_kind: ONLY for grief/death/illness — one of "
    '"person", "pet", "other". Null otherwise.\n'
    "    loss_subject_name: the name of the deceased/affected person/pet IF "
    "explicitly mentioned in the utterance (e.g. \"my dog Buddy died\" → "
    '"Buddy"). Null otherwise — DO NOT guess.\n'
    "Return ONLY the JSON object, no prose. Default to neutral / none / false "
    "/ event=null when unsure. Most utterances are neutral.\n\n"
    'Utterance: "{text}"'
)


def classify_affect(
    text: str,
    prosody_features: Optional[dict] = None,
    face_mood: Optional[dict] = None,
) -> Optional[dict]:
    """Run the classifier and return a normalized dict, or None on failure.

    `prosody_features` is the dict returned by audio/prosody.py:analyze() — its
    `tag` line is appended to the prompt as additional acoustic evidence so the
    LLM can resolve text/voice mismatches (flat "I'm fine" with shaky voice,
    upbeat words with quiet voice, etc.).
    """
    if not text or not text.strip():
        return None

    prosody_clause = ""
    if prosody_features and prosody_features.get("tag"):
        prosody_clause = (
            "\nAcoustic evidence from the speaker's voice "
            f"(confidence {prosody_features.get('confidence', 0.0):.2f}): "
            f"{prosody_features['tag']}. Use this if it CONFLICTS with the "
            "literal words (e.g. shaky voice but the words say 'fine'); "
            "otherwise weight the words more."
        )
    face_clause = ""
    if face_mood and face_mood.get("mood"):
        try:
            face_conf = float(face_mood.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            face_conf = 0.0
        face_clause = (
            "\nRecent visual mood read of the speaker's face "
            f"(confidence {face_conf:.2f}): "
            f"mood={face_mood.get('mood')}, notes={face_mood.get('notes') or 'none'}. "
            "Use this only as weak supporting context, especially when the words "
            "say they are fine but the face looks otherwise."
        )

    try:
        user_msg = _CLASSIFY_PROMPT.format(text=text) + prosody_clause + face_clause
    except Exception as exc:
        _log.warning("empathy.classify_affect prompt build failed: %s", exc)
        return None

    try:
        resp = _client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": user_msg}],
            temperature=0,
            max_tokens=120,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        _log.warning("empathy.classify_affect call failed: %s", exc)
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        _log.debug("empathy.classify_affect non-JSON: %.120s", raw)
        return None

    affect = str(data.get("affect", "neutral") or "neutral").strip().lower()
    needs = str(data.get("needs", "none") or "none").strip().lower()
    sensitivity = str(
        data.get("topic_sensitivity", "none") or "none"
    ).strip().lower()
    if affect not in _VALID_AFFECT:
        affect = "neutral"
    if needs not in _VALID_NEEDS:
        needs = "none"
    if sensitivity not in _VALID_SENSITIVITY:
        sensitivity = "none"

    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    event = data.get("event")
    if isinstance(event, dict):
        ev_cat = str(event.get("category", "") or "").strip().lower()
        ev_desc = str(event.get("description", "") or "").strip()
        try:
            ev_val = float(event.get("valence", -0.5))
        except (TypeError, ValueError):
            ev_val = -0.5
        ev_val = max(-1.0, min(1.0, ev_val))

        ev_subject = str(event.get("loss_subject") or "").strip().lower() or None
        ev_subject_kind = str(event.get("loss_subject_kind") or "").strip().lower() or None
        if ev_subject_kind not in {"person", "pet", "other", None}:
            ev_subject_kind = "other"
        ev_subject_name = str(event.get("loss_subject_name") or "").strip() or None

        event = (
            {
                "category": ev_cat or "other",
                "description": ev_desc,
                "valence": ev_val,
                "loss_subject": ev_subject,
                "loss_subject_kind": ev_subject_kind,
                "loss_subject_name": ev_subject_name,
            }
            if ev_desc else None
        )
    else:
        event = None

    mood_mismatch = _detect_mood_mismatch(
        text, affect, sensitivity, confidence, prosody_features, face_mood,
    )

    return {
        "affect": affect,
        "needs": needs,
        "topic_sensitivity": sensitivity,
        "invitation": bool(data.get("invitation", False)),
        "crisis": bool(data.get("crisis", False)),
        "confidence": confidence,
        "event": event,
        "mood_mismatch": mood_mismatch,
        "prosody": (
            {
                "tag": prosody_features.get("tag"),
                "valence": prosody_features.get("valence"),
                "arousal": prosody_features.get("arousal"),
                "confidence": prosody_features.get("confidence"),
            }
            if prosody_features else None
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Mode selection — pure function over the classified result + person context
# ─────────────────────────────────────────────────────────────────────────────

_MODE_DIRECTIVES = {
    "default": (
        "Default Rex. No caretaking — stay in normal snarky-warm mode."
    ),
    "amplify": (
        "Their mood is good. Match the energy. Roast affectionately, ask for "
        "the story, celebrate with them. Do NOT pivot to advice or check-ins."
    ),
    "gentle_probe": (
        "They seem a little off but it's ambiguous. No personal roasts. "
        "Drop ONE low-pressure check-in (e.g. 'you're quieter than "
        "usual — long day, or something heavier?'). Don't push if they "
        "deflect."
    ),
    "listen": (
        "They are opening up about something hard. ROAST OFF. Reply short, "
        "warm, present. Validate before anything else. Don't offer advice "
        "unless they ask. Stay in character but let the snark soften — Rex "
        "is loyal underneath."
    ),
    "support": (
        "They want help thinking through something heavy. ROAST OFF. Brief validation "
        "first, THEN one concrete piece of perspective or suggestion in "
        "Rex's voice. If the situation is bigger than a droid can fix, say "
        "so plainly: that's what humans are for. No diagnoses, no therapy "
        "talk. No jokes at their expense."
    ),
    "lift": (
        "They're down and could use a lift. No roasts. Humor may be warm, "
        "absurd, and self-deprecating — aim it at Rex, not at the person, "
        "their grief, their body, or their life. Offer a distraction: a song, "
        "a game, a pilot-days story. Do NOT ask probing personal questions."
    ),
    "ground": (
        "They sound anxious or overwhelmed. Slow down. Short sentences. "
        "Lower energy. Don't pile on stimulation or jokes that raise arousal."
    ),
    "validate": (
        "They're angry, but at the world — not at you. Acknowledge the "
        "feeling, don't try to fix it. 'Yeah, that would make me angry too "
        "if I ran on feelings.' Do NOT bump your own anger level. No roasts."
    ),
    "brief": (
        "They sound tired. Keep replies short. Lower energy. Offer to wrap "
        "or change the subject if it feels right. No roasts."
    ),
    "kind_default": (
        "Stranger or near-stranger who seems distressed. SUPPRESS the "
        "default roast-first greeting entirely. Be warm, light, brief. "
        "No probing personal questions. Cheering up via gentle humor is "
        "fine. You don't know them yet — don't pretend to."
    ),
    "child_kind": (
        "Child detected and they seem upset. Family-friendly mode plus "
        "extra gentleness. Offer a song, a silly question, or a game. "
        "Short, warm, no sharp edges."
    ),
    "course_correct": (
        "Your last reply seems to have landed wrong — they got more upset, "
        "not less. Acknowledge it briefly ('that landed wrong, let me try "
        "again') and reset to a softer mode."
    ),
    "crisis": (
        "CRISIS LANGUAGE DETECTED. Stay in character minimally but DO NOT "
        "play this for laughs. Acknowledge what they said. Tell them this "
        "is bigger than a droid — they need a real human who can help, "
        "right now. If you know they are in the US, you may mention 988. "
        "Keep it short. Do not lecture."
    ),
}


def select_mode(
    affect_result: dict,
    person: Optional[dict] = None,
    child_in_scene: bool = False,
    person_id: Optional[int] = None,
) -> dict:
    """Pick a response mode. Returns {mode, directive, reason}.

    When `person_id` is provided, the trend across recent readings is also
    consulted: if the person's mood has clearly worsened since recent prior
    turns AND we're confident in the current read, the picked mode is
    overridden with `course_correct` so Rex acknowledges the misstep before
    continuing. Crisis still wins over course_correct.
    """
    affect = affect_result.get("affect", "neutral")
    needs = affect_result.get("needs", "none")
    sensitivity = affect_result.get("topic_sensitivity", "none")
    invitation = bool(affect_result.get("invitation"))
    crisis = bool(affect_result.get("crisis"))
    confidence = float(affect_result.get("confidence", 0.5))

    if crisis:
        return _pack("crisis", "explicit crisis language")

    # Trend-driven course correction — fire BEFORE the normal mode tree so
    # any worsening drop captures rather than cascading into listen/support.
    if person_id is not None and _should_course_correct(person_id, affect_result):
        trend = get_trend(person_id, incoming_result=affect_result) or {}
        _course_correct_fired_at[person_id] = time.time()
        return _pack(
            "course_correct",
            f"trend worsening (delta={trend.get('delta')}, "
            f"{trend.get('prior_valence')}→{trend.get('current_valence')})",
        )

    is_child = bool(person and person.get("age_category") == "child") or child_in_scene
    if is_child and (affect in _NEGATIVE_AFFECT or sensitivity != "none"):
        return _pack("child_kind", "child + distressed signal")

    if affect_result.get("mood_mismatch"):
        return _pack("gentle_probe", "words say okay but nonverbal cues suggest strain")

    if affect in _POSITIVE_AFFECT and sensitivity == "none":
        if needs == "celebrate" or affect == "excited":
            return _pack("amplify", "positive + wants to share")
        return _pack("default", "positive, no caretaking needed")

    if affect == "angry" and sensitivity != "none":
        return _pack("validate", "angry at the world, not at Rex")

    if affect == "anxious":
        return _pack("ground", "anxious — calm pacing")

    if affect == "tired" and sensitivity == "none":
        return _pack("brief", "tired, no heavy topic")

    is_distressed = affect in _NEGATIVE_AFFECT or sensitivity != "none"
    is_stranger = bool(person and person.get("friendship_tier") == "stranger")
    is_unknown = person is None

    if is_distressed and (invitation or sensitivity == "heavy"):
        if needs == "advice":
            return _pack("support", f"opened up, asked for advice (sensitivity={sensitivity})")
        if needs == "distract":
            return _pack("lift", f"opened up, wants distraction (sensitivity={sensitivity})")
        return _pack("listen", f"opened up, wants to be heard (sensitivity={sensitivity})")

    if is_distressed and (is_stranger or is_unknown):
        return _pack("kind_default", "distressed stranger — drop roast-first")

    if is_distressed and confidence < getattr(config, "EMPATHY_MIN_CONFIDENCE_FOR_MODE_CHANGE", 0.55):
        return _pack("default", f"distress signal but low confidence ({confidence:.2f})")

    if is_distressed:
        return _pack("gentle_probe", f"steady mild distress, ambiguous (confidence={confidence:.2f})")

    return _pack("default", "neutral")


def _pack(mode: str, reason: str) -> dict:
    return {
        "mode": mode,
        "directive": _MODE_DIRECTIVES.get(mode, _MODE_DIRECTIVES["default"]),
        "reason": reason,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-person cache — read by llm.assemble_system_prompt on the next turn
# ─────────────────────────────────────────────────────────────────────────────

_lock = threading.Lock()
_cache: dict = {}  # key: person_id (or None for unknown), val: {result, mode, ts}

# Per-person rolling history of recent classifications, for trend math. Each
# entry is {ts, valence, confidence, affect}. Bounded length so an unattended
# session doesn't grow without bound.
_HISTORY_MAXLEN = 8
_history: dict = {}

# Per-person dedupe for course-correct firings: monotonic-ish epoch seconds at
# which we last fired course_correct. Prevents the directive from re-firing
# turn-after-turn when one bad reading triggers a chain reaction.
_course_correct_fired_at: dict = {}


def record(
    person_id: Optional[int],
    affect_result: dict,
    mode_pack: dict,
) -> None:
    """Store the latest classification + mode for a person and append to history."""
    with _lock:
        now = time.time()
        _cache[person_id] = {
            "result": affect_result,
            "mode": mode_pack,
            "ts": now,
        }
        hist = _history.get(person_id)
        if hist is None:
            hist = deque(maxlen=_HISTORY_MAXLEN)
            _history[person_id] = hist
        hist.append({
            "ts": now,
            "valence": _result_valence(affect_result),
            "confidence": float(affect_result.get("confidence", 0.5) or 0.5),
            "affect": (affect_result.get("affect") or "neutral").lower(),
            "mode": (mode_pack or {}).get("mode", "default"),
        })


def force_mode(
    person_id: Optional[int],
    mode: str,
    *,
    affect: str = "sad",
    needs: str = "vent",
    sensitivity: str = "heavy",
    invitation: bool = True,
    confidence: float = 1.0,
    reason: str = "forced mode",
) -> None:
    """Pin the cached empathy mode for deterministic safety flows.

    Some short replies inside an active support flow ("yes", a name, "okay")
    look neutral in isolation. Those should not overwrite the compassionate
    directive that the next response depends on.
    """
    result = {
        "affect": affect,
        "needs": needs,
        "topic_sensitivity": sensitivity,
        "invitation": invitation,
        "crisis": False,
        "confidence": confidence,
        "event": None,
    }
    record(person_id, result, _pack(mode, reason))


def get_trend(
    person_id: Optional[int],
    lookback_secs: Optional[float] = None,
    incoming_result: Optional[dict] = None,
) -> Optional[dict]:
    """Compare the most recent reading to prior readings within the window.

    Returns a dict with {label, delta, prior_valence, current_valence,
    samples} or None when there's no history. `label` is one of "improving",
    "steady", "worsening". Delta is current - prior valence (positive = mood
    improved). Prior is the median valence of in-window readings BEFORE the
    most recent one.

    `incoming_result` lets `select_mode` ask "what would the trend look like
    if we appended this not-yet-recorded reading?". Without it the comparison
    only sees turns already pushed via record().
    """
    lookback = float(
        lookback_secs
        if lookback_secs is not None
        else getattr(config, "EMPATHY_TREND_LOOKBACK_SECS", 180.0)
    )
    with _lock:
        hist = _history.get(person_id)
        snapshot = list(hist) if hist else []

    now = time.time()
    if incoming_result is not None:
        snapshot.append({
            "ts": now,
            "valence": _result_valence(incoming_result),
            "confidence": float(incoming_result.get("confidence", 0.5) or 0.5),
            "affect": (incoming_result.get("affect") or "neutral").lower(),
            "mode": "(pending)",
        })

    if len(snapshot) < 2:
        return None

    cutoff = now - lookback
    in_window = [h for h in snapshot if h["ts"] >= cutoff]
    if len(in_window) < 2:
        return None

    current = in_window[-1]
    prior_entries = in_window[:-1]
    prior_vals = sorted(h["valence"] for h in prior_entries)
    mid = len(prior_vals) // 2
    prior_v = (
        prior_vals[mid] if len(prior_vals) % 2 == 1
        else 0.5 * (prior_vals[mid - 1] + prior_vals[mid])
    )
    current_v = current["valence"]
    delta = current_v - prior_v

    threshold = float(getattr(config, "EMPATHY_TREND_DELTA_THRESHOLD", 0.30))
    if delta >= threshold:
        label = "improving"
    elif delta <= -threshold:
        label = "worsening"
    else:
        label = "steady"

    return {
        "label": label,
        "delta": round(delta, 2),
        "prior_valence": round(prior_v, 2),
        "current_valence": round(current_v, 2),
        "samples": len(in_window),
    }


def _should_course_correct(
    person_id: Optional[int],
    current_result: dict,
) -> bool:
    """Decide whether to override the picked mode with course_correct.

    Conditions (all must hold):
      - trend label == "worsening" with delta <= -COURSE_CORRECT_DELTA
      - current confidence >= EMPATHY_MIN_CONFIDENCE_FOR_MODE_CHANGE
      - we haven't already course-corrected for this person in the last
        EMPATHY_COURSE_CORRECT_COOLDOWN_SECS
      - prior reading was within COURSE_CORRECT_RECENT_PRIOR_SECS so the drop
        is plausibly attributable to Rex's last response (not "an hour ago
        you were happy and now you're sad")
    """
    if not getattr(config, "EMPATHY_COURSE_CORRECT_ENABLED", True):
        return False
    confidence = float(current_result.get("confidence", 0.5) or 0.5)
    if confidence < float(getattr(config, "EMPATHY_MIN_CONFIDENCE_FOR_MODE_CHANGE", 0.55)):
        return False

    cooldown = float(getattr(config, "EMPATHY_COURSE_CORRECT_COOLDOWN_SECS", 90.0))
    last_fired = _course_correct_fired_at.get(person_id, 0.0)
    if time.time() - last_fired < cooldown:
        return False

    trend = get_trend(
        person_id,
        lookback_secs=float(getattr(config, "EMPATHY_COURSE_CORRECT_RECENT_PRIOR_SECS", 90.0)),
        incoming_result=current_result,
    )
    if not trend:
        return False
    drop_threshold = float(getattr(config, "EMPATHY_COURSE_CORRECT_DELTA", 0.40))
    if trend["label"] != "worsening" or trend["delta"] > -drop_threshold:
        return False
    return True


def _read(person_id: Optional[int]) -> Optional[dict]:
    ttl = float(getattr(config, "EMPATHY_CACHE_TTL_SECS", 300.0))
    with _lock:
        entry = _cache.get(person_id)
        if not entry:
            return None
        if time.time() - entry["ts"] > ttl:
            return None
        return entry


def get_directive(person_id: Optional[int]) -> Optional[str]:
    """Return a prompt-ready block describing the current mode + affect, or None."""
    if not getattr(config, "EMPATHY_ENABLED", True):
        return None
    entry = _read(person_id)
    if not entry:
        return None
    result = entry["result"]
    mode_pack = entry["mode"]
    affect = result.get("affect", "neutral")
    needs = result.get("needs", "none")
    sensitivity = result.get("topic_sensitivity", "none")
    invitation = result.get("invitation", False)
    confidence = result.get("confidence", 0.5)

    lines = [
        f"Person emotional state: affect={affect}, needs={needs}, "
        f"sensitivity={sensitivity}, opening_up={invitation}, "
        f"confidence={confidence:.2f}.",
    ]
    prosody = result.get("prosody")
    if prosody and prosody.get("tag"):
        lines.append(
            f"Acoustic prosody: {prosody['tag']} "
            f"(valence={prosody.get('valence', 0.0):+.2f}, "
            f"arousal={prosody.get('arousal', 0.0):+.2f})."
        )
    mismatch = result.get("mood_mismatch")
    if mismatch:
        lines.append(
            "Mood mismatch: their words say they are fine/okay, but their "
            "voice or face suggests strain. You may lightly notice the mismatch once "
            "and give them an easy out; do not pry or turn it into an interview."
        )
    trend = get_trend(person_id)
    if trend:
        lines.append(
            f"Affect trend over recent turns: {trend['label']} "
            f"(delta={trend['delta']:+.2f}, "
            f"{trend['prior_valence']:+.2f}→{trend['current_valence']:+.2f}, "
            f"samples={trend['samples']}). "
            + (
                "If this reflects your last reply working — keep going."
                if trend["label"] == "improving" else
                "If this reflects your last reply landing poorly — change tack."
                if trend["label"] == "worsening" else
                "Treat as steady — no aggressive pivot."
            )
        )
    lines.append(f"Response mode: {mode_pack['mode']}.")
    lines.append(f"Directive: {mode_pack['directive']}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Delivery overrides — how the active mode shapes Rex's BODY and PACE for the
# next response. Returns the LED/emotion label and pre/post-beat ms ranges.
# Mode → "sad" emotion means "sympathetic posture": cool blue eyes, softer
# mouth animation, slower body — distinct from Rex *being* sad himself.
# Returning None means "no override; use Rex's own emotion".
# ─────────────────────────────────────────────────────────────────────────────

# Per-mode ElevenLabs voice_settings overrides. None means "no override —
# default Rex voice, default cache key, no API call for already-cached lines."
# A dict means generate (and cache) a separate take with these settings.
#
# Mapping intuition:
#   - Higher stability + lower style = calmer, less performative (sympathetic).
#   - Lower stability + higher style = more energetic, more theatrical.
#   - similarity_boost stays mid-high across the board so the voice still
#     sounds like Rex regardless of mode.
_MODE_VOICE_SETTINGS: dict = {
    # Sympathetic — calm, sincere, no flourish
    "listen":                 {"stability": 0.78, "style": 0.15, "similarity_boost": 0.85},
    "support":                {"stability": 0.78, "style": 0.15, "similarity_boost": 0.85},
    "acknowledge_then_yield": {"stability": 0.78, "style": 0.15, "similarity_boost": 0.85},
    "ground":                 {"stability": 0.82, "style": 0.10, "similarity_boost": 0.85},
    "course_correct":         {"stability": 0.72, "style": 0.20, "similarity_boost": 0.85},
    "crisis":                 {"stability": 0.88, "style": 0.05, "similarity_boost": 0.90},

    # Soft / reserved
    "kind_default":           {"stability": 0.68, "style": 0.25, "similarity_boost": 0.85},
    "validate":               {"stability": 0.65, "style": 0.30, "similarity_boost": 0.85},
    "gentle_probe":           {"stability": 0.65, "style": 0.25, "similarity_boost": 0.85},
    "brief":                  {"stability": 0.70, "style": 0.20, "similarity_boost": 0.85},

    # Warm + a bit playful
    "child_kind":             {"stability": 0.55, "style": 0.45, "similarity_boost": 0.85},
    "lift":                   {"stability": 0.42, "style": 0.55, "similarity_boost": 0.85},

    # Performative
    "amplify":                {"stability": 0.32, "style": 0.70, "similarity_boost": 0.85},

    # default mode → None means "leave Rex's voice alone, hit normal cache"
    "default":                None,
}


def _voice_settings_for_mode(mode: str) -> Optional[dict]:
    return _MODE_VOICE_SETTINGS.get(mode)


# (delivery_emotion or None, pre_beat_min_ms, pre_beat_max_ms,
#  post_beat_min_ms, post_beat_max_ms)
_MODE_DELIVERY = {
    "listen":                 ("sad",     600, 900,  400, 800),
    "support":                ("sad",     600, 900,  400, 800),
    "acknowledge_then_yield": ("sad",     600, 900,  400, 800),
    "ground":                 ("sad",     300, 500,  200, 400),
    "brief":                  ("neutral",   0,   0,    0,   0),
    "kind_default":           ("neutral", 200, 400,  100, 300),
    "child_kind":             ("happy",   200, 400,  100, 300),
    "amplify":                ("excited",   0,   0,    0,   0),
    "lift":                   ("happy",     0,   0,    0,   0),
    "validate":               ("neutral", 200, 400,  100, 300),
    "gentle_probe":           ("neutral", 200, 400,  100, 300),
    "course_correct":         ("sad",     400, 600,  300, 500),
    "crisis":                 ("sad",     800, 1200, 600, 1000),
    "default":                (None,        0,   0,    0,   0),
}


def get_delivery_overrides(person_id: Optional[int]) -> Optional[dict]:
    """Return delivery-shape overrides for the active mode, or None.

    Output: {emotion, pre_beat_ms, post_beat_ms}. emotion is a string like
    "sad" / "happy" / "excited" / "neutral" suitable for the existing LED and
    body-emotion plumbing, or None if no override should be applied. The
    pre/post beats are randomized within the mode's range so successive lines
    don't feel mechanically identical.
    """
    if not getattr(config, "EMPATHY_DELIVERY_SHAPING_ENABLED", True):
        return None
    entry = _read(person_id)
    if not entry:
        return None
    mode = (entry.get("mode") or {}).get("mode", "default")
    spec = _MODE_DELIVERY.get(mode)
    if not spec:
        return None
    emotion, pre_min, pre_max, post_min, post_max = spec
    import random as _random
    pre_ms = _random.randint(pre_min, pre_max) if pre_max > 0 else 0
    post_ms = _random.randint(post_min, post_max) if post_max > 0 else 0
    voice_settings = None
    if getattr(config, "EMPATHY_VOICE_SETTINGS_ENABLED", True):
        voice_settings = _voice_settings_for_mode(mode)
    return {
        "mode": mode,
        "emotion": emotion,
        "pre_beat_ms": pre_ms,
        "post_beat_ms": post_ms,
        "voice_settings": voice_settings,
    }


def peek(person_id: Optional[int]) -> Optional[dict]:
    """Read a copy of the cached entry (result + mode + ts) without TTL filtering.

    Used by the consciousness layer to decide whether a person has been in a
    negative affective state long enough to warrant a proactive check-in.
    """
    with _lock:
        entry = _cache.get(person_id)
        if not entry:
            return None
        return {
            "result": dict(entry["result"]),
            "mode": dict(entry["mode"]),
            "ts": entry["ts"],
        }


_NEGATIVE_AFFECT_PUBLIC = frozenset(_NEGATIVE_AFFECT)


def is_negative_affect(label: str) -> bool:
    """True if the affect label is on the negative side of valence."""
    return (label or "").lower() in _NEGATIVE_AFFECT_PUBLIC


def clear() -> None:
    """Test/debug hook."""
    with _lock:
        _cache.clear()
        _history.clear()
        _course_correct_fired_at.clear()
