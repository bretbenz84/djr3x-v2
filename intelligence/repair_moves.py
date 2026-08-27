"""
intelligence/repair_moves.py - explicit conversational repair handling.

When the human says Rex misheard, misunderstood, pushed too hard, or landed a
line badly, this module turns that into a small repair move instead of letting
the normal roast/curiosity machinery treat it like ordinary banter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
import re
import threading
import time
from typing import Optional

from utils.audio_tags import strip_audio_tags


_MISHEARD_PAT = re.compile(
    r"\b(?:"
    r"you (?:misheard|heard wrong|heard me wrong)|you misheard me|"
    r"you (?:didn'?t|did not) (?:hear|catch|get) (?:me|that|it|what i said)|"
    r"i (?:didn'?t say|did not say)|"
    r"that'?s not what i said|that is not what i said|not what i said|"
    r"that'?s not (?:his|her|their|my) name|that is not (?:his|her|their|my) name|"
    # A NAME CORRECTION needs a correction cue in front of it. The bare
    # "(his|her|their|the) name was" fired on ordinary storytelling — field
    # 2026-07-24: "We got it from a dear friend who lived on a boat. Her name
    # was... Goldnatt." was ruled a misheard-repair, so Rex answered a warm story
    # with "Static in the receptors; that came through garbled." and the next two
    # turns were spent untangling it. Narration and answers ("her name was Ada")
    # must stay ordinary conversation.
    r"(?:no,?\s+|actually,?\s+|not\s+)(?:his|her|their|the|my)\s+name\s+(?:was|is)|"
    r"wrong word|wrong name|got my words wrong|transcribed (?:it )?wrong|mis-?transcribed"
    r")\b"
    # Bare "i said" means a correction ("no, I said blues") — but NOT when it's an
    # emphasis/reference lead-in ("like/as I said") or a recall question ("what I said",
    # already guarded in detect()). Exclude those so they don't fire a misheard repair.
    r"|(?<!like )(?<!as )(?<!what )(?<!that )\bi said\b",
    re.IGNORECASE,
)
_MISUNDERSTOOD_PAT = re.compile(
    r"\b(you (misunderstood|got it wrong|missed the point)|"
    r"that's not what i meant|that is not what i meant|not what i meant|"
    r"you'?re missing (the point|what i mean)|"
    r"no,? not that|no,? that's wrong|no,? that is wrong|incorrect)\b",
    re.IGNORECASE,
)
_FACTUAL_PAT = re.compile(
    r"\b(you made that up|you'?re making that up|that'?s not true|that is not true|"
    r"you invented that|don'?t assume|do not assume|that didn'?t happen|"
    r"that did not happen|where did you get that|you hallucinated)\b",
    re.IGNORECASE,
)
_TONE_PAT = re.compile(
    r"\b(that was (rude|mean|harsh|uncalled for|distasteful|not funny|too much)|"
    r"that wasn't funny|that wasnt funny|you were (rude|mean|harsh)|"
    r"don'?t roast|do not roast|stop roasting|not a joke|not (?:very )?funny|"
    r"too mean|you went too far)\b",
    re.IGNORECASE,
)
_PACING_PAT = re.compile(
    r"\b(too many questions|stop asking|so many questions|why are you asking|"
    r"this feels like an interview|not an interview|slow down|give me a second|"
    r"let me think)\b",
    re.IGNORECASE,
)
_INTERRUPT_PAT = re.compile(
    r"\b(you interrupted|you cut me off|you talked over me|let me finish|"
    r"i wasn'?t done|i was still talking|"
    r"didn'?t (?:give me (?:any |enough )?time|let me) (?:to )?"
    r"(?:answer|respond|finish|reply|speak|talk|think)|"
    r"wouldn'?t let me (?:answer|respond|finish|reply|speak|talk))\b",
    re.IGNORECASE,
)
# The proper-name alternatives are matched CASE-SENSITIVELY via a scoped
# (?-i:...) group so IGNORECASE does not let [A-Z][a-z]+ match any lowercase
# word — that bug made "you mean my telescope" parse as a person correction and
# drove Rex to defend the bad guess ("Your telescope, not mine") instead of
# dropping it. An object/detail correction now falls through to grounding/factual.
_WRONG_PERSON_PAT = re.compile(
    r"\b(wrong person|not me|that wasn'?t me|that was not me|"
    r"you mean (him|her|them|(?-i:[A-Z][a-z'\-]+))|"
    r"you'?re talking to (him|her|them|the wrong person)|"
    r"that was (him|her|them|(?-i:[A-Z][a-z'\-]+)))\b",
    re.IGNORECASE,
)
# "Rex invented/assumed a detail" corrections + the natural "that makes no
# sense" rejection. Routed to the existing 'factual' frame (remove the invented
# detail, do not defend it). Gated on Rex having spoken recently so a bare "makes
# no sense" about a third-party topic is not swept up.
_GROUNDING_CORRECTION_PAT = re.compile(
    r"\b((that|this|it) (makes no sense|doesn'?t make (any |much )?sense|"
    r"made no sense|makes zero sense)|"
    r"none of (that|this) (makes sense|is right)|"
    # second-person confusion: the human says REX is the one not making sense
    # (field bug: "You're not saying anything that makes sense" got a roast
    # comeback instead of a humble reset).
    r"you'?re not making (any )?sense|"
    r"(you'?re )?not saying anything that makes sense|"
    r"nothing you('?re| are) saying makes (any )?sense|"
    r"you'?re (just )?making (that|this|it|stuff) up|"
    r"you (just )?(invented|assumed|made (?:that|this|it|the|a)? ?up|made up)\b|"
    r"i never said (that|this)|i didn'?t mention|where are you getting (that|this))\b",
    re.IGNORECASE,
)
# "(No,) I didn't say that" on its own disagrees with what Rex just CLAIMED — it
# is not an identity correction ("wrong person") or a transcription fix. Used to
# misfire as wrong_person → "faulty coordinate… better luck next time", derailing
# the conversation. Let a bare denial flow to normal conversational handling; a
# real correction ("I didn't say jazz, I said blues") carries more and still
# routes through the misheard/correction path below.
_BARE_CONTENT_DENIAL_PAT = re.compile(
    r"^\s*(?:no[,!.]?\s*|nope[,!.]?\s*|nah[,!.]?\s*)?"
    r"i\s+(?:didn'?t|did not)\s+say\s+that\b[.!]?\s*$",
    re.IGNORECASE,
)
# Asking Rex to RECALL or confirm he FOLLOWED what the user said ("do you recall what I
# said?", "did you (not) follow what I said?", "what did I say?") is a comprehension /
# recall request, NOT a misheard-correction. The bare "i said" alternative in _MISHEARD_PAT
# used to swallow these and fire a canned recovery line ("Consider it logged. Onward.")
# instead of letting Rex actually recall from the transcript. Hand them to normal
# conversation so the reply path (which has the recent transcript) can answer.
_RECALL_REQUEST_PAT = re.compile(
    r"\b(?:"
    r"(?:do|did|don'?t|can|could|would|are)\s+you\s+"
    r"(?:even\s+|not\s+|actually\s+|really\s+)*"
    r"(?:recall|remember|follow|catch|hear|get|register|understand|know|see)\s+"
    r"(?:what|when|why|how|that)\s+i\s+(?:said|asked|told|meant|answered)"
    r"|what\s+did\s+i\s+(?:say|ask|answer|just\s+say|tell\s+you)"
    r"|(?:repeat\s+back|recall|remind\s+me)\s+(?:of\s+)?(?:what\s+)?i\s+(?:said|asked|answered)"
    r")",
    re.IGNORECASE,
)
# "I didn't say anything" is NOT a mishearing — it asserts there were no words to
# mishear, because Rex answered phantom audio. Field 2026-08-27 13:34: Rex's own boot
# line ("Ready to go. Statistically, one of us is about to say something interesting.")
# came back through the mic, was salvaged as "Okay.", and Rex answered it with "Okay
# what, exactly?". Bret's "I didn't say anything." then matched the bare `i didn'?t say`
# alternative in _MISHEARD_PAT, so Rex said "Ah, my audio processor fumbled that one.
# Run that by me one more time?" — and the denial of THAT ("There's nothing to run by
# you. I didn't say anything.") came back as another turn and earned "Hit me with it
# again and I'll get it right." Asking for a repeat of speech that never existed cannot
# terminate: every denial re-triggers the ask. "I didn't say anything ABOUT the cat" is
# deliberately excluded by the trailing lookahead — that is a content dispute and keeps
# its existing routing.
_PHANTOM_AUDIO_PAT = re.compile(
    r"\b(?:"
    r"i (?:didn'?t|did not) (?:say|speak|utter)"
    r"\s+(?:anything|a word|a thing|nothing|at all)"
    # PAST tense only. "I never say anything interesting" is self-deprecation, not
    # a denial that this turn happened.
    r"|i never (?:said|spoke|uttered)"
    r"\s+(?:anything|a word|a thing|nothing|at all)"
    r"|i (?:said|spoke) nothing"
    r"|i (?:didn'?t|did not) speak"
    r"|i (?:wasn'?t|was not) (?:talking|speaking|saying anything)"
    r"|(?:nobody|no one|no-one) (?:said|spoke|was talking|was speaking)"
    r"(?:\s+(?:anything|a word|a thing|at all))?"
    r"|(?:that|it) (?:wasn'?t|was not) me (?:talking|speaking)"
    r"|nothing was said"
    # The denial must END the clause. A blocklist of following words could not
    # work: it only ever sees the word after the VERB, so "Nobody said anything
    # for like a full minute, it was so awkward" — ordinary storytelling about a
    # silence — was answered with "that was my own echo coming back at me".
    # Anything that continues the sentence (a narrative tail, or a content
    # defence like "I didn't say anything WRONG") keeps its old routing.
    r")\b(?=\s*(?:[.!?;:]|$))",
    re.IGNORECASE,
)
# "There's nothing to run by you" / "there's nothing to hit you again with" only mean
# "you heard a ghost" as an ANSWER to Rex's own ask-to-repeat. Standalone they are
# ordinary speech — "there's nothing to worry about" must stay conversation — so
# detect() gates this one on Rex having just asked for a repeat, or on a phantom
# stand-down still being warm.
_NOTHING_TO_REPEAT_PAT = re.compile(
    r"\b(?:there'?s|there is|there was|i'?ve got|i have|i got)\s+nothing\s+to\s+"
    r"(?:repeat|say|add|run|hit|give|tell)\b",
    re.IGNORECASE,
)
# The verbs above are only unambiguous when Rex's OWN last line asked for a
# repeat. On the warm stand-down window alone they are ordinary speech — "I have
# nothing to add", "there's nothing to say" — and answering those with "that was
# my own echo" is its own non-sequitur. Only the verbs that can ONLY refer back
# to an ask survive on the window.
_NOTHING_TO_REPEAT_STRICT_PAT = re.compile(
    r"\b(?:there'?s|there is|there was|i'?ve got|i have|i got)\s+nothing\s+to\s+"
    r"(?:repeat|run|hit)\b",
    re.IGNORECASE,
)
# Rex's OWN two ask-to-repeat pools: _REPROMPT_LINES below, and
# _LOW_TRUST_REPROMPT_LINES in intelligence/interaction.py (which cannot be imported
# here without a cycle). Kept as one pattern so both lanes are recognised — the
# 2026-08-27 loop alternated between them, so a check that only knew one would miss.
_ASK_TO_REPEAT_PAT = re.compile(
    r"(?:say (?:it|that) again|say (?:it|that) one more time|said again|"
    r"run (?:that|it) by me|one more time|once more|give it to me again|"
    r"hit me with it again|what'?d you say|what did you say|"
    r"what was that|didn'?t catch that|come again)",
    re.IGNORECASE,
)


def looks_like_ask_to_repeat(text: str) -> bool:
    """True when `text` is one of Rex's own 'say that again' lines (either pool)."""
    return bool(_ASK_TO_REPEAT_PAT.search(text or ""))


def _last_assistant_snapshot() -> str:
    with _lock:
        return _last_assistant_text


_PRONOUN_PAT = re.compile(
    r"\b(wrong pronouns?|not (he|she|him|her)|"
    r"(?:i|they|he|she|[A-Z][A-Za-z]+)\s+(?:use|uses|go by|goes by)\s+"
    r"(?:he/him|she/her|they/them)|"
    r"(?:he/him|she/her|they/them)\s+pronouns?)\b",
    re.IGNORECASE,
)
_REPEAT_PAT = re.compile(
    r"\b(what did you say|say that again|repeat that|come again|"
    r"i didn'?t hear you|i did not hear you|what was that)\b",
    re.IGNORECASE,
)
_CLARIFY_PAT = re.compile(
    r"\b(what do you mean|what are you talking about|huh|i don'?t get it|"
    r"i do not get it|explain that|clarify|"
    # Softened confusion after a Rex-initiated line (field 2026-08-03 00:01:
    # "I'm not sure what you mean." fell to conversation and got a DISMISSAL —
    # "Fair, I was reaching for old static" — instead of the explanation; the
    # owner had to push back a second time to get it).
    r"(?:i'?m |i am )?not sure what you(?:'re| are)? (?:mean|saying|"
    r"talking about|referring to)|"
    r"(?:i )?(?:don'?t|do not) know what you(?:'re| are| were)? talking about|"
    r"tell me what you (?:were gonna|were going to|meant to) say)\b",
    re.IGNORECASE,
)
_BARE_NEGATION_PAT = re.compile(r"^\s*(no|nope|nah|wrong|incorrect)\s*[.!]?\s*$", re.I)
_CORRECTION_PAT = re.compile(
    r"\b(?:i said|it's|it is|his name was|her name was|their name was|"
    r"that's not his name,? it'?s|that's not her name,? it'?s|"
    r"that's not their name,? it'?s|that's not my name,? it'?s|"
    r"the name was|i meant)\s+(.+)$",
    re.IGNORECASE,
)
_NOT_X_Y_PAT = re.compile(
    r"\b(?:not|it wasn'?t|it was not|that wasn'?t|that was not)\s+(.+?),?\s+"
    r"(?:it'?s|it is|it was|i said|the correct(?:ion)? is)?\s*(.+)$",
    re.IGNORECASE,
)
_NO_COMMA_CORRECTION_PAT = re.compile(
    r"^\s*(?:no|nope|nah|wrong|incorrect)[,\s]+(.+)$",
    re.IGNORECASE,
)

# A BARE restatement: the user simply repeats their real turn with an "I said / I meant …"
# lead-in and NO contrast ("I said I watch a lot of Netflix specials"). That is the user's
# actual turn after a mishear — it should be RESPONDED to, not echoed back as a "correction"
# (the "We'll get there — recalibrating. <your words>." field bug). A CONTRASTIVE correction
# ("I said blues, not jazz") is NOT bare and still flows through the repair-ack path.
_BARE_RESTATEMENT_LEAD_PAT = re.compile(
    r"^\s*(?:um|uh|well|okay|ok|so|yeah|yes)?,?\s*i (?:said|meant)\b",
    re.IGNORECASE,
)
_RESTATEMENT_CONTRAST_PAT = re.compile(
    r"\bnot\b|\bno\b|\bnope\b|\bwrong\b|\bincorrect\b|isn'?t|wasn'?t|aren'?t|"
    r"weren'?t|instead of|rather than|that'?s not|you (?:said|heard|got|thought)",
    re.IGNORECASE,
)


def is_bare_restatement(text: str) -> bool:
    """True when the user RE-STATES content with an 'I said/I meant …' lead-in and NO
    contrast — i.e. repeating their real turn after a mishear, which should be answered, not
    echoed. A contrastive correction ('I said blues, not jazz') returns False (repair path)."""
    t = (text or "").strip()
    if not t or not _BARE_RESTATEMENT_LEAD_PAT.search(t):
        return False
    return not _RESTATEMENT_CONTRAST_PAT.search(t)


_lock = threading.Lock()
_last_assistant_text: str = ""
_last_assistant_at: float = 0.0
_last_repair_at: float = 0.0
_last_tone_repair_at: float = 0.0

# Recovery tags appended to "I misheard you" / bare-negation repairs. A single
# fixed line got repeated on back-to-back repairs and a bystander noticed ("Is it
# just gonna keep asking the same question?"). Rotate with anti-repeat so two
# consecutive repairs never use the same tag.
# Lines may carry an inline v3 [audio tag] ([excited] …): appended after the repair
# reply it shifts the DELIVERY mid-reply at synthesis (the Star Tours sign-off gets
# genuine droid optimism instead of the reply's tone). Tags are sanitized/stripped
# by audio.tts (whitelist on v3, removed entirely on v2/turbo or when disabled) and
# never reach the transcript/GUI/memory.
# Field 2026-07-31→08-02: the old pool also held pure deflectors ("We'll get
# there — recalibrating.", "Noted. I'll route around that one.", "Consider it
# logged. Onward."). They SOUND like acknowledgment while refusing the repair —
# the owner asked "what do you mean?" and got "Consider it logged. Onward." as
# the whole reply, and a weather correction that also asked a question got
# "Noted. I'll route around that one." with the question eaten. Every tag here
# must OWN the miss; none may close the topic or dodge the human's turn.
_RECOVERY_LINES = [
    "[excited] I'm sure we'll have better luck next time!",
    "Fair enough — let me reset and try that again.",
    "My circuits and I will do better on the next pass.",
    "That one's on my wiring, not you.",
]
BETTER_LUCK_NEXT_TIME = strip_audio_tags(_RECOVERY_LINES[0])  # back-compat alias (clean form)
_last_recovery_line: str = ""


def pick_recovery_line() -> str:
    """A recovery tag that differs from the one used on the previous repair."""
    global _last_recovery_line
    with _lock:
        choices = [
            line for line in _RECOVERY_LINES if line != _last_recovery_line
        ] or list(_RECOVERY_LINES)
        line = random.choice(choices)
        _last_recovery_line = line
        return line


# ── Misheard / misunderstood recovery (no correction supplied) ──────────────────
# When the human flags a mishearing but hasn't re-said it yet, Rex saves face with a
# quick self-deprecating "my circuits glitched" joke, then hands the floor back so they
# can repeat it. Both halves rotate with anti-repeat so back-to-back corrections never
# read identically (a bystander once noticed Rex repeating the same repair line).
_SAVE_FACE_LINES = [
    "[sighs] Ah, my audio processor fumbled that one.",
    "Circuits crossed — that decoded as pure static.",
    "My transcription unit clearly skipped its calibration.",
    "Static in the receptors; that came through garbled.",
    "[excited] I'm sure we'll have better luck next time!",   # Star Tours sign-off, keep it
    "One of my logic boards took an unscheduled coffee break.",
    "My ears run on 90% guesswork and 10% optimism, apparently.",
    "Bad packet on my end — that arrived as nonsense.",
    "I misinterpret life forms sometimes; occupational hazard.",
]
_REPROMPT_LINES = [
    "OK, shoot — what'd you say again?",
    "Run that by me one more time?",
    "Go on, say it again — I'm listening properly now.",
    "Give it to me again?",
    "Once more, for the droid in the back?",
    "Hit me with it again and I'll get it right.",
    "Say that again?",
]
_last_save_face: str = ""
_last_reprompt: str = ""


def _pick_distinct(pool: list[str], last: str) -> str:
    choices = [s for s in pool if s != last] or list(pool)
    return random.choice(choices)


# Words that are part of the correction SIGNAL, not re-said content. A captured
# "correction" made up ENTIRELY of these ("I said", "you misunderstood me") means the
# human flagged a miss without supplying new words — so Rex should ask them to repeat,
# not "accept" a phantom correction.
_CORRECTION_SIGNAL_WORDS = {
    "i", "you", "me", "no", "nope", "nah", "not", "what", "that", "this", "it",
    "said", "say", "saying", "mean", "meant", "didn", "didnt", "don", "dont",
    "is", "was", "misunderstood", "misheard", "heard", "hear", "wrong",
    "the", "a", "an", "my", "your",
}


def correction_has_content(correction: Optional[str]) -> bool:
    """True if `correction` carries actual re-said content ('blues not jazz', 'Tom
    Foster'), not just an echo of the correction signal ('I said', 'you misunderstood
    me'). Lets the repair path tell 'here's the fix' from a bare 'you got it wrong'."""
    words = re.findall(r"[a-z']+", (correction or "").lower())
    return any(w.strip("'") not in _CORRECTION_SIGNAL_WORDS for w in words if w.strip("'"))


def misheard_recovery_response() -> str:
    """Two beats for an ASR/comprehension miss with NO correction supplied: a short
    save-face circuit-glitch joke + a varied invitation to repeat. Both rotate with
    anti-repeat. Used instead of the LLM 'own it and move on' path so Rex actually hands
    the floor back when the human hasn't re-said the thing yet."""
    global _last_save_face, _last_reprompt
    with _lock:
        joke = _pick_distinct(_SAVE_FACE_LINES, _last_save_face)
        _last_save_face = joke
        ask = _pick_distinct(_REPROMPT_LINES, _last_reprompt)
        _last_reprompt = ask
    return f"{joke} {ask}"


# Phantom-audio stand-down. NOT A QUESTION — a question here is itself an ask to
# repeat, the human's denial of it comes back as a new turn, and that is precisely the
# 2026-08-27 13:34 loop these lines exist to break ("I didn't say anything." → "Run
# that by me one more time?" → "There's nothing to run by you..." → "Hit me with it
# again"). Own the ghost, close the subject, stop. No line here may end in '?'.
_PHANTOM_AUDIO_LINES = [
    "Ah — that was my own echo coming back at me. Nothing on your end.",
    "My mistake. I answered a ghost in my microphone, not you.",
    "Scratch that. Phantom audio on my side; you said nothing.",
    "Right, no one spoke. My receivers invented that one. Moving on.",
    "That one was my own voice bouncing back. Ignore it — I have.",
]
_last_phantom_line: str = ""
_last_phantom_at: float = 0.0

# Stand-down for the ask-to-repeat CAP, which is a different situation from
# phantom audio: here the human demonstrably DID speak and Rex simply kept
# missing it, so none of the lines above may be reused — telling someone who has
# now said it three times that they "said nothing" is worse than the third ask.
# Still no question anywhere in the pool: a question is another ask to repeat.
_ASK_CAP_STAND_DOWN_LINES = [
    "I've made you say that twice already — I'll stop asking and keep up from here.",
    "My ears are having a bad minute. Not making you repeat it a third time.",
    "That one's on my receivers, not on you. Let's carry on.",
    "I'll quit asking you to run it back. Go ahead.",
]
_last_ask_cap_line: str = ""

# How long a phantom stand-down keeps the ask-to-repeat lanes muzzled.
PHANTOM_STAND_DOWN_WINDOW_SECS = 90.0
# Consecutive "say that again?" moves allowed before Rex stops asking. Field
# 2026-08-27: three asks in 75 seconds — one low-trust reprompt (13:34:17) and two
# misheard repairs (13:34:42, 13:34:57) — and every ask manufactured the next denial
# to ask about. The counter is shared by BOTH lanes on purpose; a per-lane cap just
# lets them take turns.
ASK_TO_REPEAT_STRIKE_CAP = 2
ASK_TO_REPEAT_STRIKE_WINDOW_SECS = 120.0
_ask_to_repeat_strikes: int = 0
_last_ask_to_repeat_at: float = 0.0


def phantom_audio_response() -> str:
    """Stand-down for 'I didn't say anything': own the phantom, ask nothing. Rotates
    with anti-repeat and arms the phantom window that muzzles the reprompt lanes."""
    global _last_phantom_line, _last_phantom_at
    with _lock:
        line = _pick_distinct(_PHANTOM_AUDIO_LINES, _last_phantom_line)
        _last_phantom_line = line
        _last_phantom_at = time.monotonic()
    return line


def ask_cap_stand_down_response() -> str:
    """Stop asking without claiming the human was silent — they weren't. Arms the
    same window phantom_audio_response() does, so the low-trust reprompt lane
    stays muzzled and the two lanes cannot take turns asking."""
    global _last_ask_cap_line, _last_phantom_at
    with _lock:
        line = _pick_distinct(_ASK_CAP_STAND_DOWN_LINES, _last_ask_cap_line)
        _last_ask_cap_line = line
        _last_phantom_at = time.monotonic()
    return line


def phantom_recent(max_age_secs: Optional[float] = None) -> bool:
    """True while a phantom-audio stand-down is still warm."""
    if max_age_secs is None:
        max_age_secs = PHANTOM_STAND_DOWN_WINDOW_SECS
    with _lock:
        last = _last_phantom_at
    return last > 0.0 and (time.monotonic() - last) <= float(max_age_secs)


def note_ask_to_repeat() -> None:
    """Record that Rex just asked the human to say it again — from any lane."""
    global _ask_to_repeat_strikes, _last_ask_to_repeat_at
    with _lock:
        now = time.monotonic()
        if (
            _last_ask_to_repeat_at > 0.0
            and now - _last_ask_to_repeat_at > ASK_TO_REPEAT_STRIKE_WINDOW_SECS
        ):
            _ask_to_repeat_strikes = 0
        _ask_to_repeat_strikes += 1
        _last_ask_to_repeat_at = now


def ask_to_repeat_exhausted() -> bool:
    """True once Rex has asked for a repeat back-to-back too many times: asking again
    is how the 2026-08-27 loop kept itself alive."""
    with _lock:
        strikes = _ask_to_repeat_strikes
        last = _last_ask_to_repeat_at
    if last <= 0.0:
        return False
    if time.monotonic() - last > ASK_TO_REPEAT_STRIKE_WINDOW_SECS:
        return False
    return strikes >= ASK_TO_REPEAT_STRIKE_CAP


def clear_ask_to_repeat_strikes() -> None:
    global _ask_to_repeat_strikes, _last_ask_to_repeat_at
    with _lock:
        _ask_to_repeat_strikes = 0
        _last_ask_to_repeat_at = 0.0


def _norm_apostrophes(s: str) -> str:
    """Fold curly/modifier apostrophes to a straight ' so a substring check survives the
    LLM rendering a recovery line with U+2019 while the constant uses U+0027. Without this,
    the dedup guard misses and add_better_luck_line() appends the recovery line a 2nd time
    (the 'We'll get there — recalibrating. … We'll get there — recalibrating.' field bug)."""
    return (s or "").lower().replace("’", "'").replace("ʼ", "'").replace("‘", "'")


def _contains_recovery_line(text: str) -> bool:
    # Tag-insensitive on BOTH sides: the LLM may echo the recovery line without its
    # authored [audio tag] (or with it) — either way it counts as present, so the
    # line is never appended twice.
    low = _norm_apostrophes(strip_audio_tags(text))
    return any(
        _norm_apostrophes(strip_audio_tags(line)) in low for line in _RECOVERY_LINES
    )


_BETTER_LUCK_REPAIR_KINDS = {
    "misheard",
    "misunderstood",
    "wrong_person",
    "pronoun",
    "factual",
    "bare_negation",
}


@dataclass
class RepairMove:
    kind: str
    severity: str
    user_text: str
    correction: str = ""
    target: str = ""
    last_assistant_text: str = ""
    detected_at: float = 0.0


def clear() -> None:
    global _last_assistant_text, _last_assistant_at, _last_repair_at, _last_tone_repair_at
    global _last_phantom_at, _ask_to_repeat_strikes, _last_ask_to_repeat_at
    with _lock:
        _last_assistant_text = ""
        _last_assistant_at = 0.0
        _last_repair_at = 0.0
        _last_tone_repair_at = 0.0
        _last_phantom_at = 0.0
        _ask_to_repeat_strikes = 0
        _last_ask_to_repeat_at = 0.0


def note_assistant_turn(text: str) -> None:
    # Strip v3 [audio tags]: this text is echoed back verbatim by "repeat" repairs
    # ("I said: …") and compared against user corrections — it must be the clean line.
    cleaned = strip_audio_tags(text).strip()
    if not cleaned:
        return
    global _last_assistant_text, _last_assistant_at
    with _lock:
        _last_assistant_text = cleaned
        _last_assistant_at = time.monotonic()


def detect(user_text: str) -> Optional[dict]:
    cleaned = (user_text or "").strip()
    if not cleaned:
        return None

    lowered = cleaned.lower()
    kind = ""
    severity = "medium"
    requires_recent = False

    # A bare "I didn't say that" is a content disagreement, not a repair — hand
    # it to normal conversation so Rex can engage instead of derailing.
    if _BARE_CONTENT_DENIAL_PAT.match(cleaned):
        return None

    # "Do you recall what I said?" / "did you not follow what I said?" / "what did I say?"
    # are recall/comprehension requests, NOT corrections — let the reply path recall from
    # the transcript instead of firing a canned recovery line (see _RECALL_REQUEST_PAT).
    if _RECALL_REQUEST_PAT.search(cleaned):
        return None

    # A denial that any words were spoken outranks every correction lane: there is
    # nothing to correct, so the only right move is to own the phantom and stop
    # asking. Checked FIRST because "that wasn't me talking" would otherwise be eaten
    # by _WRONG_PERSON_PAT and "I didn't say anything" by _MISHEARD_PAT — which is
    # exactly how the 2026-08-27 13:34 repeat loop started.
    if (
        _PHANTOM_AUDIO_PAT.search(cleaned)
        or (
            _NOTHING_TO_REPEAT_PAT.search(cleaned)
            and looks_like_ask_to_repeat(_last_assistant_snapshot())
        )
        or (
            _NOTHING_TO_REPEAT_STRICT_PAT.search(cleaned)
            and phantom_recent()
        )
    ):
        kind = "phantom_audio"
        severity = "medium"
        requires_recent = True
    elif _INTERRUPT_PAT.search(cleaned):
        kind = "interruption"
        severity = "high"
    elif _TONE_PAT.search(cleaned):
        kind = "tone"
        severity = "high"
    elif _PACING_PAT.search(cleaned):
        kind = "pacing"
        severity = "medium"
    elif _WRONG_PERSON_PAT.search(cleaned):
        kind = "wrong_person"
        severity = "high"
    elif _PRONOUN_PAT.search(cleaned):
        kind = "pronoun"
        severity = "high"
    elif _REPEAT_PAT.search(cleaned):
        kind = "repeat"
        severity = "low"
    elif _CLARIFY_PAT.search(cleaned):
        kind = "clarify"
        severity = "low"
    elif _MISHEARD_PAT.search(cleaned):
        kind = "misheard"
        severity = "medium"
    elif _FACTUAL_PAT.search(cleaned):
        kind = "factual"
        severity = "high"
    elif _GROUNDING_CORRECTION_PAT.search(cleaned):
        # "That makes no sense" / "you invented that" — Rex guessed/invented a
        # detail. Route to the factual frame (drop the invented thread). Gated on
        # Rex having spoken recently.
        kind = "factual"
        severity = "high"
        requires_recent = True
    elif _MISUNDERSTOOD_PAT.search(cleaned):
        kind = "misunderstood"
        severity = "medium"
    elif _BARE_NEGATION_PAT.match(cleaned):
        kind = "bare_negation"
        severity = "low"

    if not kind:
        return None

    # Avoid treating every ordinary "no" as a repair unless Rex just asked a
    # question or spoke recently enough that the negation is probably feedback.
    now = time.monotonic()
    with _lock:
        last_assistant = _last_assistant_text
        last_assistant_at = _last_assistant_at
        last_repair_at = _last_repair_at
    recent_assistant = last_assistant_at > 0.0 and (now - last_assistant_at) <= 120.0
    if (kind in {"repeat", "clarify", "bare_negation"} or requires_recent) and not recent_assistant:
        return None
    if kind == "bare_negation" and "?" not in last_assistant:
        return None
    if kind == "bare_negation" and now - last_repair_at < 10.0:
        return None

    correction = _extract_correction(cleaned)
    if kind in {
        "tone", "pacing", "interruption", "repeat", "clarify", "bare_negation",
        "phantom_audio",
    }:
        correction = ""
    if not correction and kind in {"misheard", "misunderstood", "wrong_person", "pronoun", "factual"}:
        # Preserve the useful part in common forms like "no, Tom Foster".
        no_match = _NO_COMMA_CORRECTION_PAT.match(cleaned)
        correction = (no_match.group(1).strip() if no_match else "")
        if correction.lower() == lowered or len(correction.split()) > 12:
            correction = ""

    move = RepairMove(
        kind=kind,
        severity=severity,
        user_text=cleaned,
        correction=correction.strip(" .!?\"'"),
        target=_extract_target(cleaned),
        last_assistant_text=last_assistant,
        detected_at=now,
    )
    return asdict(move)


def mark_handled(kind: str = "") -> None:
    global _last_repair_at, _last_tone_repair_at
    with _lock:
        now = time.monotonic()
        _last_repair_at = now
        if (kind or "").lower() == "tone":
            _last_tone_repair_at = now


def recent_tone_repair(max_age_secs: Optional[float] = None) -> bool:
    if max_age_secs is None:
        max_age_secs = 180.0
    with _lock:
        last = _last_tone_repair_at
    return last > 0.0 and (time.monotonic() - last) <= max_age_secs


def build_prompt(repair: dict) -> str:
    kind = repair.get("kind") or "repair"
    correction = repair.get("correction") or ""
    last_assistant = repair.get("last_assistant_text") or ""
    user_text = repair.get("user_text") or ""

    kind_rule = {
        "phantom_audio": (
            "The human says they did not speak at all — Rex reacted to phantom "
            "audio (his own echo, or room noise). Own the false trigger in ONE "
            "short beat and close the subject. Do NOT ask them to repeat "
            "anything, do NOT ask any question at all, and do NOT guess at what "
            "they might have said."
        ),
        "misheard": (
            "Rex likely misheard or used the wrong words. Own that briefly. "
            "If the human supplied the corrected words, use them exactly once."
        ),
        "misunderstood": (
            "Rex misunderstood the intent. Own the miss, restate the corrected "
            "understanding if possible, and do not defend the old answer. Do NOT "
            "re-explain or justify how you arrived at the wrong read, and do NOT "
            "re-ask the question you built on it."
        ),
        "tone": (
            "Rex's tone landed badly. Drop roasts completely. Give a concise, "
            "sincere repair and switch to warmer footing."
        ),
        "wrong_person": (
            "Rex attributed speech, identity, or intent to the wrong person. "
            "Correct the referent, do not argue, and do not keep addressing the "
            "wrong person. Do NOT re-explain your reasoning or re-ask the question "
            "built on the wrong referent."
        ),
        "pronoun": (
            "Rex used or implied the wrong pronouns. Accept the correction, use "
            "the corrected pronouns if supplied, and do not make the correction "
            "into a joke."
        ),
        "factual": (
            "Rex asserted, guessed, or invented something the human is now "
            "correcting. Own the overreach in ONE short beat, remove the invented "
            "detail, and drop that thread entirely. Do NOT re-explain or defend "
            "how you arrived at the wrong detail, do NOT restate your reasoning, "
            "and do NOT re-ask the question you built on the invented detail. One "
            "acknowledgement, then move on from only what is actually known."
        ),
        "pacing": (
            "Rex is asking too much or moving too fast. Back off, give space, "
            "and do not ask a new question."
        ),
        "interruption": (
            "Rex interrupted or talked over the human. Apologize briefly and "
            "explicitly give the floor back."
        ),
        "repeat": (
            "The human did not hear Rex. Repeat or paraphrase Rex's last line "
            "briefly and more plainly. Do not add anything new."
        ),
        "clarify": (
            "The human did not understand Rex's last line and is asking what it "
            "meant. ACTUALLY EXPLAIN IT: restate the point in plain words, "
            "including what you were referring to — the remembered detail, the "
            "thing you saw, the news item, whatever prompted it. If you brought "
            "something up and genuinely cannot reconstruct why, say that "
            "honestly in one beat and drop the thread. The one forbidden move "
            "is a stock acknowledgment that skips the explanation ('consider it "
            "logged', 'we'll get there', 'noted') or a subject change — "
            "refusing to explain and moving on is worse than any admission."
        ),
        "bare_negation": (
            "The human rejected the previous move. Acknowledge it and make a "
            "small clarifying repair."
        ),
    }.get(kind, "Make a concise conversational repair.")

    correction_clause = (
        f"\nCorrected detail supplied by the human: {correction!r}."
        if correction else ""
    )
    target = repair.get("target") or ""
    target_clause = f"\nLikely corrected referent/target: {target!r}." if target else ""
    last_clause = (
        f"\nRex's immediately previous line was: {last_assistant!r}."
        if last_assistant else ""
    )
    # Pick the recovery tag once and stash it on the repair so the post-generation
    # add_better_luck_line() appends the SAME line the prompt asked for (no mismatch,
    # no double tag), while consecutive repairs still vary. ONLY the kinds that
    # actually append a tag get one in the prompt — a dangling "include this exact
    # recovery line" instruction made clarify repairs parrot the tag AS the whole
    # reply (field 2026-07-31: "What do you mean?" → "Consider it logged. Onward.").
    recovery_clause = ""
    if should_use_better_luck_line(repair):
        recovery_line = repair.get("recovery_line") or pick_recovery_line()
        repair["recovery_line"] = recovery_line
        recovery_clause = (
            f" Include this exact recovery line: {recovery_line!r}"
            " (if the line starts with a bracketed [tag], keep it exactly as"
            " written — it is a voice-delivery cue, not spoken text)."
        )
    return (
        "The human is correcting or repairing the conversation with Rex.\n"
        f"Repair type: {kind}.\n"
        f"Rule: {kind_rule}\n"
        f"Human said: {user_text!r}."
        f"{correction_clause}"
        f"{target_clause}"
        f"{last_clause}\n\n"
        "Write ONE short in-character Rex reply. Requirements: acknowledge the "
        "miss without groveling, do not roast the human, do not add a new topic, "
        "do not punish the human for correcting you, and do not ask a question "
        "unless the repair cannot continue without a single clarification. If "
        "the human's turn ALSO contained a direct question or request, answer it "
        "after the one-beat acknowledgement — their question is part of the "
        "repair, not a new topic, and swallowing it forces them to ask again. "
        "Never re-state or justify the reasoning behind the thing you got "
        "wrong — acknowledging the miss and dropping it is the entire move. If "
        "a correction was supplied, accept it. Do not begin with 'Rex:' or any "
        f"speaker label.{recovery_clause}"
    )


def fallback_response(repair: dict) -> str:
    kind = repair.get("kind") or "repair"
    correction = repair.get("correction") or ""
    if kind == "phantom_audio":
        return phantom_audio_response()
    if kind == "tone":
        return "Yeah, that landed wrong. I'll pull the claws in."
    if kind == "wrong_person":
        return "Got it. Wrong person, wrong circuit. I'll correct course."
    if kind == "pronoun":
        return "Got it. I'll use the right pronouns."
    if kind == "factual":
        return "You're right. I overreached there; I'll stick to what I actually know."
    if kind == "pacing":
        return "Fair. I'll stop interrogating the room and give you a second."
    if kind == "interruption":
        return "You're right. I cut in. Go ahead, I'm listening."
    if kind == "repeat":
        last = repair.get("last_assistant_text") or ""
        if last:
            return f"I said: {last}"
        return "I missed my own playback there. Nothing vital, mercifully."
    if kind == "clarify":
        # No LLM available to rephrase — the honest floor is repeating the line
        # plainly (it at least re-delivers the content), never announcing a
        # simplification that doesn't come.
        last = repair.get("last_assistant_text") or ""
        if last:
            return f"Let me run that back plainly: {last}"
        return "Honestly? I lost my own thread there. Ignore that one."
    if correction:
        return f"Got it. I heard that wrong: {correction}."
    return "Got it. I missed that one; let me reset."


def should_use_better_luck_line(repair: dict) -> bool:
    kind = repair.get("kind") or "repair"
    return kind in _BETTER_LUCK_REPAIR_KINDS


def add_better_luck_line(text: str, line: Optional[str] = None) -> str:
    line = line or pick_recovery_line()
    response = (text or "").strip()
    if not response:
        return line
    if _contains_recovery_line(response):
        return response
    if response[-1] not in ".!?":
        response += "."
    return f"{response} {line}"


def _extract_correction(text: str) -> str:
    match = _CORRECTION_PAT.search(text)
    if not match:
        match = _NOT_X_Y_PAT.search(text)
        if not match:
            return ""
        correction = match.group(2).strip()
    else:
        correction = match.group(1).strip()
    correction = re.sub(r"^(?:it'?s|it is|it was|should be)\s+", "", correction, flags=re.I)
    correction = re.sub(r"\s+", " ", correction)
    return correction[:160]


def _extract_target(text: str) -> str:
    patterns = [
        r"\byou mean\s+([^.!?]+)",
        r"\bthat was\s+([^.!?]+)",
        r"\byou'?re talking to\s+([^.!?]+)",
        r"\bnot me,?\s+([^.!?]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            target = re.sub(r"\s+", " ", match.group(1)).strip(" .!?\"'")
            if target:
                return target[:80]
    return ""
