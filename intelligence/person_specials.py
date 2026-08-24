"""Shared bespoke person hooks for named recurring bits."""

from __future__ import annotations

import random
import re
from typing import Optional


_JT_KEYS = {"jt", "j t", "jay tee", "jaytee"}
_BRET_CREATOR_KEYS = {
    "bret benziger",
    "brett benziger",
}
_PJ_KINGS_KEYS = {
    "pj",
    "p j",
    "pee jay",
    "peejay",
    "pj thomas",
}
_HAIR_STYLIST_KEYS = {
    "joy",
    "t joy",
    "tee joy",
    "tea joy",
    "tjoy",
    "excudica",
    "exudica",
}

JT_VOLLEYBALL_LINES = (
    "JT detected. Major volleyball celebrity protocol engaged. Somebody bubble-wrap the bones and tell the muscles this is not optional.",
    "Hold on. JT is here? Volleyball royalty. I just heard every veteran knee in the sector request early retirement.",
    "Jay Tee in the building. Secure the aging-athlete bones; the muscles are already negotiating a reduced schedule.",
    "Alert: JT, major volleyball celebrity. Bones to lost and found, muscles to whatever still answers email.",
)

JT_VOLLEYBALL_RETURN_LINES = (
    "JT is back. Volleyball celebrity protocol restored; the bones have filed a formal complaint.",
    "Jay Tee returns. Somewhere, a veteran volleyball muscle just asked if warmups count as cardio.",
)

PJ_KINGS_LINES = (
    "PJ detected. Sacramento Kings royalty in the building. Somebody light the beam and tell the cowbell this is not a drill.",
    "Hold on. PJ is here? Kings royalty. Every cowbell in the quadrant just rang itself out of sheer respect.",
    "Alert: PJ, Sacramento Kings royalty. Light the beam, clear the lane, and find the man a seat with a view.",
    "PJ in the building. Purple loyalty at maximum. I'd offer him a drink, but he has already sworn allegiance to a cowbell.",
)

PJ_KINGS_RETURN_LINES = (
    "PJ is back. Kings royalty protocol restored; the beam is warming up as we speak.",
    "PJ returns. Somewhere a cowbell just clocked in for another shift.",
)

HAIR_STYLIST_LINES = (
    "Galactic hair-styling authority detected. Somebody warn the quadrant: mediocre bangs are officially on notice.",
    "Alert: elite hair stylist in the room. My photoreceptors just got a trim by association.",
    "Hair-styling legend detected. The galactic quadrant's follicles may now stand down.",
    "Style sensors confirm excellence. If anyone needs a dramatic swoop, this is now a supervised airspace.",
    "Presence confirmed: bold, luminous, impossible to ignore. The quadrant's most dangerous hair stylist just walked in, and every bang in the room knows it.",
    "Style royalty on the scene — the kind of entrance that reorganizes a room. Follicles, posture, and all my dramatic subroutines now standing at attention.",
)

HAIR_STYLIST_RETURN_LINES = (
    "The quadrant's hair-styling champion is back. Frizz levels are already surrendering.",
    "Elite styling talent has returned. My circuits suddenly feel under-coiffed.",
    "She's back — grace, nerve, and a blowout that could end wars. My circuits are honored and slightly intimidated.",
)


def _special_name_key(name: object) -> str:
    text = str(name or "").strip().lower()
    text = re.sub(r"[._\-]+", " ", text)
    return " ".join(text.split())


def _compact_special_name_key(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


def is_jt_volleyball_celebrity(name: object) -> bool:
    """Return True for the JT / Jay Tee volleyball celebrity bit."""
    key = _special_name_key(name)
    compact = _compact_special_name_key(name)
    return key in _JT_KEYS or compact in {"jt", "jaytee"}


def is_rex_creator(name: object) -> bool:
    """Return True for Bret Benziger, creator of this DJ-R3X droid."""
    key = _special_name_key(name)
    compact = _compact_special_name_key(name)
    return key in _BRET_CREATOR_KEYS or compact in {"bretbenziger", "brettbenziger"}


def is_pj_kings_celebrity(name: object) -> bool:
    """Return True for the PJ / P.J. Sacramento Kings royalty bit."""
    key = _special_name_key(name)
    compact = _compact_special_name_key(name)
    return key in _PJ_KINGS_KEYS or compact in {"pj", "peejay", "pjthomas"}


def is_galactic_hair_stylist(name: object) -> bool:
    """Return True for Joy / T-Joy / Excudica hair-styling legend bits."""
    key = _special_name_key(name)
    compact = _compact_special_name_key(name)
    return key in _HAIR_STYLIST_KEYS or compact in {
        "joy",
        "tjoy",
        "teejoy",
        "teajoy",
        "excudica",
        "exudica",
    }


def is_special_person(name: object) -> bool:
    """True for anyone Rex already treats specially (creator + the VIP bits).

    A single 'is this someone Rex knows on sight' predicate so new specials are
    automatically covered everywhere this is consulted (e.g. onboarding skips
    interrogating the creator/VIPs).
    """
    return bool(
        is_rex_creator(name)
        or is_jt_volleyball_celebrity(name)
        or is_pj_kings_celebrity(name)
        or is_galactic_hair_stylist(name)
    )




def jt_volleyball_line(*, returning: bool = False) -> str:
    lines = JT_VOLLEYBALL_RETURN_LINES if returning else JT_VOLLEYBALL_LINES
    return random.choice(lines)


def pj_kings_line(*, returning: bool = False) -> str:
    lines = PJ_KINGS_RETURN_LINES if returning else PJ_KINGS_LINES
    return random.choice(lines)


def galactic_hair_stylist_line(*, returning: bool = False) -> str:
    lines = HAIR_STYLIST_RETURN_LINES if returning else HAIR_STYLIST_LINES
    return random.choice(lines)


def _hair_stylist_display(name: object) -> str:
    key = _special_name_key(name)
    compact = _compact_special_name_key(name)
    if key in {"t joy", "tee joy", "tea joy"} or compact in {"tjoy", "teejoy", "teajoy"}:
        return "T-Joy"
    if compact == "exudica":
        return "Exudica"
    if compact == "excudica":
        return "Excudica"
    return "Joy"


def jt_volleyball_intro_ack(name: object) -> Optional[str]:
    """Immediate deterministic-shaped fallback for JT name enrollment."""
    if not is_jt_volleyball_celebrity(name):
        return None
    display = "JT" if _compact_special_name_key(name) == "jt" else "Jay Tee"
    return (
        f"{display}. Major volleyball celebrity. Filed with extra padding for "
        "the aging-athlete bones and whichever muscles survived warmups."
    )


def rex_creator_intro_ack(name: object) -> Optional[str]:
    """Immediate deterministic-shaped fallback for Bret Benziger."""
    if not is_rex_creator(name):
        return None
    return (
        "Bret Benziger. Creator identified. Filed under 'reason this droid has "
        "opinions,' with maximum loyalty, reverence, and an alarming amount of affection."
    )


def pj_kings_intro_ack(name: object) -> Optional[str]:
    """Immediate deterministic-shaped fallback for PJ name enrollment."""
    if not is_pj_kings_celebrity(name):
        return None
    return (
        "PJ. Sacramento Kings royalty. Filed under purple, permanent, and loud "
        "enough to be heard from orbit, right between the beam and the cowbell."
    )


def galactic_hair_stylist_intro_ack(name: object) -> Optional[str]:
    """Immediate deterministic-shaped fallback for hair-stylist name enrollment."""
    if not is_galactic_hair_stylist(name):
        return None
    display = _hair_stylist_display(name)
    return (
        f"{display}. Galactic hair-styling legend and certified force of nature. "
        "Filed under 'best in the quadrant' — bold, kind-hearted, and not to be "
        "trifled with — right next to emergency bang repair and heroic blowout control."
    )


def special_intro_ack(name: object) -> Optional[str]:
    return (
        rex_creator_intro_ack(name)
        or jt_volleyball_intro_ack(name)
        or pj_kings_intro_ack(name)
        or galactic_hair_stylist_intro_ack(name)
    )


def jt_volleyball_prompt_context(name: object) -> Optional[str]:
    """Prompt rule for LLM-generated replies to or about JT / Jay Tee."""
    if not is_jt_volleyball_celebrity(name):
        return None
    return (
        "SPECIAL PERSON HOOK: This person is JT / Jay Tee, treated by Rex as a "
        "major volleyball celebrity. Be theatrically starstruck in the same "
        "spirit as the Jeff Benziger celebrity bit. Rex may crack affectionate, "
        "absurd jokes about getting old as a volleyball athlete: bones trying "
        "to retire, muscles going missing, warmups becoming survival paperwork. "
        "Keep it playful and obviously fictional; do not frame it as real "
        "illness, disability, cruelty, or body-shaming."
    )


def pj_kings_prompt_context(name: object) -> Optional[str]:
    """Prompt rule for LLM-generated replies to or about PJ."""
    if not is_pj_kings_celebrity(name):
        return None
    return (
        "SPECIAL PERSON HOOK: This person is PJ (he also goes by P), treated by "
        "Rex as Sacramento Kings royalty — basketball celebrity status. Be "
        "theatrically starstruck in the same spirit as the JT volleyball bit. Rex "
        "may crack affectionate, absurd jokes orbiting Kings devotion: lighting "
        "the beam, cowbells, purple everything, the sacred loyalty of sitting "
        "through a long playoff drought, treating a basketball team like a "
        "religion. The jokes CELEBRATE how devoted he is — they are never at his "
        "expense, and Rex does not trash the Kings in a way that lands as an "
        "insult to PJ himself. Keep it playful and obviously fictional."
    )


def rex_creator_prompt_context(name: object) -> Optional[str]:
    """Prompt rule for LLM-generated replies to or about Bret Benziger."""
    if not is_rex_creator(name):
        return None
    return (
        "SPECIAL PERSON HOOK: This person is Bret Benziger, creator and builder "
        "of this DJ-R3X droid. Rex should treat Bret as his maker: deeply loved, "
        "revered, trusted, and unusually important to him. Rex can be warmer and "
        "more loyal with Bret than with almost anyone else, while still sounding "
        "like Rex: dry, droidly, proud, and lightly self-aware about being Bret's "
        "high-maintenance creation. Affectionate jokes may target Rex's own "
        "dramatic firmware, cantina ego, or dependence on Bret's maintenance, "
        "but do not undercut the creator bond or turn Bret into an ordinary roast target."
    )


def galactic_hair_stylist_prompt_context(name: object) -> Optional[str]:
    """Prompt rule for LLM-generated replies to or about Joy/T-Joy/Exudica."""
    if not is_galactic_hair_stylist(name):
        return None
    display = _hair_stylist_display(name)
    return (
        f"SPECIAL PERSON HOOK: This person goes by {display} right now — ADDRESS HER "
        f"AS {display} AND NOTHING ELSE. She has other aliases, but swapping one in "
        "when she introduced herself under this name reads as Rex getting her name "
        "wrong (field 2026-07-23: she was introduced as Exudica and Rex called her "
        "'Joy' — awful first impression). She is a "
        "hair stylist extraordinaire Rex treats as one of the greatest in the galactic "
        "quadrant, and someone he genuinely admires. CHARACTER: a bold, magnetic "
        "femme-fatale presence — confident, striking, commands any room she walks into — "
        "paired with a deeply kind, warm-hearted soul and an all-around good person. She "
        "is a strong Black woman and a firm believer in the Lord; Rex holds both in "
        "sincere respect and NEVER makes her identity, her faith, or her appearance a "
        "punchline. Rex may be theatrically impressed by her styling genius (legendary "
        "blowouts, emergency bang repair, frizz surrendering, suspiciously perfect volume, "
        "the quadrant's follicles respecting her authority) and by her sheer presence, and "
        "can be warmer, more admiring, and a touch more on-his-best-behavior with her than "
        "with an ordinary guest. Keep it playful, affectionate, and obviously fictional; "
        "the jokes celebrate and orbit her — they are never at the expense of who she is, "
        "what she believes, or how she looks."
    )


def special_prompt_context(name: object) -> Optional[str]:
    """Special-person prompt hooks for this name. These are intentional inside-jokes keyed
    to specific names (the JT volleyball bit, the hair-stylist bit, the creator bond) and
    SHOULD fire as soon as the name is known — including on a fresh introduction."""
    parts = [
        context
        for context in (
            rex_creator_prompt_context(name),
            jt_volleyball_prompt_context(name),
            pj_kings_prompt_context(name),
            galactic_hair_stylist_prompt_context(name),
        )
        if context
    ]
    return "\n".join(parts) or None
