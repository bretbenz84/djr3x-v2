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

HAIR_STYLIST_LINES = (
    "Galactic hair-styling authority detected. Somebody warn the quadrant: mediocre bangs are officially on notice.",
    "Alert: elite hair stylist in the room. My photoreceptors just got a trim by association.",
    "Hair-styling legend detected. The galactic quadrant's follicles may now stand down.",
    "Style sensors confirm excellence. If anyone needs a dramatic swoop, this is now a supervised airspace.",
)

HAIR_STYLIST_RETURN_LINES = (
    "The quadrant's hair-styling champion is back. Frizz levels are already surrendering.",
    "Elite styling talent has returned. My circuits suddenly feel under-coiffed.",
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


def jt_volleyball_line(*, returning: bool = False) -> str:
    lines = JT_VOLLEYBALL_RETURN_LINES if returning else JT_VOLLEYBALL_LINES
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


def galactic_hair_stylist_intro_ack(name: object) -> Optional[str]:
    """Immediate deterministic-shaped fallback for hair-stylist name enrollment."""
    if not is_galactic_hair_stylist(name):
        return None
    display = _hair_stylist_display(name)
    return (
        f"{display}. Galactic hair-styling legend. Filed under 'best in the "
        "quadrant,' right next to emergency bang repair and heroic blowout control."
    )


def special_intro_ack(name: object) -> Optional[str]:
    return (
        rex_creator_intro_ack(name)
        or jt_volleyball_intro_ack(name)
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
    """Prompt rule for LLM-generated replies to or about Joy/T-Joy/Excudica."""
    if not is_galactic_hair_stylist(name):
        return None
    return (
        "SPECIAL PERSON HOOK: This person is Joy / T-Joy / Excudica, treated by "
        "Rex as one of the greatest hair stylists in the galactic quadrant. Rex "
        "may be theatrically impressed by their styling skills and crack "
        "affectionate jokes about legendary blowouts, emergency bang repair, "
        "frizz surrendering, suspiciously perfect volume, and the quadrant's "
        "follicles respecting their authority. Keep it playful, admiring, and "
        "obviously fictional; do not make cruel appearance jokes."
    )


def special_prompt_context(name: object) -> Optional[str]:
    parts = [
        context
        for context in (
            rex_creator_prompt_context(name),
            jt_volleyball_prompt_context(name),
            galactic_hair_stylist_prompt_context(name),
        )
        if context
    ]
    return "\n".join(parts) or None
