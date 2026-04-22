"""
awareness/social.py — Social context analysis for DJ-R3X.
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from world_state import world_state

_log = logging.getLogger(__name__)

_DISENGAGED_ENGAGEMENT = frozenset({"low", "none", "disengaged"})
_CHILD_AGE_VALUES = frozenset({"child", "teen"})

# Proxemics zones that indicate active engagement vs passive/distant presence
_ENGAGED_ZONES = frozenset({"intimate", "social"})


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def check_disengagement(people_list: list[dict]) -> list:
    """
    Return a list of person IDs that show disengagement signals:
    facing away, low engagement, or retreating to public distance zone.
    """
    disengaged = []
    for person in people_list:
        pose = (person.get("pose") or "").lower()
        engagement = (person.get("engagement") or "high").lower()
        dist = (person.get("distance_zone") or "social").lower()

        if (
            "away" in pose
            or engagement in _DISENGAGED_ENGAGEMENT
            or dist == "public"
        ):
            pid = person.get("id") or person.get("face_id")
            if pid is not None:
                disengaged.append(pid)

    return disengaged


def analyze_crowd(people_list: list[dict]) -> dict:
    """
    Derive interaction context from the current people list and update world_state.crowd.

    interaction_mode:
      one_on_one  — exactly 1 person present
      small_group — 2–3 people present
      crowd       — 4+ people, at least one actively engaged
      performance — 4+ people but all disengaged/distant, or 0 people

    dominant_speaker is read from world_state.crowd (set by the speaker ID system).

    Returns dict with: interaction_mode, dominant_speaker, disengaged_people.
    """
    count = len(people_list)
    disengaged = check_disengagement(people_list)
    engaged_count = count - len(disengaged)

    if count == 0:
        mode = "performance"
    elif count == 1:
        mode = "one_on_one"
    elif count <= 3:
        mode = "small_group"
    elif engaged_count > 0:
        mode = "crowd"
    else:
        mode = "performance"

    crowd = world_state.get("crowd")
    dominant_speaker = crowd.get("dominant_speaker")

    crowd["interaction_mode"] = mode
    crowd["disengaged_people"] = disengaged
    crowd["last_updated"] = datetime.now(timezone.utc).isoformat()
    world_state.update("crowd", crowd)

    return {
        "interaction_mode": mode,
        "dominant_speaker": dominant_speaker,
        "disengaged_people": disengaged,
    }


def detect_child_present() -> bool:
    """
    Return True if any person in world_state.people has age_category or age_estimate
    of 'child' or 'teen'. Rex switches to family-friendly mode when this is True.
    """
    for person in world_state.get("people"):
        age = (
            person.get("age_category")
            or person.get("age_estimate")
            or "adult"
        ).lower()
        if age in _CHILD_AGE_VALUES:
            return True
    return False
