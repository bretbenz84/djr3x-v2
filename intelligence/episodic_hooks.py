"""
episodic_hooks.py — thin capture hooks that write Rex's first-person episodic memory
(rex.db) from the consciousness loop.

Extracted verbatim from consciousness.py to keep that module smaller. Every function
here is a LEAF: it takes plain arguments, lazily calls into `memory.episodes` (which is
itself gated + failure-safe), and swallows errors — it never calls back into
consciousness, so there is no import cycle and no behavior coupling. The actual write
gating (kill switch + test-runner suppression) lives in memory.episodes; these wrappers
just add convenience, a scene-change dedupe, and the once-per-run startup image latch.
"""

from __future__ import annotations

import logging
import threading

import config

_log = logging.getLogger(__name__)


def person_seen(person_id, name) -> None:
    if not isinstance(person_id, int):
        return
    try:
        from memory import episodes
        episodes.record_person_seen(person_id, name)
    except Exception as exc:
        _log.debug("episodic person_seen failed: %s", exc)


def made_laugh(person_id, name, kind: str = "smile") -> None:
    try:
        from memory import episodes
        episodes.record_made_laugh(person_id if isinstance(person_id, int) else None, name, kind=kind)
    except Exception as exc:
        _log.debug("episodic made_laugh failed: %s", exc)


def animal(species, position=None) -> None:
    try:
        from memory import episodes
        episodes.record_animal(species, position=position)
    except Exception as exc:
        _log.debug("episodic animal failed: %s", exc)


_last_scene_episode_sig = None


def _known_visible_names(snapshot: Optional[dict] = None) -> list[str]:
    """Names of recognized people currently visible, so a scene memory records WHO was
    there ('Bret at his desk') instead of 'a man at a desk'. Reads from the given
    snapshot's people (or world_state), resolving each person_db_id to a name."""
    try:
        if snapshot is not None:
            people = (snapshot or {}).get("people") or []
        else:
            from world_state import world_state
            people = world_state.get("people") or []
    except Exception:
        return []
    names: list[str] = []
    seen: set[int] = set()
    for person in people:
        if not isinstance(person, dict):
            continue
        if person.get("face_visible") is False or person.get("face_missing"):
            continue
        pid = person.get("person_db_id")
        try:
            pid = int(pid) if pid is not None else None
        except (TypeError, ValueError):
            pid = None
        if pid is None or pid in seen:
            continue
        seen.add(pid)
        try:
            from memory import people as people_mem
            row = people_mem.get_person(pid)
        except Exception:
            row = None
        name = (row or {}).get("name")
        if name and name not in names:
            names.append(name)
    return names


def _join_names(names: list[str]) -> str:
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def scene_changed(snapshot: dict) -> None:
    """Log a 'scene' episode when the observed environment MATERIALLY changes (deduped
    on a signature, so it captures 'the room got cluttered/crowded' transitions, not
    every tick). Cheap dict reads; the write itself is gated in memory.episodes."""
    global _last_scene_episode_sig
    try:
        env = (snapshot or {}).get("environment") or {}
        scene_type = str(env.get("scene_type") or "").strip()
        lighting = str(env.get("lighting") or "").strip()
        crowd = str(env.get("crowd_density") or "").strip()
        desc = str(env.get("description") or "").strip()
        if not (scene_type or desc):
            return
        sig = (scene_type, lighting, crowd, desc[:80])
        if sig == _last_scene_episode_sig:
            return
        _last_scene_episode_sig = sig
        if desc:
            summary = f"I looked around the room: {desc}"
        else:
            parts = [p for p in (
                scene_type or "",
                f"{lighting} light" if lighting else "",
                f"{crowd} crowd" if crowd else "",
            ) if p]
            if not parts:
                return
            summary = "The room was " + ", ".join(parts) + "."
        # Name who was there (the room description is generic and doesn't identify
        # recognized people), unless their name is already in the text.
        names = [n for n in _known_visible_names(snapshot)
                 if n.lower() not in summary.lower()]
        if names:
            who = _join_names(names)
            verb = "was" if len(names) == 1 else "were"
            summary = summary.rstrip(" .") + f". {who} {verb} there."
        from memory import episodes
        episodes.record_scene(
            summary,
            detail={"scene_type": scene_type, "lighting": lighting,
                    "crowd_density": crowd, "description": desc},
        )
    except Exception as exc:
        _log.debug("episodic scene failed: %s", exc)


_startup_image_captured = False


def startup_image(frame) -> None:
    """Once per run: ONE cheap GPT image caption of Rex's first look at the room, logged
    to rex.db as a 'scene' episode ("When I powered up, I saw: …"). The GPT call runs OFF
    the tick (background thread) so it never delays consciousness; gated like all episodic
    writes (no GPT call under the test runner / when disabled)."""
    global _startup_image_captured
    if _startup_image_captured or frame is None:
        return
    if not bool(getattr(config, "EPISODIC_STARTUP_IMAGE_ENABLED", True)):
        _startup_image_captured = True
        return
    try:
        from memory import episodes
        if episodes._suppressed():   # disabled / under the test runner → no GPT call
            _startup_image_captured = True
            return
    except Exception:
        return
    _startup_image_captured = True   # one-shot: latch BEFORE spawning so we fire once

    def _work(frame=frame) -> None:
        try:
            from vision import scene as _scene
            # Resolve names here (not at call time): startup_image latches on the first
            # frame, BEFORE that tick's person-recognition runs, so reading world_state
            # now — just before the captioning GPT call — picks up who was identified.
            known = _known_visible_names()
            caption = _scene.quick_caption(frame, known_people=known)
            if caption:
                from memory import episodes
                episodes.record_episode(
                    "scene", f"When I powered up, I saw: {caption}",
                    detail={"source": "startup_image_caption"}, salience=0.55,
                )
                _log.info("episodic: startup image caption logged")
        except Exception as exc:
            _log.debug("startup image caption failed: %s", exc)

    threading.Thread(target=_work, daemon=True, name="startup-image-caption").start()


def visit_departure(person_id, name, arrival_secs, departure_secs) -> None:
    if arrival_secs is None:
        return  # never recorded an arrival this session → no duration
    try:
        from memory import episodes
        episodes.record_visit_departure(
            person_id if isinstance(person_id, int) else None, name,
            max(0.0, float(departure_secs) - float(arrival_secs)),
            detail={"arrival_secs": arrival_secs, "departure_secs": departure_secs},
        )
    except Exception as exc:
        _log.debug("episodic visit_departure failed: %s", exc)


def celebrity(person_id, name, celebrity_name, returning: bool = False) -> None:
    try:
        from memory import episodes
        episodes.record_celebrity(
            person_id if isinstance(person_id, int) else None, name, celebrity_name,
            returning=bool(returning),
        )
    except Exception as exc:
        _log.debug("episodic celebrity failed: %s", exc)


def checkin(person_id, name, summary, detail=None) -> None:
    try:
        from memory import episodes
        episodes.record_checkin(
            person_id if isinstance(person_id, int) else None, name, summary, detail=detail,
        )
    except Exception as exc:
        _log.debug("episodic checkin failed: %s", exc)


def celebration(person_id, name, summary, detail=None) -> None:
    try:
        from memory import episodes
        episodes.record_greeting_event(
            "celebration", summary,
            person_id=person_id if isinstance(person_id, int) else None,
            person_name=name, detail=detail,
        )
    except Exception as exc:
        _log.debug("episodic celebration failed: %s", exc)


def greeting_from_label(label, person_id, name) -> None:
    """Log a memorable first-sight greeting (birthday / celebration / milestone /
    reunion / check-in), keyed on the dispatched tier's `label`. Called only inside the
    `if queued:` success block, so it reflects a greeting Rex ACTUALLY spoke — never a
    candidate dropped under ENFORCE. The label is set by the tier that won the priority
    chain, so this captures exactly the fired tier (ordinary greetings log nothing). We
    key on the label (vs the branch-local tier vars) to avoid touching the greeting
    chain itself; a label rename just silently stops the capture (graceful)."""
    if not isinstance(person_id, int):
        return
    who = name or "them"
    lab = str(label or "")
    kind = summary = None
    detail: dict = {}
    if lab.startswith("startup birthday (T-0)"):
        kind, summary, detail = "birthday_wish", f"I wished {who} a happy birthday.", {"t_minus": 0}
    elif lab.startswith("startup birthday (T-"):
        kind, summary = "birthday_wish", f"I reminded {who} their birthday is coming up soon."
    elif lab.startswith("first-sight celebration"):
        kind, summary = "celebration", f"I celebrated some good news with {who}."
    elif lab.startswith("startup milestone (#"):
        import re as _re
        m = _re.search(r"#(\d+)", lab)
        n = m.group(1) if m else "?"
        kind, summary, detail = "milestone", f"I marked {who}'s visit #{n}.", {"visit_number": n}
    elif lab.startswith("startup long-absence"):
        kind, summary = "reunion", f"I welcomed {who} back after a long while away."
    elif lab.startswith("first-sight emotional check-in"):
        kind, summary = "emotional_checkin", f"I checked in on {who} when I saw them."
    else:
        return  # ordinary greeting → not a memorable event
    try:
        from memory import episodes
        if kind == "emotional_checkin":
            episodes.record_checkin(person_id, name, summary, detail=detail)
        else:
            episodes.record_greeting_event(kind, summary, person_id=person_id, person_name=name, detail=detail)
    except Exception as exc:
        _log.debug("episodic greeting event failed: %s", exc)
