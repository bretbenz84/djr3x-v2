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
import re
import threading

import config

_log = logging.getLogger(__name__)

# Low-signal words ignored when comparing two scene captions for "is this a new scene?"
_CAPTION_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "with", "this", "that",
    "room", "scene", "space", "area", "shows", "show", "appears", "there", "image",
    "view", "looks", "seen", "visible", "some", "front", "while", "into", "from",
})


# (session_id, person_id) pairs already logged as "I saw X", so a person seen across a
# whole session yields ONE record, not one per detection tick ("I saw Bret" ×39).
_person_seen_this_session: set = set()


def person_seen(person_id, name) -> None:
    if not isinstance(person_id, int):
        return
    try:
        from memory import episodes
        key = (episodes._session(), person_id)
        if key in _person_seen_this_session:
            return
        _person_seen_this_session.add(key)
        episodes.record_person_seen(person_id, name)
    except Exception as exc:
        _log.debug("episodic person_seen failed: %s", exc)


def made_laugh(person_id, name, kind: str = "smile", topic=None) -> None:
    try:
        from memory import episodes
        episodes.record_made_laugh(
            person_id if isinstance(person_id, int) else None, name,
            kind=kind, topic=topic,
        )
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
    there ('Bret at his desk') instead of 'a man at a desk'. Thin wrapper over the
    canonical vision-layer resolver (`vision.face.visible_known_names`)."""
    try:
        from vision import face
        return face.visible_known_names(snapshot)
    except Exception:
        return []


def _visible_known_people(snapshot=None) -> list:
    """(person_id, name) pairs for recognized people present — for ATTRIBUTING a scene
    to a person (face match), not just naming them in prose."""
    try:
        from vision import face
        return list(face.visible_known_people(snapshot))
    except Exception:
        return []


def _sole_known_person(snapshot=None):
    """(id, name) of the single recognized person present, or (None, None) when zero or
    more than one — a scene is attributed only when there is no ambiguity about who."""
    people = _visible_known_people(snapshot)
    if len(people) == 1:
        return people[0]
    return (None, None)


def _caption_tokens(text: str) -> set:
    return {
        w for w in re.findall(r"[a-z]+", (text or "").lower())
        if len(w) > 3 and w not in _CAPTION_STOPWORDS
    }


def _caption_materially_differs(prev: str, current: str) -> bool:
    """True when two scene captions describe a meaningfully different scene (token
    overlap below SCENE_CAPTURE_SIMILARITY_THRESHOLD). No prior caption → treat as
    notable (the first look is always worth keeping)."""
    if not (prev or "").strip():
        return True
    a, b = _caption_tokens(prev), _caption_tokens(current)
    if not a or not b:
        return True
    overlap = len(a & b) / float(len(a | b))
    return overlap < float(getattr(config, "SCENE_CAPTURE_SIMILARITY_THRESHOLD", 0.55))


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
        known = _visible_known_people(snapshot)
        names = [n for _pid, n in known if n.lower() not in summary.lower()]
        if names:
            who = _join_names(names)
            verb = "was" if len(names) == 1 else "were"
            summary = summary.rstrip(" .") + f". {who} {verb} there."
        # Attribute to the single recognized person present (face match) so the scene
        # joins Rex's history WITH them; ambiguous/empty stays unattributed.
        pid, pname = (known[0] if len(known) == 1 else (None, None))
        from memory import episodes
        episodes.record_scene(
            summary,
            detail={"scene_type": scene_type, "lighting": lighting,
                    "crowd_density": crowd, "description": desc},
            person_id=pid, person_name=pname,
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
            known_people = _visible_known_people()
            known = [n for _pid, n in known_people]
            caption = _scene.quick_caption(frame, known_people=known)
            if caption:
                from memory import episodes
                # Before recording THIS run's snapshot, compare it to the PREVIOUS run's
                # so Rex can remark on a change of scenery (a different room, outdoors, a
                # new place). Queued for consciousness to speak; no-op if unchanged.
                _maybe_queue_scenery_remark(caption)
                # Keep only a notable or person-present startup scene — otherwise it's
                # generic boilerplate ("a tidy room with white walls") that repeats every
                # boot and drags down recall quality.
                require = bool(getattr(config, "SCENE_CAPTURE_REQUIRE_PERSON_OR_CHANGE", True))
                notable = _caption_materially_differs(_previous_startup_caption(), caption)
                if require and not known_people and not notable:
                    _log.info("episodic: startup scene skipped (no person present, unchanged)")
                    return
                pid, pname = (known_people[0] if len(known_people) == 1 else (None, None))
                episodes.record_scene(
                    f"When I powered up, I saw: {caption}",
                    detail={"source": "startup_image_caption", "caption": caption},
                    person_id=pid, person_name=pname,
                )
                _log.info(
                    "episodic: startup image caption logged (person=%s notable=%s)",
                    pname or "none", notable,
                )
        except Exception as exc:
            _log.debug("startup image caption failed: %s", exc)

    threading.Thread(target=_work, daemon=True, name="startup-image-caption").start()


# Set by the startup-image worker when this run's scene differs from the previous run's;
# consciousness pops it via take_scenery_remark() and speaks it once.
_pending_scenery_remark = None


def _previous_startup_caption() -> str:
    """The caption Rex stored on his PREVIOUS run's power-up (or '' if none)."""
    try:
        import json as _json
        from memory import episodes, rex_db
        row = rex_db.fetchone(
            "SELECT summary, detail FROM rex_episodes WHERE kind = 'scene' "
            "AND detail LIKE '%startup_image_caption%' AND session_id != ? "
            "ORDER BY created_at DESC, id DESC LIMIT 1",
            (episodes._session(),),
        )
        if not row:
            return ""
        detail = row["detail"]
        if detail:
            try:
                cap = (_json.loads(detail) or {}).get("caption")
                if cap:
                    return str(cap).strip()
            except Exception:
                pass
        # Fallback for episodes written before the caption was stored in detail:
        # strip the "When I powered up, I saw: " prefix off the summary.
        summary = str(row["summary"] or "")
        marker = "I saw: "
        return summary.split(marker, 1)[1].strip() if marker in summary else ""
    except Exception as exc:
        _log.debug("previous startup caption lookup failed: %s", exc)
        return ""


def _maybe_queue_scenery_remark(current_caption: str) -> None:
    global _pending_scenery_remark
    if not bool(getattr(config, "SCENERY_CHANGE_REMARK_ENABLED", True)):
        return
    prev = _previous_startup_caption()
    if not prev:
        return  # first run (or no prior snapshot) — nothing to compare against
    try:
        from intelligence import llm
        remark = llm.scenery_change_remark(prev, current_caption)
    except Exception as exc:
        _log.debug("scenery_change_remark failed: %s", exc)
        return
    if remark:
        _pending_scenery_remark = remark
        _log.info("episodic: scenery-change remark queued: %r", remark)


def take_scenery_remark():
    """Pop the queued scenery-change remark (consciousness speaks it once), or None."""
    global _pending_scenery_remark
    remark = _pending_scenery_remark
    _pending_scenery_remark = None
    return remark


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


def exploration(summary, person_name=None, person_id=None, detail=None) -> None:
    """Rex went on a self-directed room wander and fixated on something.

    Routed to record_scene (kind "scene", already surfaced by episodic recall). The
    write gate + test-runner suppression live inside memory.episodes, so this leaf
    stays a thin wrapper like the others.
    """
    try:
        from memory import episodes
        episodes.record_scene(
            summary, detail=detail,
            person_id=person_id if isinstance(person_id, int) else None,
            person_name=person_name,
        )
    except Exception as exc:
        _log.debug("episodic exploration failed: %s", exc)


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
