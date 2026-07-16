"""
speech_engine.py — Rex's proactive-speech engine: the primitives every proactive line flows
through. Pulled out of consciousness.py to group "the mechanism of speaking" in one place.

These are the gating + delivery + governor-arbitration functions (can_proactive_speak,
generate_and_speak[_presence], speak_async, the purpose-claim trio, observe/mark/finish
governor candidate, etc.). They operate on consciousness's proactive-speech STATE (the
pending event, the locks, the last-spoke timestamps) and a handful of its helpers, reached
lazily via the `_c` consciousness proxy (resolves at call time — no import cycle: consciousness
imports this module at top and re-exports each function as a shim, e.g. `_generate_and_speak =
speech_engine.generate_and_speak`, so existing call sites keep working). `note_rex_utterance`
and the shared state deliberately STAY in consciousness (cross-cutting bookkeeping touched by
interaction.py and many steps).

NOTE ON INTRA-ENGINE CALLS: engine functions call each other via the consciousness shim
(`can_proactive_speak`, etc.) rather than the local name, so the extraction is fully
transparent — patching `consciousness._<fn>` overrides the call exactly as it did when these
functions lived in consciousness.
"""

from __future__ import annotations

import inspect
import logging
import random
import threading
import time
from typing import Callable, Optional

import config
import state as state_module
from state import State
from awareness.situation import assessor as _situation_assessor, SituationProfile
from utils import conv_log

_log = logging.getLogger(__name__)


class _ConsciousnessProxy:
    """Lazy handle to consciousness, so this module imports cleanly in either load order
    (consciousness re-exports these functions as shims, so a top-level import here would be
    circular). Every `_c.<name>` access resolves at call time — consciousness is always
    fully loaded by the time an engine function actually runs."""
    __slots__ = ()

    def __getattr__(self, name):
        from intelligence import consciousness
        return getattr(consciousness, name)


_c = _ConsciousnessProxy()


def can_proactive_speak(*, salient: bool = False, reactive: bool = False) -> bool:
    # salient=True marks a high-value, time-sensitive event (e.g. a new animal
    # arriving in frame) that may interrupt a normal ACTIVE conversation and ignore
    # the proactive pacing cooldown — otherwise a priority-85 reaction is starved by
    # the same gate that low-priority idle chatter (which submits via a different
    # path) bypasses. It STILL respects DJ playback, active games, awaiting-a-reply,
    # an in-flight proactive line, and not talking over live speech, and the governor
    # still arbitrates its priority.
    #
    # reactive=True marks a DIRECT reaction to the person right now (e.g. waving back
    # at a wave). Unlike salient, it also breaks through the "awaiting a reply to Rex's
    # own question" and active-conversation gates and the pacing cooldown, so a wave is
    # acknowledged promptly even when Rex just asked something — but it STILL must not
    # talk over live speech, music, a game, an open tell-about/onboarding flow, or the
    # give-space window after a heavy moment.
    if not _c._can_speak():
        return False

    # An open "tell me about someone" briefing owns the floor outright —
    # nothing proactive (salient or not) may barge in until the flow exits.
    # Live-logged failure: idle banter won a governor cycle mid-briefing and
    # derailed the collection; the flow's re-anchor question is the backstop,
    # not the norm. Checked before the salient bypasses on purpose.
    try:
        from intelligence import interaction as _interaction
        if _interaction.tell_about_flow_active():
            return False
        # A first-meeting onboarding burst likewise owns the floor — no idle
        # banter / smile reaction may barge in until the burst exits.
        if _interaction.onboarding_flow_active():
            return False
    except Exception:
        pass

    # A room-exploration session owns the floor outright while Rex is wandering,
    # surveying, and narrating what he finds — no other proactive behavior may barge
    # in until the walk ends. The mode speaks its own lines by enqueuing directly, so
    # denying here does NOT gag the exploration itself.
    try:
        from intelligence import exploration as _exploration
        if _exploration.active():
            return False
    except Exception:
        pass

    # GIVE SPACE after a heavy/grief disclosure: suppress NON-salient proactive speech
    # (idle banter, holiday/plans, environment snark, small talk) for the sober window,
    # so Rex doesn't proactively re-open or probe a heavy topic the user stepped back
    # from. Genuinely reactive salient events (e.g. an animal arriving) still pass, and
    # Rex always still RESPONDS when spoken to — this only gates volunteering.
    if not salient:
        try:
            from intelligence import callback_engine as _cb_engine
            if _cb_engine.recently_heavy():
                return False
        except Exception:
            pass

    try:
        from features import dj as dj_mod
        if (
            bool(getattr(config, "DJ_SUPPRESS_CONVERSATION_DURING_PLAYBACK", True))
            and dj_mod.is_playing()
        ):
            return False
    except Exception:
        pass

    try:
        from features import games as games_mod
        if hasattr(games_mod, "suppresses_conversation_interruptions"):
            if games_mod.suppresses_conversation_interruptions():
                return False
        elif games_mod.is_active():
            return False
    except Exception:
        pass

    current_state = state_module.get_state()
    if (
        not salient
        and not reactive
        and current_state == State.ACTIVE
        and not getattr(config, "CONSCIOUSNESS_ALLOW_PROACTIVE_IN_ACTIVE", False)
    ):
        return False

    if not reactive and _c.is_waiting_for_response():
        return False
    if _c._proactive_speech_pending.is_set():
        return False
    try:
        if _situation_assessor.is_interaction_busy():
            return False
    except Exception:
        pass

    if not salient and not reactive:
        with _c._turn_lock:
            last_spoken = _c._last_proactive_speech_at
        min_gap = max(0.0, float(getattr(config, "CONSCIOUSNESS_PROACTIVE_MIN_GAP_SECS", 0.0)))
        if min_gap and (time.monotonic() - last_spoken) < min_gap:
            return False

    try:
        from audio import speech_queue, output_gate
        if speech_queue.is_speaking() or output_gate.is_busy():
            return False
    except Exception:
        return False
    return True


def speak_async(
    text: str,
    emotion: str = "neutral",
    *,
    wait_secs: Optional[float] = None,
    purpose: Optional[str] = None,
    label: str = "",
    governed: bool = True,
    on_done: Optional[Callable[[], None]] = None,
    on_spoke: Optional[Callable[[], None]] = None,
    force_salient: bool = False,
    reactive: bool = False,
) -> bool:
    # `on_spoke` fires once the line is committed to the speech queue (only the
    # ENFORCE winner reaches here) — the place for "I fired this" bookkeeping that
    # must NOT happen for a losing candidate. See generate_and_speak's note.
    # force_salient=True lets a high-value event (e.g. an animal arrival) speak
    # during an ACTIVE conversation and skip the pacing cooldown (see
    # can_proactive_speak); it still yields to live user speech below.
    # reactive=True is a direct reaction to the person (e.g. wave-back) that also
    # breaks through the awaiting-a-reply / active-conversation gates (see
    # can_proactive_speak); it still yields to live speech / music / games.
    def _do_speak(candidate_id: Optional[str]) -> bool:
        try:
            if not _c._can_proactive_speak(salient=force_salient, reactive=reactive):
                _c._mark_governor_candidate(candidate_id, "dropped", "can_proactive_speak_false")
                return False
            if not text or not text.strip():
                _c._mark_governor_candidate(candidate_id, "dropped", "empty_text")
                return False
            # Claim the proactive floor NOW — BEFORE the (possibly seconds-long) TTS
            # pre-cache below. can_proactive_speak() consults _proactive_speech_pending,
            # so without claiming here a SECOND proactive candidate sails through its own
            # check during our ensure_cached() window and both speak ~seconds apart
            # (live-logged 2026-06-20: idle_banter + visual_curiosity stacked). Cleared
            # on every early-out below and after playback completes (_on_done).
            _c._proactive_speech_pending.set()
            # Yield the floor if the user has already started talking. This line was
            # decided + generated before now; pre-cache its audio so the mic re-check
            # lands right before playback (not ~1s before it, the window in which Rex
            # used to start talking over a reply that began during TTS generation),
            # then bail if the user beat us to it — the interaction turn loop will
            # pick them up from the un-attenuated rolling buffer.
            if bool(getattr(config, "PROACTIVE_SPEECH_YIELD_ENABLED", True)):
                try:
                    from audio import tts
                    tts.ensure_cached(text, emotion=emotion)
                except Exception as exc:
                    _log.debug("proactive pre-cache failed: %s", exc)
                try:
                    from audio import barge_guard
                    if barge_guard.user_speaking_now():
                        _c._mark_governor_candidate(candidate_id, "dropped", "user_speaking")
                        _c._proactive_speech_pending.clear()
                        _log.info(
                            "[consciousness] proactive line yielded — user already speaking: %r",
                            text,
                        )
                        return False
                except Exception as exc:
                    _log.debug("proactive yield check failed: %s", exc)
            from audio import speech_queue
            # log_text=False: we log_rex this line at enqueue (below). Without this
            # tts.speak would ALSO log it at playback, double-printing every proactive
            # line in the conversation log (the cosmetic "duplicate" in the field log).
            done = speech_queue.enqueue(text, emotion, priority=0, log_text=False)
            _c._mark_governor_candidate(candidate_id, "accepted", "current_behavior_enqueued_speech")
            should_open_wait_on_done = (
                on_done is None and (wait_secs is not None or _c._utterance_expects_reply(text))
            )

            def _on_done() -> None:
                done.wait()
                try:
                    if on_done is not None:
                        on_done()
                    elif should_open_wait_on_done:
                        _c.begin_response_wait(wait_secs)
                finally:
                    _c._proactive_speech_pending.clear()

            threading.Thread(target=_on_done, daemon=True, name="speech-pending-clear").start()
            try:
                conv_log.log_rex(text)
            except Exception as exc:
                _log.debug("conversation log write failed for proactive speech: %s", exc)
            _c.note_rex_utterance(
                text,
                wait_secs=wait_secs,
                open_response_wait=False,
                source=purpose,
            )
            if on_spoke is not None:
                try:
                    on_spoke()
                except Exception as exc:
                    _log.debug("on_spoke callback failed: %s", exc)
            return True
        except Exception as exc:
            _c._mark_governor_candidate(candidate_id, "dropped", "speak_async_error")
            _c._proactive_speech_pending.clear()
            _log.debug("speak_async error: %s", exc)
            return False

    if governed and _c._governor_enforcing():
        # ENFORCE: submit a candidate carrying the deferred enqueue; only the tick's
        # winner speaks. (governed=False callers — e.g. generate_and_speak's own
        # already-arbitrated winner — bypass this and speak directly, no double pass.)
        candidate_id = _c._observe_governor_candidate(
            purpose=purpose or "direct_speech",
            label=label,
            suggested_text=text,
            emotion=emotion,
            wait_secs=wait_secs,
            requires_llm=False,
            # Tell the governor this is a salient/reactive move so its scoring waives
            # the same cadence/active-state gates can_proactive_speak waives for these
            # flags — otherwise the deferred speak_fn (which DOES honor salient) never
            # runs because the candidate is rejected at scoring. The speak_fn still
            # re-checks the real hard gates (DJ/games/flows/live speech).
            metadata={"salient": bool(force_salient), "reactive": bool(reactive)},
            speak_fn=lambda: _do_speak(None),
        )
        return candidate_id is not None

    candidate_id = None
    if governed:
        candidate_id = _c._observe_governor_candidate(
            purpose=purpose or "direct_speech",
            label=label,
            suggested_text=text,
            emotion=emotion,
            wait_secs=wait_secs,
            requires_llm=False,
        )
    return _do_speak(candidate_id)


def claim_proactive_purpose(
    purpose: str,
    *,
    priority: Optional[int] = None,
    label: str = "",
) -> Optional[str]:
    try:
        from intelligence import conversation_agenda
        return conversation_agenda.claim_proactive_purpose(
            purpose,
            priority=priority,
            label=label,
        )
    except Exception as exc:
        _log.debug("proactive purpose claim failed: %s", exc)
        return None


def release_proactive_purpose(token: Optional[str]) -> None:
    try:
        from intelligence import conversation_agenda
        conversation_agenda.release_proactive_claim(token)
    except Exception:
        pass


def proactive_purpose_current(token: Optional[str]) -> bool:
    try:
        from intelligence import conversation_agenda
        return conversation_agenda.proactive_claim_is_current(token)
    except Exception:
        return True


def apply_proactive_directive(prompt: str, purpose: Optional[str]) -> str:
    if not purpose:
        return prompt
    try:
        from intelligence import conversation_agenda
        return conversation_agenda.with_proactive_directive(prompt, purpose)
    except Exception:
        return prompt


def governor_source() -> str:
    try:
        for frame in inspect.stack(context=0):
            name = frame.function
            if name.startswith("_step_") or name.startswith("_do_"):
                return name
    except Exception:
        pass
    return "consciousness"


def governor_speech_metadata() -> dict:
    metadata = {
        "waiting_for_response": _c.is_waiting_for_response(),
        "can_speak": _c._can_speak(),
    }
    try:
        current_state = state_module.get_state()
        metadata["state"] = getattr(current_state, "name", str(current_state))
        metadata["active_state_proactive_blocked"] = (
            current_state == State.ACTIVE
            and not getattr(config, "CONSCIOUSNESS_ALLOW_PROACTIVE_IN_ACTIVE", False)
        )
    except Exception:
        pass
    try:
        from features import games as games_mod
        if hasattr(games_mod, "suppresses_conversation_interruptions"):
            metadata["game_interruptions_suppressed"] = bool(
                games_mod.suppresses_conversation_interruptions()
            )
        elif hasattr(games_mod, "is_active"):
            metadata["game_interruptions_suppressed"] = bool(games_mod.is_active())
    except Exception:
        pass
    metadata["proactive_speech_pending"] = _c._proactive_speech_pending.is_set()
    try:
        metadata["interaction_busy"] = _situation_assessor.is_interaction_busy()
    except Exception:
        pass
    try:
        from audio import output_gate, speech_queue
        metadata["speech_queue_speaking"] = speech_queue.is_speaking()
        metadata["output_gate_busy"] = output_gate.is_busy()
    except Exception:
        metadata["output_gate_status_error"] = True
    with _c._turn_lock:
        last_spoken = _c._last_proactive_speech_at
    if last_spoken:
        recent_gap = time.monotonic() - last_spoken
        metadata["seconds_since_rex_spoke"] = recent_gap
        min_gap = max(0.0, float(getattr(config, "CONSCIOUSNESS_PROACTIVE_MIN_GAP_SECS", 0.0)))
        if min_gap and recent_gap < min_gap:
            metadata["cooldown_active"] = True
            metadata["cooldown_reason"] = "proactive_speech_cooldown"
            metadata["cooldown_remaining_secs"] = max(0.0, min_gap - recent_gap)
    try:
        metadata["can_proactive_speak"] = _c._can_proactive_speak()
    except Exception:
        pass
    return metadata


def observe_governor_candidate(
    *,
    purpose: Optional[str],
    label: str = "",
    prompt: str = "",
    suggested_text: str = "",
    emotion: str = "neutral",
    wait_secs: Optional[float] = None,
    priority: Optional[int] = None,
    target_person_id: Optional[int] = None,
    target_label: str = "",
    requires_llm: bool = True,
    source: Optional[str] = None,
    metadata: Optional[dict] = None,
    speak_fn: Optional[Callable[[], None]] = None,
) -> Optional[str]:
    try:
        from intelligence.action_governor import CandidateMove, governor
        if not governor.active():
            return None
        merged = _c._governor_speech_metadata()
        if metadata:
            merged.update(metadata)
        # Apply the end-of-thread grace + question-budget gates that the legacy
        # conversation_agenda claim enforced. The governor (single decider under
        # ENFORCE) bypasses the claim, and neither it nor can_proactive_speak
        # replicates these gates — so surface them as metadata → rejection reasons,
        # otherwise ENFORCE would fire proactive questions during grace or past the
        # budget. (In shadow/legacy this only enriches the candidate log; the claim
        # still gates the actual speak.)
        try:
            from intelligence import conversation_agenda as _agenda
            _p = purpose or "direct_speech"
            if _agenda.proactive_grace_blocks(_p):
                merged["grace_suppressed"] = True
            if _agenda.proactive_budget_blocks(_p):
                merged["question_budget_exhausted"] = True
        except Exception:
            pass
        candidate = CandidateMove(
            source=source or _c._governor_source(),
            purpose=purpose or "direct_speech",
            label=label or purpose or "",
            prompt=prompt,
            suggested_text=suggested_text,
            emotion=emotion,
            priority=priority,
            target_person_id=target_person_id,
            target_label=target_label,
            requires_llm=requires_llm,
            wait_secs=wait_secs,
            metadata=merged,
            speak_fn=speak_fn,
        )
        if (
            speak_fn is not None
            and governor.enforcing
            and not governor.has_active_cycle()
        ):
            # Off-tick ENFORCE submit (e.g. generate_and_speak / speak_async called
            # from a spawned worker thread like _do_live_vision_comment): the cycle is
            # thread-local to the consciousness tick, so observe() here would only
            # standalone-log and the speak_fn would never run → the line is silently
            # dropped. Route through the cross-thread buffer so the next tick arbitrates
            # it (same path idle banter uses from the interaction thread).
            governor.submit_external(candidate)
            return candidate.candidate_id
        return governor.observe(candidate)
    except Exception as exc:
        _log.debug("action governor observe failed: %s", exc)
        return None


def mark_governor_candidate(candidate_id: Optional[str], outcome: str, reason: str = "") -> None:
    if not candidate_id:
        return
    try:
        from intelligence.action_governor import governor
        governor.mark_outcome(candidate_id, outcome, reason)
    except Exception as exc:
        _log.debug("action governor outcome update failed: %s", exc)


def start_governor_cycle(profile: SituationProfile) -> None:
    try:
        from intelligence.action_governor import governor
        governor.start_cycle(profile=profile)
    except Exception as exc:
        _log.debug("action governor cycle start failed: %s", exc)


def governor_enforcing() -> bool:
    try:
        from intelligence.action_governor import governor
        return bool(governor.enforcing)
    except Exception:
        return False


def finish_governor_cycle() -> None:
    try:
        from intelligence.action_governor import governor
        decision = governor.finish_cycle()
        # ENFORCE mode: the governor is the single decider — run ONLY the winning
        # candidate's deferred speak work (losers stay silent). Shadow mode returns
        # here without acting; each mechanism already spoke for itself.
        if decision is None or not governor.enforcing:
            return
        if decision.action == "speak" and decision.selected is not None:
            speak_fn = getattr(decision.selected.candidate, "speak_fn", None)
            if callable(speak_fn):
                try:
                    speak_fn()
                except Exception as exc:
                    _log.debug("governor winner speak_fn failed: %s", exc)
    except Exception as exc:
        _log.debug("action governor cycle finish failed: %s", exc)


def generate_and_speak(
    prompt: str,
    emotion: str = "neutral",
    *,
    wait_secs: Optional[float] = None,
    purpose: Optional[str] = None,
    priority: Optional[int] = None,
    label: str = "",
    metadata: Optional[dict] = None,
    on_spoke: Optional[Callable[[], None]] = None,
    pre_speak_check: Optional[Callable[[], bool]] = None,
) -> bool:
    # `on_spoke` runs only when this line ACTUALLY speaks (inside the task, after a
    # successful enqueue) — NOT when it's merely queued/submitted. So a caller's
    # "I fired this" bookkeeping (cooldown arm, mark_acknowledged, _fired marker)
    # belongs here, not on the return: under ENFORCE a losing candidate never speaks
    # and so never marks itself done. (Legacy: the held purpose-claim blocks a
    # re-fire until the task finishes, so there is no double-fire window.)
    #
    # `pre_speak_check` re-runs the CALLER's own gates at speak time, for
    # candidates whose suitability can change between submit and the governor
    # win (e.g. the lull callback's empathy/sober-room gates after a heavy
    # disclosure lands mid-tick). Returning False (or raising) drops the line.
    def _task(token):
        try:
            if token is not None and not _c._proactive_purpose_current(token):
                return
            if not _c._can_proactive_speak():
                return
            if pre_speak_check is not None:
                try:
                    if not pre_speak_check():
                        return
                except Exception as exc:
                    _log.debug("pre_speak_check failed — dropping line: %s", exc)
                    return
            from intelligence.llm import get_response
            text = get_response(_c._apply_proactive_directive(prompt, purpose))
            # Opener-diversity backstop for ambient proactive chatter (celebration/emotional
            # check-ins): drop a line that opens with the same word as a recent line — the
            # "Good… Good…" field stack. Scoped to chit-chat purposes, so salient reactions
            # are untouched; on a drop the proactive beat simply yields (no canned fallback).
            if text and _c._proactive_opener_repeats(text, purpose):
                _log.info("[speech_engine] proactive line dropped — opener repeats a recent line: %r", text)
                return
            if text and (token is None or _c._proactive_purpose_current(token)):
                if _c._speak_async(text, emotion, wait_secs=wait_secs, governed=False):
                    if on_spoke is not None:
                        try:
                            on_spoke()
                        except Exception as exc:
                            _log.debug("on_spoke callback failed: %s", exc)
        except Exception as exc:
            _log.debug("generate_and_speak error: %s", exc)
        finally:
            if token is not None:
                _c._release_proactive_purpose(token)

    if _c._governor_enforcing():
        # ENFORCE: submit a candidate carrying the deferred speak work; the cycle
        # resolver runs only the tick's winner. No conversation_agenda claim — the
        # governor subsumes that gate. Speak-time still re-checks can_proactive_speak.
        candidate_id = _c._observe_governor_candidate(
            purpose=purpose,
            label=label,
            prompt=prompt,
            emotion=emotion,
            wait_secs=wait_secs,
            priority=priority,
            requires_llm=True,
            metadata=metadata,
            speak_fn=lambda: threading.Thread(
                target=lambda: _task(None), daemon=True
            ).start(),
        )
        return candidate_id is not None

    # LEGACY (shadow): observe (log only) + conversation_agenda claim + speak now.
    candidate_id = _c._observe_governor_candidate(
        purpose=purpose,
        label=label,
        prompt=prompt,
        emotion=emotion,
        wait_secs=wait_secs,
        priority=priority,
        requires_llm=True,
        metadata=metadata,
    )
    token = None
    if purpose:
        token = _c._claim_proactive_purpose(
            purpose,
            priority=priority,
            label=label or purpose,
        )
        if token is None:
            _c._mark_governor_candidate(
                candidate_id,
                "dropped",
                "conversation_agenda_claim_rejected",
            )
            return False
    _c._mark_governor_candidate(candidate_id, "accepted", "current_behavior_queued_llm")
    threading.Thread(target=lambda: _task(token), daemon=True).start()
    return True


def generate_and_speak_presence(
    prompt: str,
    label: str,
    tag_key,
    emotion: str = "neutral",
    *,
    purpose: str = "presence_reaction",
    priority: Optional[int] = None,
    startup_greeting_name: Optional[str] = None,
    question_key: Optional[str] = None,
    question_depth: int = 1,
    direct_text: Optional[str] = None,
) -> bool:
    """
    Presence-reaction variant of generate_and_speak.

    All gating now flows through _should_fire_presence() before this is called.
    The tag_key is used to coalesce duplicate queued reactions for the same
    person (newer replaces older).
    """
    speech_text = str(direct_text or "").strip()
    candidate_id = _c._observe_governor_candidate(
        purpose=purpose,
        label=label,
        prompt=prompt,
        suggested_text=speech_text,
        emotion=emotion,
        priority=priority,
        target_person_id=tag_key if isinstance(tag_key, int) else None,
        target_label=str(tag_key) if not isinstance(tag_key, int) else "",
        requires_llm=not bool(speech_text),
    )
    token = _c._claim_proactive_purpose(purpose, priority=priority, label=label)
    if token is None:
        _c._mark_governor_candidate(
            candidate_id,
            "dropped",
            "conversation_agenda_claim_rejected",
        )
        return False
    _c._mark_governor_candidate(
        candidate_id,
        "accepted",
        "current_behavior_queued_direct_speech" if speech_text else "current_behavior_queued_llm",
    )
    if not speech_text:
        prompt = _c._apply_proactive_directive(prompt, purpose)

    def _wait_proactive_clear(grace_secs: float) -> bool:
        """Wait briefly for a transient proactive block to clear instead of silently
        dropping the line. A Whisper-hallucination 'user turn' blocks proactive
        speech for ~1-2s — that race used to swallow the startup greeting entirely
        (field log 2026-07-03: greeting accepted by the governor, never spoken,
        Rex silent for a minute). Purpose-current stays a hard cancel."""
        deadline = time.monotonic() + max(0.0, grace_secs)
        while True:
            if not _c._proactive_purpose_current(token):
                return False
            if _c._can_proactive_speak():
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.5)

    def _task():
        if not _c._presence_reaction_lock.acquire(blocking=False):
            _log.debug("generate_and_speak_presence: reaction already in progress, skipping — %s", label)
            _c._release_proactive_purpose(token)
            return
        grace = float(getattr(config, "PRESENCE_SPEAK_GRACE_SECS", 8.0))
        try:
            if not _wait_proactive_clear(grace):
                _log.info(
                    "consciousness: presence reaction dropped before generation — %s "
                    "(proactive blocked past %.0fs grace or superseded)", label, grace,
                )
                return
            if speech_text:
                text = speech_text
            else:
                from intelligence.llm import get_response
                text = get_response(prompt)
            if not text or not text.strip():
                _log.info("consciousness: presence reaction dropped — empty generation (%s)", label)
                return
            if startup_greeting_name and not speech_text:
                text = _c._ensure_named_startup_greeting(text, startup_greeting_name)

            delay = getattr(config, "PRESENCE_REACTION_DELAY_SECS", 2.0)
            if delay > 0:
                time.sleep(delay)

            if not _wait_proactive_clear(grace):
                _log.info(
                    "consciousness: presence reaction dropped after generation — %s "
                    "(proactive blocked past %.0fs grace or superseded)", label, grace,
                )
                return

            from audio import speech_queue
            tag = f"presence:{tag_key}"
            _log.info("consciousness: firing presence reaction — %s: %r", label, text[:120])
            _c._last_presence_reaction_at[tag_key] = time.monotonic()
            done = speech_queue.enqueue(text, emotion, priority=1, tag=tag)
            if isinstance(tag_key, int) and _c._presence_line_counts_as_greeting(label, purpose):
                try:
                    from memory import people as people_mod
                    people_mod.record_greeting(tag_key)
                except Exception as exc:
                    _log.debug("record greeting failed for person_id=%s: %s", tag_key, exc)
            expects_reply = _c._utterance_expects_reply(text)
            _c.note_rex_utterance(
                text,
                open_response_wait=False,
                source=purpose,
                topic=label,
                target_person_id=tag_key if isinstance(tag_key, int) else None,
            )
            if expects_reply:
                def _open_wait_after_presence_done() -> None:
                    done.wait()
                    _c.begin_response_wait()

                threading.Thread(
                    target=_open_wait_after_presence_done,
                    daemon=True,
                    name="presence-response-wait",
                ).start()
            _c._record_proactive_question(
                tag_key if isinstance(tag_key, int) else None,
                text,
                label=label,
                purpose=purpose,
                question_key=question_key,
                question_depth=question_depth,
            )
            if (
                purpose in {"memory_followup", "celebration_checkin", "emotional_checkin"}
                and isinstance(tag_key, int)
            ):
                _c.note_memory_hint(text, tag_key)
        except Exception as exc:
            _log.debug("generate_and_speak_presence error: %s", exc)
        finally:
            _c._release_proactive_purpose(token)
            _c._presence_reaction_lock.release()

    threading.Thread(target=_task, daemon=True).start()
    return True
