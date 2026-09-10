"""I/O adapter for the conversational learner, called after speech trust/echo gates."""
from datetime import datetime, timezone
from functools import wraps
import logging
import os
import time

import numpy as np

import config
from intelligence.voice_learning import Learner, Policy, Sample, visible_faces
from audio import speaker_id, voice_score
from memory import people, conversations, voice_recordings

log = logging.getLogger(__name__)


def serialized(method):
    @wraps(method)
    def run(self, *args, **kwargs):
        with self.learner.lock:
            return method(self, *args, **kwargs)
    return run


def enabled():
    return speaker_id.active_backend() == 'campplus'


class Runtime:
    def __init__(self, interaction):
        from hardware import servos
        self.i = interaction
        policy = Policy(
            min_turns=config.VOICE_LEARNING_MIN_TURNS,
            min_voiced=config.VOICE_LEARNING_MIN_VOICED_SECS,
            sample_min_voiced=config.VOICE_LEARNING_SAMPLE_MIN_VOICED_SECS,
            anchor_cosine=config.VOICE_LEARNING_ANCHOR_COSINE,
            short_anchor_cosine=config.VOICE_LEARNING_SHORT_ANCHOR_COSINE,
            cluster_cosine=config.VOICE_LEARNING_CLUSTER_COSINE,
            foreign_cosine=config.VOICE_LEARNING_FOREIGN_COSINE,
            bearing_tolerance=config.VOICE_LEARNING_BEARING_TOLERANCE_DEG,
            window_secs=config.VOICE_LEARNING_WINDOW_SECS,
            resume_secs=config.VOICE_LEARNING_RESUME_SECS)
        self.learner = Learner(policy, hold=servos.hold_voice_enrollment_mic,
                               release=servos.release_voice_enrollment_hold)
        self.last_sample = None
        self.turn_person = None
        self.saved = False

    def tick(self):
        from hardware import servos
        i = self.i
        self.learner.tick(conversations.transcript_version()[0])
        if self.last_sample and time.monotonic() - self.last_sample.ended_at > self.learner.policy.window_secs:
            self.last_sample = None
        if (i._game_suppresses_conversation() or i._shutdown_requested()
                or getattr(config, 'INTERACTION_PAUSED', False) or servos.manual_override_enabled()):
            self.learner.cancel('game_shutdown_or_manual_control')
            self.last_sample = None

    @serialized
    def request(self, pid, name):
        if self.i._game_suppresses_conversation() or speaker_id.comparable_print_count(pid):
            return False
        return self.learner.request(pid, name or 'friend', conversations.transcript_version()[0],
                                    self.i.world_state.get('people') or [])

    def _sample(self, audio, text):
        i = self.i
        capture = i._utterance_observations
        start, end = capture.get('started_at'), capture.get('ended_at')
        if audio is None or start is None or end is None:
            return None
        if not i._turn_transcript_trusted() or i._is_non_speech_vocalization(text):
            return None
        # Audio recorded during Rex's own speech or mic/body movement is never
        # training material, even if ASR produced plausible human words.
        if start < float(i.echo_cancel.last_playback_ended_at() or 0):
            return None
        from intelligence import motion_controller
        from intelligence import voice_learning
        if motion_controller.is_moving() or start < voice_learning._last_mic_motion_at:
            return None
        bearing = i._last_voice_bearing
        if bearing and bearing.get('utterance_t0') != start:
            bearing = None
        window_ids = {r.get('person_id') for r in i._last_scan_windows if r.get('person_id') is not None}
        mixed = len(window_ids) > 1 or any(r.get('change_suspected') for r in i._last_scan_windows)
        # Deliberately don't read person_db_id/confidence/is_speaking from the
        # active-mouth observation. Only its independent face snapshots count.
        rows = capture.get('faces', capture.get('visual')) or []
        faces = [r.get('faces', []) for r in rows]
        faces.append(i.world_state.get('people') or [])
        if rows:
            stamps = sorted(float(r['monotonic_at']) for r in rows if r.get('monotonic_at') is not None)
            if (len(stamps) != len(rows) or stamps[0] < start or stamps[-1] > end
                    or stamps[0] - start > 1 or end - stamps[-1] > 1
                    or any(b-a > 1 for a, b in zip(stamps, stamps[1:]))):
                return None
        elif len(visible_faces(faces[-1])) > 1:
            return None  # no interval evidence to maintain the addressed face track
        embedding = speaker_id.get_embedding(audio)
        if embedding is None:
            return None
        rate = int(config.AUDIO_SAMPLE_RATE)
        return Sample(np.asarray(audio, dtype=np.float32).copy(), embedding, rate,
                      speaker_id.voiced_secs(audio), str(text), float(start), float(end),
                      faces=faces, bearing=bearing.get('bearing_deg') if bearing else None,
                      direction_conflict=bool(bearing and (bearing.get('head_disagrees') or len(bearing.get('clusters') or []) > 1)),
                      mixed=mixed, ranked=list(i._last_scan_ranked), metadata={
                          'captured_at': datetime.now(timezone.utc).isoformat(),
                          'capture_device': os.environ.get('AUDIO_DEVICE_NAME', 'system default input'),
                          'capture_started_monotonic': start, 'capture_ended_monotonic': end,
                          'direction_available': bearing is not None,
                          'direction_spread_deg': bearing.get('spread_deg') if bearing else None,
                          'visible_person_ids': sorted({p.get('person_db_id') for p in faces[-1] if p.get('person_db_id') is not None})})

    @serialized
    def seed(self, pid, name, audio):
        # Only the sample processed after this turn's echo/trust gates can seed
        # a name handler. Background refresh threads cannot borrow another turn.
        s = self.last_sample
        if (s is None or audio is None or not np.array_equal(s.audio, audio)
                or speaker_id.comparable_print_count(pid) > 0):
            return False
        ok = self.learner.seed(pid, name, conversations.transcript_version()[0], s, explicit=True)
        log.info('[voice_learning] person_id=%s reason=%s', pid, self.learner.last_reason)
        return ok

    @serialized
    def process(self, audio, text):
        """Return a deterministic confirmation/question response, if needed.

        Ordinary accepted samples keep flowing to the conversation. turn_person
        is contextual only until saved=True; neither silently grants learning.
        """
        self.turn_person = None
        self.saved = False
        self.last_sample = None
        self.tick()
        i = self.i
        if i._game_suppresses_conversation() or not i._turn_transcript_trusted():
            return None
        from memory.name_validation import extract_referred_person_name, is_name_correction
        if is_name_correction(text):
            self.learner.cancel('identity_corrected')
            return None
        if extract_referred_person_name(text):
            # A bystander naming Jeff is not Jeff confirming his own voice.
            # Leave last_sample empty so downstream handlers cannot seed it.
            if self.learner.pending and not self.learner.pending.confirmed:
                self.learner.cancel('third_party_reference')
            return None
        s = self._sample(audio, text)
        self.last_sample = s
        if s is None:
            return None
        # User actions take priority over the mic lease. The normal router still
        # executes them; this just discards provisional enrollment first.
        import re
        imperative = re.sub(r'^(?:(?:hey\s+)?rex[, ]*|please\s+|(?:can|could|would) you\s+)*', '', text.strip(), flags=re.I)
        if (re.match(r'^(?:wave|turn|rotate|come|drive|back up|shut down|sleep|forget)\b', imperative, re.I)
                or (re.match(r'^(?:raise|lower|move|lift|drop|put)\b', imperative, re.I)
                    and re.search(r'\b(?:arm|hand|left|right|forward|back)\b', imperative, re.I))):
            self.learner.cancel('user_action')
            return None
        p = self.learner.pending
        if p and re.search(r"\b(?:i(?:'m| am) not|that (?:was|is) not|isn'?t here|not (?:me|the one speaking))\b", text, re.I):
            self.learner.cancel('identity_corrected')
            return None
        p = self.learner.pending
        if p and p.confirmed:
            if self.learner.observe(s, conversations.transcript_version()[0]):
                self.turn_person = (p.person_id, p.name)
                if self.learner.ready():
                    try:
                        ok = voice_recordings.save_batch(p.person_id, p.samples, voice_score.biometric_type(),
                                                        limit=config.VOICE_LEARNING_MAX_RECORDINGS,
                                                        print_limit=config.VOICE_LEARNING_MAX_PRINTS)
                    except Exception:
                        log.exception('[voice_learning] atomic save failed')
                        self.learner.cancel('storage_failed')
                        return None
                    if ok:
                        self.saved = True
                        i._last_confident_voice_at[p.person_id] = time.monotonic()
                        log.info('[voice_learning] SAVED person_id=%s name=%r samples=%d voiced=%.2f',
                                 p.person_id, p.name, len(p.samples), sum(x.voiced for x in p.samples))
                        self.learner.cancel('saved')
                    else:
                        self.learner.cancel('person_removed')
            log.info('[voice_learning] reason=%s', self.learner.last_reason)
            return None
        if p and p.asked_at is not None:
            if self.learner.confirm(s):
                self.turn_person = (p.person_id, p.name)
                # Acknowledges the answer, never a saved voiceprint.
                return f'Thanks, {p.name.split()[0]}. What have you been up to lately?'
            log.info('[voice_learning] confirmation reason=%s', self.learner.last_reason)
            return None
        # Self-introductions that name an existing person need no extra yes/no.
        name = i._extract_self_identified_name(text)
        person = people.find_person_by_name(name) if name else None
        if person is None:
            bare = text.strip().rstrip('.!?')
            if len(bare.split()) >= 2:
                person = people.find_person_by_name(bare)
        if person and not speaker_id.comparable_print_count(person['id']):
            self.seed(person['id'], person['name'], audio)
            return None
        faces = visible_faces(i.world_state.get('people') or [])
        missing = [f for f in faces if f.get('person_db_id') is not None
                   and speaker_id.comparable_print_count(f['person_db_id']) == 0]
        strong_known = bool(s.ranked and float(s.ranked[0][2]) >= self.learner.policy.foreign_cosine
                            and (len(s.ranked) < 2 or s.ranked[0][2] - s.ranked[1][2] >= speaker_id.required_ambiguity_margin(s.ranked)))
        if not p and not strong_known:
            for face in missing:
                pid = face.get('person_db_id')
                if pid is not None and speaker_id.comparable_print_count(pid) == 0:
                    person = people.get_person(pid) or {}
                    if self.request(pid, person.get('name')):
                        break
        # Established voices grow through the very same multi-turn process.
        # A strong existing print plus a visible matching face anchors the batch;
        # thin/weak nearest-neighbour matches cannot bootstrap themselves.
        if not self.learner.pending and s.ranked and not missing:
            pid, name, score, _ = s.ranked[0]
            margin = score - s.ranked[1][2] if len(s.ranked) > 1 else 1.
            count = speaker_id.comparable_print_count(pid)
            if (score >= .80 and margin >= .10 and 0 < count <= config.VOICE_LEARNING_MAX_PRINTS - self.learner.policy.min_turns
                    and any(f.get('person_db_id') == pid for f in faces)):
                self.learner.seed(pid, name, conversations.transcript_version()[0], s, explicit=True)
        return self.learner.question()

    def speak_response(self, text):
        p = self.learner.pending
        question = bool(p and not p.confirmed and p.asked_at is None)
        if question:
            self.learner.prepare_question()
        # This handler runs before ordinary attribution: log as unassigned, and
        # never turn the confirmation's provisional name into a personal memory.
        i = self.i
        i.conv_memory.add_to_transcript('user', self.last_sample.text)
        i.conv_log.log_heard(None, self.last_sample.text)
        completed = i._speak_blocking(text, emotion='curious')
        if question:
            self.learner.question_spoken(completed)
        if completed:
            i.conv_memory.add_to_transcript('Rex', text)
            i.conv_log.log_rex(text)
            i._register_rex_utterance(text)
            i._session_exchange_count += 1
