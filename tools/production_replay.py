#!/usr/bin/env python3
"""Offline production-turn replay with injected model/audio sinks.

Run in a fresh process only. Hardware, network, and Metal model imports are
blocked before importing interaction. Databases live in a temporary directory.
Fixture format: [{"text": "...", "reply": "scripted model response"}, ...].
This verifies routing/context/delivery plumbing, not model quality or live latency.
"""
from __future__ import annotations
import argparse
import json
import socket
import sqlite3
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fixture', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--text-only', action='store_true',
                        help='exercise the no-audio branch instead of streamed sentence delivery')
    args = parser.parse_args()
    fixtures = json.loads(args.fixture.read_text())
    # Deliberately process-scoped: background workers must never regain access
    # to real transports/databases when a test context manager exits.
    sys.modules['mlx'] = None
    sys.modules['mlx_whisper'] = None
    def forbidden(*a, **kw):
        raise RuntimeError('real device/network access forbidden in production replay')
    socket.socket.connect = forbidden
    import serial
    serial.Serial = forbidden
    import sounddevice as sd
    for name in ('play', 'InputStream', 'OutputStream', 'RawInputStream', 'RawOutputStream'):
        setattr(sd, name, forbidden)
    import config
    temp = tempfile.TemporaryDirectory(prefix='rex-replay-')
    root = Path(temp.name)
    config.DB_PATH = str(root / 'people.db')
    config.REX_DB_PATH = str(root / 'rex.db')
    config.PLACE_DB_PATH = str(root / 'places.db')
    config.TTS_CACHE_DIR = str(root / 'tts')
    config.NO_AUDIO_MODE = True
    config.MEMORY_SEMANTIC_RECALL_ENABLED = False
    config.CONVERSATION_ARC_ENABLED = False
    config.SELF_EMOTION_CLASSIFY_ENABLED = False
    config.WEB_SEARCH_ENABLED = False
    config.ONBOARDING_ENABLED = False
    from setup_assets import DB_SCHEMA
    with sqlite3.connect(config.DB_PATH) as db:
        db.executescript(DB_SCHEMA)
        db.execute("INSERT INTO people (id, name) VALUES (1, 'Replay User')")
    from intelligence import llm_compat
    calls = []
    active = {'reply': ''}
    def create(client, **kwargs):
        calls.append({'model': kwargs.get('model'), 'messages': kwargs.get('messages'),
                      'stream': kwargs.get('stream', False)})
        if kwargs.get('stream'):
            return iter([NS(choices=[NS(delta=NS(content=active['reply'], tool_calls=None))])])
        return NS(choices=[NS(message=NS(content=''))])
    llm_compat.create = create
    # Older utility classifiers bypass the shim. Inject their transport too,
    # so the report counts them and never spends time in network retries.
    from openai.resources.chat.completions import Completions
    Completions.create = create
    from intelligence import interaction as I
    from audio import speech_queue
    from memory import conversations
    import state
    from state import State
    # Keep routing and prompt generation real; replace output and physical acts.
    audio_lines = []
    def enqueue(text, *a, **kwargs):
        audio_lines.append(text)
        for key in ('on_synth_start', 'on_start'):
            callback = kwargs.get(key)
            if callback:
                callback()
        done = speech_queue.DoneEvent()
        done.started = done.played = True
        done.set()
        return done
    speech_queue.enqueue = enqueue
    config.NO_AUDIO_MODE = args.text_only
    config.AUDIO_OUTPUT_SUPPRESSED = False
    I._text_only_mode = True
    I._start_latency_filler_timer = lambda: __import__('threading').Event()
    I._prefetch_stream_audio = lambda *a, **kw: None
    # Turn processing can schedule physical flourishes even with no audio.
    from hardware import servos
    servos.set_servo = forbidden
    report = []
    state.set_state(State.ACTIVE)
    for fixture in fixtures:
        active['reply'] = fixture['reply']
        first_call, first_audio = len(calls), len(audio_lines)
        accepted = I.submit_text(fixture['text'], person_id=1, person_name='Replay User')
        report.append({'input': fixture['text'], 'accepted': accepted,
                       'calls': calls[first_call:], 'delivered': audio_lines[first_audio:],
                       'transcript': conversations.get_session_transcript()})
    args.out.write_text(json.dumps(report, indent=2))
    # Keep the temporary DB alive through interpreter teardown; no restoration
    # to the owner's DB paths occurs in this process.
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
