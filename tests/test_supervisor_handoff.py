"""Temporary OS leases and fake streams; no microphone or controller launch."""
from pathlib import Path
import tempfile
import unittest
from utils.supervisor_handoff import SupervisorHandoff, Lease


class Stream:
    def __init__(self): self.closed=False;self.fail=False
    def close(self):
        if self.fail: raise OSError('fake close failure')
        self.closed=True


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'controller'
        self.running=False
        self.gate=SupervisorHandoff(self.path,lambda:self.running,pid=123)
        self.gate.start()
        self.addCleanup(self.gate.registration.close)
        self.addCleanup(self.gate.microphone.close)

    def test_stream_close_is_ack_and_listener_resumes(self):
        raw=Stream();stream=self.gate.open_stream(lambda:raw)
        probe=Lease(self.path.with_name('controller.v3-microphone'))
        self.assertFalse(probe.acquire())
        self.running=True
        stream.close()
        self.assertTrue(probe.acquire())
        self.assertIsNone(self.gate.open_stream(lambda:self.fail('opened during controller')))
        probe.close();self.running=False
        resumed=self.gate.open_stream(Stream)
        self.assertIsNotNone(resumed)
        stream.close()  # Late repeated cleanup cannot release a newer stream.
        self.assertFalse(probe.acquire())
        resumed.close()
        self.gate.close()

    def test_close_failure_retains_lease_and_prevents_reopen(self):
        raw=Stream();stream=self.gate.open_stream(lambda:raw);raw.fail=True
        with self.assertRaises(OSError):stream.close()
        self.assertIsNone(self.gate.open_stream(lambda:self.fail('reopened after failed close')))
        with self.assertRaises(RuntimeError):self.gate.close()
        raw.fail=False;stream.close();self.gate.close()

    def test_intent_race_after_lease_prevents_open(self):
        checks=iter((False,True));self.gate.controller_running=lambda:next(checks)
        self.assertIsNone(self.gate.open_stream(lambda:self.fail('opened during race')))
        self.assertIsNone(self.gate.microphone.handle)

    def test_constructor_failure_releases_microphone(self):
        def fail():raise OSError('fake open failure')
        with self.assertRaises(OSError): self.gate.open_stream(fail)
        self.assertIsNone(self.gate.microphone.handle)

    def test_registration_is_live_lock_not_pid_file(self):
        probe=Lease(self.gate.registration.path)
        self.assertFalse(probe.acquire())
        self.gate.close()
        self.assertTrue(probe.acquire());probe.close()
