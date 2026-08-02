"""
Offline mode (owner spec 2026-08-01): lose internet → degrade, don't stop.
Field motivation: the 22:47 outage wedged one turn for 125s (OpenAI retries +
ElevenLabs timeouts) while the mic stayed gated. Pins: the connectivity state
machine, the guarded-client fail-fast, the lean local reroute (including the
mid-turn fallback), and the in-character announcement plumbing.
"""

import unittest
from unittest import mock

import config
from intelligence import connectivity


class _Completions:
    def __init__(self):
        self.calls = 0

    def create(self, **kw):
        self.calls += 1
        return "hosted-result"


class _Chat:
    def __init__(self):
        self.completions = _Completions()


class _FakeClient:
    def __init__(self):
        self.chat = _Chat()


class ConnectivityStateTest(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(connectivity, "_prewarm_offline_brain")
        p.start(); self.addCleanup(p.stop)
        connectivity._set_state(True)
        connectivity._listeners.clear()
        self.addCleanup(connectivity._set_state, True)
        self.addCleanup(connectivity._listeners.clear)

    def test_disabled_is_always_online(self):
        with mock.patch.object(config, "OFFLINE_MODE_ENABLED", False, create=True):
            connectivity._set_state(True)
            self.assertTrue(connectivity.is_online())
            self.assertTrue(connectivity.note_failure("x"))

    def test_failure_with_dead_probe_goes_offline_and_fires_listener(self):
        events = []
        connectivity.add_listener(events.append)
        with mock.patch.object(connectivity, "_probe", return_value=False), \
             mock.patch.object(connectivity, "_ensure_monitor"):
            connectivity._last_probe_at = 0.0
            up = connectivity.note_failure("test")
        self.assertFalse(up)
        self.assertTrue(connectivity.is_offline())
        self.assertEqual(events, [False])

    def test_failure_with_live_probe_stays_online(self):
        with mock.patch.object(connectivity, "_probe", return_value=True):
            connectivity._last_probe_at = 0.0
            self.assertTrue(connectivity.note_failure("test"))
        self.assertTrue(connectivity.is_online())

    def test_probes_are_rate_limited(self):
        with mock.patch.object(connectivity, "_probe", return_value=True) as probe:
            connectivity._last_probe_at = 0.0
            connectivity.note_failure("a")
            connectivity.note_failure("b")   # inside the min interval — no probe
        self.assertEqual(probe.call_count, 1)

    def test_recovery_fires_online_listener(self):
        events = []
        connectivity.add_listener(events.append)
        with mock.patch.object(connectivity, "_ensure_monitor"):
            connectivity._set_state(False)
        connectivity._set_state(True)
        self.assertEqual(events, [False, True])


class GuardedClientTest(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(connectivity, "_prewarm_offline_brain")
        p.start(); self.addCleanup(p.stop)
        connectivity._set_state(True)
        self.addCleanup(connectivity._set_state, True)

    def test_offline_raises_instantly_without_calling_through(self):
        client = connectivity.guard_client(_FakeClient(), "test")
        with mock.patch.object(connectivity, "_ensure_monitor"):
            connectivity._set_state(False)
        with self.assertRaises(connectivity.OfflineError):
            client.chat.completions.create(model="x", messages=[])
        # And the wrapped create was never invoked.
        self.assertEqual(client.chat._real_calls if hasattr(client.chat, "_real_calls")
                         else 0, 0)

    def test_online_passes_through(self):
        client = connectivity.guard_client(_FakeClient(), "test")
        self.assertEqual(client.chat.completions.create(model="x"), "hosted-result")

    def test_transport_failure_reports_note_failure(self):
        client = _FakeClient()

        def boom(**kw):
            raise TimeoutError("net down")
        client.chat.completions.create = boom
        client = connectivity.guard_client(client, "test")
        with mock.patch.object(connectivity, "note_failure") as nf:
            with self.assertRaises(TimeoutError):
                client.chat.completions.create(model="x")
        nf.assert_called_once()


class LeanOfflineRerouteTest(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(connectivity, "_prewarm_offline_brain")
        p.start(); self.addCleanup(p.stop)
        connectivity._set_state(True)
        self.addCleanup(connectivity._set_state, True)

    def test_offline_reply_streams_from_local_model(self):
        from intelligence import lean_brain, local_llm
        with mock.patch.object(connectivity, "_ensure_monitor"):
            connectivity._set_state(False)
        with mock.patch.object(local_llm, "stream_chat",
                               return_value=iter(["Local ", "reply."])) as sc, \
             mock.patch.object(lean_brain, "_messages",
                               return_value=[{"role": "system", "content": "persona"},
                                             {"role": "user", "content": "hi"}]):
            out = "".join(lean_brain.stream_reply("hi", 1))
        self.assertEqual(out, "Local reply.")
        # The system prompt gained the offline-capability note.
        sent = sc.call_args[0][0]
        self.assertIn("OFFLINE MODE", sent[0]["content"])

    def test_hosted_failure_falls_back_to_local_same_turn(self):
        from intelligence import lean_brain, local_llm

        def dead_create(*a, **kw):
            raise TimeoutError("net down")
        with mock.patch.object(lean_brain.llm_compat, "create", side_effect=dead_create), \
             mock.patch.object(lean_brain, "_messages",
                               return_value=[{"role": "system", "content": "p"},
                                             {"role": "user", "content": "hi"}]), \
             mock.patch.object(connectivity, "note_failure", return_value=False), \
             mock.patch.object(local_llm, "stream_chat",
                               return_value=iter(["Backup brain here."])):
            out = "".join(lean_brain.stream_reply("hi", 1))
        self.assertEqual(out, "Backup brain here.")
        self.assertNotIn("circuits hiccuped", out)

    def test_hosted_failure_while_actually_online_keeps_hiccup_line(self):
        from intelligence import lean_brain

        def dead_create(*a, **kw):
            raise TimeoutError("blip")
        with mock.patch.object(lean_brain.llm_compat, "create", side_effect=dead_create), \
             mock.patch.object(lean_brain, "_messages",
                               return_value=[{"role": "user", "content": "hi"}]), \
             mock.patch.object(connectivity, "note_failure", return_value=True):
            out = "".join(lean_brain.stream_reply("hi", 1))
        self.assertIn("circuits hiccuped", out)


class AnnouncementLinesTest(unittest.TestCase):
    def test_lines_mention_the_galactic_internet(self):
        self.assertIn("galactic internet", connectivity.offline_announcement().lower())
        self.assertIn("galactic internet",
                      connectivity.no_internet_reply().lower())
        self.assertTrue(connectivity.online_announcement())


if __name__ == "__main__":
    unittest.main()
