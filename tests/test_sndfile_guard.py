"""Regression tests for audio/sndfile_guard — the libsndfile MP3-init crash fix.

Two threads opening MP3s through soundfile at once raced mpg123's
non-thread-safe global init inside libsndfile and SIGBUS'd the process
(Bus error: 10, 2026-08-14 — crash report showed both threads inside
libsndfile's mpeg_init). The guard wraps soundfile.read and
SoundFile.__init__ with one re-entrant process lock.

The native race itself can't be unit-tested (losing it kills the
interpreter), so these tests verify the lock plumbing instead: the wrappers
serialize concurrent calls, re-enter cleanly (sf.read constructs a SoundFile
inside the guarded read), pass data through unchanged, and install
idempotently.

Run:  venv/bin/python -m unittest tests.test_sndfile_guard
"""
from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from audio import sndfile_guard  # noqa: E402


class SndfileGuardTest(unittest.TestCase):
    def setUp(self):
        # Full save/restore so each test installs fresh and the global
        # soundfile module leaves this file exactly as it entered.
        self._saved = (
            sf.read,
            sf.SoundFile.__init__,
            sndfile_guard._installed,
            sndfile_guard._orig_read,
            sndfile_guard._orig_sf_init,
        )
        sndfile_guard._installed = False
        self.assertTrue(sndfile_guard.install())

    def tearDown(self):
        (
            sf.read,
            sf.SoundFile.__init__,
            sndfile_guard._installed,
            sndfile_guard._orig_read,
            sndfile_guard._orig_sf_init,
        ) = self._saved

    def test_install_is_idempotent(self):
        wrapped_read = sf.read
        wrapped_init = sf.SoundFile.__init__
        self.assertTrue(sndfile_guard.is_installed())
        self.assertTrue(sndfile_guard.install())   # second call: no re-wrap
        self.assertIs(sf.read, wrapped_read)
        self.assertIs(sf.SoundFile.__init__, wrapped_init)
        # The wrappers are actually in place (not the originals).
        self.assertIsNot(sf.read, sndfile_guard._orig_read)
        self.assertIsNot(sf.SoundFile.__init__, sndfile_guard._orig_sf_init)

    def test_concurrent_reads_are_serialized(self):
        """Six threads through sf.read must never overlap inside the decoder.

        Without the guard the 30 ms sleep makes overlap certain; with it the
        observed concurrency must stay at exactly 1 — the property that keeps
        two MP3 opens out of mpeg_init simultaneously.
        """
        conc = 0
        max_conc = 0
        meter = threading.Lock()
        results = []

        def probe(*args, **kwargs):
            nonlocal conc, max_conc
            with meter:
                conc += 1
                max_conc = max(max_conc, conc)
            time.sleep(0.03)
            with meter:
                conc -= 1
            return ("decoded", 22050)

        sndfile_guard._orig_read = probe
        threads = [
            threading.Thread(target=lambda: results.append(sf.read("fake.mp3")))
            for _ in range(6)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
        self.assertTrue(all(not t.is_alive() for t in threads), "reads wedged")
        self.assertEqual(max_conc, 1, "guarded sf.read calls overlapped")
        self.assertEqual(results, [("decoded", 22050)] * 6)

    def test_concurrent_soundfile_opens_are_serialized(self):
        """Direct SoundFile construction (sf.write/info path) is also guarded."""
        conc = 0
        max_conc = 0
        meter = threading.Lock()

        def probe_init(self_sf, *args, **kwargs):
            nonlocal conc, max_conc
            with meter:
                conc += 1
                max_conc = max(max_conc, conc)
            time.sleep(0.03)
            with meter:
                conc -= 1

        sndfile_guard._orig_sf_init = probe_init
        threads = [
            threading.Thread(target=lambda: sf.SoundFile("fake.mp3"))
            for _ in range(6)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
        self.assertTrue(all(not t.is_alive() for t in threads), "opens wedged")
        self.assertEqual(max_conc, 1, "guarded SoundFile opens overlapped")

    def test_reentrant_open_inside_read_does_not_deadlock(self):
        """sf.read internally constructs a SoundFile: guarded open inside
        guarded read, same thread. The RLock must let it through."""
        with tempfile.TemporaryDirectory(prefix="sndfile_guard_") as td:
            wav = Path(td) / "tone.wav"
            tone = np.sin(np.linspace(0.0, 40.0, 2000)).astype(np.float32)
            sf.write(str(wav), tone, 16000)

            done = threading.Event()
            out = {}

            def run():
                # Real sf.read -> real SoundFile.__init__, both wrapped.
                out["audio"], out["sr"] = sf.read(
                    str(wav), dtype="float32", always_2d=False
                )
                done.set()

            worker = threading.Thread(target=run, daemon=True)
            worker.start()
            self.assertTrue(done.wait(timeout=5.0), "guarded sf.read deadlocked")
            self.assertEqual(out["sr"], 16000)
            self.assertEqual(len(out["audio"]), len(tone))
            np.testing.assert_allclose(out["audio"], tone, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
