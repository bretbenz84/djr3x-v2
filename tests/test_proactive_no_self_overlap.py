"""A proactive line must never preempt Rex's own in-flight speech (field bug 2026-06-30:
the "Strawberry pillows…" visual-curiosity line was cut off mid-sentence when an idle line
barged in 2s later).

The consciousness proactive path was already gated by can_proactive_speak (it drops a line
while speech_queue.is_speaking()); the idle path reached _speak_blocking directly and skipped
that check. _speak_proactive now drops the ambient line when Rex is already speaking.
"""

import unittest
from unittest import mock

from intelligence import interaction as ix


class ProactiveNoSelfOverlapTest(unittest.TestCase):
    def _run(self, *, rex_speaking, guard_enabled=True):
        with mock.patch.object(ix.speech_queue, "is_speaking", return_value=rex_speaking), \
             mock.patch("audio.output_gate.is_busy", return_value=False), \
             mock.patch.object(ix, "_speak_blocking", return_value=True) as sb, \
             mock.patch.object(ix.config, "PROACTIVE_SPEECH_YIELD_ENABLED", False), \
             mock.patch.object(ix.config, "PROACTIVE_NO_SELF_OVERLAP_ENABLED", guard_enabled):
            out = ix._speak_proactive("Late still suits you, apparently.", label="idle_banter")
        return out, sb

    def test_dropped_while_rex_speaking(self):
        out, sb = self._run(rex_speaking=True)
        self.assertFalse(out)              # line dropped, not spoken
        sb.assert_not_called()             # never reached the speech queue → no preempt

    def test_speaks_when_not_speaking(self):
        out, sb = self._run(rex_speaking=False)
        self.assertTrue(out)
        sb.assert_called_once()

    def test_kill_switch_restores_old_behavior(self):
        # Guard off → it would enqueue even while speaking (the old, buggy behavior).
        out, sb = self._run(rex_speaking=True, guard_enabled=False)
        self.assertTrue(out)
        sb.assert_called_once()


if __name__ == "__main__":
    unittest.main()
