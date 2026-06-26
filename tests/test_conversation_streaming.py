"""Read-along streaming of Rex's reply into the GUI transcript.

A streamed reply must fill the conversation panel sentence-by-sentence AS it is
generated (so the text leads the TTS), instead of appearing as one block after
playback finishes. These cover the three pieces:
  * the bridge growing a single Rex bubble in place (append/finish),
  * conv_log's GUI-only stream helpers + the file-only `to_gui=False` path,
  * the panel re-rendering when the last line's text grows under a stable seq.
"""

import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication
    from gui.conversation_panel import ConversationPanel
    _HAVE_QT = True
except Exception:  # pragma: no cover - PySide6 not installed in this environment
    _HAVE_QT = False


class StreamingBridgeTests(unittest.TestCase):
    def _bridge(self):
        from gui.state_bridge import GUIDashboardBridge

        return GUIDashboardBridge()

    def _lines(self, bridge):
        return bridge.get_snapshot()["conversation_lines"]

    def test_first_append_creates_line_then_grows_in_place(self):
        b = self._bridge()
        b.append_rex_stream("Oh, bold choice.")
        b.append_rex_stream("Real bold.")
        lines = self._lines(b)
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["kind"], "rex")
        self.assertEqual(lines[0]["text"], "Oh, bold choice. Real bold.")
        # The seq stays stable across the grow — that's what lets it be ONE bubble.
        self.assertEqual(lines[0]["seq"], 1)

    def test_finish_overwrites_canonical_text_and_starts_fresh_next_time(self):
        b = self._bridge()
        b.append_rex_stream("Oh, bold choice.")
        b.finish_rex_stream("Oh, bold choice. Real bold.")
        lines = self._lines(b)
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["text"], "Oh, bold choice. Real bold.")
        # The NEXT reply must not grow onto the finished bubble.
        b.append_rex_stream("Next reply entirely.")
        lines = self._lines(b)
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[1]["text"], "Next reply entirely.")

    def test_intervening_line_starts_a_fresh_bubble(self):
        b = self._bridge()
        b.append_rex_stream("First reply.")
        # A human turn lands between replies — the streaming line is no longer last.
        b.add_conversation_line("Bret", "hey rex", "user")
        b.append_rex_stream("Second reply.")
        texts = [ln["text"] for ln in self._lines(b)]
        self.assertEqual(texts, ["First reply.", "hey rex", "Second reply."])

    def test_finish_without_active_stream_is_safe(self):
        b = self._bridge()
        b.finish_rex_stream("nothing was streaming")
        self.assertEqual(self._lines(b), [])

    def test_blank_append_is_ignored(self):
        b = self._bridge()
        b.append_rex_stream("   ")
        self.assertEqual(self._lines(b), [])


class ConvLogStreamingTests(unittest.TestCase):
    def test_log_rex_to_gui_false_skips_the_panel_mirror(self):
        from utils import conv_log

        conv_log.clear_dedupe_state()
        with mock.patch.object(conv_log, "_mirror_to_gui") as mirror:
            conv_log.log_rex("a streamed reply, file only", to_gui=False)
        mirror.assert_not_called()

    def test_log_rex_default_still_mirrors(self):
        from utils import conv_log

        conv_log.clear_dedupe_state()
        with mock.patch.object(conv_log, "_mirror_to_gui") as mirror:
            conv_log.log_rex("a normal reply")
        mirror.assert_called_once()

    def test_log_rex_stream_gated_on_gui_enabled(self):
        from utils import conv_log

        fake = mock.MagicMock()
        with mock.patch("gui.state_bridge.gui_bridge", fake):
            with mock.patch.object(conv_log.config, "GUI_ENABLED", False, create=True):
                conv_log.log_rex_stream("hi")
            fake.append_rex_stream.assert_not_called()
            with mock.patch.object(conv_log.config, "GUI_ENABLED", True, create=True):
                conv_log.log_rex_stream("hi")
            fake.append_rex_stream.assert_called_once_with("hi")

    def test_finish_rex_stream_forwards_when_gui_enabled(self):
        from utils import conv_log

        fake = mock.MagicMock()
        with mock.patch("gui.state_bridge.gui_bridge", fake):
            with mock.patch.object(conv_log.config, "GUI_ENABLED", True, create=True):
                conv_log.finish_rex_stream("final text")
        fake.finish_rex_stream.assert_called_once_with("final text")


@unittest.skipUnless(_HAVE_QT, "PySide6 not available")
class StreamingPanelRenderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.panel = ConversationPanel()
        self.panel.resize(320, 220)
        self.panel.show()
        self.app.processEvents()
        self.addCleanup(self.panel.deleteLater)

    def _feed(self, text, seq=1):
        self.panel.set_snapshot({
            "conversation_lines": [
                {"seq": seq, "ts": 1000, "speaker": "Rex", "text": text, "kind": "rex"}
            ]
        })
        self.app.processEvents()

    def test_growing_last_line_under_same_seq_rerenders(self):
        # The bug this guards: keying the re-render on seq ALONE skips the grow, so a
        # reply streamed into one bubble would only show its first sentence.
        self._feed("Oh, bold choice.")
        self.assertIn("Oh, bold choice.", self.panel._log.toPlainText())
        self.assertNotIn("Real bold.", self.panel._log.toPlainText())

        self._feed("Oh, bold choice. Real bold.")  # same seq, grown text
        self.assertIn("Real bold.", self.panel._log.toPlainText())


if __name__ == "__main__":
    unittest.main()
