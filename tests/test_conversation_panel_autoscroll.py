"""Auto-scroll behavior for the GUI chat transcript (gui/conversation_panel.py).

The transcript must keep the newest line pinned to the bottom of the view as the
conversation grows (the long-standing "text runs off below the window" bug), while
still letting a reader scroll up through history without being yanked back down.

QTextBrowser relays its HTML out lazily, so the scrollbar maximum is stale right after
setHtml — these tests guard the rangeChanged + ensureCursorVisible fix that pins to the
*true* bottom once layout settles. Runs headless via the Qt 'offscreen' platform.
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication
    from gui.conversation_panel import ConversationPanel
    _HAVE_QT = True
except Exception:  # pragma: no cover - PySide6 not installed in this environment
    _HAVE_QT = False


def _lines(n):
    return [{
        "seq": i, "ts": 1000 + i,
        "speaker": "R3X" if i % 2 else "Bret",
        "text": f"message {i}: " + ("blah " * 12),
        "kind": "rex" if i % 2 else "user",
    } for i in range(n)]


@unittest.skipUnless(_HAVE_QT, "PySide6 not available")
class ConversationAutoScrollTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.panel = ConversationPanel()
        self.panel.resize(320, 220)   # small viewport so the transcript overflows
        self.panel.show()
        self.app.processEvents()
        self.addCleanup(self.panel.deleteLater)

    def _feed(self, n):
        self.panel.set_snapshot({"conversation_lines": _lines(n)})
        self.app.processEvents()
        return self.panel._log.verticalScrollBar()

    def _at_bottom(self, bar):
        return bar.value() >= bar.maximum() - 6

    def test_long_transcript_is_scrolled_to_bottom(self):
        bar = self._feed(60)
        self.assertGreater(bar.maximum(), 0, "transcript should overflow the viewport")
        self.assertTrue(self._at_bottom(bar), "newest line must be at the bottom")

    def test_new_lines_stick_to_bottom(self):
        self._feed(60)
        for n in (61, 62, 70, 90):
            bar = self._feed(n)
            self.assertTrue(self._at_bottom(bar), f"did not stick to bottom at {n} lines")

    def test_scrolled_up_reader_is_not_yanked_down(self):
        bar = self._feed(90)
        bar.setValue(0)              # reader scrolls up to the top
        self.app.processEvents()
        bar = self._feed(95)         # 5 new lines arrive
        self.assertLess(bar.value(), bar.maximum() - 6,
                        "reader was yanked to the bottom while reading history")

    def test_autoscroll_reengages_after_returning_to_bottom(self):
        bar = self._feed(90)
        bar.setValue(0)
        self.app.processEvents()
        self._feed(95)
        bar = self.panel._log.verticalScrollBar()
        bar.setValue(bar.maximum())  # reader scrolls back to the bottom
        self.app.processEvents()
        bar = self._feed(100)
        self.assertTrue(self._at_bottom(bar), "auto-scroll should re-engage at the bottom")

    def test_resize_keeps_bottom_pinned(self):
        self._feed(90)
        self.panel.resize(320, 120)  # shrink the viewport
        self.app.processEvents()
        bar = self.panel._log.verticalScrollBar()
        self.assertTrue(self._at_bottom(bar), "resize must keep the newest line in view")


if __name__ == "__main__":
    unittest.main()
