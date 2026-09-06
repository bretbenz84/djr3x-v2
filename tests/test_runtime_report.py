import types
import unittest
from utils import runtime_report

class RuntimeReportTest(unittest.TestCase):
    def test_report_is_allowlisted_and_does_not_load_models(self):
        import sys
        before = set(sys.modules)
        result = runtime_report.snapshot(types.SimpleNamespace(
            OPENAI_API_KEY='never export this', LEAN_BRAIN_ENABLED=True))
        self.assertNotIn('never export this', str(result))
        self.assertEqual(result['owners']['reply'], 'lean')
        self.assertGreater(result['resources']['peak_rss_bytes'], 0)
        self.assertEqual(before, set(sys.modules))
