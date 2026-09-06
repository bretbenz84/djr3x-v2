"""Production submit_text seam in a separate process with blocked real I/O."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

class ProductionReplayTest(unittest.TestCase):
    def test_two_turns_use_the_live_reply_prompt_and_delivered_history(self):
        root = Path(__file__).resolve().parents[1]
        fixture = root / 'tests/fixtures/lean_production_replay.json'
        with tempfile.TemporaryDirectory() as temp:
            report = Path(temp) / 'report.json'
            run = subprocess.run([sys.executable, str(root / 'tools/production_replay.py'),
                str(fixture), '--out', str(report)], cwd=root, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=45)
            self.assertEqual(run.returncode, 0, run.stdout)
            turns = json.loads(report.read_text())
        expected = json.loads(fixture.read_text())
        self.assertEqual(len(turns), 2)
        for turn, case in zip(turns, expected):
            self.assertTrue(turn['accepted'])
            self.assertEqual(' '.join(turn['delivered']), case['reply'])
            primary = [call for call in turn['calls'] if call['stream']]
            self.assertEqual(len(primary), 1)
        history = [call for call in turns[1]['calls'] if call['stream']][0]['messages']
        self.assertTrue(any(expected[0]['reply'] == row.get('content') for row in history))
