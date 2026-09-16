"""Replay real failed sensor status and verify recovery without motor I/O."""
import csv
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class FrontRightHealthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        source = Path(cls.tmp.name)/'health.cpp'
        cls.binary = Path(cls.tmp.name)/'health'
        source.write_text('''#include <iostream>
#include "tof_fr_filter.h"
int main() { TofFrHealth h; unsigned t; int valid;
while(std::cin >> t >> valid) std::cout << tof_fr_health_step(h, valid, t) << '\\n'; }
''')
        subprocess.run(['clang++','-std=c++17','-Wall','-Wextra','-Werror','-I',str(ROOT/'firmware/djr3x_motion'),str(source),'-o',str(cls.binary)],check=True)

    def replay(self, rows):
        p = subprocess.run([str(self.binary)],input=''.join(f'{t} {int(v)}\n' for t,v in rows),text=True,capture_output=True,check=True)
        return [int(x) for x in p.stdout.split()]

    def test_live_failure_flicker_stays_quarantined(self):
        with open(ROOT/'tests/fixtures/tof_fr_quality_2026_09_15.csv') as f:
            rows=[(int(r['t']),int(r['status']) in (0,3,4)) for r in csv.DictReader(f)]
        out=self.replay(rows)
        first=out.index(0)
        self.assertLess(first,25)
        self.assertEqual(set(out[first:]),{0})

    def test_single_bad_frame_does_not_discard_real_obstacle(self):
        self.assertEqual(self.replay([(i*80,i!=20) for i in range(50)]),[1]*50)

    def test_recovery_needs_sixteen_fresh_valid_samples(self):
        vals=[False]*8+[True]*15+[False]+[True]*16
        out=self.replay([(i*80,v) for i,v in enumerate(vals)])
        self.assertEqual(out[7:-1],[0]*(len(vals)-8))
        self.assertEqual(out[-1],1)

    def test_cached_readings_and_time_gap_cannot_recover(self):
        rows=[(i*80,False) for i in range(8)]+[(640,True)]*30+[(1000,True)]
        self.assertEqual(self.replay(rows)[7:],[0]*(len(rows)-7))

    def test_healthy_start_and_clock_wrap(self):
        rows=[((0xffffff00+i*80)&0xffffffff,True) for i in range(30)]
        self.assertEqual(self.replay(rows),[1]*30)
