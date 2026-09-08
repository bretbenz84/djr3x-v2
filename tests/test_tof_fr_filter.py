"""Compile the actual front-right firmware filter and replay sensor evidence. No I/O."""
import csv
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FrontRightPersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='rex-tof-filter-')
        cls.addClassCleanup(cls.temp.cleanup)
        source = Path(cls.temp.name)/'filter.cpp'
        cls.binary = Path(cls.temp.name)/'filter'
        source.write_text(r'''
#include <iostream>
#include "tof_fr_filter.h"
#include "tof_filter.h"
int main() {
  TofFrFilter fr; TofFilt legacy;
  uint32_t t; int mm, status;
  while (std::cin >> t >> mm >> status) {
    const int input = tof_fr_input_mm(mm, status == 0 || status == 3, status == 4);
    const int output = tof_fr_filter_step(fr, input, t);
    // The unchanged legacy status policy manufactured clear from invalid frames.
    const int old_input = (status == 0 || status == 3) ? mm : TOF_L1X_OUT_OF_RANGE_MM;
    std::cout << output << ' ' << tof_filter_step(legacy, old_input) << '\n';
  }
}
''')
        subprocess.run(['clang++','-std=c++17','-Wall','-Wextra','-Werror',
                        '-I',str(ROOT/'firmware/djr3x_motion'),str(source),'-o',str(cls.binary)],
                       check=True,capture_output=True,text=True)

    def run_filter(self, values):
        rows = [(i*80, x, 0) if isinstance(x,int) else x for i,x in enumerate(values)]
        result = subprocess.run([str(self.binary)], input=''.join(f'{t & 0xffffffff} {mm} {st}\n' for t,mm,st in rows),
                                text=True,capture_output=True,check=True)
        return [tuple(map(int,line.split())) for line in result.stdout.splitlines()]

    def test_short_close_bursts_do_not_replace_clear_range(self):
        for length in (1,2,3,4):
            with self.subTest(length=length):
                outputs=self.run_filter([3000]+([65]*length+[3000])*4)
                self.assertTrue(all(fr==3000 for fr,old in outputs))
                if length>=2:self.assertTrue(any(old==65 for fr,old in outputs))

    def test_persistent_close_is_accepted_after_300_ms_not_two_reads(self):
        outputs=self.run_filter([3000]+[65]*7)
        self.assertEqual([r[0] for r in outputs[:5]],[3000]*5)
        self.assertEqual(outputs[5][0],65)  # samples at 80..400 ms
        self.assertEqual(outputs[2][1],65)  # the old two-frame guard failed here

    def test_close_drop_under_old_400_mm_threshold_also_needs_confirmation(self):
        outputs=self.run_filter([350]+[65]*5)
        self.assertEqual(outputs[1],(350,65))
        self.assertEqual(outputs[-1][0],65)

    def test_clear_observation_resets_close_proof(self):
        outputs=self.run_filter([3000,65,65,65,3000,65,65,65,65,3000])
        self.assertTrue(all(fr==3000 for fr,old in outputs))

    def test_invalid_measurement_cannot_complete_or_extend_close_proof(self):
        seq=[(0,3000,0),(80,65,0),(160,65,0),(240,65,7),
             (320,65,0),(400,65,0),(480,65,0),(560,65,0)]
        self.assertTrue(all(fr==3000 for fr,old in self.run_filter(seq)))

    def test_missing_or_duplicate_sample_times_cannot_supply_persistence(self):
        seq=[(0,3000,0),(80,65,0)]+[(80,65,0)]*20+[(400,65,0),(480,65,0)]
        self.assertTrue(all(fr==3000 for fr,old in self.run_filter(seq)))

    def test_consistent_noisy_near_readings_pass(self):
        outputs=self.run_filter([3000,65,83,70,62,76])
        self.assertEqual(outputs[-1][0],76)

    def test_different_range_bands_do_not_accumulate_one_candidate(self):
        outputs=self.run_filter([3000]+[65,600]*12)
        self.assertTrue(all(fr==3000 for fr,old in outputs))

    def test_smooth_physical_approach_has_no_extra_latency(self):
        ranges=list(range(1600,39,-40))
        self.assertEqual([r[0] for r in self.run_filter(ranges)],ranges)

    def test_cold_start_already_near_is_conservative(self):
        self.assertEqual(self.run_filter([160]*8),[(160,160)]*8)

    def test_far_glitches_do_not_release_confirmed_obstacle(self):
        outputs=self.run_filter([160,4000,4000,160,4000,160])
        self.assertEqual([x[0] for x in outputs],[160]*6)

    def test_persistent_clear_releases_without_rearming_each_step(self):
        outputs=self.run_filter([160]+[4000]*17)
        self.assertEqual([x[0] for x in outputs[:4]],[160,160,160,460])
        self.assertEqual(outputs[-1][0],4000)
        self.assertTrue(all(b[0]-a[0]<=300 for a,b in zip(outputs,outputs[1:])))

    def test_status_failures_hold_instead_of_manufacturing_clear(self):
        outputs=self.run_filter([(0,160,0),(80,165,7),(160,163,6),(240,157,1),(320,158,0)])
        self.assertEqual([x[0] for x in outputs],[160,160,160,160,158])
        self.assertGreater(outputs[1][1],400)

    def test_explicit_out_of_range_can_confirm_clear(self):
        outputs=self.run_filter([(0,160,0),(80,10,4),(160,10,4),(240,10,4)])
        self.assertEqual(outputs[-1][0],460)

    def test_millisecond_wrap_preserves_confirmation_duration(self):
        t=0xffffff00
        outputs=self.run_filter([(t,3000,0)]+[(t+80*i,65,0) for i in range(1,6)])
        self.assertEqual(outputs[-1][0],65)
        self.assertTrue(all(fr==3000 for fr,old in outputs[:-1]))

    def test_recorded_stationary_data_preserves_sustained_near_range(self):
        with (ROOT/'tests/fixtures/tof_fr_stationary_2026_09_07.csv').open() as f:
            rows=list(csv.DictReader(f))
        outputs=self.run_filter([(int(r['t_ms']),int(r['raw_mm']),int(r['status'])) for r in rows])
        self.assertEqual(len(outputs),543)
        self.assertEqual([old for fr,old in outputs],[int(r['old_filtered_mm']) for r in rows])
        self.assertTrue(all(150<=fr<=168 for fr,old in outputs))
        self.assertEqual(sum(old>300 for fr,old in outputs),10)


if __name__=='__main__':unittest.main()
