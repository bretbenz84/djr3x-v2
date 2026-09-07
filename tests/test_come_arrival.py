"""Compile and exercise firmware measured arrival and deceleration."""
import subprocess
import tempfile
import unittest
from pathlib import Path


class ComeArrivalTest(unittest.TestCase):
    def test_real_distance_approach_and_invalid_sensors(self):
        root = Path(__file__).resolve().parents[1]
        source = r'''
#include "come_arrival.h"
#include <cassert>
int main() {
  assert(ComeArrival::range_m(-1, -1) < 0);
  assert(fabsf(ComeArrival::range_m(-1, 2100)-2.1f) < .001f);
  assert(ComeArrival::speed(-1, 1.3f, .4f, .35f) == 0);
  assert(!ComeArrival::arrived(-1, 1.3f));
  assert(ComeArrival::arrived(1.32f, 1.3f));
  assert(!ComeArrival::arrived(1.4f, 1.3f));
  // The caller starts three metres out: passing .6m is NOT arrival.
  assert(!ComeArrival::arrived(3.f-.6f, 1.3f));
  float range=3.f, speed=0, prev_speed=0;
  bool slowing=false;
  for(int i=0; i<1500; ++i) {
    float target=ComeArrival::speed(range,1.3f,.4f,.35f);
    assert(target>=0 && target<=.40001f);
    float delta=fmaxf(-.35f*.02f,fminf(.35f*.02f,target-speed));
    speed+=delta;
    if(target<.39f && range<1.6f) slowing=true;
    range-=speed*.02f;
    if(ComeArrival::arrived(range,1.3f)) break;
    prev_speed=speed;
  }
  assert(slowing);
  assert(range<1.34f && range>1.28f);
  assert(speed<.10f); // ordinary arrival reaches the stop line at a crawl
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            cpp = Path(tmp)/'check.cpp'
            cpp.write_text(source)
            binary = Path(tmp)/'check'
            subprocess.run(['c++', '-std=c++11', '-I', str(root/'firmware/djr3x_motion'),
                            str(cpp), '-o', str(binary)], check=True, capture_output=True)
            subprocess.run([str(binary)], check=True, capture_output=True)
