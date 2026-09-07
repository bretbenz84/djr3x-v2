"""Run the real firmware heading controller through obstacle/return trajectories."""
import subprocess
import tempfile
import unittest
from pathlib import Path


class TravelHeadingTest(unittest.TestCase):
    def test_compiled_controller_trajectories(self):
        root = Path(__file__).resolve().parents[1]
        source = r'''
#include "travel_heading.h"
#include <cassert>
int main() {
  for (float sign : {-1.f, 1.f}) {
    TravelHeading h;
    float yaw = 0;
    // Obstacle correction must win, even as we turn farther off course.
    for (int i=0; i<100; ++i) {
      float w = h.correction(yaw, true, .02f, false, sign*.20f);
      assert(fabsf(w-sign*.20f) < .0001f);
      yaw += w*.02f;
    }
    assert(fabsf(yaw) > .39f);
    // Brief clear gaps cannot begin steering back into the obstacle.
    for (int i=0; i<10; ++i) assert(h.correction(yaw,true,.02f,true,0)==0);
    assert(h.correction(yaw,true,.02f,false,sign*.2f)==sign*.2f);
    // Once clear, return smoothly, with no change of destination bearing.
    for (int i=0; i<500; ++i) {
      float w=h.correction(yaw,true,.02f,true,0);
      assert(fabsf(w)<=.25001f);
      yaw+=w*.02f;
    }
    assert(fabsf(yaw)<.036f);
  }
  TravelHeading wrap;
  wrap.correction(3.13f,true,.02f,false,0);
  float w=0;
  for(int i=0;i<30;++i) w=wrap.correction(-3.05f,true,.02f,true,0);
  assert(w<0 && w>-.2f); // shortest route across +/-pi
  TravelHeading lost;
  lost.correction(0,true,.02f,false,.2f);
  lost.correction(.3f,false,.02f,true,0);
  for(int i=0;i<100;++i) assert(lost.correction(.3f,true,.02f,true,0)==0);
  assert(lost.correction(.3f,false,.02f,false,.2f)==.2f);
  TravelHeading narrow;
  narrow.correction(0,true,.02f,false,0);
  for(int i=0;i<100;++i) assert(narrow.correction(.4f,true,.02f,false,0)==0);
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            cpp = Path(tmp)/'check.cpp'
            cpp.write_text('#include <initializer_list>\n' + source)
            binary = Path(tmp)/'check'
            subprocess.run(['c++', '-std=c++11', '-I', str(root/'firmware/djr3x_motion'),
                            str(cpp), '-o', str(binary)], check=True, capture_output=True)
            subprocess.run([str(binary)], check=True, capture_output=True)
