"""Compile real firmware battery logic with fake INA226/NVS; never touch hardware."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FW = ROOT / "firmware/djr3x_motion"


class GamepadUnplugTests(unittest.TestCase):
    def test_hold_and_battery_interlock(self):
        with tempfile.TemporaryDirectory(prefix="rex-unplug-test-") as tmp:
            work = Path(tmp)
            for name in ("battery.cpp", "battery.h", "calib.h", "gamepad_unplug.h"):
                shutil.copyfile(FW / name, work / name)
            (work / "Arduino.h").write_text(r'''
#pragma once
#include <stdint.h>
#include <cstdio>
#include <cmath>
static uint32_t clock_ms = 1000;
inline uint32_t millis() { return clock_ms; }
struct SerialMock { void println(const char*) {} };
static SerialMock Serial;
''')
            (work / "Wire.h").write_text(r'''
#pragma once
#include "calib.h"
struct WireMock {
  int reg=0, writes=0, byte=0, value=0;
  float mv=13400, ma=150;
  bool current_ok=true, voltage_ok=true;
  void beginTransmission(int) { writes=0; }
  void write(int v) { if (!writes++) reg=v; }
  int endTransmission(bool=true) { return 0; }
  int requestFrom(int, int) {
    byte=0;
    if (reg==1 && !current_ok) return 0;
    if (reg==2 && !voltage_ok) return 0;
    value = reg==0xfe ? 0x5449 : reg==2 ? int(mv/1.25f) :
        int(ma*BATT_SHUNT_MICROOHM/(BATT_CURRENT_SIGN*2500.0f));
    return 2;
  }
  int read() { return byte++ ? (value & 255) : ((value >> 8) & 255); }
};
static WireMock Wire;
''')
            (work / "Preferences.h").write_text(r'''
#pragma once
struct Preferences {
  void begin(const char*, bool) {}
  float getFloat(const char*, float d) { return d; }
  bool getBool(const char*, bool) { return true; }
  void putBool(const char*, bool) {}
  void putFloat(const char*, float) {}
};
''')
            (work / "context.h").write_text(r'''
#pragma once
#include "calib.h"
#define LOCK_STATE() do {} while (0)
#define UNLOCK_STATE() do {} while (0)
inline float clampf(float x,float a,float b) { return x<a?a:x>b?b:x; }
struct Context { bool charging=false; int batt_mv=-1,batt_ma=0,batt_soc=-1; };
static Context g_ctx;
''')
            (work / "pins.h").write_text("")
            (work / "proto_io.h").write_text(r'''
#pragma once
inline void emit_log(const char*,const char*) {}
inline void emit_event_kv(const char*,const char*,const char*) {}
''')
            (work / "test.cpp").write_text(r'''
#include <cassert>
#include "battery.cpp"
#include "gamepad_unplug.h"
void tick(int n=1) { while(n--) { clock_ms+=1000; battery_tick(); } }
void lock() { battery_request_charge_assert(true); tick(); assert(g_ctx.charging); }
int main() {
  GamepadUnplugHold h;
  assert(!h.step(true,true,0));
  assert(!h.step(true,true,1999));
  assert(h.step(true,true,2000));
  assert(!h.step(true,true,8000));
  h.step(false,true,8001);
  h.step(true,true,9000);
  h.step(true,false,10999); // moving a stick restarts the hold
  assert(!h.step(true,true,11000));
  assert(!h.step(true,true,12999));
  assert(h.step(true,true,13000));
  h=GamepadUnplugHold(); // disconnect cancels the old hold
  assert(!h.step(true,true,15000));
  h=GamepadUnplugHold();
  assert(!h.step(true,true,0xffffff00u));
  assert(h.step(true,true,0x6d0u)); // unsigned clock wrap

  battery_init();
  assert(g_ctx.charging); // persisted charger latch survives reboot
  Wire.ma=-1000; Wire.mv=14200; tick(30);
  battery_request_charge_assert(false); tick();
  assert(g_ctx.charging); // still plugged in
  Wire.mv=13400; Wire.ma=150; // physically unplugged; old EMA still negative
  battery_request_charge_assert(false); tick();
  assert(!g_ctx.charging); // next tick, without EMA or 90-second wait
  tick(5); assert(!g_ctx.charging); // no stale-voltage re-lock

  lock(); Wire.ma=-300;
  battery_request_charge_assert(false); tick(); assert(g_ctx.charging);
  Wire.ma=150; Wire.mv=14200;
  battery_request_charge_assert(false); tick(); assert(g_ctx.charging);
  Wire.mv=13400; Wire.current_ok=false;
  battery_request_charge_assert(false); tick(); assert(g_ctx.charging);
  Wire.current_ok=true; Wire.voltage_ok=false;
  battery_request_charge_assert(false); tick(); assert(g_ctx.charging);
  Wire.voltage_ok=true;
  s_ma_ema=150; s_mv_ema=13400; s_chg_ticks=0;
  tick(89); assert(g_ctx.charging); // automatic path retains the long guard
  Wire.current_ok=false; tick(); assert(g_ctx.charging); // no stale proof
  Wire.current_ok=true; tick(89); assert(g_ctx.charging);
  tick(); assert(!g_ctx.charging);
}
''')
            binary = work / "test"
            subprocess.run(["clang++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                            "-I", str(work), str(work / "test.cpp"), "-o", str(binary)],
                           check=True, capture_output=True, text=True)
            result = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
