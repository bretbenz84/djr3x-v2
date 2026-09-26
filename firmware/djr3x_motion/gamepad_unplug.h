// Deliberate operator declaration, not automatic cable detection.
#pragma once
#include <stdint.h>

struct GamepadUnplugHold {
  bool holding = false;
  bool fired = false;
  uint32_t since = 0;

  bool step(bool chord, bool neutral, uint32_t now) {
    if (!chord) { holding = false; fired = false; return false; }
    if (!neutral) { holding = false; return false; }
    if (fired) return false;  // must release the shoulders before retrying
    if (!holding) { holding = true; since = now; }
    if ((uint32_t)(now - since) < 2000u) return false;
    fired = true;
    return true;
  }
};
