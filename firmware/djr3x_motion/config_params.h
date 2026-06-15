// config_params.h — runtime parameter application + clamp helpers.
#pragma once
#include <ArduinoJson.h>
#include "context.h"   // clampf/clampu live here

// Apply a `config` command. Reads known keys from `cmd`, clamps each to its
// hard cap, writes them into g_ctx.params under the state lock, and copies the
// effective (post-clamp) params into `out`. Returns true if any value was
// clamped (=> ack.reason "clamped"). Unknown keys are ignored.
bool apply_config(JsonObjectConst cmd, MotionParams& out);
