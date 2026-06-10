/*
 * diag_head.ino — TEST A: bright speak wave with serial REMOVED entirely.
 *
 * Replicates the production firmware's speak animation exactly — same zone
 * table, same wave math, same 50 fps frame cap (SPEAK_FRAME_MS 20), same
 * bright "excited" yellow, eyes lit solid gold — but with NO Serial.begin(),
 * no UART interrupts, no command parsing, no host. The audio level that
 * normally arrives via SPEAK_LEVEL is synthesized on-board (slow sine,
 * 180-255, updated every 33 ms like real TTS levels).
 *
 * Interpretation:
 *   - Still flickers yellow → purple/blue: genuine LED-output corruption
 *     (electrical / strip timing). Serial is exonerated; chase the hardware
 *     path with a meter.
 *   - Clean: the flicker was never LED corruption — it is mangled serial
 *     input being rendered faithfully. The fix moves entirely to the serial
 *     link (drops during show(), parsing robustness).
 *
 * Restore the real firmware afterwards:
 *   arduino-cli upload -p /dev/cu.usbmodem1301 --fqbn arduino:avr:uno arduino/head_nano
 */

#include <FastLED.h>
#include <math.h>

#define DATA_PIN    6
#define NUM_EYES    2
#define NUM_MOUTH   80
#define NUM_LEDS    (NUM_EYES + NUM_MOUTH)
#define MOUTH_START NUM_EYES
#define NUM_ZONES   5

// ── Identical to production head_nano.ino ──────────────────────────────────
const uint8_t PIXEL_ZONE[NUM_MOUTH] PROGMEM = {
    4, 4, 4, 4, 4, 4, 4, 4,
    4, 3, 3, 3, 3, 3, 3, 4,
    3, 3, 2, 2, 2, 2, 3, 3,
    3, 2, 1, 1, 1, 1, 2, 3,
    3, 2, 1, 0, 0, 1, 2, 3,
    3, 2, 1, 0, 0, 1, 2, 3,
    3, 2, 1, 1, 1, 1, 2, 3,
    3, 3, 2, 2, 2, 2, 3, 3,
    4, 3, 3, 3, 3, 3, 3, 4,
    4, 4, 4, 4, 4, 4, 4, 4,
};

CRGB leds[NUM_LEDS];

// "excited" yellow in wire order (GRB mouth driven via RGB declaration):
// physical (R=255, G=200, B=0) → stored (200, 255, 0). Same as production.
const uint8_t MC_R = 200, MC_G = 255, MC_B = 0;

#define SPEAK_LEAD     0.30f
#define SPEAK_WINDOW   1.70f
#define SPEAK_FRAME_MS 20          // same 50 fps cap as production

uint8_t  speakLevel       = 220;
float    speakPhase       = 0.0f;
uint32_t lastMs           = 0;
uint32_t lastSpeakFrameMs = 0;
uint32_t lastLevelMs      = 0;
float    levelPhase       = 0.0f;

void tickSpeak(float dt) {
    float speed = 1.5f + (speakLevel / 255.0f) * 6.5f;
    speakPhase += speed * dt;
    if (speakPhase >= (float)NUM_ZONES) speakPhase -= (float)NUM_ZONES;

    uint32_t now = millis();
    if ((uint32_t)(now - lastSpeakFrameMs) < SPEAK_FRAME_MS) return;
    lastSpeakFrameMs = now;

    float peak    = 0.30f + (speakLevel / 255.0f) * 0.70f;
    float ambient = 0.12f;

    for (uint8_t i = 0; i < NUM_MOUTH; i++) {
        float zone = (float)pgm_read_byte(&PIXEL_ZONE[i]);
        uint8_t ledIdx = i + MOUTH_START;
        float diff = speakPhase - zone;
        if (diff < -SPEAK_LEAD) diff += (float)NUM_ZONES;
        float pulse = 0.0f;
        if (diff >= -SPEAK_LEAD && diff <= (SPEAK_WINDOW - SPEAK_LEAD)) {
            pulse = sin(PI * (diff + SPEAK_LEAD) / SPEAK_WINDOW);
            if (pulse < 0.0f) pulse = 0.0f;
        }
        float brightness = ambient + pulse * peak;
        if (brightness > 1.0f) brightness = 1.0f;
        uint8_t sc = (uint8_t)(brightness * 255.0f);
        leds[ledIdx] = CRGB(scale8(MC_R, sc), scale8(MC_G, sc), scale8(MC_B, sc));
    }
    FastLED.show();
}

void setup() {
    // NOTE: no Serial.begin() anywhere — the UART stays disabled for the
    // whole test. Any flicker seen here cannot be serial-related.
    FastLED.addLeds<WS2812B, DATA_PIN, RGB>(leds, NUM_LEDS);
    FastLED.setBrightness(255);
    FastLED.setDither(0);
    delay(100);
    FastLED.clear();
    FastLED.show();
    delay(10);
    FastLED.show();

    // Eyes solid gold, set once — like ACTIVE during speech (blink omitted to
    // keep the test single-variable).
    leds[0] = CRGB(255, 200, 0);
    leds[1] = CRGB(255, 200, 0);
    FastLED.show();

    lastMs = millis();
}

void loop() {
    uint32_t now = millis();

    // Synthesize loud TTS audio levels: 180-255 sine, stepped every 33 ms —
    // the same range and update rate the host sends during the bright wave.
    if ((uint32_t)(now - lastLevelMs) >= 33) {
        lastLevelMs = now;
        levelPhase += 0.3f;
        if (levelPhase >= TWO_PI) levelPhase -= TWO_PI;
        speakLevel = (uint8_t)(217.5f + 37.5f * sinf(levelPhase));
    }

    float dt = (now - lastMs) * 0.001f;
    lastMs = now;
    if (dt > 0.1f) dt = 0.1f;

    tickSpeak(dt);
}
