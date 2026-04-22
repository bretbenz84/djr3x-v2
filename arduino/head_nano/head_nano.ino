/*
 * head_nano.ino — DJ-R3X Head LED Board (v2)
 *
 * Hardware
 * --------
 *   82 WS2812B NeoPixels on D6 (FastLED)
 *   Pixels 0–1   : left and right eyes
 *   Pixels 2–81  : mouth (80-pixel trapezoid PCB)
 *
 * Mouth layout — 10 rows × 8 cols, serpentine wiring
 * ---------------------------------------------------
 *   Even rows (0,2,4,6,8) wire left→right.
 *   Odd  rows (1,3,5,7,9) wire right→left.
 *   Physical center of the array is at grid position (row=4.5, col=3.5).
 *
 *   Centre cluster (zone 0): pixels 37, 38, 45, 46 (mouth offset +2)
 *     37 = row4 col3   38 = row4 col4
 *     45 = row5 col4*  46 = row5 col3*   (* serpentine reversal)
 *
 *   Zones by Euclidean distance from (4.5, 3.5):
 *     Zone 0  dist < 1.0   — centre cluster        ( 4 pixels)
 *     Zone 1  dist < 2.4   — inner ring             (12 pixels)
 *     Zone 2  dist < 3.2   — middle ring            (16 pixels)
 *     Zone 3  dist < 4.5   — outer ring             (28 pixels)
 *     Zone 4  dist ≥ 4.5   — outermost edges        (20 pixels)
 *
 * Serial protocol — 115200 baud, ASCII, newline-terminated
 * ---------------------------------------------------------
 *   SPEAK:{emotion}       Start speaking animation.
 *                         emotion = neutral | happy | excited | sad | angry
 *   SPEAK_LEVEL:{0-255}   Update audio intensity — drives pulse speed + brightness.
 *                         Send as often as needed; non-blocking.
 *   SPEAK_STOP            Mouth off immediately; eyes unchanged; blinking
 *                         suspended until next EYE: or ACTIVE command.
 *   IDLE                  Mouth off; eyes breathe slowly at last EYE: colour;
 *                         blink system activates (or stays active) immediately.
 *   ACTIVE                Mouth off; preserve the current eye colour and
 *                         resume blinking. Falls back to bright white only
 *                         if no eye colour has been set yet.
 *   EYE:{r},{g},{b}       Set both eyes to RGB colour; blinking resumes.
 *   OFF                   All 82 pixels off immediately; blinking suspended.
 */

#include <FastLED.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Pin / layout constants
// ---------------------------------------------------------------------------

#define DATA_PIN    6
#define NUM_EYES    2
#define NUM_MOUTH   80
#define NUM_LEDS    (NUM_EYES + NUM_MOUTH)   // 82 total; eyes first, mouth second
#define MOUTH_START NUM_EYES                 // mouth pixels begin at index 2
#define NUM_ZONES   5

#define BAUD_RATE  115200
#define SERIAL_BUF 64

// ---------------------------------------------------------------------------
// Zone lookup table  (stored in flash — saves ~80 bytes of SRAM)
// ---------------------------------------------------------------------------
//
// Layout:   10 rows × 8 pixels.  Rows alternate L→R / R→L (serpentine).
// Symmetry: the table is symmetric top↔bottom and left↔right, which gives
//           the concentric diamond / ellipse pattern radiating from centre.
//
//   Row 0  top edge  (even, L→R)
//   Row 1            (odd,  R→L : phys cols 7→0)
//   Row 2            (even, L→R)
//   Row 3            (odd,  R→L)
//   Row 4  ← zone-0 pixels 35,36 are at this row, cols 3 & 4
//   Row 5  ← zone-0 pixels 43,44 are at this row, cols 4 & 3 (serpentine)
//   Row 6
//   Row 7
//   Row 8
//   Row 9  bottom edge (odd, R→L)

const uint8_t PIXEL_ZONE[NUM_MOUTH] PROGMEM = {
    4, 4, 4, 4, 4, 4, 4, 4,   // row 0 — top edge
    4, 3, 3, 3, 3, 3, 3, 4,   // row 1
    3, 3, 2, 2, 2, 2, 3, 3,   // row 2
    3, 2, 1, 1, 1, 1, 2, 3,   // row 3
    3, 2, 1, 0, 0, 1, 2, 3,   // row 4  ← pixels 35,36 = zone 0
    3, 2, 1, 0, 0, 1, 2, 3,   // row 5  ← pixels 43,44 = zone 0
    3, 2, 1, 1, 1, 1, 2, 3,   // row 6
    3, 3, 2, 2, 2, 2, 3, 3,   // row 7
    4, 3, 3, 3, 3, 3, 3, 4,   // row 8
    4, 4, 4, 4, 4, 4, 4, 4,   // row 9 — bottom edge
};

// ---------------------------------------------------------------------------
// Emotion colour table
// ---------------------------------------------------------------------------

struct EmotionColor { uint8_t r, g, b; };

#define EMO_NEUTRAL  0
#define EMO_HAPPY    1
#define EMO_EXCITED  2
#define EMO_SAD      3
#define EMO_ANGRY    4
#define EMO_COUNT    5

// IMPORTANT — mouth colour encoding (leds[2..81]):
// The eye pixels (leds[0-1]) are RGB-ordered LEDs; the mouth PCB uses
// GRB-ordered WS2812B.  Both share one data line so FastLED uses a single
// colour order (RGB, matching the eyes).  For mouth pixels this means the
// first wire byte is interpreted by the GRB strip as GREEN, not RED.
// All mouth colours must therefore have R↔G swapped relative to the
// intended physical colour:
//   physical (R, G, B) on GRB mouth strip → store as EmotionColor { G, R, B }
const EmotionColor EMOTION_COLORS[EMO_COUNT] PROGMEM = {
    { 140, 255,   0 },   // neutral  — warm amber   (physical R=255 G=140 B=0 → swap → 140,255,0)
    { 200,   0, 255 },   // happy    — cyan blue    (physical R=0   G=200 B=255 → swap → 200,0,255)
    { 200, 255,   0 },   // excited  — yellow       (physical R=255 G=200 B=0 → swap → 200,255,0)
    {   0,  40, 200 },   // sad      — blue-purple  (physical R=40  G=0   B=200 → swap → 0,40,200)
    {   0, 255,   0 },   // angry    — red          (physical R=255 G=0   B=0 → swap → 0,255,0)
};

// ---------------------------------------------------------------------------
// LED array
// ---------------------------------------------------------------------------

CRGB leds[NUM_LEDS];

// ---------------------------------------------------------------------------
// Animation state
// ---------------------------------------------------------------------------

enum AnimMode : uint8_t {
    ANIM_OFF,
    ANIM_SPEAK,
    ANIM_IDLE,
    ANIM_ACTIVE,
    ANIM_SLEEP,
};

AnimMode animMode = ANIM_OFF;

// Speaking state
EmotionColor speakColor  = { 255, 140, 0 };  // default: neutral amber
uint8_t      speakLevel  = 0;                 // 0–255 audio intensity
float        speakPhase  = 0.0f;              // wave front 0.0 – NUM_ZONES

// Idle breathing state
float        idlePhase   = 0.0f;              // 0.0 – TWO_PI

// Eye brightness scale for idle breathing (0.0–1.0).
// tickIdle() updates this continuously.  Non-idle modes reset it to 1.0 via
// setEyes() so full colour is restored.  tickBlink() reads this to restore
// the correct mid-breath level after a blink, and saves it at blink-start.
float        eyeBrightness      = 1.0f;
float        blinkSavedBrightness = 1.0f;     // captured at blink-start

uint32_t     lastMs      = 0;

// ---------------------------------------------------------------------------
// Eye blink state machine
// ---------------------------------------------------------------------------
//
// Rex blinks at random human-like intervals (2–8 s) with a 100–400 ms closed
// duration.  10 % of blinks are double-blinks: eyes reopen briefly (200–400 ms)
// then close again for a second blink before returning to normal.
//
// Three non-blocking states driven by millis():
//
//   BLINK_OPEN        — eyes showing eyeColor; waiting for next blink interval
//   BLINK_CLOSED      — eyes dark for blinkDuration ms
//   BLINK_DOUBLE_WAIT — eyes restored; short pause before the second blink
//
// eyeColor tracks the *intended* eye colour so a blink always restores to the
// current colour even if an EYE: command arrives mid-blink.
//
// eyesActive = false suspends all blinking.  Set false by OFF and SPEAK_STOP;
// set true by EYE: (non-black) and ACTIVE.  The eyes remain physically visible
// in leds[] when blinking is suspended — only new blink triggers are blocked.
//
// Set BLINK_ENABLED = false to freeze eyes open for debugging.

bool BLINK_ENABLED = true;

enum BlinkState : uint8_t {
    BLINK_OPEN,
    BLINK_CLOSED,
    BLINK_DOUBLE_WAIT,
};

CRGB       eyeColor      = CRGB::Black;  // intended colour; restored after blink
bool       eyesActive    = false;        // false → blink triggers suspended
BlinkState blinkState    = BLINK_OPEN;
bool       isSecondBlink = false;        // true during the 2nd leg of a double-blink
uint32_t   blinkTimer    = 0;           // millis() at start of current blink state
uint32_t   blinkInterval = 4000;        // ms to wait before next blink (overwritten in setup)
uint32_t   blinkDuration = 0;           // ms eyes stay closed / paused between double blinks

// ---------------------------------------------------------------------------
// Serial
// ---------------------------------------------------------------------------

char    serialBuf[SERIAL_BUF];
uint8_t serialPos = 0;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

inline uint8_t clampByte(int v) {
    if (v < 0)   return 0;
    if (v > 255) return 255;
    return (uint8_t)v;
}

// setEyes — store the intended colour and write to the LED buffer.
//
// If a blink is in progress (BLINK_CLOSED), only eyeColor is updated; leds[]
// stays dark so the blink isn't interrupted.  tickBlink() restores eyeColor to
// leds[] when the blink ends.
//
// Transitioning from inactive (eyesActive was false) to active resets the blink
// timer so Rex doesn't blink the instant his eyes come on.
//
// NOTE: does NOT call FastLED.show() — the caller is responsible.
void setEyes(uint8_t r, uint8_t g, uint8_t b) {
    bool wasActive = eyesActive;

    eyeColor      = CRGB(r, g, b);
    eyeBrightness = 1.0f;   // non-idle modes always use full brightness
    eyesActive    = ((r | g | b) != 0);

    // Update leds[] only when eyes are not mid-blink.
    if (blinkState != BLINK_CLOSED) {
        leds[0] = eyeColor;
        leds[1] = eyeColor;
    }
    // When eyes become active for the first time (or return from an off state),
    // start a fresh blink cycle — avoids an immediate blink right after enable.
    if (eyesActive && !wasActive) {
        blinkTimer    = millis();
        blinkInterval = 2000UL + (uint32_t)random(6001);
        blinkState    = BLINK_OPEN;
        isSecondBlink = false;
    }
}

inline void mouthOff() {
    for (uint8_t i = MOUTH_START; i < NUM_LEDS; i++) leds[i] = CRGB::Black;
}

static uint8_t parseEmotion(const char *s) {
    if (strcmp(s, "happy")   == 0) return EMO_HAPPY;
    if (strcmp(s, "excited") == 0) return EMO_EXCITED;
    if (strcmp(s, "sad")     == 0) return EMO_SAD;
    if (strcmp(s, "angry")   == 0) return EMO_ANGRY;
    return EMO_NEUTRAL;
}

// ---------------------------------------------------------------------------
// Eye blink tick — call every loop()
// ---------------------------------------------------------------------------
//
// Only calls FastLED.show() when the blink state changes (i.e. rarely), so it
// does not interfere with tickSpeak()'s continuous animation rate.  tickSpeak()
// only writes mouth pixels (index 2+) and never touches leds[0]/leds[1], so
// both functions share leds[] safely without coordination.

void tickBlink() {
    if (!BLINK_ENABLED || !eyesActive) return;

    uint32_t now = millis();

    switch (blinkState) {

        case BLINK_OPEN:
            // Wait for the interval then snap eyes off to start the blink.
            // Capture the current breathing brightness so we restore to the
            // same mid-breath level rather than jumping to full eyeColor.
            if (now - blinkTimer >= blinkInterval) {
                blinkSavedBrightness = eyeBrightness;
                leds[0]       = CRGB::Black;
                leds[1]       = CRGB::Black;
                FastLED.show();
                blinkTimer    = now;
                blinkDuration = 100UL + (uint32_t)random(301);   // 100–400 ms closed
                blinkState    = BLINK_CLOSED;
            }
            break;

        case BLINK_CLOSED:
            // Hold closed, then restore eye colour at the saved breathing level.
            if (now - blinkTimer >= blinkDuration) {
                uint8_t sc = (uint8_t)(blinkSavedBrightness * 255.0f);
                leds[0]    = CRGB(scale8(eyeColor.r, sc),
                                  scale8(eyeColor.g, sc),
                                  scale8(eyeColor.b, sc));
                leds[1]    = leds[0];
                FastLED.show();
                blinkTimer = now;

                if (!isSecondBlink && (random(10) == 0)) {
                    // 10 % chance: double-blink — brief open pause, then blink again.
                    blinkDuration = 200UL + (uint32_t)random(201);  // 200–400 ms pause
                    blinkState    = BLINK_DOUBLE_WAIT;
                } else {
                    // Normal recovery: reset for next independent blink.
                    isSecondBlink = false;
                    blinkInterval = 2000UL + (uint32_t)random(6001);  // 2–8 s
                    blinkState    = BLINK_OPEN;
                }
            }
            break;

        case BLINK_DOUBLE_WAIT:
            // Eyes are open; wait the inter-blink pause, then close again.
            // Re-capture brightness for the second blink.
            if (now - blinkTimer >= blinkDuration) {
                blinkSavedBrightness = eyeBrightness;
                leds[0]       = CRGB::Black;
                leds[1]       = CRGB::Black;
                FastLED.show();
                isSecondBlink = true;                              // prevent triple-blink
                blinkTimer    = now;
                blinkDuration = 100UL + (uint32_t)random(301);   // 100–400 ms closed
                blinkState    = BLINK_CLOSED;
            }
            break;
    }
}

// ---------------------------------------------------------------------------
// Command dispatch
// ---------------------------------------------------------------------------

void handleCommand(char *cmd) {

    // SPEAK_LEVEL:{0-255}  — check before SPEAK: to avoid prefix collision
    if (strncmp(cmd, "SPEAK_LEVEL:", 12) == 0) {
        speakLevel = clampByte(atoi(cmd + 12));
        return;
    }

    // SPEAK_STOP — mouth off; eyes unchanged in leds[] but blink suspended
    // until the next EYE: or ACTIVE re-enables it.
    //
    // Intentionally idempotent: mouthOff() + FastLED.show() run unconditionally
    // even if animMode is already ANIM_OFF.  The Pi may send this command multiple
    // times as a reliability measure; redundant calls are harmless and guarantee
    // the mouth pixels reach the off state.
    if (strcmp(cmd, "SPEAK_STOP") == 0) {
        animMode   = ANIM_OFF;
        eyesActive = false;
        blinkState = BLINK_OPEN;   // reset so next activation starts cleanly
        mouthOff();
        FastLED.show();
        return;
    }

    // SPEAK:{emotion}
    if (strncmp(cmd, "SPEAK:", 6) == 0) {
        uint8_t emo = parseEmotion(cmd + 6);
        EmotionColor ec;
        memcpy_P(&ec, &EMOTION_COLORS[emo], sizeof(EmotionColor));
        speakColor = ec;
        speakPhase = 0.0f;
        animMode   = ANIM_SPEAK;
        // Eyes not touched — blink continues at current eyeColor/eyesActive state.
        return;
    }

    // IDLE — mouth off; eyes breathe slowly; blink system active.
    if (strcmp(cmd, "IDLE") == 0) {
        // Clear mouth pixels FIRST — before touching any other state — so that
        // pixels 2-81 (MOUTH_START … NUM_LEDS-1) are guaranteed black the
        // instant this command is processed, regardless of what animMode was
        // previously.  A single stale non-black value in the buffer (e.g. from
        // tickSpeak's ambient floor) would otherwise survive until the next
        // FastLED.show() writes it out.
        mouthOff();
        FastLED.show();

        animMode      = ANIM_IDLE;
        idlePhase     = 0.0f;
        eyeBrightness = 1.0f;   // tickIdle will update from here on first tick
        // Activate blink system if it was suspended (e.g. after SPEAK_STOP).
        // Only start if eyeColor is non-black — no point blinking dark eyes.
        if (!eyesActive && (eyeColor.r | eyeColor.g | eyeColor.b)) {
            eyesActive    = true;
            blinkTimer    = millis();
            blinkInterval = 2000UL + (uint32_t)random(6001);
            blinkState    = BLINK_OPEN;
            isSecondBlink = false;
        }
        return;
    }

    // ACTIVE — mouth off; preserve current eye colour; blink resumes.
    if (strcmp(cmd, "ACTIVE") == 0) {
        animMode = ANIM_ACTIVE;
        mouthOff();
        if (eyeColor.r || eyeColor.g || eyeColor.b) {
            setEyes(eyeColor.r, eyeColor.g, eyeColor.b);
        } else {
            setEyes(255, 255, 255);
        }
        FastLED.show();
        return;
    }

    // EYE:{r},{g},{b} — set eye colour; blink resumes.
    if (strncmp(cmd, "EYE:", 4) == 0) {
        int r, g, b;
        if (sscanf(cmd + 4, "%d,%d,%d", &r, &g, &b) == 3) {
            setEyes(clampByte(r), clampByte(g), clampByte(b));
            FastLED.show();
        }
        return;
    }

    // OFF — all pixels off; blink suspended until EYE: or ACTIVE.
    if (strcmp(cmd, "OFF") == 0) {
        animMode      = ANIM_OFF;
        eyeColor      = CRGB::Black;
        eyesActive    = false;
        blinkState    = BLINK_OPEN;
        isSecondBlink = false;
        FastLED.clear();
        FastLED.show();
        return;
    }

    // SLEEP — mouth pulses red (breathing); eyes off; blink suspended.
    if (strcmp(cmd, "SLEEP") == 0) {
        animMode      = ANIM_SLEEP;
        eyeColor      = CRGB::Black;
        eyesActive    = false;
        blinkState    = BLINK_OPEN;
        isSecondBlink = false;
        leds[0]       = CRGB::Black;
        leds[1]       = CRGB::Black;
        mouthOff();
        FastLED.show();
        return;
    }

    // Unknown — ignore silently
}

// ---------------------------------------------------------------------------
// Speaking pulse animation
// ---------------------------------------------------------------------------
//
// A sine-shaped wave front advances from zone 0 outward to zone 4, looping
// continuously.  Speed and peak brightness both scale with speakLevel.
//
// For each pixel at zone Z, brightness is:
//   diff = speakPhase - Z           (how far the wave has passed this zone)
//   pulse = sin(π × (diff+LEAD) / WINDOW)   for diff in [-LEAD, WINDOW-LEAD]
//
// LEAD  = 0.30  slight pre-glow before the wave arrives
// WINDOW= 1.70  total pulse width in zone units (enter 0.30 before, exit 1.40 after peak)
//
// An ambient floor (0.12) keeps the mouth dimly lit at all times while speaking.
//
// NOTE: only writes to mouth pixels (index MOUTH_START and above).
// Eye pixels leds[0] and leds[1] are left alone so tickBlink() owns them.

#define SPEAK_LEAD    0.30f
#define SPEAK_WINDOW  1.70f

void tickSpeak(float dt) {
    // Wave speed: 1.5 zones/s at level 0 → 8.0 zones/s at level 255
    float speed = 1.5f + (speakLevel / 255.0f) * 6.5f;
    speakPhase += speed * dt;
    if (speakPhase >= (float)NUM_ZONES) speakPhase -= (float)NUM_ZONES;

    // Peak brightness: 0.30 at level 0 → 1.00 at level 255
    float peak    = 0.30f + (speakLevel / 255.0f) * 0.70f;
    float ambient = 0.12f;

    for (uint8_t i = 0; i < NUM_MOUTH; i++) {
        float zone = (float)pgm_read_byte(&PIXEL_ZONE[i]);
        // Mouth pixels start at index MOUTH_START (2); zone table is 0-indexed
        uint8_t ledIdx = i + MOUTH_START;
        float diff = speakPhase - zone;

        // Wrap so waves look continuous when front passes zone 4 → zone 0
        if (diff < -SPEAK_LEAD) diff += (float)NUM_ZONES;

        float pulse = 0.0f;
        if (diff >= -SPEAK_LEAD && diff <= (SPEAK_WINDOW - SPEAK_LEAD)) {
            pulse = sin(PI * (diff + SPEAK_LEAD) / SPEAK_WINDOW);
            if (pulse < 0.0f) pulse = 0.0f;
        }

        float brightness = ambient + pulse * peak;
        if (brightness > 1.0f) brightness = 1.0f;

        uint8_t sc = (uint8_t)(brightness * 255.0f);
        leds[ledIdx] = CRGB(
            scale8(speakColor.r, sc),
            scale8(speakColor.g, sc),
            scale8(speakColor.b, sc)
        );
    }
    FastLED.show();
}

// ---------------------------------------------------------------------------
// Idle animation — slow eye breathing
// ---------------------------------------------------------------------------
//
// Eyes pulse gently between 30 % and 100 % of eyeColor using a sine wave
// (period ≈ 7.8 s at 0.8 rad/s).  eyeBrightness is updated every tick so
// tickBlink() can save the mid-breath level at blink-start and restore to
// it exactly after the blink ends, avoiding a jarring brightness jump.
//
// leds[] is NOT written during BLINK_CLOSED — tickBlink() owns the eye
// pixels while the eyes are dark, and will restore them with the saved level.
//
// IMPORTANT — mouth pixels (indices 2-81 / MOUTH_START … NUM_LEDS-1):
//   tickIdle() intentionally writes ONLY leds[0] and leds[1] (the two eyes).
//   Mouth pixels are cleared by mouthOff() in the IDLE command handler and
//   are never modified here.  Any future edit that writes a mouth pixel
//   inside tickIdle() is a bug.

void tickIdle(float dt) {
    idlePhase += 0.8f * dt;
    if (idlePhase >= TWO_PI) idlePhase -= TWO_PI;

    // Brightness: 0.30 at trough → 1.00 at peak
    eyeBrightness = 0.30f + 0.35f * (1.0f + sinf(idlePhase));

    // Let tickBlink() own leds[] while eyes are closed.
    if (blinkState == BLINK_CLOSED) return;

    // Only eye pixels — mouth pixels are never written here.
    uint8_t sc = (uint8_t)(eyeBrightness * 255.0f);
    leds[0] = CRGB(scale8(eyeColor.r, sc),
                   scale8(eyeColor.g, sc),
                   scale8(eyeColor.b, sc));
    leds[1] = leds[0];
    // leds[2] … leds[NUM_LEDS-1] (mouth) are intentionally NOT modified.
    FastLED.show();
}

// ---------------------------------------------------------------------------
// Sleep animation — slow red mouth breathing
// ---------------------------------------------------------------------------
//
// All 80 mouth pixels ramp linearly from 0 to 30 % brightness over 4 s then
// back to 0 over the next 4 s (triangle wave, 8 s period).  Peak R value is
// 76 out of 255 (≈ 30 %).  Mouth pixels are GRB-ordered; R↔G is swapped in
// all mouth writes so the physical display is red (see EMOTION_COLORS note).
//
// FastLED.show() is only called when the red byte changes (≈ every 52 ms at
// this ramp rate) — avoids hammering the WS2812B bus hundreds of times per
// second, which causes visible colour glitches.
//
// Eyes are NOT touched — they are off in SLEEP state.

void tickSleep() {
    uint32_t now   = millis();
    // Triangle wave: phase 0→0.5 ramps up, 0.5→1.0 ramps down.
    float    phase = (float)(now % 8000UL) / 8000.0f;      // 0.0 – 1.0
    float    tri   = (phase < 0.5f) ? (phase * 2.0f)
                                    : (2.0f - phase * 2.0f); // 0.0 – 1.0
    // Cap at 30 % brightness (76 / 255 ≈ 29.8 %)
    uint8_t  r     = (uint8_t)(tri * 76.0f);

    // Only push a new frame when the value has actually changed.
    // This throttles show() to ~19 calls/s and eliminates bus hammering.
    static uint8_t lastR = 255;   // sentinel: non-zero so first call always fires
    if (r == lastR) return;
    lastR = r;

    for (uint8_t i = MOUTH_START; i < NUM_LEDS; i++) {
        leds[i] = CRGB(0, r, 0);   // GRB mouth via RGB FastLED: swap R↔G → physical red
    }
    FastLED.show();
}

// ---------------------------------------------------------------------------
// Main animation tick — call every loop()
// ---------------------------------------------------------------------------

void tickAnimation() {
    if (animMode == ANIM_OFF || animMode == ANIM_ACTIVE) return;

    uint32_t now     = millis();
    float    dt      = (now - lastMs) * 0.001f;   // seconds since last tick
    lastMs           = now;

    if (dt > 0.1f) dt = 0.1f;   // clamp: ignore stalls > 100 ms (e.g. first tick)

    if (animMode == ANIM_SPEAK) { tickSpeak(dt); return; }
    if (animMode == ANIM_IDLE)  { tickIdle(dt);  return; }
    if (animMode == ANIM_SLEEP) { tickSleep();   return; }
}

// ---------------------------------------------------------------------------
// setup / loop
// ---------------------------------------------------------------------------

void setup() {
    FastLED.addLeds<WS2812B, DATA_PIN, RGB>(leds, NUM_LEDS);
    FastLED.setBrightness(255);

    // WS2812B pixels can latch random data on power-on before the first show().
    // A brief delay lets the supply voltage stabilise so the reset pulse is
    // clean, then we explicitly zero every pixel — including pixel 2 (MOUTH_START)
    // which is the first mouth pixel and the most likely to stay lit from glitch.
    delay(50);
    FastLED.clear();   // fill leds[] with CRGB::Black
    FastLED.show();    // push zeros to every pixel on the strip
    // Belt-and-suspenders: zero the buffer a second time and show again.
    // The first show() resets any latched state; the second guarantees all
    // 82 pixels — especially the mouth — start in a known-off state.
    FastLED.clear();
    FastLED.show();

    // Seed PRNG from floating analog pin for varied blink timing across reboots.
    // NOTE: analogRead() is called AFTER both show() calls so it cannot
    // interfere with the WS2812B data line timing.
    randomSeed(analogRead(A0));

    Serial.begin(BAUD_RATE);
    serialPos     = 0;
    lastMs        = millis();

    // Initialise blink state machine — first blink fires somewhere in 2–8 s.
    blinkTimer    = millis();
    blinkInterval = 2000UL + (uint32_t)random(6001);
    blinkState    = BLINK_OPEN;
    isSecondBlink = false;
}

void loop() {
    // Serial command reader — buffer until newline, then dispatch.
    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c == '\n' || c == '\r') {
            if (serialPos > 0) {
                serialBuf[serialPos] = '\0';
                handleCommand(serialBuf);
                serialPos = 0;
            }
        } else if (serialPos < SERIAL_BUF - 1) {
            serialBuf[serialPos++] = c;
        }
        // If buffer overflows, discard characters until next newline.
    }

    tickAnimation();   // mouth animation (speak wave, idle)
    tickBlink();       // eye blink state machine (all modes)
}