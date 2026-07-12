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
 *   Physical center of the array is at grid position (row=4.5, col=3.5):
 *   the "center row pair" is rows 4 and 5.
 *
 * Speaking animation — equalizer bars
 * ------------------------------------
 *   Eight vertical bars (one per column) open outward from the center row
 *   pair like a VU meter. Bar height tracks SPEAK_LEVEL with fast-attack /
 *   slow-decay ballistics; each column wobbles on its own phase so the bars
 *   dance independently, with a center-weighted envelope so the mouth reads
 *   as a mouth (tall in the middle, tapered at the corners). A falling
 *   "peak dot" hangs above each bar. Peak brightness is capped at
 *   SPEAK_MAX_BRIGHT (45 %) and only the center row pair keeps a dim floor —
 *   quiet passages close the mouth to a thin lit line instead of lighting
 *   the whole panel.
 *
 * Serial protocol — 115200 baud, ASCII, newline-terminated
 * ---------------------------------------------------------
 *   SPEAK:{emotion}       Start speaking animation. Also sets the mouth's
 *                         emotion colour used by the idle glow afterwards.
 *                         emotion = neutral | happy | excited | sad | angry | curious
 *   SPEAK_LEVEL:{0-255}   Update audio intensity — drives bar height + motion.
 *                         Send as often as needed; non-blocking.
 *   SPEAK_STOP            Mouth returns to the dim idle glow (current emotion
 *                         colour); eyes unchanged; blinking suspended until next
 *                         EYE: or ACTIVE command.
 *   IDLE                  Mouth idle glow; eyes breathe slowly at last EYE: colour;
 *                         blink system activates (or stays active) immediately.
 *   ACTIVE                Mouth idle glow; preserve the current eye colour and
 *                         resume blinking. Falls back to bright white only
 *                         if no eye colour has been set yet.
 *   EYE:{r},{g},{b}       Set both eyes to RGB colour; blinking resumes. If the
 *                         board was in OFF mode (e.g. fresh reboot mid-session,
 *                         re-lit by the host's eye keep-alive heartbeat), this
 *                         also re-enters ACTIVE so the mouth glow resumes.
 *   OFF                   All 82 pixels off immediately; blinking suspended.
 *
 * Mouth idle glow
 * ---------------
 *   Whenever Rex is awake and not speaking (ACTIVE / IDLE / after SPEAK_STOP),
 *   the whole mouth pulses gently between GLOW_MIN and GLOW_MAX brightness
 *   (15–25 %) of the current emotion colour, so the mouth is never fully dark
 *   while he's "on". Speaking brightens it through the existing wave animation,
 *   then it settles back to the glow. OFF / SLEEP / FADEOFF keep their existing
 *   dark / red-breathing behaviour. The glow writes ONLY mouth pixels — eyes
 *   stay exclusively owned by setEyes()/tickIdle()/tickBlink().
 *
 *   When the glow starts from a dark mouth (boot / OFF / waking from SLEEP),
 *   it ramps in from 0 over GLOW_RAMP_IN_MS (4 s) — program launch breathes
 *   the mouth in rather than snapping it on. A SPEAK: cancels the ramp. The
 *   shutdown mirror is FADEOFF, which fades the frozen frame (glowing mouth
 *   included) to black over HEAD_FADEOFF_MS (4 s).
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
#define MOUTH_COLS  8
#define MOUTH_ROWS  10
#define MOUTH_HALF  5                        // half-rows per side of the center pair

#define BAUD_RATE  115200
#define SERIAL_BUF 64

// mouthIdx — LED index for a (row, col) grid position, accounting for the
// serpentine wiring: even rows run L→R, odd rows run R→L.
static inline uint8_t mouthIdx(uint8_t row, uint8_t col) {
    uint8_t c = (row & 1) ? (uint8_t)(MOUTH_COLS - 1 - col) : col;
    return MOUTH_START + row * MOUTH_COLS + c;
}

// ---------------------------------------------------------------------------
// Emotion colour table
// ---------------------------------------------------------------------------

struct EmotionColor { uint8_t r, g, b; };

#define EMO_NEUTRAL  0
#define EMO_HAPPY    1
#define EMO_EXCITED  2
#define EMO_SAD      3
#define EMO_ANGRY    4
#define EMO_CURIOUS  5
#define EMO_COUNT    6

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
    {   0, 180, 255 },   // curious  — purple       (physical R=180 G=0   B=255 → swap → 0,180,255)
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

// Mouth colour — set by SPEAK:{emotion}; used by BOTH the speaking wave and the
// idle glow, so the mouth settles back to the same emotion colour it spoke in.
// Default matches EMOTION_COLORS[EMO_NEUTRAL] (wire-order, R↔G pre-swapped for
// the GRB mouth strip — see the EMOTION_COLORS note).
EmotionColor mouthColor  = { 140, 255, 0 };   // neutral amber (wire order)
uint8_t      speakLevel  = 0;                 // 0–255 audio intensity

// Speaking equalizer state — one bar per mouth column (see tickSpeak).
float        colHeight[MOUTH_COLS];           // smoothed bar height, 0–MOUTH_HALF half-rows
float        colPeak[MOUTH_COLS];             // falling peak-dot height per column
float        colPhase[MOUTH_COLS];            // per-column wobble phase (seeded in setup)

// Mouth idle glow (ACTIVE / IDLE / post-speech): gentle sine pulse between
// GLOW_MIN and GLOW_MAX of mouthColor (~6.3 s period at 1.0 rad/s).
// glowLastScale throttles FastLED.show() to actual brightness-byte changes
// (~8 frames/s) so the glow doesn't hammer the WS2812B bus or starve the
// serial port; 255 is a sentinel forcing a redraw on the next tick (the real
// scale never exceeds GLOW_MAX*255 = 64).
#define GLOW_MIN  0.15f
#define GLOW_MAX  0.25f
#define GLOW_RATE 1.0f                        // rad/s
float        glowPhase     = 0.0f;            // 0.0 – TWO_PI
uint8_t      glowLastScale = 255;             // sentinel: force first frame

// Glow ramp-in: when the glow starts from a dark mouth (boot / OFF, or waking
// from SLEEP), brightness ramps 0 → full glow over GLOW_RAMP_IN_MS so program
// launch breathes the mouth in instead of snapping it on.  glowRampStartMs is
// the ramp's millis() origin; 0 = no ramp active.  Speaking cancels the ramp
// (the mouth is fully lit by the wave, so the post-speech glow is full level).
// The shutdown mirror is FADEOFF, which fades the frozen frame — glowing
// mouth included — to black over HEAD_FADEOFF_MS.
#define GLOW_RAMP_IN_MS 4000
uint32_t     glowRampStartMs = 0;

inline void forceGlowRefresh() { glowLastScale = 255; }

inline void startGlowRamp()   { glowRampStartMs = millis() | 1; }  // |1: never the 0 sentinel
inline void cancelGlowRamp()  { glowRampStartMs = 0; }

// Mouth watchdog. The mouth animation (ANIM_SPEAK) is free-running and is only
// stopped by a host command (SPEAK_STOP/ACTIVE/IDLE/OFF). Those bytes can be
// DROPPED on the lossy serial link — a running mouth calls show() every frame,
// which disables interrupts and can lose inbound UART bytes, and the eye keep-
// alive heartbeat contends for the link at stop time. If every stop byte is lost
// the mouth would pulse forever. So: if we are in ANIM_SPEAK but no SPEAK or
// SPEAK_LEVEL has arrived for SPEAK_TIMEOUT_MS, self-extinguish the mouth. The
// host stops sending SPEAK_LEVEL the instant playback ends, so this auto-clears
// the mouth even if the SPEAK_STOP is never received. SPEAK_TIMEOUT_MS must exceed
// the longest real gap between SPEAK_LEVEL writes (host ~30 Hz, delta-throttled);
// 1500 ms is safe — validate on real TTS audio before lowering it.
#define SPEAK_TIMEOUT_MS 1500
uint32_t     lastSpeakActivityMs = 0;         // millis() of last SPEAK/SPEAK_LEVEL

// FADEOFF: smooth shutdown fade — freeze the current frame (eyes) and ramp master
// brightness to 0 over HEAD_FADEOFF_MS, then go dark. A lifelike "powering down".
// headFadeLastBright throttles the fade's show() calls to actual brightness-byte
// changes (255 steps over 4 s ≈ one change per 15.7 ms ≈ 64 fps — a steady,
// regular cadence). With dithering disabled, frames between byte changes are
// bit-identical, so pushing them would only hammer the WS2812B bus and drop
// inbound serial bytes (show() disables interrupts) for zero visual gain.
#define HEAD_FADEOFF_MS 4000
bool         headFading         = false;
uint8_t      headFadeLastBright = 255;
uint32_t     headFadeStartMs     = 0;
uint8_t      headFadeStartBright = 255;

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
    if (strcmp(s, "curious") == 0) return EMO_CURIOUS;
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
        lastSpeakActivityMs = millis();   // feed the mouth watchdog
        return;
    }

    // SPEAK_STOP — mouth returns to the dim idle glow at the current emotion
    // colour; eyes unchanged in leds[] but blink suspended until the next EYE:
    // or ACTIVE re-enables it.
    //
    // Intentionally idempotent: re-entering ANIM_ACTIVE + forcing a glow frame
    // is harmless if already there.  The Pi may send this command multiple
    // times as a reliability measure; the forced refresh guarantees the mouth
    // leaves the bright speak frame for the glow within one loop pass.
    if (strcmp(cmd, "SPEAK_STOP") == 0) {
        animMode   = ANIM_ACTIVE;
        eyesActive = false;
        blinkState = BLINK_OPEN;   // reset so next activation starts cleanly
        forceGlowRefresh();        // snap mouth from speak frame to glow now
        return;
    }

    // SPEAK:{emotion}
    if (strncmp(cmd, "SPEAK:", 6) == 0) {
        uint8_t emo = parseEmotion(cmd + 6);
        EmotionColor ec;
        memcpy_P(&ec, &EMOTION_COLORS[emo], sizeof(EmotionColor));
        mouthColor = ec;
        animMode   = ANIM_SPEAK;
        cancelGlowRamp();       // speech lights the mouth fully; glow resumes at full level
        lastSpeakActivityMs = millis();   // reset watchdog at utterance start
        // Eyes not touched — blink continues at current eyeColor/eyesActive state.
        return;
    }

    // IDLE — mouth idle glow; eyes breathe slowly; blink system active.
    if (strcmp(cmd, "IDLE") == 0) {
        if (animMode == ANIM_OFF || animMode == ANIM_SLEEP) {
            startGlowRamp();    // mouth was dark — breathe the glow in over 4 s
        }
        animMode      = ANIM_IDLE;
        idlePhase     = 0.0f;
        eyeBrightness = 1.0f;   // tickIdle will update from here on first tick
        forceGlowRefresh();     // repaint the mouth at glow level immediately
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

    // ACTIVE — mouth idle glow; preserve current eye colour; blink resumes.
    if (strcmp(cmd, "ACTIVE") == 0) {
        if (animMode == ANIM_OFF || animMode == ANIM_SLEEP) {
            startGlowRamp();    // mouth was dark — breathe the glow in over 4 s
        }
        animMode = ANIM_ACTIVE;
        forceGlowRefresh();
        if (eyeColor.r || eyeColor.g || eyeColor.b) {
            setEyes(eyeColor.r, eyeColor.g, eyeColor.b);
        } else {
            setEyes(255, 255, 255);
        }
        FastLED.show();
        return;
    }

    // EYE:{r},{g},{b} — set eye colour; blink resumes.  If the board is in OFF
    // mode, an EYE means "awake again" (the host's eye keep-alive heartbeat
    // re-lighting us, e.g. after a firmware reboot mid-session) — re-enter
    // ACTIVE so the mouth idle glow resumes too.  Never interrupts SPEAK /
    // IDLE / SLEEP.
    if (strncmp(cmd, "EYE:", 4) == 0) {
        if (headFading) return;   // don't re-light the eyes mid shutdown-fade
        int r, g, b;
        if (sscanf(cmd + 4, "%d,%d,%d", &r, &g, &b) == 3) {
            if (animMode == ANIM_OFF && (r | g | b)) {
                animMode = ANIM_ACTIVE;
                startGlowRamp();   // dark → awake: breathe the glow in over 4 s
                forceGlowRefresh();
            }
            setEyes(clampByte(r), clampByte(g), clampByte(b));
            FastLED.show();
        }
        return;
    }

    // FADEOFF — smooth shutdown fade of the current frame (eyes) to black, then
    // dark. Idempotent: a repeat during an in-progress fade is ignored. Freezes
    // the eyes as-is and stops the mouth; the actual ramp runs in loop().
    if (strcmp(cmd, "FADEOFF") == 0) {
        if (!headFading) {
            // Re-assert dither-off at fade start: nothing in this sketch
            // re-enables dithering today, but the dim fade tail is exactly
            // where stray dithering shows as flicker, so guarantee it here
            // rather than trusting setup() alone.
            FastLED.setDither(0);
            headFading          = true;
            headFadeStartMs     = millis();
            headFadeStartBright = FastLED.getBrightness();
            headFadeLastBright  = headFadeStartBright;  // frame already shows at this level
            animMode            = ANIM_OFF;   // stop mouth; leave eyes lit to fade
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
// Speaking animation — audio-reactive equalizer bars
// ---------------------------------------------------------------------------
//
// Eight vertical bars — one per mouth column — open outward from the center
// row pair (rows 4/5), VU-meter style.  Per frame, each column:
//
//   1. Wobbles: a per-column sine (distinct rate + phase per column, faster
//      when loud) modulates the shared audio level so the bars dance
//      independently instead of pumping in lockstep.
//   2. Is shaped: a fixed center-weighted envelope keeps the middle columns
//      tallest and the corners tapered, so the lit region reads as a mouth.
//   3. Is smoothed: fast attack / slow decay ballistics — bars snap up on a
//      syllable and settle down through the gaps, closing to a thin center
//      line during pauses.
//   4. Trails a peak dot: a dim marker rides the bar's recent maximum and
//      falls slowly, adding motion above the bars.
//
// Brightness is capped at SPEAK_MAX_BRIGHT (45 %) — the old wave hit 100 % —
// and only the center row pair keeps a dim floor, so quiet passages no
// longer light the whole panel.  The bar tip is anti-aliased (fractional
// height renders as a dimmed pixel) to keep motion smooth on a 10-row grid.
//
// NOTE: only writes to mouth pixels (index MOUTH_START and above).
// Eye pixels leds[0] and leds[1] are left alone so tickBlink() owns them.

#define SPEAK_MAX_BRIGHT 0.45f   // brightness of a fully lit bar pixel
#define SPEAK_FLOOR      0.10f   // dim floor on the center row pair only
#define SPEAK_ATTACK     18.0f   // bar rise responsiveness (per second)
#define SPEAK_DECAY       6.0f   // bar fall responsiveness (per second)
#define SPEAK_PEAK_FALL   3.5f   // peak-dot fall speed (half-rows per second)

// Center-weighted column envelope: middle columns reach full height, corner
// columns top out lower — the lit shape tapers like a mouth, not a box.
const float COL_ENV[MOUTH_COLS] = {
    0.55f, 0.75f, 0.92f, 1.0f, 1.0f, 0.92f, 0.75f, 0.55f
};

// Frame cap. Unthrottled, tickSpeak rendered + show()ed on EVERY loop pass
// (~400 fps): back-to-back frames leave only FastLED's minimum reset gap,
// which crowds the WS2812 latch window — pixels that miss the latch
// reinterpret the next frame's leading bits and flash wrong hues. It also
// kept interrupts disabled for ~2.5 ms per show, >50 % of wall time, eating
// the host's SPEAK_LEVEL bytes. 50 fps is far beyond smooth for this motion,
// leaves ≥17 ms of latch headroom per frame, and frees the UART. Bar
// ballistics use the real elapsed time between frames, so motion speed is
// independent of the render cadence.
#define SPEAK_FRAME_MS 20
uint32_t lastSpeakFrameMs = 0;

void tickSpeak() {
    // Frame-rate cap: skip the update + render + show until the next slot.
    uint32_t now = millis();
    uint32_t elapsedMs = now - lastSpeakFrameMs;
    if (elapsedMs < SPEAK_FRAME_MS) return;
    lastSpeakFrameMs = now;

    float dt = elapsedMs * 0.001f;
    if (dt > 0.1f) dt = 0.1f;   // clamp: ignore stalls (e.g. first frame after idle)

    float level = speakLevel * (1.0f / 255.0f);

    // --- Update per-column bar heights + peak dots -------------------------
    for (uint8_t c = 0; c < MOUTH_COLS; c++) {
        // Per-column wobble: distinct rate per column, faster when loud.
        colPhase[c] += (2.2f + 0.55f * c) * (0.6f + 1.4f * level) * dt;
        if (colPhase[c] >= TWO_PI) colPhase[c] -= TWO_PI;
        float wobble = 0.60f + 0.40f * sinf(colPhase[c]);

        // Target height in half-rows (0 – MOUTH_HALF from the center pair).
        float target = level * wobble * COL_ENV[c] * 5.2f;
        if (target > (float)MOUTH_HALF) target = (float)MOUTH_HALF;

        // Fast attack, slow decay.
        float rate = (target > colHeight[c]) ? SPEAK_ATTACK : SPEAK_DECAY;
        float k = rate * dt;
        if (k > 1.0f) k = 1.0f;
        colHeight[c] += (target - colHeight[c]) * k;

        // Peak dot rides the maximum, then falls slowly.
        if (colHeight[c] > colPeak[c]) {
            colPeak[c] = colHeight[c];
        } else {
            colPeak[c] -= SPEAK_PEAK_FALL * dt;
            if (colPeak[c] < 0.0f) colPeak[c] = 0.0f;
        }
    }

    // --- Render -------------------------------------------------------------
    for (uint8_t c = 0; c < MOUTH_COLS; c++) {
        uint8_t peakRow = (uint8_t)colPeak[c];
        if (peakRow > MOUTH_HALF - 1) peakRow = MOUTH_HALF - 1;
        bool showPeak = (colPeak[c] > colHeight[c] + 0.6f);

        for (uint8_t d = 0; d < MOUTH_HALF; d++) {
            // Bar fill at this half-row: >=1 full, 0–1 anti-aliased tip, <=0 off.
            float fill = colHeight[c] - (float)d;
            if (fill > 1.0f) fill = 1.0f;
            if (fill < 0.0f) fill = 0.0f;

            float bright = fill * SPEAK_MAX_BRIGHT;
            if (d == 0 && bright < SPEAK_FLOOR) bright = SPEAK_FLOOR;
            if (showPeak && d == peakRow && bright < SPEAK_MAX_BRIGHT * 0.7f) {
                bright = SPEAK_MAX_BRIGHT * 0.7f;
            }

            uint8_t sc = (uint8_t)(bright * 255.0f);
            CRGB px = CRGB(scale8(mouthColor.r, sc),
                           scale8(mouthColor.g, sc),
                           scale8(mouthColor.b, sc));
            leds[mouthIdx(4 - d, c)] = px;   // upper half
            leds[mouthIdx(5 + d, c)] = px;   // lower half
        }
    }
    FastLED.show();
}

// ---------------------------------------------------------------------------
// Mouth idle glow — dim emotional pulse while awake and not speaking
// ---------------------------------------------------------------------------
//
// All 80 mouth pixels pulse together between GLOW_MIN (15 %) and GLOW_MAX
// (25 %) of mouthColor on a slow sine (~6.3 s period at GLOW_RATE rad/s), so
// the mouth shows a soft version of the current emotion colour whenever Rex
// is on but quiet.  Runs in ANIM_ACTIVE and ANIM_IDLE.
//
// Writes ONLY mouth pixels (MOUTH_START …) and returns true when the frame
// changed — the caller decides when to FastLED.show(), so IDLE mode can fold
// the glow and the eye breathing into a single show per frame.  Throttled to
// brightness-byte changes (~8 fps); forceGlowRefresh() makes the next tick
// repaint unconditionally (used on mode entry so a stale speak frame or
// mouthOff() never lingers).
//
// NOTE: eye pixels leds[0]/leds[1] are never written here — eyes stay
// exclusively owned by setEyes()/tickIdle()/tickBlink().

bool tickMouthGlow(float dt) {
    glowPhase += GLOW_RATE * dt;
    if (glowPhase >= TWO_PI) glowPhase -= TWO_PI;

    float mid  = (GLOW_MIN + GLOW_MAX) * 0.5f;
    float amp  = (GLOW_MAX - GLOW_MIN) * 0.5f;
    float brightness = mid + amp * sinf(glowPhase);

    // Ramp-in from dark (program launch / wake): scale the glow 0 → 1 over
    // GLOW_RAMP_IN_MS, then drop the ramp state once complete.
    if (glowRampStartMs) {
        uint32_t elapsed = millis() - glowRampStartMs;
        if (elapsed >= GLOW_RAMP_IN_MS) {
            glowRampStartMs = 0;
        } else {
            brightness *= (float)elapsed / (float)GLOW_RAMP_IN_MS;
        }
    }

    uint8_t sc = (uint8_t)(brightness * 255.0f);
    if (sc == glowLastScale) return false;   // nothing visible changed
    glowLastScale = sc;

    CRGB c = CRGB(scale8(mouthColor.r, sc),
                  scale8(mouthColor.g, sc),
                  scale8(mouthColor.b, sc));
    for (uint8_t i = MOUTH_START; i < NUM_LEDS; i++) leds[i] = c;
    return true;
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
//   Mouth pixels are owned by tickMouthGlow() while awake; any edit that
//   writes a mouth pixel inside tickIdle() is a bug.
//
// Returns true when the eye frame changed — the caller decides when to
// FastLED.show() (folded with the mouth glow into one show per frame), and
// the byte-change throttle (~45 fps at this breathing rate) keeps show()
// calls — which disable interrupts and can drop inbound serial bytes — off
// the loop()'s hot path.

bool tickIdle(float dt) {
    idlePhase += 0.8f * dt;
    if (idlePhase >= TWO_PI) idlePhase -= TWO_PI;

    // Brightness: 0.30 at trough → 1.00 at peak
    eyeBrightness = 0.30f + 0.35f * (1.0f + sinf(idlePhase));

    // Let tickBlink() own leds[] while eyes are closed.
    if (blinkState == BLINK_CLOSED) return false;

    // Only push a new frame when the scaled brightness byte actually changed.
    static uint8_t lastSc = 255;
    uint8_t sc = (uint8_t)(eyeBrightness * 255.0f);
    if (sc == lastSc) return false;
    lastSc = sc;

    // Only eye pixels — mouth pixels are never written here.
    leds[0] = CRGB(scale8(eyeColor.r, sc),
                   scale8(eyeColor.g, sc),
                   scale8(eyeColor.b, sc));
    leds[1] = leds[0];
    // leds[2] … leds[NUM_LEDS-1] (mouth) are intentionally NOT modified.
    return true;
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
    if (animMode == ANIM_OFF) return;

    uint32_t now     = millis();
    float    dt      = (now - lastMs) * 0.001f;   // seconds since last tick
    lastMs           = now;

    if (dt > 0.1f) dt = 0.1f;   // clamp: ignore stalls > 100 ms (e.g. first tick)

    if (animMode == ANIM_SPEAK) {
        // Mouth watchdog backstop: if no SPEAK/SPEAK_LEVEL has arrived recently the
        // host has stopped speaking and the stop command was dropped — settle the
        // mouth into the idle glow ourselves. Only touches the mouth + animMode;
        // eyes/blink are left alone (the heartbeat keeps them lit), so this is a
        // clean stop.
        if ((uint32_t)(now - lastSpeakActivityMs) > SPEAK_TIMEOUT_MS) {
            animMode = ANIM_ACTIVE;
            forceGlowRefresh();
            return;   // glow takes over on the next tick
        }
        tickSpeak();
        return;
    }
    if (animMode == ANIM_SLEEP) { tickSleep(); return; }

    // ANIM_ACTIVE / ANIM_IDLE — mouth idle glow, plus eye breathing in IDLE.
    // Both ticks only mark the buffer; a single show() pushes the combined frame.
    bool changed = tickMouthGlow(dt);
    if (animMode == ANIM_IDLE) changed = tickIdle(dt) || changed;
    if (changed) FastLED.show();
}

// ---------------------------------------------------------------------------
// setup / loop
// ---------------------------------------------------------------------------

void setup() {
    FastLED.addLeds<WS2812B, DATA_PIN, RGB>(leds, NUM_LEDS);
    FastLED.setBrightness(255);
    // Temporal dithering OFF. FastLED's binary dithering simulates extra
    // brightness depth by alternating pixel values between frames, which only
    // looks smooth with a very fast, steady show() cadence. This sketch
    // deliberately throttles show() to value changes (glow ~8 fps, idle
    // breathing ~45 fps, fade ~64 fps), so at low brightness the dither
    // alternation is visible as flicker — worst in the dim tail of FADEOFF.
    // With dithering off, dim levels quantize to steady values instead.
    FastLED.setDither(0);

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

    // Stagger the equalizer columns' wobble phases so the bars never start
    // (or drift) in lockstep. Heights/peaks start closed.
    for (uint8_t c = 0; c < MOUTH_COLS; c++) {
        colPhase[c]  = c * 0.9f;
        colHeight[c] = 0.0f;
        colPeak[c]   = 0.0f;
    }

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

    if (headFading) {
        // Shutdown fade: freeze the current frame and ramp master brightness to 0
        // over HEAD_FADEOFF_MS, then go dark. Skip the normal ticks so nothing
        // redraws (or blinks) over the fading eyes.
        //
        // Non-blocking millis()-based step: nothing in this branch (or in the
        // serial reader above it) blocks, so the refresh cadence stays steady.
        // show() is pushed only when the computed brightness BYTE changes —
        // the linear ramp makes those changes land every ~15.7 ms (≈64 fps),
        // a tight, regular cadence. Re-showing bit-identical frames between
        // steps (the old behaviour: show() every loop pass, ~400 fps) added
        // nothing visually with dithering off and cost serial bytes, since
        // every show() disables interrupts for ~2.5 ms.
        uint32_t elapsed = millis() - headFadeStartMs;
        if (elapsed >= HEAD_FADEOFF_MS) {
            headFading = false;
            eyeColor   = CRGB::Black;
            eyesActive = false;
            animMode   = ANIM_OFF;
            FastLED.setBrightness(headFadeStartBright);  // restore for next use
            FastLED.clear();
            FastLED.show();
        } else {
            uint8_t b = (uint8_t)(
                (uint32_t)headFadeStartBright * (HEAD_FADEOFF_MS - elapsed) / HEAD_FADEOFF_MS);
            if (b != headFadeLastBright) {
                headFadeLastBright = b;
                FastLED.setBrightness(b);
                FastLED.show();   // re-show the frozen leds[] at the new level
            }
        }
        return;
    }

    tickAnimation();   // mouth animation (speak wave, idle)
    tickBlink();       // eye blink state machine (all modes)
}