// DJ Rex LED Panels
// Requires FASTLED Library - https://github.com/FastLED/FastLED

#include <FastLED.h>
#include <math.h>


// How many NeoPixels are attached
#define NUM_LEDS 98
#define BRIGHTNESS 90

// Setup the LED Matrix
#define LED_PIN    6
#define LED_TYPE    WS2811
#define COLOR_ORDER GRB
#define FRAMES_PER_SECOND  30

#define ARRAY_SIZE(A) (sizeof(A) / sizeof((A)[0]))

#define DECAYTIME 80;

// define the 4 LED block starting LED numbers
#define PanelAStart 0
#define PanelA1 PanelAStart + 12
#define PanelA2 PanelA1 + 12
#define PanelA3 PanelA2 + 4

#define PanelBStart PanelAStart + 32
#define PanelB1 PanelBStart + 8
#define PanelB2 PanelB1 + 4
#define PanelB3 PanelB2 + 4

#define PanelCStart PanelBStart + 32
#define PanelC1 PanelCStart + 20
#define PanelC2 PanelC1 + 4
#define PanelC3 PanelC2 + 4

uint16_t IntervalTime[NUM_LEDS];
unsigned long LEDMillis[NUM_LEDS];
bool LEDOn[NUM_LEDS];
const uint8_t StartLEDNum[9] = {PanelA1, PanelA2, PanelA3, PanelB1, PanelB2, PanelB3, PanelC1, PanelC2, PanelC3};
uint8_t Bar1Length = 4;
uint8_t Bar2Length = 4;
uint8_t Bar3Length = 4;

CRGB DJLEDs[NUM_LEDS];
uint8_t Brightness = 150;
#define BLOCKBRIGHTNESS 225

uint8_t LEDBrightness[NUM_LEDS] = { BRIGHTNESS,BRIGHTNESS };
uint8_t LEDMinBrightness[NUM_LEDS] = { BRIGHTNESS,BRIGHTNESS };

// Command loop processing times
unsigned long previousMillis = millis();
unsigned long interval = 5;

unsigned long LEDUpdateMillis = millis();
unsigned long LEDUpdateInterval = 20;

unsigned long FadeMillis = millis();
unsigned long FadeInterval = 0;

uint16_t DecayTime = DECAYTIME;

#define cRED 0xFF0000
#define cBLUE 0x0000FF
#define cWHITE 0xFFFFFF
#define cGOLD 0xFFDD88

#define cRED2 0x500000
#define cBLUE2 0x000055
#define cWHITE2 0x608888

const CRGB SmallLEDColors[9] = { cRED2, cRED2, cWHITE2, cWHITE2, cBLUE2, cBLUE2, cRED2, cWHITE2, cWHITE2 };
const CRGB BlockLEDColors[4] = { cRED, cWHITE, cGOLD, cBLUE };

byte LEDIndex = 0;
bool inout = 0;
byte State = 0;

// ---------------------------------------------------------------------------
// Serial command parser
// ---------------------------------------------------------------------------
#define SERIAL_BUF 32
char    serialBuf[SERIAL_BUF];
uint8_t serialPos = 0;

// ---------------------------------------------------------------------------
// Chest mode
// ---------------------------------------------------------------------------
enum ChestMode : uint8_t {
    CM_STARTUP,       // host-requested: ShortCircuit once, then auto-switch to IDLE
    CM_IDLE,          // default: RandomBlocks2 at normal brightness
    CM_ACTIVE,        // Rex awake: RandomBlocks2 brighter
    CM_SPEAK_NEUTRAL, // speaking neutral: RandomBlocks2
    CM_SPEAK_EXCITED, // speaking excited: racing gold chases + rapid colour pops
    CM_SPEAK_SAD,     // speaking sad: slow blue sighs (draining bars, breathing blocks)
    CM_SPEAK_ANGRY,   // speaking angry: red alert (scanner bars + alarm blocks + slams)
    CM_SPEAK_HAPPY,   // speaking happy: bouncing gold heads + cheery pops + confetti
    CM_COMPLIMENT,    // reacting to a compliment: blue glow + gold/white sparkles (self-ends ~2.5s)
    CM_FADEOFF,       // shutdown: smoothly fade the current frame to black, then OFF
    CM_SLEEP,         // sleep: very dim slow red breath
    CM_CHARGE,        // Rex off: 3x8 SOC gauge; upward pulse while charging
    CM_OFF,           // all off
    CM_MANUAL,        // NEXT command: cycle gPatterns[] manually
};
ChestMode chestMode = CM_OFF;
uint32_t complimentStartMs = 0;   // millis() when CM_COMPLIMENT began (self-timeout)
uint8_t chargeSoc = 0;             // 0..100, supplied by off-state battery monitor
bool chargeConnected = false;

// FADEOFF: freeze the last rendered frame and ramp master brightness to 0 over
// CHEST_FADEOFF_MS, then go fully OFF — a lifelike "powering down" fade.
#define CHEST_FADEOFF_MS 4000
uint32_t chestFadeStartMs    = 0;
uint8_t  chestFadeStartBright = 0;

// setup() function -- runs once at startup --------------------------------

void setup() {

	Serial.begin(115200);

	//DataSetup();

	pinMode(13, OUTPUT);
	digitalWrite(13, HIGH);

	// tell FastLED about the LED strip configuration
	FastLED.addLeds<LED_TYPE, LED_PIN, COLOR_ORDER>(DJLEDs, NUM_LEDS).setCorrection(TypicalLEDStrip);


	// set master brightness control
	FastLED.setBrightness(BRIGHTNESS);
	FastLED.clear();
	FastLED.show();

	randomSeed(analogRead(0));
	// Seed the Array
	for (byte x = 0; x < NUM_LEDS; x++) {
		IntervalTime[x] = random16(3000);
		LEDMillis[x] = millis();
		LEDOn[x] = 0;
	}

	// Initialise serial command buffer.
	serialPos = 0;

	// Stay dark until the host explicitly sends STARTUP. This keeps the panels
	// off after a serial disconnect/reset during program shutdown.
	DecayTime = 80;
	FadeInterval = 0;
	FadeMillis = millis();
	chestMode = CM_OFF;
}

// List of patterns to cycle through.  Each is defined as a separate function below.
typedef void (*SimplePatternList[])();
SimplePatternList gPatterns = {
  LEDsOff,	// 0
  RandomBlocks2,
  AllRed,	// 2
  AllGreen,	//3
  ShortCircuit,	// 4
  ConfettiRedWhite, // 10
  rainbow,
  rainbowWithGlitter,
  confetti,
  juggle,
  bpm
};

uint8_t gCurrentPatternNumber = 1; // Index number of which pattern is current



uint8_t gHue = 0; // rotating "base color" used by many of the patterns
uint8_t gSat = 0; // saturation value
bool updown = 0;

// ---------------------------------------------------------------------------
// Mode dispatcher — called every loop tick
// ---------------------------------------------------------------------------
//
// CM_STARTUP: runs ShortCircuit() directly; detects completion when ShortCircuit
//             internally sets gCurrentPatternNumber = 0, then switches to CM_IDLE.
// CM_MANUAL:  uses gPatterns[gCurrentPatternNumber]() so NEXT still works.
// All other modes call a specific pattern function, with FastLED.setBrightness
// set at command time (in handleCommand).

void runCurrentMode() {
	switch (chestMode) {
		case CM_STARTUP:
			ShortCircuit();
			if (gCurrentPatternNumber == 0) {
				// ShortCircuit completed — switch to idle
				chestMode = CM_IDLE;
				gCurrentPatternNumber = 1;
				FastLED.setBrightness(BRIGHTNESS);
			}
			break;

		case CM_IDLE:
		case CM_SPEAK_NEUTRAL:
			RandomBlocks2();
			break;

		case CM_ACTIVE:
			RandomBlocks2();
			break;

		case CM_SPEAK_EXCITED:
			excitedPulse();
			break;

		case CM_SPEAK_SAD:
			sadSigh();
			break;

		case CM_SPEAK_ANGRY:
			angryFlash();
			break;

		case CM_SPEAK_HAPPY:
			happyBounce();
			break;

		case CM_COMPLIMENT:
			complimentFlash();
			// Self-terminate so a missed follow-up command can't leave the panel
			// stuck flashing; hand back to the awake/active pattern.
			if (millis() - complimentStartMs > 2500UL) {
				chestMode = CM_ACTIVE;
				FastLED.setBrightness(200);
			}
			break;

		case CM_FADEOFF: {
			// Freeze the last frame (don't redraw a pattern) and ramp brightness
			// down with elapsed time for a smooth, frame-rate-independent fade.
			uint32_t elapsed = millis() - chestFadeStartMs;
			if (elapsed >= CHEST_FADEOFF_MS) {
				FastLED.clear();
				chestMode = CM_OFF;
			} else {
				FastLED.setBrightness(
					(uint8_t)((uint32_t)chestFadeStartBright * (CHEST_FADEOFF_MS - elapsed) / CHEST_FADEOFF_MS));
			}
			break;
		}

		case CM_SLEEP:
			sleepBreath();
			break;

		case CM_CHARGE:
			chargeGauge();
			break;

		case CM_OFF:
			LEDsOff();
			break;

		case CM_MANUAL:
			gPatterns[gCurrentPatternNumber]();
			break;
	}
}

// loop() function -- runs repeatedly as long as board is on ---------------

void loop() {
	readSerial();

	if (millis() - previousMillis > interval) {
		previousMillis = millis();
		// Call the current mode function, updating the 'leds' array
		runCurrentMode();
		// RandomEyes only in active modes — skip during sleep/off and while fading
		// (the fade freezes the last frame, so nothing should redraw onto it).
		if (chestMode != CM_SLEEP && chestMode != CM_OFF && chestMode != CM_FADEOFF) RandomEyes();
	}

	if (millis() - LEDUpdateMillis > LEDUpdateInterval) {

		LEDUpdateMillis = millis();
		FastLED.show();
	}


	// do some periodic updates
	EVERY_N_MILLISECONDS(20) {
		gHue++;  // slowly cycle the "base color" through the rainbow
		if (updown) {
			gSat++;
			if (gSat == 255) updown = 0;
		}
		else {
			gSat--;
			if (gSat == 0) updown = 1;
		}
	}
	// EVERY_N_SECONDS( 5 ) { nextPattern(); } // change patterns periodically

}

void SetMode(uint16_t mode)
{

	switch (mode)
	{
	case 0: //  LEDs off
		gCurrentPatternNumber = 0;
		break;
	case 1: //  LEDs default
		gCurrentPatternNumber = 1;
		break;

	case 99:
		nextPattern();
		break;
	default:
		break;
	}

}

void nextPattern()
{
	// add one to the current pattern number, and wrap around at the end
	gCurrentPatternNumber = (gCurrentPatternNumber + 1) % ARRAY_SIZE(gPatterns);
	// skip 0
	if (gCurrentPatternNumber == 0) gCurrentPatternNumber = 1;
}

// ---------------------------------------------------------------------------
// Serial command handler
// ---------------------------------------------------------------------------
//
// Commands (newline-terminated, 115200 baud):
//   STARTUP          — play ShortCircuit once then switch to RandomBlocks2
//   IDLE             — RandomBlocks2 at normal brightness (default)
//   ACTIVE           — RandomBlocks2 at higher brightness
//   SPEAK:{emotion}  — emotion-specific pattern:
//                        neutral  → RandomBlocks2
//                        excited  → racing gold chases + rapid colour pops, full brightness
//                        sad      → slow blue sighs, dim
//                        angry    → red alert (scanner bars, alarm blocks, slams)
//                        happy    → bouncing gold heads + cheery pops + confetti
//   SPEAK_STOP       — return to IDLE (end of speech)
//   SLEEP            — very dim slow red breathing pulse
//   CHARGE:{soc}:{0|1} — 3x8 SOC gauge; final field says charger attached
//   OFF              — all LEDs off
//   NEXT             — cycle to next pattern in gPatterns[]

void handleCommand(char *cmd) {
	if (strcmp(cmd, "STARTUP") == 0) {
		DecayTime = 80;
		FadeInterval = 0;
		FadeMillis = millis();
		gCurrentPatternNumber = 1;
		FastLED.setBrightness(BRIGHTNESS);
		chestMode = CM_STARTUP;

	} else if (strcmp(cmd, "IDLE") == 0 || strcmp(cmd, "SPEAK_STOP") == 0) {
		FastLED.setBrightness(BRIGHTNESS);
		chestMode = CM_IDLE;

	} else if (strcmp(cmd, "ACTIVE") == 0) {
		FastLED.setBrightness(200);
		chestMode = CM_ACTIVE;

	} else if (strncmp(cmd, "SPEAK:", 6) == 0) {
		const char *emotion = cmd + 6;
		if (strcmp(emotion, "excited") == 0) {
			FastLED.setBrightness(255);
			chestMode = CM_SPEAK_EXCITED;
		} else if (strcmp(emotion, "sad") == 0) {
			FastLED.setBrightness(55);
			chestMode = CM_SPEAK_SAD;
		} else if (strcmp(emotion, "angry") == 0) {
			FastLED.setBrightness(255);
			chestMode = CM_SPEAK_ANGRY;
		} else if (strcmp(emotion, "happy") == 0) {
			FastLED.setBrightness(200);   // match ACTIVE's energy — happy shouldn't be dimmer than idle chat
			chestMode = CM_SPEAK_HAPPY;
		} else {
			// neutral or unknown emotion
			FastLED.setBrightness(BRIGHTNESS);
			chestMode = CM_SPEAK_NEUTRAL;
		}

	} else if (strcmp(cmd, "SLEEP") == 0) {
		FastLED.setBrightness(BRIGHTNESS);
		chestMode = CM_SLEEP;

	} else if (strncmp(cmd, "CHARGE:", 7) == 0) {
		int soc = 0;
		int connected = 0;
		if (sscanf(cmd + 7, "%d:%d", &soc, &connected) == 2) {
			chargeSoc = (uint8_t)constrain(soc, 0, 100);
			chargeConnected = connected != 0;
			FastLED.setBrightness(55);  // visible while off, never room-filling
			chestMode = CM_CHARGE;
		}

	} else if (strcmp(cmd, "OFF") == 0) {
		FastLED.setBrightness(BRIGHTNESS);
		chestMode = CM_OFF;
		FastLED.clear();
		FastLED.show();

	} else if (strcmp(cmd, "NEXT") == 0) {
		nextPattern();
		chestMode = CM_MANUAL;

	} else if (strcmp(cmd, "COMPLIMENT") == 0) {
		FastLED.setBrightness(200);
		complimentStartMs = millis();
		chestMode = CM_COMPLIMENT;

	} else if (strcmp(cmd, "FADEOFF") == 0) {
		// Smoothly fade the current frame to black (shutdown). Idempotent: a
		// repeat during an in-progress fade is ignored so the fade isn't restarted.
		if (chestMode != CM_FADEOFF) {
			chestFadeStartMs     = millis();
			chestFadeStartBright = FastLED.getBrightness();
			chestMode = CM_FADEOFF;
		}
	}
	// Unknown commands are silently ignored.
}

// Physical pixel for a logical gauge level, where level 0 is the BOTTOM of the
// column and level 7 the top. Each 8-pixel panel is wired TOP-DOWN — pixel 0 is the
// topmost LED — so filling from index 0 lit the gauge from the top and emptied
// downward, backwards for a battery (owner 2026-07-24: "the LEDs are inverted. The
// top are showing solid while the bottom are off"). Flip GAUGE_BOTTOM_UP to 0 if the
// panels are ever rewired the other way.
#define GAUGE_BOTTOM_UP 1
static inline uint8_t gaugePixel(uint8_t colStart, uint8_t level) {
	return colStart + (GAUGE_BOTTOM_UP ? (uint8_t)(7 - level) : level);
}

// Fixed colour ladder for the 8-pixel gauge, bottom (level 0) to top (level 7):
// red -> red/orange -> orange -> yellow -> yellow/green -> green -> green/blue ->
// blue (owner spec 2026-07-24). The half-steps are midpoint blends of their two
// neighbours, so the column ramps smoothly instead of stepping between five flat
// bands. Each pixel's colour is a property of its POSITION, not of the current
// charge — the charge only decides how many are lit, so the bar always reads the
// same way and the level is judged by where it stops.
// A switch, not a CRGB[8] table: this sketch sits at 89% DRAM and an array would
// spend 24 bytes of it, while the switch costs flash only.
static CRGB gaugeLevelColor(uint8_t level) {
	switch (level) {
		case 0:  return CRGB(130,   0,   0);   // red
		case 1:  return CRGB(130,  22,   0);   // red/orange
		case 2:  return CRGB(130,  45,   0);   // orange
		case 3:  return CRGB(120, 100,   0);   // yellow
		case 4:  return CRGB( 60, 110,  15);   // yellow/green
		case 5:  return CRGB(  0, 120,  30);   // green
		case 6:  return CRGB(  0,  82,  90);   // green/blue
		default: return CRGB(  0,  45, 150);   // blue
	}
}

// Off-state charge display. The three first 8-pixel bars (A/B/C) are parallel
// vertical gauges filling from the BOTTOM. Filled pixels show SOC in the band
// colour; while attached, a bright energy packet repeatedly climbs from the current
// fill boundary toward the top.
void chargeGauge() {
	const uint8_t columns[3] = { PanelAStart, PanelBStart, PanelCStart };
	// Each pixel owns exactly one eighth of the range, so pixel k lights the moment
	// the charge passes (k-1)*12.5% (owner spec 2026-07-24). That is a CEILING, not
	// a round-to-nearest: rounding lit pixel k only at the MIDDLE of its band, so
	// the bar lagged a whole LED behind the charge for most of the range (13% still
	// showed one pixel, 26% two, 88% seven). Ceiling of 0 is 0, so an empty pack is
	// dark and any non-zero charge keeps at least the red pixel lit.
	//   LED 1 >0%   LED 2 >12.5%  LED 3 >25%    LED 4 >37.5%
	//   LED 5 >50%  LED 6 >62.5%  LED 7 >75%    LED 8 >87.5%
	uint8_t filled = (uint8_t)(((uint16_t)chargeSoc * 8 + 99) / 100);
	if (filled > 8) filled = 8;

	FastLED.clear();
	for (uint8_t col = 0; col < 3; col++) {
		for (uint8_t level = 0; level < filled; level++)
			DJLEDs[gaugePixel(columns[col], level)] = gaugeLevelColor(level);
	}

	if (!chargeConnected) return;
	if (filled < 8) {
		const uint8_t travel = 8 - filled;
		const uint8_t phase = (millis() / 170UL) % (travel + 2); // two-beat gap
		if (phase < travel) {
			const uint8_t level = filled + phase;   // climbs UPWARD from the fill line
			for (uint8_t col = 0; col < 3; col++)
				DJLEDs[gaugePixel(columns[col], level)] = CRGB(80, 190, 255);
		}
	} else {
		// At full there is nowhere left to climb; breathe the TOP pixel instead —
		// in its OWN ladder colour, so a full pack still reads blue rather than
		// swapping to an unrelated teal.
		const uint8_t glow = beatsin8(18, 70, 255);
		CRGB top = gaugeLevelColor(7);
		top.nscale8_video(glow);
		for (uint8_t col = 0; col < 3; col++)
			DJLEDs[gaugePixel(columns[col], 7)] = top;
	}
}

void readSerial() {
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
		// Buffer overflow: discard characters until the next newline.
	}
}

// Turns on block of 4 LEDs based on start number
void LEDBlockOn(uint8_t LEDStart, CRGB Color, int Brightness)
{
	byte i;
	for (i = 0; i < 4; i++) {
		DJLEDs[LEDStart + i] = Color;
	}
}

// Random Eyes
// Simple mostly solid eyes with a bit of flicker to them.
void RandomEyes()
{
  byte i;
  byte y;
  int pos;

  for (pos = 96; pos < NUM_LEDS; pos++) {
    if (!LEDOn[pos]) {	// Fade LEDs up or down
        DJLEDs[pos].maximizeBrightness(LEDBrightness[pos]);
        if (LEDBrightness[pos] < BRIGHTNESS) LEDBrightness[pos]++;
//      DJLEDs[pos].fadeToBlackBy(8);
    }
    else {
        DJLEDs[pos].maximizeBrightness(LEDBrightness[pos]);
        if (LEDBrightness[pos] > LEDMinBrightness[pos]) LEDBrightness[pos]--;
    }
    if (millis() - LEDMillis[pos] > IntervalTime[pos]) {
      if (!LEDOn[pos]) { // LED Off - turn in on
          DJLEDs[pos] = cGOLD;
        IntervalTime[pos] = random(200, 1600);
        LEDMillis[pos] = millis();
        LEDOn[pos] = 1;
        LEDMinBrightness[pos] = random(BRIGHTNESS * 0.2, BRIGHTNESS);
      }
      else {	// Turn the LED off
        IntervalTime[pos] = random(200, 2000);
        LEDMillis[pos] = millis();
        LEDOn[pos] = 0;
      }
    }
  }
}
// Random Blocks
// 8 LED bars are random individual
// 4 LED groups are random together as groups of 4.
void RandomBlocks()
{
	byte i;
	byte y;
	int pos;

	// Fade all LEDs
//	fadeToBlackBy(DJLEDs, NUM_LEDS, 10);

	// Turn On LEDs
	// Random single bars
	//int pos = random16(8);
	for (i = 0; i < 8; i++) {
		for (y = 0; y < 3; y++) {
			pos = PanelAStart + i + y * 20;

			if (!LEDOn[pos]) {	// LED off, fade it
				DJLEDs[pos].fadeToBlackBy(8);
			}
			else {
				//	DJLEDs[pos].fadeLightBy(-5);
			}
			if (millis() - LEDMillis[pos] > IntervalTime[pos]) {
				if (!LEDOn[pos]) { // LED Off - turn in on
					DJLEDs[pos] = SmallLEDColors[random(0, 9)];// CHSV(random(100, 200), random(0, 200), 255); //CHSV(gHue + random8(64), 200, 0);
					IntervalTime[pos] = random(500, 2500);
					LEDMillis[pos] = millis();
					LEDOn[pos] = 1;
				}
				else {	// Turn the LED off
					IntervalTime[pos] = random(500, 3000);
					LEDMillis[pos] = millis();
					LEDOn[pos] = 0;
				}
			}
		}

	}

	// Large Block LEDs
	for (i = 0; i < 3; i++) {
		for (y = 0; y < 3; y++) {
			pos = PanelA1 + i * 4 + y * 20;

			if (!LEDOn[pos]) {	// LED off, fade it
				for (byte x = 0; x < 4; x++) {
					DJLEDs[pos + x].fadeToBlackBy(8);
				}
			}
			else {
				//	DJLEDs[pos].fadeLightBy(-5);
			}
			if (millis() - LEDMillis[pos] > IntervalTime[pos]) {
				if (!LEDOn[pos]) { // LED Off - turn in on
					byte c = random(0, 3);
					//Serial.println(c);
					LEDBlockOn(pos, BlockLEDColors[c], BLOCKBRIGHTNESS);//CHSV(random(100, 200), random(0, 200), 255), 50);
					IntervalTime[pos] = random(200, 1500);
					LEDMillis[pos] = millis();
					LEDOn[pos] = 1;
				}
				else {	// Turn the LED off
					IntervalTime[pos] = random(200, 2000);
					LEDMillis[pos] = millis();
					LEDOn[pos] = 0;
				}
			}
		}

	}

}

// Random Blocks
// 8 LED bars are bars and random individual
// 4 LED groups are random together as groups of 4.
void RandomBlocks2()
{
	byte i;
	byte y;
	int pos;

	// Fade all LEDs
//	fadeToBlackBy(DJLEDs, NUM_LEDS, 10);

	// Turn On LEDs
	// Random single bars
	//int pos = random16(8);

	// Bar 1

	// Blink end LED;
	pos = PanelAStart + Bar1Length;
	DJLEDs[pos].fadeToBlackBy(8);
	if (millis() - LEDMillis[pos] > IntervalTime[pos]) {
		if (!LEDOn[pos]) { // LED Off - turn in on


			IntervalTime[pos] = random(200, 500);
			LEDMillis[pos] = millis();
			LEDOn[pos] = 1;
		}
		else {	// Turn the LED off
			DJLEDs[pos] = SmallLEDColors[random(0, 9)];
			IntervalTime[pos] = random(200, 500);
			LEDMillis[pos] = millis();
			LEDOn[pos] = 0;
		}
	}

	// Change the Length
	pos = PanelAStart;
	if (millis() - LEDMillis[pos] > IntervalTime[pos]) {
			IntervalTime[pos] = random(1000, 3500);
			LEDMillis[pos] = millis();
			Bar1Length = random(1, 7);
			//Serial.print("bar1length: ");
			//Serial.println(Bar1Length);
			for (i = 0; i < 8; i++) {
				pos = PanelAStart + i;
				if (i < Bar1Length)	DJLEDs[pos] = cRED2;
				else DJLEDs[pos] = 0;
			}
	}


	// Bar 2
	for (i = 0; i < 8; i++) {
		//for (y = 1; y < 3; y++) {
		y = 1;
			pos = PanelBStart + i;// + y * 20;

			if (!LEDOn[pos]) {	// LED off, fade it
				DJLEDs[pos].fadeToBlackBy(8);
			}
			else {
				//	DJLEDs[pos].fadeLightBy(-5);
			}
			if (millis() - LEDMillis[pos] > IntervalTime[pos]) {
				if (!LEDOn[pos]) { // LED Off - turn in on
					DJLEDs[pos] = SmallLEDColors[random(0, 9)];// CHSV(random(100, 200), random(0, 200), 255); //CHSV(gHue + random8(64), 200, 0);
					IntervalTime[pos] = random(500, 2500);
					LEDMillis[pos] = millis();
					LEDOn[pos] = 1;
				}
				else {	// Turn the LED off
					IntervalTime[pos] = random(500, 3000);
					LEDMillis[pos] = millis();
					LEDOn[pos] = 0;
				}
			}
		//}

	}

	// Bar 3

// Blink end LED;
	pos = PanelCStart + Bar3Length;
	DJLEDs[pos].fadeToBlackBy(8);
	if (millis() - LEDMillis[pos] > IntervalTime[pos]) {
		if (!LEDOn[pos]) { // LED Off - turn in on


			IntervalTime[pos] = random(200, 500);
			LEDMillis[pos] = millis();
			LEDOn[pos] = 1;
		}
		else {	// Turn the LED off
			DJLEDs[pos] = SmallLEDColors[random(0, 9)];
			IntervalTime[pos] = random(200, 500);
			LEDMillis[pos] = millis();
			LEDOn[pos] = 0;
		}
	}

	// Change the Length
	pos = PanelCStart;
	if (millis() - LEDMillis[pos] > IntervalTime[pos]) {
		IntervalTime[pos] = random(1000, 3500);
		LEDMillis[pos] = millis();
		Bar3Length = random(1, 7);
		for (i = 0; i < 8; i++) {
			pos = PanelCStart + i;
			if (i < Bar3Length)	DJLEDs[pos] = cWHITE2;
			else DJLEDs[pos] = 0;
		}
	}



	 //Large Block LEDs
	for (i = 0; i < 9; i++) {

			pos = StartLEDNum[i];

			if (!LEDOn[pos]) {	// LED off, fade it
				for (byte x = 0; x < 4; x++) {
					DJLEDs[pos + x].fadeToBlackBy(8);
				}
			}
			else {
				//	DJLEDs[pos].fadeLightBy(-5);
			}
			if (millis() - LEDMillis[pos] > IntervalTime[pos]) {
				if (!LEDOn[pos]) { // LED Off - turn in on
					byte c = random(0, 4);
					//Serial.println(c);
					LEDBlockOn(pos, BlockLEDColors[c], BLOCKBRIGHTNESS);//CHSV(random(100, 200), random(0, 200), 255), 50);
					IntervalTime[pos] = random(200, 1500);
					LEDMillis[pos] = millis();
					LEDOn[pos] = 1;
				}
				else {	// Turn the LED off
					IntervalTime[pos] = random(200, 2000);
					LEDMillis[pos] = millis();
					LEDOn[pos] = 0;
				}
			}


	}

}


void rainbow()
{
	// FastLED's built-in rainbow generator
	fill_rainbow(DJLEDs, NUM_LEDS, gHue, 7);

}

void rainbowWithGlitter()
{
	// built-in FastLED rainbow, plus some random sparkly glitter
	rainbow();
	addGlitter(80);
}

void addGlitter(fract8 chanceOfGlitter)
{
	if (random8() < chanceOfGlitter) {
		DJLEDs[random16(NUM_LEDS)] += CRGB::White;

	}
}

void addGlitter4(fract8 chanceOfGlitter)
{
	for (byte i = 0; i < 4; i++) {
		if (random8() < chanceOfGlitter) {
			DJLEDs[random16(NUM_LEDS)] += CRGB::White;

		}
	}
}

void confetti()
{
	// random colored speckles that blink in and fade smoothly
	fadeToBlackBy(DJLEDs, NUM_LEDS, 10);

	int pos = random16(NUM_LEDS);
	DJLEDs[pos] += CHSV(gHue + random8(64), 200, 255);

}
void ConfettiRedWhite()
{
	// random colored speckles that blink in and fade smoothly
	fadeToBlackBy(DJLEDs, NUM_LEDS, 10);

	int pos = random16(NUM_LEDS);
	DJLEDs[pos] += CHSV(0, gSat, 192);

}


void bpm()
{
	// colored stripes pulsing at a defined Beats-Per-Minute (BPM)
	uint8_t BeatsPerMinute = 62;
	CRGBPalette16 palette = PartyColors_p;
	uint8_t beat = beatsin8(BeatsPerMinute, 64, 255);
	for (int i = 0; i < NUM_LEDS; i++) { //9948
		DJLEDs[i] = ColorFromPalette(palette, gHue + (i * 2), beat - gHue + (i * 10));

	}
}

void juggle() {
	// eight colored dots, weaving in and out of sync with each other
	fadeToBlackBy(DJLEDs, NUM_LEDS, 20);

	byte dothue = 0;
	for (int i = 0; i < 8; i++) {
		DJLEDs[beatsin16(i + 7, 0, NUM_LEDS - 1)] |= CHSV(dothue, 200, 255);

		dothue += 32;
	}
}
void AllRed() {
	fill_solid(DJLEDs, NUM_LEDS, CRGB(255, 0, 0));

}

void AllGreen() {
	fill_solid(DJLEDs, NUM_LEDS, CRGB(0, 255, 0));

}

void AllBlue() {
	fill_solid(DJLEDs, NUM_LEDS, CRGB(0, 0, 255));

}


void ShortCircuit() {
	if (millis() - FadeMillis > FadeInterval) {
		addGlitter4(150);
		DecayTime--;
		FadeInterval += 4;
		FadeMillis = millis();
	}

	if (DecayTime == 0) {
		DecayTime = DECAYTIME;
		gCurrentPatternNumber = 0;
		FadeInterval = 0;
	}
	fadeToBlackBy(DJLEDs, NUM_LEDS, 10);

}

void LEDsOff() {
	// Turn LEDs off and keep them off until another mode command arrives.
	FastLED.clear();
}

// ---------------------------------------------------------------------------
// New animation functions
// ---------------------------------------------------------------------------

// sleepBreath — all 98 pixels pulse dim red with an 8-second sine-wave period.
// RandomEyes is suppressed while in CM_SLEEP so the last two pixels also breathe.
void sleepBreath() {
    uint32_t now    = millis();
    float    phase  = (float)(now % 8000UL) / 8000.0f;   // 0.0 → 1.0
    float    bright = 0.5f * (1.0f - cosf(TWO_PI * phase)); // sine 0→1→0
    uint8_t  b      = (uint8_t)(bright * 50.0f);           // 0 – 50 (very dim)
    fill_solid(DJLEDs, NUM_LEDS, CRGB(b, 0, 0));
}

// angryFlash — red alert. Instead of strobing the whole panel on/off, this
// keeps the control-panel character of RandomBlocks2 but furious:
//
//   * Bars (3 × 8 LEDs)  — a hot red-orange scanner head sweeps each bar
//     back and forth (different speed/phase per bar), leaving a decaying
//     trail via the per-frame fade.
//   * Blocks (9 × 4 LEDs) — asynchronous alarm strobes: short red bursts
//     with random gaps, some burning orange-hot, like warning lights all
//     tripping at once.
//   * Slam — every 1.2–2.6 s the whole panel flashes full red once and
//     decays in ~200 ms: the old strobe's punch, kept as an accent.
//   * A dim red simmer floor keeps the panel from ever going black.
//
// Eyes (pixels 96–97) stay owned by RandomEyes, as in every awake mode.
void angryFlash() {
    static uint32_t lastRunMs   = 0;
    static uint32_t nextSlamMs  = 0;
    static bool     blockHot[9] = { false };

    uint32_t now = millis();

    // Mode (re)entry: a gap since the last frame means we just switched in.
    // Kick the shared per-pixel timers to short angry intervals — otherwise
    // stale RandomBlocks2 intervals (up to 3 s) leave the blocks frozen —
    // and schedule the first slam almost immediately for instant drama.
    if (now - lastRunMs > 250UL) {
        for (byte i = 0; i < 9; i++) {
            uint8_t pos = StartLEDNum[i];
            LEDMillis[pos]    = now;
            IntervalTime[pos] = random(40, 300);
            LEDOn[pos]        = 0;
        }
        nextSlamMs = now + 150;
    }
    lastRunMs = now;

    // Decay everything from the previous frame (trails, strobes, slam tail).
    // Eyes excluded — RandomEyes owns pixels 96+.
    fadeToBlackBy(DJLEDs, 96, 20);

    // Bars: one hot scanner head per bar, sweeping at slightly different
    // rates so the three bars never sync up.
    const uint8_t barStarts[3] = { PanelAStart, PanelBStart, PanelCStart };
    for (byte b = 0; b < 3; b++) {
        uint8_t head = beatsin8(88 + b * 9, 0, 7, 0, b * 85);
        DJLEDs[barStarts[b] + head] = CRGB(255, 40, 0);
    }

    // Blocks: independent alarm strobes. Bursts rewrite their colour every
    // frame (so the fade can't dim them); gaps let the fade swallow them.
    for (byte i = 0; i < 9; i++) {
        uint8_t pos = StartLEDNum[i];
        if (now - LEDMillis[pos] > IntervalTime[pos]) {
            LEDMillis[pos] = now;
            if (!LEDOn[pos]) {
                LEDOn[pos]        = 1;
                IntervalTime[pos] = random(80, 260);   // short furious burst
                blockHot[i]       = (random(0, 4) == 0);  // 1 in 4 burns orange
            } else {
                LEDOn[pos]        = 0;
                IntervalTime[pos] = random(120, 500);  // dark gap
            }
        }
        if (LEDOn[pos]) {
            LEDBlockOn(pos, blockHot[i] ? CRGB(255, 90, 0) : CRGB(255, 0, 0),
                       BLOCKBRIGHTNESS);
        }
    }

    // Slam: one full-panel red flash, then let the fade eat it.
    if (now >= nextSlamMs) {
        fill_solid(DJLEDs, 96, CRGB(255, 0, 0));
        nextSlamMs = now + random(1200, 2600);
    }

    // Simmer floor: the panel never goes fully dark while he's angry.
    for (byte i = 0; i < 96; i++) DJLEDs[i] |= CRGB(18, 0, 0);
}

// complimentFlash — an "aw shucks" shimmer (triggered when Rex is complimented):
// a soft blue glow washes the panel while gold and white sparkles shower across
// it for the ~2.5 s the mode runs (self-ended by runCurrentMode).
//
// PSU note (learned the hard way): white lights all 3 channels at once, so the
// old FULL-PANEL white flash browned out the WS2811s and rendered as RED. This
// effect keeps white to a couple of sparkle pixels at a time; the sustained
// wash is single-channel blue. If sparkles ever look red/orange, lower
// SPARKLE_CHANCE to thin them out rather than brightening anything.
#define SPARKLE_CHANCE 40   // per attempt, 2 attempts/frame ≈ 60 sparkles/s
void complimentFlash() {
    fadeToBlackBy(DJLEDs, 96, 30);   // sparkles glint out in ~75 ms

    // Soft blue wash, always present under the sparkles.
    for (byte i = 0; i < 96; i++) DJLEDs[i] |= CRGB(0, 0, 45);

    // Gold / white sparkle shower.
    for (byte s = 0; s < 2; s++) {
        if (random8() < SPARKLE_CHANCE) {
            DJLEDs[random8(96)] = (random8() < 100) ? CRGB(255, 255, 255)
                                                    : CRGB(255, 200, 80);   // gold
        }
    }
}

// excitedPulse — the idle panel on caffeine. Same control-panel vocabulary as
// RandomBlocks2 (that's what makes it read "Rex", not a generic light show)
// but pumping ~4x faster: a gold head races up each of the three bars and
// wraps around (each bar at its own tempo), the nine blocks pop in rapid
// red-heavy bursts with gold/white/blue accents, and white glitter rides the
// whole thing. Distinct from angry: racing and celebratory, no slams, no
// menace. Runs at full master brightness (set in handleCommand).
void excitedPulse() {
    static uint32_t lastRunMs   = 0;
    static uint8_t  blockCol[9] = { 0 };
    uint32_t now = millis();

    // Mode (re)entry: kick the shared per-pixel timers to excited tempo so
    // stale intervals from the previous mode can't stall the blocks.
    if (now - lastRunMs > 250UL) {
        for (byte i = 0; i < 9; i++) {
            uint8_t pos = StartLEDNum[i];
            LEDMillis[pos]    = now;
            IntervalTime[pos] = random(30, 200);
            LEDOn[pos]        = 0;
        }
    }
    lastRunMs = now;

    fadeToBlackBy(DJLEDs, 96, 25);   // snappy trails

    // Bars: a gold head races up each bar and wraps, each at its own tempo.
    const uint8_t barStarts[3] = { PanelAStart, PanelBStart, PanelCStart };
    for (byte b = 0; b < 3; b++) {
        uint8_t head = scale8(beat8(140 + b * 30), 7);
        DJLEDs[barStarts[b] + head] = cGOLD;
    }

    // Blocks: rapid pops — half red, half gold/white/blue accents.
    for (byte i = 0; i < 9; i++) {
        uint8_t pos = StartLEDNum[i];
        if (now - LEDMillis[pos] > IntervalTime[pos]) {
            LEDMillis[pos] = now;
            if (!LEDOn[pos]) {
                LEDOn[pos]        = 1;
                IntervalTime[pos] = random(60, 220);
                blockCol[i]       = (random8() < 128) ? 0 : random(1, 4);
            } else {
                LEDOn[pos]        = 0;
                IntervalTime[pos] = random(60, 320);
            }
        }
        if (LEDOn[pos]) LEDBlockOn(pos, BlockLEDColors[blockCol[i]], BLOCKBRIGHTNESS);
    }

    addGlitter(60);   // white sparks riding the energy
}

// sadSigh — slow blue melancholy. Everything breathes on long cycles: each
// bar's level slowly swells and slumps like a sigh (12–20 s per cycle, the
// three bars drifting out of phase), and each block breathes dim blue on its
// own slow period. Stateless (pure beatsin8), so no entry kick is needed.
// Runs dim (master brightness 55, set in handleCommand).
void sadSigh() {
    fadeToBlackBy(DJLEDs, 96, 5);   // long, mournful trails as the bars slump

    // Bars: sighing levels, out of phase with each other.
    const uint8_t barStarts[3] = { PanelAStart, PanelBStart, PanelCStart };
    for (byte b = 0; b < 3; b++) {
        uint8_t len = beatsin8(3 + b, 0, 8, 0, b * 70);   // 3–5 bpm sighs
        for (byte i = 0; i < len; i++) DJLEDs[barStarts[b] + i] = CRGB(0, 0, 200);
    }

    // Blocks: each breathes dim blue on its own slow period and phase.
    for (byte i = 0; i < 9; i++) {
        uint8_t lvl = beatsin8(2 + (i & 3), 15, 160, 0, i * 28);
        LEDBlockOn(StartLEDNum[i], CRGB(0, 0, lvl), BLOCKBRIGHTNESS);
    }
}

// happyBounce — cheerful and playful, halfway between idle's calm and
// excited's frenzy: a gold head BOUNCES end-to-end on each bar (bounce, not
// wrap — that's the visual signature separating happy from excited's racing),
// the blocks pop at a relaxed pace in gold/white/blue (no red — red reads
// angry/excited), and rainbow confetti drifts over the top as a nod to the
// old happy pattern.
void happyBounce() {
    static uint32_t lastRunMs   = 0;
    static uint8_t  blockCol[9] = { 1 };
    uint32_t now = millis();

    // Mode (re)entry: kick the shared per-pixel timers (see excitedPulse).
    if (now - lastRunMs > 250UL) {
        for (byte i = 0; i < 9; i++) {
            uint8_t pos = StartLEDNum[i];
            LEDMillis[pos]    = now;
            IntervalTime[pos] = random(100, 500);
            LEDOn[pos]        = 0;
        }
    }
    lastRunMs = now;

    fadeToBlackBy(DJLEDs, 96, 12);   // soft playful trails

    // Bars: gold heads bounce end-to-end, each at its own happy tempo.
    const uint8_t barStarts[3] = { PanelAStart, PanelBStart, PanelCStart };
    for (byte b = 0; b < 3; b++) {
        uint8_t head = beatsin8(45 + b * 8, 0, 7, 0, b * 85);
        DJLEDs[barStarts[b] + head] = cGOLD;
    }

    // Blocks: relaxed cheerful pops in white/gold/blue.
    for (byte i = 0; i < 9; i++) {
        uint8_t pos = StartLEDNum[i];
        if (now - LEDMillis[pos] > IntervalTime[pos]) {
            LEDMillis[pos] = now;
            if (!LEDOn[pos]) {
                LEDOn[pos]        = 1;
                IntervalTime[pos] = random(150, 500);
                blockCol[i]       = random(1, 4);   // cWHITE / cGOLD / cBLUE
            } else {
                LEDOn[pos]        = 0;
                IntervalTime[pos] = random(150, 700);
            }
        }
        if (LEDOn[pos]) LEDBlockOn(pos, BlockLEDColors[blockCol[i]], BLOCKBRIGHTNESS);
    }

    // Rainbow confetti drifting over the top.
    if (random8() < 90) DJLEDs[random8(96)] += CHSV(gHue + random8(64), 200, 255);
}
