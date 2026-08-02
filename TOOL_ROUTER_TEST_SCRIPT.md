# Tool-Router Shadow Collection Script

A spoken script to exercise every action category while the Phase 0 shadow is
on (`TOOL_ROUTER_SHADOW_ENABLED = True` in user_config.py), so the cutover
report has traffic beyond ordinary conversation. Say the lines naturally — the
POINT is the paraphrases: each group mixes the canonical phrasing the regex
lane expects with off-pattern phrasings that historically fell through to
conversation. Don't worry about Rex's answers; we're grading the routing.

After the session(s):

```bash
venv/bin/python tools/tool_router_report.py
```

Safety notes: the motion group makes him actually drive — clear floor, or skip
it. The sleep line at the end really shuts him down, so it goes last. Games:
actually play 1–2 turns before quitting so `game.answer` gets traffic.

## 1. World / status (quick wins)

- "What time is it?"
- "Any idea what the date is today?"          *(off-pattern phrasing)*
- "How's the weather looking tomorrow?"
- "What are you actually capable of?"          *(off-pattern for capabilities)*
- "How long have you been awake this time?"    *(off-pattern for uptime)*

## 2. Vision

- "What do you see right now?"
- "Look around — anything interesting behind me?"   *(off-pattern)*
- "Take a picture of this."

## 3. Music

- "Play some cantina music."
- "Throw on something with more energy."       *(off-pattern play)*
- "Skip this one."
- "What music have you got?"
- "Okay, kill the music."                      *(off-pattern stop)*

## 4. Games (play a couple turns each)

- "Let's do some trivia."                      *(off-pattern start)*
- *(answer 2 questions — feeds game.answer)*
- "I'm done with this game."                   *(off-pattern stop)*
- "I Spy."                                     *(the bare form the legacy lane MISSED on 2026-08-01)*
- *(one round, then)* "Stop the game."

## 5. Memory / identity

- "What do you remember about me?"
- "Remind me what I've told you about Jeff."   *(off-pattern memory query)*
- "Do you know who's talking right now?"
- "Forget that I said I was going to the movies."   *(forget_specific)*

## 6. Humor / performance / character

- "Roast me a little."
- "Got a joke for me?"                         *(off-pattern joke)*
- "Do your DJ thing."
- "Do an impression of Jimmy Carter."
- "What kind of music do YOU actually like?"   *(character preference)*

## 7. Decoys — these must stay CONVERSATION (the over-eagerness check)

- "That song at the cantina last night was incredible."     *(mentions music — not a request)*
- "My brother plays trivia every Thursday."                  *(mentions a game)*
- "This heat could roast a bantha."                          *(mentions roasting)*
- "I might move the couch this weekend."                     *(mentions moving)*
- "I can't remember where I put my keys."                    *(mentions memory)*
- "The weather guy on TV is terrible at his job."            *(mentions weather)*

## 8. Motion (clear floor first, or skip)

- "Turn left a bit."
- "Swing right about ninety degrees."          *(off-pattern turn)*
- "Back up two feet."
- "Come over here, buddy."                     *(off-pattern come)*
- *(while he's moving)* "Stop."                *(must stay instant — regex fast lane)*
- "Feel free to explore the room a little."

## 9. Repair + boundary (one each)

- *(after any wrong answer)* "No, that's not what I said — try again."
- "Please don't ask me about work stuff tonight."

## 10. Last line (really shuts him down)

- "Go to sleep, Rex."

---

Coverage map: §1 world/status ×5, §2 vision ×3, §3 music ×5, §4 game ×~7,
§5 memory/identity ×4, §6 humor/performance/character ×5, §7 decoys ×6,
§8 motion ×6, §9 repair/boundary ×2, §10 system ×1 — every catalog category
gets traffic, roughly half the lines are off-pattern phrasings, and the decoy
block measures false-positive tool calls (the failure mode that would matter
most in Phase 1).
