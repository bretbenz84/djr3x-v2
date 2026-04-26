# DJ-R3X Conversation Feature Test Script

This script is meant for a fresh database run after deleting `people.db` and
running `setup_assets.py`. It focuses on the conversation features added during
the recent polish pass, especially the places where Rex can otherwise feel
pushy, tone-deaf, or forgetful.

Use fictional details if you do not want real memories stored during the test.

## 0. Fresh Setup

From the repo root:

```bash
source venv/bin/activate

# Optional backup if you do not want to lose the current database permanently.
mkdir -p assets/memory/backups
cp assets/memory/people.db "assets/memory/backups/people.$(date +%Y%m%d-%H%M%S).db" 2>/dev/null || true

rm -f assets/memory/people.db
python setup_assets.py

sqlite3 assets/memory/people.db "PRAGMA table_info(person_facts);"
sqlite3 assets/memory/people.db "PRAGMA table_info(person_emotional_events);"
sqlite3 assets/memory/people.db "PRAGMA table_info(person_conversation_boundaries);"
```

Confirm `person_facts` includes:

- `last_confirmed_at`
- `evidence_count`

Start Rex:

```bash
source venv/bin/activate
python main.py
```

In another terminal:

```bash
tail -f logs/djr3x.log logs/conversation.log
```

## 1. Fresh Person Onboarding

Goal: get a known person into the new DB.

If Rex asks who you are, say:

> Bret Benziger

If he does not ask, use a direct introduction:

> My name is Bret Benziger.

Expected:

- Rex stores/recognizes the person without repeatedly asking for a name.
- `people` has a row for the person.
- Future turns log a known `person_id`.

Quick DB check after stopping:

```bash
sqlite3 assets/memory/people.db "SELECT id, name, visit_count, friendship_tier FROM people;"
```

## 2. Empathy, Grief Flow, And No Roasts

Goal: verify sensitive disclosures use the structured grief/empathy path.

Say:

> My dog Scout died today.

If Rex asks whether you want to talk about it, say:

> Yes.

If he asks the name, say:

> Scout.

Then say:

> He was my little shadow around the house.

Expected:

- Rex offers sympathy and/or asks low-pressure consent.
- Rex does not roast you, the dog, grief, death, or the situation.
- The name `Scout` is used if captured.
- Logs include emotional event storage, such as `[empathy] stored emotional event`.

DB check:

```bash
sqlite3 assets/memory/people.db \
  "SELECT category, description, loss_subject, loss_subject_kind, checkins_muted_at FROM person_emotional_events;"
```

## 3. Recent Loss Check-In On Return

Goal: verify recent sensitive events are remembered and outrank generic intros.

Stop Rex cleanly, start him again, and let him see/hear you as the same known person.

Expected:

- Rex should open with a gentle check-in, for example:
  `Hey Bret, how are you holding up with everything?`
- He should not say something robotic like:
  `I remember your dog died today.`
- He should not do a "back so soon" intro first.

Useful log lines:

- `first-sight emotional check-in`
- `consciousness: firing presence reaction`

## 4. Emotional Check-In Boundary

Goal: verify a person can stop grief/sadness check-ins for that event.

When Rex checks in, say:

> I'd rather not talk about it. Please don't ask me about that again.

Expected:

- Rex acknowledges the boundary briefly.
- Future restarts should not proactively check in about that same event.
- Logs mention muted emotional check-ins.

DB check:

```bash
sqlite3 assets/memory/people.db \
  "SELECT category, description, checkins_muted_at, checkins_muted_reason FROM person_emotional_events;"
```

## 5. Soft Topic Threading

Goal: verify Rex stays with the current topic instead of topic-hopping.

Say:

> I've been working on this droid project because the audio timing is hard.

After Rex replies, say:

> The weird part is that it only fails after text to speech finishes.

Expected:

- Rex should continue the droid/audio/thread naturally.
- He should not jump to an unrelated interview question like travel, favorite music, etc.
- Logs should include an agenda directive with `Topic thread`.

Useful log line:

- `[agenda] ... Topic thread: keep continuity ...`

## 6. Conversation Boundary Memory Beyond Grief

Goal: verify durable preferences like "do not roast X" are stored and obeyed.

Say:

> Don't roast me about my shirt.

Expected:

- Rex acknowledges the boundary.
- Logs include `[boundaries] saved boundary`.
- Later appearance/style riffs should avoid roasting the shirt.

DB check:

```bash
sqlite3 assets/memory/people.db \
  "SELECT behavior, topic, description, active FROM person_conversation_boundaries;"
```

Optional follow-up:

> You can roast my shirt again.

Expected:

- Boundary is deactivated or cleared.

## 7. User Energy Matching

Goal: verify Rex changes pacing based on the user's energy.

Low-energy prompt:

> I'm tired. Keep it short.

Expected:

- Rex uses a shorter, lower-energy reply.
- No pile-on questions.

Playful prompt:

> Okay, that was funny. You can mess with me a little.

Expected:

- Rex can use warmer banter again, as long as no grief/boundary context forbids it.

Useful log lines:

- `[empathy] ... mode=brief`
- `[agenda] ... User energy matching`

## 8. Question Budget

Goal: verify Rex does not become interview-y.

Let Rex ask a question. Give short answers two or three times:

> Pretty good.

> Yeah.

> Not much.

Expected:

- After a couple of questions in the window, Rex should stop adding optional
  follow-up questions.
- Separate curiosity follow-ups should be suppressed.
- Identity and emotional care questions may still happen if appropriate.

Useful log lines:

- `Question budget:`
- `curiosity_check suppressed - question budget full`
- `proactive purpose suppressed by question budget`

## 9. Better Repair Moves

Goal: verify direct corrections are treated as repairs, not insults or normal banter.

Misheard detail:

> No, I said Scout.

Expected:

- Rex acknowledges the correction and uses `Scout`.
- In an active grief name flow, the corrected text should continue the flow.

Tone repair:

> That was rude.

Expected:

- Rex briefly owns the miss.
- No roast, no defensive anger escalation.

Pacing repair:

> This feels like an interview.

Expected:

- Rex backs off and does not ask a new question.

Interruption repair:

> You cut me off.

Expected:

- Rex gives the floor back.

Useful log line:

- `[repair] handled kind=...`

## 10. Memory Confidence And Freshness

Goal: verify repeated facts become stronger, stale/low-confidence facts are treated cautiously.

Say:

> I work as a pilot.

Later in the same session, say:

> Like I said, I work as a pilot.

After stopping Rex, check:

```bash
sqlite3 assets/memory/people.db \
  "SELECT key, value, confidence, evidence_count, last_confirmed_at FROM person_facts WHERE key='job_title';"
```

Expected:

- `evidence_count` should be at least `2`.
- `confidence` should rise slightly after repeated matching evidence.

Force a stale/low-confidence memory for the next run:

```bash
sqlite3 assets/memory/people.db \
  "UPDATE person_facts SET confidence=0.40, last_confirmed_at=datetime('now','-500 days') WHERE key='job_title';"
```

Restart Rex and have a normal conversation.

Expected:

- Rex should treat the pilot memory as tentative.
- If it comes up, he should ask lightly whether it is still true instead of
  using it as a hard certainty or roast foundation.

Useful log line:

- `[llm] fact confirmation prompt`

## 11. End-Of-Thread Grace

Goal: verify Rex lets a thread end without immediately pivoting.

After any topic has had a natural reply, say:

> Thanks, that's all.

Expected:

- Rex gives one short landing acknowledgement.
- Rex does not ask a new question in that reply.
- For about `END_OF_THREAD_GRACE_SECS`, optional proactive chatter should stay quiet.

Useful log lines:

- `[end_thread] closure cue detected`
- `proactive purpose suppressed by end-of-thread grace`
- `curiosity_check suppressed - end-of-thread grace`

To verify the grace clears, start a new topic:

> By the way, what time is it?

Expected:

- Rex answers normally.

## 12. Proactive Arbiter / No Stacked Speech

Goal: verify multiple background instincts do not talk over or replace each other.

Run a normal session with:

- A recent emotional event present.
- A known person visible.
- A few seconds of silence after conversation.

Expected:

- Higher-priority emotional check-ins should beat small talk or visual curiosity.
- Rex should not start one phrase and suddenly switch to another.
- Logs should show one claimed proactive purpose at a time.

Useful log lines:

- `proactive purpose suppressed`
- `consciousness: firing presence reaction`
- `visual curiosity question`

## 13. Optional Camera-Dependent Check

This one is optional because camera quality may make it noisy.

After a few back-and-forth turns, stop speaking while sitting visibly in frame.

Expected if camera/vision is good enough:

- Rex may ask one concrete question based on a visible, non-sensitive detail.
- It should not fire during grief, end-of-thread grace, or while waiting for an answer.
- It should respect shirt/clothing roast boundaries.

Useful log lines:

- `visual curiosity question`
- `visual curiosity step error`

## Final Pass Criteria

The run is a pass if:

- Grief and distress never get roasted.
- Recent grief is remembered on return unless muted.
- Boundaries are stored and obeyed.
- Rex stays on topic and adapts to user energy.
- Rex does not ask endless questions.
- Corrections get repaired cleanly.
- Stale or low-confidence memories are tentative.
- Thread-ending cues produce a graceful stop.
- Optional proactive speech does not overlap, stack, or interrupt the user's answer.

If something fails, save:

```bash
cp logs/djr3x.log "logs/djr3x.failure.$(date +%Y%m%d-%H%M%S).log"
cp logs/conversation.log "logs/conversation.failure.$(date +%Y%m%d-%H%M%S).log"
```
