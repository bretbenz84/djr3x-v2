# Photographic memory

Rex learns individual pets from human-labelled photographs. The local detector
still supplies animal detections. With `PHOTO_MEMORY_ENABLED = True` (default),
a background worker copies the detector crop, asks OpenAI to validate and refine
its bounding box, and crops the **original camera pixels** locally. No generated
or retouched animal image is used as a reference.

Rex compares a fresh crop with the saved photographs. A confident match says
“Oh hey, it's Max again!” An uncertain match asks “I'm not sure what this dog's
name is. Can you tell me?” Answer “Max”, “That's Max”, or “His name is Max”.
A successful save gets an explicit spoken acknowledgment. Names come from the
human's answer, never a guess from household pet facts. “No, that's Toby” can
correct a recent animal greeting when its conversation frame still owns the
reply. An unrelated conversation cannot label the photo.

Show one animal at a time for now. Multiple detections or multiple subjects in
a crop abstain; seeing a second animal invalidates an outstanding singular photo
question. A blurry, tiny or occluded subject, failed API request, missing reference
file, or oversized gallery falls back to an unnamed reaction. Existing animal
speech cooldowns/session caps remain, so scans do not produce constant greetings.
Directed-look reports retain their duplicate-reaction suppression; use an ordinary
clear sighting to teach a pet.

For objects, use explicit commands:

- “Remember this chair as Captain's chair.”
- “Do you recognize this chair?”
- “What do you call this chair?”

The class must match a local detector label, and exactly one object of that class
must be visible. These commands capture a fresh frame and detect on that exact
frame. Screens/devices remain excluded by the existing detector. Objects are
checked on request, without ambient API calls for every piece of furniture.

## Storage and limits

`data/photo_memory/` contains `album.json` and UUID-named JPEG crops. The directory
is gitignored, created on the first confirmed save, and persists across restarts.
Unlabelled captures stay in memory and expire after 120 seconds. Back up this
directory to preserve learned photographs. To forget all visual identities, stop
Rex and remove it. It is separate from pet facts and human face/voice enrollment.

Each human-confirmed name has up to three reference views, scoped to the person
teaching it. Predictions never add reference images automatically. Teach the
same name again after an uncertain view to add another angle. Pet comparisons
include all saved pets despite detector dog/cat wobble. Names shared by different
teachers remain separate identities.

`PHOTO_MEMORY_MODEL = None` uses the existing `VISION_MODEL`. Requests send crops
to OpenAI as [multiple image inputs](https://developers.openai.com/api/docs/guides/images-vision).
One background animal worker starts at most one check per 30 seconds while
animals remain visible. Each API call has an eight-second timeout with retries
disabled. A check uses one localization call and, when references exist, one
comparison call. Local detection does not wait for these calls. A photo older
than 20 seconds cannot produce a delayed named greeting. The default comparison
budget is 12 identities, with up to three views each; exceeding it abstains rather
than silently dropping a sibling from the comparison.

The match floor is 0.92 with a 0.20 margin over the runner-up. These are model
scores, not calibrated probabilities; similar pets and difficult lighting still
need household field testing. Set `PHOTO_MEMORY_ENABLED = False` in
`user_config.py` to restore legacy pet-name guessing. Other `PHOTO_MEMORY_*`
settings live in `config.py`.

Teaching requires trusted transcription and a resolved speaker allowed to learn.
The answer window expires 60 seconds after the question actually speaks. Saved
bytes belong to that question's immutable observation ID; newer camera frames
never replace the photograph being labelled.

## Verification

Run `venv/bin/python tools/run_lean_checks.py photo_memory animal_pet_name_guess
animal_returns animal_detector pet_during_directed_look` on one shell line.
The runner uses temporary albums/databases and blocks real network, audio and
serial I/O. Coverage includes original-pixel cropping, persistence, abstention,
malformed results, concurrent detections, stale/ambiguous/untrusted answers, the
real speech-handler path, and legacy reactions with the feature disabled. These
tests do not measure real-camera recognition quality.
