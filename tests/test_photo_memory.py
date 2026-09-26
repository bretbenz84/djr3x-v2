"""Photographic identity contracts: exact capture binding, persistence, abstention.

No camera, network, audio or real personal data is used.
"""
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np
import config
from memory import photo_album as album
from vision import photo_memory as PM


class PhotoCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        for patch in (
            mock.patch.object(config, "PHOTO_MEMORY_ENABLED", True),
            mock.patch.object(config, "PHOTO_MEMORY_DIR", self.temp.name),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        PM.reset()
        self.addCleanup(PM.reset)
        fact_patch = mock.patch("memory.facts.add_fact")
        fact_patch.start()
        self.addCleanup(fact_patch.stop)
        self.frame = np.zeros((200, 300, 3), dtype=np.uint8)
        self.frame[:, 150:] = 255
        self.record = {"species": "dog", "box": (0, 0, 150, 200)}

    def capture(self, token="one", **values):
        PM._observations[token] = {
            "created": time.monotonic(), "status": "unknown", "kind": "animal",
            "label": "dog", "jpeg": b"original photo", **values,
        }
        return token

    def located(self):
        return {"usable": True, "target_count": 1, "recognition_ready": True, "box": [0, 0, 1, 1]}


class AlbumTests(PhotoCase):
    def test_human_label_survives_session_reset(self):
        token = self.capture()
        self.assertTrue(PM.arm_question(token))
        self.assertEqual(PM.answer("That's Max", owner_id=1, trusted=True), "Max")
        PM.reset()
        refs = album.references("animal", "cat")
        self.assertEqual(refs[0]["name"], "Max")
        self.assertEqual(refs[0]["images"], [b"original photo"])

    def test_bounded_views_and_separate_owners(self):
        for i in range(6):
            album.save(str(i).encode(), name="Max", kind="animal", label="dog", owner_id=1)
        album.save(b"other", name="Max", kind="animal", label="dog", owner_id=2)
        refs = album.references("animal", "dog")
        self.assertEqual(len(refs), 2)
        self.assertEqual(refs[0]["images"], [b"3", b"4", b"5"])
        self.assertEqual(len(list(Path(self.temp.name).glob("*.jpg"))), 4)

    def test_corrupt_index_is_not_overwritten(self):
        path = Path(self.temp.name) / "album.json"
        path.write_text("broken")
        with self.assertRaises(ValueError):
            album.save(b"photo", name="Max", kind="animal", label="dog", owner_id=1)
        self.assertEqual(path.read_text(), "broken")

    def test_missing_reference_abstains_instead_of_hiding_sibling(self):
        row = album.save(b"photo", name="Max", kind="animal", label="dog", owner_id=1)
        (Path(self.temp.name) / row["photos"][0]).unlink()
        with self.assertRaises(FileNotFoundError):
            album.references("animal", "dog")


class LearningTests(PhotoCase):
    def test_no_delivered_question_no_learning(self):
        self.capture()
        self.assertIsNone(PM.answer("Max", owner_id=1, trusted=True))
        self.assertEqual(album.references("animal", "dog"), [])

    def test_exact_prompt_photo_not_newest_detection(self):
        PM.arm_question(self.capture("asked"))
        self.capture("newer", jpeg=b"different dog")
        PM.answer("No, that's Toby", owner_id=1, trusted=True)
        self.assertEqual(album.references("animal", "dog")[0]["images"], [b"original photo"])

    def test_expired_answer_or_capture_cannot_train(self):
        token = self.capture()
        PM.arm_question(token)
        PM._pending = (token, time.monotonic() - 61)
        self.assertIsNone(PM.answer("Max", owner_id=1, trusted=True))
        PM.arm_question(token)
        PM._observations[token]["created"] -= 121
        self.assertIsNone(PM.answer("Max", owner_id=1, trusted=True))

    def test_untrusted_speech_and_unknown_speaker_cannot_train(self):
        PM.arm_question(self.capture())
        self.assertIsNone(PM.answer("Max", owner_id=1, trusted=False))
        self.assertIsNone(PM.answer("Max", owner_id=None, trusted=True))
        self.assertEqual(album.references("animal", "dog"), [])

    def test_nonanswers_never_become_names(self):
        for text in ["yes", "I don't know", "not Max", "Max is outside", "turn left",
                     "move forward", "I'm going camping tomorrow", "who is that?", "no", "maybe Max"]:
            with self.subTest(text=text):
                self.assertIsNone(PM._answer_name(text))
        self.assertEqual(PM._answer_name("It's Biscuit."), "Biscuit")

    def test_recognized_greeting_only_accepts_explicit_correction(self):
        PM.arm_question(self.capture(status="recognized", name="Max"))
        self.assertIsNone(PM.answer("Toby", owner_id=1, trusted=True))
        self.assertEqual(PM.answer("No, that's Toby", owner_id=1, trusted=True), "Toby")


class ComparisonTests(PhotoCase):
    def test_api_image_numbers_are_bound_to_each_actual_image(self):
        from vision import scene
        from intelligence import connectivity
        client = mock.Mock()
        client.with_options.return_value = client
        client.chat.completions.create.return_value.choices = [
            mock.Mock(message=mock.Mock(content='{"usable": true}'))]
        with mock.patch.object(scene, "_get_client", return_value=client), \
             mock.patch.object(connectivity, "guard_client", side_effect=lambda c, _: c):
            PM._request("Compare image 1 against image 2", [b"observation", b"reference"])
        content = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        self.assertEqual([c["type"] for c in content],
                         ["text", "text", "image_url", "text", "image_url"])
        self.assertEqual(content[1]["text"], "IMAGE 1:")
        self.assertEqual(content[3]["text"], "IMAGE 2:")
        import base64
        self.assertEqual(base64.b64decode(content[2]["image_url"]["url"].split(",")[1]), b"observation")
        self.assertEqual(base64.b64decode(content[4]["image_url"]["url"].split(",")[1]), b"reference")

    def test_comparison_mapping_tracks_reordered_multiview_gallery(self):
        import json
        refs = [{"id": "second", "name": "Toby", "images": [b"toby1", b"toby2"]},
                {"id": "first", "name": "Max", "images": [b"max"]}]
        response = {"match_id": "second", "confidence": .99, "runner_up_confidence": .01,
                    "distinctive_evidence": "face markings"}
        with mock.patch.object(PM, "_request", return_value=response) as api:
            result = PM._compare({"status": "unknown", "jpeg": b"new", "label": "dog"}, refs)
        prompt, images = api.call_args.args
        mapping = json.loads(prompt.split("Reference mapping: ")[1].split(". Return JSON")[0])
        self.assertEqual(mapping, [{"id": "second", "image_numbers": [2, 3]},
                                   {"id": "first", "image_numbers": [4]}])
        self.assertEqual(images, [b"new", b"toby1", b"toby2", b"max"])
        self.assertEqual(result["name"], "Toby")

    def test_partial_view_can_be_taught_but_cannot_identify(self):
        located = {**self.located(), "recognition_ready": False,
                   "reason": "One dog held by a person; face turned away"}
        with mock.patch.object(PM, "_request", return_value=located):
            teaching = PM._analyze(self.frame, self.record, "animal", compare=False)
            recognition = PM._analyze(self.frame, self.record, "animal", compare=True)
        self.assertTrue(teaching["jpeg"])
        self.assertFalse(teaching["recognition_ready"])
        self.assertEqual(recognition["status"], "unusable")

    def test_partial_view_with_references_reaches_actual_comparison(self):
        ref = album.save(b"ref", name="Toby", kind="animal", label="dog", owner_id=1)
        located = {**self.located(), "recognition_ready": False}
        comparison = {"match_id": ref["id"], "confidence": .98, "runner_up_confidence": .01,
                      "distinctive_evidence": "Visible white blaze and ear shape match"}
        with mock.patch.object(PM, "_request", side_effect=[located, comparison]) as api:
            result = PM._analyze(self.frame, self.record, "animal")
        self.assertEqual(api.call_count, 2)
        self.assertEqual(result["name"], "Toby")

    def test_multiple_or_missing_target_count_cannot_pass_usable_flag(self):
        for count in (None, 0, 2):
            with self.subTest(count=count), mock.patch.object(PM, "_request", return_value={
                    **self.located(), "target_count": count}):
                self.assertEqual(PM._analyze(self.frame, self.record, "animal", compare=False),
                                 {"status": "unusable"})

    def test_padded_view_preserves_original_pixels(self):
        from PIL import Image
        import io
        with mock.patch.object(PM, "_request", return_value=self.located()):
            result = PM._analyze(self.frame, self.record, "animal")
        image = Image.open(io.BytesIO(result["jpeg"]))
        self.assertEqual(image.size, (250, 200))
        self.assertEqual(image.getpixel((75, 100)), (0, 0, 0))
        self.assertEqual(image.getpixel((220, 100)), (255, 255, 255))
        self.assertFalse((Path(self.temp.name) / "album.json").exists())

    def test_tight_model_bounds_cannot_recrop_reference(self):
        from PIL import Image
        import io
        located = {"usable": True, "target_count": 1, "recognition_ready": True, "box": [.2, .2, .6, .8]}
        with mock.patch.object(PM, "_request", return_value=located) as api:
            result = PM._analyze(self.frame, self.record, "animal", compare=False)
        self.assertEqual(Image.open(io.BytesIO(result["jpeg"])).size, (250, 200))
        self.assertEqual(result["jpeg"], api.call_args.args[1][0])

    def test_padding_at_frame_edge_clamps_without_wrapping(self):
        view = PM._candidate_crop(self.frame, (250, 140, 50, 60))
        self.assertEqual(view.shape[:2], (90, 80))
        self.assertTrue(np.all(view == 255))

    def test_padded_multiple_subjects_cannot_be_saved(self):
        with mock.patch.object(PM, "_request", return_value={
                "usable": False, "reason": "Two animals in expanded view"}):
            result = PM._analyze(self.frame, self.record, "animal", compare=False)
        self.assertEqual(result["status"], "unusable")
        self.assertNotIn("jpeg", result)

    def test_matching_never_self_trains(self):
        ref = album.save(b"reference", name="Max", kind="animal", label="dog", owner_id=1)
        comparison = {"match_id": ref["id"], "confidence": .98,
                      "runner_up_confidence": .2, "distinctive_evidence": "matching muzzle patch"}
        with mock.patch.object(PM, "_request", side_effect=[self.located(), comparison]) as api:
            result = PM._analyze(self.frame, self.record, "animal")
        self.assertEqual(result["name"], "Max")
        self.assertEqual(api.call_args.args[1][1:], [b"reference"])
        self.assertEqual(album.references("animal", "dog")[0]["photos"], ref["photos"])

    def test_similar_dogs_and_invented_ids_abstain(self):
        ref = album.save(b"reference", name="Max", kind="animal", label="dog", owner_id=1)
        for changes in [{"confidence": .8}, {"runner_up_confidence": .9},
                        {"match_id": "invented"}, {"distinctive_evidence": ""},
                        {"confidence": float("nan")}, {"runner_up_confidence": -1}]:
            comparison = {"match_id": ref["id"], "confidence": .98,
                          "runner_up_confidence": .2, "distinctive_evidence": "patch", **changes}
            with self.subTest(changes=changes), mock.patch.object(
                    PM, "_request", side_effect=[self.located(), comparison]):
                result = PM._analyze(self.frame, self.record, "animal")
            self.assertEqual(result["status"], "unknown")

    def test_invalid_crops_and_multiple_subjects_fail_closed(self):
        for box in [[0, 0, 0, 1], [0, 0, float("nan"), 1], [-1, 0, 1, 1]]:
            with self.subTest(box=box), mock.patch.object(
                    PM, "_request", return_value={"usable": True, "target_count": 1, "recognition_ready": True, "box": box}):
                with self.assertRaises(ValueError):
                    PM._analyze(self.frame, self.record, "animal")
        with mock.patch.object(PM, "_request", return_value={"usable": False}):
            self.assertEqual(PM._analyze(self.frame, self.record, "animal")["status"], "unusable")


class PetCommandTests(PhotoCase):
    def test_human_label_saves_partial_view_with_honest_acknowledgement(self):
        from vision import camera, animal_detector
        with mock.patch.object(camera, "capture_pet_still", return_value=self.frame), \
             mock.patch.object(animal_detector, "detect_animals", return_value=[self.record]), \
             mock.patch.object(PM, "_request", return_value={
                 **self.located(), "recognition_ready": False}):
            line = PM.pet_command("This is my dog Max.", owner_id=1, trusted=True)
        self.assertIn("saved", line)
        self.assertIn("clearer view", line)
        self.assertEqual(album.references("animal", "dog")[0]["name"], "Max")

    def run_command(self, text, animals=None, results=None, trusted=True):
        from vision import camera, animal_detector
        with mock.patch.object(camera, "capture_pet_still", return_value=self.frame), \
             mock.patch.object(animal_detector, "detect_animals", return_value=animals if animals is not None else [self.record]), \
             mock.patch.object(PM, "_analyze", side_effect=results or [
                 {"status": "unknown", "jpeg": b"new photo", "kind": "animal", "label": "dog"}]):
            return PM.pet_command(text, owner_id=1, trusted=trusted)

    def test_known_pet_bare_introduction_saves_toby(self):
        from memory import facts
        with mock.patch.object(facts, "get_pets", return_value=[{"name": "Toby", "species": "dog"}]):
            self.assertIn("saved", self.run_command("This is Toby."))
        self.assertEqual(album.references("animal", "dog")[0]["name"], "Toby")

    def test_two_visible_dogs_do_not_get_one_names_photo(self):
        line = self.run_command("This is my dog Max.", animals=[self.record, self.record])
        self.assertIn("haven't saved", line)
        self.assertEqual(album.references("animal", "dog"), [])

    def test_query_identifies_each_dog_and_rejects_duplicate_identity(self):
        for second, expected in [("toby", "Toby"), ("max", "not sure")]:
            with self.subTest(second=second):
                line = self.run_command("What dogs do you see?", animals=[self.record, self.record], results=[
                    {"status": "recognized", "name": "Max", "identity_id": "max"},
                    {"status": "recognized", "name": "Toby", "identity_id": second},
                ])
                self.assertIn(expected, line)
                if second == "max":
                    self.assertNotIn("recognize", line)

    def test_unknown_query_arms_only_after_delivery(self):
        line = self.run_command("What dog do you see?")
        self.assertIn("Can you tell me", line)
        self.assertIsNone(PM._pending)
        self.assertTrue(PM.arm_question(line.observation))
        self.assertEqual(PM.answer("Max", owner_id=1, trusted=True), "Max")

    def test_untrusted_introduction_cannot_save(self):
        self.assertIn("haven't saved", self.run_command("This is my dog Max.", trusted=False))
        self.assertEqual(album.references("animal", "dog"), [])

    def test_unclear_photo_does_not_claim_saved(self):
        line = self.run_command("This is my dog Max.", results=[{"status": "unusable"}])
        self.assertIn("clearer view", line)
        self.assertEqual(album.references("animal", "dog"), [])

    def test_new_introduction_retires_previous_photo_question(self):
        PM.arm_question(self.capture())
        self.assertIn("saved", self.run_command("This is my dog Max."))
        self.assertIsNone(PM._pending)

    def test_human_introduction_is_not_claimed_as_pet(self):
        from memory import facts
        with mock.patch.object(facts, "get_pets", return_value=[]):
            self.assertIsNone(self.run_command("This is Jeremy."))


class WorkerTests(PhotoCase):
    def test_api_failure_finishes_worker_without_writing_album(self):
        with mock.patch.object(PM, "_analyze", side_effect=TimeoutError("offline")):
            record = PM.observe(self.frame, [self.record])[0]
            deadline = time.monotonic() + 2
            while PM.result(record["photo_observation"]).get("status") == "checking" and time.monotonic() < deadline:
                threading.Event().wait(.01)
        self.assertEqual(PM.result(record["photo_observation"])["status"], "unavailable")
        self.assertFalse(PM._busy)
        self.assertFalse((Path(self.temp.name) / "album.json").exists())

    def test_second_dog_invalidates_outstanding_singular_question(self):
        PM.arm_question(self.capture())
        PM.observe(self.frame, [self.record, self.record])
        self.assertIsNone(PM.answer("Max", owner_id=1, trusted=True))

    def test_detection_is_nonblocking_and_no_identity_transfers(self):
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        def slow(*args):
            entered.set()
            release.wait(2)
            finished.set()
            return {"status": "recognized", "name": "Max"}
        with mock.patch.object(PM, "_analyze", side_effect=slow):
            first = PM.observe(self.frame, [self.record])[0]
            self.assertTrue(entered.wait(1))
            later = PM.observe(self.frame, [self.record])[0]
            self.assertNotIn("photo_observation", later)
            self.assertNotIn("name", later)
            release.set()
            self.assertTrue(finished.wait(1))
        self.assertIn("photo_observation", first)

    def test_two_dogs_never_arm_ambiguous_learning(self):
        with mock.patch.object(PM, "_analyze") as analyze:
            records = PM.observe(self.frame, [self.record, self.record])
        analyze.assert_not_called()
        self.assertEqual([r["photo_memory"] for r in records], ["multiple", "multiple"])
        self.assertFalse(PM.arm_question(None))

    def test_disabled_never_starts_network(self):
        with mock.patch.object(config, "PHOTO_MEMORY_ENABLED", False), mock.patch.object(PM, "_request") as api:
            self.assertEqual(PM.observe(self.frame, [self.record]), [self.record])
        api.assert_not_called()


class IntegrationTests(PhotoCase):
    def test_only_delivered_reaction_arms_photo_answer(self):
        from intelligence import consciousness as C
        token = self.capture()
        pending = {"dog": {"species": "dog", "photo_memory": "checking", "photo_observation": token,
                           "last_seen_at": time.monotonic(), "kind": "arrival"}}
        with mock.patch.object(C, "_pending_animal_arrivals", pending), \
             mock.patch.object(C, "_session_is_signing_off", return_value=False), \
             mock.patch.object(C, "_furry_sibling_spoke_recently", return_value=False), \
             mock.patch.object(C, "_animal_remark_covered_by_report", return_value=False), \
             mock.patch.object(C, "_answering_a_directed_look", return_value=False), \
             mock.patch.object(C, "_prime_emotion_frame"), \
             mock.patch.object(C.episodic_hooks, "animal"), \
             mock.patch.object(C, "_animal_presence", {}), \
             mock.patch.object(C, "_animal_reacted_at", {}), \
             mock.patch.object(C, "_animal_species_reacted_at", {}), \
             mock.patch.object(C, "_speak_async", return_value=True) as speak:
            self.assertTrue(C._fire_pending_animal_arrival_reaction())
            self.assertIsNone(PM._pending)
            speak.call_args.kwargs["on_spoke"]()
            self.assertEqual(PM._pending[0], token)
            self.assertFalse(pending)

    def test_real_speech_pipeline_binds_pet_answer(self):
        self._speech_answer("That's Max.")

    def test_bare_pet_name_in_real_speech_pipeline(self):
        self._speech_answer("Max.")

    def test_pet_answer_with_unknown_human_visible_never_enrolls_human(self):
        self._speech_answer("That's Max.", unknown_face=True)

    def test_logged_unprompted_dog_introduction_saves_in_speech_pipeline(self):
        self._speech_answer("This is my dog Max.", direct=True)

    def test_known_sole_face_can_explicitly_teach_without_voice_learning(self):
        self._speech_answer("This is my dog Max.", direct=True, sole_face=True)

    def test_known_sole_face_can_answer_photo_question(self):
        self._speech_answer("That's Max.", sole_face=True)

    def test_photo_label_permission_does_not_promote_ambiguous_speaker(self):
        from intelligence import interaction as I
        good = {"status": "known", "person_id": 1, "learning_allowed": False,
                "basis": "continuous sole-face conversation", "conflicts": []}
        for change in ({"status": "ambiguous"}, {"person_id": 2},
                       {"conflicts": ["other speaker"]}, {"basis": "unconfirmed"}):
            with self.subTest(change=change), mock.patch.object(
                    I, "_current_turn_speaker_evidence", {"resolution": {**good, **change}}):
                self.assertFalse(I._photo_label_speaker_confirmed(1))
        with mock.patch.object(I, "_current_turn_speaker_evidence", {"resolution": good}):
            self.assertTrue(I._photo_label_speaker_confirmed(1))
            self.assertTrue(I._turn_speaker_uncertain())  # biometric/automatic learning guard remains

    def test_logged_identity_question_uses_album_in_speech_pipeline(self):
        self._speech_answer("What dog do you see?", direct=True, query=True)

    def _speech_answer(self, text, unknown_face=False, direct=False, query=False, sole_face=False):
        # Reuse the audio/identity fixture, not a mock of the speech handler.
        from tests.test_voice_learning import RuntimeTests, face
        from intelligence import dialogue_act, conversation_state
        from memory import conversations
        fixture = RuntimeTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture._mock_speech_pipeline()
        I = fixture.I
        if sole_face:
            from intelligence.attribution import Resolution
            fixture.stack.enter_context(mock.patch.object(I, "_resolve_turn_attribution", return_value=
                Resolution("known", 1, "Bret", "continuous sole-face conversation", learning_allowed=False)))
        for state in (dialogue_act, conversation_state):
            state.clear()
            self.addCleanup(state.clear)
        conversations.clear_transcript()
        self.addCleanup(conversations.clear_transcript)
        fixture.faces = [face(1)] + ([face(None, 500)] if unknown_face else [])
        for attr, value in {"_last_speaker_turn": None, "_pending_offscreen_identify": None,
                            "_identity_prompt_until": fixture.now + 30 if unknown_face else 0.,
                            "_pending_onboarding": None}.items():
            fixture.stack.enter_context(mock.patch.object(I, attr, value))
        if direct:
            from vision import camera, animal_detector
            fixture.stack.enter_context(mock.patch.object(camera, "capture_pet_still", return_value=self.frame))
            fixture.stack.enter_context(mock.patch.object(animal_detector, "detect_animals", return_value=[self.record]))
            observed = {"status": "unknown", "jpeg": b"fresh photo", "label": "dog", "kind": "animal"}
            if query:
                ref = album.save(b"reference", name="Max", kind="animal", label="dog", owner_id=1)
                observed.update(status="recognized", name="Max", identity_id=ref["id"])
            fixture.stack.enter_context(mock.patch.object(PM, "_analyze", return_value=observed))
        else:
            token = self.capture()
            PM.arm_question(token)
            question = dialogue_act.note_rex_turn("What is this dog's name?", source="world.animal_arrival",
                                                 target_person_id=1, expected_reply_types=["answer"])
            question.created_at = fixture.now
        audio = fixture.prepare(mouth=None)
        I._utterance_observations['faces'] = I._utterance_observations.pop('visual')
        with mock.patch.object(I, "_enroll_new_person") as enroll:
            I._handle_speech_segment(audio, transcribed_text=text,
                                    raw_best_id_override=1, raw_best_name_override="Bret Benziger",
                                    speaker_score_override=.95)
        self.assertEqual([r["name"] for r in album.references("animal", "dog")], ["Max"],
                         (I._speak_blocking.call_args_list, I._current_turn_speaker_evidence,
                          dialogue_act.frames_snapshot(), PM._pending))
        enroll.assert_not_called()
        if direct:
            I._reply_token_stream.assert_not_called()
        self.assertTrue(any(("recognize Max" if query else "saved") in c.args[0]
                            for c in I._speak_blocking.call_args_list))

    def test_named_reaction_uses_photo_not_species_confirmation(self):
        from intelligence import consciousness as C
        C._animal_confirmed_pet["dog"] = "Wrong Dog"
        self.addCleanup(C._animal_confirmed_pet.clear)
        token = self.capture(status="recognized", name="Max")
        _, line = C._animal_reaction_frame_and_line(
            {"species": "dog", "photo_memory": "checking", "photo_observation": token})
        self.assertEqual(line, "Oh hey, it's Max again!")
        _, line = C._animal_reaction_frame_and_line({"species": "dog", "kind": "return"})
        self.assertNotIn("Wrong Dog", line)

    def test_reaction_composition_does_not_arm_learning(self):
        from intelligence import consciousness as C
        token = self.capture()
        _, line = C._animal_reaction_frame_and_line(
            {"species": "dog", "photo_memory": "checking", "photo_observation": token})
        self.assertIn("Can you tell me?", line)
        self.assertIsNone(PM._pending)

    def test_actual_answer_takeover_saves_and_acknowledges(self):
        from intelligence import interaction as I
        PM.arm_question(self.capture())
        with mock.patch.object(I, "_speak_blocking") as speak:
            line = I._photo_memory_takeover("That's Max", person_id=1, trusted=True, answering_animal=True)
        self.assertIn("saved", line)
        speak.assert_called_once()
        self.assertEqual(album.references("animal", "dog")[0]["name"], "Max")

    def test_unrelated_turn_cannot_answer_photo_question(self):
        from intelligence import interaction as I
        PM.arm_question(self.capture())
        with mock.patch.object(I, "_speak_blocking"):
            self.assertIsNone(I._photo_memory_takeover(
                "No, that's Jeremy", person_id=1, trusted=True, answering_animal=False))
        self.assertEqual(album.references("animal", "dog"), [])

    def test_scene_submits_detector_frame(self):
        from vision import scene
        with mock.patch.object(scene.local_animal_detector, "detect_animals", return_value=[self.record]), \
             mock.patch.object(scene, "_confirm_persistent_animals", side_effect=lambda x: x), \
             mock.patch.object(PM, "observe", return_value=[{**self.record, "photo_memory": "checking"}]) as observe, \
             mock.patch.object(scene.world_state, "update"):
            result = scene.detect_animals_local(self.frame)
        self.assertIs(observe.call_args.args[0], self.frame)
        self.assertEqual(result[0]["photo_memory"], "checking")

    def test_explicit_object_teaching_and_recall(self):
        from vision import camera, animal_detector
        observed = {"status": "unknown", "jpeg": b"chair photo"}
        with mock.patch.object(camera, "get_frame", return_value=self.frame), \
             mock.patch.object(animal_detector, "detect_objects", return_value=[{"label": "chair"}]), \
             mock.patch.object(PM, "_analyze", return_value=observed):
            line = PM.object_command("remember this chair as Captain's chair", owner_id=1, trusted=True)
        self.assertIn("saved", line)
        self.assertEqual(album.references("object", "chair")[0]["name"], "Captain's chair")


if __name__ == "__main__":
    unittest.main()
