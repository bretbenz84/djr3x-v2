"""Vision call hygiene (field 2026-08-02 21:43: 'what do you see' took 17.5s —
full 1920x1080 frame at auto detail, no timeout, racing the periodic scan).

Four rules under test: frames are downscaled before upload (except face
enrollment), room-level directed looks use low detail while held-object looks
keep auto, every vision call carries a timeout, and the periodic scan defers
while a user-initiated directed look is in flight.
"""

import io
import unittest
from unittest import mock

import numpy as np
from PIL import Image

import config
from vision import image_utils, scene


def _frame(h=1080, w=1920):
    return (np.random.rand(h, w, 3) * 255).astype(np.uint8)


_FAKE_JSON = (
    '{"target_summary":"a room","target_visible":true,'
    '"subject_type":"room_feature","visible_people_count":1,"animals":[],'
    '"notable_details":[],"roast_angle":"","confidence":"high"}'
)


class _FakeClient:
    def __init__(self, captured):
        self._captured = captured
        outer = self

        class _Completions:
            @staticmethod
            def create(**kw):
                outer._captured.update(kw)

                class _Msg:
                    content = _FAKE_JSON

                class _Choice:
                    message = _Msg()

                class _Resp:
                    choices = [_Choice()]

                return _Resp()

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


class DownscaleTest(unittest.TestCase):
    def test_encode_downscales_to_max_dim(self):
        raw = image_utils.encode_jpeg_bytes(_frame(), max_dim=1024)
        w, h = Image.open(io.BytesIO(raw)).size
        self.assertEqual(max(w, h), 1024)
        self.assertAlmostEqual(w / h, 1920 / 1080, places=2)

    def test_encode_untouched_below_max_dim(self):
        raw = image_utils.encode_jpeg_bytes(_frame(480, 640), max_dim=1024)
        self.assertEqual(Image.open(io.BytesIO(raw)).size, (640, 480))


class CallHygieneTest(unittest.TestCase):
    def _captured_call(self, **kwargs):
        captured = {}
        with mock.patch.object(scene, "_get_client",
                               return_value=_FakeClient(captured)):
            scene.analyze_directed_attention(_frame(), **kwargs)
        return captured

    def test_room_level_look_is_low_detail_and_downscaled(self):
        captured = self._captured_call(utterance="what do you see?")
        image_url = captured["messages"][0]["content"][0]["image_url"]
        self.assertEqual(image_url["detail"], "low")
        # base64 of a 1024px q85 JPEG is far below a full-res one (~1.5MB+)
        self.assertLess(len(image_url["url"]), 900_000)

    def test_held_object_look_keeps_auto_detail(self):
        captured = self._captured_call(
            utterance="what am I holding?",
            target_hint="the object the person is holding up",
        )
        self.assertEqual(
            captured["messages"][0]["content"][0]["image_url"]["detail"], "auto"
        )

    def test_every_call_carries_a_timeout(self):
        captured = self._captured_call(utterance="what do you see?")
        self.assertEqual(
            captured["timeout"],
            float(getattr(config, "VISION_REQUEST_TIMEOUT_SECS", 12.0)),
        )

    def test_face_enrollment_keeps_full_resolution(self):
        captured = {}
        with mock.patch.object(scene, "_get_client",
                               return_value=_FakeClient(captured)):
            scene._call_gpt4o(_frame(), "prompt", "face_enrollment")
        room = self._captured_call(utterance="what do you see?")
        self.assertGreater(
            len(captured["messages"][0]["content"][0]["image_url"]["url"]),
            len(room["messages"][0]["content"][0]["image_url"]["url"]) * 2,
        )


class ScanDeferralTest(unittest.TestCase):
    def test_flag_set_during_directed_look_and_cleared_after(self):
        seen = []

        def _inner(*a, **k):
            seen.append(scene._directed_look_active)
            return {}

        with mock.patch.object(scene, "_analyze_directed_attention_locked",
                               side_effect=_inner):
            scene.analyze_directed_attention(_frame())
        self.assertEqual(seen, [True])
        self.assertFalse(scene._directed_look_active)

    def test_flag_cleared_even_on_error(self):
        with mock.patch.object(scene, "_analyze_directed_attention_locked",
                               side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                scene.analyze_directed_attention(_frame())
        self.assertFalse(scene._directed_look_active)


if __name__ == "__main__":
    unittest.main()
