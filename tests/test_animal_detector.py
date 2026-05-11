import unittest
from unittest import mock


class _FakeCategory:
    def __init__(self, name, score):
        self.category_name = name
        self.score = score


class _FakeBox:
    def __init__(self, origin_x, origin_y, width, height):
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.width = width
        self.height = height


class _FakeDetection:
    def __init__(self, name, score, box):
        self.categories = [_FakeCategory(name, score)]
        self.bounding_box = box


class LocalAnimalDetectorTests(unittest.TestCase):
    def test_close_releases_detector_and_resets_load_state(self):
        from vision import animal_detector

        fake_detector = mock.Mock()
        old_detector = animal_detector._detector
        old_mp = animal_detector._mp
        old_attempted = animal_detector._load_attempted
        old_ok = animal_detector._load_ok

        try:
            animal_detector._detector = fake_detector
            animal_detector._mp = object()
            animal_detector._load_attempted = True
            animal_detector._load_ok = True

            animal_detector.close()

            fake_detector.close.assert_called_once()
            self.assertIsNone(animal_detector._detector)
            self.assertIsNone(animal_detector._mp)
            self.assertFalse(animal_detector._load_attempted)
            self.assertFalse(animal_detector._load_ok)
        finally:
            animal_detector._detector = old_detector
            animal_detector._mp = old_mp
            animal_detector._load_attempted = old_attempted
            animal_detector._load_ok = old_ok

    def test_records_from_detections_keeps_configured_animals(self):
        from vision import animal_detector

        detections = [
            _FakeDetection("dog", 0.91, _FakeBox(900, 420, 260, 240)),
            _FakeDetection("chair", 0.99, _FakeBox(100, 100, 300, 300)),
        ]

        with mock.patch.object(animal_detector.config, "LOCAL_ANIMAL_DETECTION_SPECIES", {"dog", "cat"}):
            records = animal_detector._records_from_detections(
                detections,
                (720, 1280, 3),
                now=42.0,
            )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["species"], "dog")
        self.assertEqual(records[0]["confidence"], 0.91)
        self.assertEqual(records[0]["last_seen"], 42.0)
        self.assertEqual(records[0]["source"], "mediapipe_object_detector")
        self.assertTrue(records[0]["furred"])
        self.assertEqual(records[0]["box"], (900.0, 420.0, 260.0, 240.0))

    def test_records_from_detections_dedupes_same_species_position(self):
        from vision import animal_detector

        detections = [
            _FakeDetection("cat", 0.84, _FakeBox(500, 360, 200, 170)),
            _FakeDetection("cat", 0.79, _FakeBox(520, 370, 180, 160)),
        ]

        with mock.patch.object(animal_detector.config, "LOCAL_ANIMAL_DETECTION_SPECIES", {"cat"}):
            records = animal_detector._records_from_detections(
                detections,
                (720, 1280, 3),
                now=42.0,
            )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["species"], "cat")


if __name__ == "__main__":
    unittest.main()
