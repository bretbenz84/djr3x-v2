"""Heavy background vision yields to foreground local speech without clearing evidence."""
import threading
import unittest
from unittest import mock

import config
from utils import local_work


class VisionPriorityTest(unittest.TestCase):
    def test_periodic_detectors_defer_and_retry_after_speech(self):
        from vision import scene, camera
        blocked = threading.Event()
        resume = threading.Event()
        class Stop:
            ticks = 0
            def is_set(self): return self.ticks >= 2
            def wait(self, timeout):
                self.ticks += 1
                if self.ticks == 1:
                    blocked.set()
                    resume.wait(2)
                return self.is_set()
        with mock.patch.object(scene, '_stop_event', Stop()), \
             mock.patch.object(camera, 'get_frame', return_value=object()), \
             mock.patch.object(scene, '_directed_look_active', True), \
             mock.patch.object(config, 'LOCAL_ANIMAL_DETECTION_ENABLED', True), \
             mock.patch.object(config, 'ANIMAL_DETECTION_ENABLED', True), \
             mock.patch.object(config, 'OBJECT_DETECTION_ENABLED', True), \
             mock.patch.object(config, 'STARTLE_DETECTION_ENABLED', False), \
             mock.patch.object(scene, 'detect_animals_local') as animals, \
             mock.patch.object(scene, 'detect_objects_local') as objects:
            with local_work.foreground():
                worker = threading.Thread(target=scene._scan_loop, args=(180,))
                worker.start()
                try:
                    self.assertTrue(blocked.wait(1))
                    animals.assert_not_called()
                    objects.assert_not_called()
                finally:
                    # Release admission before letting the second loop tick run.
                    pass
            resume.set()
            worker.join(2)
            self.assertFalse(worker.is_alive())
            animals.assert_called_once()
            objects.assert_called_once()

    def test_place_cold_load_defers_and_can_stop_while_speech_owns_compute(self):
        from perception import place_service, place_embedder
        stop = mock.Mock()
        stop.is_set.return_value = False
        stop.wait.return_value = True  # shutdown during deferred startup
        with mock.patch.object(place_service, '_stop', stop), \
             mock.patch.object(place_embedder, 'load_place_embedder') as load, \
             local_work.foreground():
            place_service._run(None, None)
        load.assert_not_called()
        stop.wait.assert_called_once_with(0.1)

    def test_place_load_can_start_after_admission_returns(self):
        from perception import place_service, place_embedder
        stop = mock.Mock()
        stop.is_set.return_value = False
        with mock.patch.object(place_service, '_stop', stop), \
             mock.patch.object(place_embedder, 'load_place_embedder', return_value=None) as load:
            place_service._run(None, None)
        load.assert_called_once()
