import unittest
from unittest import mock


class CameraReconnectTests(unittest.TestCase):
    def test_reconnect_callback_fires_once_with_downtime(self):
        from vision import camera

        calls = []
        callback = lambda downtime: calls.append(downtime)
        old_callbacks = list(camera._reconnect_callbacks)
        old_offline_since = camera._offline_since
        try:
            with camera._reconnect_lock:
                camera._reconnect_callbacks.clear()
                camera._offline_since = None
            camera.register_on_reconnect(callback)

            with mock.patch.object(camera.time, "monotonic", side_effect=[100.0, 104.5]):
                camera._mark_camera_offline()
                camera._notify_camera_reconnected_if_needed()
                camera._notify_camera_reconnected_if_needed()

            self.assertEqual(calls, [4.5])
        finally:
            with camera._reconnect_lock:
                camera._reconnect_callbacks[:] = old_callbacks
                camera._offline_since = old_offline_since


if __name__ == "__main__":
    unittest.main()
