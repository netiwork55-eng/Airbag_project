"""Camera discovery must prefer configuration and fall back safely."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


WEB_APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEB_APP_DIR))

import app  # noqa: E402


class FakeCapture:
    def __init__(self, opened: bool, readable: bool = False) -> None:
        self.opened = opened
        self.readable = readable
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def read(self):
        frame = np.full((4, 4, 3), 80, dtype=np.uint8) if self.readable else None
        return self.readable, frame

    def release(self) -> None:
        self.released = True


class CameraDiscoveryTests(unittest.TestCase):
    def test_invalid_configuration_does_not_silently_select_legacy_detector(self):
        with mock.patch.object(app.detector, "PROJECT_CONFIG_PATH") as config:
            config.exists.return_value = False
            with self.assertRaises(FileNotFoundError):
                app.detector.load_project_detection_config()
            config.exists.return_value = True
            config.read_text.return_value = "invalid-json"
            with self.assertRaises(RuntimeError):
                app.detector.load_project_detection_config()
            config.read_text.return_value = "[]"
            with self.assertRaises(ValueError):
                app.detector.load_project_detection_config()

    def test_preferred_camera_is_checked_first_then_fallback(self) -> None:
        captures = {
            1: FakeCapture(False),
            0: FakeCapture(True, True),
        }

        with mock.patch.object(
            app.detector,
            "open_camera",
            side_effect=lambda index: captures.get(index, FakeCapture(False)),
        ) as open_camera:
            cap, selected = app.detector.open_available_camera(1, max_index=2)

        self.assertIs(cap, captures[0])
        self.assertEqual(selected, 0)
        self.assertEqual([call.args[0] for call in open_camera.call_args_list[:2]], [1, 0])
        self.assertTrue(captures[1].released)

    def test_opened_device_without_frames_is_skipped(self) -> None:
        unreadable = FakeCapture(True, False)
        readable = FakeCapture(True, True)

        with mock.patch.object(
            app.detector,
            "open_camera",
            side_effect=lambda index: unreadable if index == 2 else readable,
        ):
            cap, selected = app.detector.open_available_camera(2, max_index=2)

        self.assertIs(cap, readable)
        self.assertEqual(selected, 0)
        self.assertTrue(unreadable.released)

    def test_black_placeholder_frame_is_not_treated_as_a_camera(self) -> None:
        blank = FakeCapture(True, True)
        blank.read = lambda: (True, np.zeros((4, 4, 3), dtype=np.uint8))
        usable = FakeCapture(True, True)

        with mock.patch.object(
            app.detector,
            "open_camera",
            side_effect=lambda index: blank if index == 1 else usable,
        ):
            cap, selected = app.detector.open_available_camera(1, max_index=1)

        self.assertIs(cap, usable)
        self.assertEqual(selected, 0)
        self.assertTrue(blank.released)


if __name__ == "__main__":
    unittest.main()
