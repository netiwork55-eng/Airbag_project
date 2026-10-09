"""Regression tests for excluding unsafe full-frame offline augmentations."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = PROJECT_ROOT / "02_training"
sys.path.insert(0, str(TRAINING_DIR))

import train_from_yolov8_exports as training  # noqa: E402


class FullFrameFilterTests(unittest.TestCase):
    def test_full_frame_threshold(self) -> None:
        self.assertEqual(training.full_frame_box_count_from_text("0 0.5 0.5 1 1\n"), 1)
        self.assertEqual(training.full_frame_box_count_from_text("0 0.5 0.5 0.98 0.98\n"), 1)
        self.assertEqual(training.full_frame_box_count_from_text("0 0.5 0.5 0.97 1\n"), 0)

    def test_copy_split_rejects_entire_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            output = root / "output"
            (source / "train" / "images").mkdir(parents=True)
            (source / "train" / "labels").mkdir(parents=True)
            (source / "train" / "images" / "unsafe.jpg").write_bytes(b"image")
            (source / "train" / "labels" / "unsafe.txt").write_text(
                "0 0.5 0.5 1 1\n1 0.5 0.5 0.2 0.2\n",
                encoding="utf-8",
            )
            rejected: list[dict] = []

            copied, objects, skipped = training.copy_split(
                source,
                output,
                "train",
                "d1",
                ["Airbag", "Mark"],
                ["Airbag", "Mark"],
                rejected_full_frame=rejected,
            )

            self.assertEqual((copied, objects, skipped), (0, 0, 1))
            self.assertEqual(len(rejected), 1)
            self.assertFalse((output / "train" / "images" / "d1_unsafe.jpg").exists())
            self.assertFalse((output / "train" / "labels" / "d1_unsafe.txt").exists())

    def test_copy_split_keeps_normal_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            output = root / "output"
            (source / "train" / "images").mkdir(parents=True)
            (source / "train" / "labels").mkdir(parents=True)
            (source / "train" / "images" / "safe.jpg").write_bytes(b"image")
            (source / "train" / "labels" / "safe.txt").write_text(
                "0 0.5 0.5 0.6 0.4\n",
                encoding="utf-8",
            )

            copied, objects, skipped = training.copy_split(
                source,
                output,
                "train",
                "d1",
                ["Airbag"],
                ["Airbag"],
            )

            self.assertEqual((copied, objects, skipped), (1, 1, 0))
            self.assertTrue((output / "train" / "images" / "d1_safe.jpg").is_file())
            self.assertEqual(
                (output / "train" / "labels" / "d1_safe.txt").read_text(encoding="utf-8"),
                "0 0.500000 0.500000 0.600000 0.400000\n",
            )

    def test_copy_split_returns_three_counts_when_split_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertEqual(
                training.copy_split(
                    root / "source",
                    root / "output",
                    "test",
                    "d1",
                    ["Airbag"],
                    ["Airbag"],
                ),
                (0, 0, 0),
            )


if __name__ == "__main__":
    unittest.main()
