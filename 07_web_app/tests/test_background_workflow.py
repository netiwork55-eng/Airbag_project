"""Regression tests สำหรับ Background split/evaluation isolation."""

from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "02_training"))

import background_workflow as workflow  # noqa: E402
import train_from_yolov8_exports as trainer  # noqa: E402


class BackgroundWorkflowTests(unittest.TestCase):
    def make_image(self, path: Path, seed: int) -> None:
        randomizer = random.Random(seed)
        pixels = bytes(randomizer.randrange(256) for _ in range(32 * 32 * 3))
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.frombytes("RGB", (32, 32), pixels).save(path)

    def test_legacy_background_remains_train_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_image(root / "old_batch" / "mouse.jpg", 1)
            summary = workflow.background_summary(root)
            self.assertEqual(summary["train"], 1)
            self.assertEqual(summary["eval_valid"], 0)
            self.assertEqual(summary["eval_test"], 0)

    def test_upload_is_split_and_manifested(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staging"
            destination = root / "background"
            for index in range(30):
                self.make_image(staging / f"image_{index:02d}.png", index + 100)

            result = workflow.install_background_upload(staging, destination, "test_batch")

            self.assertEqual(result["installed"], 30)
            self.assertGreaterEqual(result["eval_valid"], workflow.MIN_EVAL_IMAGES)
            self.assertGreaterEqual(result["eval_test"], workflow.MIN_EVAL_IMAGES)
            self.assertTrue((destination / workflow.BACKGROUND_MANIFEST).is_file())
            self.assertTrue(workflow.background_integrity(destination)["ok"])

    def test_exact_duplicate_in_same_upload_is_installed_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staging"
            destination = root / "background"
            self.make_image(staging / "first.png", 7)
            (staging / "copy.png").write_bytes((staging / "first.png").read_bytes())

            result = workflow.install_background_upload(staging, destination, "duplicate_batch")

            self.assertEqual(result["installed"], 1)
            self.assertEqual(result["skipped_duplicates"], 1)

    def test_eval_images_are_not_returned_as_train(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_image(root / "train" / "train.png", 1)
            self.make_image(root / "eval_valid" / "valid.png", 2)
            self.make_image(root / "eval_test" / "test.png", 3)
            self.assertEqual([path.name for path in workflow.background_images(root, "train")], ["train.png"])

    def test_trainer_copies_only_train_background(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "background"
            output = root / "combined"
            self.make_image(source / "train" / "train.png", 11)
            self.make_image(source / "eval_valid" / "valid.png", 12)
            self.make_image(source / "eval_test" / "test.png", 13)

            copied = trainer.copy_background_images(source, output)

            self.assertEqual(copied, 1)
            self.assertEqual(len(list((output / "train" / "images").glob("*"))), 1)
            self.assertEqual(len(list((output / "train" / "labels").glob("*"))), 1)


if __name__ == "__main__":
    unittest.main()
