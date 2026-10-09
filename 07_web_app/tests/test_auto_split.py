"""Regression tests สำหรับ Auto Split โดยใช้ TemporaryDirectory ไม่แตะ Dataset จริง."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


WEB_APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEB_APP_DIR))

import app  # noqa: E402


class AutoSplitTests(unittest.TestCase):
    def make_unsplit_dataset(self, root: Path, count: int = 12) -> Path:
        staging = root / "staging"
        (staging / "images").mkdir(parents=True)
        (staging / "labels").mkdir(parents=True)
        for index in range(count):
            (staging / "images" / f"item_{index}.jpg").write_bytes(f"image-{index}".encode())
            (staging / "labels" / f"item_{index}.txt").write_text(
                f"{index % 2} 0.5 0.5 0.2 0.2\n",
                encoding="utf-8",
            )
        return staging

    def test_unsplit_pairs_become_train_valid_test(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            staging = self.make_unsplit_dataset(root)
            pairs, notes = app.discover_unsplit_yolo_pairs(staging, [])
            result = app.create_auto_split_export(pairs, root / "output", ["Airbag", "Defect"], notes)

            self.assertEqual(result["counts"], {"train": 9, "valid": 2, "test": 1})
            export_dir = Path(result["path"])
            self.assertTrue((export_dir / "data.yaml").is_file())
            self.assertTrue((export_dir / app.AUTO_SPLIT_MANIFEST).is_file())
            for split, expected in result["counts"].items():
                self.assertEqual(app.count_images(export_dir / split / "images"), expected)
                self.assertEqual(len(list((export_dir / split / "labels").glob("*.txt"))), expected)

    def test_prepare_detection_upload_runs_auto_split_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            staging = self.make_unsplit_dataset(root)
            with patch.object(app, "read_project_config", return_value={"target_classes": ["Airbag", "Defect"]}):
                result = app.prepare_detection_upload(staging, root / "workspace")

            self.assertEqual(result["ready_exports"], 0)
            self.assertEqual(result["auto_split"]["counts"], {"train": 9, "valid": 2, "test": 1})
            self.assertEqual(len(result["installed_paths"]), 1)

    def test_existing_train_and_validation_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for split in ("train", "valid"):
                (root / split / "images").mkdir(parents=True)
                (root / split / "labels").mkdir(parents=True)
                (root / split / "images" / f"{split}.jpg").write_bytes(b"image")
                (root / split / "labels" / f"{split}.txt").write_text(
                    "0 0.5 0.5 0.2 0.2\n",
                    encoding="utf-8",
                )
            (root / "data.yaml").write_text(
                "train: train/images\nval: valid/images\nnames: [Airbag]\n",
                encoding="utf-8",
            )
            self.assertTrue(app.dataset_has_usable_split(root))

    def test_missing_image_label_pair_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "images").mkdir()
            (root / "labels").mkdir()
            (root / "images" / "missing.jpg").write_bytes(b"image")
            (root / "labels" / "other.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "no matching label"):
                app.discover_unsplit_yolo_pairs(root, [])



if __name__ == "__main__":
    unittest.main()
