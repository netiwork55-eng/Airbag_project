"""Regression tests for dataset preview counts shown in the workspace."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


WEB_APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEB_APP_DIR))

import app  # noqa: E402


class DatasetPreviewTests(unittest.TestCase):
    def test_readme_text_files_are_not_counted_as_yolo_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            dataset = Path(temp) / "example.yolov8"
            (dataset / "train" / "images").mkdir(parents=True)
            (dataset / "train" / "labels").mkdir(parents=True)
            (dataset / "data.yaml").write_text("names: [Airbag]\n", encoding="utf-8")
            (dataset / "README.dataset.txt").write_text("metadata", encoding="utf-8")
            (dataset / "README.roboflow.txt").write_text("metadata", encoding="utf-8")
            (dataset / "train" / "images" / "sample.jpg").write_bytes(b"image")
            (dataset / "train" / "labels" / "sample.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n",
                encoding="utf-8",
            )

            preview = app.dataset_folder_preview(dataset)

            self.assertEqual(preview["labels"], 1)
            self.assertEqual(preview["yolo_exports"][0]["labels"], 1)

    def test_latest_upload_manifest_excludes_older_dataset_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            dataset = Path(temp) / "example.yolov8"
            old_batch = dataset / "raw_images" / "older"
            latest_batch = dataset / "raw_images" / "latest"
            old_batch.mkdir(parents=True)
            latest_batch.mkdir(parents=True)
            (old_batch / "old.jpg").write_bytes(b"old")
            (latest_batch / "new_01.jpg").write_bytes(b"new")
            (latest_batch / "new_02.png").write_bytes(b"new")

            manifest = app.record_latest_dataset_upload(dataset, [str(latest_batch)])
            page = app.latest_dataset_upload_page(dataset, offset=0, limit=12)

            self.assertEqual(manifest["count"], 2)
            self.assertEqual(page["total"], 2)
            self.assertEqual(
                [sample["name"] for sample in page["samples"]],
                ["new_01.jpg", "new_02.png"],
            )
            self.assertNotIn("old.jpg", [sample["name"] for sample in page["samples"]])

    def test_latest_upload_gallery_is_paginated_from_the_same_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            dataset = Path(temp) / "example.yolov8"
            latest_batch = dataset / "raw_images" / "latest"
            latest_batch.mkdir(parents=True)
            for index in range(5):
                (latest_batch / f"sample_{index}.jpg").write_bytes(b"image")

            app.record_latest_dataset_upload(dataset, [str(latest_batch)])
            first = app.latest_dataset_upload_page(dataset, offset=0, limit=2)
            second = app.latest_dataset_upload_page(dataset, offset=first["next_offset"], limit=2)

            self.assertTrue(first["has_more"])
            self.assertEqual(first["next_offset"], 2)
            self.assertEqual(second["offset"], 2)
            self.assertEqual(second["next_offset"], 4)
            self.assertEqual(
                {sample["name"] for sample in first["samples"]}.intersection(
                    sample["name"] for sample in second["samples"]
                ),
                set(),
            )


if __name__ == "__main__":
    unittest.main()
