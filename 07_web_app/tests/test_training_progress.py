"""Regression tests for ETA parsing after a project folder is copied to another PC."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


WEB_APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEB_APP_DIR))

import app  # noqa: E402


class TrainingProgressTests(unittest.TestCase):
    def make_detector_log(self, root: Path) -> Path:
        log_path = root / "20260715_092510_airbag_detector.log"
        log_path.write_text(
            """
      1/40         0G     0.5000     0.2000     0.9000         20        416: 100% 40/40 5.0s/it 3:20
      2/40         0G     0.4000     0.1500     0.8000         20        416: 100% 40/40 5.0s/it 4:00
                 Class     Images  Instances: 100% 2/2 8.0s/it 16.0s
            """.strip(),
            encoding="utf-8",
        )
        return log_path

    def test_historical_elapsed_uses_log_duration_not_file_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            log_path = self.make_detector_log(Path(temp))
            os.utime(log_path, (2_000_000_000, 2_000_000_000))

            progress = app.parse_training_progress(str(log_path), status="finished")

            self.assertEqual(progress["elapsed_seconds"], 440.0)
            self.assertEqual(progress["seconds_per_epoch"], 220.0)
            self.assertEqual(progress["remaining_seconds"], 0.0)

    def test_validation_tqdm_row_is_not_used_as_training_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            log_path = self.make_detector_log(Path(temp))
            progress = app.parse_training_progress(str(log_path))

            self.assertEqual(progress["epoch"], 2)
            self.assertEqual(progress["total_epochs"], 40)
            self.assertEqual(progress["batch"], 40)
            self.assertEqual(progress["total_batches"], 40)

    def test_classifier_completed_hours_are_available_for_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            log_path = Path(temp) / "20260715_092510_brand_classifier.log"
            log_path.write_text("50 epochs completed in 2.757 hours.\n", encoding="utf-8")

            seconds, epochs = app.logged_training_duration_summary(str(log_path))

            self.assertEqual(epochs, 50)
            self.assertAlmostEqual(seconds, 9925.2)

    def test_completed_early_stopping_is_reported_honestly(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            log_path = Path(temp) / "20260703_093103_airbag_detector.log"
            log_path.write_text(
                "EarlyStopping: Training stopped early as no improvement was observed.\n"
                "13 epochs completed in 1.915 hours.\n"
                "Training complete: C:/runs/weights/best.pt\n"
                "Copied trained model to: C:/models/best.pt\n",
                encoding="utf-8",
            )

            status = app.infer_train_status_from_log(str(log_path))

            self.assertEqual(status["status"], "finished")
            self.assertTrue(status["early_stopped"])
            self.assertEqual(status["completion_reason"], "early_stopping")
            self.assertIn("epoch 13", status["status_detail"])

    def test_normal_completion_is_not_marked_as_early_stopping(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            log_path = Path(temp) / "20260715_092510_airbag_detector.log"
            log_path.write_text(
                "40 epochs completed in 1.928 hours.\n"
                "Training complete: C:/runs/weights/best.pt\n",
                encoding="utf-8",
            )

            status = app.infer_train_status_from_log(str(log_path))

            self.assertEqual(status["completion_reason"], "completed")
            self.assertFalse(status["early_stopped"])

    def test_training_run_cache_invalidates_when_log_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            log_path = Path(temp) / "20260715_092510_airbag_detector.log"
            log_path.write_text("Training complete: C:/runs/weights/best.pt\n", encoding="utf-8")
            app.TRAIN_RUN_CACHE["key"] = None
            app.TRAIN_RUN_CACHE["records"] = []

            with mock.patch.object(app, "TRAIN_LOG_DIR", Path(temp)), mock.patch.object(
                app,
                "infer_train_status_from_log",
                wraps=app.infer_train_status_from_log,
            ) as infer_status:
                first = app.train_run_records()
                first_call_count = infer_status.call_count
                cached = app.train_run_records()
                cached_call_count = infer_status.call_count
                log_path.write_text(
                    "1/2 0G 0.5 0.2 0.8 10 416\n"
                    "Training complete: C:/runs/weights/best.pt\n",
                    encoding="utf-8",
                )
                app.train_run_records()
                refreshed_call_count = infer_status.call_count

            self.assertEqual(first, cached)
            self.assertEqual(first_call_count, 1)
            self.assertEqual(cached_call_count, 1)
            self.assertEqual(refreshed_call_count, 2)


if __name__ == "__main__":
    unittest.main()
