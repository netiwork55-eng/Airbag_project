"""Safety checks for training API requests that must not start a process."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np


WEB_APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEB_APP_DIR))

import app  # noqa: E402


class TrainingApiGuardTests(unittest.TestCase):
    def test_normal_reference_preflight_is_a_supported_workflow(self) -> None:
        runtime = {"ready": False, "model": None, "reason": "Build first"}
        with (
            mock.patch.object(app, "read_project_config", return_value={}),
            mock.patch.object(app, "normal_reference_runtime_status", return_value=runtime),
            mock.patch.object(app, "latest_detector_model_path", return_value=None),
        ):
            response = app.api_train_preflight("normal_reference")
        payload = json.loads(response.body)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["train_type"], "normal_reference")
        self.assertTrue(any(check["label"] == "Normal Reference engine" for check in payload["checks"]))

    def test_reference_fingerprint_changes_only_for_real_input_changes(self) -> None:
        settings = {"inspection_size": 512, "min_good_images": 5, "min_region_ratio": 0.002}
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "good.jpg"
            image_path.write_bytes(b"reviewed-known-good")
            first = app.normal_reference_training_fingerprint([image_path], settings)
            self.assertEqual(first, app.normal_reference_training_fingerprint([image_path], settings))
            image_path.write_bytes(b"changed-known-good")
            self.assertNotEqual(first, app.normal_reference_training_fingerprint([image_path], settings))

    def test_reviewed_known_good_corpus_uses_all_confirmed_folders_and_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            (first / "a.jpg").write_bytes(b"same-image")
            (second / "copy.jpg").write_bytes(b"same-image")
            (second / "b.jpg").write_bytes(b"different-image")
            records = [
                {"id": "first", "label": "First", "meta": {"dataset_type": "Known-good Reference"}, "review_status": "reviewed", "needs_info": False},
                {"id": "second", "label": "Second", "meta": {"dataset_type": "Known-good Reference"}, "review_status": "reviewed", "needs_info": False},
                {"id": "pending", "label": "Pending", "meta": {"dataset_type": "Known-good Reference"}, "review_status": "needs_info", "needs_info": True},
            ]
            folders = {"first": first, "second": second, "pending": root / "pending"}
            with (
                mock.patch.object(app, "list_dataset_folder_records", return_value=records),
                mock.patch.object(app, "dataset_folder_path", side_effect=lambda name: folders[name]),
            ):
                corpus = app.reviewed_known_good_corpus()

        self.assertEqual(corpus["dataset_count"], 2)
        self.assertEqual(corpus["source_images"], 3)
        self.assertEqual(corpus["unique_images"], 2)
        self.assertEqual(corpus["duplicates_removed"], 1)
        self.assertEqual([item["id"] for item in corpus["datasets"]], ["first", "second"])

    def test_reference_fingerprint_is_independent_of_file_name(self) -> None:
        settings = {"inspection_size": 512, "min_good_images": 5, "min_region_ratio": 0.002}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first.jpg"
            renamed = root / "renamed.jpg"
            first.write_bytes(b"identical-content")
            renamed.write_bytes(b"identical-content")
            self.assertEqual(
                app.normal_reference_training_fingerprint([first], settings),
                app.normal_reference_training_fingerprint([renamed], settings),
            )

    def test_reference_train_accepts_the_confirmed_corpus_without_selected_folder(self) -> None:
        state_before = dict(app.REFERENCE_TRAIN_STATE)
        fake_thread = mock.Mock()
        fake_thread.start = mock.Mock()
        corpus = {
            "paths": [Path("one.jpg"), Path("two.jpg"), Path("three.jpg")],
            "datasets": [{"id": "first", "recipe": {"model": "same_front", "target": "Airbag"}}, {"id": "second", "recipe": {"model": "same_front", "target": "Airbag"}}],
            "dataset_count": 2,
            "source_images": 4,
            "unique_images": 3,
            "duplicates_removed": 1,
        }
        try:
            app.REFERENCE_TRAIN_STATE.update({"running": False, "status": "idle"})
            with (
                mock.patch.object(app, "reviewed_known_good_corpus", return_value=corpus),
                mock.patch.object(app, "read_project_config", return_value={"normal_reference": {}}),
                mock.patch.object(app, "normal_reference_training_fingerprint", return_value="fingerprint"),
                mock.patch.object(app, "refresh_train_state_locked"),
                mock.patch.object(app, "TRAIN_PROCESS", None),
                mock.patch.object(app.vision_service.runtime.ownership, "start_thread", return_value=fake_thread) as thread_factory,
            ):
                response = app.api_train_normal_reference({"scope": "reviewed_known_good", "min_good_images": 3, "inspection_size": 512})
            payload = json.loads(response.body)
            self.assertEqual(response.status_code, 202)
            self.assertTrue(payload["accepted"])
            self.assertEqual(payload["state"]["total_images"], 3)
            self.assertEqual(payload["state"]["source_images"], 4)
            self.assertEqual(payload["state"]["datasets"], ["first", "second"])
            self.assertEqual(thread_factory.call_args.args[1][2]["inspection_size"], 512)
            self.assertIs(thread_factory.call_args.args[0],app.run_normal_reference_training)
        finally:
            app.REFERENCE_TRAIN_STATE.clear()
            app.REFERENCE_TRAIN_STATE.update(state_before)

    def test_reviewed_ng_metrics_are_kept_outside_training(self) -> None:
        class FakeInspector:
            def inspect(self, image, *, object_box):
                status = "NG" if int(image[0, 0, 2]) > 100 else "OK"
                return app.InspectionResult(status, 0.9, 1.0, 0.5, [])

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ng_path = root / "ng.jpg"
            ok_path = root / "ok.jpg"
            cv2.imencode(".jpg", np.full((20, 20, 3), (0, 0, 255), np.uint8))[1].tofile(str(ng_path))
            cv2.imencode(".jpg", np.zeros((20, 20, 3), np.uint8))[1].tofile(str(ok_path))
            records = {
                "confirmed_ng": [{"source_image": str(ng_path), "object": {"box": [0, 0, 20, 20]}}],
                "false_alarm": [{"source_image": str(ok_path), "object": {"box": [0, 0, 20, 20]}}],
            }
            with mock.patch.object(app, "inspection_review_records", side_effect=lambda status: records[status]):
                reference = {"recipe_datasets": [{"id": "part-a"}]}
                for group in records.values():
                    for record in group:
                        record["evaluation_scope"] = app.reference_review_scope(reference)
                metrics = app.evaluate_reference_on_reviewed_cases(FakeInspector(), reference)
        self.assertEqual(metrics["ng_recall"], 1.0)
        self.assertEqual(metrics["false_ok_rate"], 0.0)
        self.assertEqual(metrics["reviewed_ok_acceptance_rate"], 1.0)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)

    def test_candidate_comparison_rejects_accuracy_regression(self) -> None:
        incumbent = {
            "normal_acceptance_rate": 0.97,
            "confirmed_ng_images": 8,
            "ng_recall": 0.875,
            "reviewed_ok_images": 6,
            "reviewed_ok_acceptance_rate": 1.0,
        }
        candidate = {
            "normal_acceptance_rate": 0.96,
            "confirmed_ng_images": 8,
            "ng_recall": 0.75,
            "reviewed_ok_images": 6,
            "reviewed_ok_acceptance_rate": 1.0,
        }
        comparison = app.compare_reference_candidate(candidate, incumbent)
        self.assertFalse(comparison["promote"])
        self.assertEqual(comparison["decision"], "rejected")
        self.assertIn("ng_recall_not_regressed", comparison["failed_checks"])

    def test_candidate_without_ng_evidence_is_marked_provisional(self) -> None:
        comparison = app.compare_reference_candidate(
            {"normal_acceptance_rate": 0.98, "confirmed_ng_images": 0, "reviewed_ok_images": 0, "independent_validation": True, "validation_images": 10},
            {"normal_acceptance_rate": 0.97},
        )
        self.assertFalse(comparison["promote"])
        self.assertTrue(comparison["checks_passed"])
        self.assertTrue(comparison["provisional"])
        self.assertEqual(comparison["decision"], "needs_test")

    def test_reference_rejects_mixed_or_unspecified_recipes(self) -> None:
        honda={"recipe":{"brand":"Honda","model":"front","target":"Airbag"}}
        toyota={"recipe":{"brand":"Toyota","model":"front","target":"Airbag"}}
        with self.assertRaises(ValueError):
            app.validate_reference_recipe([honda,toyota])
        with self.assertRaises(ValueError):
            app.validate_reference_recipe([{"recipe":{}},honda])
        app.validate_reference_recipe([honda,honda])
        app.validate_reference_recipe([honda])

    def test_training_set_acceptance_cannot_activate_candidate(self) -> None:
        for independent, count in [(False, 10), (True, 0), (None, 10)]:
            comparison = app.compare_reference_candidate({
                "normal_acceptance_rate": 1.0,
                "independent_validation": independent, "validation_images": count,
            })
            self.assertFalse(comparison["promote"])
            self.assertIn("independent_holdout_required", comparison["failed_checks"])

    def test_review_event_updates_one_folder_and_keeps_event_count(self) -> None:
        result = app.InspectionResult("NG", 0.2, 1.2, 0.8, [])
        source = np.full((40, 40, 3), 80, np.uint8)
        annotated = source.copy()
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            app, "INSPECTION_REVIEW_DIR", Path(temp_dir)
        ):
            first = app.save_inspection_review(
                source, annotated, result,
                object_box=(2, 2, 38, 38), object_label="Part", object_confidence=0.9,
                source_name="camera_0",
            )
            second = app.save_inspection_review(
                source, annotated, result,
                object_box=(2, 2, 38, 38), object_label="Part", object_confidence=0.9,
                source_name="camera_0", existing_review_id=first["id"],
            )
            manifests = list(Path(temp_dir).glob("*/review.json"))
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(second["event_frame_count"], 2)
        self.assertEqual(len(manifests), 1)

    def test_cvat_sync_uploads_confirmed_event_and_draft_box(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            review_dir = root / "ng_event"
            review_dir.mkdir()
            source = review_dir / "source.jpg"
            source.write_bytes(b"jpeg")
            record = {
                "id": "ng_event", "status": "confirmed_ng", "source_image": str(source),
                "regions": [{"x": 2, "y": 3, "width": 10, "height": 12}],
            }
            app.write_json(review_dir / "review.json", record)

            def fake_cvat(path, **kwargs):
                if path == "/api/tasks" and kwargs.get("method") == "POST":
                    return {"id": 12}
                if path == "/api/tasks/12/data/":
                    return {}
                if path == "/api/tasks/12":
                    return {"labels": [{"id": 4, "name": "NG"}]}
                if path == "/api/tasks/12/annotations/":
                    self.assertEqual(kwargs["payload"]["shapes"][0]["points"], [2.0, 3.0, 12.0, 15.0])
                    return {}
                raise AssertionError(path)

            with (
                mock.patch.object(app, "INSPECTION_REVIEW_DIR", root),
                mock.patch.object(app, "inspection_review_records", return_value=[record]),
                mock.patch.object(app, "cvat_connection_config", return_value={"configured": True, "base_url": "http://cvat", "token": "x", "project_id": None}),
                mock.patch.object(app, "cvat_request", side_effect=fake_cvat),
                mock.patch.object(app, "read_project_config", return_value={"project_name": "Part"}),
            ):
                synced = app.sync_confirmed_reviews_to_cvat()
            saved = app.read_json(review_dir / "review.json")
        self.assertEqual(synced["draft_boxes"], 1)
        self.assertEqual(saved["cvat_sync"]["status"], "synced")
        self.assertEqual(saved["cvat_sync"]["task_id"], 12)
        self.assertEqual(saved["cvat_sync"]["frame_index"], 0)

    def test_cvat_pull_accumulates_manual_boxes_as_yolo_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reviews = root / "reviews"
            archive = root / "archive"
            review_dir = reviews / "event_1"
            review_dir.mkdir(parents=True)
            source = review_dir / "source.jpg"
            cv2.imwrite(str(source), np.zeros((100, 200, 3), dtype=np.uint8))
            record = {
                "id": "event_1", "status": "pending", "source_image": str(source),
                "cvat_sync": {"status": "synced", "task_id": 9, "frame_index": 0},
            }
            app.write_json(review_dir / "review.json", record)

            def fake_cvat(path, **_kwargs):
                if path == "/api/tasks/9":
                    return {"labels": [{"id": 7, "name": "Scratch"}]}
                if path == "/api/tasks/9/annotations/":
                    return {"shapes": [{
                        "type": "rectangle", "frame": 0, "label_id": 7,
                        "points": [20, 10, 120, 60], "outside": False, "source": "manual",
                    }]}
                raise AssertionError(path)

            with (
                mock.patch.object(app, "INSPECTION_REVIEW_DIR", reviews),
                mock.patch.object(app, "CVAT_ANNOTATED_DIR", archive),
                mock.patch.object(app, "cvat_request", side_effect=fake_cvat),
            ):
                result = app.pull_cvat_annotations()
            saved = app.read_json(review_dir / "review.json")
            yolo = (archive / "labels" / "event_1.txt").read_text(encoding="utf-8").strip()
            index = app.read_json(archive / "index.json")

        self.assertEqual(result["imported"], 1)
        self.assertEqual(saved["status"], "confirmed_ng")
        self.assertEqual(saved["cvat_annotations"][0]["label"], "Scratch")
        self.assertEqual(yolo, "0 0.350000 0.350000 0.500000 0.500000")
        self.assertEqual(index["images"], 1)
        self.assertEqual(index["classes"], ["Scratch"])

    def test_ok_first_config_preserves_locator_but_disables_mark_decision(self) -> None:
        detector_path = WEB_APP_DIR.parent / "models" / "Airbag_Mark_20260824_154238_best.pt"
        config = app.normalize_project_config({
            "active_detector_model": str(detector_path),
            "active_locator_model": str(detector_path),
            "use_legacy_defect_signal": True,
        })

        self.assertEqual(config["inspection_strategy"], "normal_reference")
        self.assertEqual(Path(config["active_detector_model"]), detector_path.resolve())
        self.assertEqual(Path(config["active_locator_model"]), detector_path.resolve())
        self.assertTrue(config["inspection_modules"]["part_locator"])
        self.assertTrue(config["inspection_modules"]["normal_reference"])
        self.assertFalse(config["inspection_modules"]["legacy_defect_assist"])
        self.assertFalse(config["use_legacy_defect_signal"])

    def test_legacy_mark_signal_cannot_be_reenabled_by_config_payload(self) -> None:
        config = app.normalize_project_config({
            "use_legacy_defect_signal": True,
            "inspection_modules": {"legacy_defect_assist": True},
        })
        self.assertFalse(config["use_legacy_defect_signal"])
        self.assertFalse(config["inspection_modules"]["legacy_defect_assist"])

    def test_unknown_train_type_is_rejected_before_state_changes(self) -> None:
        before = app.read_project_config()

        response = app.api_start_train(
            train_type="unknown",
            epochs=10,
            imgsz=416,
            resume=False,
            target_class="",
            target_classes="",
            project_name="Should Not Be Saved",
            project_brand="",
            project_model="",
            augmentation_mode="recommended",
            augmentation_config="{}",
        )

        self.assertEqual(response.status_code, 400)
        self.assertFalse(json.loads(response.body)["ok"])
        self.assertEqual(app.read_project_config(), before)

    def test_invalid_augmentation_json_is_rejected_before_state_changes(self) -> None:
        before = app.read_project_config()

        response = app.api_start_train(
            train_type="airbag_detector",
            epochs=10,
            imgsz=416,
            resume=False,
            target_class="Airbag",
            target_classes="Airbag, Mark",
            project_name="Should Not Be Saved",
            project_brand="",
            project_model="",
            augmentation_mode="recommended",
            augmentation_config="{",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid augmentation", json.loads(response.body)["error"])
        self.assertEqual(app.read_project_config(), before)

    def test_normal_reference_status_never_promotes_legacy_log(self) -> None:
        project = {"inspection_strategy": "normal_reference"}
        runtime = {
            "strategy": "normal_reference",
            "ready": False,
            "model": None,
            "source_dataset": "",
            "training_images": 0,
            "trained_at": "",
            "reason": "Build a Normal Reference first.",
        }
        state_before = dict(app.TRAIN_STATE)
        try:
            app.TRAIN_STATE.update(
                {"running": False, "status": "idle", "log_path": "", "train_type": ""}
            )
            with (
                mock.patch.object(app, "refresh_train_state_locked"),
                mock.patch.object(app, "read_project_config", return_value=project),
                mock.patch.object(app, "normal_reference_runtime_status", return_value=runtime),
                mock.patch.object(app, "latest_train_log_path") as latest_log,
            ):
                response = app.api_train_status()
            payload = json.loads(response.body)
            self.assertEqual(payload["status"], "idle")
            self.assertEqual(payload["status_source"], "normal_reference")
            self.assertEqual(payload["log_path"], "")
            self.assertFalse(payload["historical_log"])
            latest_log.assert_not_called()
        finally:
            app.TRAIN_STATE.clear()
            app.TRAIN_STATE.update(state_before)

    def test_camera_worker_refuses_to_start_without_reference(self) -> None:
        project = {"inspection_strategy": "normal_reference"}
        runtime = {"ready": False, "reason": "Build a Normal Reference first."}
        with (
            mock.patch.object(app, "read_project_config", return_value=project),
            mock.patch.object(app, "normal_reference_runtime_status", return_value=runtime),
            mock.patch.object(app, "stop_camera_worker") as stop_worker,
        ):
            with self.assertRaisesRegex(RuntimeError, "Normal Reference"):
                app.start_camera_worker(0, 0.5, False, False)
        stop_worker.assert_not_called()

    def test_ready_generic_runtime_clears_stale_legacy_model_error(self) -> None:
        runtime = {
            "strategy": "normal_reference",
            "ready": True,
            "model": "reference.npz",
            "reason": "",
        }
        state_before = dict(app.STATE)
        worker_before = app.WORKER_THREAD
        try:
            app.STATE.update({"running": False, "status": "model_error", "error": "legacy Airbag model missing"})
            app.WORKER_THREAD = None
            with (
                mock.patch.object(app, "read_project_config", return_value={}),
                mock.patch.object(app, "normal_reference_runtime_status", return_value=runtime),
            ):
                response = app.status()
            payload = json.loads(response.body)
            self.assertEqual(payload["status"], "ready")
            self.assertEqual(payload["error"], "")
        finally:
            app.STATE.clear()
            app.STATE.update(state_before)
            app.WORKER_THREAD = worker_before

    def test_normal_reference_models_are_in_module_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            older = model_dir / "Inspection_20260901.npz"
            active = model_dir / "Inspection_20260902_deep_patch.npz"
            older.write_bytes(b"reference-v1")
            active.write_bytes(b"reference-v2")
            with (
                mock.patch.object(app, "NORMAL_REFERENCE_MODEL_DIR", model_dir),
                mock.patch.object(
                    app,
                    "read_project_config",
                    return_value={"normal_reference": {"active_model": str(active)}},
                ),
            ):
                records = app.normal_reference_run_records()

        self.assertEqual(len(records), 2)
        self.assertTrue(all(record["train_type"] == "normal_reference" for record in records))
        self.assertTrue(next(record for record in records if record["id"] == active.name)["active"])
        self.assertEqual(next(record for record in records if record["id"] == active.name)["engine"], "deep_patch")


if __name__ == "__main__":
    unittest.main()
