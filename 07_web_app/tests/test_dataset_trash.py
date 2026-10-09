"""Permanent deletion only affects explicitly confirmed entries in the trash."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
import asyncio


class DatasetTrashTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "trash"
        self.entry = self.root / "deleted_example"
        self.entry.mkdir(parents=True)
        (self.entry / "image.jpg").write_bytes(b"test")
        self.outside = Path(self.temp.name) / "active"
        self.outside.mkdir()
        (self.outside / "keep.jpg").write_bytes(b"keep")
        self.patches = [
            mock.patch.object(app, "DELETED_DATASETS_DIR", self.root),
            mock.patch.object(app, "dataset_mutation_blocked_response", return_value=None),
            mock.patch.object(app, "clear_analysis_cache"),
        ]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_delete_removes_only_confirmed_entry(self):
        response = app.api_permanently_delete_dataset(self.entry.name, {"confirm_id": self.entry.name})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(json.loads(response.body)["permanent"])
        self.assertFalse(self.entry.exists())
        self.assertTrue((self.outside / "keep.jpg").exists())

    def test_http_delete_contract_and_repeated_request(self):
        async def check():
            for expected in (200, 404):
                messages = []
                async def receive():
                    return {"type": "http.request", "body": json.dumps({"confirm_id": self.entry.name}).encode(), "more_body": False}
                async def send(message):
                    messages.append(message)
                await app.app({"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "DELETE",
                    "scheme": "http", "path": f"/api/deleted-dataset-folders/{self.entry.name}", "query_string": b"",
                    "headers": [(b"host", b"127.0.0.1"), (b"content-type", b"application/json")], "server": ("127.0.0.1", 80), "client": ("127.0.0.1", 1234), "root_path": ""}, receive, send)
                self.assertEqual(messages[0]["status"], expected)
        asyncio.run(check())

    def test_requires_matching_confirmation(self):
        response = app.api_permanently_delete_dataset(self.entry.name, {"confirm_id": "wrong"})
        self.assertEqual(response.status_code, 400)
        self.assertTrue(self.entry.exists())

    def test_rejects_paths_outside_trash(self):
        for name in ("..", "../active", "..\\active", str(self.outside), "."):
            with self.subTest(name=name):
                response = app.api_permanently_delete_dataset(name, {"confirm_id": name})
                self.assertEqual(response.status_code, 400)
        self.assertTrue((self.outside / "keep.jpg").exists())

    def test_missing_entry_returns_not_found(self):
        response = app.api_permanently_delete_dataset("missing", {"confirm_id": "missing"})
        self.assertEqual(response.status_code, 404)

    def test_busy_system_blocks_deletion(self):
        with mock.patch.object(app, "dataset_mutation_blocked_response", return_value=app.JSONResponse({"ok": False}, status_code=409)):
            response = app.api_permanently_delete_dataset(self.entry.name, {"confirm_id": self.entry.name})
        self.assertEqual(response.status_code, 409)
        self.assertTrue(self.entry.exists())

    def test_file_lock_returns_error_without_success(self):
        with mock.patch.object(app.shutil, "rmtree", side_effect=PermissionError("locked")):
            response = app.api_permanently_delete_dataset(self.entry.name, {"confirm_id": self.entry.name})
        self.assertEqual(response.status_code, 409)
        self.assertFalse(json.loads(response.body)["ok"])

    def test_redirected_entry_is_rejected(self):
        with mock.patch.object(app.os.path, "isjunction", return_value=True):
            response = app.api_permanently_delete_dataset(self.entry.name, {"confirm_id": self.entry.name})
        self.assertEqual(response.status_code, 400)
        self.assertTrue(self.entry.exists())

    def test_cleanup_preserves_runs_and_unrelated_datasets(self):
        exports = Path(self.temp.name) / "01_yolov8_exports"
        output = exports.parent / "03_output"
        affected = output / "old_yolov8_combined"
        unaffected = output / "current_yolov8_combined"
        for directory, source in ((affected, "deleted"), (unaffected, "current")):
            (directory / "train").mkdir(parents=True)
            (directory / "train" / "image.jpg").write_bytes(b"image")
            (directory / "runs").mkdir()
            (directory / "runs" / "best.pt").write_bytes(b"model")
            (directory / "source_manifest.json").write_text(json.dumps({"files": [{"group": "exports", "path": source + "/image.jpg"}]}))
        with mock.patch.object(app, "AIRBAG_EXPORTS_DIR", exports):
            removed = app.cleanup_deleted_dataset_artifacts("deleted")
        self.assertEqual(len(removed), 2)
        self.assertFalse((affected / "train").exists())
        self.assertTrue((affected / "runs" / "best.pt").exists())
        self.assertTrue((unaffected / "train" / "image.jpg").exists())
