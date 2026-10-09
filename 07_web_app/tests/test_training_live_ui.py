"""Regression coverage for the live-training workspace layout and state wiring."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


WEB_APP_DIR = Path(__file__).resolve().parents[1]
TEMPLATE = (WEB_APP_DIR / "templates" / "training.html").read_text(encoding="utf-8")
THEME = (WEB_APP_DIR / "static" / "training-theme.css").read_text(encoding="utf-8")
CONFIGURE = (WEB_APP_DIR / "static" / "training-configure.css").read_text(encoding="utf-8")
STYLE = (WEB_APP_DIR / "static" / "style.css").read_text(encoding="utf-8")
REDESIGN = (WEB_APP_DIR / "static" / "training-redesign.css").read_text(encoding="utf-8")
POLISH = (WEB_APP_DIR / "static" / "training-polish-v2.css").read_text(encoding="utf-8")
RESULTS = (WEB_APP_DIR / "static" / "training-results-v2.css").read_text(encoding="utf-8")


class TrainingLiveUiTests(unittest.TestCase):
    def test_user_workflow_starts_at_dataset_and_has_four_steps(self) -> None:
        self.assertNotIn('data-sidebar-step="project"', TEMPLATE)
        self.assertNotIn('data-workflow-step="project"', TEMPLATE)
        self.assertIn('id="project-setup"', TEMPLATE)
        self.assertIn('hidden aria-hidden="true"', TEMPLATE)
        workflow = re.findall(
            r'class="workspace-signal-item" data-workflow-step="([^"]+)"', TEMPLATE
        )
        self.assertEqual(workflow, ["dataset", "configure", "training", "results"])
        self.assertIn('ขั้นตอน 1 จาก 4 · เลือกหรือเพิ่ม Dataset', TEMPLATE)

    def test_dataset_creation_owns_visible_job_identity(self) -> None:
        self.assertIn('id="newFolderName"', TEMPLATE)
        self.assertIn('id="newDatasetBrand"', TEMPLATE)
        self.assertIn('id="newDatasetModel"', TEMPLATE)
        self.assertIn('ชื่อชิ้นงาน / Dataset', TEMPLATE)
        self.assertIn('projectConfig.brand = metadata.brand || "";', TEMPLATE)
        self.assertIn('projectConfig.model = metadata.model || "";', TEMPLATE)

    def test_live_state_is_derived_from_backend_busy_state(self) -> None:
        self.assertIn(
            'document.body.classList.toggle("train-live-view", studioView === "training" && busy);',
            TEMPLATE,
        )
        self.assertIn(
            'document.body.classList.toggle("train-review-idle", studioView === "training" && !busy && !referenceComplete);',
            TEMPLATE,
        )
        self.assertIn("if (busy) {\n          trainReviewPreview.hidden = true;", TEMPLATE)

    def test_reference_completion_returns_to_original_train_review(self) -> None:
        self.assertNotIn('id="referenceTrainingResult"', TEMPLATE)
        self.assertNotIn("reference-result-metrics", THEME)
        self.assertIn("showReferenceTrainingCompletion(payload, state, notifyCompletion);", TEMPLATE)
        self.assertIn('inspection-last-completed-training', TEMPLATE)
        self.assertIn('id="trainCompleteModal"', TEMPLATE)
        self.assertIn("openTrainCompletePopup", TEMPLATE)
        self.assertIn("train-complete-modal", THEME)
        self.assertIn('referenceTrainingState.completed = false;', TEMPLATE)
        self.assertIn('referenceTrainingState.prepareNew = true;', TEMPLATE)
        self.assertNotIn('state.status === "finished") {\n            const justFinished = referenceTrainingState.running;\n            referenceTrainingState.running = false;', TEMPLATE)
        self.assertNotIn("body.studio-view-training.reference-training-complete", THEME)
        self.assertIn("referenceTrainingDetail.hidden = true;", TEMPLATE)

    def test_busy_training_does_not_lock_navigation_to_train_page(self) -> None:
        sync_view = re.search(
            r"function syncStudioView\(.*?\) \{(?P<body>.*?)\n      \}",
            TEMPLATE,
            re.DOTALL,
        )
        self.assertIsNotNone(sync_view)
        self.assertNotIn('if (busy) {\n          setStudioView("training")', sync_view.group("body"))
        running_branch = re.search(
            r'if \(state\.running \|\| state\.status === "training" \|\| state\.status === "stopping"\) \{(?P<body>.*?)return true;',
            TEMPLATE,
            re.DOTALL,
        )
        self.assertIsNotNone(running_branch)
        self.assertNotIn('setStudioView("training")', running_branch.group("body"))
        self.assertIn("Train ทำงานเบื้องหลัง", TEMPLATE)

    def test_reference_progress_bar_uses_backend_width(self) -> None:
        rule = re.search(
            r"body\.studio-view-training\.reference-training-busy \.progress-track > span \{(?P<body>.*?)\}",
            THEME,
            re.DOTALL,
        )
        self.assertIsNotNone(rule)
        self.assertNotIn("width: 34%", rule.group("body"))
        self.assertIn("animation: reference-progress-heartbeat", rule.group("body"))
        self.assertIn("Width remains the real backend value", rule.group("body"))
        self.assertIn("trainProgressBar.style.width = `${stageProgress}%`;", TEMPLATE)

    def test_home_page_is_removed_and_dataset_is_the_default(self) -> None:
        self.assertNotIn('id="training-home"', TEMPLATE)
        self.assertNotIn('data-studio-view="home"', TEMPLATE)
        self.assertNotIn('data-studio-target="home"', TEMPLATE)
        self.assertNotIn('>Home<', TEMPLATE)
        self.assertIn('let studioView = "dataset";', TEMPLATE)
        self.assertIn('let restoredView = "dataset";', TEMPLATE)
        self.assertIn('id="trainingSidebarBrandButton"', TEMPLATE)

    def test_selected_upload_files_remain_readable_in_light_theme(self) -> None:
        self.assertIn('html[data-training-theme="light"] body.studio-view-dataset .upload-selection-chips b', POLISH)
        self.assertIn('color: #173149 !important', POLISH)
        self.assertIn('.upload-clear-button:not([hidden])', POLISH)
        self.assertIn('position: static !important', POLISH)
        self.assertIn('width: auto !important', POLISH)

    def test_candidate_rejection_and_direct_cvat_are_visible_to_user(self) -> None:
        self.assertIn('Train เสร็จ · ทดสอบโมเดลใน Results', TEMPLATE)
        self.assertIn('โมเดลที่ใช้อยู่ยังไม่เปลี่ยน', TEMPLATE)
        self.assertIn('/api/inspection/cvat/status', TEMPLATE)
        self.assertIn('/api/inspection/cvat/sync', TEMPLATE)
        self.assertIn('draft_boxes', TEMPLATE)

    def test_results_prioritizes_human_review_with_evidence_on_right(self) -> None:
        self.assertIn('id="inspectionReviewTotal"', TEMPLATE)
        self.assertIn('id="inspectionReviewAnnotated"', TEMPLATE)
        self.assertIn('body.studio-view-training #inspectionReviewPanel', RESULTS)
        self.assertIn('grid-template-columns: minmax(0, 7fr) minmax(0, 3fr)', RESULTS)
        self.assertIn('#inspectionReviewPanel', RESULTS)
        self.assertIn('grid-column: 1 !important', RESULTS)
        self.assertIn('#evaluationWorkspace', RESULTS)
        self.assertIn('#training-history-section.history-panel', RESULTS)
        # The backend bridge owns remote pulls; page polling reads local state.
        self.assertNotIn('fetch("/api/inspection/cvat/pull"', TEMPLATE)
        self.assertIn('/api/inspection/cvat/status', TEMPLATE)

    def test_busy_train_page_uses_one_full_width_column(self) -> None:
        rule = re.search(
            r"body\.studio-view-training\.train-is-busy \.status-layout\s*\{(?P<body>.*?)\}",
            THEME,
            re.DOTALL,
        )
        self.assertIsNotNone(rule)
        self.assertIn("grid-template-columns: minmax(0, 1fr) !important", rule.group("body"))
        self.assertIn("width: 100% !important", rule.group("body"))

    def test_busy_state_hides_idle_review_and_history(self) -> None:
        hidden_rule = re.search(
            r"body\.studio-view-training\.train-is-busy \.history-panel,.*?"
            r"body\.studio-view-training\.train-is-busy #trainReviewPreview,.*?"
            r"body\.studio-view-training\.train-is-busy \.evaluation-workspace\s*\{(?P<body>.*?)\}",
            THEME,
            re.DOTALL,
        )
        self.assertIsNotNone(hidden_rule)
        self.assertIn("display: none !important", hidden_rule.group("body"))

    def test_live_dashboard_prioritizes_progress_and_hides_duplicate_overflow(self) -> None:
        dashboard = re.search(
            r"body\.studio-view-training\.train-is-busy \.status-dashboard\s*\{(?P<body>.*?)\}",
            THEME,
            re.DOTALL,
        )
        progress = re.search(
            r"body\.studio-view-training\.train-is-busy \.training-progress-card\s*\{(?P<body>.*?)\}",
            THEME,
            re.DOTALL,
        )
        context = re.search(
            r"body\.studio-view-training\.train-is-busy \.train-context-strip\s*\{(?P<body>.*?)\}",
            THEME,
            re.DOTALL,
        )
        track = re.search(
            r"body\.studio-view-training\.train-is-busy \.progress-track\s*\{(?P<body>.*?)\}",
            THEME,
            re.DOTALL,
        )
        self.assertIsNotNone(dashboard)
        self.assertIsNotNone(progress)
        self.assertIsNotNone(context)
        self.assertIsNotNone(track)
        self.assertIn("order: 2", progress.group("body"))
        self.assertIn('"quality" !important', dashboard.group("body"))
        self.assertIn("height: auto !important", dashboard.group("body"))
        self.assertIn("max-height: none !important", dashboard.group("body"))
        self.assertIn("display: none !important", context.group("body"))
        self.assertIn("height: 12px !important", track.group("body"))
        self.assertIn(".live-quality-panel", THEME)
        self.assertIn(".live-quality-metrics", THEME)
        self.assertIn(".live-quality-chart", THEME)
        raw_log = re.search(
            r"body\.studio-view-training\.train-is-busy \.raw-log-store\s*\{(?P<body>.*?)\}",
            THEME,
            re.DOTALL,
        )
        self.assertIsNotNone(raw_log)
        self.assertIn("display: none !important", raw_log.group("body"))

    def test_busy_geometry_has_one_stylesheet_owner(self) -> None:
        self.assertIn("Live Train layout contract", THEME)
        self.assertNotIn("body.studio-view-training.train-is-busy .status-dashboard", STYLE)
        self.assertNotIn("body.studio-view-training.train-is-busy .status-dashboard", REDESIGN)
        self.assertNotIn("body.studio-view-training.train-is-busy .validation-snapshot", REDESIGN)
        self.assertIn("body.studio-view-training:not(.train-is-busy) .status-layout", STYLE)
        self.assertIn("body.studio-view-training:not(.train-is-busy) .status-layout", REDESIGN)

    def test_live_copy_is_thai_and_latest_theme_is_loaded(self) -> None:
        self.assertNotIn("Training process is currently running.", TEMPLATE)
        self.assertIn("สถานะการ Train", TEMPLATE)
        self.assertIn("กำลัง Train", TEMPLATE)
        self.assertRegex(TEMPLATE, r'/static/training-theme\.css\?v=\d+')
        self.assertIn('/static/style.css?v=162', TEMPLATE)
        self.assertIn('/static/training-redesign.css?v=115', TEMPLATE)

    def test_configure_layout_has_one_final_cascade_owner(self) -> None:
        self.assertNotIn("training-configure-fix.css", TEMPLATE)
        self.assertFalse((WEB_APP_DIR / "static" / "training-configure-fix.css").exists())
        theme_href = re.search(r'/static/training-theme\.css\?v=\d+', TEMPLATE)
        configure_href = re.search(r'/static/training-configure\.css\?v=\d+', TEMPLATE)
        self.assertIsNotNone(theme_href)
        self.assertIsNotNone(configure_href)
        self.assertLess(configure_href.start(), theme_href.start())
        self.assertNotIn("QA v29 — Configure layout", THEME)
        self.assertIn("Configure screen final layout contract", CONFIGURE)
        self.assertIn("position: static !important", CONFIGURE)
        self.assertIn("gap: 6px !important", CONFIGURE)

    def test_readiness_and_augmentation_modals_are_localized(self) -> None:
        self.assertIn("รายละเอียดความพร้อมก่อน Train", TEMPLATE)
        self.assertIn("ตรวจความพร้อม", TEMPLATE)
        self.assertIn("เพิ่มความหลากหลายของภาพอย่างปลอดภัย", TEMPLATE)
        self.assertIn("ใช้การตั้งค่านี้", TEMPLATE)
        self.assertNotIn("Custom values are checked and limited before training.", TEMPLATE)
        self.assertNotIn("Recommended defaults restored in this preview.", TEMPLATE)

    def test_modal_layout_keeps_actions_visible(self) -> None:
        self.assertIn("max-height: min(860px, calc(100dvh - 44px)) !important", THEME)
        self.assertIn("position: sticky !important", THEME)
        self.assertIn("grid-template-rows: auto minmax(0, 1fr) !important", THEME)

    def test_laptop_layout_does_not_squeeze_home_or_train_cards(self) -> None:
        self.assertIn("@media (min-width: 821px) and (max-width: 1280px)", THEME)
        self.assertIn("grid-template-columns: repeat(3, minmax(0, 1fr)) !important", THEME)
        self.assertIn("white-space: normal !important", THEME)

    def test_wide_home_overview_cannot_overflow_into_lower_cards(self) -> None:
        self.assertIn("grid-template-rows: 80px 202px 106px 360px !important", THEME)
        self.assertIn("row-gap: 18px !important", THEME)
        self.assertIn("grid-template-rows: 18px 80px !important", THEME)
        self.assertIn("height: 106px !important", THEME)
        self.assertIn("overflow: hidden !important", THEME)

    def test_sidebar_rows_and_icon_frames_share_one_contract(self) -> None:
        self.assertIn("grid-template-columns: minmax(0, 1fr) !important", THEME)
        self.assertIn("inline-size: 100% !important", THEME)
        self.assertIn("flex: 0 0 34px !important", THEME)


if __name__ == "__main__":
    unittest.main()
