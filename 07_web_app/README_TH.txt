Web / Inspection backend — 16 ตุลาคม 2026

รันจากรากโปรเจกต์: ..\venv\Scripts\python.exe 07_web_app/app.py
เปิด http://127.0.0.1:8000/inspection หรือ start_web_app.bat
/ และ /training เข้าหน้า /inspection; /engineering ถูกถอดแล้ว Dataset/Jobs/Models ใช้ service เดิมจากหน้าเดียวกัน

แกนระบบ inspection/: schema, store, pipeline, validation, runtime, api, anomaly_worker, worker_client
Frontend: templates/inspection.html, static/inspection.css, static/inspection.js
ใช้ pipeline เดียวกันในการทดสอบภาพและ Production; ตัดสินผลที่ schema.decision
Anomalib worker แยก process; ไม่มีการเรียนรู้หรือเปลี่ยนโมเดล Production อัตโนมัติ

API หลัก (local /api/vision)
GET /status, /recipes, /images, /reports, /training, /evidence
POST /images: multipart files, label GOOD/NG/UNKNOWN, group เป็นรอบ/ชิ้นเดียวกัน
POST /images/import: folder ของ Dataset เดิมและ label ที่ผู้ใช้ยืนยัน
POST /camera/capture: เก็บหนึ่งภาพและปล่อยกล้อง ภาพเริ่มเป็น UNKNOWN
POST /images/{id}/confirm: label GOOD หรือ NG
POST /recipes: config และ parent (ถ้ามี)
POST /train: recipe_id, area_id, algorithm padim/patchcore, image_ids
POST /training/stop: หยุด Candidate ไม่เปลี่ยน Production
POST /test: recipe_id, image_id → ตรวจและบันทึก evidence
POST /validate: recipe_id, image_ids → metrics/report/eligible/blockers
POST /activate: recipe_id, report_id; POST /rollback
POST /production/start, /production/stop; GET /production/frame
POST /review/{event_id}: status confirmed_ng/false_alarm/needs_review
POST /cvat/send: event_ids ที่เลือกและทบทวนแล้ว
GET /evidence/{event_id}/preview, /maps: แสดงแผนที่หรือข้อมูลแผนที่จริง

Activate ต้องผ่านรายงานของสูตรและไฟล์รุ่นเดียวกัน รวมภาพทดสอบที่ไม่เปลี่ยน label/group
Fixed locator ไม่รองรับกล้องวางเข้า–ยกออก ใช้ Template/Features ที่ตรวจ absence ได้
ผล invalid/error = UNABLE_TO_INSPECT; uncertain = REVIEW ไม่บังคับเป็น PASS/NG
CVAT ใช้ integration เดิม แยก lock จากกล้อง ไม่มี /api/annotation/* ของ Studio ภายในแล้ว

ทดสอบจาก root:
.venv-inspection\Scripts\python.exe -m pytest 07_web_app/tests -q
node 07_web_app/tests/test_live_polling.cjs
node --check 07_web_app/static/inspection.js
API tests ใช้ httpx2 ใน worker environment; ไม่ข้ามชุด API แล้วอ้างว่าผ่านทั้งหมด

ข้อมูล/ข้อจำกัด/ผลจริงดู ../NEW_SYSTEM_TH.md และ ../work/RENOVATION_CONSOLIDATION_REPORT.md

กล้อง Terminal ใช้ 01_camera/airbag_webcam_detection.py เดิม ไม่มี --candidate-recipe
ผลรอบกล้องอ่านจาก /api/inspection/parts ที่ Review → Terminal camera / Legacy history
/start_stream, /stop_stream, /latest_frame.jpg, /status ของเว็บเก่าถูกถอดแล้ว ใช้ /api/vision/status ของ Core
