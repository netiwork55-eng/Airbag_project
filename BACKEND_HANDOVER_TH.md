# ส่งต่องาน Backend — 16 ตุลาคม 2026

อ่าน NEW_SYSTEM_TH.md และ work/RENOVATION_CONSOLIDATION_REPORT.md ก่อน; source snapshot เก่าย้ายออกนอกโปรเจกต์แล้วตามรายงานหลัก

## ขอบเขต
FastAPI เดิม app.py mount inspection/api.py ที่ /api/vision และ UI /inspection
recipe ไม่ขึ้นกับ YOLO/Brand/Train page; app.py reuse services เดิมของ Dataset/Jobs/Models/Test/Review; /engineering และ legacy web stream ถูกถอดแล้ว
schema.py กำหนด contract/decision; pipeline.py ทำ quality/align/ROI/methods; runtime.py ทำ camera/latest-frame/cycle/evidence
store.py เก็บ atomic JSON, hashes, immutable candidates, activation และ rollback
validation.py กัน reference/training/calibration/same-group ออกจากคะแนน
anomaly_worker.py เรียก official Anomalib และ exporter; worker_client.py ดูแล local subprocess/timeout/cancel

## Invariants ที่ห้ามเปลี่ยนเงียบ ๆ
- required invalid → UNABLE, required NG → NG, uncertainty → REVIEW; ไม่มีผล required ที่ใช้ได้ห้าม PASS
- ทดลอง/Train ไม่เปลี่ยน production.json
- activate ตรวจ exact recipe hash, artifacts, report integrity และ labels/groups
- camera มีเจ้าของเดียว; capture read/release ใน thread เดียว และ timeout ไม่ปลด busy ขณะ thread ยังไม่จบ
- เก็บเฟรมล่าสุดเท่านั้น; source timestamp/age, one event ต่อ stable place/remove cycle
- maps ต้องเป็นของ evidence frame เดียวกัน ไม่วางแผนที่เก่าบนเฟรมใหม่
- ถ้าเก็บ evidence ไม่ได้ ให้หยุดและแจ้ง UNABLE
- Human Review ไม่ auto-send CVAT หรือ retrain; CVAT งานที่เลือกใช้ lock แยกจาก inspection

## ข้อมูลและ compatibility
inspection_data/ เป็น store หลักใหม่; project_config.json/models/normal_reference เป็นระบบเดิม
ยังไม่มี new production.json เพราะไม่มี independent NG ผ่าน validation
อย่าลบข้อมูลเดิมจนตรวจผู้ใช้/recipe/model references และมีรุ่นใหม่ที่ผ่านจริง
ลบ internal Annotation Studio API/helper แล้ว /annotation ส่งไป CVAT เดิม
startup ไม่ auto-repair path หรือเลือก model ล่าสุด; repair_after_move ใช้ชื่อเดิมเท่านั้น

## ทดสอบและสภาพแวดล้อม
Python 3.12, base runtime ../venv, worker .venv-inspection อ่าน torch/torchvision base ผ่าน .pth
รัน .venv-inspection/Scripts/python.exe -m pytest 07_web_app/tests -q
JS: node --check 07_web_app/static/inspection.js และ node 07_web_app/tests/test_live_polling.cjs
ผล JSON/ภาพ/รายงานอยู่ work ตามวันที่; สคริปต์ replay/ทดลองเก่าอยู่ external archive ตามรายงานหลัก ไม่ใช่ runtime
ตัวเลขผ่าน tests ไม่รับรอง camera/defect recall อ่าน REPORT_TH.md สำหรับข้อจำกัด
LifeCam รับภาพจริงแล้ว 14 ต.ค.; Demo ใช้คราบจำลองชิ้นเดียวและ repeatability ยังไม่ผ่าน ไม่รับรอง defect accuracy; CVAT เป็น connector ตามต้องการ
Hash ไม่เป็น security signature; server loopback ไม่ใช่ multi-user authenticated deployment
เปลี่ยน engine/runtime หรือ camera setup ต้อง validation ใหม่ก่อนใช้งานจริง

## จุดเชื่อมที่ต้องรักษา
Terminal เดิมเขียน inspection_cycles; /api/inspection/parts อ่านไฟล์เดียวกันใน Review ของ /inspection
Legacy jobs/config ยังใช้กับ Terminal เมื่อไม่มี Core Production; Candidate Terminal ถูกถอดตามผู้ใช้แล้ว
Core Production ของ Terminal และเว็บใช้ runtime.py/FramePump เดียวกันเมื่อมีสูตรผ่าน Validation
อย่าเพิ่ม worker กล้องใน app.py; ป้องกัน Model/Job/Settings ผ่าน vision_service.busy() และ OS project lease เดิม
Tests ใช้ข้อมูลชั่วคราว ทดสอบ external owner ด้วย subprocess โดยไม่เปิดกล้องหรือแก้ production.json
