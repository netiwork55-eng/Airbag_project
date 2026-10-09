# ส่งต่องาน Backend — 6 ตุลาคม 2026

อ่าน NEW_SYSTEM_TH.md ก่อน คู่มือเก่าถูกสำรองใน work/renovation-20261006/before/

## ขอบเขต
FastAPI เดิม app.py mount inspection/api.py ที่ /api/vision และ UI /inspection
recipe ไม่ขึ้นกับ YOLO/Brand/Train page; compatibility shell ยังใช้ dependency เก่าที่ Engineering ต้องใช้
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
real-data experiments/replay/API training scripts และผล JSON อยู่ work/renovation-20261006/
ตัวเลขผ่าน tests ไม่รับรอง camera/defect recall อ่าน REPORT_TH.md สำหรับข้อจำกัด
ไม่มี NG ที่ยืนยัน กล้องไม่ได้ต่อ และ CVAT ติดต่อไม่ได้ขณะทดสอบ
Hash ไม่เป็น security signature; server loopback ไม่ใช่ multi-user authenticated deployment
เปลี่ยน engine/runtime หรือ camera setup ต้อง validation ใหม่ก่อนใช้งานจริง
