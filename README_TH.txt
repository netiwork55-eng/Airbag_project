Industrial Visual Inspection / Airbag_Project
อัปเดต 6 ตุลาคม 2026

เริ่มระบบ: start_web_app.bat แล้วเปิด http://127.0.0.1:8000/inspection
Workflow: Product → Image setup → Part location → Areas → Methods → Test → Production → Review
Train เฉพาะพื้นที่ที่ใช้ AI ไม่บังคับทุกงานผ่านหน้า Train
เป้าหมายตรวจ Defect หลายประเภทชิ้นงาน Brand เป็นเพียงตัวช่วยแยกรุ่น ไม่ใช่เป้าหมายหลัก

คู่มือหลัก: NEW_SYSTEM_TH.md
ผลตรวจและข้อจำกัดล่าสุด: work/renovation-20261006/REPORT_TH.md
การตัดสินใจ KEEP/REFACTOR/REPLACE/REMOVE: work/renovation-20261006/AUDIT_TH.md
คู่มือเครื่องใหม่: MOVE_TO_NEW_MACHINE_TH.txt
คู่มือ backend: BACKEND_HANDOVER_TH.md

ระบบใหม่ใช้ OpenCV สำหรับภาพ/จัดแนว/ตรวจเงื่อนไข และ Anomalib สำหรับความผิดปกติผิว
ใช้ ONNX Runtime เมื่อ export/parity ผ่าน มี PASS / NG / REVIEW / UNABLE_TO_INSPECT กลาง
Candidate แยกจาก Production ต้อง Validation สูตรเดียวกันก่อน Activate และย้อนรุ่นได้
ภาพดีอ้างอิงและภาพเรียนรู้ต้องยืนยัน GOOD; ภาพทดสอบต้องแยกชิ้น/รอบถ่าย
ไม่มีเพดาน 160 ภาพ แต่ข้อมูลมากไม่ใช่หลักฐานความแม่นยำ ต้องวัดกับ GOOD และ NG

ข้อมูลระบบใหม่: inspection_data/
ข้อมูล Dataset เดิม: 05_train_new_object/01_yolov8_exports/
หน้าเดิมสำหรับ Engineering: /engineering (เก็บเพื่อข้อมูล/เปรียบเทียบ ยังไม่ลบโมเดลที่ใช้อยู่)
ใช้ CVAT ที่ติดตั้งเดิมเท่านั้น ส่งภาพที่เลือกจาก Human Review ไม่มีการส่งหรือ retrain อัตโนมัติ

สถานะสำคัญ: ยังไม่รับรองการใช้งานจริงในโรงงาน
PaDiM รับ GOOD 41/42 และแจ้ง NG ผิด 1 ภาพ; PatchCore PASS 28/42, NG 2, REVIEW 12
ผลเป็นภาพ crop จากข้อมูลเดิม ไม่มี verified NG จึงยังวัดการปล่อยชิ้นเสียผ่านไม่ได้
ยังไม่ได้ทดสอบกล้องจริง และ CVAT ติดต่อไม่ได้ขณะตรวจ ไม่มี Candidate ใหม่ถูกเปิดเป็น Production
อ่านรายงานก่อนนำตัวเลขไปอ้างอิง อย่าเรียกผล GOOD-only ว่า Defect accuracy

ติดตั้งใช้ Python 3.12 / runtime ที่ตรึงรุ่นใน requirements-runtime.lock
worker Anomalib แยก .venv-inspection; requirements-inspection.lock ระบุรุ่นที่ตรวจแล้ว
ไม่คัดลอก venv ไปเครื่องใหม่ ให้สร้างใหม่ตามคู่มือ ไม่แก้ production model path เป็นไฟล์ล่าสุด
เก็บ credentials ใน environment/.env ไม่ใส่ source, recipe หรือรายงาน
