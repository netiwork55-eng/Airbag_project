Data และเครื่องมือเรียนรู้เดิม — 6 ตุลาคม 2026

Workflow หลักใหม่อยู่ /inspection ไม่บังคับ Train ทุกงาน
ข้อมูลในโฟลเดอร์นี้ยังใช้ได้ผ่าน Setup → นำเข้า Dataset เดิม
01_yolov8_exports: ภาพ/export/metadata; 02_background_images: ภาพพื้นหลัง detector
03_output: combined data/checkpoints/run outputs; 04_camera_collection: ภาพจากหน้างาน
ห้ามลบทั้งหมดเพื่อแก้หน้าค้าง เพราะมีต้นฉบับและหลักฐานการฝึก

ระบบตรวจใหม่เก็บภาพนำเข้าและโมเดลที่ ../inspection_data/
Surface defects ใช้ official Anomalib PaDiM/PatchCore เฉพาะภาพ GOOD ที่ยืนยันแล้ว
Reference match/Brightness ใช้ OpenCV ไม่ต้อง Train
เลือกข้อมูลตามรุ่นชิ้นงานและสภาพตรวจ ไม่ผสมหลายรุ่นเพียงเพื่อเพิ่มจำนวน
Background ไม่ใช่ภาพ GOOD ของชิ้นงาน ให้ระบุ UNKNOWN จนตรวจสอบ

ไม่มีเพดาน 160 ภาพสำหรับการเรียนรู้ใหม่
ขั้นต่ำ 20 ภาพไม่ซ้ำและ 5 capture groups; แยก fit/calibration/validation ตามชิ้นหรือรอบถ่าย
ภาพใกล้กันจากชิ้นเดียวกันต้องอยู่กลุ่มเดียวกัน ไม่ตั้งชื่อใหม่เพื่อให้ดูเป็นภาพทดสอบอิสระ
ใช้ NG ที่ยืนยันสำหรับวัด false accept และ defect recall ไม่ใช้เพียงจำนวน GOOD ผ่าน
เปลี่ยน ROI/ภาพอ้างอิง/การจัดแนว/ภาพเข้าแล้วต้องสร้าง Candidate ที่ตรงกันและทดสอบใหม่

YOLO export เดิมยังใช้กับ Engineering ได้เมื่อมี label ที่ตรงงาน:
My_Export/data.yaml, train/images, train/labels, valid/images, valid/labels
อ่าน class names จาก data.yaml ไม่สมมติ class ID ข้ามชุดข้อมูล
ตัวเตรียมข้อมูลเดิม: ../02_training/train_from_yolov8_exports.py
ไม่ได้เป็น engine บังคับของระบบใหม่ และผล Airbag/Brand ไม่ใช่คะแนน Defect

ลบ Dataset เดิมผ่าน Engineering จะย้ายถังขยะก่อน ลบถาวรต้องยืนยัน ID
โมเดลที่เรียนรู้ไปแล้วไม่ลืมข้อมูลเมื่อ Dataset ถูกลบ
ภาพที่นำเข้า inspection_data และ evidence มีวงจรชีวิตแยก ต้องไม่ลบสิ่งที่ recipe อ้างอิง
คู่มือหลัก ../NEW_SYSTEM_TH.md; ผลล่าสุด ../work/renovation-20261006/REPORT_TH.md
