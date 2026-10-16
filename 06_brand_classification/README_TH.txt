Brand Classification — โมดูลเสริมเดิม
อัปเดต 16 ตุลาคม 2026

Brand ใช้ช่วยแยกรุ่น/ประเภทชิ้นงานเท่านั้น ไม่ตัดสิน Defect และไม่ใช่เงื่อนไขบังคับของ inspection recipe ใหม่
ผู้ใช้เลือก Product/Recipe ใน /inspection; ไม่จำเป็นต้อง Train Brand ก่อนตรวจ
ยังเก็บโมเดล/ข้อมูล/เครื่องมือเดิมเพราะมีงานเก่าอ้างอิง ไม่เปลี่ยน active_brand_model อัตโนมัติ

01_brand_exports: ข้อมูลต้นทาง; 02_unknown_background_images: Unknown จริง
03_output_dataset: classification data/runs; 04_trained_models: โมเดลเดิม
โมเดลเดิมมีคลาส Honda/Toyota/Unknown ตาม contract ที่ตรวจไว้ ต้องตรวจใหม่เมื่อเปลี่ยนโมเดล
การจำแนกแบรนด์ถูกไม่ใช่หลักฐานว่าจับรอยขีด คราบ รอยบุบ หรือชิ้นส่วนขาดได้

เครื่องมือเดิม train_brand_classifier.py --prepare-only หรือ --epochs 50
เรียกผ่าน ..\venv\Scripts\python.exe จากตำแหน่งที่ script รองรับ
อย่ารัน CLI ซ้อนกล้อง/Train เว็บ เพราะ CLI ไม่ได้แชร์ lock ใน process เว็บ
ค่าระบบเดิมอยู่ ../07_web_app/project_config.json ยังใช้กับ Terminal เดิมเมื่อไม่มี validated Core Production; /engineering ถูกถอดแล้ว
ไม่มีการ Train Brand ใหม่ในรอบ renovation นี้

การตรวจใหม่และข้อจำกัด: ../NEW_SYSTEM_TH.md
หลักฐานการทดสอบล่าสุด: ../work/RENOVATION_CONSOLIDATION_REPORT.md
