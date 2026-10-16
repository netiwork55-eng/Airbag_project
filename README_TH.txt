Airbag_Project — จุดเริ่มใช้งานเดียว
อัปเดต 16 ตุลาคม 2026

เปิดกล้อง: ใช้โปรแกรมเดิมจาก root
..\venv\Scripts\python.exe 01_camera\airbag_webcam_detection.py
ภาพสด/กรอบ/FPS อยู่หน้าต่าง OpenCV เดิม ไม่ใช้ --candidate-recipe

เปิดพื้นที่จัดการ: start_web_app.bat → http://127.0.0.1:8000/inspection
Setup/Dataset → Test/Validation/Jobs/Models → Production → Evidence/Review
ดูผลจากกล้องเดิม: Review → Terminal camera / Legacy history → Refresh
อ่านภาพและผลรอบวาง–ยกจาก 03_dataset/inspection_cycles โดยตรง ไม่ทำสำเนา
NG จากเครื่องไม่ใช่ NG ที่มนุษย์ยืนยัน ไม่มี live preview ของ Terminal บนเว็บ
ไม่เปิดกล้องสองโปรแกรมพร้อมกัน Shared ownership ป้องกันการแย่งงานในโปรเจกต์

สถานะจริง
- ยังไม่มี Core Production ที่ผ่าน Validation และ Activate
- Terminal จึงยังใช้ Normal Reference เดิมตาม 07_web_app/project_config.json
- เมื่อมี Production ที่ผ่านจริง Terminal เดิมรองรับ Shared Core อยู่แล้ว
- Candidate ทดสอบภาพผ่าน /inspection ข้อมูล Candidate เดิมยังอยู่
- ถอด /engineering และ legacy web stream แล้ว ไม่ใช้ /start_stream หรือ /latest_frame.jpg
- LifeCam เคยรับภาพจริง 1920×1080 เมื่อ 14 ต.ค.; ไม่ใช่การยืนยันว่ากล้องพร้อมทุกครั้ง
- Demo คราบเป็นคราบจำลองบนชิ้นเดียวกัน และ GOOD re-placement ยังมี alignment failure
- ยังไม่รับรอง Defect accuracy/การใช้งานโรงงาน ไม่ Train/Activate/เปลี่ยน threshold อัตโนมัติ

คู่มือปัจจุบัน: NEW_SYSTEM_TH.md (รวมแผนที่ทั้งโฟลเดอร์และความสัมพันธ์)
สถานะงาน/ข้อจำกัด/ผลทดสอบ: work/RENOVATION_CONSOLIDATION_REPORT.md
Backend: BACKEND_HANDOVER_TH.md; ย้ายเครื่อง: MOVE_TO_NEW_MACHINE_TH.txt
รายงานลงวันที่อื่นใน work เป็นหลักฐานย้อนหลัง ไม่ใช่คำสั่งเปิดระบบปัจจุบัน
สำเนาโค้ด/สคริปต์ทดลองย้อนหลังย้ายออกไป Airbag_Project_Backups แล้วตามรายงานหลัก

Dataset/Model/Evidence เดิมเป็นข้อมูลสำคัญ ไม่ลบเพียงเพราะมีชื่อเก่า
.venv-inspection เป็น worker environment ที่ใช้งานจริง แยกจาก ../venv ของกล้อง/เว็บ
.git ใช้ย้อนการแก้ Source ไม่ใช่สำรองข้อมูลที่ Git-ignore ต้องสำรองข้อมูลแยก
