# Industrial Visual Inspection — อัปเดต Consolidation 9 ตุลาคม 2026

ระบบปรับใน FastAPI เดิม เปิด http://127.0.0.1:8000/inspection
หน้า / และ /training เข้าหน้าใหม่นี้ ไม่เปิด Candidate ใหม่ทับโมเดลเดิม
Airbag/Toyota เป็นตัวอย่าง ไม่จำกัดประเภทชิ้นงาน และ Brand ไม่ใช่ตัวตัดสิน Defect

## Camera Runtime เดิม

ยังรันด้วย `..\venv\Scripts\python.exe 01_camera\airbag_webcam_detection.py` เหมือนเดิม ไม่มี camera CLI หรือหน้าต่างใหม่
รักษาการเปิดกล้อง, OpenCV window, ภาพสด, กรอบ, FPS, q/Esc และ s/Space ในเส้นทาง Normal Reference
เมื่อมี Production Recipe ของ Core ใหม่ที่ Activate อย่างถูกต้อง ไฟล์เดิมเรียก `inspection.runtime.Runtime` เพื่อใช้ Pipeline, Decision, Cycle และ Evidence ร่วมกับเว็บ
ตรวจรายงานและข้อมูลต้นทางอีกครั้งก่อนเริ่ม ไม่เลือก Candidate เอง ไม่แก้ project_config และไม่ปรับ resolution/model ให้ตรงกันโดยเงียบ ๆ
หาก active Recipe ผิดหรือกล้องไม่ตรง จะหยุดพร้อมเหตุผล ไม่ fallback ไปโมเดลเก่า
ถ้ายังไม่มี Production Recipe ที่ Activate คงเส้นทางโมเดลเดิมไว้ระหว่าง Migration; ณ รอบนี้ยังไม่มี Production Recipe และยังไม่ได้สลับการตรวจจริง
กล้อง Terminal ยังใช้ลูปอ่านภาพเดิม ค่าอายุภาพเริ่มหลัง read สำเร็จ ไม่ใช่ hardware exposure timestamp จึงต้องทดสอบ buffer/latency กับกล้องและโมเดลจริง
Terminal, capture utility เดิม และ Web workflow ใช้ project lease ที่มี OS lock ร่วมกัน ป้องกัน process อื่นใน project แย่งงานกล้อง/เปลี่ยน configuration ระหว่างตรวจ; Web แสดง external owner แต่ไม่แชร์ camera handle หรือผลสดของ Terminal
สามารถทดสอบ Candidate ที่บันทึกแล้วด้วยโปรแกรมเดิม: เพิ่ม `--candidate-recipe recipe_<idจริง> --camera-index <indexจริง>` ใช้ Core เดียวกัน แสดง CANDIDATE TEST - NOT PRODUCTION และเก็บ source=candidate/candidate_sample ไม่ Activate เอง
Candidate แบบวาง–ยกต้องมี Locator; fixed/cropped recipe เดิมยังใช้ Test ภาพได้ แต่ไม่รับรอง live cycle โปรแกรม Core ใช้ resolution ที่ Recipe กำหนดและต้องตรง full-frame reference; legacy mode รักษาค่ากล้องเดิม
ก่อน Production validation ต้องกำหนด Min defect recall ระดับชิ้นงานจาก requirement จริง ค่าเริ่มต้นเว้นว่าง; REVIEW/UNABLE ไม่นับเป็น NG rejection และ object recall ของ detector เป็นคนละเกณฑ์
Engine 2026.10.09.1 ต้อง Validate ใหม่ก่อน Production รายงาน/Recipe/threshold เดิมไม่ได้แก้ย้อนหลัง
Regression รอบล่าสุดผ่าน 327 tests และ 5 subtests รวม mock camera และการชนกันข้าม process จริง; JavaScript 2 suites ผ่าน ไม่ใช่หลักฐานตรวจตำหนิหรือกล้องจริง Browser automation ใช้ไม่ได้รอบนี้; HTTP smoke ผ่าน
รายละเอียด KEEP/MIGRATED/REMOVED/BLOCKED และจุดย้อนกลับ: work/RENOVATION_CONSOLIDATION_REPORT.md
โค้ดใหม่มีผลเมื่อเปิดโปรแกรมครั้งถัดไป ไม่ได้ patch/restart process กล้องหรือ server ที่ผู้ใช้เปิดค้างก่อนแก้

## ขั้นตอนใช้งาน
1. Setup: สร้าง/เลือก Product เลือกภาพชิ้นดีอ้างอิง ตั้งกล้องและคุณภาพภาพ
2. Part location: Template หรือ Features เมื่อชิ้นงานขยับได้ แล้วลาก Areas บนภาพอ้างอิง
3. เลือกวิธีตรวจแต่ละพื้นที่: Reference match, Brightness หรือ Surface defects
4. เฉพาะ Surface defects ต้องเลือกภาพ GOOD และสร้าง Candidate ด้วย Anomalib
5. Test: ใช้ GOOD/NG แยกจากภาพเรียนรู้และรอบถ่ายเดียวกัน ดูผลรายภาพและเกณฑ์ผิดพลาด
6. Activate ได้เมื่อสูตรรุ่นนั้นผ่าน Validation; เปลี่ยนพื้นที่/ภาพอ้างอิง/เกณฑ์ต้องบันทึกและทดสอบใหม่
7. Production: กล้องดูต่อเนื่อง วางเข้า–ยกออกทีละชิ้น บันทึกหนึ่งผลต่อรอบพร้อมหลักฐาน
8. Human review: Confirmed NG, False Alarm, Needs Review; เลือกส่งภาพไป CVAT เดิมเมื่อต้อง annotation
9. ใช้ข้อมูลที่ทบทวนแล้วสร้าง Candidate ใหม่ ไม่มีการ Train หรือเปลี่ยน Production อัตโนมัติ

## วิธีตรวจที่ทำงานแล้ว
| วิธี | การใช้งาน | เทคโนโลยีและข้อจำกัด |
|---|---|---|
| Reference match | รูปร่าง/ชิ้นส่วนที่ต้องมีหลังจัดแนว | OpenCV template matching; ภาพไร้รายละเอียดตรวจไม่ได้ ไม่ใช้แทนการตรวจผิวทุกชนิด |
| Brightness | ช่วงความสว่างเฉลี่ยที่กำหนดได้ | OpenCV; ไม่อ้างว่าเป็นการวัดสีหรือขนาดเชิงมาตรวิทยา |
| Surface defects | ผิวผิดปกติที่มีภาพชิ้นดี | official Anomalib PaDiM / PatchCore; ต้องประเมินกับข้อมูลจริง |
| Part location | จัดตำแหน่ง ROI ก่อนตรวจ | OpenCV template translation หรือ ORB + RANSAC ภายใน tolerance |

หนึ่ง Product มีหลายพื้นที่และใช้ต่างวิธีกัน ไม่บังคับ Train ทุกงาน
ยังไม่มี detector/segmentation ตำหนิใหม่ที่ผ่านการฝึกและทดสอบ เลือก TorchVision ที่มีอยู่แล้วเมื่อมี label ที่เหมาะสม ไม่ติดตั้ง MMDetection ซ้ำ
Fixed position ใช้ทดสอบภาพ/fixture แต่ยังเปิด Production แบบวางเข้า–ยกออกไม่ได้ เพราะต้องมี locator ยืนยันการเข้า/ออก

## ผลตรวจและ runtime
- PASS: ภาพ/พื้นที่ใช้ได้ ทุกพื้นที่บังคับผ่าน
- NG: พบเงื่อนไขไม่ผ่าน/ความผิดปกติตามเกณฑ์
- REVIEW: ตรวจสำเร็จแต่ใกล้ขอบเกณฑ์หรือคำตัดสินยังไม่นิ่ง
- UNABLE_TO_INSPECT: ภาพ แสง ความคม ตำแหน่ง ROI โมเดลไม่พร้อม หรือ inference ล้มเหลว

inspection/schema.py รวมคำตัดสิน ไม่เปลี่ยนข้อผิดพลาดเป็น PASS
Web camera runtime เก็บเฟรมล่าสุดช่องเดียว ไม่สะสมคิวในแอป ตรวจอายุภาพและบันทึกเวลา source; Terminal รักษาลูปอ่านกล้องเดิมตามข้อกำหนดด้านบน
รอผลต่อเนื่องตาม stable_frames ก่อนยืนยัน คงผลรุนแรงกว่าที่นิ่งแล้วจนยกออก หยุดกลางรอบเป็น UNABLE
locator หายเป็นสัญญาณยกออก ต้องพิสูจน์กับพื้นว่าง มือบัง และชิ้นขยับจริงก่อนหน้างาน

## Candidate / Production
Recipe เก็บ Product, camera, quality, locator, areas, methods, thresholds, decision และ validation limits
บันทึกเป็นรุ่น immutable; hash ผูก reference/model/metadata และรายงาน Validation
ไฟล์เปลี่ยนหรือข้อมูลไม่ตรงจะไม่ทำงานต่อเงียบ ๆ; ภาพ/label/group เปลี่ยนต้องทดสอบใหม่
Production เก็บ pointer ปัจจุบันและรุ่นก่อนเพื่อ Rollback การทดลองไม่เปลี่ยน pointer
Hash ตรวจความสอดคล้อง ไม่ใช่ลายเซ็นดิจิทัลหรือการป้องกันผู้มีสิทธิ์แก้ไฟล์ทั้งหมด
หลังเปลี่ยน code/runtime/camera/สภาพหน้างานต้องทดสอบใหม่ รายงานเดิมไม่รับรองชุดติดตั้งใหม่

## Anomalib
worker แยก process และ .venv-inspection จากเว็บ/กล้อง ใช้ Anomalib core 2.6.2
ใช้ official TorchVision ResNet18 V1 ที่มีใน cache และตรวจ hash ไม่รับโมเดล pickle จาก upload
ใช้ Anomalib ทำ fitting/coreset/anomaly map ไม่เขียน PatchCore/PaDiM เอง
ไม่มีเพดาน 160 ภาพ โหลดภาพฝึกทีละภาพ แต่ feature ภายในโมเดลยังใช้หน่วยความจำตามจำนวนข้อมูล
ขั้นต่ำ 20 GOOD ไม่ซ้ำและ 5 capture groups แยก fit/calibration; กลุ่ม reference ไม่อยู่ calibration
capture group หมายถึงชิ้น/รอบถ่ายเดียวกัน ต้องกำหนดให้ถูก ชื่อภาพต่างกันไม่ได้พิสูจน์ความเป็นอิสระ
threshold เริ่มจาก calibration GOOD และ REVIEW margin ต้องทดสอบ NG ก่อน Production
ใช้ ONNX Runtime เมื่อ export ด้วย PyTorch และตรวจคะแนน แผนที่ และคำตัดสิน calibration ตรงกับ native ภายใน tolerance
ถ้าไม่ผ่านใช้ native พร้อมเหตุผล; parity ไม่ใช่หลักฐานความแม่นยำ Defect

## ข้อมูลและ CVAT
inspection_data/ เก็บ images, recipes, models, tests, evidence, reviews, training และ production.json เมื่อเปิดใช้
ภาพ deduplicate ตาม decoded pixels; เก็บ source image, timestamp, ผล/เหตุผล, recipe hash, area scores/maps เมื่อมี
Human Review ไม่แก้ label ภาพต้นฉบับอัตโนมัติ เพราะ verdict ของเหตุการณ์มีบริบทต่างกัน
นำเข้าจาก Dataset เดิมได้ การลบ Dataset เดิมไม่ได้ลบภาพที่นำเข้าหรือทำให้โมเดลลืมข้อมูล
ไม่ลบหลักฐานหรือโมเดลที่ยังอ้างอิงเพื่อทำให้หน้าว่าง ต้องกำหนด retention/backup สำหรับหน้างาน
ใช้ CVAT เดิมผ่าน API ไม่มี Annotation Studio ซ้ำ; ส่งเฉพาะภาพที่เลือก/ทบทวนแล้ว
CVAT ใช้ lock แยกจาก inspection ไม่ครอบครองกล้องหรือหยุด Production

## ผลที่พิสูจน์แล้วและข้อจำกัด
- OpenCV API ทดสอบ upload → recipe → validate → activate → evidence → review และ CVAT adapter จำลอง
- Anomalib เรียนรู้ภาพจริง 172 ภาพ (fit 138/calibration 34) ทดสอบ GOOD แยก 42 ภาพจาก manifest เดิม
- PaDiM: PASS 41, NG 1; PatchCore: PASS 28, NG 2, REVIEW 12; ทั้งคู่ไม่มี UNABLE ในชุดนี้
- เป็นภาพ crop แล้ว ไม่พิสูจน์ raw camera/locator และไม่ได้แสดงว่าดีกว่าโมเดลเดิม
- ไม่มี verified NG จึงวัด false accept/defect recall ไม่ได้ ไม่มีสูตรใหม่เปิดเป็น Production
- replay วางเข้า–ยกออกและการปล่อยกล้องจำลองผ่าน ยังไม่ทดสอบกล้องจริงหรือรันยาว
- มี pypylon ใน runtime; probe ที่รายงานไว้วันที่ 8 ต.ค. ไม่พบ Basler device รอบพัฒนานี้ไม่ได้เปิด/ตรวจกล้องจริง จึงยังไม่รับรอง adapter กับ hardware
- CVAT localhost:8080 ติดต่อ/ยืนยันตัวตนไม่ได้ตอนตรวจ จึงยังไม่ผ่าน annotation round trip จริง

## ติดตั้งและสิทธิ์การใช้งาน
เปิด start_web_app.bat; server ใช้ loopback ไม่เพิ่มพอร์ตภายนอก
เครื่องใหม่ใช้ Python 3.12 และ setup_new_machine.bat ดู MOVE_TO_NEW_MACHINE_TH.txt
requirements-runtime.lock ตรึง runtime เดิม; requirements-inspection.lock ตรึง dependency ใหม่
ONNX Runtime ปิด telemetry; worker ไม่มี cloud logger หรือส่งภาพ/โมเดลไป cloud
license ของ library กับ pretrained weights อาจต่างกัน: TorchVision ระบุให้ตรวจเงื่อนไขข้อมูลที่ใช้สร้าง weights สำหรับการใช้งานนั้น ไม่ถือ BSD ของ code เป็นสิทธิ์ใช้ weights เชิงพาณิชย์โดยอัตโนมัติ

## ระบบเดิมที่ยังอยู่
/engineering เก็บ workflow/data/model เดิมเพื่ออ่านและเปรียบเทียบจนสูตรใหม่ผ่านจริง
DeepPatch/YOLO ไม่เป็นข้อบังคับของ engine ใหม่ แต่ compatibility shell ยัง import dependency เก่า
ลบ Annotation Studio API/helper ที่ซ้ำ CVAT และไม่มีผู้เรียกแล้ว ไม่ลบ annotation/data เดิม
หยุด auto-repair ตอน startup ที่อาจเลือกโมเดลอื่น ไม่มีการแทน active model ด้วยไฟล์ล่าสุดอัตโนมัติ
รายละเอียด: work/renovation-20261006/AUDIT_TH.md และ REPORT_TH.md
