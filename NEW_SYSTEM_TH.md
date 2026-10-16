# Industrial Visual Inspection — สถานะและทางใช้งาน 16 ตุลาคม 2026

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
กล้อง Terminal โหมด Core ใช้ OpenCVCamera/FramePump ชุดเดียวกับ Web แล้ว เก็บเฉพาะภาพล่าสุดพร้อม sequence/host read time และ capture metadata ใน Evidence; Legacy branch คงลูป/ค่ากล้องเดิม Host read time ไม่ใช่ hardware exposure timestamp จึงยังต้องทดสอบ buffer/latency จริง
Terminal, capture utility เดิม และ Web workflow ใช้ project lease ที่มี OS lock ร่วมกัน ป้องกัน process อื่นใน project แย่งงานกล้อง/เปลี่ยน configuration ระหว่างตรวจ; Web แสดง external owner แต่ไม่แชร์ camera handle หรือผลสดของ Terminal
ตั้งแต่ 15 ตุลาคม 2026 ถอดโหมด Candidate ของ Terminal แล้ว ใช้คำสั่งเปิดกล้องเดิมข้างต้นเพียงทางเดียว; ทดสอบ Candidate ด้วยภาพผ่าน Test/Validation ใน `/inspection` ข้อมูล Recipe และ Evidence เดิมยังอยู่
Candidate แบบวาง–ยกต้องมี Locator; fixed/cropped recipe เดิมยังใช้ Test ภาพได้ แต่ไม่รับรอง live cycle โปรแกรม Core ใช้ resolution ที่ Recipe กำหนดและต้องตรง full-frame reference; legacy mode รักษาค่ากล้องเดิม
ก่อน Production validation ต้องกำหนด Min defect recall ระดับชิ้นงานจาก requirement จริง ค่าเริ่มต้นเว้นว่าง; REVIEW/UNABLE ไม่นับเป็น NG rejection และ object recall ของ detector เป็นคนละเกณฑ์
Engine 2026.10.09.2 ต้อง Validate ใหม่ก่อน Production รายงาน/Recipe/threshold เดิมไม่ได้แก้ย้อนหลัง
Regression Phase 3 หลังรวม live capture ผ่าน 327 tests และ 5 subtests รวม mock camera, cross-process ownership Dataset lifecycle และ Terminal/Web live parity; JavaScript 9 tests ผ่าน ไม่ใช่หลักฐานตรวจตำหนิหรือกล้องจริง Browser automation ใช้ไม่ได้รอบนี้; HTTP smoke ผ่าน
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
Web camera runtime เก็บเฟรมล่าสุดช่องเดียว ไม่สะสมคิวในแอป ตรวจอายุภาพและบันทึกเวลา source; Terminal โหมด Core ใช้ FramePump เดียวกัน ส่วน Legacy รักษาลูปเดิมตามข้อกำหนดด้านบน
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

## หลักฐานปัจจุบันและข้อจำกัด
- Hardware วันที่ 14 ต.ค. รับภาพ LifeCam 1920×1080 จริงเข้า Core ได้ แต่ Recipe 256×256 เดิมหยุด UNABLE ก่อน Anomaly
- Demo Full-frame OpenCV ใช้คราบจำลองบนชิ้นเดียวกัน ไม่ใช่โมเดล AI ใหม่ และ ROI ไม่ใช่ขอบเขตตำหนิจริง
- Repeatability ยังไม่ผ่าน: GOOD ที่วางใหม่มี alignment failure; หยุดรอบทดสอบเมื่อผู้ใช้สั่งถอด Candidate จาก Terminal
- ตัวเลข PaDiM/PatchCore บน crop เดิมเป็นผลทดลองย้อนหลัง ชุด Airbag_Toyota 268 ภาพเป็นชิ้นเดียวขยับหลายตำแหน่งตามผู้ใช้ยืนยัน ไม่ใช่ independent part validation
- ไม่มี Production ที่ผ่านการรับรอง ห้ามอ้างว่า unit/integration tests หรือ GOOD-only พิสูจน์ Defect recall
- ผลและภาพจริงเดิมยังอยู่ รายละเอียดล่าสุดใช้ work/RENOVATION_CONSOLIDATION_REPORT.md

## แผนที่ทั้งโปรเจกต์และความสัมพันธ์
| ตำแหน่ง | หน้าที่และสิ่งที่เชื่อม |
|---|---|
| 01_camera | Terminal/OpenCV ตัวเดิม; ใช้ Shared Core เมื่อมี validated Production หรือ Normal Reference เดิมตาม config; capture_training_images เป็นเครื่องมือเก็บภาพเดิม |
| 02_training | CLI เตรียมข้อมูล/ฝึก detector เดิม เรียกโดย Backend และ dataset utility; ไม่ใช่ workflow บังคับก่อนตรวจ |
| 03_dataset | ผลรอบกล้อง, human review, capture/staging/trash/CVAT และ utility เดิม; /inspection อ่านผ่าน API เดิม |
| 04_verify | CLI ประเมิน detector; ไม่ใช่ตัวรับรอง Defect accuracy ของ Core |
| 05_train_new_object | Dataset Library ต้นทาง, backgrounds และผลฝึกเดิมที่ Training/Jobs/Models อ้างอิง; ไม่ใช่ source pipeline ใหม่ |
| 06_brand_classification | เครื่องมือ/ข้อมูล/model ของ Brand เสริม; ยังมี config เดิมอ้างอิง ไม่ใช้ Brand ตัดสิน Defect |
| 07_web_app | FastAPI และ UI /inspection; app.py ให้บริการ Dataset/Jobs/Model/History เดิม, inspection/ เป็น Core; tests ตรวจทั้งสองร่วมกัน |
| inspection_data | Store ของภาพ/Recipe/Model metadata/Validation/Evidence/Core runtime; โมเดลและ reference ผูก hash |
| models และ jobs | น้ำหนัก/model artifact และ Job เดิม; Terminal/config และหน้า Jobs/Models ใช้ร่วมกัน ไม่คัดลอกเข้าระบบใหม่ |
| tools/maintenance | ติดตั้ง worker และซ่อม path เมื่อย้ายเครื่องแบบสั่งเอง ไม่รัน repair ตอน startup |
| work | รายงาน ผลทดลอง ภาพ/วิดีโอ/สไลด์ และหลักฐานย้อนหลัง; ไม่เป็น source/runtime ที่ต้องเปิดอีกชุด |
| .venv-inspection | Environment ของ Anomalib worker/tests; setup เชื่อม torch/torchvision จาก ../venv ไม่ใช่ implementation ซ้ำ |
| .env | credentials/การตั้งค่าเฉพาะเครื่อง; เก็บเป็นความลับ ไม่ใช่ source และไม่ย้ายค่าลงรายงาน |
| __pycache__ / .pytest_cache | cache ที่สร้างใหม่ได้ ไม่ใช่ implementation หรือข้อมูลอ้างอิง |
| .git / .gitignore | ประวัติ Source/ขอบเขต tracking; ไม่ครอบคลุม backup Dataset/Model/Evidence ที่ ignore |
| .vscode / pyrightconfig.json / __init__.py | การตั้งค่า editor/package marker; ไม่ใช่ camera entrypoint |
| requirements*, setup_new_machine.bat, start_web_app.bat | dependency lock, setup และ launcher เดิม; ไม่ติดตั้ง/สลับรุ่นเองขณะตรวจ |
| README_TH.txt, AGENTS.md, BACKEND_HANDOVER_TH.md, MOVE_TO_NEW_MACHINE_TH.txt | จุดเริ่มใช้งาน กติกางาน คู่มือ backend และย้ายเครื่อง; อ่านสถานะล่าสุดจากรายงานหลัก |

ผลจาก Terminal เดิมเข้าถึงที่ Review → Terminal camera / Legacy history ใน /inspection
ผ่าน /api/inspection/parts และภาพต้นฉบับใน 03_dataset/inspection_cycles; Human Review ใช้ /api/inspection/reviews
Core Evidence อ่าน inspection_data/evidence แยกชนิดผลให้ชัด ไม่แปลง machine NG เป็น human-confirmed NG
การเข้าถึงจากหน้าเดียวไม่ได้หมายความว่า Legacy และ Core ตัดสินเหมือนกันแล้ว: Legacy ยังต้องอยู่จนตัวแทนผ่าน field validation
ถอด legacy web camera worker และ /start_stream, /stop_stream, /latest_frame.jpg, /status แล้ว; เว็บใช้ /api/vision runtime เท่านั้น
ไม่ย้อนนำ /engineering กลับมา Dataset/Jobs/Models/History ใช้ backend เดิมผ่าน /inspection
สำเนา source และสคริปต์ทดลอง 81 ไฟล์ใน work ถูกสำรองพร้อม SHA256 ไปนอกโปรเจกต์ก่อนถอด ดู archive ในรายงานหลัก
