# Airbag_Project — Renovation & Consolidation

อัปเดต 9 ตุลาคม 2026 · Project: `D:\Users\dev06\Desktop\Airbag_Project`

## สถานะรอบนี้

ลงมือแก้ Source Code ข้าม Camera, Dataset utility, Inspection Core, Backend, UI, legacy reference และ Tests แล้ว เป็นช่วงแรกของ consolidation **ยังไม่อ้างว่าทั้ง Project เหลือ runtime เดียวสมบูรณ์** เพราะยังไม่มี full-frame Candidate/NG validation ที่ทดแทนงาน legacy จริงได้ และงาน dataset/job บางส่วนยังต้องย้ายก่อนถอด `/engineering`

ใช้โค้ดปัจจุบันตรวจผู้เรียกและผลกระทบก่อนแก้ ไม่ทำ Audit ทั้งระบบซ้ำ ไม่เปลี่ยน actual Production configuration ไม่เพิ่ม AI model/framework/dependency

## KEEP

- `07_web_app/inspection` เป็น Core เดิม: quality → locator/alignment → areas → method → decision → cycle/evidence; ไม่มี Core หรือ inference pipeline ใหม่
- Entry point `01_camera/airbag_webcam_detection.py`, หน้าต่าง OpenCV, FPS, q/Esc, s/Space และ legacy camera controls ที่ยังใช้อยู่
- Dataset preparation, locator/brand utilities และ legacy artifacts ที่ยังอ้างโดย config/jobs; ข้อมูล training/calibration/validation/evidence เดิมยังอยู่
- PaDiM/PatchCore/OpenCV, optional detection/segmentation/ONNX และ CVAT connector เดิม ไม่เพิ่มโมเดลหรือขยาย workflow เหล่านี้

## MIGRATED / MERGED / REFACTORED

### 1. ปิดช่องว่างการอนุมัติ Validation

- เพิ่ม `validation.min_defect_recall` สำหรับคำตัดสินระดับชิ้นงาน แยกจาก object recall ของ detection/segmentation
- ค่าเริ่มต้น **None/ช่องว่าง**; ต้องให้เจ้าของงานกำหนดเกณฑ์ก่อนอนุมัติ Production ไม่ยืม default90% ของ detector มาเดาความต้องการหน้างาน
- Strict recall นับเฉพาะ NG ที่ตัดสิน NG; REVIEW/UNABLE ไม่นับเป็นการ reject สำเร็จ จึงไม่ให้ GOOD จำนวนมากกลบกรณี NG ทั้งหมดเข้า REVIEW แล้วผ่าน aggregate review rate
- UI โหลด Recipe เก่าโดยเว้นเกณฑ์นี้ว่าง และ New product ล้างเกณฑ์ของงานก่อน ไม่เติมเป้าหมายย้อนหลัง
- เพิ่ม Engine version เป็น `2026.10.09.1`; reports เก่าต้อง Validate ใหม่ การอ่าน history ยังทำได้ ไม่ rewrite recipes/models/reports เดิม

ไฟล์หลัก: `inspection/schema.py`, `validation.py`, `__init__.py`, `static/inspection.js`, `templates/inspection.html` ใต้ `07_web_app`

### 2. Candidate ใช้กล้อง Terminal เดิมและ Core เดียวกันได้โดยไม่ Activate

- เพิ่ม **option ใน entry point เดิม** `--candidate-recipe`; ไม่สร้าง executable, หน้าเว็บกล้อง หรือ camera CLI entry ใหม่
- ใช้ `Runtime.candidate_pipeline()` และ readiness/pipeline/cycle/evidence ชุดเดียวกับ Production พร้อมแสดง `CANDIDATE TEST - NOT PRODUCTION` บนภาพสดและ Terminal
- Evidence แยก `source=candidate` / `candidate_sample`; ไม่สร้าง Ground Truth และไม่แก้ `production.json`
- เมื่อผู้ใช้เลือก Core recipe อย่างชัดเจน ใช้ขนาดภาพที่ระบุใน recipe และตรวจว่าตรงกับ full-frame reference ไม่ resize ภาพเพื่อให้ผ่าน; camera index ต้องตรง ไม่รับ exposure/gain override ของ USB อย่างผิดหน่วย
- หากไม่ระบุ Candidate และไม่มี Production สูตรใหม่ ยังคงเส้นทางเดิมที่ใช้งานอยู่จนมี replacement ผ่านการทดสอบจริง
- Candidate continuous place/remove ต้องใช้ Locator; fixed/cropped recipes เดิมไม่ถูกเปลี่ยนให้ผ่านเอง

รูปแบบคำสั่งเมื่อมี Candidate และ camera setup จริงแล้ว (ตัวแทนข้อความ ไม่ใช่ ID ที่สร้างขึ้น):

```text
..\venv\Scripts\python.exe 01_camera\airbag_webcam_detection.py --candidate-recipe <saved_recipe_id> --camera-index <actual_index>
```

รอบนี้ไม่ได้รันคำสั่งเปิดกล้องนี้ และไม่ได้สร้าง Candidate จากภาพสมมติใน project

### 3. รวมการถือสิทธิ์ Runtime ข้าม process

- ใช้ `filelock` ที่ติดตั้งอยู่แล้ว สร้าง project lease ใน Core เพียงจุดเดียว (`inspection/ownership.py`) แก้ปัญหาจริงที่ Web กับ Terminal ใช้ in-process locks คนละชุด
- Terminal ทั้ง legacy/Core, capture utility เดิม, Core test/capture/activate/train/Production และ legacy Web camera/reference-training/YOLO-training lifecycle ใช้ lease ร่วมกัน
- ถือสิทธิ์จน worker/child training และ finalization จบ ไม่ปล่อยทันทีเมื่อ HTTP request จบ; thread-start failure คืนสิทธิ์; OS คืน lock เมื่อ process ตาย จึงไม่ใช้ metadata/PID เก่าเป็นหลักฐานว่ากล้องยังถูกครอง
- `/api/vision/status` แสดง `external_runtime`; UI บอก owner และปิดงานที่ขัดกัน การอ่าน history/status ยังใช้ได้
- ไม่ mark training record ของ process อื่นเป็น failed เพียงเพราะอีก Web instance เริ่มทำงานขณะมี owner
- `capture_training_images.py` ใช้ lease เดียวกัน คง capture/settings/UI เดิม และ release เมื่อเปิดกล้องไม่สำเร็จ

ขอบเขต: ป้องกัน entry points ที่ปรับแล้วใน **project นี้** ไม่ใช่ global device lock สำหรับโปรแกรมกล้องจากผู้ผลิต/โปรแกรมภายนอก/สำเนา project อื่น หรือ process เก่าที่เปิดค้างด้วยโค้ดก่อนแก้ ไม่ได้ hot-patch หรือ Restart กล้อง/server ของผู้ใช้

### 4. Legacy failure ต้องไม่กลับเป็น OK

- Classical `NormalReferenceInspector` ไม่คืน identity image เงียบ ๆ หลัง ECC ล้มเหลวอีกต่อไป ตรวจ transform finite/ไม่ singular/ไม่ reflection
- การจัดแนวล้มเหลว, ไม่มีพื้นที่ตรวจที่มองเห็น หรือ model output/threshold ไม่ finite ให้ REVIEW พร้อม `input_issue` และไม่มี defect region ปลอม
- Training reference ทาง classical ก็ปฏิเสธ alignment ที่ล้มเหลว ไม่เรียนรู้ต่อจาก identity fallback
- DeepPatch กับ Classical ใช้ `AlignmentError` ตัวเดียวกัน ถอด class ซ้ำ โดยไม่เปลี่ยน model weights/threshold ที่เก็บจริง

## REMOVED จริง

จาก `03_dataset/airbag_negative_mining_pipeline.py`:

- ถอด model fallback loader ที่หมด caller (`load_inference_model`) และ model-selection constants ที่เกี่ยวข้อง
- ถอด duplicate inference helpers `calculate_iou`, `overlaps_any_airbag`, `draw_labeled_box` และ constants ของกล้องชุดที่เลิกใช้
- ถอด `ultralytics.YOLO` import ของ utility นี้ และ `PROJECT_DIR` alias ที่หมดหน้าที่; **ไม่ uninstall ultralytics** เพราะกล้อง/locator/brand/training ส่วนอื่นยังใช้
- คง dataset preparation logic ที่ `02_training/train_airbag_model.py` import จริง
- แก้ `--camera` ให้ส่งเฉพาะ camera arguments ไปยัง entry point เดิม ป้องกัน argument parser ของกล้องได้รับ `--camera/--prepare-dataset` จาก utility แล้ว error; camera index ที่ระบุส่งต่อจริง

ตรวจชื่อผู้เรียกจาก source/import/CLI/script/config ที่เกี่ยวข้องและทดสอบ delegation แล้วก่อนถอด ไม่มีการ comment code ไว้หรือสร้าง `_old`/`v2` ไว้แทนการลบ

ไม่อ้างว่าถอดหน้า/API ที่ยังมีผู้ใช้งานแล้ว: `/engineering`, legacy live reference และ dataset/job APIs ยังอยู่ตาม BLOCKED/NEXT ด้านล่าง

## TESTED

| การตรวจ | ผล / ขอบเขต |
|---|---|
| Qualification regression ช่วงแรก | 58 passed; explicit policy, imbalance/REVIEW loophole, provenance/activation และกล้องเดิมที่จำลอง |
| Classical alignment/failure ช่วงถัดมา | 37 passed; รวม save/load และ failure cases ไม่มีการเทียบ accuracy หน้างาน |
| Full regression หลัง integration/cleanup | **327 passed, 5 subtests passed** ด้วย `.venv-inspection\Scripts\python.exe -B -m pytest 07_web_app/tests -q -p no:cacheprovider` |
| Cross-process integration | เปิด Python process ชั่วคราวใน temp directory จริง ทดสอบ lock contention, Web mutation rejected/status readable, Terminal/capture utility หยุดก่อนเปิดกล้อง, process crash คืน lock, worker-start failure คืนสิทธิ์ |
| Candidate integration | ไม่มี Production pointer/report ก็ใช้ explicit Candidate test ได้; source แยกชัด; Production pointer ที่มีอยู่ใน fixture ไม่เปลี่ยน; camera เป็น Mock |
| JavaScript | `node --check` ผ่าน; `node --test` 2 suites ผ่าน (live polling + defect policy round trip/old recipe/New product/external owner UI) |
| Cross-module syntax | AST parse source ของ Camera, Training, Dataset, Verification, Brand, Web/Core และ tools ผ่าน |
| HTTP smoke | temporary server8011: `/inspection`, status, recipes, reports, evidence ตอบ200; Productionไม่มี/runningfalse; fieldใหม่อยู่ในHTML |
| Browser จริง | **ยังไม่ยืนยันรอบนี้**: CUA ล้มเหลวก่อนเปิดหน้า (`failed to write kernel assets ... os error 3`) ไม่ใช้ HTTP/VM tests อ้างว่าเป็น browser test |
| Data preservation | SHA256 ของ existing JSON records **412 ไฟล์** เท่า checkpoint ทุกไฟล์; `project_config.json` ทุก byte ไม่เปลี่ยน; ไม่มี production pointer ใหม่ |

Tests ใช้ภาพสังเคราะห์/temporary fixtures เพื่อตรวจ software contracts ไม่ใช่หลักฐาน Defect Accuracy หรือกล้องจริง ไม่ Train ด้วย Dataset จริง ไม่ Activate Production ไม่เรียก CVAT และไม่ติดตั้ง dependencies

temporary server8011 ปิดแล้ว; serverเดิมที่8000ไม่ได้ Restart โค้ดใหม่ใช้เมื่อเริ่มโปรแกรมครั้งถัดไปตามวิธีเดิม

## จุดย้อนกลับ

- เดิมไม่มี `.git`; สร้าง local Git repository พร้อม **baseline commit `f2decd0`** จาก source ก่อนแก้ ไม่ทำ remote/push
- บันทึก implementation ที่ผ่าน tests เป็น commit ถัดจาก baseline; ดู `git log --oneline` เพื่อเลือก revert เฉพาะ renovation commit ไม่ควร restore ทั้ง baseline ทับ config ที่อาจเปลี่ยนภายหลัง
- `.gitignore` แยก environments/runtime data/models/jobs/work/history/logs ออกจาก source checkpoints; ไม่ลบไฟล์จริง
- สำเนาที่สร้างก่อนรับคำสั่ง Git ย้ายออกจาก Project แล้ว อยู่ที่ `D:\Users\dev06\Documents\Codex\2026-10-07\ai-defect-inspection-airbag-project-desktop-2\work\consolidation-before-20261009-092137` ไม่มีสำเนา source ชุดใหม่ค้างใน Project; ไม่แตะ historical backups ของผู้ใช้

## BLOCKED

- **ยังถอด legacy inspection ทั้งสายไม่ได้:** ไม่มี full-frame Candidate ของชิ้นงานจริงที่พิสูจน์ locator/ROI/anomaly/decision/cycle เทียบกับงานเดิมแล้ว จึงคงความสามารถที่ผู้ใช้ใช้งานระหว่าง migration โดยระบุว่าเป็น legacy
- Core data ยังไม่มี human-confirmed NG หรือ confirmed capture groups สำหรับรับรอง defect recall; current cropped/fixed recipes ไม่ใช่ full-camera replacement และไม่มี Production activation
- ยังไม่พิสูจน์ LifeCam/Basler จริงในรอบนี้ ไม่เปลี่ยน focus/exposure/gain และไม่สร้าง threshold/model/recipe เพื่อให้ validation ผ่าน
- `/engineering` ยังมี dataset upload/preview/undo/trash/metadata และ legacy job workflows ที่ต้องย้าย/reuse ให้ครบก่อนถอดหน้า/API; ยังไม่อ้างว่า migration ส่วนนี้เสร็จ

## NEXT — งานจำเป็นที่เหลือ

1. ใช้ภาพเต็ม GOOD/NG ที่คนยืนยันพร้อม part/capture groups และ actual camera setup เพื่อทำ Candidate ผ่านโปรแกรมเดิม ทดสอบ quality/pose/ROI/NG/การวาง–ยก/เวลาต่อชิ้น และกำหนด acceptance requirement จริง
2. ย้ายเฉพาะ dataset/job/history ที่จำเป็นเข้าการเข้าถึงจาก `/inspection` โดย reuse services เดิม รักษาความหมาย machine result เทียบ human confirmation; ทดสอบ old/new records ก่อนถอด `/engineering` และ APIs ที่หมดหน้าที่
3. เมื่อ replacement ผ่าน field validation แล้ว ถอด legacy reference/decision/cycle/live branches ที่หมด caller ในรอบเดียวกัน พร้อมตรวจ locator/brand/model artifact references ไม่ลบข้อมูลตามโค้ด
4. กลับมาทำ browser smoke ด้วยเครื่องมือที่ใช้งานได้ และทดสอบ hardware ownership/disconnect/latency จริงก่อนรับรองหน้างาน

ไม่ใช้ผล software tests รอบนี้อ้างว่า Production-ready และไม่ถือว่าการคง legacy ระหว่างรอ replacement เป็นเป้าหมายถาวร
