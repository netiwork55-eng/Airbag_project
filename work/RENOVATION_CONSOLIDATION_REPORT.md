# Airbag_Project — Renovation & Consolidation

อัปเดต 9 ตุลาคม 2026 · Phase 4 · Project: `D:\Users\dev06\Desktop\Airbag_Project`

## Phase 4 — System Verification & Legacy Retirement Readiness

ฐาน `37ab956`; checkpoint ก่อนแก้ `phase4-start-20261009` และหลังแก้ `phase4-verified-20261009` ตรวจ Source/Runtime ปัจจุบันเฉพาะ integration ที่เกี่ยวข้อง ไม่เปลี่ยน Camera entrypoint, Recipe, model weights, thresholds หรือ Production configuration ไม่ Train/Activate และไม่เปิดกล้องจริง

### ผลตรวจและสิ่งที่แก้จริง

- `/inspection` ยังเป็น workflow หลัก Dataset/Jobs/Models/Legacy History ใช้ backend เดิม; `/engineering` ตอบ404 และ `/jobs` redirect `/inspection#test` ไม่มีการนำหน้าเก่ากลับมา
- Terminal Core กับ Web ใช้ `Runtime.open_capture` → `FramePump` → `Pipeline` → `Cycle` → `evidence` ร่วมกันแล้วตั้งแต่ Phase3 ยืนยันด้วย integration tests ทั้ง Candidate และ Production, ภาพเต็ม, ROI/quality, stale/disconnect, metadata และ incomplete cycle ส่วน legacy fallback ยังคงอยู่เมื่อไม่มี Core Production ที่ activate
- แก้ Production UI เมื่อ status request ล้มเหลว: ล้างสีเขียว/ภาพสดเก่า แสดงไม่ทราบสถานะ ปิด Start/เปลี่ยนสูตร/Train/Validate/Activate/Rollback ระหว่างติดต่อไม่ได้; คงปุ่ม Stop ที่เคยใช้ได้ไว้ เพราะ network failure ไม่ได้แปลว่า worker หยุดแล้ว
- ป้าย Production จาก engine เก่าแสดงต้อง Validate ด้วยสี review; Activate ติดตามสถานะ busy, report eligibility/engine และ dirty recipe พร้อมคืนสถานะอย่างถูกต้องเมื่อเชื่อมต่อได้ การเปิดใช้จริงยังผ่าน backend validation gate เดิม
- แก้ Core lifecycle: หลัง driver แจ้ง `Camera release failed` การเรียก Stop ซ้ำไม่กลบ error เป็น “Production is stopped” และการเปิด capture ใหม่ใน Runtime เดิมถูกปฏิเสธก่อนแตะ driver ต้องให้ผู้ใช้จัดการ process/device ก่อนเริ่มใหม่ ไม่มี automatic restart/recovery และไม่อ้างว่าการคืน hardware สำเร็จจาก thread ที่จบแล้ว
- พิสูจน์บั๊กด้วย tests ที่ fail ก่อนแก้: UI 3กรณี และ camera lifecycle 2กรณี หลังแก้ผ่านทั้งหมด Engine ยังเป็น `2026.10.09.2` เพราะไม่ได้เปลี่ยน image processing, scoring หรือ decision contract

### Verification ที่ทำแล้ว / ขอบเขต

| การตรวจ | ผลจริง |
|---|---|
| Python regression/integration | **329 tests + 5 subtests ผ่าน**; ครอบคลุม Camera/legacy entrypoint/Core/ownership/data lifecycle/validation/training-job contracts/model compatibility/evidence |
| JavaScript VM/DOM fixtures | **12 tests ผ่าน** รวม status offline→recovery, stale engine, activation busy, Dataset/Jobs/Legacy/Human review; ไม่ใช่ Browser tests |
| Latest-code HTTP smoke บน temporary8011 | หน้า/asset และ12 API groups: status, recipes, images, reports, evidence, dataset folders, trash, jobs, model registry, training runs, reviews, legacy parts อ่านได้200 |
| ข้อมูลเก่า | Core test evidence 3รายการ (PASS1/NG2) และ preview ทั้ง3เปิดได้; legacy parts API อ่านได้30รายการในหน้าที่ตอบกลับ, unreadable0 และภาพตัวอย่างเปิดได้; ไม่มีการเปลี่ยน machine result เป็น human label |
| Mutation integration | Upload/Preview/Metadata/Undo/Trash/Restore และ review/recipe/validation ใช้ temporary fixtures; ไม่ทำ Undo/Delete กับ Dataset จริง |
| Ownership/capture | real threads/OS file lock กับกล้องจำลอง, subprocess contention, timeout/start/release failure ผ่าน; ไม่ใช่การทดสอบ LifeCam/Basler จริง |
| Browser | **ยังค้าง**: ทั้ง `cua_repl` และ skill computer-use ผ่าน `node_repl` เริ่มไม่ได้ด้วย `failed to write kernel assets ... os error 3`; ไม่รับรอง layout/button/form/responsive จาก HTTP200 |

temporary server8011 ปิดหลังอ่านข้อมูล ไม่ Restart server8000 หรือโปรแกรมกล้องของผู้ใช้ คำสั่ง regression ใช้ชุดเดียวกับส่วน Phase3 ด้านล่าง ไม่มี dependency เพิ่ม

ตรวจ SHA-256 ก่อน/หลังของ data/config/artifact ที่เลือก **8,261ไฟล์: เปลี่ยน0 / หาย0** (รวม72 model artifacts); manifest เก็บนอก Project ใน workspace ของงาน ไม่รวม `.git`, environments, caches, `work`, `.vscode` และ runtime lock metadata ไม่อ้างว่าเป็น hash ทุกไฟล์ใน Project ตรวจ Git diff แล้วมีเฉพาะ runtime, UI, regression tests และรายงานนี้; `project_config.json`/dependency configuration ไม่เปลี่ยน

### Candidate readiness จากข้อมูลปัจจุบัน

- Core Recipe7รายการยังเป็น `fixed`; ไม่มี Production pointer สูตรเหล่านี้ใช้ image tests ได้ตามเงื่อนไขเดิม แต่ **ไม่ใช่ Candidate Live ที่พร้อม place/remove**
- Core image metadata392รายการ: label GOOD391, UNKNOWN1; `confirmed_at` สำหรับ GOOD/NG =0, confirmed capture group =0 จึงยังใช้รับรอง validation ไม่ได้ การติดป้าย GOOD เดิมไม่ใช่ human confirmation
- คลังนี้ไม่มี human-confirmed NG; NG ใน Evidence เป็น machine test เท่านั้น ไม่ใช่ ground truth สถิตินี้เฉพาะ Core store ไม่ได้อ้างว่าภาพทั้งหมดใน project ถูกตรวจโดยมนุษย์แล้ว
- ต้องมี full-frame reference ที่ยืนยันจริง, capture settings/group provenance, ROI/model input ที่ตรงกัน, independent GOOD/NG และข้อกำหนด defect ก่อนสร้าง/ประเมิน Candidate หน้างาน ห้ามรับรองโมเดลที่เคยเรียนจาก cropped ROI ว่าเข้ากับสูตรใหม่โดยอัตโนมัติ

### Legacy retirement decision จาก caller และ contract ปัจจุบัน

| ส่วน | สถานะ | หลักฐาน/เงื่อนไขก่อนถอด |
|---|---|---|
| YOLO Part Locator (`part_locator.analyze_part`) | **REUSE IN CORE — BLOCKED** | Web/Terminal/legacy reference ยังเรียกอยู่ คืน bounding box ไม่มี reference transform; พิจารณา reuse เฉพาะเมื่อ Template/Features ไม่ครอบคลุมงานจริง ต้องพิสูจน์ full-frame translation/rotation/scale, partial part, alignment และ source ROI ก่อน ไม่มี adapter ใหม่ในรอบนี้ |
| Optional Brand Classifier | **KEEP** | ใช้เลือก/แยกชนิดงานแบบ optional ใน legacy; ไม่ใช่ defect decision และยังไม่มี requirement ให้ Core เลือกสูตรด้วย Brand คง shared classifier ที่มี caller ไม่เพิ่ม Brand gate |
| Deep Patch Reference / RecipeBoundInspector | **REPLACE — BLOCKED** | ยังใช้ใน live/job tests และ artifact contract เดิม; PaDiM/PatchCore ใน Core เป็นตัวแทนเชิงหน้าที่ แต่ต้องผ่าน full-frame ROI/model compatibility + independent NG validation ก่อนย้ายงานจริง ห้ามลบ weights/history |
| Legacy DecisionStabilizer / PartCycle | **REPLACE — BLOCKED** | Core Cycle มีตัวแทนแล้วใน software tests แต่ OK/REVIEW/NG และ incomplete policy ของ legacy ต่างกัน ต้องเทียบลำดับภาพเดียวกันและหนึ่งผลต่อชิ้นจริงก่อนถอด callers |
| Legacy Camera Acquisition | **REPLACE — BLOCKED เฉพาะ legacy branch** | Core branch ใช้ capture ร่วมและไม่มี synchronous loop ซ้ำแล้ว; legacy index/fallback/driver settings ยังเป็น contract เดิม ต้องพิสูจน์ exact-device/full resolution/freshness/release บนกล้องจริงก่อนย้าย branch นี้ |
| Legacy UI/API/helpers ที่ถอด Phase2–3 | **READY TO REMOVE → REMOVED แล้ว** | `/engineering`, UI assets/helpers ที่หมด caller และ Core capture loop ซ้ำถูกถอดแล้ว; regression ปัจจุบันยังผ่าน |

**Phase4 ไม่ลบ Legacy Live เพิ่ม**: ส่วนที่ตรวจยังมี caller และ field-validation blocker จริง ไม่ได้เก็บเพียงเพื่อความสะดวก ไม่มี dependency ใดถูกถอดโดยยังมี caller และไม่มี implementation ใหม่ซ้อน การแก้รอบนี้อยู่ที่ UI status และ Runtime lifecycle เดิม

### ขั้นตอน Candidate ครั้งแรก (ผู้ใช้ทำเมื่อข้อมูลพร้อม)

1. ระบุชนิด/ตำแหน่ง/ขนาดต่ำสุดของ defect, tolerance การเลื่อน/หมุน/scale, เวลาต่อชิ้น และเกณฑ์ FA/FR/strict recall/REVIEW/UNABLE ที่เจ้าของงานยอมรับก่อน ไม่ใช้ค่า default เป็นการอนุมัติหน้างาน
2. นัดช่วงหยุดงาน ให้ผู้ใช้ปิด Terminal เดิมด้วย q/Esc และตรวจคืนกล้องก่อนใช้ Capture เดิมใน Setup หรือ utility เดิม ห้ามเปิดซ้อน จด LifeCam indexจริง, full resolution, exposure/gain/focus/autofocus/white balance ที่ driver รองรับ, ระยะกล้องและการจัดแสง ค่าจาก OpenCV อาจเป็น sentinel ต้องเทียบภาพจริง; Basler ยังไม่ได้ทดสอบ
3. เก็บภาพเต็มต้นฉบับ GOOD และ NG ที่ผู้ตรวจยืนยัน ระบุ part ID, defect type/ขนาด/ตำแหน่ง, capture group และผู้ตรวจในบันทึกหน้างานที่โยง filename/image ID; UI metadata/review ไม่ได้เป็นระบบจัด defect taxonomy อัตโนมัติ ชิ้นเดียวกันหรือรอบเดียวกันรวมกลุ่มเดียวกัน รวมภาพเลื่อน/หมุน/ถ่ายซ้ำของชิ้นนั้น ห้ามกระจายไป train/calibration/independent validation คนละชุด
4. ใน `/inspection` Setup เลือกภาพเต็มอ้างอิงและยืนยัน GOOD/group เฉพาะที่ตรวจแล้ว ตั้ง Candidate แยกด้วย Template/Features ที่เหมาะกับภาพจริง ลาก inspection areas ใน reference coordinates ตรวจ source polygon หลัง alignment และ model/reference/ROI/input hash ให้ตรงกัน ใช้ anomaly ที่มีอยู่; ถ้า model input ไม่ตรงให้หยุดเพื่อขอทำงานปรับ/Train ด้วยข้อมูลจริงในรอบที่อนุญาต ไม่ยืม cropped Recipe เก่ามาบังคับให้ผ่าน
5. เก็บภาพ calibration ทั้งยอมรับ/ไม่ยอมรับสำหรับแสงและความคมแล้วจึงกำหนด quality/locator/anomaly limits ตามข้อมูลจริง แยกกลุ่มจาก validation ถ้ายังไม่ได้ calibrate ให้ระบุว่า exploratory และยังไม่รับรอง fault coverage; `min_blur=0` คือปิด blur gate
6. เมื่อ Candidate พร้อมและไม่มี process ถือกล้อง ให้รันไฟล์เดิมด้วย Python environment เดิมจาก Project root (แทน placeholders ด้วยค่าจริง ไม่ copy bracket ไปใช้):

```text
..\venv\Scripts\python.exe 01_camera\airbag_webcam_detection.py --candidate-recipe <saved_recipe_id> --camera-index <actual_index>
```

7. ต้องเห็น `CANDIDATE TEST - NOT PRODUCTION`; ตรวจขนาดภาพเต็ม/กรอบ/FPS/quality/locator/ROI แล้ววาง–ยก GOOD/NG ทีละชิ้นตาม matrix ด้านล่าง `s`/Space เก็บ sample ไม่ใช่ human confirmation และไม่แทน cycle completion; q/Esc หยุด ไม่กด Activate
8. อ่าน Review พร้อม source=`candidate`/`candidate_sample`, image ID, Recipe hash, area/source polygon, score/threshold, capture metadata, frame age/sequence, complete flag และเหตุผลของ decision ผูกกับ field log/part ID/group แยก human truth จาก machine result ตรวจ Production pointer ไม่เปลี่ยน
9. หาก ROI หลุด, ภาพผิดขนาด, locator/quality fail, ข้อมูลไม่ครบ หรือ release error ให้หยุดและเก็บ failure evidence ห้ามแก้ threshold เพื่อทำให้ตัวอย่างนั้นผ่าน หากเก็บ evidence ไม่ได้ ให้เก็บ error log และไม่ถือว่ารอบนั้นผ่าน ผู้ใช้เป็นผู้ตัดสินใจ restart/reconnect หลังตรวจ process/device

### Field test matrix และผลที่ต้องเก็บ

| กรณี | สิ่งที่วัด/ตรวจ |
|---|---|
| Confirmed GOOD/NG ทุก defect type และขนาดใกล้ขอบเขต | ผลต่อชิ้น/ต่อชนิด defect, heatmap/score/threshold/area coverage, false accept/reject; จำนวนตัวอย่างและเกณฑ์ให้เจ้าของงานกำหนด ไม่กำหนดในรอบนี้ |
| ชิ้นงานกึ่งกลาง/เลื่อน/หมุน/scale ภายในขอบเขตและเลยขอบเขต | transform, locator score, source polygon ครอบคลุมตำหนิจริง; invalid alignment/required ROI ต้องไม่ PASS |
| เบลอ แสงมืด/สว่าง เงาสะท้อน บัง/ชิ้นงานขาดขอบ | Quality/Locator failure พร้อม reason; ตรวจ blur gate เปิดหลัง calibration แล้ว ถ้าภาพไม่เหมาะยัง PASS ต้องบันทึกเป็น failure ของ setup |
| วางค้าง/สั่น/ยก/วางเร็ว/ชิ้นถัดไป และทดสอบชิ้นเดิมซ้ำ | ผลหนึ่งรายการต่อ complete cycle, state holding/settling, ไม่เอาผลหรือ heatmap ชิ้นก่อนมาตัดสินชิ้นใหม่; แยก repeatability จากจำนวน independent parts |
| ภาพค้าง/ประมวลผลช้า/กล้องหลุด/หยุดกลางชิ้น | UNABLE หรือ incomplete ตามเหตุ; ไม่สร้าง complete PASS ที่ยังตรวจไม่จบ, stale image ไม่กลับมาใช้, error และ ownership/release บันทึกได้ การถอดสายทำโดยผู้ใช้ในช่วงทดสอบที่ปลอดภัย |
| Evidence/Review และ Validation | อ่านภาพ/Recipe/model/area/threshold ย้อนกลับได้, human confirmation ไม่เขียนทับ machine result, split groups ไม่รั่ว, record เก่าไม่มี field ใหม่ต้องไม่ถูกเติม ground truth เอง |

นิยามตาม `inspection/validation.py`: **False Accept = NG→PASS / confirmed independent NG**, **False Reject = GOOD→NG / confirmed independent GOOD**, **Strict Defect Recall = NG→NG / confirmed independent NG**; NG→REVIEW/UNABLE ไม่นับว่าจับ NG สำเร็จ รายงาน REVIEW/UNABLE แยกเสมอ การไม่มี false accept ไม่เพียงพอหากทุกชิ้นได้ UNABLE

Validation metrics ปัจจุบันนับ **ภาพที่ eligible** ไม่ได้ deduplicate เป็นจำนวนชิ้นอิสระอัตโนมัติ; บันทึกผลต่อชิ้น/defect type และ repeat runs แยกใน field log ห้ามเพิ่มจำนวน independent samples ด้วยหลายภาพของชิ้นเดิม `mean_ms/p95_ms` คือ image-processing ของ rows ที่ทดสอบ ไม่ใช่ throughput โรงงาน; `cycle_elapsed_ms` คือเวลา observed place-to-removal รวม settling/removal confirmation ให้เก็บ median/P95 cycle time, inspected frames, frame age และ repeatability ต่อชิ้นจริงแยกกัน

### Manual Browser checklist ที่ยังต้องทำ

ใช้ server โค้ดล่าสุดเมื่อผู้ใช้พร้อม restart ตามขั้นตอนปกติ แล้ว Hard refresh; ตรวจ desktop และหน้าจอแคบ/zoom: navigation4แท็บ, ฟอร์มและปุ่มไม่ล้น/ซ้อน, canvas/ROI ตรงภาพเมื่อย่อจอ, errors อ่านได้และไม่มี console error

- Setup/Dataset: เปิดรายการจริงแบบอ่านอย่างเดียว; Upload→Preview→Metadata→Undo→Trash→Restore ให้ใช้ **temporary dataset แยก** ตรวจ token/ภาพเปลี่ยนหลัง upload ทำให้ Undo ถูกปฏิเสธ และ metadata ไม่ยืนยัน label เอง
- Test: โหลดสูตรเก่าและสูตร fixture/ทดสอบแยก ตรวจ Jobs/Models/Training history อ่านได้, ไม่มีการ Train/Use/Activate จากการเปิดรายการ; backend error ต้องแสดง ไม่ใช่ลิสต์ว่างที่ดูสำเร็จ
- Production: เมื่อไม่มี active recipe ปิด Start; old-engine pointer ต้องแสดงต้อง Validate; runtimeภายนอกทำให้ปุ่มที่ชนกัน disabled; จำลอง network offline ใน browser แล้วสีเขียว/ภาพเก่าหายและกลับได้เมื่อ online โดยไม่เปิดกล้องทดสอบ
- Review: เปิด Core Test/Legacy history/Production Evidence แยก source, complete และ human review; preview/คะแนน/เหตุผล/Recipe ถูกต้อง ไม่กด Human confirmation กับข้อมูลจริงเพื่อทดลอง ใช้ temporary fixture ถ้าจะทดสอบการเขียน

### เงื่อนไขถอด Legacy Live ถาวร

ต้องผ่าน full-frame Camera/Locator/Alignment/ROI และ input/model compatibility กับงานเดิม; independent human GOOD/NG validation ผ่านเกณฑ์ที่เจ้าของงานกำหนด; field sequence/cycle/failure/repeatability/timing/evidence ผ่าน และ browser checklist เสร็จ ยืนยันว่าความสามารถจำเป็นรวม optional product identification มีตัวแทนหรือเจ้าของงานยืนยันว่าไม่ต้องใช้ จากนั้นตรวจทุก Web/Terminal/Job/Training/Verification caller ใหม่ก่อนถอด branch และรัน regression โดยเก็บ artifacts/history พร้อม Git rollback; **ยังไม่ผ่านเงื่อนไขเหล่านี้ในรอบนี้**

---

## Phase 3 — ผลที่ลงมือทำแล้ว

Terminal **โหมด Core Candidate/Production** และ Web ใช้การรับภาพ, Recipe/Pipeline, Decision/Cycle และ Evidence ชุดเดียวกันเพิ่มขึ้นจริง ไม่สร้างแอปกล้อง/CLI/Pipeline ใหม่ ไม่เรียกกล้องจริงในรอบนี้ ไม่ Train หรือ Activate Production

**ยังไม่ได้ถอด Legacy Live ทั้งสาย** และไม่ได้เปลี่ยน process กล้องที่ผู้ใช้เปิดอยู่ให้เป็น Core โดยอัตโนมัติ

### MERGED / REFACTORED

- `Runtime.open_capture()` ใน Core เดิมเป็นทางเปิดกล้องร่วม ใช้ `OpenCVCamera`/`BaslerCamera` และ `FramePump` เดิม; Web กับ Terminal Core ไม่ตั้งวงรับภาพคนละแบบอีก
- Terminal USB ใช้ exact camera index ตาม Recipe ไม่เรียก fallback scanner; คงหน้าต่าง OpenCV, กรอบ, FPS, q/Esc, s/Space และ FPS/buffer settings เดิมไว้
- Recipe ที่เว้น width/height ว่างใช้ขนาดจาก driver ไม่ยืม640×480จาก legacy defaults; ภาพที่ส่งมาต้องตรง reference/Recipe โดยไม่มีการ resize เพื่อให้ตรวจผ่าน
- `FramePump` เก็บเฉพาะภาพล่าสุด ให้ frame sequence และเวลาที่ host รับภาพ เหมือนกันทั้ง Web/Terminal; `inspect_frame()` ตรวจอายุภาพรวมเวลาประมวลผลตามเกณฑ์เดิม
- ใช้ `Runtime.make_cycle()` สร้าง Cycle จาก Decision settings ของ Recipe จุดเดียวกัน การถือผล/วาง–ยก/Evidence ยังใช้ Core เดิม
- Live evidence เพิ่ม `capture.requested`, `capture.actual` และ timing scope ใช้ metadata จาก driver เดิม; เวลาเป็น host read completion **ไม่ใช่ sensor exposure timestamp** และ driver values ไม่ใช่หลักฐานว่า focus/แสงเหมาะสม
- Capture thread ถือ project lease ของตัวเองจนอ่าน/คืนกล้องจบ ถ้า caller หยุด timeout แต่ reader ยังอยู่ process อื่นยังไม่ได้สิทธิ์; metadata/start-thread/worker-start failure คืนทรัพยากรตามเส้นทางที่ทดสอบ
- ข้อความ Terminal ไม่แสดงว่า Camera released เมื่อการคืนกล้องล้มเหลว
- Engine version `2026.10.09.2`: การเปลี่ยน acquisition/timing ต้อง Validate ใหม่ก่อน Production; ไม่แก้ reports/recipes/threshold/production pointer เดิมย้อนหลัง

### REMOVED จริงใน Phase 3

- การใช้ legacy camera scanner/fallback สำหรับ **Core branch** ใน Terminal
- ลูป `cap.read()` และการสร้างเวลาภาพแบบ synchronous ที่ซ้ำกับ Core ใน branch ดังกล่าว ใช้ latest-frame pump แทน
- การสร้าง Core Cycle และการตั้งค่ากล้องซ้ำใน Terminal ย้ายไป runtime เดิม
- `locate_primary_object()` helper ที่ไม่มี caller หลังตรวจทั้ง Camera/Web/Training/Verification/tools และ tests; คง `analyze_primary_object()` ที่ผู้เรียกยังใช้อยู่
- Import/timestamp assignment ที่หมดหน้าที่หลังใช้ FramePump

ไม่ลบ scanner/cap.read ของ **Legacy branch** ที่ยังเป็นทางใช้งานจริง ไม่ลบ model weights/history และไม่ถอด dependencies ที่ยังใช้อยู่

### KEEP / สิ่งที่ยังต้องพิสูจน์ก่อนย้าย

| ความสามารถ | เหตุผลที่ยังไม่ถอด |
|---|---|
| YOLO Part Locator เดิม | `part_locator.analyze_part()` คืน box/confidence เป็นหลัก; Core ต้องมี transform, alignment limits และ source ROI polygon การแปลง box เป็น alignment เองจะเปลี่ยนความหมาย model/reference ต้องพิสูจน์ translation/rotation/scale/partial-part บนภาพจริงก่อน |
| Optional Brand identification | Web/Terminal reuse classifier เดิมอยู่แล้ว; Core Recipe ยังไม่มี requirement ว่าต้องใช้ Brand เลือกงานใด ไม่เพิ่ม Brand เป็น defect gate หรือย้ายเข้า Core เพื่อให้ครบจำนวน |
| Legacy stabilization / PartCycle | OK/REVIEW/NG, การถือ NG และ incomplete-cycle policy ต่างจาก Core PASS/NG/REVIEW/UNABLE; ไม่แทนที่จากชื่อ field ที่คล้ายกัน ต้องเทียบลำดับภาพและผลรายชิ้นกับงานจริง |
| Deep Patch / legacy reference | Model/ROI/input contract ของงานจริงยังเป็น cropped/fixed; ยังไม่มี full-frame Candidate พร้อม NG validation ที่ใช้แทนได้ |
| Legacy camera acquisition | คงค่ากล้อง/พฤติกรรม fallback ของ branch ที่ผู้ใช้กำลังใช้ ไม่เปลี่ยนร่วมกับการย้าย Core capture ในครั้งเดียว |

Terminal Core รอบนี้ยังเป็น USB/OpenCV ตาม contract เดิม; Basler provider ของ Core/Web คงอยู่ ไม่อ้างว่าทดสอบ pylon หรือเพิ่ม Basler Terminal สำเร็จ

### TESTED

- **327 Python tests + 5 subtests ผ่าน** หลังการแก้และ Cleanup ทั้งหมด
- **9 JavaScript tests ผ่าน** รวม navigation/data request/error/stale preview ของ Phase2 และ Core polling; ไม่ใช่ browser tests
- ใหม่/ปรับ: Terminal/Web ใช้ Candidate และลำดับภาพ fixture เดียวกัน ให้ state/cycle/ROI/quality/capture/evidence เท่ากัน รวม NG; explicit Candidate ไม่เปลี่ยน Production pointer
- ทดสอบ stale frames ไม่เป็น PASS, disconnect เก็บ incomplete UNABLE, ภาพเต็มไม่ถูกย่อ, camera index ไม่ fallback, blank resolution ไม่ใช้ legacy default, รูป/Recipe/capture metadata เชื่อมกัน
- ใช้ real thread + OS file lock กับ **กล้องจำลอง** ทดสอบ capture ค้าง/close timeout, lock คงอยู่จน reader จบ, start failure และ metadata failure คืนทรัพยากร
- Phase2 integration ยังผ่าน: Dataset lifecycle ใช้ temporary directories เท่านั้น, legacy/core records, jobs/models และ human confirmation ไม่เปลี่ยนความหมาย
- HTTP smoke แบบอ่านอย่างเดียว: `/inspection` และ9 APIs ของ Dataset/Jobs/Models/Runs/Legacy history/Evidence/Status/Recipes ตอบ200บน temporary server8011
- **Browser automation ยังใช้ไม่ได้**: `failed to write kernel assets ... os error 3`; ไม่รับรอง Layout/Button/Form ผ่าน browser จากผล HTTP/VM
- Hash records/config ก่อน Phase3 **462ไฟล์** และ model artifacts **72ไฟล์** ไม่เปลี่ยน/ไม่หาย; `project_config.json` ไม่มี diff
- temporary server8011 ปิดแล้ว กล้องเดิมและ server8000ไม่ได้ Restart ไม่ได้เปิดกล้องเพิ่ม

คำสั่ง regression:

```text
.venv-inspection\Scripts\python.exe -B -m pytest 07_web_app/tests -q -p no:cacheprovider
node --test 07_web_app/tests/test_live_polling.cjs 07_web_app/tests/test_workspace_ui.cjs 07_web_app/tests/test_inspection_validation_ui.cjs
```

### Candidate กับกล้องจริง — ยังไม่ได้ทำ / ต้องเก็บหลักฐาน

สถานะที่อ่านได้: Recipe7รายการยังเป็น `fixed`, ไม่มี Production pointer จึงยังไม่มี Candidate ที่ใช้แทน live place/remove ได้ ห้ามเลือกสูตรเก่าแล้วบังคับผ่านหรือเดา threshold

1. ผู้ใช้หยุดโปรแกรมที่ถือกล้องด้วย q/Esc ก่อนเริ่ม Candidate; อย่าเปิดสองโปรแกรมแย่งกล้อง จด serial/index, full resolution, focus, exposure/gain/driver settings และการจัดแสงจริง
2. ใน Setup ใช้ภาพเต็มจริงจากกล้องเดียวกันและ capture groups ที่ยืนยันได้ ทำ Candidate แยก เลือก Template/Features ตามการเคลื่อนจริง ตั้ง ROI หลัง alignment และใช้ model/reference/input setup ที่เข้ากันได้ โมเดลจาก cropped recipe เดิมไม่ถือว่าใช้แทนได้อัตโนมัติ
3. ถ้าข้อมูลยังไม่พอ calibrate quality/locator/anomaly หรือโมเดลยังไม่ตรง input contract ให้หยุดเก็บข้อมูล ไม่ Train/เปลี่ยนค่าเพื่อทำให้ผ่าน; การ Train ด้วยข้อมูลจริงต้องเป็นงานที่ผู้ใช้อนุญาต
4. เริ่ม **ไฟล์เดิม** จาก Project root โดยแทน placeholders ด้วย Recipe ID/index จริง:

```text
..\venv\Scripts\python.exe 01_camera\airbag_webcam_detection.py --candidate-recipe <saved_recipe_id> --camera-index <actual_index>
```

5. ตรวจข้อความ `CANDIDATE TEST - NOT PRODUCTION`, ขนาดเต็ม, ภาพสด/กรอบ/FPS, Quality/Locator และ ROI บนชิ้นงานจริง ทดสอบ GOOD/NG ที่คนยืนยัน, การเลื่อน/หมุนในขอบเขต, ชิ้นงานขาดขอบ, blur/แสงผิด, วาง–ยก และทำซ้ำแต่ละชิ้น
6. เก็บ Evidence ID, image/recipe hash, camera requested/actual, source polygon, score/threshold, frame age/sequence, decision/complete/cycle time และ ground truth/capture group แยกจาก machine result; s/Space เป็น sample ไม่ใช่ Human confirmation
7. ทดสอบหยุด/ถอดกล้องระหว่างชิ้นงาน: ต้องไม่เกิด complete PASS ที่ยังตรวจไม่จบ และต้องคืนกล้องก่อนเริ่มรอบใหม่ FramePump อาจข้ามเฟรมเมื่อ inference ช้า ต้องวัด latency และความเสถียรจริง ไม่ใช้ unit-test timing รับรอง throughput
8. ตรวจ source=`candidate`/`candidate_sample`, Production pointer ไม่เปลี่ยน, หนึ่งผลต่อรอบที่สมบูรณ์ ใช้ independent GOOD/NG ประเมิน False Accept/False Reject/Defect Recall, repeatability และเวลา/ชิ้น
9. ถอด Legacy Live ได้เมื่อ full-frame Candidate ครอบคลุมงานเดิม, alignment/ROI/optional product identification ได้รับการยืนยัน, fail-safe/cycle/evidence ผ่านกล้องจริง และ independent validation ผ่านเกณฑ์ที่เจ้าของงานกำหนดแล้ว

### Phase2 Manual Browser Verification ที่ยังค้าง

หลังเริ่ม serverด้วยโค้ดล่าสุดตามวิธีเดิมและ Hard refresh:
- Setup → Dataset library: Preview/Metadata ถูกต้อง ใช้ **temporary dataset แยกเท่านั้น** สำหรับ Upload→Undo→Trash→Restore ตรวจว่าไม่เปลี่ยน Ground Truth และไม่กระทบชุดก่อน
- Test → Jobs/Models/History: Core/Legacy แยกชัด การเปิดรายการไม่ Train/Activate, Legacy Test แสดงภาพและข้อผิดพลาดถูกต้อง
- Review: Machine result/Human confirmation/Legacy Test/Production Evidence แยกกัน การอ่านไม่ยืนยันผลเอง
- `/engineering`404, `/jobs` เข้า Test, Setup→Test→Production→Review ไม่มี console error/ปุ่มล้นจอ ตรวจ disabled state เมื่อ runtime busy และเมื่อ backend ติดต่อไม่ได้

### Git checkpoints

- Phase1: `9f3f8a1`; Phase2 migration/cleanup: `65434bf` → `bfe7520`
- ก่อน Phase3: tag `phase3-start-20261009` ที่ `bfe7520`
- หลัง Phase3: tag `phase3-live-20261009` ชี้ commit ของโค้ด/tests/รายงานรอบนี้ (`git log --oneline`)
- ไม่มี backup source ชุดใหม่ใน Project ไม่ push ไม่แตะ untracked files เดิม

---

## บันทึก Phase 2 (งานที่เสร็จก่อนหน้า)

## สถานะ

ย้ายความสามารถ Dataset/Jobs/Models/Legacy History ที่จำเป็นเข้าสู่ `/inspection` และถอดหน้า `/engineering` จริงแล้ว ไม่คัดลอกหน้าเก่า ไม่สร้าง Pipeline/Storage/API อีกชุด ข้อมูลเก่าอ่านจากที่เดิม

Phase 1 Camera/Candidate/Ownership/Validation safety คงเดิม ไม่เปลี่ยน Production configuration, model weights, threshold หรือ Train ข้อมูลจริง **ยังไม่อ้างว่าระบบทั้งหมดเหลือ inference implementation เดียว**: Legacy Live ยังรอ replacement ที่ผ่าน Field Validation

## KEEP

- `07_web_app/inspection` เป็น Shared Core; `01_camera/airbag_webcam_detection.py` และ OpenCV window/วิธีรันเดิมไม่แก้ใน Phase 2
- Candidate test, project lease, explicit defect-recall requirement, fail-safe และ Production validation gate จาก Phase 1
- Dataset folder services/API, Job test service, model registry, legacy cycle/review stores และ Core evidence/training records เดิม
- Legacy reference/locator/brand runtime และ backend เฉพาะทางสำหรับสร้าง/ตั้งค่า legacy model/jobs ยังจำเป็นระหว่างรอ replacement ผ่านหน้างาน ไม่สร้างหน้า Train แยกขึ้นมาทดแทน
- Dependencies ของ AI/กล้องที่ยังใช้จริง ไม่ uninstall library จากการลบหน้าเว็บเพียงอย่างเดียว

## MIGRATED / MERGED / REFACTORED

### Workflow หลัก

| ใน `/inspection` | ความสามารถ / Service ที่ reuse |
|---|---|
| Setup → Dataset library | Create, Upload files/folder/ZIP, Preview, Metadata, Undo latest upload, Trash/Restore ผ่าน `/api/dataset-folders*` และ `/api/deleted-dataset-folders*` เดิม |
| Setup → Build candidate | Core learning เดิม ไม่ย้าย Legacy Train-first workflow มาครอบ |
| Test → Jobs / Models / Training history | อ่าน Core training jobs และ legacy jobs/runs; แสดง Core model metadata และ legacy/optional model registry; ทดสอบภาพกับ saved legacy job ผ่าน service เดิมโดยไม่ Activate |
| Review → Evidence | Core evidence แสดง Machine result / source / Human review แยกกัน; Production Evidence แยกจาก Test/Candidate |
| Review → Legacy history | Legacy cycles และ review records อ่านตรง API/store เดิม พร้อมภาพ/score/threshold และปุ่ม Human review ที่บันทึกเฉพาะเมื่อผู้ใช้กด |

ส่วนเพิ่มเติมเป็น disclosure ที่พับได้ในหน้าเดิม โหลดประวัติเมื่อเปิด/Refresh ไม่ poll training/history/CVAT เพิ่ม ไม่มีหน้าใหม่หรือ API route ใหม่

- `/api/models/registry` เดิมเพิ่มรายการ `inspection_models` จาก Core store โดยอ่าน metadata ไม่โหลด AI model; ไม่มีปุ่ม Activate legacy model ใน library นี้
- `/api/jobs` เดิมเพิ่มลิงก์รายงาน Legacy Test ที่บันทึกอยู่; reuse endpoint ดาวน์โหลดรายงานเดิม ไม่คัดลอก/แปลงรายงานเป็น Production validation
- Legacy job test แสดงภาพ preview และผลเดิมชัดว่า Machine result/Test only ไม่มีการสร้าง human label เอง
- `/jobs` bookmark เดิมส่งตรง `/inspection#test`; `/` และ `/training` ยังส่ง `/inspection`; `/engineering` ถูกถอดและตอบ404
- Metadata เป็นคำอธิบาย dataset เท่านั้น ไม่ใช้ `Known-good Reference` หรือ metadata `reviewed` เป็น Ground Truth ของ Core

### Data safety ที่พบขณะย้าย

- การอ่าน metadata แปลงชื่อประเภทเก่าในหน่วยความจำเท่านั้น ไม่เขียน/ซ่อมไฟล์ย้อนหลังเพียงเพราะเปิด Preview หรือรายการ Dataset
- Undo เดิมรับ paths/remove-folder จาก browser และลบไฟล์จริง ถูกแทนที่ใน endpoint เดิมด้วย **server manifest + upload ID + SHA256 ของทุกไฟล์ในชุด** รวม annotation
- ตรวจไฟล์ทั้งหมดก่อนเริ่มย้าย ถ้า token เก่า/ไฟล์เปลี่ยน/manifest ไม่รองรับให้ปฏิเสธโดยไม่แตะไฟล์
- Undo ย้ายเฉพาะชุดล่าสุดไป Trash เดิม คืนได้เป็น Dataset แยก ไม่ลบชุดก่อนหรือ metadata ของ Dataset หลัก
- Upload ก่อน migration ที่ไม่มี safe manifest และ special background upload ที่ไฟล์อยู่นอก folder จะไม่มีปุ่ม safe Undo; ใช้การจัดการข้อมูลเดิมอย่างชัดเจน ไม่สร้าง manifest/อ้างว่า undo ข้อมูลที่ไม่ทราบที่มาได้
- ชุด Upload ในวินาทีเดียวกันใช้ staging/batch directory ไม่ซ้ำ ป้องกันชื่อภาพเหมือนกันทับข้อมูลก่อนหน้า
- ภาพ/ผลเก่าไม่มี Human confirmation ยังคงไม่ยืนยัน; Legacy NG ไม่กลายเป็น confirmed NG โดยอัตโนมัติ

## REMOVED จริง

- Route `/engineering` และ `training_page`
- `templates/training.html`, `templates/model_test.html`
- `static/inspection-area-editor.js` ที่ใช้เฉพาะหน้าเดิม
- `static/style.css` และ `static/training-*.css` ทั้ง9ไฟล์ที่หมดผู้ใช้หลังย้าย ไม่ได้ลบ `inspection.css`
- Duplicate category-upload endpoint `/api/datasets/upload` และ helpers เฉพาะมัน; ใช้ folder-upload endpoint เดิมเป็นทางเข้าหลัก
- `/api/datasets/summary` สำหรับ dashboard เก่า และ `/api/train/recommended-settings` ซึ่งหมด UI caller; internal summary/preflight ที่ยังมีผู้ใช้คงอยู่
- ฟังก์ชันเก่ารวม16ตัวใน `app.py` ที่หมดผู้เรียก รวม recommendation/ETA, detector dashboard analysis, legacy category-upload และ parser ที่ใช้เฉพาะ dashboard นั้น
- Unused `math` import
- 21 tests ที่ผูกกับ layout/CSS/Train-first navigation ของหน้าที่ถอดออก; ย้าย polling regression มาทดสอบ Core UI ที่ยังใช้งานจริง

ตรวจ imports, function references, HTML/JS URLs, CLI/scripts, runtime caller และ tests ก่อน/หลังลบ Source ลดสุทธิประมาณ36,000บรรทัด ส่วนใหญ่เป็น UI/CSS ที่เลิกใช้ ไม่มีการ rename เป็น `_old`/`_backup`

## TESTED

| ขอบเขต | ผล |
|---|---|
| ก่อนถอด UI/assets เก่า | 336 tests + 5 subtests ผ่าน |
| หลัง Cleanup และแก้ Upload collision | **317 tests + 5 subtests ผ่าน**: `.venv-inspection\Scripts\python.exe -B -m pytest 07_web_app/tests -q -p no:cacheprovider` |
| Python integration ใหม่ | 11 tests: Upload→Preview→Metadata→Undo→Restore, Trash/Restore, stale/changed batch, annotation hash, same-second uploads, metadata read-only, machine/human separation, old/broken jobs, Core/legacy registry และ retired routes |
| JavaScript | **9 tests ผ่าน** ใน `test_live_polling.cjs`, `test_inspection_validation_ui.cjs`, `test_workspace_ui.cjs`; `node --check` ผ่าน |
| UI contract | อ่าน HTML route จริง ตรวจ ID ซ้ำ/controls; VM รัน controller จริง ตรวจ request payload, stale preview, errors, Trash response, Job preview, human-status/XSS escaping และอ่าน history โดยไม่มี mutation |
| Cross-module syntax/callers | AST parse67ไฟล์ Camera/Training/Dataset/Verification/Brand/Web/tools ผ่าน; ไม่พบ caller ของ assets/routes/functions ที่ถอดใน active source |
| HTTP smoke กับข้อมูลจริง แบบอ่านอย่างเดียว | `/inspection` และ9 APIs ของ Dataset/Trash/Jobs/Models/Training/Evidence/Legacy history ตอบ200 บน temporary server8011 |
| Browser จริง | **ยังไม่ผ่านการยืนยัน**: CUA เปิดไม่ได้ (`failed to write kernel assets ... os error 3`); HTTP/VM ไม่ใช่ browser test |
| Data/model preservation | SHA256 ของไฟล์ภาพ/data/config/model artifacts เดิม **8,213ไฟล์ ไม่เปลี่ยนและไม่หาย**; ไม่พบ diff ใน `project_config.json` |
| Runtime preservation | ไม่มี diff ใน Camera, Core runtime/ownership/validation; regression เดิมผ่าน ไม่เปิด/แย่งกล้องจริง |

จำนวน tests ลดจาก327ใน Phase1 เพราะลบ21 tests ของ UI ที่ถูกถอด และเพิ่ม11 integration tests; ไม่ใช่ skip failing regression Tests ภาพชั่วคราวยืนยัน software behaviour เท่านั้น ไม่ยืนยัน Defect Accuracy

ระหว่างทำงานพบ Legacy camera เพิ่ม cycle ใหม่12รายการ (JSON+JPG รวม24ไฟล์) ซึ่งระบุ source=camera_0 และมี process กล้องเดิมกำลังรัน จึงรักษาไว้ทั้งหมด ไม่ใช่ข้อมูลทดสอบที่ผู้ช่วยสร้าง และไม่ถือเป็นผลยืนยันกล้อง/accuracy ของโค้ดใหม่

temporary server8011 ปิดแล้ว ไม่ Restart server8000หรือโปรแกรมกล้องของผู้ใช้ โค้ดใหม่มีผลเมื่อเริ่มโปรแกรมตามวิธีเดิมครั้งถัดไป

## Git checkpoints

- `f2decd0`: baseline ก่อน Phase1
- `9f3f8a1` / tag `phase2-start-20261009`: จุดเริ่ม Phase2
- `65434bf`: migration ที่ผ่าน regression ก่อนถอด UI/assets
- Cleanup และรายงานนี้อยู่ใน commit ถัดจาก migration (`git log --oneline`)

ไม่มีสำเนา Source ใหม่ใน Project ไม่แตะ untracked files เดิมของผู้ใช้ ไม่มี push/remote เปลี่ยนแปลง ไฟล์ hash สำหรับตรวจรักษาข้อมูลเก็บใน chat workspace นอก Project

## BLOCKED / ข้อจำกัดที่ยังเหลือ

1. **Legacy Live runtime**: ยังไม่ถอด เพราะไม่มี full-frame Candidate ที่พิสูจน์ locator/alignment/ROI/anomaly/decision/cycle เทียบงานจริงด้วย human-confirmed GOOD/NG จึงคงเฉพาะ runtime และ backend/utility ที่รองรับมัน ไม่คงหน้า `/engineering`
2. **Browser layout/interaction**: เครื่องมือเปิด browser ล้มเหลว ต้องตรวจด้วยมือ/เครื่องมือที่ใช้งานได้ตามขั้นตอนด้านล่าง
3. **Hardware/accuracy**: ไม่ได้ทดสอบ LifeCam/Basler, disconnect, latency หรือ Defect Recall จริงในรอบนี้ ไม่ Train, Activate Production หรือเปลี่ยน threshold
4. Legacy cycle API/UI แสดงล่าสุด100รายการตาม limit เดิม; ข้อมูลเก่าทั้งหมดยังคงอยู่บน disk ส่วน Legacy Test reports เข้าถึงจากรายการ Jobs ได้

## NEXT / Manual smoke checklist

หลังเริ่ม server ด้วยโค้ดล่าสุดตามวิธีเดิม เปิด `/inspection` และ Hard refresh (ไม่ต้องเปิดกล้องหรือ Production):

1. Setup: เปิด Dataset library เลือก folder เดิม ตรวจ Preview/Metadata และกลับเข้าหน้าใหม่แล้วข้อมูลไม่เปลี่ยนเอง ไม่เห็น GOOD/NG confirmation จากชื่อ folder
2. ใช้ **folder ทดลองแยก**: Upload ภาพที่มีอยู่สองชุดชื่อไฟล์ซ้ำ ตรวจว่าอยู่ครบทั้งสองชุด → Undo ล่าสุด → ชุดแรกยังอยู่ → Trash Restore ชุดที่ Undo กลับเป็น folder แยก; Metadata ไม่ทำให้ภาพเป็น Ground Truth
3. Test: เปิด Jobs/Models ตรวจ Core candidate training กับ Legacy jobs/runs/model artifacts แยกประเภท; เลือก Legacy job ที่เข้ากันได้และภาพจริงเมื่อต้องการทดสอบ ต้องเห็น Test only/ภาพ preview และไม่มี Production activation
4. Review: ดู Core Test/Candidate/Production source แยกกัน เปิด Legacy history ตรวจ Machine NG กับ human status แยกกัน; การอ่านไม่ยืนยันผลเอง และไม่มี CVAT polling เพิ่ม
5. ตรวจ `/engineering`404, `/jobs` เข้าหน้าTest, ไม่มี request หา CSS/JS หน้าเก่าหรือ errors ใน console; ตรวจที่ความกว้างจอใช้งานจริงว่า disclosure/ปุ่มไม่ล้น
6. เก็บ GOOD/NG ที่มนุษย์ยืนยันพร้อม Capture groups และ full-frame camera setup จริง เพื่อพิสูจน์ Candidate ผ่านโปรแกรมกล้องเดิมก่อนพิจารณาถอด Legacy Live/backend ที่ยังใช้กับมัน

เกณฑ์จบ Phase2 ด้าน Source/API/Regression ผ่านตามขอบเขตด้านบน; Browser และ Field Validation ยังไม่รับรอง
