# Airbag_Project — Renovation & Consolidation

อัปเดต 9 ตุลาคม 2026 · Phase 2 · Project: `D:\Users\dev06\Desktop\Airbag_Project`

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
