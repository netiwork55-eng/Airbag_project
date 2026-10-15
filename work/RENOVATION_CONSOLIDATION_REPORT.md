# Airbag_Project — Renovation & Consolidation

อัปเดต 14 ตุลาคม 2026 · Final Sprint — controlled stain demonstration · Project: `D:\Users\dev06\Desktop\Airbag_Project`

## ขยายหน้าต่างกล้องเดิม — 15 ตุลาคม 2026

- ตามคำขอผู้ใช้ ปรับ OpenCV preview เป็น WINDOW_NORMAL รักษาสัดส่วน เริ่มขนาด 1280x960 และลากขยาย/maximize ได้ ทั้งเส้นทางกล้องเดิมที่มีอยู่
- ไม่ resize ภาพ inference/Evidence ไม่เปลี่ยน camera resolution, threshold, model หรือ Recipe
- Checkpoint ก่อนแก้: tag `camera-display-before-20261015` ที่ `69f6539`
- Regression เดิม 23 tests ผ่าน (camera/core/entrypoint/full-frame); เพิ่ม mock window operations เพื่อไม่เปิด UI ระหว่าง tests
- เรียกเปิดโปรแกรมเดิมแล้ว จากนั้นผู้ใช้แจ้งว่าจะตรวจหน้าต่างเอง จึงไม่ทำ UI verification ต่อ ไม่อ้างว่าตรวจภาพสดหรือขนาดหน้าต่างจริงผ่านแล้ว

## ถอดโหมด Candidate จาก Terminal — 15 ตุลาคม 2026

ตามขอบเขตที่ผู้ใช้ยืนยัน: เหลือวิธีเปิดกล้องเดิม เก็บ `/inspection`, Shared Core และข้อมูลเดิมทั้งหมด ยุติแผนทดสอบ Terminal Candidate 5 GOOD / 5 คราบจำลองในรอบนี้ ไม่อ้างว่าทำครบหรือผ่าน repeatability

- Checkpoint ก่อนลบ: `d9c6bd0`; ก่อนหน้านั้น `b6aa927` เก็บผล Demo เดิม
- ลบ argument, branch เลือก Candidate, parameter เฉพาะโหมด, ข้อความ overlay/console และการสร้าง candidate_sample จาก Terminal จริง ไม่มี alias หรือโปรแกรมกล้องตัวแทน
- เปิดตามเดิม: `..\venv\Scripts\python.exe 01_camera\airbag_webcam_detection.py` คงหน้าต่าง OpenCV, กรอบ, FPS และปุ่มเดิม การเลือก Production ที่ผ่านการ Activate อย่างถูกต้องยังคงตาม logic เดิม; ไม่มีการ Activate หรือแก้ configuration ในงานนี้
- `/inspection` Test/Validation, Recipe, Evidence และ Runtime ownership ยังอยู่ ข้อมูล candidate_sample เก่ายังอ่านได้
- Data hash ก่อน/หลัง: 1,162 ไฟล์ข้อมูล/โมเดล/metadata/configuration เดิม hash ตรงกัน ไม่มีข้อมูลสำคัญหาย; `.runtime-owner.json` เป็นสถานะ ownership ชั่วคราวซึ่งหายไประหว่างตรวจ จึงแยกจากข้อมูลถาวร ไม่ได้ลบด้วย cleanup
- Regression กล้อง/entrypoint/capture/cycle 46 tests ผ่าน; API/ownership/evidence/validation preview 28 tests ผ่าน; JavaScript UI contract 6 tests ผ่าน รวม Python 74 + JS 6 ไม่ใช่ Browser หรือ Field Accuracy validation
- คำสั่ง Candidate เก่าถูกปฏิเสธก่อนเข้า runtime/เปิดกล้องใน regression test; production Terminal/Web parity และการปล่อยกล้องเมื่อผิดพลาดยังผ่าน
- ไม่ restart หรือแย่งกล้องจาก process ที่เปิดอยู่; source ใหม่มีผลในการเปิดครั้งถัดไป
- สำรองภาพ/Evidence ก่อนการถอดไว้ภายนอก Source Tree ที่ `D:\Users\dev06\Documents\Codex\Airbag_Project_Backups` (ZIP ของ repeatability ตรวจ SHA256 แต่ละไฟล์แล้ว)

## Final Repeatability Verification — 15 ตุลาคม 2026 (ยุติตามคำสั่งผู้ใช้ให้ถอดโหมด Terminal Candidate)

- G01 clarification: user confirms repeated actual removals/reinsertions during pre-ready positioning. The three machine-completed records are therefore NOT proof of duplicate-cycle software failure; exact physical matching remains unverified. Keep that interval as setup, not three scored trials.
- 15 Oct 2026 14:06 ICT: original Terminal ended with Camera did not deliver an image; camera release logged, incomplete UNABLE evidence `event_232f24c9725e4157b371493b006e1105`, last image black. Cause unconfirmed. Re-enumerated LifeCam index 1 and restarted original command unchanged. At 15:17 ICT G02 fresh sample `event_a35f115180624a58837c098f9b0c9736`: 1920x1080, UNABLE alignment score 0.390274, age 125 ms, processing 124.612 ms. Exposure -6/focus 8/autofocus 1 driver values match earlier. Part still tilted; table framing differs; asked about physical setup movement and requested guide orientation correction. Restart began with part present, so entry/cycle not verified. No threshold, Recipe or source change.

- 15 Oct 2026 11:09 ICT: G01 removal confirmed by user and empty raw image `event_e289c8572127460596bc96042437be30` (also E02, second distinct empty interval): UNABLE, locator 0.234186, age 78 ms, processing 78.717 ms. One requested GOOD placement/removal attempt is complete physically, stationary result UNABLE; end-to-end cycle mapping is NOT verified. Three earlier machine-completed records (REVIEW/NG/NG at 11:05:03/23/38) precede the stationary sample, and none was created after the final removal. User clarification requested on intervening physical movements; do not count these as three verified trials. Requested G02 with guide orientation corrected physically, unchanged recipe/thresholds. Current counts: 1 requested GOOD attempt (failed alignment), 0 simulated-stain attempts, 2/3 verified empty intervals; no repeatability/accuracy claim.

- 15 Oct 2026 11:06 ICT: G01 stationary GOOD sample (user-confirmed condition) `event_0bff90344ba644f7a25fd600dccbb73b`: UNABLE_TO_INSPECT, template score 0.541420 below unchanged 0.800 limit; no inspection correlation score because alignment failed. Actual 1920x1080, age 109 ms, processing 84.047 ms; brightness 155.918, sharpness 77.173, blur gate still disabled. Part is visible with hands outside inspection region; orientation differs from reference, and existing template alignment corrects translation only. Physical removal and software-cycle completion still pending. No threshold/source-code change.

- 15 Oct 2026 11:03 ICT: E01 verified from fresh raw image after user removed the part: `event_928af877f0a74169b7f960d241bb3d56`, image `image_a0c5bcca04df49644e766ae7f874ea482ad02586dc2b5ff9da451ff1f15e220a`. Empty scene returned UNABLE_TO_INSPECT, locator 0.220566 / part_present=false, no area correlation, no PASS, no new completed-cycle record. 1920x1080, frame 6172, age 109 ms, processing 72.905 ms. Counts: 1/3 empty-scene tests; 0/5 completed GOOD and 0/5 completed simulated-stain physical placements. G01 requested with original reference orientation.

- 15 Oct 2026 11:01 ICT: original Terminal Space shortcut successfully saved `event_92d983849866417faa0b859c865e4868` (1920x1080, frame 2232, age 109 ms, processing 86.747 ms). Raw image still contains the part despite requested EMPTY condition. Locator score 0.331528; UNABLE_TO_INSPECT before area correlation; sample only, complete=false. Not counted as an empty-scene or completed physical-cycle trial. Await actual part removal or clarification of a possibly frozen image; no threshold or source-code change.

รับงานต่อจาก staged Sprint และ commitผลเดิมเป็น `b6aa927`; checkpointก่อนรอบใหม่ `repeatability-start-20261015` ใช้ Candidate `recipe_a08d724e03d5421f82b0a985671a4053` hashเดิม ไม่เปลี่ยน Threshold/Locator/Recipe, ไม่ Train/Activate/ถอดLegacy ผู้ใช้ยืนยันจัดไกด์วางแบบไม่ทำความเสียหายและพื้นที่ว่างพร้อมแล้ว

- เป้าหมายคือGOOD5 placements, authorized simulated-stain5 placements, EMPTY3 observations; ทุกplacementต้องมีground-truth conditionที่ผู้ใช้ระบุ, stationary sample, removalและEvidenceที่จับคู่ได้ จำนวนเฟรมและcycle recordsไม่ใช่จำนวนชิ้นจริง
- เปิดTerminalเดิมสองครั้งแรกแล้วเกิด `Camera did not deliver an image`; หลักฐานrawเป็นภาพดำ เก็บincomplete UNABLE2records ไม่มีรอบphysicalที่ยืนยันผ่าน ระบุแยกจากผลตรวจตำหนิ Adapterเดิมที่1920×1080ก็เปิดไม่สำเร็จในช่วงหนึ่ง ไม่สรุปสาเหตุเป็นสาย/driverจากerrorเพียงอย่างเดียว
- หลังผู้ใช้สั่งเปิดใหม่ ตรวจDirectShowพบLifeCam index1และไม่มีOS lease holder แล้วเปิดTerminalCandidateเดิมอีกครั้ง รอหลักฐานE01ก่อนเริ่มนับการทดสอบ ห้ามถือการเปิดหน้าต่างสำเร็จเป็นการทดสอบผ่าน
- Targeted regressionวันนี้41testsผ่านใน8.54s ไม่มีsource change Browser automationก่อนหน้านี้ถูกปฏิเสธสิทธิ์ควบคุมChrome จึงใช้manual smoke testเมื่อจบการเก็บภาพ ไม่ใช้HTTP200เป็นหลักฐานBrowserผ่าน
- สำรองภาพทดลอง, Recipe, raw sourceและCore EvidenceของSprintเดิมแล้ว107ไฟล์ใน `D:\Users\dev06\Documents\Codex\Airbag_Project_Backups\sprint-before-repeatability-20261014-145041.zip` (141261805bytes) ตรวจCRCและSHA256แต่ละไฟล์ในarchiveตรงต้นทาง Archive SHA256=`8520e9ed2abff8ac3afbb88cd60ef49ca96b709867fa5b467faefcad6bba0e91` เป็นสำเนาบนเครื่องเดียวกัน ไม่ใช่off-site backup
- ตรวจข้อความสไลด์10หน้าเดิม: Classical CV/AI และข้อจำกัดaccuracyแยกไว้แล้ว จุดที่ควรเสริมในสไลด์ผลคือกรอบสีแดงเป็นInspection ROI ไม่ใช่ขอบเขตคราบที่โมเดลหาเจอ ยังไม่อัดDemo videoเพราะrepeatabilityยังไม่ยืนยัน

Trial log: `work/repeatability-20261015/physical-trials.json`; acquisition failure diagnostic: `camera-delivery-diagnostic.json` จะแสดงจำนวนphysicalที่ยืนยันหลังจบรอบเท่านั้น

---

## Final 3-Day Sprint — Controlled stain demonstration (14 ตุลาคม 2026)

ฐาน Source `3f07c11`; checkpoint `sprint-start-20261014` เป้าหมายรอบนี้คือการพิสูจน์คราบหนึ่งประเภทด้วยระบบเดิม ไม่มีการเปลี่ยน Architecture/เพิ่ม AI model ชิ้นงานที่ผู้ใช้ระบุในภาพก่อนหน้าเป็นคราบสีขาวใต้โลโก้ ส่วน **การเก็บภาพ GOOD/NG รอบ Sprint ผู้ใช้ยืนยันว่าเป็นชิ้นเดียวกันและเพิ่มคราบเพื่อทดสอบ** จึงเป็น physical simulation ไม่ใช่ independent production NG และไม่ยืนยันว่าชิ้นนี้เป็นชิ้นเดียวกับ legacy268ภาพ

### วัตถุประสงค์ / Workflow / วิธีที่ใช้จริง

Prototype นี้รับภาพเต็ม ตรวจความพร้อมภาพ จัดแนวชิ้นงาน แล้วเปรียบเทียบพื้นที่ผิวด้านหน้ากับ GOOD reference เพื่อแสดง Decision และเก็บหลักฐาน ใช้ `/inspection` เป็น Setup/Test/Review และ `01_camera/airbag_webcam_detection.py` เป็น Terminal/OpenCV เดิม ทั้งสองใช้ Camera adapter, FramePump, Pipeline, Decision, Cycle และ Evidence เดียวกัน ขอบเขต Demo แคบลงเหลือคราบบน ROI ผิว ไม่ได้ตรวจผิวทุกด้าน โลโก้ ขอบ หรือความปลอดภัยการทำงานของ Airbag

PaDiM/PatchCore มี artifact เดิมแต่ input signature ไม่ตรง Full-frame ใหม่ จึงไม่เรียก inference และไม่ Train เพิ่ม ใช้ `presence` เดิมใน Core ซึ่งคำนวณ OpenCV normalized reference correlation; **เป็น Classical CV ไม่ใช่ AI anomaly model** การที่ชิ้นคราบได้ NG ไม่เพียงพอจะบอกว่าระบุขอบเขตคราบได้หรือแยกคราบออกจากเงา/มือได้

### จุดติดขัดแรกและการทดลองที่ทำจริง

ไม่พบหลักฐานว่าต้องแก้โค้ด alignment จากภาพก่อนหน้า จึงควบคุมการวางชิ้นงานและใช้ implementation เดิมก่อน ตรวจ DirectShow อีกครั้งพบ LifeCam index1, Sharing-Media0, OBS2; PnP LifeCam OK, ไม่มี Terminal/OS lease holder, Web Core stopped และ Production=None ก่อนเปิดแต่ละรอบใช้ lease เดิมและปล่อยสำเร็จ ไม่ kill/restart process

| การรับภาพจริงใน Sprint | GOOD ก่อนเพิ่มคราบ | เพิ่มคราบจำลอง |
|---|---:|---:|
| Actual resolution | 1920×1080 | 1920×1080 |
| ช่วงรับภาพ | ประมาณ8s | ประมาณ8s |
| Consumer frames | 227 | 226 |
| Producer FPS | 29.963 | 29.907 |
| ภาพเต็มที่เก็บหลังช่วงเริ่มต้น2s | 11 | 11 |
| Exposure / Focus / Autofocus จาก driver | −6 / 8 / on | −6 / 8 / on |

ไม่ได้เปลี่ยน exposure/focus; autofocus ยังเปิดจึงอาจเปลี่ยน focus เองได้ gain−1เป็น sentinel ใช้ไฟล์ `good-capture.json` / `ng-capture.json` สำหรับ timestamps, quality และ frame age ระดับเฟรม FPS ในตารางเป็นการรับภาพ ไม่ใช่ defect throughput

Reference `good_00.png` ผู้ใช้ยืนยัน GOOD และวางให้ถ่ายแล้ว เลือก logo anchor `[880,320,1140,445]` ที่ไม่รวมคราบกับ ROI ผิว `[820,445,1200,635]` หน่วยpxบน1920×1080 ก่อนประมวลผลชุดคราบ ใช้ min_score=.8/min_matches12 และ pose limits เดิม ไม่ลดเกณฑ์ Geometry

| Geometry / raw surface score | GOOD อื่น10ภาพ (ไม่รวมreference) | คราบจำลอง11ภาพ |
|---|---:|---:|
| Template accepted | 10/10 | 11/11 |
| ORB accepted | 10/10 | 11/11 |
| Template-aligned surface correlation | .932057–.945505 | .647617–.701649 |
| ORB-aligned surface correlation | .909575–.945944 | .695855–.714388 |
| Template-aligned ROI brightness | 43.056–43.977 | 47.099–48.548 |

เป็นหลักฐานว่าการวางคงที่ทำให้ Core เดิมจัดแนวผ่านได้ในภาพรอบนี้ ไม่ใช่ทุกมุม/แสง และความสว่างผิวที่เปลี่ยนเป็นตัวแปรกวน คะแนนต่างไม่ใช่ข้อพิสูจน์ว่าตอบสนองเฉพาะคราบ ทุกภาพ GOOD/คราบรอบนี้อยู่ physical group `sprint-20261014-same-physical-part-01` ห้ามแยกไป train/calibration/independent validation คนละกลุ่ม

### Candidate ที่ผู้ใช้อนุญาตอย่างชัดเจน

ผู้ใช้ตอบ “อนุญาต Candidate OpenCV สำหรับ Demo เท่านั้น” หลังเห็นข้อเสนอและตัวเลข จึงสร้าง **`recipe_a08d724e03d5421f82b0a985671a4053`**, hash `385f5cfde1d9675b2b8e5e65842031238ddff54d16788135b86a0091f61e1b6d` Reference `image_05d021d36a55996ca690a6b62d6e74fd43adca08b4c4830a606573a632b74d1e` โดย `Store.put_image/save_recipe` เดิม ไม่มี training job ไม่มี Production pointer

- REVIEW band คือช่องว่างระหว่าง max observed simulated-NG `.7016487718` กับ min observed GOOD `.9320566058`; threshold=midpoint `.8168526888`, review_margin=half-gap `.1152039170` เป็น empirical demo calibration จากชิ้นเดียว ไม่ใช่ค่าหน้างานที่ validated แล้ว และไม่ใช้ภาพชุดเดียวกันรายงาน accuracy
- Quality ยังคง limits1–254/min_blur0 ตามขอบเขต diagnostic ที่แจ้งไว้: ยัง **ไม่ calibrated** สำหรับคราบจริง จึงไม่อ้าง quality gate พร้อมโรงงาน
- Runtime.candidate_pipeline ผ่าน warm-up กับ referenceตัวเอง (ไม่นับเป็น independent proof) และเปิด Terminal เดิมด้วย explicit Candidate สำเร็จ ไม่เปิด legacy pipeline ซ้อน
- นำเข้าเฉพาะ GOOD reference ใหม่ให้ Core; ภาพคราบและ samples จาก Terminal คง machine/unconfirmed records ไม่อัปเกรดเป็น human-confirmed production NG การยืนยัน physical simulation เก็บใน manifest ของการทดลอง

### ผล Live / ข้อจำกัด

ผล Live อยู่ `work/sprint-20261014/live-evidence-index.json` เชื่อมไป Core Evidence จริงและ Recipe เดียวกัน บันทึกผลสุดท้ายหลังปิดกล้องไว้ต่อด้านล่าง การกด s คือ unconfirmed `candidate_sample`, complete=false; ไม่ใช่หนึ่งชิ้นที่จบ Cycle

**ผลสุดท้ายของรอบ Sprint:** Terminal Candidate เดิมเปิดได้จริง และปล่อยกล้องสำเร็จ ได้ `candidate_sample` NG14รายการ (กด s, complete=false) คะแนน .549348–.562304, Pipeline73.739–86.299ms เฉลี่ย80.513ms และ frame age78–109ms เป็นค่าของ samples ที่บันทึกเท่านั้น ไม่ใช่ all-frame distribution หรือเวลาครบหนึ่งชิ้น ไม่ใช่ AI inference

มี `candidate` complete=true NG3records: `event_bcaf72d9eda1448ab47183cba02b662a` (125frames/15.032s), `event_fc364934e6fb40729aee309a13975234` (3frames/.719s), `event_5d22d474df934043a8446e767db700a6` (628frames/81.031s) จำนวน3recordsไม่ได้แปลว่า3ชิ้นอิสระหรือหนึ่งผลต่อรอบถูกต้องแล้ว ยังไม่มี time-aligned video/physical cycle tally เพื่อจับคู่ทุก record ภาพของrecordสุดท้ายมีมือบังROIตอนยกและscore−.045069; จึงใช้เป็นหลักฐานข้อจำกัดของ occlusion ไม่ใช่ความไวต่อคราบ

ผู้ใช้แจ้งทดสอบGOODและปิดแล้ว แต่ไม่พบsaved GOOD sampleใหม่จากTerminal จึงขอวางGOODอีกครั้งและรับภาพใหม่ด้วย Runtime.open_capture/inspect_frameเดิม10s โดยคงRecipeทุกค่า **134เฟรมได้UNABLE_TO_INSPECTทั้งหมด** ที่locator; Template score .381545–.393618 ต่ำกว่า .8 ภาพ `good_retest_00.png` แสดงชิ้นงานหมุนต่างจากreference ลองORBเดิมบน4ภาพที่เก็บแบบofflineก็ไม่ผ่าน (inlier ratio .406–.500) ไม่เปลี่ยนCandidateเป็นORB/ลดthreshold

บันทึกcandidate_sample UNABLE4ภาพไว้ในCore (เช่น `event_dfa91654977f467baabb10a37831ffff`) ไม่มีcycleที่จบเพราะlocatorไม่เคยพบชิ้นงานในรอบนี้ ดังนั้น **ยังไม่เป็น Demo ที่ทำซ้ำผ่านทั้งGOOD/คราบหลังยกวางกลับ** จุดถัดไปคือควบคุมตำแหน่ง/มุมด้วยจุดอ้างอิงหรือfixtureและพิสูจน์การยกวาง ไม่ใช่เพิ่มAI/ปรับthresholdเพื่อบังคับPASS การทดสอบที่สำเร็จจริงคือacquisition→quality→alignment→ROI→OpenCV method→NG→candidate Evidenceในภาพคราบตำแหน่งควบคุม ส่วนGOODทดสอบกลับล้มเหลวอย่างfail-safe

**Software / UI / Data checks:** focused regression41testsผ่าน (full-image evidence, existing Terminal/Core, unified capture, capture-group splitting); full regressionล่าสุดก่อนSprint336+5subtests ไม่รันซ้ำทั้งชุดเพราะไม่มีsource change ตรวจHTTP /inspection, status, recipes และNG Evidence previewตอบ200และProduction=None แต่ **Browserจริงยังค้าง** ไม่มีbrowserautomationที่เรียกได้ ไม่อ้างHTTPเป็นBrowser pass Manual: เปิดSetupเลือกCandidateนี้ ตรวจreference/ROI/quality-disabled warning; TestเปิดNGและUNABLEของสูตรเดียวกัน; Reviewเทียบraw/previewและsource; Productionต้องยังinactiveและไม่มีvalidationreportที่eligible อย่าTrain/Activate

ตรวจSHA256ข้อมูลเดิม879ไฟล์ในinspection_data/jobs/modelsและrootconfigก่อน–หลัง **ไม่มีไฟล์เดิมเปลี่ยน/หาย**; เพิ่มเฉพาะGOOD reference1ภาพ, Candidate1สูตร, rawimages/Evidenceใหม่จากการทดลอง ข้อมูลกล้องรอบใหม่และprovenanceอยู่`work/sprint-20261014`; `artifact-hashes.json`เชื่อมhashไฟล์หลักฐาน ไม่มีsource/dependency/legacy/Production change ไม่มีการTrainหรือclaimaccuracy กล้องปล่อยแล้วและOS leaseว่าง

รายงานฉบับนี้เป็นเอกสารหลักสำหรับส่งงาน; สไลด์10หน้า `work/sprint-20261014/AIRBAG_SPRINT_RESULTS_20261014.pptx` ใช้ข้อเท็จจริงและภาพจากการทดลองพร้อมข้อจำกัด ตรวจ renderทุกหน้า/package/font/layout/import แล้ว แต่ไม่ได้เปิดตรวจในPowerPoint Video Demoยังไม่ได้อัด และไม่สร้างวิดีโอจำลองให้เหมือนการตรวจสำเร็จ

### ขั้นตอน Demo ที่ทำตามได้

1. ตรวจ device index และไม่มี process ถือกล้องก่อนรัน อย่ายึด index1 ถาวรหลัง replug เปิดด้วย environment เดิมจาก project root:
คำสั่ง Terminal Candidate ที่เคยใช้ในขั้นตอนนี้ถูกถอดเมื่อ 15 ตุลาคม 2026 ตามคำสั่งผู้ใช้; เก็บผลเดิมเป็นประวัติ ทดสอบ Candidate ด้วยภาพผ่าน `/inspection` ต่อได้
2. ใช้กล้อง1920×1080 ตำแหน่ง/แสงตาม `good_00.png`; วางชิ้นงานแล้วเอามือออก รอผลนิ่ง กด s ครั้งเดียวเพื่อเก็บ raw sample ยกออกและรอพื้นที่ว่าง ≥2s เพื่อปิด cycle ตรวจชื่อ Candidate และป้าย NOT PRODUCTION
3. ตรวจ GOOD ก่อนเพิ่มคราบและหลังเอาคราบออก ตลอดจนชิ้นคราบจำลองเดิม บันทึกตามจริงเมื่อ REVIEW/UNABLE/NG ผิดคาด ไม่ปรับ threshold หน้างานเพื่อทำให้ PASS
4. Review ต้องอ่านทั้ง raw image, source polygon, score, threshold, quality, locator และ cycle completion; มือบังROI/เงา/การยกวางต้องแยกจากตำหนิจริง ห้ามนำค่าเวลาจากตัวอย่างเฟรมไปอ้างเป็นเวลาหนึ่งชิ้น
5. ปิดด้วย q/Esc รอ Camera released และตรวจ lease ว่าง ไม่ Activate Production/ถอด Legacy Live ใช้ภาพและ Evidence นี้สำรองการนำเสนอจนกว่าจะอัด Demo video จริงได้ ไม่ทำ animation ภาพนิ่งแล้วเรียกว่าการทดสอบ Live

### แผนส่งงาน 3 วัน / Stop condition

วัน1: เก็บคู่GOOD/คราบจริงผ่านกล้องและพิสูจน์ alignmentในตำแหน่งควบคุม; ทดลอง Candidate Classical CV ที่อนุญาตและเขียนผลล้มเหลวควบคู่ วัน2: ทดสอบยกวางซ้ำ GOOD/คราบและพื้นที่ว่าง ตรวจ false rejection/มือบัง/แสงจากข้อมูลจริง แก้เฉพาะ critical problem และสรุปผล วัน3: freeze revision/Recipe ที่พิสูจน์ได้ จัดสไลด์พร้อมข้อจำกัด อัดหน้าต่างโปรแกรมเดิมและซ้อมอธิบาย หาก input/model/field validation ยังไม่พร้อม ให้หยุดขยาย software และส่งขอบเขตที่พิสูจน์ได้ ไม่อ้างว่า AI/Production พร้อม

สำหรับ Production ต้องมีชิ้น GOOD/NG จริงอิสระจาก reference/calibration, defect typeและขนาด, ROI coverage ที่ทวนสอบ, controlled settings, place/remove repeatability และคุณภาพภาพที่ calibrate; คำนวณ False Accept, False Reject, Strict Defect Recall พร้อม REVIEW/UNABLE rates ตาม policyที่เจ้าของงานยืนยัน แล้วจึงผ่าน Validation Gate เดิม ไม่คำนวณอัตราผิดพลาดจากคราบจำลองชิ้นเดียวหรือจำนวนframesแทนจำนวนชิ้น

---

## Full-frame Candidate preparation — ข้อมูลจริงและขอบเขตที่ผู้ใช้ยืนยัน

ฐาน `ddb5b60`; checkpoint ก่อนแก้ `full-frame-preparation-start-20261014` ผู้ใช้ระบุว่าชิ้นงานในภาพ Live มีจุดเสีย จึง **ไม่ใช้เป็น GOOD/reference สำหรับการเรียนรู้** และต้องตรวจผิว โลโก้ และขอบทั้งหมดที่กล้องมองเห็น ไม่ตัดโลโก้/ขอบออกเพื่อทำให้ตรวจผ่าน พื้นที่ด้านข้าง/ด้านหลังที่ไม่เห็นในภาพยังรับรองไม่ได้

### พบชุด GOOD เดิมแล้ว — แก้ความเข้าใจจาก Core store

- `05_train_new_object/01_yolov8_exports/Airbag_Toyota/20260929_153627_auto_split` มี **268ภาพเต็ม1280×720** และ labels268ไฟล์: class0=`Airbag`, class1=`Toyota` อย่างละ268กรอบ เป็นกรอบวัตถุและโลโก้ ไม่ใช่ defect annotations
- `dataset_meta.json` ระบุ Known-good Reference, review_status=reviewed, reviewed_at=2026-09-29T15:36:34, auto_filled=false จึงมีแหล่ง GOOD เดิมในระบบจริง ไม่ได้มีเพียง Core cropped images ไม่ลบทิ้ง/ขอถ่ายใหม่ทั้งหมดโดยไม่ตรวจชุดนี้
- ผู้ใช้ยืนยันในรอบนี้ว่า **ทั้ง268ภาพเป็นชิ้นเดียวที่ขยับหลายตำแหน่ง** ชื่อไฟล์อยู่23ก.ย.08:58:23–09:01:12; จึงนับเป็นกลุ่มชิ้นงานเดียว ห้ามใช้ split เดิม train215/valid40/test13 เป็นหลักฐานกลุ่มอิสระ Manifestเดิมเป็น automatic_class_aware_split ไม่ได้ยืนยัน physical part groups
- ยังไม่มี capture settings/EXIF camera provenance ที่เทียบกับ LifeCamรอบล่าสุดได้ และความละเอียด1280×720ต่างจาก1920×1080 ห้าม upscale ภาพเก่าแล้วอ้างว่าเป็น reference1080pจากกล้องเดียวกัน; การใช้ camera1280×720 เป็นอีก setup ต้องตรวจ FOV/pose/lighting จริงก่อน ไม่เปลี่ยน camera configuration ในรอบนี้
- Core storeปัจจุบัน393ภาพยังไม่มี confirmed label/group แต่ข้อเท็จจริงนี้ **ไม่ใช่ข้อสรุปว่าไม่มี reviewed GOOD dataset นอก Core** ยังไม่นำเข้าทั้ง268ภาพ/เขียน label ใหม่ เพราะต้องแก้ input setup และ group provenance ก่อน เพื่อไม่สร้างสำเนาที่ใช้ Trainไม่ได้

### ภาพจริง / Geometry feasibility ที่ลงมือทดสอบแล้ว

ใช้ `inspection.pipeline.align/match` เดิมบนภาพที่เก็บไว้ ไม่มีการ Train หรือบันทึก Recipeใหม่ เปลี่ยนเฉพาะ method/กรอบใน memoryสำหรับ diagnostic ใช้ locator limits จากสูตรเดิมโดยไม่ปรับตัวเลขเพื่อทำให้ผ่าน **ผล accepted ด้านล่างเป็น software gate ของ geometry เท่านั้น ไม่ใช่การตรวจ GOOD/NG และไม่ใช่เกณฑ์ที่ calibrateสำหรับ Full-frameแล้ว**

| ชุด/บริเวณ locator | Template accepted | ORB accepted | ขอบเขต |
|---|---:|---:|---|
| ภาพ1080pจริง: referenceภาพวางชิ้นงานบนโต๊ะ, locatorภายในชิ้นงาน | 0/2ภาพอื่น | 0/2ภาพอื่น | self-match1/1ทั้งสองวิธีไม่นับเป็นหลักฐานการจัดแนวข้ามภาพ |
| ชุด1280×720เดิม: กรอบภายใน Airbagจาก annotation | 3/267 | 1/267 | ตัดภาพreferenceเองออกจากจำนวนเปรียบเทียบ |
| ชุด1280×720เดิม: กรอบ Toyota logoจาก annotation | 0/267 | 2/267 | บริเวณสะท้อนแสงยังไม่เป็น anchorที่เสถียรในชุดที่ลอง |

ภาพ1080pมีทั้งช่วงถือชิ้นงานเอียง/ยกและวางบนโต๊ะ รูปทรงฉายในภาพ แสง และขนาดไม่คงที่ Templateเดิมทำ translationเท่านั้น; ORBใช้ partial affine (translation/rotation/uniform scale) ไม่ได้แก้ perspectiveหรือผิวโค้งทุกแบบ ผลยังไม่สนับสนุนการเลือกวิธีใดเป็น live-ready ไม่สรุปว่า algorithmทั้งหมดใช้ไม่ได้จาก reference/regionที่ลอง และไม่ลด min_score/min_matches เพื่อฝืนผ่าน

Referenceที่ใช้ diagnostic `core_compatibility_raw.png` มี global brightness120.09 แต่ ROIตัวอย่างบนผิวเพียง50.50; global sharpness85.67 และ ROI143.24 (Laplacian variance) แสดงว่าค่าคุณภาพภาพรวมมีส่วนของโต๊ะ/เสื้อ/มือ ไม่ใช่ค่าผิวชิ้นงานล้วน มีแสงสะท้อนบนโลโก้และจุดสีอ่อนบนผิว ซึ่งยังไม่กำหนด defect type/ขนาดเอง ภาพ Liveนี้ไม่ใช้เป็น normal reference ตามคำยืนยันล่าสุดของผู้ใช้

แนวทางที่ reuse Coreได้: เก็บ raw full-frameเดิม, เลือก locatorบนรายละเอียดของชิ้นงาน, กำหนดหลาย rectangular Inspection Areas หลัง alignment แยกผิว/โลโก้/ขอบและตรวจ source polygons ด้วยภาพจริง Coreรองรับสูงสุด16areas ยังไม่มี arbitrary part maskสำหรับ anomaly ROI; สี่เหลี่ยมครอบขอบโค้งอาจกินพื้นหลัง ต้องจัดฉาก/ตำแหน่งให้คงที่และตรวจ coverage ห้ามอ้างว่าลบ backgroundแล้วหรือมองเห็นครบทั้งชิ้นจากภาพด้านเดียว ROIตัวอย่างที่ใช้วิเคราะห์ไม่ใช่ขอบเขตสุดท้ายของงาน “ทั้งหมด”

### Model compatibility / ข้อมูลที่ยังขาด

PaDiM2และPatchCore2ตัวเดิมมี input_signatureของ cropped recipeเดียวกัน ไม่มีตัวใดตรงกับ full-frame reference/camera/locator/area/qualityที่ทดลองเชิงgeometry Pipelineเดิมตรวจ signatureก่อน predict จึงยังไม่เรียกโมเดลเหล่านี้กับROIใหม่ Workerเดิมทำ prepare→aligned ROI→resize224×224→normalization; การที่tensorขนาด224เหมือนกันไม่ได้ทำให้คนละROI/referenceเข้ากัน ต้องพิจารณาขนาดตำหนิหลังย่อROIและทดสอบ NGจริงก่อนรับรองพื้นที่กว้าง

ขั้นต่ำตาม software guardเดิมสำหรับการ Train: ≥20ภาพGOODไม่ซ้ำ, ≥5กลุ่มอิสระ, fit≥10/calibration≥4 และไม่ให้ reference groupอยู่calibration **ไม่ใช่จำนวนที่รับรอง accuracy** ขณะนี้268ภาพเดิมคือกลุ่มเดียวตามผู้ใช้ยืนยัน ห้ามสร้าง group IDsใหม่แยกตามไฟล์หรือ train/valid/testเพื่อผ่าน guard ต้องเพิ่มชิ้นงาน/กลุ่มอิสระที่ยืนยันGOODและมี setupตรงกัน พร้อมกันนั้นกัน GOOD/NGจากชิ้นอื่นสำหรับ independent validation และแยก defect types/ขนาด/ตำแหน่งจริง

### แก้ software blocker ที่พิสูจน์ได้

ตัวแบ่งกลุ่มเดิมสุ่มเลือกcalibrationก่อนแล้ว discard reference group ทำให้ชุด20ภาพ/5กลุ่มที่พอใช้งานเกิดcalibrationว่างเมื่อ referenceถูกสุ่มมาเป็นกลุ่มแรก พิสูจน์ด้วย regressionที่failในกรณี part_d ก่อนแก้ ย้ายการแบ่งกลุ่มเดิมมา `split_training_groups` ใน workerไฟล์เดิม และตัดreferenceออกจากรายชื่อที่เลือก **ก่อน** เลือกcalibration ไม่เพิ่ม training pipeline/โมเดล ไม่ลดข้อกำหนดจำนวนภาพ/กลุ่มและไม่ทำให้ชิ้นเดียวผ่าน

- Testsใหม่7กรณีผ่าน: referenceทุกกลุ่ม, fit/calibrationไม่ทับกัน, deterministic split, ภาพจำนวนมากของชิ้นเดียวถูกปฏิเสธ และกลุ่มที่มีภาพcalibrationไม่พอยังถูกปฏิเสธ
- Full regression: **336 Python tests +5 subtests ผ่าน** ไม่มีการ fit/model construction ใน testsใหม่ ไม่ถือว่า testsเป็น defect accuracy
- Browserจริงยังไม่ได้ตรวจในรอบนี้: ไม่มี callable browser/computer-use tool และ environmentโปรเจกต์ไม่มี Playwright/Selenium ไม่ติดตั้งเพิ่ม ไม่ใช้ HTTPแทนBrowser pass; manual checklistจากPhase4ยังค้าง
- ไม่ Train, ไม่เปลี่ยน image labels/Recipe/model weights/thresholds/Production, ไม่เปิดLiveโดยใช้inputผิดเงื่อนไข ไม่ถอดLegacyLive Source changeเฉพาะการแบ่งกลุ่มใน workerและtests

### หลักฐานและขั้นตอนถัดไป

`work/full-frame-preparation-20261014/` เก็บ `legacy-data-provenance.json` (268 paths/hashes/annotation provenanceและคำยืนยันชิ้นเดียว), `geometry-diagnostics.json`, `legacy-reference-selection.json`, `legacy-geometry-diagnostics.json`, `legacy-logo-diagnostics.json`; ภาพต้นฉบับและDatasetยังอยู่ที่เดิม ตรวจ540ไฟล์ของToyotaก่อน–หลังgeometry testไม่เปลี่ยน

1. ใช้ภาพ/กรอบเดิมช่วยกำหนดพื้นที่ชิ้นงานและขอบเขตความแปรผัน โดยคง268ภาพในกลุ่มเดียว และเก็บชิ้นที่ผู้ใช้แจ้งว่ามีตำหนิแยกจากGOOD ไม่กำหนดlabelทุกไฟล์แทนผู้ใช้
2. จัดภาพชิ้นงานGOODที่ยืนยันแล้วใน setupเดียวกับกล้องที่จะใช้จริง: ชิ้นงานวางมั่นคง, มือออกก่อนตรวจ, ระยะ/มุม/แสงคงที่ ไม่ยกชิ้นงานเปลี่ยน perspectiveระหว่างรอบ เก็บภาพ full-frameต้นฉบับและsettingsพร้อม part/group IDs
3. พิสูจน์ Templateสำหรับtranslationในตำแหน่งควบคุม หรือ ORBสำหรับin-plane rotation/scaleด้วยภาพจริงก่อนเลือกตัวใด บันทึก ROIผิว/โลโก้/ขอบทุกพื้นที่ที่ต้องตรวจและsource polygons; ถ้ายังไม่ผ่าน ให้หาสาเหตุ/ประเมิน reuse legacy locatorตามหลักฐาน ไม่ต่อ YOLOเข้าCoreแล้วอ้างว่าได้alignmentครบ
คำสั่ง Terminal Candidate ที่เคยใช้ในขั้นตอนนี้ถูกถอดเมื่อ 15 ตุลาคม 2026 ตามคำสั่งผู้ใช้; เก็บผลเดิมเป็นประวัติ ทดสอบ Candidate ด้วยภาพผ่าน `/inspection` ต่อได้

---

## Hardware Verification — 14 ตุลาคม 2026

**รับภาพจากกล้องจริงแล้ว** ด้วย `Runtime.open_capture` → `OpenCVCamera` → `FramePump` เดิม และทดสอบภาพจริงกับ `Pipeline`/Decision/Evidence เดิม ไม่มี Camera Application/CLI/Pipeline ใหม่ ไม่มี Source Code change, Train, Activate, เปลี่ยน threshold หรือสร้าง Recipe/ground truth ไม่อ้างว่าตรวจ Defect Accuracy ผ่าน

### อุปกรณ์ / ownership

- รอบก่อนวันที่9พบ LifeCam ที่ index0 แต่ก่อนทดสอบมี Terminal เริ่มถือ lock PID23068 จึงไม่แย่งกล้อง และผู้ใช้ปิดด้วย q/Esc ตามคำขอ รอบต่อวันที่14 Windows ไม่พบกล้องจริง จนผู้ใช้ต่อใหม่ จึง enumerate อีกครั้งแทนใช้ index เดิม
- ปัจจุบัน: **Microsoft® LifeCam Studio(TM), DirectShow index1**; USB `VID_045E&PID_0772&MI_00`, instance `6&3E88034&0&0000`; index0=Sharing-Media, index2=OBS Virtual Camera ชื่อ/indexอ่านจาก DirectShow COM enumeration ไม่ใช่การเดาหรือ probe ทุกกล้อง
- Driver ที่ตรวจพบ: Microsoft `5.20.102.0` / `oem70.inf`; backend ที่เปิดจริง=`DSHOW` ไม่มี Basler ใน pylon enumeration
- ก่อนเปิดไม่มี Terminal camera process/OS lease holder; Web8000รายงาน legacy worker และ Coreหยุดอยู่ ใช้ project lease ทุกช่วงรับภาพและ enumerate stream modes ไม่ Kill/Restart โปรแกรมใด
- Mode list ที่ driver ประกาศผ่าน `IAMStreamConfig` วันที่14 (YUY2): 160×120, 176×144, 320×240, 352×288, 424×240, 640×360, 640×480, 800×448, 1280×720, 1920×1080 ที่ nominal30FPS; 800×600 และ960×544 ที่ nominal20FPS **ไม่ได้ทดสอบรับภาพทุก mode**

### ผลรับภาพจริง

| รายการ | Driver default | Full-frame session |
|---|---:|---:|
| Actual resolution | 640×480 | 1920×1080 |
| ช่วงวัด | 10.032s | 25.094s |
| Frames ที่ consumer อ่าน | 298 | 732 |
| Producer FPS จาก sequence/host timestamps | 29.98 | 29.95 |
| Consumer FPS | 29.98 | 29.35 |
| Sequence ที่ consumer ข้าม | 0 | 15 |
| ภาพดำล้วน / pixels ซ้ำติดกัน | 0 / 0 | 0 / 0 |
| Brightness mean (gray0–255) | 98.08 | 103.19 |
| Sharpness mean (Laplacian variance) | 779.00 | 134.29 |
| Frame age ตอน consumer รับ P95 / max | 0 / 16ms | 16 / 32ms |
| Release | สำเร็จ | สำเร็จ |

FramePump เป็น latest-slot จึงข้ามเฟรมได้เมื่อ consumer วัด Laplacian/hash/เขียนภาพช้ากว่า producer; 15 sequence ที่ข้าม **ไม่ใช่ตัววัด USB/sensor frame loss** ไม่มี sensor frame counter จึงยังรับรองว่าไม่มี hardware drop ไม่ได้ Host `time.monotonic()` ใช้ `GetTickCount64()` resolution15.625ms; age0ms ไม่ได้แปลว่า latency เป็นศูนย์ และตัวเลขทั้งหมดไม่ใช่ sensor-to-display latency

ค่าจาก driver ก่อน/หลัง: exposure **−6 driver units**, gain **−1 (unsupported/sentinel จึงไม่สรุปเป็นค่า gain จริง)**, focus **7 driver units**, autofocus **1/on**; OpenCV FPS property=−1 จึงใช้ timestamp วัด FPS แทน ไม่มีการ set exposure/gain/focus/autofocus เปลี่ยนเพียง requested width/height ของ session จากโหมดที่ประกาศไว้ หลังปิด1080p เปิดใหม่โดยไม่กำหนดขนาดได้640×480และค่าควบคุมเดิม

เห็นชิ้นงานบนโต๊ะจากภาพจริง ไม่ใช่ black placeholder; ภาพเต็มมีแสงสะท้อนบริเวณโลโก้และมือ/ฉากหลัง ต้องกำหนด lighting/ROI/focus ตามตำหนิที่ต้องจับก่อนรับรองความคม ค่า sharpness ต่าง resolution เปรียบเทียบตรง ๆ ไม่ได้ และไม่มี threshold ที่ calibrate แล้วจึงไม่ตัดสินว่าภาพชัดพอสำหรับ defect ขนาดใด

### Core integration กับภาพจริง / จุดที่หยุดอย่างถูกต้อง

- รับ1080pต่ออีกประมาณ10วินาที → `Pipeline.run` กับ Recipeเดิม `recipe_00edecc594b149358ae66695d927730d`: **299เฟรมได้ UNABLE_TO_INSPECT** ที่ `alignment` เพราะ full frameไม่ตรง reference256×256 ไม่มี resize/crop เพื่อบังคับผ่าน
- Quality metrics ถูกคำนวณจากภาพจริง; Recipeนี้มี brightness limits1–254 และ **min_blur=0 / blur gateปิด** คงค่าเดิมไว้ ไม่ใช้ผลผ่าน brightness เป็นหลักฐานว่าคุณภาพพร้อมโรงงาน
- Pipeline latency mean18.861ms / P95 27.692ms / max47.092ms; host frame ageหลัง Pipeline mean19.204ms / P95 32ms / max47ms เป็น **เส้นทาง quality→alignment rejection→decision** ยังไม่มี anomaly inference จึงไม่ใช่ throughput ของระบบตรวจตำหนิครบชุด
- ทดสอบภาพจริงเดียวกันกับ Recipeทั้งหมด7รายการ ได้ UNABLE ที่alignmentทั้งหมด; referenceทุกสูตรเป็น256×256, locator=`fixed`, camera indexในสูตรเดิม=0 ซึ่งปัจจุบันเป็น Sharing-Media `Runtime.candidate_pipeline` ปฏิเสธ fixed สำหรับ place/remove ก่อนเปิดกล้อง ไม่แก้สูตรเดิมให้ใช้ index1 เงียบ ๆ
- เก็บ Evidence ใหม่ `event_cdd18ccf676d4a41a3939ff5e4a062ad`, source=`test`, complete=false, verification_kind=`real_camera_incompatible_recipe_check` พร้อม Recipe hash, required area/settings, failure reason, full raw imageและ requested/actual capture metadata; `camera_requested` ใน Recipeยังเป็นindex0 ส่วน `capture.requested/actual` ของ diagnosticเป็นindex1 แสดงความไม่เข้ากันตามจริง **ไม่ใช่ Candidate Live run**
- Image `image_28c0d8c93f241d38c5c94bc559b1a2302088167e8ee14ef9da4cc864f100dceb` เป็น1920×1080, labelUNKNOWN, ไม่ยืนยัน label/group; Evidence อ่านได้จาก APIเดิมและ raw image HTTP200 ส่วน aligned preview404ตาม contract เพราะยังไม่มี location evidence ไม่สร้าง ROI/heatmapปลอม
- **ยังไม่ทดสอบผ่าน**: alignment/ROI success, anomaly score, place/remove cycle, NG recall, physical unplug/reconnect ระหว่างกำลังอ่าน และ soak testระยะยาว; ไม่รัน Terminal Candidateด้วย cropped/fixed recipeที่ไม่เข้ากัน ไม่เรียกผลจาก299เฟรมนี้ว่า field validation สำเร็จ

### หลักฐาน / ข้อมูลเดิม / งานถัดไป

หลักฐานอยู่ `work/hardware-verification-20261014/`: `device-modes.json`, `summary.json`, `*_frames.json`, `core-compatibility.json`, `core-frames.json`, ภาพ `driver_default_first/last.png`, `full_1920x1080_first/last.png`, `core_compatibility_raw.png` เป็นภาพจริงไม่แก้ไข ส่วน Evidence/Image ใหม่อยู่ Core storeตาม IDด้านบน

ตรวจ hash ไฟล์เดิม **876ไฟล์** ใน inspection_data/jobs/models และ project_config ก่อน–หลัง Core test: ไม่เปลี่ยน/ไม่หาย เพิ่มเฉพาะภาพUNKNOWN+Evidenceจริงและ diagnostic artifacts ตรวจ Productionยังเป็นNone และ OS leaseว่างหลังปิดกล้อง **31 regression tests** ของ capture/Terminal/Core/ownership ผ่านหลังทดสอบ hardware ไม่มี source/dependency/config diff; บันทึกรายงานใน Git แยกจากหลักฐานภาพที่อยู่ในพื้นที่ข้อมูล

ก่อนทดสอบ Candidateครั้งแรก ต้องเลือกภาพเต็มที่ผู้ตรวจยืนยันGOOD พร้อม capture groupจริง, ออกแบบ locator/ROIให้ตรงชิ้นงานและการเลื่อน/หมุน, calibrate qualityจากภาพจริง, ตรวจ model/reference/input compatibility และมี NGจริงพร้อม defect type/ขนาด ไม่มีการรับรองจุดสี/รอยใดในภาพครั้งนี้ว่าเป็นตำหนิเอง เมื่อสูตรเข้ากันแล้วจึงใช้ Terminalไฟล์เดิมด้วย explicit Candidateและ indexที่ enumerateได้ ณ เวลานั้น; ห้ามยึด index0/1 ถาวรจากรายงานนี้

---

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
คำสั่ง Terminal Candidate ที่เคยใช้ในขั้นตอนนี้ถูกถอดเมื่อ 15 ตุลาคม 2026 ตามคำสั่งผู้ใช้; เก็บผลเดิมเป็นประวัติ ทดสอบ Candidate ด้วยภาพผ่าน `/inspection` ต่อได้
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
คำสั่ง Terminal Candidate ที่เคยใช้ในขั้นตอนนี้ถูกถอดเมื่อ 15 ตุลาคม 2026 ตามคำสั่งผู้ใช้; เก็บผลเดิมเป็นประวัติ ทดสอบ Candidate ด้วยภาพผ่าน `/inspection` ต่อได้
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
