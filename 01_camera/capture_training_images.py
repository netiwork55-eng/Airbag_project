"""
เครื่องมือถ่ายรูปจาก webcam เพื่อเก็บข้อมูลต้นฉบับสำหรับสร้าง dataset

ไฟล์นี้ทำหน้าที่ควบคุม exposure, brightness และ focus ของกล้อง แล้วบันทึก frame
ลง SAVE_DIR เท่านั้น ยังไม่สร้าง label และยังไม่นำรูปเข้า dataset train อัตโนมัติ
ผู้รับช่วงควรตรวจ CAMERA_INDEX และ SAVE_DIR ก่อนใช้งานกับเครื่องใหม่

ปุ่มควบคุม:
    SPACE or s  Save current frame
    a           Auto light mode
    m           Manual light mode
    [           Decrease exposure / reduce light
    ]           Increase exposure / add light
    -           Decrease brightness
    =           Increase brightness
    u           Enable autofocus
    l           Lock current/manual focus
    r           Scan best focus
    f           Decrease focus
    g           Increase focus
    q or ESC    ออกจากโปรแกรม
"""

from datetime import datetime
from pathlib import Path

import cv2


CAMERA_INDEX = 1
WINDOW_NAME = "Webcam Photo Capture"
SAVE_DIR = Path.home() / "Downloads" / "Background"

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30

AUTO_LIGHT = True

EXPOSURE_VALUE = -8
BRIGHTNESS = 70
CONTRAST = 145
SATURATION = 80
GAIN = 0

WB_VALUE = 4500

# ถ้าถ่ายใกล้ 35 อาจชัด
# ถ้าถ่ายไกล ให้ลอง 80, 120, 160 หรือใช้ปุ่ม r สแกนหา
START_WITH_AUTOFOCUS = True
FOCUS_VALUE = 80

FOCUS_MIN = 0
FOCUS_MAX = 255
FOCUS_STEP = 5


def set_camera_property(cap, prop, value, name):
    """ตั้งค่ากล้องหนึ่งรายการและแจ้งเตือนเมื่อ driver ไม่รองรับ; ไม่หยุดโปรแกรม."""
    cap.set(prop, value)
    actual = cap.get(prop)
    print(f"{name}: set={value}, actual={actual}")
    return actual


def apply_auto_light(cap):
    """คืน exposure/brightness เป็นโหมดอัตโนมัติตามที่ OpenCV backend รองรับ."""
    print("\nAuto light mode ON")
    set_camera_property(cap, cv2.CAP_PROP_AUTO_EXPOSURE, 0.75, "Auto Exposure")
    set_camera_property(cap, cv2.CAP_PROP_AUTO_WB, 1, "Auto White Balance")
    set_camera_property(cap, cv2.CAP_PROP_GAIN, 0, "Gain")


def apply_manual_light(cap, exposure, brightness):
    """ปิด auto exposure แล้วตั้งแสงแบบ manual เพื่อให้ภาพชุดเดียวกันสม่ำเสมอ."""
    print("\nManual light mode ON")
    set_camera_property(cap, cv2.CAP_PROP_AUTO_EXPOSURE, 0.25, "Auto Exposure")
    set_camera_property(cap, cv2.CAP_PROP_EXPOSURE, exposure, "Exposure")

    set_camera_property(cap, cv2.CAP_PROP_AUTO_WB, 0, "Auto White Balance")
    set_camera_property(cap, cv2.CAP_PROP_WB_TEMPERATURE, WB_VALUE, "White Balance")

    set_camera_property(cap, cv2.CAP_PROP_BRIGHTNESS, brightness, "Brightness")
    set_camera_property(cap, cv2.CAP_PROP_CONTRAST, CONTRAST, "Contrast")
    set_camera_property(cap, cv2.CAP_PROP_SATURATION, SATURATION, "Saturation")
    set_camera_property(cap, cv2.CAP_PROP_GAIN, GAIN, "Gain")


def enable_autofocus(cap):
    """เปิด autofocus หากกล้องรองรับ โดยปล่อยให้ผู้ใช้ใช้ manual focus ต่อได้."""
    print("\nAutofocus ON")
    set_camera_property(cap, cv2.CAP_PROP_AUTOFOCUS, 1, "Autofocus")


def apply_manual_focus(cap, focus_value):
    """ปิด autofocus และล็อกค่า focus ที่ผู้ใช้เลือกสำหรับการเก็บภาพต่อเนื่อง."""
    print("\nManual focus mode ON")
    set_camera_property(cap, cv2.CAP_PROP_AUTOFOCUS, 0, "Autofocus")
    actual = set_camera_property(cap, cv2.CAP_PROP_FOCUS, focus_value, "Focus")
    return actual


def get_focus_score(frame):
    """
    วัดความคมของภาพจากกลางภาพ
    ค่ายิ่งสูง = ยิ่งคม
    """
    h, w = frame.shape[:2]

    x1 = int(w * 0.25)
    x2 = int(w * 0.75)
    y1 = int(h * 0.25)
    y2 = int(h * 0.75)

    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    return cv2.Laplacian(gray, cv2.CV_64F).var()


def read_stable_frame(cap, warmup=5):
    """ทิ้ง frame ช่วงปรับแสง/focus แล้วคืน frame ล่าสุดที่อ่านสำเร็จ."""
    frame = None
    for _ in range(warmup):
        ok, frame = cap.read()
        if not ok:
            return None
    return frame


def scan_best_focus(cap):
    """
    สแกนค่า focus หลายค่า แล้วเลือกค่าที่ภาพคมสุด
    เหมาะมากตอนวางชิ้นงานระยะใหม่
    """
    print("\nScanning best focus...")

    set_camera_property(cap, cv2.CAP_PROP_AUTOFOCUS, 0, "Autofocus")

    best_focus = FOCUS_VALUE
    best_score = -1

    for focus in range(FOCUS_MIN, FOCUS_MAX + 1, FOCUS_STEP):
        cap.set(cv2.CAP_PROP_FOCUS, focus)

        frame = read_stable_frame(cap, warmup=6)
        if frame is None:
            continue

        score = get_focus_score(frame)

        if score > best_score:
            best_score = score
            best_focus = focus

        print(f"Focus {focus:3d} | sharpness score: {score:.2f}")

    actual = apply_manual_focus(cap, best_focus)

    print(f"\nBest focus = {best_focus}, actual = {actual}, score = {best_score:.2f}")
    return int(best_focus), best_score


def main() -> None:
    """เปิดกล้อง รับปุ่มควบคุม และบันทึก frame ที่ผู้ใช้เลือกลง SAVE_DIR."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam {CAMERA_INDEX}. "
            "Check that the camera is connected and not used by another app."
        )

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    set_camera_property(cap, cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH, "Width")
    set_camera_property(cap, cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT, "Height")
    set_camera_property(cap, cv2.CAP_PROP_FPS, FPS, "FPS")

    current_exposure = EXPOSURE_VALUE
    current_brightness = BRIGHTNESS
    current_focus = FOCUS_VALUE
    focus_score = 0
    light_mode = "AUTO"
    focus_mode = "AUTO" if START_WITH_AUTOFOCUS else "MANUAL"

    if START_WITH_AUTOFOCUS:
        enable_autofocus(cap)
    else:
        apply_manual_focus(cap, current_focus)

    if AUTO_LIGHT:
        apply_auto_light(cap)
        light_mode = "AUTO"
    else:
        apply_manual_light(cap, current_exposure, current_brightness)
        light_mode = "MANUAL"

    print(f"\nSaving photos to: {SAVE_DIR}")
    print("SPACE/s = save")
    print("a = auto light, m = manual light")
    print("[ / ] = exposure down/up")
    print("- / = = brightness down/up")
    print("u = autofocus")
    print("l = lock/manual focus")
    print("r = scan best focus")
    print("f / g = focus down/up")
    print("q or ESC = quit")

    for _ in range(30):
        cap.read()

    saved_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: failed to read frame from webcam.")
                break

            focus_score = get_focus_score(frame)

            preview = frame.copy()

            cv2.putText(
                preview,
                "SPACE/s: save | a:auto light | m:manual light | [/]: exposure | -/=: brightness",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                preview,
                "u: autofocus | l: lock focus | r: scan focus | f/g: focus | q/ESC: quit",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                preview,
                f"Saved:{saved_count} | Light:{light_mode} | FocusMode:{focus_mode} | Focus:{current_focus} | Sharp:{focus_score:.0f}",
                (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if key == ord("a"):
                apply_auto_light(cap)
                light_mode = "AUTO"

            elif key == ord("m"):
                apply_manual_light(cap, current_exposure, current_brightness)
                light_mode = "MANUAL"

            elif key == ord("["):
                current_exposure -= 1
                apply_manual_light(cap, current_exposure, current_brightness)
                light_mode = "MANUAL"
                print(f"Exposure decreased: {current_exposure}")

            elif key == ord("]"):
                current_exposure += 1
                apply_manual_light(cap, current_exposure, current_brightness)
                light_mode = "MANUAL"
                print(f"Exposure increased: {current_exposure}")

            elif key == ord("-"):
                current_brightness = max(0, current_brightness - 5)
                apply_manual_light(cap, current_exposure, current_brightness)
                light_mode = "MANUAL"
                print(f"Brightness decreased: {current_brightness}")

            elif key == ord("="):
                current_brightness = min(255, current_brightness + 5)
                apply_manual_light(cap, current_exposure, current_brightness)
                light_mode = "MANUAL"
                print(f"Brightness increased: {current_brightness}")

            elif key == ord("u"):
                enable_autofocus(cap)
                focus_mode = "AUTO"
                print("Autofocus enabled. Wait 1-2 seconds for camera to adjust.")

            elif key == ord("l"):
                # ล็อก manual ด้วยค่า current_focus ที่ใช้อยู่
                apply_manual_focus(cap, current_focus)
                focus_mode = "MANUAL"
                print(f"Focus locked at: {current_focus}")

            elif key == ord("r"):
                current_focus, best_score = scan_best_focus(cap)
                focus_mode = "MANUAL"
                print(f"Best focus locked: {current_focus}, score: {best_score:.2f}")

            elif key == ord("f"):
                current_focus = max(FOCUS_MIN, current_focus - FOCUS_STEP)
                apply_manual_focus(cap, current_focus)
                focus_mode = "MANUAL"
                print(f"Focus decreased: {current_focus}")

            elif key == ord("g"):
                current_focus = min(FOCUS_MAX, current_focus + FOCUS_STEP)
                apply_manual_focus(cap, current_focus)
                focus_mode = "MANUAL"
                print(f"Focus increased: {current_focus}")

            elif key in (ord("s"), 32):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                output_path = SAVE_DIR / f"airbag_mark_{timestamp}.jpg"

                success = cv2.imwrite(str(output_path), frame)

                if success:
                    saved_count += 1
                    print(f"Saved {saved_count}: {output_path}")
                else:
                    print(f"Failed to save: {output_path}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Camera closed. Total saved: {saved_count}")


if __name__ == "__main__":
    main()
