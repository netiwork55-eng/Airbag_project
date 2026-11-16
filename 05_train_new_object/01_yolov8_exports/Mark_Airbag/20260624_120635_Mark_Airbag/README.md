# Mark_Airbag_V2 YOLOv8-ready dataset

This folder was converted from `Mark_Airbag_V2.zip` to Ultralytics YOLOv8/YOLO11 format.

## Summary

- Total images: 128
- Train images: 102
- Val images: 26
- Image size: 1280x720
- Total labels: 128
- Total boxes: 731

## Classes

Original `obj.names`:
- 0: รอยเปื้อน
- 1: Airbag

YOLOv8 `data.yaml` uses English-safe names:
- 0: Mark
- 1: Airbag

Label indexes were not changed.

## Structure

```text
Mark_Airbag_V2_YOLOv8_ready/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

## Train command

Run from inside this folder:

```bash
yolo detect train model=yolov8n.pt data=data.yaml imgsz=640 epochs=100 batch=8 name=mark_airbag_v2
```

Or with an absolute path:

```bash
yolo detect train model=yolov8n.pt data="D:/Users/is00001/Desktop/YOLO/Airbag_Project/05_train_new_object/01_yolov8_exports/Mark_Airbag_V2/data.yaml" imgsz=640 epochs=100 batch=8 name=mark_airbag_v2
```

## QA

- Missing images/labels: 0
- Bad YOLO label rows: 0
- Train boxes: 574
- Val boxes: 157
- Train class counts: {0: 473, 1: 101}
- Val class counts: {0: 131, 1: 26}
