# Project Explanation – Lab 1: YOLOv8 Object Detection & Tracking

## Overview

This document provides a detailed explanation of the concepts, design decisions, and
implementation details behind Lab 1 for the **CMPG 313 Artificial Intelligence** practical
assessment.

---

## Background: What is Object Detection?

Object detection is a computer vision task that answers two questions simultaneously:

1. **What** objects are present in an image or video frame?
2. **Where** are those objects located (bounding boxes)?

Classical approaches relied on hand-crafted features (e.g., Histogram of Oriented Gradients)
combined with sliding-window classifiers. Modern deep-learning models such as the **YOLO**
(You Only Look Once) family solve this in a single forward pass through a convolutional neural
network, achieving near-real-time performance.

---

## YOLOv8 Architecture

YOLOv8, released by [Ultralytics](https://github.com/ultralytics/ultralytics) in 2023, is the
latest generation of the YOLO family. Key design choices include:

| Component | Description |
|-----------|-------------|
| **Backbone** | CSP (Cross-Stage Partial) network for feature extraction |
| **Neck** | PANet (Path Aggregation Network) for multi-scale feature fusion |
| **Head** | Anchor-free detection head that predicts bounding boxes, class probabilities, and (optionally) masks |
| **Loss** | Distribution Focal Loss (DFL) + Binary Cross-Entropy |

The pretrained `yolov8n.pt` ("nano") weights used in this lab were trained on the **COCO**
dataset, which covers 80 object categories including people, vehicles, animals, and everyday
items.

---

## Implementation Details

### 1. Image Detection (`detect_image`)

```python
results = model.predict(source=IMAGE_PATH, save=False, conf=0.25)
annotated = results[0].plot()
cv2.imwrite(IMAGE_OUTPUT, annotated)
```

- `conf=0.25` filters out detections with a confidence score below 25 %.
- `results[0].plot()` returns a NumPy array with bounding boxes and labels rendered directly
  onto the image.
- The annotated frame is written to disk using OpenCV's `imwrite`.

### 2. Video Tracking (`track_video`)

```python
results = model.track(source=frame, persist=True, conf=0.25, verbose=False)
```

- `model.track()` extends detection with the **ByteTrack** algorithm, assigning consistent
  integer IDs to each object across consecutive frames.
- `persist=True` keeps the tracker state alive between calls so IDs are maintained throughout
  the entire video.
- Frames are decoded with `cv2.VideoCapture`, annotated, and re-encoded with
  `cv2.VideoWriter`.

### 3. Real-Time Webcam Detection (`detect_webcam`)

```python
cap = cv2.VideoCapture(0)   # device index 0 = default webcam
```

- Frames are captured in a loop, passed through the model, and displayed with
  `cv2.imshow` at near-real-time speed.
- The session ends when the user presses **q**.

---

## Confidence Threshold

A confidence threshold of **0.25** (25 %) was chosen as a balance between:

- **Recall** – detecting as many true objects as possible.
- **Precision** – avoiding too many false positives.

Lowering this value produces more (potentially noisy) detections; raising it filters down to
only high-certainty results.

---

## Limitations & Future Work

| Limitation | Potential Improvement |
|------------|----------------------|
| Uses generic COCO weights | Fine-tune on a custom dataset for domain-specific objects |
| Nano model trades accuracy for speed | Experiment with `yolov8s`, `yolov8m`, or `yolov8l` |
| No GPU acceleration configured | Add `device="cuda"` for NVIDIA GPU inference |
| Webcam demo not recorded | Integrate `cv2.VideoWriter` to save the webcam session |

---

## References

- Ultralytics YOLOv8 Documentation: <https://docs.ultralytics.com>
- OpenCV Documentation: <https://docs.opencv.org>
- COCO Dataset: <https://cocodataset.org>
- Redmon, J. et al. (2016). *You Only Look Once: Unified, Real-Time Object Detection*. CVPR.
