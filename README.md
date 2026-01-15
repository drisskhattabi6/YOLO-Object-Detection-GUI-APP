# YOLO Object Detection -- GUI

A focused, user-friendly **desktop application** (Tkinter) for running inference with YOLO `.pt` models on images, videos, or a webcam. This README covers only the desktop app and how you can use your own YOLO model for testing.

---

## 📌 Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Installation](#installation)
* [Usage](#usage)
* [Using Your Own YOLO Model](#using-your-own-yolo-model)
* [Project Structure](#project-structure)
* [Troubleshooting](#troubleshooting)
* [License & Acknowledgments](#license--acknowledgments)

---

# Overview

This is a **desktop application** for fast, visual testing of YOLO models. It’s designed for local use on your machine (images, videos, webcam) and makes it easy to switch models, tweak thresholds, and save annotated outputs — no training steps, no cloud required.

![Desktop App](imgs/img1.png)

---

# Features

* Dark-themed Tkinter GUI optimized for desktop
* Load images, videos, or use webcam for real-time detection
* Dynamic model loading: swap `.pt` models at runtime
* Auto-detects class names and number of classes from the model
* Adjustable confidence & IoU thresholds
* Save annotated images (in `runs/detect/predict*` by default)
* Simple, clear visual results: counts, per-class summaries, confidence bars

---

# Installation

### Prerequisites

* Python 3.9+
* pip
* (optional) Webcam for real-time testing

### Clone repo

```bash
git clone https://github.com/drisskhattabi6/YOLO-Object-Detection.git
cd YOLO-Object-Detection
```

### Create & activate virtual environment

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` in your terminal when it’s active.

### Install dependencies

```bash
pip install -r requirements.txt
```

**Main dependencies**

* `ultralytics` — run YOLO inference from `.pt` models
* `opencv-python` — image/video processing
* `pillow` — image handling for GUI
* `numpy` — numerical operations

---

# Usage

### Run the desktop app

Ensure the virtual environment is active, then:

```bash
python app.py
```

### Quick workflow

1. Launch app — it will try to auto-load `models/default_yolo.pt`.
2. Use **Change Model** to load any `.pt` model from disk.
3. Choose input: **Image**, **Video**, or **Webcam**.
4. Adjust **Confidence** and **IoU** sliders as needed.
5. Click **Detect Objects**.
6. Save annotated output with **Save Result**.

---

# Using Your Own YOLO Model

This app is designed for testing **any Ultralytics-style `.pt` YOLO model**. No code edits needed in the typical case — just place your model in `models/` and load it.

**Brief Answers**

1. **Different YOLO Version?**
   ✅ **YES** — if the model is in **Ultralytics `.pt` format** (works with modern Ultralytics outputs).

   * Supported (Ultralytics `.pt`): YOLOv5, YOLOv8, YOLOv9, YOLOv10, YOLOv11 — ✅
   * Not supported directly: YOLOv3 / YOLOv4 in Darknet `.weights`/`.cfg` format — ❌ (convert to Ultralytics `.pt` first)

2. **Model trained on the same dataset (Pascal VOC) but different run/size?**
   ✅ **YES** — different checkpoints, sizes (nano/s/m/l), or hyperparameter variants will work without changes.

3. **Model trained on a different dataset (COCO, custom)?**
   ✅ **YES — generally no code changes required.** The app reads class names from the model file.

**Notes**

* The `voc_classes` array in code is **documentation only** and not required for inference.
* The app will **auto-detect** number of classes and their names from the loaded model and display them.
* If you have a Darknet model, convert it into an Ultralytics-compatible `.pt` before using the app.

---

# Project Structure

Current structure (desktop-app focused):

```
YOLODetectorApp/
├── app.py
├── models/
│   └── default_yolo.pt
├── test_images/  
├── README.md
└── requirements.txt
```

---

# Troubleshooting

* **Model fails to load**

  * Ensure the file is an Ultralytics `.pt` model.
  * Try re-saving or exporting the model with Ultralytics if it was converted.

* **Too many false positives**

  * Increase confidence threshold (e.g., 0.4–0.6).

* **Missing detections**

  * Lower confidence threshold (e.g., 0.15–0.25) or check that class names match expected classes.

* **Webcam problems**

  * Check camera is not used by another app and that permissions are granted.

---

# License & Acknowledgments

* Provided for educational purposes.
* Thanks to **Ultralytics**, **OpenCV**, and **Pillow** for their libraries.
