# 💡 LaserLineSkin — Laser Triangulation for Skin Surface Analysis

LaserLineSkin is a **deterministic image processing application** designed to detect and analyze **laser line deformation on skin-like surfaces** using **laser triangulation principles**.

The application is implemented with **Python, OpenCV, and Streamlit**, and focuses on:
- subpixel laser line extraction,
- robustness to textured surfaces (skin),
- local anomaly detection (folds, depressions),
- and quantitative deformation measurement.

> ⚠️ This project is **algorithmic only (no AI / no deep learning)** and is intended for **educational, research, and prototyping purposes**.

---

## 📌 Key Features

- ✅ Laser line enhancement using **Difference of Gaussians (DoG)**
- ✅ **Subpixel laser center extraction** (centroid-based)
- ✅ Robust handling of **skin-like textured surfaces**
- ✅ Local baseline estimation to highlight **surface anomalies**
- ✅ Interactive **Streamlit GUI**
- ✅ Export of results (CSV, processed images, overlays)
- ✅ Adjustable parameters (smoothing, confidence, baseline window)

---

## 🧠 Method Overview

The processing pipeline follows these steps:

1. **Preprocessing & Ridge Enhancement**
   - CLAHE for local contrast normalization
   - Difference of Gaussians to enhance the laser ridge
   - Horizontal smoothing to reduce speckle noise

2. **Subpixel Laser Line Extraction**
   - Column-wise centroid estimation
   - Confidence score based on laser energy (“mass”)
   - Rejection of unreliable columns only (no global outlier suppression)

3. **Baseline Estimation**
   - Very smooth local baseline (Savitzky–Golay)
   - Emphasizes local surface deformations

4. **Metrics Computation**
   - Peak deformation
   - Half-maximum width
   - Absolute deformation area

---

## 🖥️ Requirements

### ✔️ System
- Windows 10 / 11
- Python **3.9 or newer** recommended

---

## 📦 Installation (Windows)

### 1️⃣ Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate
