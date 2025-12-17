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
```

### 2️⃣ Install required dependencies
```bash
pip install numpy pandas opencv-python scipy streamlit
```


💡 If you encounter issues with OpenCV:
```bash
pip install opencv-python-headless
```

### ▶️ Running the Application
From a terminal or command prompt:
```bash
python -m streamlit run "C:\Users\adham\Downloads\laser_gui_app.py"
```

Streamlit will automatically open the application in your web browser.

#### 📂 Input & Output
Streamlit will automatically open the application in your web browser.
Input

Image formats supported:

- .png

- .jpg

- .jpeg

The image should contain a visible laser line projected on a surface.


### Output

## 📈 Interactive visualization of:
- detected laser line (green)
- estimated baseline (orange)
## 📊 Quantitative metrics

## 📁 Export options:
- CSV profile (x, y, deviation, confidence)

- Processed images (ridge response)

- Overlay images

## ⚙️ Adjustable Parameters (GUI)

The Streamlit interface allows fine control over the laser line extraction and deformation analysis pipeline.

| Parameter | Description |
|---------|------------|
| `mm/px` | Conversion factor from pixels to millimeters. Must be set after calibration using a reference object. |
| `min_mass_ratio` | Confidence threshold for laser detection. Columns with insufficient laser energy are discarded. |
| `smooth_win` | Smoothing window size applied to the extracted laser centerline. Smaller values preserve sharp anomalies; larger values produce smoother profiles. |
| `baseline_win` | Window size used to compute the local baseline. Larger values highlight local surface anomalies more clearly. |
| Debug options | Enable visualization of the vertical search band and the laser confidence (`mass`) for debugging and analysis. |

---

## ⚠️ Limitations

- No intrinsic camera–laser calibration is included.
- The `mm/px` factor must be manually provided after experimental calibration.
- Images used for demonstration purposes are illustrative and not clinically validated.
- This software is **not intended for medical diagnosis or clinical use**.

---

## 🎓 Academic Context

This project was developed in the context of:
- computer vision,
- optical triangulation,
- surface profiling,
- medical imaging prototyping.

It demonstrates how **classical image processing techniques** (without AI or deep learning) can be applied to **complex, textured biological surfaces** such as skin.

---

## 📜 License

This project is released for **educational and research purposes** only.  
You are free to use, modify, and extend it with proper attribution.

---

## 👤 Author

**Adham Ahmed Salah Ali**  
**Hadi El Ayoubi**  

Engineering Student — Robotics & Image Processing  
Polytech Dijon
