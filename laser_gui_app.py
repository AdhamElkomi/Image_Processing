import io
import math
import numpy as np
import pandas as pd
import cv2
from scipy.signal import savgol_filter
import streamlit as st

# PDF (ReportLab)
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
)
from reportlab.lib.utils import ImageReader


# =========================================================
# ------------------------ UI STYLE ------------------------
# =========================================================
st.set_page_config(page_title="LaserLineSkin — Skin Aesthetics Report", page_icon="💡", layout="wide")
_CSS = """
<style>
    .stApp { background: #0f172a; color: #e5e7eb; font-family: 'Segoe UI', sans-serif; }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    h1, h2, h3 { color: #f8fafc; font-weight: 650; }
    hr { border: 1px solid #1e293b; margin: 1rem 0; }
    .metric-box { background: #111827; border: 1px solid #1f2937;
                  padding: 1rem; border-radius: 1rem; text-align: center; }
    .metric-title { font-size: 0.9rem; color: #94a3b8; margin-bottom: 0.4rem; }
    .metric-value { font-size: 1.5rem; font-weight: bold; color: #f1f5f9; }
    .small { font-size: 0.85rem; color: #94a3b8; }
    .accent { color: #60a5fa; }
    .stDownloadButton button { border-radius: 12px; background: #1e40af; color: white; }
    .stDownloadButton button:hover { background: #2563eb; color: white; }
    .footer { border-top: 1px solid #1e293b; margin-top: 2rem; padding-top: 0.8rem;
              text-align: center; font-size: 0.85rem; color:#94a3b8; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center'>💡 LaserLineSkin</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center; color:#60a5fa'>Analyse esthétique cutanée — imagerie + triangulation laser</h3>", unsafe_allow_html=True)
st.write(
    "**Objectif :** produire un **bilan esthétique** (relief, texture, homogénéité, rougeur) "
    "et générer un **rapport PDF clinique-style**.\n\n"
    "⚠️ **Ce rapport n’est pas un diagnostic médical** et ne remplace pas un avis dermatologique."
)

# =========================================================
# --------------------- CORE METHODS -----------------------
# =========================================================

def apply_preprocessing(gray: np.ndarray, method: str, ksize: int = 5, sigma: float = 1.2) -> np.ndarray:
    """Pré-traitement (réduction bruit) pour la peau: Gaussian / Median / Bilateral / NLM."""
    g = gray.copy()
    if method == "Gaussian":
        # sigma en pixels
        g = cv2.GaussianBlur(g, (0, 0), sigmaX=sigma, sigmaY=sigma)
    elif method == "Median":
        k = max(3, ksize | 1)
        g = cv2.medianBlur(g, k)
    elif method == "Bilateral":
        # d ~ taille voisinage, sigmaColor ~ radiométrique, sigmaSpace ~ spatial
        d = max(5, ksize | 1)
        g = cv2.bilateralFilter(g, d=d, sigmaColor=25, sigmaSpace=9)
    elif method == "NLM":
        # OpenCV fastNlMeansDenoising: h = force du filtrage
        g = cv2.fastNlMeansDenoising(g, None, h=10, templateWindowSize=7, searchWindowSize=21)
    return g


def enhance_laser_ridge(gray: np.ndarray) -> np.ndarray:
    """Renforcement d’une ligne laser fine (crête) : CLAHE + DoG + lissage directionnel."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    g = clahe.apply(gray)
    g = cv2.GaussianBlur(g, (0, 0), 1.0)

    sigma1, sigma2 = 1.2, 3.0
    g1 = cv2.GaussianBlur(g, (0, 0), sigma1)
    g2 = cv2.GaussianBlur(g, (0, 0), sigma2)
    dog = cv2.subtract(g1, g2).astype(np.float32)

    # Lissage horizontal pour renforcer une crête globalement horizontale
    dog = cv2.GaussianBlur(dog, (0, 0), 2.0, 0.0)
    return dog


def ridge_to_uint8(ridge: np.ndarray) -> np.ndarray:
    return cv2.normalize(ridge, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def extract_line_subpixel(
    ridge_img: np.ndarray,
    band_auto: bool = True,
    y0_hint=None,
    min_mass_ratio: float = 0.10,
    smooth_win: int = 41
):
    """
    Extraction subpixel: centroid sur une bande autour de y0 (projection verticale).
    Conserve anomalies; rejette seulement colonnes avec signal faible (mass).
    """
    h, w = ridge_img.shape

    if band_auto or y0_hint is None:
        proj = np.maximum(ridge_img, 0).sum(axis=1)
        y0 = int(np.argmax(proj))
    else:
        y0 = int(y0_hint)

    band = max(20, h // 25)
    y_min = max(0, y0 - band)
    y_max = min(h - 1, y0 + band)

    ys = np.full(w, np.nan, dtype=float)
    mass = np.zeros(w, dtype=np.float32)
    y_coords = np.arange(y_min, y_max + 1, dtype=np.float32)

    for x in range(w):
        col = ridge_img[y_min:y_max + 1, x].copy()
        col = np.maximum(col, 0)
        s = float(col.sum())
        mass[x] = s
        if s < 1e-6:
            continue
        ys[x] = float((y_coords * col).sum() / s)

    xs = np.arange(w)
    mmax = float(np.max(mass)) + 1e-6
    keep = (mass >= (min_mass_ratio * mmax)) & (~np.isnan(ys))
    xs_k, ys_k = xs[keep], ys[keep]

    if len(xs_k) < 30:
        valid = ~np.isnan(ys)
        return xs[valid], ys[valid], y0, band, mass

    ys_interp = np.interp(xs, xs_k, ys_k)

    if smooth_win % 2 == 0:
        smooth_win += 1
    smooth_win = min(smooth_win, (w // 3) * 2 + 1)

    if smooth_win >= 9:
        ys_smooth = savgol_filter(ys_interp, window_length=smooth_win, polyorder=2)
    else:
        ys_smooth = ys_interp

    return xs, ys_smooth, y0, band, mass


def local_baseline(y: np.ndarray, win: int = 301) -> np.ndarray:
    """Baseline locale lisse (composante lente) pour isoler anomalies locales."""
    if win % 2 == 0:
        win += 1
    win = min(win, (len(y) // 2) * 2 + 1)
    if win < 9:
        return y.copy()
    return savgol_filter(y, window_length=win, polyorder=2)


# ---------------- Texture / Color analytics (no AI) ----------------

def texture_metrics(gray: np.ndarray, roi=None):
    """
    Indicateurs texture (grain / rugosité):
    - variance locale moyenne (Laplacian energy)
    - gradient magnitude moyen
    - "micro-contrast" (std)
    """
    if roi is not None:
        x, y, w, h = roi
        g = gray[y:y+h, x:x+w]
    else:
        g = gray

    g = g.astype(np.float32)

    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    lap_energy = float(np.mean(lap**2))

    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx*gx + gy*gy)
    grad_mean = float(np.mean(grad_mag))

    micro_contrast = float(np.std(g))

    return {
        "lap_energy": lap_energy,
        "grad_mean": grad_mean,
        "micro_contrast": micro_contrast,
    }


def redness_index(bgr: np.ndarray, roi=None):
    """Indice de rougeur simple: R / (G + B + eps)."""
    if roi is not None:
        x, y, w, h = roi
        img = bgr[y:y+h, x:x+w]
    else:
        img = bgr

    b, g, r = cv2.split(img.astype(np.float32))
    eps = 1e-6
    ri = r / (g + b + eps)
    return float(np.mean(ri)), float(np.std(ri))


def tone_uniformity_lab(bgr: np.ndarray, roi=None):
    """Homogénéité du teint via LAB: std de L* et a* (rouge/vert)."""
    if roi is not None:
        x, y, w, h = roi
        img = bgr[y:y+h, x:x+w]
    else:
        img = bgr

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)
    return {
        "L_std": float(np.std(L)),
        "A_std": float(np.std(A)),
        "B_std": float(np.std(B)),
        "L_mean": float(np.mean(L)),
    }


def wrinkle_proxy(gray: np.ndarray, roi=None):
    """
    Proxy rides (sans IA):
    - accentuer structures fines (top-hat / black-hat) + seuil adaptatif
    - densité de pixels 'ride-like'
    """
    if roi is not None:
        x, y, w, h = roi
        g = gray[y:y+h, x:x+w]
    else:
        g = gray

    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    # Morphologie: black-hat (structures sombres fines)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blackhat = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, se)

    # Normalisation + seuillage adaptatif
    bh = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    bw = cv2.adaptiveThreshold(bh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 21, 2)

    density = float(np.mean(bw > 0))  # proportion pixels ride-like
    strength = float(np.mean(bh))     # intensité moyenne ride-like

    return {"wrinkle_density": density, "wrinkle_strength": strength, "blackhat_u8": bh, "wrinkle_mask": bw}


# ------------------- Scoring (0–100) -------------------

def clamp01(x): 
    return max(0.0, min(1.0, x))

def score_from_metrics(tex, color_tone, redness, wrinkle, relief):
    """
    Convertit des mesures en scores (0–100).
    IMPORTANT: ces mappings sont heuristiques → affichés comme 'indices esthétiques'.
    """
    # Texture: plus lap_energy/grad_mean/micro_contrast élevés => peau plus "rugueuse"
    tex_r = clamp01((tex["lap_energy"] / 8000.0))          # échelle heuristique
    tex_g = clamp01((tex["grad_mean"] / 25.0))
    tex_c = clamp01((tex["micro_contrast"] / 35.0))
    roughness = clamp01(0.45*tex_r + 0.35*tex_g + 0.20*tex_c)

    # Tone uniformity: std élevés => teint moins uniforme
    tone_var = clamp01((color_tone["L_std"] / 18.0) * 0.6 + (color_tone["A_std"] / 12.0) * 0.4)

    # Redness: moyenne de R/(G+B) - seuils heuristiques
    red_mean, red_std = redness
    red_level = clamp01((red_mean - 0.85) / 0.35)  # commence à monter au-delà de ~0.85

    # Wrinkles: densité + force
    w = clamp01((wrinkle["wrinkle_density"] / 0.12) * 0.6 + (wrinkle["wrinkle_strength"] / 40.0) * 0.4)

    # Relief: pic deviation mm (si dispo) sinon px
    relief_level = 0.0
    if relief is not None:
        relief_level = clamp01(relief["peak_abs_mm"] / 1.0)  # 1mm = élevé (heuristique)

    # Scores: 100 = meilleur
    smoothness_score = int(round(100 * (1 - roughness)))
    uniformity_score = int(round(100 * (1 - tone_var)))
    redness_score = int(round(100 * (1 - red_level)))
    wrinkle_score = int(round(100 * (1 - w)))
    relief_score = int(round(100 * (1 - relief_level)))

    global_score = int(round(
        0.25*smoothness_score +
        0.20*uniformity_score +
        0.15*redness_score +
        0.25*wrinkle_score +
        0.15*relief_score
    ))

    return {
        "Smoothness": smoothness_score,
        "Uniformity": uniformity_score,
        "Redness": redness_score,
        "Wrinkles": wrinkle_score,
        "Relief": relief_score,
        "Global": global_score
    }


# =========================================================
# ------------------------ PDF -----------------------------
# =========================================================

def cv_to_png_bytes(img_bgr_or_gray):
    """Convert OpenCV image to PNG bytes."""
    if img_bgr_or_gray is None:
        return None
    if len(img_bgr_or_gray.shape) == 2:
        img = img_bgr_or_gray
    else:
        img = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2RGB)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return None
    return buf.tobytes()

def make_pdf_report(
    patient_id: str,
    session_date: str,
    method_filter: str,
    params: dict,
    metrics_table: list,
    scores: dict,
    images: dict
) -> bytes:
    """
    Génère un PDF clinique-style.
    images: dict name->png bytes
    metrics_table: list of [Param, Value, Interpretation]
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            leftMargin=18*mm, rightMargin=18*mm,
                            topMargin=16*mm, bottomMargin=16*mm)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1", fontSize=16, leading=18, spaceAfter=10))
    styles.add(ParagraphStyle(name="H2", fontSize=12.5, leading=14, spaceAfter=8))
    styles.add(ParagraphStyle(name="Small", fontSize=9.5, leading=12, textColor=colors.grey))
    styles.add(ParagraphStyle(name="Body", fontSize=10.5, leading=13))

    story = []

    # Header
    story.append(Paragraph("Bilan esthétique cutané non invasif", styles["H1"]))
    story.append(Paragraph(
        f"<b>ID patient (anonymisé)</b> : {patient_id} &nbsp;&nbsp; "
        f"<b>Date</b> : {session_date}<br/>"
        f"<b>Méthode</b> : Imagerie optique + extraction subpixel (laser) + analyse texture/couleur<br/>"
        f"<b>Filtrage</b> : {method_filter}",
        styles["Body"]
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "⚠️ <b>Notice :</b> ce rapport fournit des <b>indices esthétiques</b> (non diagnostiques) "
        "et ne se substitue pas à un examen médical.",
        styles["Small"]
    ))
    story.append(Spacer(1, 10))

    # Images page
    story.append(Paragraph("Résumé visuel", styles["H2"]))
    img_rows = []
    for label in ["original", "preproc", "ridge", "overlay", "wrinkle_mask"]:
        if label in images and images[label] is not None:
            ir = ImageReader(io.BytesIO(images[label]))
            w, h = ir.getSize()
            target_w = 85*mm
            target_h = target_w * (h / max(w, 1))
            img_rows.append([RLImage(io.BytesIO(images[label]), width=target_w, height=target_h),
                             Paragraph(f"<b>{label}</b>", styles["Small"])])

    if img_rows:
        tbl = Table(img_rows, colWidths=[90*mm, 80*mm])
        tbl.setStyle(TableStyle([
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ]))
        story.append(tbl)
    story.append(Spacer(1, 12))

    # Scores
    story.append(Paragraph("Scores esthétiques (0–100)", styles["H2"]))
    score_data = [["Domaine", "Score", "Interprétation"]]
    def interpret_score(s):
        if s >= 85: return "Excellent"
        if s >= 70: return "Bon"
        if s >= 50: return "Moyen"
        return "À améliorer"
    for k in ["Global","Smoothness","Uniformity","Wrinkles","Redness","Relief"]:
        v = scores.get(k, 0)
        score_data.append([k, str(v), interpret_score(v)])

    score_tbl = Table(score_data, hAlign="LEFT", colWidths=[45*mm, 25*mm, 80*mm])
    score_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.lightgrey]),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(score_tbl)
    story.append(Spacer(1, 10))

    # Detailed metrics
    story.append(Paragraph("Mesures quantitatives", styles["H2"]))
    data = [["Paramètre", "Valeur", "Commentaire"]]
    data += metrics_table

    met_tbl = Table(data, hAlign="LEFT", colWidths=[55*mm, 35*mm, 60*mm])
    met_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0B1220")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.lightgrey]),
    ]))
    story.append(met_tbl)
    story.append(Spacer(1, 12))

    # Method & parameters
    story.append(Paragraph("Paramètres de traitement (traçabilité)", styles["H2"]))
    p_lines = []
    for kk, vv in params.items():
        p_lines.append(f"<b>{kk}</b> : {vv}")
    story.append(Paragraph("<br/>".join(p_lines), styles["Small"]))
    story.append(Spacer(1, 10))

    # Conclusion text (professional tone)
    story.append(Paragraph("Synthèse esthétique", styles["H2"]))
    g = scores.get("Global", 0)
    if g >= 85:
        synth = ("Les indices calculés indiquent une peau globalement en <b>excellent état esthétique</b>, "
                 "avec une texture homogène et une absence de signal marqué d’inflammation. "
                 "Le relief mesuré reste compatible avec un profil cutané régulier.")
    elif g >= 70:
        synth = ("Les indices calculés indiquent un état esthétique <b>globalement bon</b>. "
                 "Des variations modérées peuvent être observées sur la texture et/ou l’homogénéité du teint. "
                 "Une routine d’entretien ciblée peut améliorer la régularité perçue.")
    elif g >= 50:
        synth = ("Les indices calculés indiquent un état esthétique <b>moyen</b>, avec des marqueurs de rugosité, "
                 "d’hétérogénéité ou de structures fines (rides/relief) plus présents. "
                 "Une optimisation de l’hydratation, de la protection UV et une prise en charge esthétique peuvent être envisagées.")
    else:
        synth = ("Les indices calculés indiquent un état esthétique <b>à améliorer</b>, "
                 "avec des marqueurs plus marqués sur texture/relief et/ou rougeur. "
                 "Une évaluation dermatologique peut être utile si ces signes s’accompagnent d’inconfort, douleur ou lésions.")
    story.append(Paragraph(synth, styles["Body"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Recommandation : réaliser un suivi longitudinal (mêmes conditions d’éclairage, distance et calibration) "
        "pour comparer l’évolution des indices.",
        styles["Small"]
    ))

    doc.build(story)
    return buffer.getvalue()


# =========================================================
# ---------------------- SIDEBAR ---------------------------
# =========================================================
st.sidebar.header("⚙️ Paramètres")

patient_id = st.sidebar.text_input("ID patient (anonymisé)", value="PAT-0001")
session_date = st.sidebar.text_input("Date (ex: 2025-12-17)", value="2025-12-17")

# Global scale
mm_per_px = st.sidebar.number_input("Échelle relief (mm/px)", value=0.10, step=0.01, min_value=0.0)

# Filtering selection for skin image analytics
filter_method = st.sidebar.selectbox("Filtrage peau (pré-traitement)", ["Gaussian", "Median", "Bilateral", "NLM"])
filter_ksize = st.sidebar.slider("Taille voisinage (ksize)", 3, 21, 7, 2)
filter_sigma = st.sidebar.slider("Sigma (Gaussian)", 0.5, 5.0, 1.2, 0.1)

# Laser parameters
min_mass_ratio = st.sidebar.slider("Laser confidence (min_mass_ratio)", 0.01, 0.50, 0.10, 0.01)
smooth_win = st.sidebar.slider("Lissage ligne laser (smooth_win)", 9, 151, 41, 2)
baseline_win = st.sidebar.slider("Fenêtre baseline (baseline_win)", 51, 701, 301, 2)

# Anomaly controls (relief)
anomaly_ratio = st.sidebar.slider("Seuil anomalie relief (ratio du max)", 0.05, 0.80, 0.35, 0.01)
min_anomaly_width = st.sidebar.slider("Largeur min anomalie (px)", 1, 200, 10, 1)

st.sidebar.markdown('<div class="small">Conseil : faites une calibration px→mm pour des résultats fiables.</div>', unsafe_allow_html=True)

# =========================================================
# ------------------------ UPLOAD --------------------------
# =========================================================
st.header("📂 Chargement de l’image")
up = st.file_uploader("Dépose ton image peau + ligne laser (PNG/JPG)", type=["png", "jpg", "jpeg"])

if up:
    file_bytes = np.asarray(bytearray(up.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("❌ Impossible de lire l'image.")
        st.stop()

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # ---------------- Preprocess (skin) ----------------
    gray_pre = apply_preprocessing(gray, filter_method, ksize=filter_ksize, sigma=filter_sigma)

    # ---------------- Laser ridge + subpixel ----------------
    ridge = enhance_laser_ridge(gray_pre)
    ridge_vis = ridge_to_uint8(ridge)

    xs, ys_smooth, y0, band, mass = extract_line_subpixel(
        ridge, min_mass_ratio=min_mass_ratio, smooth_win=smooth_win
    )

    relief_metrics = None
    overlay = None

    if len(xs) > 10:
        baseline = local_baseline(ys_smooth, win=baseline_win)
        deviation_px = ys_smooth - baseline
        deviation_mm = deviation_px * mm_per_px if mm_per_px > 0 else None

        abs_dev = np.abs(deviation_px)
        max_dev = float(np.max(abs_dev)) if abs_dev.size else 0.0
        anomaly_threshold = anomaly_ratio * max_dev if max_dev > 0 else 0.0

        idx = np.where(abs_dev >= anomaly_threshold)[0] if max_dev > 0 else np.array([], dtype=int)
        anomaly_detected = idx.size >= int(min_anomaly_width)

        anomaly_range = None
        anomaly_peak_idx = None
        if anomaly_detected:
            anomaly_range = (int(xs[idx[0]]), int(xs[idx[-1]]))
            anomaly_peak_idx = int(idx[np.argmax(abs_dev[idx])])

        peak_idx = int(np.nanargmax(np.abs(deviation_px)))
        peak_px = float(deviation_px[peak_idx])
        peak_mm = float(peak_px * mm_per_px) if mm_per_px > 0 else None

        # width half max
        vmax = float(np.max(np.abs(deviation_px)))
        thr = 0.5 * vmax if vmax > 0 else 0
        above = np.where(np.abs(deviation_px) >= thr)[0]
        w_hm_px = float(xs[above[-1]] - xs[above[0]]) if above.size else 0.0
        w_hm_mm = float(w_hm_px * mm_per_px) if mm_per_px > 0 else None

        area_abs_px = float(np.trapz(np.abs(deviation_px), xs))
        area_abs_mm = float(area_abs_px * (mm_per_px ** 2)) if mm_per_px > 0 else None

        relief_metrics = {
            "peak_abs_px": abs(peak_px),
            "peak_abs_mm": abs(peak_mm) if peak_mm is not None else None,
            "w_halfmax_px": w_hm_px,
            "w_halfmax_mm": w_hm_mm,
            "area_abs_px": area_abs_px,
            "area_abs_mm": area_abs_mm,
            "anomaly_detected": anomaly_detected,
            "anomaly_threshold_px": anomaly_threshold,
            "anomaly_range": anomaly_range,
            "mass_mean": float(np.mean(mass)),
            "mass_min": float(np.min(mass)),
            "mass_max": float(np.max(mass))
        }

        # Overlay on ridge
        overlay = cv2.cvtColor(ridge_vis, cv2.COLOR_GRAY2BGR)

        # detected line (green) + baseline (orange)
        for x, y in zip(xs, ys_smooth):
            yy = int(np.clip(y, 0, overlay.shape[0] - 1))
            overlay[yy, int(x)] = (0, 255, 0)
        for x, yb in zip(xs, baseline):
            yy = int(np.clip(yb, 0, overlay.shape[0] - 1))
            overlay[yy, int(x)] = (255, 165, 0)

        # highlight anomaly
        if anomaly_detected and anomaly_range is not None:
            x_start, x_end = anomaly_range
            cv2.rectangle(overlay, (x_start, 0), (x_end, overlay.shape[0]-1), (255, 0, 0), 2)
            if anomaly_peak_idx is not None:
                xpk = int(xs[anomaly_peak_idx])
                ypk = int(np.clip(ys_smooth[anomaly_peak_idx], 0, overlay.shape[0]-1))
                cv2.circle(overlay, (xpk, ypk), 6, (0, 0, 255), -1)
                cv2.putText(overlay, "ANOMALY", (max(0, xpk-70), max(20, ypk-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)

    # ---------------- Skin analytics (texture/color/wrinkles) ----------------
    tex = texture_metrics(gray_pre)
    red_mean, red_std = redness_index(img_bgr)
    tone = tone_uniformity_lab(img_bgr)
    wrinkles = wrinkle_proxy(gray_pre)

    scores = score_from_metrics(tex, tone, (red_mean, red_std), wrinkles, relief_metrics)

    # =========================================================
    # ---------------------- DISPLAY UI ------------------------
    # =========================================================
    col1, col2 = st.columns(2)
    col1.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Image — Original", use_column_width=True)
    col2.image(gray_pre, caption=f"Pré-traitement peau ({filter_method})", use_column_width=True)

    col3, col4 = st.columns(2)
    col3.image(ridge_vis, caption="Ridge response (ligne laser renforcée)", use_column_width=True)
    if overlay is not None:
        col4.image(overlay, caption="Overlay: ligne (vert), baseline (orange), anomalie (bleu/rouge)", use_column_width=True)
    else:
        col4.info("Overlay non disponible (ligne laser non détectée ou signal insuffisant).")

    st.subheader("🧬 Indicateurs peau (texture / couleur / rides)")
    wcol1, wcol2 = st.columns(2)
    wcol1.image(wrinkles["blackhat_u8"], caption="Black-hat (structures fines sombres)", use_column_width=True)
    wcol2.image(wrinkles["wrinkle_mask"], caption="Masque ride-like (seuil adaptatif)", use_column_width=True)

    st.header("📊 Scores esthétiques (0–100)")
    s1, s2, s3 = st.columns(3)
    s1.markdown(f'<div class="metric-box"><div class="metric-title">Global Skin Quality</div><div class="metric-value">{scores["Global"]}</div></div>', unsafe_allow_html=True)
    s2.markdown(f'<div class="metric-box"><div class="metric-title">Smoothness</div><div class="metric-value">{scores["Smoothness"]}</div></div>', unsafe_allow_html=True)
    s3.markdown(f'<div class="metric-box"><div class="metric-title">Wrinkles</div><div class="metric-value">{scores["Wrinkles"]}</div></div>', unsafe_allow_html=True)

    s4, s5, s6 = st.columns(3)
    s4.markdown(f'<div class="metric-box"><div class="metric-title">Uniformity</div><div class="metric-value">{scores["Uniformity"]}</div></div>', unsafe_allow_html=True)
    s5.markdown(f'<div class="metric-box"><div class="metric-title">Redness</div><div class="metric-value">{scores["Redness"]}</div></div>', unsafe_allow_html=True)
    s6.markdown(f'<div class="metric-box"><div class="metric-title">Relief</div><div class="metric-value">{scores["Relief"]}</div></div>', unsafe_allow_html=True)

    # ============================ Metrics table ============================
    metrics_rows = []
    metrics_rows.append(["Texture (Laplacian energy)", f"{tex['lap_energy']:.1f}", "Plus élevé = grain/relief micro plus marqué"])
    metrics_rows.append(["Texture (grad mean)", f"{tex['grad_mean']:.2f}", "Plus élevé = transitions fines plus nombreuses"])
    metrics_rows.append(["Micro-contrast (std)", f"{tex['micro_contrast']:.2f}", "Plus élevé = contraste local plus fort"])

    metrics_rows.append(["Redness index (mean)", f"{red_mean:.3f}", "Plus élevé = rougeur plus présente"])
    metrics_rows.append(["Redness index (std)", f"{red_std:.3f}", "Plus élevé = rougeur plus hétérogène"])

    metrics_rows.append(["Tone L* mean", f"{tone['L_mean']:.1f}", "Luminosité moyenne (LAB)"])
    metrics_rows.append(["Tone L* std", f"{tone['L_std']:.2f}", "Plus élevé = teint moins uniforme"])
    metrics_rows.append(["Tone a* std", f"{tone['A_std']:.2f}", "Variabilité rouge/vert (hétérogénéité)"])

    metrics_rows.append(["Wrinkle density", f"{wrinkles['wrinkle_density']:.3f}", "Proportion ride-like (proxy)"])
    metrics_rows.append(["Wrinkle strength", f"{wrinkles['wrinkle_strength']:.1f}", "Intensité moyenne ride-like (proxy)"])

    if relief_metrics is not None:
        metrics_rows.append(["Relief peak |Δ| (px)", f"{relief_metrics['peak_abs_px']:.2f}", "Amplitude max du relief détecté"])
        if relief_metrics["peak_abs_mm"] is not None:
            metrics_rows.append(["Relief peak |Δ| (mm)", f"{relief_metrics['peak_abs_mm']:.2f}", "Amplitude max convertie en mm (si calibré)"])
        metrics_rows.append(["Relief width (½ max) px", f"{relief_metrics['w_halfmax_px']:.1f}", "Largeur au demi-maximum"])
        if relief_metrics["w_halfmax_mm"] is not None:
            metrics_rows.append(["Relief width (½ max) mm", f"{relief_metrics['w_halfmax_mm']:.2f}", "Largeur au demi-maximum en mm"])
        metrics_rows.append(["Relief area |Δ|", f"{relief_metrics['area_abs_px']:.0f}", "Aire absolue (proxy sévérité locale)"])
        metrics_rows.append(["Laser confidence mass (mean)", f"{relief_metrics['mass_mean']:.1f}", "Plus élevé = signal laser plus fiable"])
        metrics_rows.append(["Relief anomaly detected", "YES" if relief_metrics["anomaly_detected"] else "NO", "Détection par seuil relatif + largeur min"])

    st.subheader("📋 Mesures détaillées")
    st.dataframe(pd.DataFrame(metrics_rows, columns=["Paramètre", "Valeur", "Commentaire"]), use_container_width=True)

    # =========================================================
    # --------------------- EXPORTS ---------------------------
    # =========================================================
    st.header("⬇️ Export")

    # Prepare images for PDF
    images_pdf = {
        "original": cv_to_png_bytes(img_bgr),
        "preproc": cv_to_png_bytes(gray_pre),
        "ridge": cv_to_png_bytes(ridge_vis),
        "overlay": cv_to_png_bytes(overlay) if overlay is not None else None,
        "wrinkle_mask": cv_to_png_bytes(wrinkles["wrinkle_mask"]),
    }

    params_pdf = {
        "mm_per_px": mm_per_px,
        "filter_method": filter_method,
        "filter_ksize": filter_ksize,
        "filter_sigma": filter_sigma,
        "min_mass_ratio": min_mass_ratio,
        "smooth_win": smooth_win,
        "baseline_win": baseline_win,
        "anomaly_ratio": anomaly_ratio,
        "min_anomaly_width": min_anomaly_width,
    }

    pdf_bytes = make_pdf_report(
        patient_id=patient_id,
        session_date=session_date,
        method_filter=filter_method,
        params=params_pdf,
        metrics_table=metrics_rows,
        scores=scores,
        images=images_pdf
    )

    st.download_button(
        "📄 Télécharger le rapport PDF (clinique-style)",
        data=pdf_bytes,
        file_name=f"LaserLineSkin_Report_{patient_id}_{session_date}.pdf",
        mime="application/pdf"
    )

    # Optional CSV export for reproducibility
    st.download_button(
        "📊 Télécharger les mesures (CSV)",
        data=pd.DataFrame(metrics_rows, columns=["Paramètre", "Valeur", "Commentaire"]).to_csv(index=False).encode("utf-8"),
        file_name=f"LaserLineSkin_Metrics_{patient_id}_{session_date}.csv",
        mime="text/csv"
    )

st.markdown(
    '<div class="footer">© 2025 — LaserLineSkin • Rapport esthétique non diagnostique • Streamlit + OpenCV + ReportLab</div>',
    unsafe_allow_html=True
)
