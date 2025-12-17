import numpy as np
import pandas as pd
import cv2
from scipy.signal import savgol_filter
import streamlit as st

# ============================
# --------- Styling ----------
# ============================
st.set_page_config(page_title="LaserLineSkin — Triangulation", page_icon="💡", layout="wide")
_CSS = """
<style>
    .stApp { background: #0f172a; color: #e5e7eb; font-family: 'Segoe UI', sans-serif; }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    h1, h2, h3 { color: #f8fafc; font-weight: 600; }
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

# ============================
# --------- Header -----------
# ============================
st.markdown("<h1 style='text-align:center'>💡 LaserLineSkin</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center; color:#60a5fa'>Analyse par triangulation laser — Imagerie médicale simplifiée</h3>", unsafe_allow_html=True)
st.write(
    "**But :** importer une image (a), renforcer la ligne laser (ridge), extraire son centre **subpixel**, "
    "calculer la **déformation (px et mm)** et exporter les résultats — méthode déterministe, sans IA."
)

# ============================
# ------- Core methods -------
# ============================

def enhance_laser_ridge(gray: np.ndarray) -> np.ndarray:
    """
    Renforce une ligne fine (crête) sur fond texturé.
    Retourne une réponse float32 (ridge response).
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    g = clahe.apply(gray)

    g = cv2.GaussianBlur(g, (0, 0), 1.0)

    sigma1, sigma2 = 1.2, 3.0
    g1 = cv2.GaussianBlur(g, (0, 0), sigma1)
    g2 = cv2.GaussianBlur(g, (0, 0), sigma2)
    dog = cv2.subtract(g1, g2).astype(np.float32)

    # Lissage horizontal (sigmaX=2.0, sigmaY=0.0) -> arguments positionnels
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
    Extraction subpixel qui CONSERVE les anomalies.
    Rejette uniquement les colonnes où le signal laser est trop faible (mass faible).
    Retour: xs, ys_smooth, y0, band, mass
    """
    h, w = ridge_img.shape

    # A) Estimer y0 via projection verticale
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

    # B) Centroid subpixel + confidence (mass)
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

    # C) Interpolation (comble seulement les trous sans signal)
    ys_interp = np.interp(xs, xs_k, ys_k)

    # D) Lissage léger (ne pas effacer le bump)
    if smooth_win % 2 == 0:
        smooth_win += 1
    smooth_win = min(smooth_win, (w // 3) * 2 + 1)

    if smooth_win >= 9:
        ys_smooth = savgol_filter(ys_interp, window_length=smooth_win, polyorder=2)
    else:
        ys_smooth = ys_interp

    return xs, ys_smooth, y0, band, mass


def local_baseline(y: np.ndarray, win: int = 301) -> np.ndarray:
    """
    Baseline locale très lisse pour faire ressortir anomalies locales.
    """
    if win % 2 == 0:
        win += 1
    win = min(win, (len(y) // 2) * 2 + 1)
    if win < 9:
        return y.copy()
    return savgol_filter(y, window_length=win, polyorder=2)

# ============================
# --------- Sidebar ----------
# ============================
st.sidebar.header("⚙️ Paramètres")
mm_per_px = st.sidebar.number_input("Facteur d'échelle (mm/px)", value=0.10, step=0.01, min_value=0.0)

min_mass_ratio = st.sidebar.slider("Seuil confiance (min_mass_ratio)", 0.01, 0.50, 0.10, 0.01)
smooth_win = st.sidebar.slider("Lissage ligne (smooth_win)", 9, 151, 41, 2)
baseline_win = st.sidebar.slider("Fenêtre baseline (baseline_win)", 51, 701, 301, 2)

show_debug_band = st.sidebar.checkbox("Afficher bande de recherche", value=False)
show_debug_mass = st.sidebar.checkbox("Afficher la confidence (mass)", value=False)

st.sidebar.markdown('<div class="small">⚠️ Ajuste mm/px après calibration.</div>', unsafe_allow_html=True)

# ============================
# --------- Uploader ---------
# ============================
st.header("📂 Chargement de l’image")
up = st.file_uploader("Dépose ton image (a) (PNG/JPG)", type=["png", "jpg", "jpeg"])

if up:
    file_bytes = np.asarray(bytearray(up.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("❌ Impossible de lire l'image.")
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1) ridge enhancement
        ridge = enhance_laser_ridge(gray)
        ridge_vis = ridge_to_uint8(ridge)

        # 2) extraction subpixel
        xs, ys_smooth, y0, band, mass = extract_line_subpixel(
            ridge, min_mass_ratio=min_mass_ratio, smooth_win=smooth_win
        )

        if len(xs) <= 10:
            st.error("❌ Ligne non détectée. Essaie une image avec une ligne plus visible.")
        else:
            # 3) baseline locale
            baseline = local_baseline(ys_smooth, win=baseline_win)

            deviation_px = ys_smooth - baseline
            deviation_mm = deviation_px * mm_per_px if mm_per_px > 0 else None

            # --- Layout images ---
            col1, col2 = st.columns(2)
            col1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Image (a) — Original", use_column_width=True)
            col2.image(ridge_vis, caption="Image ridge-like — renforcement ligne laser", use_column_width=True)

            # Debug band
            if show_debug_band:
                ridge_band = cv2.cvtColor(ridge_vis, cv2.COLOR_GRAY2BGR)
                y_min = max(0, y0 - band)
                y_max = min(ridge_band.shape[0] - 1, y0 + band)
                cv2.rectangle(ridge_band, (0, y_min), (ridge_band.shape[1]-1, y_max), (255, 165, 0), 2)
                st.image(ridge_band, caption=f"Bande de recherche autour de y0={y0} (±{band})", use_column_width=True)

            # Debug mass (confidence)
            if show_debug_mass:
                mass_norm = (mass / (mass.max() + 1e-6) * 255.0).astype(np.uint8)
                mass_img = np.tile(mass_norm[None, :], (60, 1))
                st.image(mass_img, caption="Confidence (mass) par colonne (clair = laser fort)", use_column_width=True)

            # Overlay sur ridge_vis
            overlay = cv2.cvtColor(ridge_vis, cv2.COLOR_GRAY2BGR)

            # ligne détectée (vert)
            for x, y in zip(xs, ys_smooth):
                yy = int(np.clip(y, 0, overlay.shape[0] - 1))
                overlay[yy, int(x)] = (0, 255, 0)

            # baseline (orange)
            for x, yb in zip(xs, baseline):
                yy = int(np.clip(yb, 0, overlay.shape[0] - 1))
                overlay[yy, int(x)] = (255, 165, 0)

            st.image(overlay, caption="Superposition : ligne détectée (vert) & baseline (orange)", use_column_width=True)

            # ============================
            # --------- Metrics ----------
            # ============================
            st.header("📊 Mesures")

            peak_idx = int(np.nanargmax(np.abs(deviation_px)))
            peak_px = float(deviation_px[peak_idx])
            peak_mm = peak_px * mm_per_px if mm_per_px > 0 else None

            vmax = float(np.max(np.abs(deviation_px)))
            thr = 0.5 * vmax if vmax > 0 else 0
            above = np.where(np.abs(deviation_px) >= thr)[0]

            w_hm_px = float(xs[above[-1]] - xs[above[0]]) if above.size else 0.0
            w_hm_mm = w_hm_px * mm_per_px if mm_per_px > 0 else None

            area_abs_px = float(np.trapz(np.abs(deviation_px), xs))
            area_abs_mm = area_abs_px * (mm_per_px ** 2) if mm_per_px > 0 else None

            mcol1, mcol2, mcol3 = st.columns(3)

            mcol1.markdown(
                f'<div class="metric-box"><div class="metric-title">📏 Pic |Δ|</div>'
                f'<div class="metric-value">{abs(peak_px):.2f}px '
                f'{(f"({abs(peak_mm):.2f}mm)" if peak_mm is not None else "")}'
                f'</div></div>',
                unsafe_allow_html=True
            )

            mcol2.markdown(
                f'<div class="metric-box"><div class="metric-title">📐 Largeur ½ max</div>'
                f'<div class="metric-value">{w_hm_px:.2f}px '
                f'{(f"({w_hm_mm:.2f}mm)" if w_hm_mm is not None else "")}'
                f'</div></div>',
                unsafe_allow_html=True
            )

            mcol3.markdown(
                f'<div class="metric-box"><div class="metric-title">📊 Aire |Δ|</div>'
                f'<div class="metric-value">{area_abs_px:.0f}px·col '
                f'{(f"({area_abs_mm:.2f}mm²)" if area_abs_mm is not None else "")}'
                f'</div></div>',
                unsafe_allow_html=True
            )

            # ============================
            # --------- Export -----------
            # ============================
            st.header("⬇️ Export")
            df = pd.DataFrame({
                "x": xs,
                "y_smooth_px": ys_smooth,
                "baseline_px": baseline,
                "deviation_px": deviation_px,
                "mass_confidence": mass
            })
            if deviation_mm is not None:
                df["deviation_mm"] = deviation_mm

            st.download_button(
                "Télécharger le profil CSV",
                df.to_csv(index=False).encode("utf-8"),
                "profile_from_ridge.csv",
                "text/csv"
            )

            _, buf_r = cv2.imencode(".png", ridge_vis)
            st.download_button("Télécharger image ridge-like", buf_r.tobytes(), "out_ridge_like.png", "image/png")

            _, buf_o = cv2.imencode(".png", overlay)
            st.download_button("Télécharger overlay", buf_o.tobytes(), "overlay_from_ridge.png", "image/png")

# ============================
# --------- Footer -----------
# ============================
st.markdown(
    '<div class="footer">© 2025 — LaserLineSkin • Interface Streamlit • Méthode déterministe (OpenCV)</div>',
    unsafe_allow_html=True
)
