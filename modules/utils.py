"""Fungsi bantu umum: path, konversi gambar, CSS, header halaman, pemilih model."""
from __future__ import annotations

import io
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
DETECTION_DIR = OUTPUT_DIR / "detection"
TRACKING_DIR = OUTPUT_DIR / "tracking"
PREPROC_DIR = OUTPUT_DIR / "preprocessing"
VIDEO_INPUT_DIR = OUTPUT_DIR / "video_input"
DATASET_DIR = BASE_DIR / "dataset"
DB_PATH = BASE_DIR / "database" / "mangovision.db"
CSS_PATH = BASE_DIR / "assets" / "style.css"

APP_NAME = "MangoVision AI"
APP_SUBTITLE = "Sistem Deteksi dan Penghitungan Buah Mangga Berbasis Computer Vision"


def ensure_dirs() -> None:
    for d in (MODELS_DIR, DETECTION_DIR, TRACKING_DIR, PREPROC_DIR, VIDEO_INPUT_DIR,
              DB_PATH.parent, DATASET_DIR / "images" / "train", DATASET_DIR / "labels" / "train"):
        d.mkdir(parents=True, exist_ok=True)


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def safe_stem(name: str) -> str:
    stem = Path(name).stem
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:60] or "file"


# ---------- konversi gambar ----------
def load_image_bgr(uploaded_file) -> np.ndarray:
    """Baca file upload Streamlit -> array BGR (koreksi orientasi EXIF ponsel)."""
    img = Image.open(io.BytesIO(uploaded_file.getvalue()))
    img = ImageOps.exif_transpose(img).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Gagal meng-encode gambar ke PNG.")
    return buf.tobytes()


def save_image(img: np.ndarray, folder: Path, prefix: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{prefix}_{timestamp()}.png"
    if not cv2.imwrite(str(path), img):
        raise IOError(f"Gagal menyimpan gambar ke {path}")
    return path


# ---------- tampilan ----------
def inject_css() -> None:
    if CSS_PATH.exists():
        st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", icon: str = "🥭") -> None:
    st.markdown(
        f"<div class='mv-header'><h1>{icon} {title}</h1><p>{subtitle}</p></div>",
        unsafe_allow_html=True,
    )


def info_card(title: str, body: str) -> None:
    st.markdown(f"<div class='mv-card'><h4>{title}</h4><p>{body}</p></div>", unsafe_allow_html=True)


# ---------- pemilih model (dipakai halaman deteksi & video) ----------
def list_models() -> list[str]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(p.name for p in MODELS_DIR.glob("*.pt"))


def model_selector(key: str = "model_name") -> str | None:
    """Tampilkan selectbox model di halaman; kembalikan path model terpilih atau None."""
    models = list_models()
    if not models:
        st.error(
            "⚠️ **Model mangga belum tersedia.** Letakkan file `best.pt` hasil training di folder "
            "`models/`, atau unggah/latih model di tab **Manajemen Model** pada halaman *Deteksi Buah Mangga*."
        )
        return None
    default = models.index("mango_ripeness.pt") if "mango_ripeness.pt" in models else (
    models.index("best.pt") if "best.pt" in models else 0
)
    name = st.selectbox("Model YOLO (folder models/)", models, index=default, key=key)
    return str(MODELS_DIR / name)
