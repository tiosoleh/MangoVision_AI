"""Fungsi preprocessing citra dengan OpenCV (tidak bergantung pada Streamlit)."""
from __future__ import annotations

import cv2
import numpy as np


def resize_keep_ratio(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if abs(scale - 1.0) < 1e-6:
        return img
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=interp)


def to_grayscale(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def gaussian_blur(img: np.ndarray, ksize: int) -> np.ndarray:
    k = int(ksize)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), 0)


def hist_equalization(img: np.ndarray) -> np.ndarray:
    """Equalization pada kanal luminans (Y) agar warna tidak rusak."""
    if img.ndim == 2:
        return cv2.equalizeHist(img)
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycc[:, :, 0] = cv2.equalizeHist(ycc[:, :, 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def clahe(img: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    c = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile), int(tile)))
    if img.ndim == 2:
        return c.apply(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = c.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def crop_percent(img: np.ndarray, left: int, right: int, top: int, bottom: int) -> np.ndarray:
    """Crop berdasarkan persen yang dibuang dari tiap sisi (0-45)."""
    h, w = img.shape[:2]
    x1, x2 = int(w * left / 100), int(w * (100 - right) / 100)
    y1, y2 = int(h * top / 100), int(h * (100 - bottom) / 100)
    if x2 - x1 < 10 or y2 - y1 < 10:
        raise ValueError("Area crop terlalu kecil. Kurangi persentase crop.")
    return img[y1:y2, x1:x2].copy()


def brightness_contrast(img: np.ndarray, brightness: int = 0, contrast: int = 0) -> np.ndarray:
    """brightness: -100..100, contrast: -100..100."""
    alpha = 1.0 + contrast / 100.0  # 0..2
    beta = float(brightness) * 1.28
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def apply_pipeline(img: np.ndarray, p: dict) -> tuple[np.ndarray, list[str]]:
    """Terapkan urutan: crop -> resize -> brightness/contrast -> grayscale -> HE/CLAHE -> blur.
    p: dict opsi. Mengembalikan (hasil, log proses)."""
    out, log = img.copy(), []
    if p.get("crop"):
        out = crop_percent(out, p["crop_l"], p["crop_r"], p["crop_t"], p["crop_b"])
        log.append(f"Cropping (kiri {p['crop_l']}%, kanan {p['crop_r']}%, atas {p['crop_t']}%, bawah {p['crop_b']}%) → {out.shape[1]}×{out.shape[0]} px")
    if p.get("resize"):
        out = resize_keep_ratio(out, int(p["max_side"]))
        log.append(f"Resize (sisi terpanjang {p['max_side']} px, rasio dipertahankan) → {out.shape[1]}×{out.shape[0]} px")
    if p.get("brightness", 0) != 0 or p.get("contrast", 0) != 0:
        out = brightness_contrast(out, p.get("brightness", 0), p.get("contrast", 0))
        log.append(f"Brightness {p.get('brightness', 0):+d}, Contrast {p.get('contrast', 0):+d}")
    if p.get("gray"):
        out = to_grayscale(out)
        log.append("Konversi Grayscale")
    if p.get("hist_eq"):
        out = hist_equalization(out)
        log.append("Histogram Equalization")
    if p.get("clahe"):
        out = clahe(out, p.get("clip", 2.0), p.get("tile", 8))
        log.append(f"CLAHE (clip limit {p.get('clip', 2.0)}, tile {p.get('tile', 8)}×{p.get('tile', 8)})")
    if p.get("blur"):
        out = gaussian_blur(out, p["blur_k"])
        log.append(f"Gaussian Blur (kernel {p['blur_k'] | 1}×{p['blur_k'] | 1})")
    if not log:
        log.append("Tidak ada proses yang dipilih; gambar tidak berubah.")
    return out, log


def laplacian_sharpness(img: np.ndarray) -> float:
    """Variansi Laplacian: nilai kecil menandakan foto blur."""
    g = to_grayscale(img)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())
