"""Deteksi buah mangga dengan Ultralytics YOLO + manajemen dataset/training/validasi."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from modules.utils import BASE_DIR, DATASET_DIR, MODELS_DIR, OUTPUT_DIR

SPLITS = ("train", "val", "test")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRAIN_LOG = OUTPUT_DIR / "train_log.txt"


# ------------------------------------------------------------------ model
@st.cache_resource(show_spinner="Memuat model YOLO ...")
def _load_model_cached(path: str, mtime: float):
    from ultralytics import YOLO  # import di sini agar aplikasi tetap terbuka walau ultralytics belum terpasang
    return YOLO(path)


def load_model(path: str):
    """Muat model (di-cache; otomatis dimuat ulang jika file berubah)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File model tidak ditemukan: {p}")
    try:
        return _load_model_cached(str(p), p.stat().st_mtime)
    except ImportError as e:
        raise ImportError("Library 'ultralytics' belum terpasang. Jalankan: pip install ultralytics") from e


def model_class_names(model) -> dict[int, str]:
    names = model.names
    return dict(names) if isinstance(names, dict) else {i: n for i, n in enumerate(names)}


def mango_class_ids(model) -> list[int] | None:
    """ID kelas yang namanya mengandung 'mango'/'mangga'. None = pakai semua kelas."""
    ids = [i for i, n in model_class_names(model).items() if "mang" in str(n).lower()]
    return ids or None



def model_warning(model) -> str | None:
    """Peringatan jika model tampaknya bukan detektor mangga."""
    names = model_class_names(model)

    if len(names) >= 80:
        return (
            "Model ini memiliki ≥80 kelas (kemungkinan COCO pretrained). "
            "Model tersebut bukan detektor mangga. "
            "Latih model dengan dataset mangga sebelum hasilnya dipakai."
        )

    # Model dengan kelas kematangan mangga dianggap valid.
    ripeness_classes = {
        "unripe",
        "ripe",
        "over ripe",
        "overripe",
        "mature",
        "harvest-ready",
    }

    normalized_names = {
        str(name).strip().lower()
        for name in names.values()
    }

    if normalized_names.intersection(ripeness_classes):
        return None

    if mango_class_ids(model) is None:
        return (
            f"Tidak ada kelas bernama 'mango/mangga' pada model "
            f"(kelas: {list(names.values())}). "
            "Semua kelas akan dihitung sebagai buah. "
            "Pastikan ini sesuai dengan dataset Anda."
        )

    return None


# ------------------------------------------------------------------ gambar
def color_for(i: int) -> tuple[int, int, int]:
    palette = [(46, 160, 67), (0, 140, 255), (255, 99, 71), (180, 70, 200), (0, 200, 200),
               (60, 60, 230), (200, 160, 0), (120, 200, 60)]
    return palette[int(i) % len(palette)]


def draw_box(img: np.ndarray, box, label: str, color) -> None:
    h, w = img.shape[:2]
    scale = max(0.4, min(w, h) / 1100)
    th = max(1, int(round(scale * 2.5)))
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, th)
    (tw, tht), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale * 0.6, max(1, th - 1))
    ty = max(y1, tht + bl + 2)
    cv2.rectangle(img, (x1, ty - tht - bl - 2), (x1 + tw + 4, ty), color, -1)
    cv2.putText(img, label, (x1 + 2, ty - bl), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.6,
                (255, 255, 255), max(1, th - 1), cv2.LINE_AA)


def detect_image(model, bgr: np.ndarray, conf: float, imgsz: int) -> tuple[list[dict], np.ndarray]:
    """Deteksi dan klasifikasi tingkat kematangan mangga."""
    allowed = mango_class_ids(model)

    res = model.predict(bgr, conf=conf, imgsz=imgsz, iou=0.55, agnostic_nms=True, max_det=300, verbose=False)[0]

    names = model_class_names(model)
    dets, out = [], bgr.copy()

    if res.boxes is not None:
        for b in res.boxes:
            cls = int(b.cls[0])

            if allowed is not None and cls not in allowed:
                continue

            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            c = float(b.conf[0])

            dets.append(dict(x1=x1, y1=y1, x2=x2, y2=y2, conf=c, cls=cls, name=names.get(cls, str(cls))))

    dets.sort(key=lambda d: (d["y1"], d["x1"]))

    for i, d in enumerate(dets, 1):
        draw_box(out, (d["x1"], d["y1"], d["x2"], d["y2"]), f"#{i} {d['name']} {d['conf']:.2f}", color_for(d["cls"]))

    return dets, out# ------------------------------------------------------------------ dataset
def dataset_status() -> dict[str, dict[str, int]]:
    """Hitung jumlah gambar & label per split."""
    status = {}
    for s in SPLITS:
        imgs = [p for p in (DATASET_DIR / "images" / s).glob("*") if p.suffix.lower() in IMG_EXT]
        lbls = list((DATASET_DIR / "labels" / s).glob("*.txt"))
        stems_i = {p.stem for p in imgs}
        stems_l = {p.stem for p in lbls}
        status[s] = dict(gambar=len(imgs), label=len(lbls), tanpa_label=len(stems_i - stems_l))
    return status


def write_data_yaml(class_names: list[str] | None = None) -> Path:
    """Tulis dataset/data.yaml dengan path absolut (agar Ultralytics tidak salah mencari folder)."""
    names = class_names or ["mango"]
    root = DATASET_DIR.resolve().as_posix()
    lines = [f"path: {root}", "train: images/train", "val: images/val", "test: images/test",
             f"nc: {len(names)}", "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    p = DATASET_DIR / "data.yaml"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


# ------------------------------------------------------------------ training (proses terpisah)
def start_training(base_model: str, epochs: int, imgsz: int, batch: int, name: str, device: str) -> subprocess.Popen:
    """Jalankan train.py sebagai proses terpisah supaya UI Streamlit tidak membeku."""
    data_yaml = write_data_yaml()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log = open(TRAIN_LOG, "w", encoding="utf-8")
    cmd = [sys.executable, str(BASE_DIR / "train.py"), "--data", str(data_yaml), "--model", base_model,
           "--epochs", str(epochs), "--imgsz", str(imgsz), "--batch", str(batch), "--name", name,
           "--device", device]
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(BASE_DIR))


def read_train_log(max_chars: int = 6000) -> str:
    if not TRAIN_LOG.exists():
        return ""
    txt = TRAIN_LOG.read_text(encoding="utf-8", errors="ignore")
    return txt[-max_chars:]


def validate_model(model_path: str, imgsz: int = 640, split: str = "val") -> dict:
    """Validasi model pada split dataset. Mengembalikan precision, recall, mAP."""
    data_yaml = write_data_yaml()
    model = load_model(model_path)
    m = model.val(data=str(data_yaml), imgsz=imgsz, split=split, verbose=False, plots=False)
    return dict(
        precision=float(m.box.mp), recall=float(m.box.mr),
        map50=float(m.box.map50), map50_95=float(m.box.map),
    )
