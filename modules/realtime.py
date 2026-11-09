"""Deteksi dan penghitungan buah mangga secara langsung (webcam) dengan YOLO + ByteTrack.

Modul ini tidak bergantung pada Streamlit; loop tampilan ada di halaman `2_Deteksi_Buah.py`.
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from modules import counting
from modules.detection import color_for, draw_box, mango_class_ids
from modules.tracking import TRACK_COLUMNS, extract_tracks
from modules.utils import TRACKING_DIR, timestamp

MAX_ROWS = 200_000  # batas baris riwayat di memori (sesi sangat panjang tidak membengkak)


# ------------------------------------------------------------------ kamera
def open_camera(index: int = 0, width: int = 1280, height: int = 720) -> cv2.VideoCapture:
    """Buka webcam dan pastikan benar-benar dapat membaca frame. Melempar RuntimeError bila gagal."""
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
    last_cap = None
    for backend in backends:
        cap = cv2.VideoCapture(int(index), backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # kurangi lag
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
        last_cap = cap
    del last_cap
    raise RuntimeError(
        f"Kamera dengan indeks {index} tidak dapat dibuka. Pastikan webcam terpasang, tidak sedang dipakai "
        "aplikasi lain (Zoom/Meet/Camera), dan izin kamera untuk aplikasi desktop diaktifkan di Pengaturan Windows. "
        "Coba ganti indeks kamera (0, 1, 2)."
    )


def release_camera(cap: cv2.VideoCapture | None) -> None:
    """Lepaskan kamera dengan aman (boleh dipanggil berulang)."""
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass


# ------------------------------------------------------------------ FPS
class FpsMeter:
    """FPS tampilan yang dihaluskan dari N frame terakhir."""

    def __init__(self, window: int = 30):
        self.times: deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        self.times.append(time.perf_counter())
        if len(self.times) < 2:
            return 0.0
        span = self.times[-1] - self.times[0]
        return (len(self.times) - 1) / span if span > 0 else 0.0


# ------------------------------------------------------------------ sesi penghitungan
class RealtimeSession:
    """Menyimpan jejak tracking selama kamera berjalan dan menghitung buah unik."""

    def __init__(self, conf: float, imgsz: int, min_frames: int):
        self.conf, self.imgsz, self.min_frames = conf, imgsz, min_frames
        self.started = time.time()
        self.rows: list[tuple] = []
        self.id_frames: dict[int, int] = {}   # tracking ID -> jumlah frame terlihat
        self.frames = 0
        self.total_detections = 0
        self.active_now = 0
        self.last_fps = 0.0
        self.error: str | None = None
        self._est_cache: dict = {}

    def update(self, tracks: list[tuple]) -> None:
        self.frames += 1
        self.active_now = sum(1 for t in tracks if t[0] >= 0)
        self.total_detections += len(tracks)
        for tid, cf, x1, y1, x2, y2 in tracks:
            if tid >= 0:
                self.id_frames[tid] = self.id_frames.get(tid, 0) + 1
            if len(self.rows) < MAX_ROWS:
                self.rows.append((self.frames, tid, cf, x1, y1, x2, y2, (x1 + x2) / 2, (y1 + y2) / 2))

    @property
    def total_ids(self) -> int:
        return len(self.id_frames)

    @property
    def confirmed(self) -> int:
        """ID yang terlihat ≥ min_frames frame. Inilah hitungan 'mangga unik' real-time."""
        return sum(1 for n in self.id_frames.values() if n >= self.min_frames)

    @property
    def duration(self) -> float:
        return time.time() - self.started

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=TRACK_COLUMNS)

    def final_estimate(self, max_gap: int = 30, ratio: float = 1.0) -> dict:
        """Estimasi setelah kamera dihentikan: ID terkonfirmasi + penggabungan tracklet yang terputus."""
        key = (max_gap, ratio, self.frames, self.min_frames)
        if key in self._est_cache:  # hindari hitung ulang setiap rerun Streamlit
            return self._est_cache[key]
        df = self.to_df()
        if df.empty:
            est = dict(estimasi=0, batas_bawah=0, id_terkonfirmasi=0, batas_atas=0, id_dibuang=0, id_digabung=0, mapping={})
        else:
            est = counting.estimate_unique(df, self.min_frames, max_gap, ratio)
        self._est_cache = {key: est}
        return est

    def save(self, name: str = "Kamera Langsung") -> Path:
        """Simpan CSV + metadata ke output/tracking agar bisa dibuka di menu Tracking & Penghitungan Buah."""
        TRACKING_DIR.mkdir(parents=True, exist_ok=True)
        stem = f"kamera_{timestamp()}"
        csv_path = TRACKING_DIR / f"{stem}_tracks.csv"
        self.to_df().to_csv(csv_path, index=False)
        meta = dict(video=name, fps=round(self.last_fps, 2), frame_interval=1, conf=self.conf, imgsz=self.imgsz,
                    frames_diproses=self.frames, durasi_detik=round(self.duration, 1), sumber="webcam")
        csv_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return csv_path


# ------------------------------------------------------------------ pemrosesan satu frame
def process_live_frame(model, frame: np.ndarray, session: RealtimeSession, allowed: list[int] | None,
                       fps: float = 0.0) -> np.ndarray:
    """Deteksi + ByteTrack pada satu frame webcam; perbarui sesi; kembalikan frame beranotasi."""
    res = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=session.conf,
                      imgsz=session.imgsz, verbose=False)[0]
    tracks = extract_tracks(res, allowed)
    session.update(tracks)
    session.last_fps = fps

    out = frame.copy()
    for tid, cf, x1, y1, x2, y2 in tracks:
        label = f"ID {tid} {cf:.2f}" if tid >= 0 else f"{cf:.2f}"
        draw_box(out, (x1, y1, x2, y2), label, color_for(tid if tid >= 0 else 0))
    _overlay(out, [f"Mangga unik: {session.confirmed}", f"Terlihat: {len(tracks)}", f"FPS: {fps:.1f}"])
    return out


def _overlay(img: np.ndarray, lines: list[str]) -> None:
    scale = max(0.5, min(img.shape[:2]) / 700)
    y = int(28 * scale)
    for line in lines:
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        y += int(32 * scale)


def reset_tracker(model) -> None:
    """Buang state tracker lama agar ID mulai dari 1 pada sesi kamera baru."""
    model.predictor = None


def allowed_classes(model) -> list[int] | None:
    return mango_class_ids(model)
