"""Metrik evaluasi: error penghitungan (Error%, MAE, RMSE, MAPE) dan Precision/Recall/F1 dari ground truth."""
from __future__ import annotations

import numpy as np
import pandas as pd


def count_error(actual: float, estimate: float) -> dict:
    """Error (%) = |Est - Aktual| / Aktual × 100%. Jika aktual = 0, persentase tidak terdefinisi (None)."""
    abs_err = abs(estimate - actual)
    pct = (abs_err / actual * 100.0) if actual > 0 else None
    return dict(abs_error=float(abs_err), error_pct=pct, bias=float(estimate - actual))


def aggregate_errors(df: pd.DataFrame) -> dict:
    """df: kolom 'aktual' & 'estimasi'. MAE, RMSE, MAPE (hanya baris dengan aktual>0)."""
    if df.empty:
        return {}
    a = df["aktual"].astype(float).to_numpy()
    e = df["estimasi"].astype(float).to_numpy()
    diff = e - a
    mask = a > 0
    return dict(
        n=int(len(df)),
        mae=float(np.mean(np.abs(diff))),
        rmse=float(np.sqrt(np.mean(diff ** 2))),
        mape=float(np.mean(np.abs(diff[mask]) / a[mask]) * 100) if mask.any() else None,
        bias=float(np.mean(diff)),
    )


# ---------------- ground truth YOLO -> precision / recall ----------------
def parse_yolo_labels(text: str, width: int, height: int) -> np.ndarray:
    """Baca label YOLO (class cx cy w h, ternormalisasi) -> array Nx4 [x1,y1,x2,y2] piksel."""
    boxes = []
    for ln, line in enumerate(text.strip().splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 5:
            raise ValueError(f"Baris {ln} label tidak valid (butuh minimal 5 nilai): '{line}'")
        _, cx, cy, w, h = parts[:5]
        cx, cy, w, h = float(cx) * width, float(cy) * height, float(w) * width, float(h) * height
        boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return np.array(boxes, dtype=float).reshape(-1, 4)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_detections(pred: np.ndarray, conf: np.ndarray, gt: np.ndarray, iou_thr: float = 0.5) -> dict:
    """Pencocokan greedy berdasarkan confidence tertinggi. Mengembalikan TP, FP, FN."""
    order = np.argsort(-conf) if len(conf) else np.array([], dtype=int)
    ious = iou_matrix(pred, gt)
    used, tp = set(), 0
    for i in order:
        if ious.shape[1] == 0:
            break
        cand = [(ious[i, j], j) for j in range(ious.shape[1]) if j not in used and ious[i, j] >= iou_thr]
        if cand:
            used.add(max(cand)[1])
            tp += 1
    fp, fn = len(pred) - tp, len(gt) - tp
    return dict(tp=tp, fp=fp, fn=fn)


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return dict(precision=p, recall=r, f1=f1)
