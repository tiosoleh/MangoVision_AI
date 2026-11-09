"""Tracking buah pada video dengan YOLO + ByteTrack (Ultralytics)."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from modules.detection import color_for, draw_box, mango_class_ids
from modules.utils import TRACKING_DIR, safe_stem, timestamp

TRACK_COLUMNS = ["frame", "track_id", "conf", "x1", "y1", "x2", "y2", "cx", "cy"]


def _ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def reencode_h264(src: Path, dst: Path) -> bool:
    """Ubah ke H.264 agar bisa diputar di browser (st.video). Return False jika gagal."""
    exe = _ffmpeg_exe()
    if not exe:
        return False
    try:
        r = subprocess.run([exe, "-y", "-i", str(src), "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                            "-preset", "veryfast", "-crf", "26", "-an", str(dst)],
                           capture_output=True, timeout=1800)
        return r.returncode == 0 and dst.exists() and dst.stat().st_size > 0
    except Exception:
        return False


def extract_tracks(res, allowed: list[int] | None) -> list[tuple]:
    """Ambil (track_id, conf, x1, y1, x2, y2) dari satu hasil YOLO (track_id = -1 bila belum ada ID).
    Dipakai bersama oleh analisis video dan kamera langsung."""
    out: list[tuple] = []
    if res.boxes is None or len(res.boxes) == 0:
        return out
    ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else None
    xyxy = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    clss = res.boxes.cls.cpu().numpy().astype(int)
    for j in range(len(xyxy)):
        if allowed is not None and clss[j] not in allowed:
            continue
        x1, y1, x2, y2 = [float(v) for v in xyxy[j]]
        out.append((int(ids[j]) if ids is not None else -1, float(confs[j]), x1, y1, x2, y2))
    return out


def video_info(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise ValueError("Video tidak dapat dibuka. Coba format MP4 (H.264).")
        return dict(fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
                    frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
                    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        cap.release()


def process_video(model, video_path: str, original_name: str, conf: float, imgsz: int,
                  frame_interval: int = 1, max_frames: int = 0,
                  progress_cb: Callable[[float, str], None] | None = None) -> dict:
    """Proses video frame demi frame (streaming, hemat memori).

    Mengembalikan dict: df (tabel tracking), video_path (hasil anotasi), csv_path, meta.
    """
    info = video_info(video_path)
    stride = max(1, int(frame_interval))
    out_fps = max(1.0, info["fps"] / stride)
    stem = f"{safe_stem(original_name)}_{timestamp()}"
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = TRACKING_DIR / f"{stem}_raw.mp4"
    final_path = TRACKING_DIR / f"{stem}_hasil.mp4"
    csv_path = TRACKING_DIR / f"{stem}_tracks.csv"

    allowed = mango_class_ids(model)
    model.predictor = None  # reset state tracker agar ID mulai dari awal untuk video baru

    results = model.track(source=video_path, tracker="bytetrack.yaml", conf=conf, imgsz=imgsz,
                          persist=True, stream=True, vid_stride=stride, verbose=False)

    writer, rows, n_proc = None, [], 0
    total_proc = max(1, info["frames"] // stride) if info["frames"] else 0
    try:
        for i, res in enumerate(results):
            frame = res.orig_img.copy()
            frame_idx = i * stride
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))
                if not writer.isOpened():
                    raise IOError("Gagal membuat file video keluaran (VideoWriter).")
            tracks = extract_tracks(res, allowed)
            n_obj = len(tracks)
            for tid, cf, x1, y1, x2, y2 in tracks:
                rows.append((frame_idx, tid, cf, x1, y1, x2, y2, (x1 + x2) / 2, (y1 + y2) / 2))
                label = f"ID {tid} {cf:.2f}" if tid >= 0 else f"{cf:.2f}"
                draw_box(frame, (x1, y1, x2, y2), label, color_for(tid if tid >= 0 else 0))
            cv2.putText(frame, f"Frame {frame_idx} | Objek: {n_obj}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, f"Frame {frame_idx} | Objek: {n_obj}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)
            n_proc += 1
            if progress_cb:
                frac = min(1.0, n_proc / total_proc) if total_proc else 0.0
                progress_cb(frac, f"Memproses frame {frame_idx} ({n_proc} frame dianalisis)")
            if max_frames and n_proc >= max_frames:
                break
    finally:
        if writer is not None:
            writer.release()
        try:  # hentikan generator bila berhenti lebih awal
            results.close()
        except Exception:
            pass

    if n_proc == 0:
        raise RuntimeError("Tidak ada frame yang berhasil diproses.")

    df = pd.DataFrame(rows, columns=TRACK_COLUMNS)
    df.to_csv(csv_path, index=False)

    playable = reencode_h264(raw_path, final_path)
    if playable:
        raw_path.unlink(missing_ok=True)
        video_out = final_path
    else:
        video_out = raw_path  # codec mp4v: bisa diunduh, tetapi mungkin tidak tampil di browser

    meta = dict(video=original_name, fps=info["fps"], frame_interval=stride, conf=conf, imgsz=imgsz,
                frames_diproses=n_proc, total_frame_video=info["frames"],
                video_hasil=video_out.name, browser_playable=playable)
    csv_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return dict(df=df, video_path=video_out, csv_path=csv_path, meta=meta)


# ---------- membaca hasil tersimpan ----------

def process_video(
    model,
    video_path: str,
    original_name: str,
    conf: float,
    imgsz: int,
    frame_interval: int = 1,
    max_frames: int = 0,
    progress_cb: Callable[[float, str], None] | None = None
) -> dict:
    """Proses video menggunakan OpenCV dan YOLO ByteTrack."""

    info = video_info(video_path)

    stride = max(1, int(frame_interval))
    out_fps = max(1.0, info["fps"] / stride)

    stem = f"{safe_stem(original_name)}_{timestamp()}"

    TRACKING_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = TRACKING_DIR / f"{stem}_raw.mp4"
    final_path = TRACKING_DIR / f"{stem}_hasil.mp4"
    csv_path = TRACKING_DIR / f"{stem}_tracks.csv"

    allowed = mango_class_ids(model)

    # Reset tracker untuk video baru
    model.predictor = None

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            "OpenCV gagal membuka video. "
            "Pastikan video menggunakan format MP4 yang kompatibel."
        )

    writer = None
    rows = []
    n_proc = 0
    frame_idx = 0

    total_proc = max(
        1,
        (info["frames"] + stride - 1) // stride
    )

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Proses frame berdasarkan interval
            if frame_idx % stride == 0:

                # Jalankan YOLO ByteTrack pada frame
                results = model.track(
                    source=frame,
                    tracker="bytetrack.yaml",
                    conf=conf,
                    imgsz=imgsz,
                    persist=True,
                    verbose=False
                )

                if not results:
                    frame_idx += 1
                    continue

                res = results[0]

                if res.orig_img is not None:
                    frame = res.orig_img.copy()

                # Inisialisasi video writer
                if writer is None:
                    h, w = frame.shape[:2]

                    writer = cv2.VideoWriter(
                        str(raw_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        out_fps,
                        (w, h)
                    )

                    if not writer.isOpened():
                        raise IOError(
                            "Gagal membuat video keluaran."
                        )

                tracks = extract_tracks(res, allowed)

                n_obj = len(tracks)

                for tid, cf, x1, y1, x2, y2 in tracks:
                    rows.append((
                        frame_idx,
                        tid,
                        cf,
                        x1,
                        y1,
                        x2,
                        y2,
                        (x1 + x2) / 2,
                        (y1 + y2) / 2
                    ))

                    label = (
                        f"ID {tid} {cf:.2f}"
                        if tid >= 0
                        else f"{cf:.2f}"
                    )

                    draw_box(
                        frame,
                        (x1, y1, x2, y2),
                        label,
                        color_for(tid if tid >= 0 else 0)
                    )

                cv2.putText(
                    frame,
                    f"Frame {frame_idx} | Objek: {n_obj}",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA
                )

                cv2.putText(
                    frame,
                    f"Frame {frame_idx} | Objek: {n_obj}",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                writer.write(frame)

                n_proc += 1

                if progress_cb:
                    frac = min(
                        1.0,
                        n_proc / total_proc
                    )

                    progress_cb(
                        frac,
                        f"Memproses frame {frame_idx} "
                        f"({n_proc} frame dianalisis)"
                    )

                if max_frames and n_proc >= max_frames:
                    break

            frame_idx += 1

    finally:
        cap.release()

        if writer is not None:
            writer.release()

    if n_proc == 0:
        raise RuntimeError(
            "Tidak ada frame yang berhasil diproses. "
            "Periksa format video atau kompatibilitas OpenCV."
        )

    df = pd.DataFrame(rows, columns=TRACK_COLUMNS)

    df.to_csv(csv_path, index=False)

    playable = reencode_h264(raw_path, final_path)

    if playable:
        raw_path.unlink(missing_ok=True)
        video_out = final_path
    else:
        video_out = raw_path

    meta = dict(
        video=original_name,
        fps=info["fps"],
        frame_interval=stride,
        conf=conf,
        imgsz=imgsz,
        frames_diproses=n_proc,
        total_frame_video=info["frames"],
        video_hasil=video_out.name,
        browser_playable=playable
    )

    csv_path.with_suffix(".json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    return dict(
        df=df,
        video_path=video_out,
        csv_path=csv_path,
        meta=meta
    )

    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(TRACKING_DIR.glob("*_tracks.csv"), key=lambda p: p.stat().st_mtime, reverse=True)


def load_tracks(csv_path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(csv_path)
    meta_p = Path(csv_path).with_suffix(".json")
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    return df, meta
