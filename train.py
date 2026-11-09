
"""
Training YOLO untuk deteksi mangga.
Optimasi untuk laptop Windows dengan RAM 12 GB.

Contoh:
    python train.py
    python train.py --epochs 100 --imgsz 640 --batch 4
    python train.py --device cpu
"""

import argparse
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Training YOLO MangoVision AI"
    )

    ap.add_argument(
        "--data",
        default=str(BASE / "dataset" / "data.yaml")
    )

    ap.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Model awal YOLO"
    )

    ap.add_argument(
        "--epochs",
        type=int,
        default=100
    )

    ap.add_argument(
        "--imgsz",
        type=int,
        default=640
    )

    ap.add_argument(
        "--batch",
        type=int,
        default=4
    )

    ap.add_argument(
        "--name",
        default="mango_yolo_improved"
    )

    ap.add_argument(
        "--device",
        default="",
        help="Kosong=otomatis, cpu=CPU, 0=GPU"
    )

    a = ap.parse_args()

    from ultralytics import YOLO

    data_path = Path(a.data)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset YAML tidak ditemukan: {data_path}"
        )

    model = YOLO(a.model)

    kwargs = {
        "data": str(data_path),
        "epochs": a.epochs,
        "imgsz": a.imgsz,
        "batch": a.batch,

        # Penyimpanan hasil training
        "project": str(BASE / "runs"),
        "name": a.name,
        "exist_ok": True,

        # Optimasi untuk Windows
        "workers": 0,
        "cache": False,

        # Optimasi training
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "cos_lr": True,
        "warmup_epochs": 3,

        # Early stopping
        "patience": 25,

        # Augmentasi gambar
        "degrees": 15.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 2.0,
        "perspective": 0.0005,

        "fliplr": 0.5,
        "flipud": 0.1,

        # Variasi warna dan pencahayaan
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,

        # Augmentasi objek
        "mosaic": 0.7,
        "mixup": 0.05,
        "close_mosaic": 10,

        # Hasil evaluasi
        "val": True,
        "plots": True,
        "save": True,
        "save_period": 10,

        # Reproducibility
        "seed": 42,
        "deterministic": True,
    }

    if a.device:
        kwargs["device"] = a.device

    print("=" * 60)
    print("MANGOVISION AI - TRAINING MODEL")
    print("=" * 60)
    print(f"Model       : {a.model}")
    print(f"Dataset     : {data_path}")
    print(f"Epoch       : {a.epochs}")
    print(f"Image Size  : {a.imgsz}")
    print(f"Batch       : {a.batch}")
    print(f"Device      : {a.device if a.device else 'Otomatis'}")
    print("=" * 60)

    model.train(**kwargs)

        # Lokasi bobot model terbaik
    best = (
        BASE
        / "runs"
        / a.name
        / "weights"
        / "best.pt"
    )

    if best.exists():
        dst = BASE / "models" / "mango_yolo_v2.pt"

        dst.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.copy2(best, dst)

        print("\n" + "=" * 60)
        print("[SELESAI] Training berhasil")
        print(f"Bobot terbaik : {best}")
        print(f"Model baru    : {dst}")
        print("=" * 60)

    else:
        print(
            "[PERINGATAN] best.pt tidak ditemukan. "
            "Periksa log training."
        )


if __name__ == "__main__":
    main()