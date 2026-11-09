"""Tes logika inti MangoVision AI (tanpa menjalankan UI).

Jalankan dari folder project:   python tests/test_core.py
Jika streamlit/ultralytics belum terpasang, keduanya di-stub; model YOLO diganti objek tiruan,
sehingga tes ini HANYA memeriksa logika kode kita, bukan akurasi model YOLO.
"""
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import streamlit  # noqa: F401
except ImportError:
    st = types.ModuleType("streamlit")
    st.cache_resource = lambda *a, **k: (lambda f: f)
    sys.modules["streamlit"] = st

import cv2
import numpy as np
import pandas as pd

from modules import counting, database, evaluation, preprocessing
from modules import detection as det
from modules import realtime as rt
from modules import tracking as trk


def test_preprocessing():
    img = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    out, log = preprocessing.apply_pipeline(img, dict(resize=True, max_side=320, crop=True, crop_l=10, crop_r=10, crop_t=5, crop_b=5,
                                                      brightness=20, contrast=10, gray=False, hist_eq=True, clahe=True, clip=2.0,
                                                      tile=8, blur=True, blur_k=4))
    assert max(out.shape[:2]) == 320 and out.ndim == 3 and len(log) >= 5
    g, _ = preprocessing.apply_pipeline(img, dict(gray=True, hist_eq=True, clahe=True))
    assert g.ndim == 2
    try:
        preprocessing.crop_percent(img[:20, :20], 45, 45, 0, 0)
        raise AssertionError("harus error")
    except ValueError:
        pass
    print("OK preprocessing")


def test_counting():
    rows = []
    def add(fr, tid, cx, cy):
        rows.append((fr, tid, 0.8, cx - 20, cy - 20, cx + 20, cy + 20, cx, cy))
    for f in range(0, 21):  add(f, 1, 100, 100)          # buah A (bagian 1)
    for f in range(25, 41): add(f, 5, 104, 102)          # buah A diberi ID baru setelah tertutup
    for f in range(0, 41):  add(f, 2, 400, 300)          # buah B
    add(10, 9, 700, 50)                                   # false positive sesaat
    df = pd.DataFrame(rows, columns=trk.TRACK_COLUMNS)
    s = counting.summarize(df)
    assert s["jumlah_id"] == 4 and s["maks_per_frame"] == 3 - 0 or s["maks_per_frame"] >= 2
    e = counting.estimate_unique(df, min_len=5, max_gap=30, max_dist_ratio=1.0)
    assert e["batas_atas"] == 4 and e["id_terkonfirmasi"] == 3 and e["estimasi"] == 2, e
    e2 = counting.estimate_unique(df, min_len=5, max_gap=2, max_dist_ratio=1.0)  # celah terlalu kecil -> tidak digabung
    assert e2["estimasi"] == 3, e2
    assert not counting.tracklet_table(df).empty and not counting.ids_per_frame(df).empty
    empty = pd.DataFrame(columns=trk.TRACK_COLUMNS)
    assert counting.summarize(empty)["total_deteksi"] == 0
    assert counting.estimate_unique(empty)["estimasi"] == 0
    print("OK counting ->", {k: v for k, v in e.items() if k != "mapping"})


def test_evaluation():
    r = evaluation.count_error(100, 94)
    assert abs(r["error_pct"] - 6.0) < 1e-9 and r["abs_error"] == 6
    assert evaluation.count_error(0, 3)["error_pct"] is None
    agg = evaluation.aggregate_errors(pd.DataFrame({"aktual": [100, 50], "estimasi": [94, 55]}))
    assert abs(agg["mae"] - 5.5) < 1e-9 and abs(agg["rmse"] - np.sqrt((36 + 25) / 2)) < 1e-9
    gt = evaluation.parse_yolo_labels("0 0.5 0.5 0.2 0.2\n0 0.1 0.1 0.1 0.1", 100, 100)
    pred = np.array([[40, 40, 60, 60], [80, 80, 90, 90]], float)
    m = evaluation.match_detections(pred, np.array([0.9, 0.8]), gt, 0.5)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1), m
    prf = evaluation.precision_recall_f1(1, 1, 1)
    assert abs(prf["f1"] - 0.5) < 1e-9
    try:
        evaluation.parse_yolo_labels("0 0.5", 10, 10)
        raise AssertionError("harus error")
    except ValueError:
        pass
    print("OK evaluation")


def test_database():
    with tempfile.TemporaryDirectory() as d:
        database.DB_PATH = Path(d) / "t.db"
        database.init_db()
        database.add_analysis("a.jpg", "Deteksi Gambar", 10, 0.7)
        database.add_analysis("b.jpg", "Deteksi Gambar", 30, 0.5)
        database.add_analysis("v.mp4", "Analisis Video", 200, 0.6)
        st_ = database.summary_stats()
        assert st_["total_gambar"] == 2 and st_["total_video"] == 1 and st_["total_buah"] == 40, st_
        h = database.get_history("Deteksi Gambar")
        assert len(h) == 2
        assert database.delete_analysis([int(h.iloc[0]["id"])]) == 1
        database.add_eval_sample("P1", 100, 94)
        assert len(database.get_eval_samples()) == 1
        database.delete_eval_samples(); database.delete_all_analysis()
        assert database.get_history().empty
    print("OK database")


# ---------- objek tiruan YOLO ----------
class T:
    def __init__(self, a): self.a = np.asarray(a)
    def cpu(self): return self
    def numpy(self): return self.a
    def tolist(self): return self.a.tolist()
    def __getitem__(self, i): return T(self.a[i]) if isinstance(self.a[i], np.ndarray) else self.a[i]
    def __float__(self): return float(self.a)
    def __int__(self): return int(self.a)


class Box:
    def __init__(self, xyxy, conf, cls): self.xyxy, self.conf, self.cls = T([xyxy]), T([conf]), T([cls])


class Boxes:
    def __init__(self, items, with_id=True):
        self.items = items
        self.xyxy = T([i[0] for i in items]); self.conf = T([i[1] for i in items]); self.cls = T([i[2] for i in items])
        self.id = T([i[3] for i in items]) if with_id and items else None
    def __len__(self): return len(self.items)
    def __iter__(self): return iter(Box(*i[:3]) for i in self.items)


class Res:
    def __init__(self, img, boxes): self.orig_img, self.boxes = img, boxes


class FakeModel:
    names = {0: "mango"}
    predictor = "x"
    def predict(self, img, conf=0.5, imgsz=640, verbose=False):
        return [Res(img, Boxes([([50, 50, 120, 140], 0.9, 0, 1), ([200, 80, 260, 160], 0.6, 0, 2)], with_id=False))]
    def track(self, source, stream=True, vid_stride=1, **kw):
        if isinstance(source, np.ndarray):  # mode webcam: satu frame -> list hasil
            self.n = getattr(self, "n", 0) + 1
            return [Res(source, Boxes([([10 + self.n, 20, 60 + self.n, 80], 0.9, 0, 1), ([150, 50, 200, 110], 0.8, 0, 2),
                                       ([300, 60, 340, 100], 0.7, 0, 100 + self.n)]))]
        return self._track_video(source, vid_stride)

    def _track_video(self, source, vid_stride):
        cap = cv2.VideoCapture(source)
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok: break
            if i % vid_stride == 0:
                yield Res(fr, Boxes([([10 + i, 20, 60 + i, 80], 0.9, 0, 1), ([150, 50, 200, 110], 0.8, 0, 2)]))
            i += 1
        cap.release()


def test_detect_and_tracking():
    m = FakeModel()
    img = np.zeros((300, 400, 3), np.uint8)
    dets, out = det.detect_image(m, img, 0.5, 640)
    assert len(dets) == 2 and out.shape == img.shape and out.sum() > 0
    assert det.model_warning(m) is None
    with tempfile.TemporaryDirectory() as d:
        vp = str(Path(d) / "in.mp4")
        w = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
        for i in range(20):
            w.write(np.full((240, 320, 3), i * 5, np.uint8))
        w.release()
        trk.TRACKING_DIR = Path(d) / "trk"
        out = trk.process_video(m, vp, "in.mp4", 0.5, 640, frame_interval=2)
        df = out["df"]
        assert len(df) == 20 and df["track_id"].nunique() == 2 and df["frame"].max() == 18, (len(df), df["frame"].max())
        assert out["csv_path"].exists() and out["video_path"].exists() and out["video_path"].stat().st_size > 0
        df2, meta = trk.load_tracks(out["csv_path"])
        assert len(df2) == 20 and meta["frame_interval"] == 2
        cap = cv2.VideoCapture(str(out["video_path"])); assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 10; cap.release()
        print("OK tracking; H.264 playable =", meta["browser_playable"])


class FakeCap:
    def __init__(self, n): self.n, self.released = n, 0
    def read(self):
        if self.n <= 0: return False, None
        self.n -= 1
        return True, np.zeros((240, 320, 3), np.uint8)
    def release(self): self.released += 1


def test_realtime():
    m = FakeModel()
    rt.reset_tracker(m)
    assert m.predictor is None
    sess = rt.RealtimeSession(conf=0.5, imgsz=320, min_frames=5)
    meter, cap = rt.FpsMeter(), FakeCap(12)
    try:
        while True:
            ok, fr = cap.read()
            if not ok: break
            out = rt.process_live_frame(m, fr, sess, rt.allowed_classes(m), meter.tick())
            assert out.shape == fr.shape and out.sum() > 0
    finally:
        rt.release_camera(cap)
    rt.release_camera(cap); rt.release_camera(None)  # aman dipanggil berulang
    assert cap.released == 2
    assert sess.frames == 12 and sess.total_ids == 14      # ID 1, 2 + 12 ID sesaat (100+n)
    assert sess.confirmed == 2, sess.confirmed             # hanya ID yang stabil >= 5 frame
    assert sess.total_detections == 36 and sess.active_now == 3
    est = sess.final_estimate()
    assert est["estimasi"] == 2 and sess.final_estimate() is est   # hasil di-cache
    with tempfile.TemporaryDirectory() as d:
        rt.TRACKING_DIR = Path(d)
        p = sess.save()
        df, meta = trk.load_tracks(p)
        assert len(df) == 36 and meta["sumber"] == "webcam" and meta["frames_diproses"] == 12
    try:  # tanpa webcam (mis. sandbox) harus ada pesan error yang jelas, bukan crash
        c = rt.open_camera(99)
        rt.release_camera(c)
    except RuntimeError as e:
        assert "Kamera" in str(e)
    print("OK realtime")


def test_dataset_status_and_yaml():
    p = det.write_data_yaml()
    txt = p.read_text()
    assert "names:" in txt and "0: mango" in txt and "path:" in txt
    p.unlink()
    print("OK data.yaml")


if __name__ == "__main__":
    for fn in (test_preprocessing, test_counting, test_evaluation, test_database, test_detect_and_tracking, test_realtime, test_dataset_status_and_yaml):
        fn()
    print("\nSEMUA TES LOLOS")
