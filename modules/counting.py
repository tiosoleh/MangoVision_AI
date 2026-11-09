"""Penghitungan buah: ringkasan deteksi/tracking dan estimasi buah unik (bukan jumlah aktual)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return dict(total_deteksi=0, jumlah_id=0, frame=0, conf_rata=0.0, maks_per_frame=0, tanpa_id=0)
    tracked = df[df["track_id"] >= 0]
    per_frame = df.groupby("frame").size()
    return dict(
        total_deteksi=int(len(df)),
        jumlah_id=int(tracked["track_id"].nunique()),
        frame=int(df["frame"].nunique()),
        conf_rata=float(df["conf"].mean()),
        maks_per_frame=int(per_frame.max()),
        tanpa_id=int((df["track_id"] < 0).sum()),
    )


def ids_per_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Jumlah objek aktif (ID unik) tiap frame."""
    t = df[df["track_id"] >= 0]
    if t.empty:
        return pd.DataFrame({"frame": [], "id_aktif": []})
    return t.groupby("frame")["track_id"].nunique().rename("id_aktif").reset_index()


def tracklet_table(df: pd.DataFrame) -> pd.DataFrame:
    """Satu baris per tracking ID: frame awal/akhir, durasi, confidence."""
    t = df[df["track_id"] >= 0]
    if t.empty:
        return pd.DataFrame(columns=["track_id", "frame_awal", "frame_akhir", "jumlah_frame", "conf_rata"])
    g = t.sort_values("frame").groupby("track_id")
    out = g.agg(frame_awal=("frame", "min"), frame_akhir=("frame", "max"),
                jumlah_frame=("frame", "nunique"), conf_rata=("conf", "mean")).reset_index()
    return out.sort_values("track_id").reset_index(drop=True)


def merge_tracklets(df: pd.DataFrame, track_ids: list[int], max_gap: int, max_dist_ratio: float) -> dict[int, int]:
    """Gabungkan tracklet yang terputus (mis. buah tertutup daun sesaat).

    Aturan: tracklet B dianggap lanjutan A bila B mulai 1..max_gap frame setelah A berakhir
    dan posisi awal B dekat posisi akhir A (jarak ≤ max_dist_ratio × diagonal kotak).
    Pasangan optimal dipilih dengan Hungarian Algorithm. Mengembalikan map id -> id grup.
    """
    parent = {i: i for i in track_ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    info = {}
    for tid in track_ids:
        d = df[df["track_id"] == tid].sort_values("frame")
        first, last = d.iloc[0], d.iloc[-1]
        diag = float(np.hypot(d["x2"] - d["x1"], d["y2"] - d["y1"]).mean())
        info[tid] = dict(start=first["frame"], end=last["frame"],
                         sc=(first["cx"], first["cy"]), ec=(last["cx"], last["cy"]), diag=max(diag, 1.0))
    n = len(track_ids)
    if n < 2:
        return {i: i for i in track_ids}
    BIG = 1e6
    cost = np.full((n, n), BIG)
    for a, ia in enumerate(track_ids):
        for b, ib in enumerate(track_ids):
            if ia == ib:
                continue
            gap = info[ib]["start"] - info[ia]["end"]
            if 0 < gap <= max_gap:
                dist = float(np.hypot(info[ia]["ec"][0] - info[ib]["sc"][0], info[ia]["ec"][1] - info[ib]["sc"][1]))
                ratio = dist / ((info[ia]["diag"] + info[ib]["diag"]) / 2)
                if ratio <= max_dist_ratio:
                    cost[a, b] = ratio
    rows, cols = linear_sum_assignment(cost)
    for r, c in zip(rows, cols):
        if cost[r, c] < BIG:
            parent[find(track_ids[c])] = find(track_ids[r])
    return {i: find(i) for i in track_ids}


def estimate_unique(df: pd.DataFrame, min_len: int = 5, max_gap: int = 30, max_dist_ratio: float = 1.0) -> dict:
    """Estimasi jumlah buah unik dari hasil tracking.

    - batas_bawah  : objek terbanyak yang terlihat dalam satu frame (minimal ada sebanyak itu buah).
    - id_terkonfirmasi : ID yang muncul ≥ min_len frame (membuang ID 'sesaat' akibat false positive).
    - estimasi     : id_terkonfirmasi setelah tracklet yang terputus digabung.
    - batas_atas   : semua ID berbeda (cenderung over-count karena ID baru muncul saat buah tertutup/kamera berputar).
    Semua angka adalah ESTIMASI, bukan jumlah buah aktual.
    """
    s = summarize(df)
    tt = tracklet_table(df)
    confirmed = tt[tt["jumlah_frame"] >= min_len]["track_id"].tolist()
    mapping = merge_tracklets(df, confirmed, max_gap, max_dist_ratio) if confirmed else {}
    n_est = len(set(mapping.values())) if mapping else 0
    return dict(
        batas_bawah=s["maks_per_frame"],
        id_terkonfirmasi=len(confirmed),
        estimasi=n_est,
        batas_atas=s["jumlah_id"],
        id_dibuang=s["jumlah_id"] - len(confirmed),
        id_digabung=len(confirmed) - n_est,
        mapping=mapping,
    )
