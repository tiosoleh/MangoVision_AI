"""Penyimpanan riwayat analisis dan sampel evaluasi dengan SQLite."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime

import pandas as pd

from modules.utils import DB_PATH


@contextmanager
def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    with _conn() as con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS analisis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nama_file TEXT NOT NULL,
                tanggal TEXT NOT NULL,
                jenis TEXT NOT NULL,
                jumlah_deteksi INTEGER DEFAULT 0,
                confidence_rata REAL DEFAULT 0,
                status TEXT DEFAULT 'Selesai',
                keterangan TEXT DEFAULT ''
            )"""
        )
        con.execute(
            """CREATE TABLE IF NOT EXISTS evaluasi (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sampel TEXT NOT NULL,
                aktual INTEGER NOT NULL,
                estimasi INTEGER NOT NULL,
                tanggal TEXT NOT NULL
            )"""
        )


def add_analysis(nama_file: str, jenis: str, jumlah_deteksi: int = 0,
                 confidence_rata: float = 0.0, status: str = "Selesai", keterangan: str = "") -> int:
    init_db()
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO analisis (nama_file, tanggal, jenis, jumlah_deteksi, confidence_rata, status, keterangan)"
            " VALUES (?,?,?,?,?,?,?)",
            (nama_file, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), jenis, int(jumlah_deteksi),
             float(confidence_rata), status, keterangan),
        )
        return cur.lastrowid


def get_history(jenis: str | None = None, limit: int | None = None) -> pd.DataFrame:
    init_db()
    q = "SELECT id, nama_file, tanggal, jenis, jumlah_deteksi, confidence_rata, status, keterangan FROM analisis"
    params: list = []
    if jenis and jenis != "Semua":
        q += " WHERE jenis = ?"
        params.append(jenis)
    q += " ORDER BY id DESC"
    if limit:
        q += f" LIMIT {int(limit)}"
    with _conn() as con:
        return pd.read_sql_query(q, con, params=params)


def delete_analysis(ids: list[int]) -> int:
    if not ids:
        return 0
    init_db()
    with _conn() as con:
        cur = con.executemany("DELETE FROM analisis WHERE id = ?", [(int(i),) for i in ids])
        return cur.rowcount


def delete_all_analysis() -> None:
    init_db()
    with _conn() as con:
        con.execute("DELETE FROM analisis")


def summary_stats() -> dict:
    """Ringkasan untuk dashboard."""
    df = get_history()
    if df.empty:
        return dict(total_gambar=0, total_video=0, total_buah=0, conf_rata=0.0)
    img = df[df["jenis"] == "Deteksi Gambar"]
    vid = df[df["jenis"] == "Analisis Video"]
    det = df[df["jenis"].isin(["Deteksi Gambar", "Analisis Video"]) & (df["jumlah_deteksi"] > 0)]
    conf = 0.0
    if not det.empty:
        w = det["jumlah_deteksi"]
        conf = float((det["confidence_rata"] * w).sum() / w.sum())
    return dict(
        total_gambar=len(img),
        total_video=len(vid),
        total_buah=int(img["jumlah_deteksi"].sum()),  # hanya deteksi gambar (bukan jumlah tracking)
        conf_rata=conf,
    )


# ---- sampel evaluasi ----
def add_eval_sample(sampel: str, aktual: int, estimasi: int) -> None:
    init_db()
    with _conn() as con:
        con.execute(
            "INSERT INTO evaluasi (sampel, aktual, estimasi, tanggal) VALUES (?,?,?,?)",
            (sampel, int(aktual), int(estimasi), datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )


def get_eval_samples() -> pd.DataFrame:
    init_db()
    with _conn() as con:
        return pd.read_sql_query("SELECT id, sampel, aktual, estimasi, tanggal FROM evaluasi ORDER BY id", con)


def delete_eval_samples() -> None:
    init_db()
    with _conn() as con:
        con.execute("DELETE FROM evaluasi")
