import pandas as pd
import streamlit as st

from modules import tracking as trk
from modules.counting import estimate_unique, summarize, tracklet_table
from modules.database import add_analysis
from modules.utils import TRACKING_DIR, page_header

page_header("Penghitungan Buah", "Dari bounding box dan tracking ID menuju estimasi buah unik", "🔢")

st.markdown("<div class='mv-card mv-warn'><h4>Bounding box ≠ tracking ID ≠ buah aktual</h4><p>Angka di halaman ini adalah <b>estimasi</b> berbasis tracking satu video, "
            "bukan identifikasi buah unik multi-view yang akurat. Validasi selalu dengan penghitungan manual pada pohon yang sama (menu Evaluasi Akurasi).</p></div>",
            unsafe_allow_html=True)

files = trk.list_track_files()
if not files:
    st.info("Belum ada hasil tracking. Proses video di menu **Analisis Video** terlebih dahulu.")
    last = st.session_state.get("last_image_count")
    if last is not None:
        st.metric("Deteksi pada gambar terakhir (satu foto)", last)
    st.stop()

names = [f.name for f in files]
default = names.index(st.session_state["track_csv_selected"]) if st.session_state.get("track_csv_selected") in names else 0
choice = st.selectbox("Pilih hasil tracking", names, index=default)
st.session_state["track_csv_selected"] = choice
df, meta = trk.load_tracks(TRACKING_DIR / choice)
if df.empty:
    st.warning("Tidak ada deteksi pada data ini.")
    st.stop()

st.subheader("Parameter estimasi")
p1, p2, p3 = st.columns(3)
min_len = p1.slider("Minimal frame per ID", 1, 30, 5, help="ID yang muncul kurang dari ini dibuang (dianggap false positive/sesaat).")
max_gap = p2.slider("Celah maks. penggabungan (frame video)", 1, 150, 30,
                    help="Dua tracklet boleh digabung jika jeda munculnya ≤ nilai ini (dalam nomor frame video asli).")
ratio = p3.slider("Jarak maks. (× ukuran kotak)", 0.2, 3.0, 1.0, 0.1, help="Posisi akhir tracklet A dan awal tracklet B harus dekat.")

s = summarize(df)
e = estimate_unique(df, min_len, max_gap, ratio)

st.subheader("Hasil")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total deteksi", s["total_deteksi"], help="Jumlah semua bounding box di seluruh frame (banyak duplikat).")
c2.metric("Jumlah tracking ID", s["jumlah_id"], help="Semua ID berbeda dari ByteTrack.")
c3.metric("Estimasi buah unik", e["estimasi"], help="ID terkonfirmasi setelah tracklet terputus digabung.")
c4.metric("Batas bawah", e["batas_bawah"], help="Objek terbanyak dalam satu frame.")

comp = pd.DataFrame({
    "Ukuran": ["Batas bawah (maks/frame)", "Estimasi buah unik", "ID terkonfirmasi", "Semua tracking ID", "Total deteksi"],
    "Jumlah": [e["batas_bawah"], e["estimasi"], e["id_terkonfirmasi"], e["batas_atas"], s["total_deteksi"]],
})
st.subheader("Perbandingan deteksi vs tracking vs estimasi")
st.bar_chart(comp.set_index("Ukuran")["Jumlah"], color="#2d9d4b")
st.dataframe(comp, use_container_width=True, hide_index=True)

st.caption(f"Rincian: {e['id_dibuang']} ID dibuang karena muncul < {min_len} frame; {e['id_digabung']} ID digabung sebagai lanjutan tracklet lain.")

with st.expander("Detail tracklet dan grup hasil penggabungan"):
    tt = tracklet_table(df)
    tt = tt[tt["jumlah_frame"] >= min_len].copy()
    tt["grup_buah"] = tt["track_id"].map(e["mapping"])
    st.dataframe(tt.round({"conf_rata": 3}), use_container_width=True, hide_index=True)

st.markdown("""
#### Cara kerja dan keterbatasan
- **Batas bawah**: jumlah objek terbanyak yang terlihat dalam satu frame. Buah yang hanya terlihat dari sisi lain pohon tidak ikut terhitung.
- **Estimasi**: ID yang muncul cukup lama dipertahankan, lalu tracklet yang terputus digabung bila posisinya berdekatan dan jedanya singkat (Hungarian Algorithm). Ini hanya menangani **tertutup sesaat**.
- **Tidak ditangani**: buah yang sama tampil di bagian video yang berjauhan (kamera sudah memutar ke sisi lain). Untuk itu dibutuhkan *multi-view association* (SfM/COLMAP, epipolar geometry, triangulasi 3D) yang **belum diimplementasikan** di aplikasi ini.
- Akibatnya hasil dapat **kurang hitung** (buah di sisi tidak terekam/tersembunyi) maupun **lebih hitung** (buah yang sama diberi ID berbeda).
- Pilih parameter dengan membandingkan beberapa nilai terhadap hitungan manual, bukan dengan mencari angka yang tampak bagus pada satu pohon.
""")

b1, b2 = st.columns(2)
if b1.button("➡️ Gunakan estimasi ini di Evaluasi Akurasi", type="primary"):
    st.session_state["estimasi_terakhir"] = dict(sampel=meta.get("video", choice), estimasi=int(e["estimasi"]))
    st.success("Estimasi dikirim ke halaman Evaluasi Akurasi.")
if b2.button("💾 Simpan ke riwayat"):
    add_analysis(meta.get("video", choice), "Penghitungan", e["estimasi"], s["conf_rata"], "Selesai",
                 f"batas_bawah={e['batas_bawah']}; id_total={s['jumlah_id']}; min_len={min_len}; gap={max_gap}; ratio={ratio}")
    st.success("Tersimpan di riwayat.")
