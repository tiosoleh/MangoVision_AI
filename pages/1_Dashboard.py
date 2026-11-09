import streamlit as st

from modules.database import get_history, summary_stats
from modules.utils import APP_NAME, APP_SUBTITLE, list_models, page_header

page_header(APP_NAME, APP_SUBTITLE)

stats = summary_stats()
c1, c2, c3, c4 = st.columns(4)
c1.metric("🖼️ Gambar Dianalisis", stats["total_gambar"])
c2.metric("🥭 Deteksi Buah (gambar)", stats["total_buah"])
c3.metric("🎬 Video Dianalisis", stats["total_video"])
c4.metric("🎯 Rata-rata Confidence", f"{stats['conf_rata']:.2f}")

models = list_models()
if not models:
    st.warning("Belum ada model YOLO di folder `models/`. Buka **Deteksi Buah Mangga → Manajemen Model** "
               "untuk mengunggah `best.pt` atau melatih model dari dataset mangga.")
else:
    st.success(f"Model tersedia: {', '.join(models)}")

st.markdown("### 📈 Grafik Hasil Deteksi")
hist = get_history()
det = hist[hist["jenis"].isin(["Deteksi Gambar", "Analisis Video"])].head(15).iloc[::-1]
if det.empty:
    st.info("Belum ada data analisis. Lakukan deteksi gambar atau analisis video terlebih dahulu.")
else:
    chart = det.assign(label=det["id"].astype(str) + " · " + det["nama_file"].str.slice(0, 18))
    left, right = st.columns([3, 2])
    with left:
        st.caption("Jumlah deteksi (bounding box) pada 15 analisis terakhir")
        st.bar_chart(chart.set_index("label")["jumlah_deteksi"], color="#2d9d4b")
    with right:
        st.caption("Confidence rata-rata")
        st.line_chart(chart.set_index("label")["confidence_rata"], color="#1b4332")

st.markdown("### 🕒 Riwayat Analisis Terbaru")
recent = hist.head(8)
if recent.empty:
    st.caption("Belum ada riwayat.")
else:
    st.dataframe(recent.drop(columns=["keterangan"]), use_container_width=True, hide_index=True)

st.markdown("### 📘 Panduan Penggunaan")
g1, g2 = st.columns(2)
with g1:
    st.markdown("""
**Alur untuk foto**
1. Siapkan model `best.pt` (menu *Deteksi Buah Mangga → Manajemen Model*).
2. Unggah foto pohon di menu *Deteksi Buah Mangga*, atur threshold, lalu **Mulai Deteksi**.
3. Unduh gambar hasil dan catat jumlahnya.
""")
with g2:
    st.markdown("""
**Alur untuk video**
1. Rekam video sambil berjalan mengelilingi pohon.
2. Proses di *Analisis Video* (ByteTrack).
3. Lihat *Tracking Buah* → *Penghitungan Buah* → bandingkan dengan hitungan manual di *Evaluasi Akurasi*.
""")
st.markdown("<div class='mv-card mv-warn'><h4>⚠️ Catatan penting</h4><p>Jumlah bounding box ≠ jumlah tracking ID ≠ jumlah buah aktual. "
            "Semua angka di aplikasi ini adalah <b>estimasi</b> yang harus divalidasi dengan penghitungan manual pada pohon yang sama.</p></div>",
            unsafe_allow_html=True)
