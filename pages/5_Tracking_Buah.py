import streamlit as st

from modules import tracking as trk
from modules.counting import ids_per_frame, summarize, tracklet_table
from modules.utils import TRACKING_DIR, page_header

page_header("Tracking Buah", "Hasil pelacakan buah antarframe menggunakan ByteTrack", "🎯")

st.markdown("<div class='mv-card'><h4>Cara membaca hasil tracking</h4><p>ByteTrack menghubungkan deteksi antarframe dan memberi <b>tracking ID</b>. "
            "Namun ID hanya konsisten selama objek terlihat terus-menerus. Ketika buah tertutup daun, atau kamera berputar mengelilingi pohon, "
            "buah yang sama dapat memperoleh ID baru. Karena itu <b>jumlah tracking ID belum tentu jumlah buah unik</b>.</p></div>",
            unsafe_allow_html=True)

files = trk.list_track_files()
if not files:
    st.info("Belum ada hasil tracking. Proses video terlebih dahulu di menu **Analisis Video**.")
    st.stop()

names = [f.name for f in files]
default = names.index(st.session_state["track_csv_selected"]) if st.session_state.get("track_csv_selected") in names else 0
choice = st.selectbox("Pilih hasil tracking", names, index=default)
st.session_state["track_csv_selected"] = choice

try:
    df, meta = trk.load_tracks(TRACKING_DIR / choice)
except Exception as e:
    st.error(f"Gagal membaca file tracking: {e}")
    st.stop()

if df.empty:
    st.warning("Tidak ada deteksi pada video ini. Coba turunkan confidence threshold atau periksa model.")
    st.stop()

s = summarize(df)
active_now = ids_per_frame(df)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total deteksi", s["total_deteksi"])
c2.metric("Jumlah tracking ID", s["jumlah_id"])
c3.metric("ID aktif rata-rata/frame", f"{active_now['id_aktif'].mean():.1f}" if not active_now.empty else "0")
c4.metric("ID aktif maksimum", int(active_now["id_aktif"].max()) if not active_now.empty else 0)
if s["tanpa_id"]:
    st.caption(f"{s['tanpa_id']} deteksi belum memperoleh ID (tracker belum mengonfirmasi objek).")
if meta:
    st.caption(f"Video: {meta.get('video')} · interval frame {meta.get('frame_interval')} · confidence {meta.get('conf')} · {meta.get('frames_diproses')} frame diproses")

st.subheader("Grafik ID aktif per frame")
if active_now.empty:
    st.info("Belum ada ID tracking.")
else:
    st.line_chart(active_now.set_index("frame")["id_aktif"], color="#2d9d4b")

tab1, tab2, tab3 = st.tabs(["📋 Tabel per ID", "🧾 Semua deteksi", "🎬 Video hasil"])
with tab1:
    tt = tracklet_table(df)
    st.caption("Satu baris per tracking ID. ID dengan jumlah frame sangat kecil biasanya false positive atau objek yang hanya terlihat sesaat.")
    st.dataframe(tt.round({"conf_rata": 3}), use_container_width=True, hide_index=True)
with tab2:
    st.dataframe(df.round(2), use_container_width=True, hide_index=True)
    st.download_button("⬇️ Unduh CSV", (TRACKING_DIR / choice).read_bytes(), file_name=choice, mime="text/csv")
with tab3:
    vname = meta.get("video_hasil")
    vpath = TRACKING_DIR / vname if vname else None
    if vpath and vpath.exists():
        if meta.get("browser_playable"):
            st.video(str(vpath))
        else:
            st.info("Format video tidak dapat ditampilkan di browser; unduh untuk memutarnya.")
        st.download_button("⬇️ Unduh video", vpath.read_bytes(), file_name=vpath.name, mime="video/mp4")
    else:
        st.info("File video hasil tidak ditemukan.")
