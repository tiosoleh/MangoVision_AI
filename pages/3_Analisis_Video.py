import streamlit as st

from modules import detection as det
from modules import tracking as trk
from modules.counting import summarize
from modules.database import add_analysis
from modules.utils import VIDEO_INPUT_DIR, model_selector, page_header, safe_stem, timestamp

page_header("Analisis Video", "Deteksi + tracking buah mangga pada video dengan YOLO dan ByteTrack", "🎬")

model_path = model_selector("vid_model")
up = st.file_uploader("Unggah video pohon mangga", type=["mp4", "avi", "mov"], key="vid_upload")

c1, c2, c3, c4 = st.columns(4)
conf = c1.slider("Confidence Threshold", 0.10, 0.90, 0.50, 0.05)
interval = c2.slider("Frame Interval", 1, 10, 2, 1, help="2 = proses tiap 2 frame. Makin besar makin cepat, tetapi tracking bisa putus.")
imgsz = c3.select_slider("Image Size", [320, 416, 512, 640, 800, 960, 1280], value=640)
max_frames = c4.number_input("Batas frame (0 = semua)", 0, 100000, 0, 50, help="Gunakan batas kecil dulu untuk uji coba.")

if up is not None:
    VIDEO_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    in_path = VIDEO_INPUT_DIR / f"{timestamp()}_{safe_stem(up.name)}{up.name[up.name.rfind('.'):]}"
    in_path.write_bytes(up.getvalue())
    try:
        info = trk.video_info(str(in_path))
    except Exception as e:
        st.error(str(e))
        st.stop()
    st.subheader("Video Asli")
    st.video(up.getvalue())
    dur = info["frames"] / info["fps"] if info["fps"] else 0
    st.caption(f"{info['width']}×{info['height']} px · {info['fps']:.1f} fps · {info['frames']} frame · ±{dur:.1f} detik")
    n_proc = (info["frames"] // interval) if not max_frames else min(max_frames, info["frames"] // interval)
    st.caption(f"Frame yang akan dianalisis: ±{n_proc}. Di CPU, 1 frame bisa memakan 0,1–1 detik; untuk video panjang gunakan interval besar atau batas frame.")

    if st.button("🚀 Mulai Analisis Video", type="primary", disabled=model_path is None):
        bar, txt = st.progress(0.0), st.empty()

        def cb(frac, msg):
            bar.progress(min(1.0, frac))
            txt.caption(msg)

        try:
            model = det.load_model(model_path)
            warn = det.model_warning(model)
            if warn:
                st.warning(warn)
            out = trk.process_video(model, str(in_path), up.name, conf, imgsz, interval, int(max_frames), cb)
            s = summarize(out["df"])
            add_analysis(up.name, "Analisis Video", s["total_deteksi"], s["conf_rata"], "Selesai",
                         f"id_tracking={s['jumlah_id']}; frame={s['frame']}; interval={interval}; conf={conf}; csv={out['csv_path'].name}")
            st.session_state["video_result"] = dict(out=out, summary=s, name=up.name)
            st.session_state["track_csv_selected"] = out["csv_path"].name
            bar.progress(1.0)
            txt.caption("Selesai.")
        except Exception as e:
            st.error(f"Analisis video gagal: {e}")
            add_analysis(up.name, "Analisis Video", 0, 0.0, "Gagal", str(e)[:200])

res = st.session_state.get("video_result")
if res:
    st.markdown("---")
    st.subheader(f"Hasil: {res['name']}")
    s, out = res["summary"], res["out"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Frame dianalisis", s["frame"])
    m2.metric("Total deteksi", s["total_deteksi"])
    m3.metric("Jumlah tracking ID", s["jumlah_id"])
    m4.metric("Confidence rata-rata", f"{s['conf_rata']:.2f}")
    vp = out["video_path"]
    if out["meta"]["browser_playable"]:
        st.video(str(vp))
    else:
        st.warning("Video hasil memakai codec yang mungkin tidak dapat diputar di browser (ffmpeg tidak tersedia). Silakan unduh dan putar di komputer.")
    st.download_button("⬇️ Unduh Video Hasil", vp.read_bytes(), file_name=vp.name, mime="video/mp4", type="primary")
    st.download_button("⬇️ Unduh Tabel Tracking (CSV)", out["csv_path"].read_bytes(), file_name=out["csv_path"].name, mime="text/csv")
    st.caption(f"Tersimpan di `output/tracking/`. Lanjutkan ke menu **Tracking Buah** dan **Penghitungan Buah**.")
    st.markdown("<div class='mv-card mv-warn'><h4>Jumlah tracking ID bukan jumlah buah</h4><p>Saat buah tertutup daun atau kamera berputar, buah yang sama dapat memperoleh ID baru. "
                "Gunakan menu <b>Penghitungan Buah</b> untuk estimasi yang lebih realistis.</p></div>", unsafe_allow_html=True)
