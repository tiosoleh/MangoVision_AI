from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import threading

import time
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from modules import detection as det
from modules import realtime as rt
from modules.counting import tracklet_table
from modules.database import add_analysis
from modules.preprocessing import clahe, laplacian_sharpness
from modules.utils import (DETECTION_DIR, MODELS_DIR, bgr_to_rgb, encode_png, list_models, load_image_bgr,
                           model_selector, page_header, safe_stem, save_image)

page_header("Deteksi Buah Mangga", "Deteksi buah dari foto atau kamera langsung menggunakan YOLO", "🔍")
tab_det, tab_model, tab_cam = st.tabs(["🖼️ Upload Gambar", "🧠 Manajemen Model", "📷 Kamera Langsung"])

# =============================================================== TAB DETEKSI
with tab_det:
    model_path = model_selector("det_model")
    c1, c2, c3 = st.columns([2, 2, 1])
    conf = c1.slider("Confidence Threshold", 0.10, 0.90, 0.50, 0.05,
                     help="Makin tinggi: lebih sedikit deteksi salah, tetapi buah bisa terlewat.")
    imgsz = c2.slider("Image Size", 320, 1280, 640, 32, help="Ukuran input YOLO. Lebih besar = lebih detail tetapi lebih lambat.")
    use_clahe = c3.checkbox("CLAHE", value=False, help="Perbaiki kontras lokal sebelum deteksi (opsional).")

    up = st.file_uploader("Unggah foto pohon mangga", type=["jpg", "jpeg", "png"], key="det_upload")
    if up is not None:
        try:
            bgr = load_image_bgr(up)
        except Exception as e:
            st.error(f"Gambar tidak dapat dibaca: {e}")
            st.stop()
        h, w = bgr.shape[:2]
        st.image(bgr_to_rgb(bgr), caption=f"{up.name} — {w}×{h} px", use_container_width=True)
        sharp = laplacian_sharpness(bgr)
        if sharp < 60:
            st.warning(f"Foto tampak kurang tajam (skor ketajaman {sharp:.0f}). Foto blur menurunkan akurasi deteksi.")

        if st.button("🚀 Mulai Deteksi", type="primary", disabled=model_path is None):
            try:
                with st.spinner("Preprocessing & deteksi YOLO ..."):
                    model = det.load_model(model_path)
                    warn = det.model_warning(model)
                    work = clahe(bgr) if use_clahe else bgr
                    dets, annotated = det.detect_image(model, work, conf, imgsz)
                    saved = save_image(annotated, DETECTION_DIR, safe_stem(up.name))
                    avg = float(sum(d["conf"] for d in dets) / len(dets)) if dets else 0.0
                    add_analysis(up.name, "Deteksi Gambar", len(dets), avg, "Selesai",
                                 f"model={Path(model_path).name}; conf={conf}; imgsz={imgsz}; clahe={use_clahe}")
                st.session_state["det_result"] = dict(name=up.name, dets=dets, img=annotated, saved=str(saved),
                                                      avg=avg, warn=warn, conf=conf)
                st.session_state["last_image_count"] = len(dets)
            except Exception as e:
                st.error(f"Deteksi gagal: {e}")
                add_analysis(up.name, "Deteksi Gambar", 0, 0.0, "Gagal", str(e)[:200])

    res = st.session_state.get("det_result")
    if res and (up is None or res["name"] == up.name):
        st.markdown("---")
        st.subheader("Hasil Deteksi")
        if res["warn"]:
            st.warning(res["warn"])
        m1, m2, m3 = st.columns(3)
        m1.metric("Jumlah Deteksi (bounding box)", len(res["dets"]))
        m2.metric("Confidence Rata-rata", f"{res['avg']:.2f}")
        m3.metric("Threshold", f"{res['conf']:.2f}")
        st.image(bgr_to_rgb(res["img"]), caption="Hasil deteksi", use_container_width=True)
        if res["dets"]:
            table = pd.DataFrame(res["dets"]).round(2)
            table.insert(0, "No", range(1, len(table) + 1))
            st.dataframe(table[["No", "name", "conf", "x1", "y1", "x2", "y2"]].rename(columns={"name": "kelas", "conf": "confidence"}),
                         use_container_width=True, hide_index=True)
        else:
            st.info("Tidak ada buah terdeteksi. Coba turunkan threshold, naikkan Image Size, atau periksa model.")
        st.download_button("⬇️ Unduh Gambar Hasil", encode_png(res["img"]),
                           file_name=f"deteksi_{safe_stem(res['name'])}.png", mime="image/png", type="primary")
        st.caption(f"Tersimpan di: `{res['saved']}`")
        st.markdown("<div class='mv-card mv-warn'><h4>Ingat</h4><p>Ini jumlah <b>bounding box pada satu foto</b>, bukan jumlah buah di seluruh pohon. "
                    "Buah di sisi lain pohon atau tertutup daun tidak ikut terhitung.</p></div>", unsafe_allow_html=True)

# =============================================================== TAB MODEL
with tab_model:
    st.markdown("<div class='mv-card'><h4>Mengapa perlu model sendiri?</h4><p>Model COCO bawaan YOLO tidak punya kelas mangga. "
                "Gunakan <code>best.pt</code> hasil training pada dataset mangga yang dianotasi. "
                "Unggah hanya file <code>.pt</code> dari sumber tepercaya.</p></div>", unsafe_allow_html=True)

    st.subheader("1. Unggah model")
    mf = st.file_uploader("Unggah model YOLO (.pt)", type=["pt"], key="model_upload")
    as_best = st.checkbox("Simpan sebagai best.pt", value=True, key="as_best")
    if mf is not None and st.button("💾 Simpan ke folder models/"):
        try:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            target = MODELS_DIR / ("best.pt" if as_best else Path(mf.name).name)
            target.write_bytes(mf.getvalue())
            st.success(f"Model tersimpan: {target.name}")
        except Exception as e:
            st.error(f"Gagal menyimpan model: {e}")
    st.caption(f"Model yang ada: {', '.join(list_models()) or '— belum ada —'}")

    st.subheader("2. Status dataset")
    st.caption("Struktur: `dataset/images/{train,val,test}` dan `dataset/labels/{train,val,test}` (format YOLO, kelas 0 = mango).")
    ds = det.dataset_status()
    st.dataframe(pd.DataFrame(ds).T.rename_axis("split").reset_index(), use_container_width=True, hide_index=True)
    ready = ds["train"]["gambar"] > 0 and ds["val"]["gambar"] > 0
    if ds["train"]["tanpa_label"] or ds["val"]["tanpa_label"]:
        st.warning("Ada gambar tanpa file label (.txt). Gambar tanpa label dianggap tidak berisi buah.")
    if st.button("📝 Buat / perbarui dataset/data.yaml"):
        p = det.write_data_yaml()
        st.code(p.read_text(encoding="utf-8"), language="yaml")

    st.subheader("3. Training model")
    t1, t2, t3, t4 = st.columns(4)
    base = t1.selectbox("Bobot awal", ["yolov8n.pt", "yolov8s.pt", "yolo11n.pt", "yolo11s.pt"],
                        help="Diunduh otomatis saat pertama kali (butuh internet). Transfer learning dari COCO — setelah training, modelnya tetap model mangga.")
    epochs = t2.number_input("Epoch", 1, 500, 50)
    tsize = t3.select_slider("Image size", [320, 416, 512, 640, 800, 960], value=640)
    batch = t4.number_input("Batch", 1, 64, 8)
    d1, d2 = st.columns(2)
    run_name = d1.text_input("Nama model hasil", "mango_yolo")
    device = d2.selectbox("Perangkat", ["", "cpu", "0"], format_func=lambda x: {"": "Otomatis", "cpu": "CPU", "0": "GPU (CUDA 0)"}[x])

    proc = st.session_state.get("train_proc")
    running = proc is not None and proc.poll() is None
    if st.button("🏋️ Mulai Training", type="primary", disabled=running or not ready):
        try:
            st.session_state["train_proc"] = det.start_training(base, int(epochs), int(tsize), int(batch), run_name.strip() or "mango_yolo", device)
            st.rerun()
        except Exception as e:
            st.error(f"Tidak dapat memulai training: {e}")
    if not ready:
        st.caption("Tombol training aktif setelah ada gambar di `train` dan `val`.")
    if proc is not None:
        if running:
            st.info("⏳ Training sedang berjalan di proses terpisah. Klik *Segarkan log* untuk melihat progres.")
        else:
            st.success("Proses training selesai" if proc.returncode == 0 else f"Training berhenti dengan kode {proc.returncode}. Lihat log.")
        if st.button("🔄 Segarkan log"):
            st.rerun()
        st.code(det.read_train_log() or "(log kosong)", language="text")

    st.subheader("4. Validasi model")
    v_model = model_selector("val_model")
    v1, v2 = st.columns(2)
    v_split = v1.selectbox("Split", ["val", "test"])
    v_size = v2.select_slider("Image size validasi", [320, 416, 512, 640, 800, 960], value=640)
    if st.button("✅ Validasi Model", disabled=v_model is None):
        try:
            with st.spinner("Validasi berjalan ..."):
                m = det.validate_model(v_model, v_size, v_split)
            a, b, c, d = st.columns(4)
            a.metric("Precision", f"{m['precision']:.3f}")
            b.metric("Recall", f"{m['recall']:.3f}")
            c.metric("mAP@50", f"{m['map50']:.3f}")
            d.metric("mAP@50-95", f"{m['map50_95']:.3f}")
        except Exception as e:
            st.error(f"Validasi gagal: {e}")

# =============================================================== TAB KAMERA LANGSUNG
with tab_cam:
    st.markdown(
        """
        <div class='mv-card'>
            <h4>Deteksi Mangga melalui Kamera HP</h4>
            <p>Kamera HP digunakan langsung melalui browser. YOLO akan mendeteksi
            mangga dan ByteTrack memberikan ID untuk tracking buah.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rt_model_path = model_selector("mobile_rt_model")

    # Pengaturan deteksi
    p1, p2, p3 = st.columns(3)

    rt_conf = p1.slider(
        "Confidence",
        0.10, 0.90, 0.50, 0.05,
        key="mobile_rt_conf",
    )

    rt_imgsz = p2.select_slider(
        "Image Size",
        [256, 320, 416, 480, 640],
        value=416,
        key="mobile_rt_imgsz",
    )

    rt_min = p3.slider(
        "Konfirmasi ID (frame)",
        2, 30, 5,
        key="mobile_rt_min",
    )

    # Pilihan kamera depan / belakang
    camera_choice = st.radio(
        "Pilih Kamera",
        ["Kamera Belakang", "Kamera Depan"],
        horizontal=True,
        key="mobile_camera_choice",
    )

    facing_mode = (
        "environment"
        if camera_choice == "Kamera Belakang"
        else "user"
    )

    st.caption(
        "Kamera belakang disarankan untuk mendeteksi buah mangga di pohon."
    )

    # ===========================================================
    # VIDEO PROCESSOR
    # ===========================================================
    class MangoVideoProcessor:
        def __init__(self):
            self.model = det.load_model(rt_model_path)
            rt.reset_tracker(self.model)

            self.allowed = rt.allowed_classes(self.model)
            self.session = rt.RealtimeSession(
                rt_conf,
                rt_imgsz,
                rt_min,
            )

            self.meter = rt.FpsMeter()
            self.lock = threading.Lock()

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")

            fps = self.meter.tick()

            with self.lock:
                output = rt.process_live_frame(
                    self.model,
                    img,
                    self.session,
                    self.allowed,
                    fps,
                )

            return av.VideoFrame.from_ndarray(
                output,
                format="bgr24",
            )

    # ===========================================================
    # WEBRTC CAMERA
    # ===========================================================
    if rt_model_path is None:
        st.warning("Silakan pilih model YOLO terlebih dahulu.")
    else:
        model_warning = det.model_warning(
            det.load_model(rt_model_path)
        )

        if model_warning:
            st.warning(model_warning)

        st.info(
            f"Model aktif: {Path(rt_model_path).name}"
        )

        ctx = webrtc_streamer(
            key=f"mangovision_camera_{facing_mode}",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=MangoVideoProcessor,
            media_stream_constraints={
                "video": {
                    "facingMode": facing_mode,
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                },
                "audio": False,
            },
            rtc_configuration={
                "iceServers": [
                    {
                        "urls": [
                            "stun:stun.l.google.com:19302"
                        ]
                    }
                ]
            },
            async_processing=True,
        )

        if ctx.state.playing:
            st.success(
                f"Kamera aktif: {camera_choice}. "
                "Hasil deteksi ditampilkan pada video."
            )
        else:
            st.caption(
                "Tekan START pada kontrol video untuk memulai kamera. "
                "Tekan STOP untuk menghentikan kamera."
            )

        # =======================================================
        # RINGKASAN SESI KAMERA
        # =======================================================
        processor = ctx.video_processor

        if processor is not None:
            sess = processor.session

            st.markdown("---")
            st.subheader("Hasil Sesi Kamera")

            k1, k2, k3, k4 = st.columns(4)

            k1.metric(
                "Mangga unik",
                sess.confirmed,
            )

            k2.metric(
                "Terlihat sekarang",
                sess.active_now,
            )

            k3.metric(
                "Total ID tracking",
                sess.total_ids,
            )

            k4.metric(
                "Durasi",
                f"{sess.duration:.0f} detik",
            )

            df_rt = sess.to_df()

            with st.expander("Tabel per ID tracking"):
                if not df_rt.empty:
                    st.dataframe(
                        tracklet_table(df_rt).round(
                            {"conf_rata": 3}
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("Belum ada data tracking.")

            if not df_rt.empty:
                st.download_button(
                    "Unduh Data Tracking CSV",
                    df_rt.to_csv(index=False).encode("utf-8"),
                    file_name="mangovision_mobile_tracking.csv",
                    mime="text/csv",
                    key="mobile_rt_download",
                )

            if st.button(
                "Simpan ke Riwayat MangoVision AI",
                key="mobile_rt_save",
            ):
                try:
                    csv_path = sess.save(
                        name="Kamera HP MangoVision AI"
                    )

                    avg = (
                        float(df_rt["conf"].mean())
                        if not df_rt.empty
                        else 0.0
                    )

                    add_analysis(
                        "Kamera HP",
                        "Kamera Langsung",
                        sess.total_detections,
                        avg,
                        "Selesai",
                        (
                            f"mangga_unik={sess.confirmed}; "
                            f"id_total={sess.total_ids}; "
                            f"frame={sess.frames}; "
                            f"conf={sess.conf}; "
                            f"csv={csv_path.name}"
                        ),
                    )

                    st.session_state[
                        "track_csv_selected"
                    ] = csv_path.name

                    st.success(
                        f"Data tersimpan: {csv_path.name}"
                    )

                except Exception as e:
                    st.error(
                        f"Gagal menyimpan sesi kamera: {e}"
                    )

            st.markdown(
                """
                <div class='mv-card mv-warn'>
                    <h4>Catatan Penghitungan</h4>
                    <p>Jumlah mangga unik merupakan estimasi dari tracking.
                    Buah yang tertutup daun atau tidak terlihat kamera
                    tidak ikut dihitung.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("<div class='mv-card'><h4>Deteksi real-time dengan webcam</h4><p>Webcam dibaca oleh komputer yang menjalankan aplikasi ini "
                "(jalankan <code>streamlit run app.py</code> di laptop Anda). YOLO mendeteksi buah, <b>ByteTrack</b> memberi ID agar buah yang sama "
                "tidak dihitung berulang. <b>Atur parameter sebelum menekan Mulai</b>: mengubah pengaturan saat kamera aktif akan menghentikan kamera.</p></div>",
                unsafe_allow_html=True)
    rt_model_path = model_selector("rt_model")

    p1, p2, p3, p4, p5 = st.columns(5)
    rt_conf = p1.slider("Confidence", 0.10, 0.90, 0.50, 0.05, key="rt_conf")
    rt_imgsz = p2.select_slider("Image Size", [256, 320, 416, 480, 640], value=416, key="rt_imgsz",
                                help="Lebih kecil = FPS lebih tinggi (penting di CPU).")
    rt_min = p3.slider("Konfirmasi ID (frame)", 2, 30, 5, key="rt_min",
                       help="Sebuah ID dihitung sebagai mangga unik bila terlihat minimal sebanyak ini frame. Mengurangi hitungan palsu dari deteksi sesaat.")
    rt_cam = p4.number_input("Indeks kamera", 0, 9, 0, key="rt_cam", help="0 = webcam bawaan laptop; coba 1 atau 2 untuk kamera eksternal.")
    rt_res = p5.selectbox("Resolusi kamera", ["640×480", "1280×720"], key="rt_res")

    b1, b2 = st.columns(2)
    start = b1.button("▶️ Mulai Kamera", type="primary", disabled=rt_model_path is None, use_container_width=True, key="rt_start")
    b2.button("⏹️ Hentikan Kamera", use_container_width=True, key="rt_stop",
              help="Menghentikan proses dan melepaskan kamera.")

    status = st.empty()
    frame_ph = st.empty()
    mcols = st.columns(5)
    mph = [c.empty() for c in mcols]

    def show_metrics(sess, fps):
        mph[0].metric("🥭 Mangga unik", sess.confirmed)
        mph[1].metric("👁️ Terlihat sekarang", sess.active_now)
        mph[2].metric("🏷️ Total ID tracking", sess.total_ids)
        mph[3].metric("⚡ FPS", f"{fps:.1f}")
        mph[4].metric("🎞️ Frame", sess.frames)

    if start:
        cap = None
        try:
            model = det.load_model(rt_model_path)
            warn = det.model_warning(model)
            if warn:
                st.warning(warn)
            rt.reset_tracker(model)
            allowed = rt.allowed_classes(model)
            sess = rt.RealtimeSession(rt_conf, rt_imgsz, rt_min)
            st.session_state["rt_session"] = sess
            w_, h_ = map(int, rt_res.replace("×", "x").split("x"))
            with st.spinner("Membuka kamera ..."):
                cap = rt.open_camera(int(rt_cam), w_, h_)
            status.success("🟢 Kamera aktif. Klik **Hentikan Kamera** untuk berhenti dan melepaskan kamera.")
            meter, last_ui = rt.FpsMeter(), 0.0
            while True:
                ok, frame = cap.read()
                if not ok:
                    sess.error = "Gagal membaca frame dari kamera (kamera terlepas atau dipakai aplikasi lain)."
                    status.error(sess.error)
                    break
                fps = meter.tick()
                out = rt.process_live_frame(model, frame, sess, allowed, fps)
                if out.shape[1] > 960:  # perkecil tampilan agar UI ringan
                    out = cv2.resize(out, (960, int(out.shape[0] * 960 / out.shape[1])))
                frame_ph.image(bgr_to_rgb(out), channels="RGB", output_format="JPEG", use_container_width=True)
                now = time.perf_counter()
                if now - last_ui > 0.3:
                    show_metrics(sess, fps)
                    last_ui = now
        except RuntimeError as e:
            status.error(str(e))
        except Exception as e:
            status.error(f"Kamera berhenti karena kesalahan: {e}")
        finally:
            rt.release_camera(cap)  # selalu dilepas, termasuk saat tombol Hentikan ditekan

    # ---------- ringkasan sesi terakhir (tampil setelah kamera dihentikan)
    sess = st.session_state.get("rt_session")
    if sess is not None and not start:
        st.markdown("---")
        st.subheader("Hasil Sesi Kamera Terakhir")
        st.caption("🔴 Kamera tidak aktif (sudah dilepas).")
        if sess.frames == 0:
            st.info("Belum ada frame yang diproses pada sesi ini.")
        else:
            est = sess.final_estimate()
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Mangga unik (real-time)", sess.confirmed, help=f"ID yang terlihat ≥ {sess.min_frames} frame.")
            k2.metric("Estimasi setelah penggabungan", est["estimasi"], help="ID terkonfirmasi setelah tracklet terputus yang berdekatan digabung.")
            k3.metric("Total ID tracking", sess.total_ids)
            k4.metric("Durasi", f"{sess.duration:.0f} dtk")
            st.caption(f"{sess.frames} frame diproses · rata-rata {sess.frames / max(sess.duration, 1e-6):.1f} frame/detik · "
                       f"{est['id_digabung']} ID digabung sebagai lanjutan ID lain · {est['id_dibuang']} ID dibuang karena terlalu singkat.")
            df_rt = sess.to_df()
            with st.expander("Tabel per ID tracking"):
                st.dataframe(tracklet_table(df_rt).round({"conf_rata": 3}), use_container_width=True, hide_index=True)
            d1, d2 = st.columns(2)
            d1.download_button("⬇️ Unduh data tracking (CSV)", df_rt.to_csv(index=False).encode("utf-8"),
                               file_name=f"kamera_tracking_{int(sess.started)}.csv", mime="text/csv", key="rt_dl")
            if d2.button("💾 Simpan ke riwayat & menu Tracking/Penghitungan", key="rt_save"):
                try:
                    csv_path = sess.save()
                    avg = float(df_rt["conf"].mean()) if not df_rt.empty else 0.0
                    add_analysis("Kamera Langsung (webcam)", "Kamera Langsung", sess.total_detections, avg, "Selesai",
                                 f"mangga_unik={sess.confirmed}; estimasi={est['estimasi']}; id_total={sess.total_ids}; "
                                 f"frame={sess.frames}; conf={sess.conf}; csv={csv_path.name}")
                    st.session_state["track_csv_selected"] = csv_path.name
                    st.success(f"Tersimpan: output/tracking/{csv_path.name}. Buka menu Tracking Buah atau Penghitungan Buah.")
                except Exception as e:
                    st.error(f"Gagal menyimpan: {e}")
        st.markdown("<div class='mv-card mv-warn'><h4>Ingat</h4><p>\"Mangga unik\" adalah <b>estimasi</b> berbasis tracking. Bila kamera bergeser jauh atau buah "
                    "tertutup lama, ByteTrack dapat memberi ID baru sehingga hitungan berlebih; buah yang tidak pernah terlihat tidak ikut terhitung. "
                    "Bandingkan selalu dengan hitungan manual di menu Evaluasi Akurasi.</p></div>", unsafe_allow_html=True)
