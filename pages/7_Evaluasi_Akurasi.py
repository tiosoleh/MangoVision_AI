import cv2
import numpy as np
import pandas as pd
import streamlit as st

from modules import detection as det
from modules import evaluation as ev
from modules.database import add_eval_sample, delete_eval_samples, get_eval_samples
from modules.utils import bgr_to_rgb, load_image_bgr, model_selector, page_header

page_header("Evaluasi Akurasi", "Bandingkan hasil sistem dengan penghitungan manual", "📏")
st.markdown("<div class='mv-card mv-warn'><h4>Gunakan data nyata</h4><p>Halaman ini tidak menyediakan data contoh. Isi dengan hasil penghitungan manual Anda "
            "dan estimasi sistem pada pohon yang sama. Untuk MAE/RMSE yang bermakna, uji beberapa pohon.</p></div>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📐 Error Penghitungan (MAE, RMSE, MAPE)", "🎯 Precision, Recall, F1 (ground truth)"])

# ===================================================== TAB 1
with tab1:
    pre = st.session_state.get("estimasi_terakhir", {})
    c1, c2, c3 = st.columns([2, 1, 1])
    sampel = c1.text_input("Nama sampel (mis. Pohon 1)", value=pre.get("sampel", ""))
    aktual = c2.number_input("Jumlah buah aktual (manual)", min_value=0, value=0, step=1)
    estimasi = c3.number_input("Estimasi sistem", min_value=0, value=int(pre.get("estimasi", 0)), step=1)

    r = ev.count_error(aktual, estimasi)
    m1, m2, m3 = st.columns(3)
    m1.metric("Absolute Error", f"{r['abs_error']:.0f}")
    m2.metric("Error (%)", f"{r['error_pct']:.2f}%" if r["error_pct"] is not None else "—")
    m3.metric("Bias (Est − Aktual)", f"{r['bias']:+.0f}", help="Positif = over-count, negatif = under-count.")
    st.latex(r"Error(\%) = \frac{|N_p - N_a|}{N_a} \times 100\%")
    if aktual == 0:
        st.caption("Error (%) tidak terdefinisi jika jumlah aktual = 0.")

    if st.button("➕ Tambahkan ke tabel pengujian", type="primary", disabled=not sampel.strip()):
        add_eval_sample(sampel.strip(), int(aktual), int(estimasi))
        st.success("Sampel ditambahkan.")
    if not sampel.strip():
        st.caption("Isi nama sampel untuk menambahkan ke tabel.")

    st.subheader("Tabel pengujian")
    df = get_eval_samples()
    if df.empty:
        st.info("Belum ada sampel pengujian.")
    else:
        df["abs_error"] = (df["estimasi"] - df["aktual"]).abs()
        df["error_%"] = np.where(df["aktual"] > 0, df["abs_error"] / df["aktual"].replace(0, np.nan) * 100, np.nan).round(2)
        df["bias"] = df["estimasi"] - df["aktual"]
        st.dataframe(df.drop(columns=["id"]), use_container_width=True, hide_index=True)
        agg = ev.aggregate_errors(df)
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Jumlah sampel", agg["n"])
        a2.metric("MAE", f"{agg['mae']:.2f}")
        a3.metric("RMSE", f"{agg['rmse']:.2f}" if agg["n"] >= 2 else "—", help="RMSE membutuhkan ≥ 2 sampel.")
        a4.metric("MAPE", f"{agg['mape']:.2f}%" if agg["mape"] is not None else "—")
        if agg["n"] < 2:
            st.caption("RMSE baru ditampilkan jika ada minimal 2 sampel pengujian.")
        st.bar_chart(df.set_index("sampel")[["aktual", "estimasi"]], color=["#1b4332", "#f4b942"])
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Unduh tabel (CSV)", df.to_csv(index=False).encode("utf-8"), "evaluasi_penghitungan.csv", "text/csv")
        if d2.button("🗑️ Hapus semua sampel"):
            delete_eval_samples()
            st.rerun()

# ===================================================== TAB 2
with tab2:
    st.markdown("Unggah **satu foto** dan **file label YOLO (.txt)** hasil anotasi manual foto tersebut. Sistem mendeteksi buah pada foto, lalu mencocokkannya dengan ground truth berdasarkan IoU.")
    model_path = model_selector("eval_model")
    e1, e2, e3 = st.columns(3)
    conf = e1.slider("Confidence", 0.10, 0.90, 0.50, 0.05, key="ev_conf")
    iou_thr = e2.slider("Ambang IoU", 0.30, 0.90, 0.50, 0.05)
    imgsz = e3.select_slider("Image size", [320, 416, 512, 640, 800, 960, 1280], value=640, key="ev_size")
    img_f = st.file_uploader("Foto", type=["jpg", "jpeg", "png"], key="ev_img")
    lbl_f = st.file_uploader("Label YOLO (.txt)", type=["txt"], key="ev_lbl")

    if st.button("▶️ Hitung Precision / Recall", disabled=model_path is None or img_f is None or lbl_f is None):
        try:
            bgr = load_image_bgr(img_f)
            h, w = bgr.shape[:2]
            gt = ev.parse_yolo_labels(lbl_f.getvalue().decode("utf-8"), w, h)
            model = det.load_model(model_path)
            dets, _ = det.detect_image(model, bgr, conf, imgsz)
            pred = np.array([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in dets], dtype=float).reshape(-1, 4)
            cf = np.array([d["conf"] for d in dets], dtype=float)
            res = ev.match_detections(pred, cf, gt, iou_thr)
            vis = bgr.copy()
            for b in gt:
                cv2.rectangle(vis, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (255, 120, 0), 2)  # biru = GT (BGR)
            for b in pred:
                cv2.rectangle(vis, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (60, 200, 60), 2)   # hijau = prediksi
            acc = st.session_state.setdefault("pr_acc", {})
            acc[img_f.name] = dict(tp=res["tp"], fp=res["fp"], fn=res["fn"], gt=len(gt), pred=len(pred))
            st.session_state["pr_vis"] = (bgr_to_rgb(vis), img_f.name)
        except Exception as e:
            st.error(f"Evaluasi gagal: {e}")

    if "pr_vis" in st.session_state:
        st.image(st.session_state["pr_vis"][0], caption=f"{st.session_state['pr_vis'][1]} — hijau: prediksi, biru: ground truth", use_container_width=True)
    acc = st.session_state.get("pr_acc", {})
    if acc:
        tab = pd.DataFrame(acc).T.rename_axis("gambar").reset_index()
        tab[["precision", "recall", "f1"]] = tab.apply(lambda r: pd.Series(ev.precision_recall_f1(r.tp, r.fp, r.fn)), axis=1)
        st.dataframe(tab.round(3), use_container_width=True, hide_index=True)
        tot = ev.precision_recall_f1(int(tab.tp.sum()), int(tab.fp.sum()), int(tab.fn.sum()))
        k1, k2, k3 = st.columns(3)
        k1.metric("Precision (gabungan)", f"{tot['precision']:.3f}")
        k2.metric("Recall (gabungan)", f"{tot['recall']:.3f}")
        k3.metric("F1-score (gabungan)", f"{tot['f1']:.3f}")
        st.caption("Nilai gabungan dihitung dari total TP/FP/FN seluruh gambar yang sudah dievaluasi. mAP@50 dapat dilihat di Deteksi Buah Mangga → Manajemen Model → Validasi.")
        if st.button("Reset akumulasi"):
            st.session_state["pr_acc"] = {}
            st.session_state.pop("pr_vis", None)
            st.rerun()
