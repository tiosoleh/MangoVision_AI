import matplotlib.pyplot as plt
import cv2
import numpy as np
import streamlit as st

from modules.database import add_analysis
from modules.preprocessing import apply_pipeline, laplacian_sharpness, to_grayscale
from modules.utils import PREPROC_DIR, bgr_to_rgb, encode_png, load_image_bgr, page_header, safe_stem, save_image

page_header("Preprocessing Citra", "Perbaiki kualitas citra dengan OpenCV sebelum deteksi", "🛠️")

up = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"], key="pre_upload")
if up is None:
    st.info("Unggah gambar untuk mulai. Semua pengaturan diterapkan secara langsung.")
    st.stop()

try:
    orig = load_image_bgr(up)
except Exception as e:
    st.error(f"Gambar tidak dapat dibaca: {e}")
    st.stop()

with st.expander("⚙️ Pengaturan preprocessing", expanded=True):
    a, b = st.columns(2)
    with a:
        do_resize = st.checkbox("Resize (rasio dipertahankan)")
        max_side = st.slider("Sisi terpanjang (px)", 256, 3000, 1280, 32, disabled=not do_resize)
        do_crop = st.checkbox("Cropping")
        cl, cr = st.columns(2)
        crop_l = cl.slider("Kiri (%)", 0, 45, 0, disabled=not do_crop)
        crop_r = cr.slider("Kanan (%)", 0, 45, 0, disabled=not do_crop)
        ct, cb = st.columns(2)
        crop_t = ct.slider("Atas (%)", 0, 45, 0, disabled=not do_crop)
        crop_b = cb.slider("Bawah (%)", 0, 45, 0, disabled=not do_crop)
        brightness = st.slider("Brightness", -100, 100, 0)
        contrast = st.slider("Contrast", -100, 100, 0)
    with b:
        gray = st.checkbox("Grayscale")
        he = st.checkbox("Histogram Equalization")
        use_clahe = st.checkbox("CLAHE")
        clip = st.slider("CLAHE clip limit", 1.0, 8.0, 2.0, 0.5, disabled=not use_clahe)
        tile = st.slider("CLAHE tile size", 2, 16, 8, disabled=not use_clahe)
        blur = st.checkbox("Gaussian Blur")
        blur_k = st.slider("Ukuran kernel (ganjil)", 3, 31, 5, 2, disabled=not blur)
        st.caption("Urutan proses: Crop → Resize → Brightness/Contrast → Grayscale → HE → CLAHE → Blur.")

params = dict(resize=do_resize, max_side=max_side, crop=do_crop, crop_l=crop_l, crop_r=crop_r, crop_t=crop_t,
              crop_b=crop_b, brightness=brightness, contrast=contrast, gray=gray, hist_eq=he, clahe=use_clahe,
              clip=clip, tile=tile, blur=blur, blur_k=blur_k)
try:
    result, log = apply_pipeline(orig, params)
except Exception as e:
    st.error(f"Preprocessing gagal: {e}")
    st.stop()

st.subheader("Perbandingan Sebelum & Sesudah")
c1, c2 = st.columns(2)
c1.image(bgr_to_rgb(orig), caption=f"Asli — {orig.shape[1]}×{orig.shape[0]} px", use_container_width=True)
c2.image(bgr_to_rgb(result), caption=f"Hasil — {result.shape[1]}×{result.shape[0]} px", use_container_width=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Ukuran asli", f"{orig.shape[1]}×{orig.shape[0]}")
m2.metric("Ukuran hasil", f"{result.shape[1]}×{result.shape[0]}")
m3.metric("Ketajaman (Laplacian)", f"{laplacian_sharpness(result):.0f}", f"{laplacian_sharpness(result) - laplacian_sharpness(orig):+.0f}")
m4.metric("Rata-rata kecerahan", f"{to_grayscale(result).mean():.0f}", f"{to_grayscale(result).mean() - to_grayscale(orig).mean():+.0f}")

with st.expander("📊 Histogram kecerahan"):
    fig, ax = plt.subplots(figsize=(8, 2.6))
    for img, lab, col in ((orig, "Asli", "#8d99ae"), (result, "Hasil", "#2d9d4b")):
        h = cv2.calcHist([to_grayscale(img)], [0], None, [256], [0, 256]).ravel()
        ax.plot(h / h.sum(), label=lab, color=col)
    ax.set_xlabel("Intensitas (0–255)"); ax.set_ylabel("Proporsi piksel"); ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Informasi Proses")
for i, line in enumerate(log, 1):
    st.markdown(f"{i}. {line}")

d1, d2 = st.columns(2)
d1.download_button("⬇️ Unduh Hasil Preprocessing", encode_png(result), file_name=f"preproc_{safe_stem(up.name)}.png",
                   mime="image/png", type="primary")
if d2.button("💾 Simpan ke folder output & riwayat"):
    try:
        p = save_image(result, PREPROC_DIR, safe_stem(up.name))
        add_analysis(up.name, "Preprocessing", 0, 0.0, "Selesai", "; ".join(log)[:300])
        st.success(f"Tersimpan: {p}")
    except Exception as e:
        st.error(f"Gagal menyimpan: {e}")
