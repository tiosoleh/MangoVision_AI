import streamlit as st

from modules.utils import APP_NAME, APP_SUBTITLE, page_header

page_header("Tentang Aplikasi", APP_SUBTITLE, "ℹ️")

st.markdown(f"""
### Deskripsi
**{APP_NAME}** adalah aplikasi Streamlit untuk mendeteksi dan memperkirakan jumlah buah mangga pada satu pohon dari foto dan video
yang direkam dengan ponsel. Aplikasi ini dibuat untuk UTS Pengolahan Citra (Precision Agriculture) dengan judul:

> *Pengembangan Sistem Estimasi Jumlah Buah Mangga pada Satu Pohon Menggunakan Smartphone Berbasis YOLO dan Computer Vision.*

### Tujuan
1. Mendeteksi buah mangga pada citra dengan YOLO.
2. Melacak buah antarframe pada video dan webcam (kamera langsung) dengan ByteTrack.
3. Mengurangi penghitungan ganda dan menghasilkan **estimasi** jumlah buah unik.
4. Mengevaluasi hasil terhadap penghitungan manual (Error %, MAE, RMSE, Precision, Recall, F1).

### Teknologi
Python · Streamlit · OpenCV · Ultralytics YOLO · ByteTrack · NumPy · Pandas · Matplotlib · Pillow · SciPy · SQLite

### Penjelasan algoritma
- **Preprocessing (OpenCV)**: resize, crop, brightness/contrast, grayscale, histogram equalization, CLAHE, Gaussian blur.
- **YOLO (You Only Look Once)**: detektor satu tahap yang menghasilkan bounding box, confidence, dan kelas dalam sekali jalan.
- **ByteTrack**: menghubungkan deteksi antarframe (termasuk deteksi ber-confidence rendah) dan memberi tracking ID.
- **Estimasi buah unik**: membuang ID yang sangat singkat, lalu menggabungkan tracklet yang terputus berdasarkan jeda waktu dan jarak
  posisi memakai *Hungarian Algorithm*.
- **Evaluasi**: Error(%) = |Estimasi − Aktual| / Aktual × 100%, MAE, RMSE, MAPE; Precision/Recall/F1 dari pencocokan IoU dengan ground truth.

### Cara penggunaan singkat
1. Siapkan model `best.pt` (unggah atau latih di *Deteksi Buah Mangga → Manajemen Model*).
2. Foto: *Deteksi Buah Mangga*. Video: *Analisis Video* → *Tracking Buah* → *Penghitungan Buah*.
3. Hitung buah secara manual pada pohon yang sama, lalu isi *Evaluasi Akurasi*.
4. Lihat atau ekspor catatan di *Riwayat Analisis*.

### Keterbatasan sistem
- **Kualitas hasil bergantung pada model.** Tanpa model yang dilatih pada dataset mangga, deteksi tidak bermakna.
- **Bounding box ≠ tracking ID ≠ buah aktual.** Satu buah bisa terdeteksi berkali-kali dan mendapat ID berbeda.
- **Multi-view association (SfM/COLMAP, triangulasi 3D) belum diimplementasikan.** Buah yang sama pada sisi pohon berbeda belum dapat dikenali sebagai satu buah; hasil penghitungan adalah estimasi.
- Buah tertutup daun, tertutup buah lain, atau berada di sisi yang tidak terekam tidak akan terhitung.
- Cahaya ekstrem, foto blur, dan angin yang menggerakkan tajuk menurunkan akurasi.
- Kamera langsung membaca webcam di komputer yang menjalankan aplikasi (lokal), bukan kamera peramban; FPS di CPU rendah.
- Pemrosesan video di CPU lambat; gunakan frame interval dan batas frame untuk uji coba.
- Penentuan kematangan/siap panen belum termasuk dalam aplikasi ini (hanya dibahas pada rancangan penelitian lanjutan).
""")
