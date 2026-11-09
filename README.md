# 🥭 MangoVision AI
**Sistem Deteksi dan Penghitungan Buah Mangga Berbasis Computer Vision** (UTS Pengolahan Citra)

> Judul UTS: *Pengembangan Sistem Estimasi Jumlah Buah Mangga pada Satu Pohon Menggunakan Smartphone Berbasis YOLO dan Computer Vision.*

## ⚠️ Baca dulu
- Aplikasi **tidak menyertakan model mangga**. Anda harus melatih `best.pt` dari dataset mangga sendiri (atau memakai model mangga dari sumber tepercaya). Model COCO bawaan YOLO **tidak** mengenal mangga.
- Angka yang dihasilkan adalah **estimasi**. *Bounding box ≠ tracking ID ≠ buah aktual.* Identifikasi buah unik antar-sudut (SfM/COLMAP) **belum** diimplementasikan.
- Jangan menulis hasil di laporan UTS kecuali Anda benar-benar menjalankan dan menghitung sendiri.

## 1. Struktur project
```
MangoVision_AI/
├── app.py                  # titik masuk + navigasi sidebar
├── train.py                # training YOLO (CLI, dipanggil juga dari aplikasi)
├── pages/                  # 9 halaman (Dashboard ... Tentang Aplikasi)
├── modules/                # preprocessing, detection, tracking, realtime, counting, evaluation, database, utils
├── assets/style.css        # tema hijau mangga
├── .streamlit/config.toml  # tema + batas upload 500 MB
├── dataset/{images,labels}/{train,val,test}   + data.yaml
├── models/                 # taruh best.pt di sini
├── output/{detection,tracking,preprocessing,video_input}/
├── database/mangovision.db # dibuat otomatis
├── tests/test_core.py      # tes logika inti
└── requirements.txt
```

## 2. Instalasi (Windows + VS Code)
1. Pasang **Python 3.11** (3.10–3.12 didukung). Centang *Add Python to PATH*.
2. Buka folder `MangoVision_AI` di VS Code → **Terminal → New Terminal**.
3. Buat virtual environment dan pasang library:
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
   Jika PowerShell menolak `activate`: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
4. (Opsional, jika punya GPU NVIDIA) pasang PyTorch CUDA dari https://pytorch.org/get-started/locally/ **sebelum** `pip install ultralytics`.
5. Di VS Code: `Ctrl+Shift+P` → *Python: Select Interpreter* → pilih `.venv`.

## 3. Menjalankan
```powershell
streamlit run app.py
```
Browser terbuka di `http://localhost:8501`. Hentikan dengan `Ctrl+C`.

## 4. Menyiapkan dataset mangga
1. Kumpulkan foto pohon mangga (ponsel, ≥12 MP, tidak blur, variasi sudut/cahaya). Minimal beberapa ratus foto agar hasil layak; 30–50 foto/pohon sesuai modul praktikum.
2. Anotasi bounding box dengan **CVAT**, **Roboflow**, atau **LabelImg**; satu kelas: `mango`. Ekspor format **YOLO**.
3. Letakkan file seperti ini (nama label = nama gambar, `.txt`):
```
dataset/images/train/foto001.jpg    dataset/labels/train/foto001.txt
dataset/images/val/foto200.jpg      dataset/labels/val/foto200.txt
dataset/images/test/...             dataset/labels/test/...
```
   Pembagian umum: 70% train / 20% val / 10% test. **Foto dari pohon/video yang sama jangan tersebar di train dan test** agar evaluasi jujur.
4. Isi satu baris label: `0 x_tengah y_tengah lebar tinggi` (semua dinormalisasi 0–1).

## 5. Training YOLO
**Dari aplikasi:** *Deteksi Buah Mangga → Manajemen Model → Training model* (cek status dataset, pilih epoch, klik *Mulai Training*, lalu *Segarkan log*). Hasil disalin ke `models/<nama>.pt`.

**Dari terminal:**
```powershell
python train.py --model yolov8n.pt --epochs 50 --imgsz 640 --batch 8
# GPU: tambahkan --device 0     CPU: --device cpu
```
Bobot awal (`yolov8n.pt`) diunduh otomatis saat pertama kali (butuh internet). Hasil lengkap ada di `runs/mango_yolo/` (kurva, confusion matrix, `weights/best.pt`). Salin/ganti nama menjadi `models/best.pt`.

**Validasi:** *Manajemen Model → Validasi Model* menampilkan Precision, Recall, mAP@50, mAP@50-95 pada split `val` atau `test`.

## 6. Panduan menu
| Menu | Fungsi |
|---|---|
| Dashboard | Ringkasan jumlah analisis, grafik, riwayat terbaru, panduan |
| Deteksi Buah Mangga | Tab *Upload Gambar*: JPG/PNG → YOLO → bounding box + confidence → unduh. Tab *Manajemen Model*: upload `.pt`, status dataset, training, validasi. Tab *Kamera Langsung*: webcam real-time dengan ByteTrack |
| Analisis Video | Upload MP4/AVI/MOV → YOLO + ByteTrack → video hasil + CSV tracking |
| Preprocessing Citra | Resize, crop, brightness/contrast, grayscale, HE, CLAHE, blur + perbandingan & histogram |
| Tracking Buah | Tabel per ID, ID aktif per frame, video hasil |
| Penghitungan Buah | Total deteksi vs ID vs **estimasi** buah unik (buang ID sesaat + gabung tracklet terputus) |
| Evaluasi Akurasi | Error %, MAE, RMSE, MAPE dari hitungan manual; Precision/Recall/F1 dari label ground truth |
| Riwayat Analisis | SQLite: lihat, hapus, export CSV |
| Tentang Aplikasi | Deskripsi, algoritma, keterbatasan |

## 6b. Kamera Langsung (webcam)
Menu **Deteksi Buah Mangga → 📷 Kamera Langsung**.
1. Pilih model (`best.pt`), atur *Confidence*, *Image Size* (kecil = FPS lebih tinggi di CPU), *Konfirmasi ID*, indeks kamera (0 = webcam bawaan) dan resolusi.
2. Klik **▶️ Mulai Kamera**. Tampil bounding box, confidence, ID tracking, hitungan *mangga unik*, jumlah terlihat, total ID, FPS, dan frame.
3. Klik **⏹️ Hentikan Kamera**: kamera dilepas, lalu muncul ringkasan sesi (hitungan real-time, estimasi setelah penggabungan tracklet, tabel per ID).
4. *Simpan ke riwayat* menulis CSV ke `output/tracking/` sehingga bisa dibuka di menu **Tracking Buah** dan **Penghitungan Buah**.

Catatan: mengubah slider/pilihan saat kamera aktif menghentikan kamera (perilaku Streamlit); atur dulu sebelum Mulai. Webcam dibaca oleh komputer yang menjalankan `streamlit run`, jadi fitur ini untuk penggunaan lokal di laptop, bukan hosting cloud. "Mangga unik" adalah estimasi: ID dihitung bila terlihat ≥ N frame; kamera yang bergeser jauh atau buah yang lama tertutup bisa menghasilkan ID baru.

## 7. Panduan pengujian (urutan disarankan)
1. `python tests/test_core.py` → tes logika (preprocessing, counting, evaluasi, database, alur tracking dengan model tiruan).
2. `streamlit run app.py` → buka semua 9 menu; tanpa model harus muncul pesan merah yang jelas, bukan crash.
3. *Preprocessing*: unggah foto, aktifkan CLAHE & resize, cek gambar berubah dan tombol unduh bekerja.
4. Latih/unggah model → *Deteksi*: foto berisi mangga, coba threshold 0.25/0.50/0.75, amati perubahan jumlah.
5. *Analisis Video*: awali dengan **batas frame 100** dan interval 2 untuk uji cepat.
6. *Tracking* → *Penghitungan* → hitung manual buah di pohon yang sama → *Evaluasi*.
7. *Riwayat*: cek catatan muncul, export CSV, hapus.

## 8. Mengatasi error
| Gejala | Solusi |
|---|---|
| `ModuleNotFoundError: streamlit/ultralytics/cv2` | Aktifkan `.venv` lalu `pip install -r requirements.txt` |
| Pesan "Model mangga belum tersedia" | Taruh `best.pt` di `models/` atau unggah di Manajemen Model |
| Banyak deteksi aneh / model ≥80 kelas | Itu model COCO, bukan mangga; latih model sendiri |
| Video hasil tidak tampil di browser | ffmpeg gagal; unduh videonya dan putar di komputer. Pastikan `imageio-ffmpeg` terpasang |
| Analisis video sangat lambat | Naikkan *Frame Interval*, turunkan *Image Size*, isi *Batas frame*, atau pakai GPU |
| `CUDA out of memory` | Turunkan batch/imgsz saat training atau pakai `--device cpu` |
| Training tidak mulai / "dataset tidak ditemukan" | Pastikan ada gambar di `images/train` dan `images/val` serta label `.txt` bernama sama |
| `lap`/`lapx` error saat ByteTrack | `pip install lapx` |
| "Kamera tidak dapat dibuka" | Tutup Zoom/Meet/aplikasi Camera, cek izin kamera Windows (Privasi → Kamera → izinkan aplikasi desktop), coba indeks kamera 1 atau 2 |
| FPS kamera rendah | Turunkan Image Size (320/416) dan resolusi 640×480; gunakan GPU bila ada |
| Port 8501 dipakai | `streamlit run app.py --server.port 8502` |
| Upload > 200 MB ditolak | Sudah dinaikkan ke 500 MB di `.streamlit/config.toml` |

## 9. Panduan screenshot dokumentasi UTS
Ambil dengan `Win+Shift+S`. Sertakan tanggal/jam dan nama file terlihat.
1. Struktur folder di VS Code Explorer.
2. Terminal saat `streamlit run app.py` berhasil.
3. Dashboard (sebelum dan sesudah ada data).
4. Preprocessing: gambar asli vs hasil + panel informasi proses.
5. Hasil training: `runs/mango_yolo/results.png` dan `confusion_matrix.png`; hasil *Validasi Model* (Precision/Recall/mAP).
6. Deteksi gambar dengan 2–3 threshold berbeda (bounding box + tabel).
7. Analisis video: video hasil dengan ID; ringkasan metrik.
8. Tracking: grafik ID aktif per frame; tabel per ID.
9. Penghitungan: perbandingan deteksi vs ID vs estimasi + penjelasan keterbatasan.
10. Evaluasi: tabel sampel, MAE/RMSE, dan (jika ada) Precision/Recall.
11. Riwayat Analisis dan file CSV hasil export.
12. Foto lapangan: cara Anda mengambil gambar mengelilingi pohon, dan lembar hitungan manual.

## 10. Rujukan
Stein, Bargoti & Underwood (2016), *Sensors* 16(11):1915 · Schönberger & Frahm (2016), *Structure-from-Motion Revisited*, CVPR · Dokumentasi Ultralytics (track/ByteTrack) · Dokumentasi COLMAP.
