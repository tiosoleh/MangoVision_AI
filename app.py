"""MangoVision AI - titik masuk aplikasi Streamlit.

Jalankan:  streamlit run app.py
"""
import streamlit as st

from modules.database import init_db
from modules.utils import APP_NAME, ensure_dirs, inject_css

st.set_page_config(page_title=f"{APP_NAME}", page_icon="🥭", layout="wide",
                   initial_sidebar_state="expanded")
ensure_dirs()
init_db()
inject_css()

pages = [
    st.Page("pages/1_Dashboard.py", title="Dashboard", icon="📊", default=True),
    st.Page("pages/2_Deteksi_Buah.py", title="Deteksi Buah Mangga", icon="🔍"),
    st.Page("pages/3_Analisis_Video.py", title="Analisis Video", icon="🎬"),
    st.Page("pages/4_Preprocessing.py", title="Preprocessing Citra", icon="🛠️"),
    st.Page("pages/5_Tracking_Buah.py", title="Tracking Buah", icon="🎯"),
    st.Page("pages/6_Penghitungan_Buah.py", title="Penghitungan Buah", icon="🔢"),
    st.Page("pages/7_Evaluasi_Akurasi.py", title="Evaluasi Akurasi", icon="📏"),
    st.Page("pages/8_Riwayat_Analisis.py", title="Riwayat Analisis", icon="🗂️"),
    st.Page("pages/9_Tentang_Aplikasi.py", title="Tentang Aplikasi", icon="ℹ️"),
]
nav = st.navigation(pages)

with st.sidebar:
    st.markdown("---")
    st.markdown(f"### 🥭 {APP_NAME}")
    st.caption("UTS Pengolahan Citra · Precision Agriculture")

nav.run()
