import streamlit as st

from modules.database import delete_all_analysis, delete_analysis, get_history
from modules.utils import page_header

page_header("Riwayat Analisis", "Seluruh analisis tersimpan di database SQLite", "🗂️")

jenis_opts = ["Semua", "Deteksi Gambar", "Analisis Video", "Kamera Langsung", "Preprocessing", "Penghitungan"]
jenis = st.selectbox("Filter jenis analisis", jenis_opts)
df = get_history(jenis)

if df.empty:
    st.info("Belum ada riwayat untuk filter ini.")
    st.stop()

st.caption(f"{len(df)} catatan")
view = df.copy()
view.insert(0, "hapus", False)
edited = st.data_editor(
    view, hide_index=True, use_container_width=True, key="riwayat_editor",
    column_config={"hapus": st.column_config.CheckboxColumn("Hapus?", help="Centang baris yang akan dihapus"),
                   "confidence_rata": st.column_config.NumberColumn("confidence", format="%.2f")},
    disabled=[c for c in view.columns if c != "hapus"],
)

c1, c2, c3 = st.columns(3)
sel = edited.loc[edited["hapus"], "id"].tolist()
if c1.button(f"🗑️ Hapus terpilih ({len(sel)})", disabled=not sel):
    n = delete_analysis(sel)
    st.success(f"{n} catatan dihapus.")
    st.rerun()
c2.download_button("⬇️ Export CSV", df.to_csv(index=False).encode("utf-8"), "riwayat_analisis.csv", "text/csv", type="primary")
with c3:
    confirm = st.checkbox("Saya yakin ingin menghapus SEMUA riwayat")
    if st.button("🧨 Hapus semua riwayat", disabled=not confirm):
        delete_all_analysis()
        st.rerun()
