# ============================================================
# 🌊 Streamlit: Prediksi Tsunami (RF vs Gradient Boosting)
# ============================================================
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="Prediksi Tsunami", page_icon="🌊", layout="wide")
st.title("🌊 Prediksi Potensi Tsunami — Random Forest vs Gradient Boosting")
st.markdown("""
Aplikasi ini memuat **model terbaik** hasil GridSearchCV dan memungkinkan kamu:
- Memilih **Random Forest** atau **Gradient Boosting**
- Atau **membandingkan** hasil keduanya sekaligus  
Input yang digunakan adalah **8 fitur terpilih** hasil seleksi fitur.

> Pastikan file model ini ada di folder yang sama:
> - `BestModel_RandomForest_KelompokTsunami.pkl`
> - `BestModel_CLF_GradientBoosting_KelompokTsunami.pkl`
""")

# ------------------------------------------------------------
# 🔧 Muat model
# ------------------------------------------------------------

with open("model/BestModel_CLF_RandomForest_KERAS.pkl", "rb") as f:
  model = pickle.load(f)

# ------------------------------------------------------------
# 🧬 Fitur terpilih (urutan harus sama dengan training)
# ------------------------------------------------------------
FEATURES = ['cdi','mmi','sig','nst','dmin','gap','latitude','longitude']

# ------------------------------------------------------------
# 📝 Input pengguna (mirip contoh yang kamu pakai)
# ------------------------------------------------------------
st.subheader("Masukkan Nilai Fitur (8 Fitur Terbaik)")
col1, col2 = st.columns(2)

with col1:
    cdi = st.number_input("CDI (0–12)", min_value=0.0, max_value=12.0, step=0.1, value=3.0,
                          help="Community Determined Intensity")
    mmi = st.number_input("MMI (0–12)", min_value=0.0, max_value=12.0, step=0.1, value=3.0,
                          help="Modified Mercalli Intensity")
    sig = st.number_input("SIG (≥0)", min_value=0.0, step=1.0, value=100.0,
                          help="Significance index")
    nst = st.number_input("NST (≥0)", min_value=0, step=1, value=10,
                          help="Jumlah stasiun yang merekam")

with col2:
    dmin = st.number_input("DMIN", min_value=0.0, step=0.01, value=0.5,
                          help="Jarak minimum stasiun–episenter")
    gap = st.number_input("GAP (0–360)", min_value=0.0, max_value=360.0, step=1.0, value=120.0,
                          help="Sudut gap maksimum antar stasiun")
    latitude = st.number_input("Latitude (-90–90)", min_value=-90.0, max_value=90.0, step=0.01, value=0.0)
    longitude = st.number_input("Longitude (-180–180)", min_value=-180.0, max_value=180.0, step=0.01, value=120.0)

X_input = pd.DataFrame([{
    'cdi': cdi, 'mmi': mmi, 'sig': sig, 'nst': nst,
    'dmin': dmin, 'gap': gap, 'latitude': latitude, 'longitude': longitude
}], columns=FEATURES)

# ------------------------------------------------------------
# ⚙️ Pilih mode: satu model atau bandingkan
# ------------------------------------------------------------
mode = st.radio(
    "Pilih Mode",
    ["Random Forest saja"],
    horizontal=True
)

# ------------------------------------------------------------
# 🔮 Util prediksi
# ------------------------------------------------------------
def predict_and_plot(model, X, title):
    pred = int(model.predict(X)[0])
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
    else:
        # fallback (jarang diperlukan untuk RF/GBC)
        proba = np.array([1-pred, pred], dtype=float)

    # Hasil teks
    if pred == 1:
        st.error(f"**{title} → Prediksi: TSUNAMI (1)**")
    else:
        st.success(f"**{title} → Prediksi: TIDAK (0)**")

    # Bar chart probabilitas (kelas 0 vs 1)
    st.markdown("Probabilitas Prediksi")
    df_proba = pd.DataFrame({"Kelas": ["0 = Tidak", "1 = Tsunami"], "Probabilitas": proba})

    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.bar(df_proba["Kelas"], df_proba["Probabilitas"], color=["green", "red"], alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probabilitas")
    ax.set_title(f"Distribusi Probabilitas — {title}")
    for i, v in enumerate(df_proba["Probabilitas"]):
        ax.text(i, min(v + 0.02, 1.0), f"{v:.2%}", ha="center", fontsize=10, fontweight="bold")
    st.pyplot(fig)

    return pred, float(proba[1])

# ------------------------------------------------------------
# 🔍 Prediksi
# ------------------------------------------------------------
if st.button("🔍 Jalankan Prediksi"):
    st.write("---")
    if mode == "Random Forest":
        predict_and_plot(model, X_input, "Random Forest")

    else:
        # Bandingkan keduanya
        colA, colB = st.columns(2)
        with colA:
            pred_rf, p1_rf = predict_and_plot(model, X_input, "Random Forest")

        st.write("---")
        st.subheader("🏁 Ringkasan Perbandingan (Input Ini)")
        c1 = st.columns(3)
        c1.metric("Prob(1) — RF", f"{p1_rf:.3f}")

st.caption("Catatan: Struktur UI & alur input meniru pola dari contoh Streamlit yang kamu kirim (judul, load model, number inputs, tombol prediksi, grafik probabilitas).")
