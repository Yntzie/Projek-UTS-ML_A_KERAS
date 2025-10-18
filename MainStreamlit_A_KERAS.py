# ============================================================
# 🌊 Streamlit: Prediksi Tsunami — Random Forest (Model Terbaik)
# ============================================================
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# ---------------------------- UI Dasar ----------------------------
st.set_page_config(page_title="Prediksi Tsunami", page_icon="🌊", layout="wide")
st.title("🌊 Prediksi Potensi Tsunami — Random Forest")
st.markdown("""
Aplikasi ini menggunakan **model terbaik (Random Forest)** hasil GridSearchCV:
> File model: `BestModel_CLF_RandomForest_KERAS.pkl`
""")

# ---------------------------- Load Model ----------------------------
@st.cache_resource
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    model_rf = load_model("model/BestModel_CLF_RandomForest_KERAS.pkl")
except Exception as e:
    st.error(f"Gagal memuat model Random Forest: {e}")
    st.stop()

# ---------------------------- Daftar Fitur ----------------------------
FEATURES_FULL = ['magnitude','cdi','mmi','sig','nst','dmin','gap','depth','latitude','longitude']
FEATURES_BEST = ['cdi','mmi','sig','nst','dmin','gap','latitude','longitude']

DEFAULTS_FOR_MISSING = {
    'magnitude': 6.8,
    'depth': 26.295
}

# ---------------------------- Input Pengguna ----------------------------
st.subheader("Masukkan Nilai Fitur (8 Fitur Terbaik)")
st.markdown("""
Keterangan fitur:
- **CDI (Community Determined Intensity)** → Intensitas guncangan yang dirasakan masyarakat (0–12).  
- **MMI (Modified Mercalli Intensity)** → Ukuran subjektif dampak gempa di permukaan (0–12).  
- **SIG (Significance)** → Tingkat signifikansi kekuatan gempa (semakin besar → semakin kuat).  
- **NST (Number of Stations)** → Jumlah stasiun seismik yang mendeteksi gempa.  
- **DMIN (Distance Min)** → Jarak minimum antara stasiun pengukur dengan episentrum (km).  
- **GAP (Azimuthal Gap)** → Distribusi arah pengukuran dari berbagai stasiun (0–360°).  
- **Latitude & Longitude** → Titik koordinat pusat gempa.
""")

c1, c2 = st.columns(2)
with c1:
    cdi = st.number_input("CDI (0–12)", 0.0, 12.0, 3.0, 0.1)
    mmi = st.number_input("MMI (0–12)", 0.0, 12.0, 3.0, 0.1)
    sig = st.number_input("SIG (≥0)", 0.0, step=1.0, value=100.0)
    nst = st.number_input("NST (≥0)", 0, step=1, value=10)
with c2:
    dmin = st.number_input("DMIN", 0.0, step=0.01, value=0.5)
    gap = st.number_input("GAP (0–360)", 0.0, 360.0, 120.0, 1.0)
    latitude = st.number_input("Latitude (-90–90)", -90.0, 90.0, 0.0, 0.01)
    longitude = st.number_input("Longitude (-180–180)", -180.0, 180.0, 120.0, 0.01)

row_best = {
    'cdi': cdi, 'mmi': mmi, 'sig': sig, 'nst': nst,
    'dmin': dmin, 'gap': gap, 'latitude': latitude, 'longitude': longitude
}

# kolom lengkap sesuai training (2 kolom sisanya diisi default)
row_full = {}
for col in FEATURES_FULL:
    if col in row_best:
        row_full[col] = float(row_best[col])
    else:
        row_full[col] = float(DEFAULTS_FOR_MISSING.get(col, 0.0))

X_input = pd.DataFrame([row_full], columns=FEATURES_FULL)

# ---------------------------- Prediksi ----------------------------
def predict_and_plot(model, X, title):
    pred = int(model.predict(X)[0])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
    else:
        proba = np.array([1 - pred, pred], dtype=float)

    # === Tampilkan hasil prediksi (success/error) ===
    if pred == 1:
        st.error(f"**{title} → Prediksi: TSUNAMI (1)**")
    else:
        st.success(f"**{title} → Prediksi: TIDAK (0)**")

    # === Visualisasi probabilitas ===
    dfp = pd.DataFrame({"Kelas": ["0 = Tidak", "1 = Tsunami"], "Prob": proba})
    fig, ax = plt.subplots(figsize=(2.8, 1.8))
    ax.bar(dfp["Kelas"], dfp["Prob"], color=["green", "red"], alpha=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Prob", fontsize=8)
    ax.set_title("Probabilitas", fontsize=9)
    for i, v in enumerate(dfp["Prob"]):
        ax.text(i, min(v + 0.03, 1.0), f"{v:.2%}", ha="center", fontsize=8)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, width="content")

    return pred, float(proba[1])

# ---------------------------- Eksekusi ----------------------------
if st.button("🔍 Jalankan Prediksi"):
    st.write("---")

    pred_rf, p1_rf = predict_and_plot(model_rf, X_input, "Random Forest")

    # ---------------------------- Ringkasan ----------------------------
    st.write("---")
    st.subheader("🏁 Ringkasan Hasil Prediksi — Random Forest")

    c1, c2 = st.columns(2)
    c1.metric("Prob(1) — Tsunami", f"{p1_rf:.3f}")
    c2.metric("Prob(0) — Tidak", f"{1 - p1_rf:.3f}")

    st.markdown("**Nilai Fitur yang Dikirim ke Model:**")
    st.dataframe(pd.DataFrame([X_input.iloc[0].to_dict()]), width="stretch")

    # tampilkan fitur terpilih (opsional)
    try:
        if hasattr(model_rf, "named_steps") and "feat_select" in model_rf.named_steps:
            selector = model_rf.named_steps["feat_select"]
            if hasattr(selector, "get_support"):
                mask = selector.get_support()
                selected_feats = np.array(FEATURES_FULL)[mask]
                st.markdown("**Fitur terpilih oleh selector dalam pipeline:**")
                st.write(", ".join(selected_feats))
    except Exception:
        pass

st.caption("Pastikan file model `.pkl` berada di folder `model/` dan urutan kolom input sesuai dengan urutan saat training.")
