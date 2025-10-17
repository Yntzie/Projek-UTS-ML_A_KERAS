# ============================================================
# 🌊 Streamlit: Prediksi Tsunami — RF vs Gradient Boosting
# ============================================================
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# ---------------------------- UI Dasar ----------------------------
st.set_page_config(page_title="Prediksi Tsunami", page_icon="🌊", layout="wide")
st.title("🌊 Prediksi Potensi Tsunami — Random Forest vs Gradient Boosting")
st.markdown("""
Aplikasi ini memuat **dua model terbaik** hasil GridSearchCV dan membandingkan hasil prediksi:
- 🌲 **Random Forest** → `BestModel_CLF_RandomForest_KERAS.pkl`  
- 🚀 **Gradient Boosting** → `BestModel_CLF_GradientBoosting_KERAS.pkl`
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

try:
    model_gbc = load_model("model/BestModel_CLF_GradientBoosting_KERAS.pkl")
except Exception as e:
    st.error(f"Gagal memuat model Gradient Boosting: {e}")
    st.stop()

# === daftar fitur ===
FEATURES_FULL = ['magnitude','cdi','mmi','sig','nst','dmin','gap','depth','latitude','longitude']
FEATURES_BEST = ['cdi','mmi','sig','nst','dmin','gap','latitude','longitude']

DEFAULTS_FOR_MISSING = {
    'magnitude': 6.8,
    'depth': 26.295
}

# ---------------------------- Form Input ----------------------------
st.subheader("Masukkan Nilai Fitur (Urutan Sama dengan Training)")

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

row_full = {}
for col in FEATURES_FULL:
    if col in row_best:
        row_full[col] = float(row_best[col])
    else:
        # kolom yang tidak diminta user (magnitude, depth) diisi default
        row_full[col] = float(DEFAULTS_FOR_MISSING.get(col, 0.0))

X_input = pd.DataFrame([row_full], columns=FEATURES_FULL)


# ---------------------------- Mode ----------------------------
mode = st.radio(
    "Pilih Mode",
    ["Random Forest saja", "Gradient Boosting saja", "Bandingkan keduanya"],
    horizontal=True
)

# ---------------------------- Util Prediksi ----------------------------
def predict_and_plot(model, X, title):
    # label
    pred = int(model.predict(X)[0])

    # probabilitas
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
    else:
        # fallback jika estimator tidak punya predict_proba
        proba = np.array([1 - pred, pred], dtype=float)

    # Teks hasil
    if pred == 1:
        st.error(f"**{title} → Prediksi: TSUNAMI (1)**")
    else:
        st.success(f"**{title} → Prediksi: TIDAK (0)**")

    # Bar chart probabilitas
    st.markdown("Probabilitas Prediksi")
    dfp = pd.DataFrame({"Kelas": ["0 = Tidak", "1 = Tsunami"], "Prob": proba})

    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.bar(dfp["Kelas"], dfp["Prob"], alpha=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probabilitas")
    ax.set_title(f"Distribusi Probabilitas — {title}")
    for i, v in enumerate(dfp["Prob"]):
        ax.text(i, min(v + 0.02, 1.0), f"{v:.2%}", ha="center", fontsize=10, fontweight="bold")
    st.pyplot(fig)

    return pred, float(proba[1])

# ---------------------------- Eksekusi ----------------------------
if st.button("🔍 Jalankan Prediksi"):
    st.write("---")

    if mode == "Random Forest saja":
        predict_and_plot(model_rf, X_input, "Random Forest")

    elif mode == "Gradient Boosting saja":
        predict_and_plot(model_gbc, X_input, "Gradient Boosting")

    else:
        colA, colB = st.columns(2)
        with colA:
            pred_rf, p1_rf = predict_and_plot(model_rf, X_input, "Random Forest")
        with colB:
            pred_gbc, p1_gbc = predict_and_plot(model_gbc, X_input, "Gradient Boosting")

        st.write("---")
        st.subheader("🏁 Ringkasan Perbandingan (Input Ini)")
        m1, m2, m3 = st.columns(3)
        m1.metric("Prob(1) — RF", f"{p1_rf:.3f}")
        m2.metric("Prob(1) — GBC", f"{p1_gbc:.3f}")
        winner = "Random Forest" if p1_rf >= p1_gbc else "Gradient Boosting"
        m3.metric("Model lebih yakin", winner)

st.caption("Pastikan kedua file model .pkl berada di direktori yang sama dengan file ini. Urutan kolom input sudah disamakan dengan urutan training.")
