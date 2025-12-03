import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Crypto Signal AI", layout="wide")
st.title("🤖 Crypto Direction AI: Buy or Sell?")
st.write("Model Deep Learning (LSTM) yang memprediksi **ARAH** pergerakan harga (Naik/Turun) dan tingkat **Volatilitas**.")

# Sidebar
st.sidebar.header("Konfigurasi")
ticker = st.sidebar.text_input("Ticker Symbol", value="BTC-USD")
# Start date dihapus sesuai request, default load data dari 2020

# --- 2. LOAD DATA (ANTI-GAGAL / RETRY LOGIC) ---
@st.cache_data
def load_data(ticker):
    max_retries = 3
    for i in range(max_retries):
        try:
            # Download data tanpa progress bar biar bersih
            # Start fix dari 2020-01-01 sesuai request
            data = yf.download(ticker, start="2020-01-01", end=pd.to_datetime("today"), progress=False)
            
            # Cek apakah data kosong
            if len(data) > 0:
                data.reset_index(inplace=True)
                return data
            
            # Kalau kosong, tunggu 1 detik sebelum coba lagi
            time.sleep(1)
        except:
            time.sleep(1)
    return None

data_load_state = st.text('Sedang memuat data...')
data = load_data(ticker)

if data is None or data.empty:
    st.error(f"Data {ticker} tidak ditemukan. Coba cek koneksi atau ganti ticker.")
    st.stop()

data_load_state.text(f'Data {ticker} berhasil dimuat!')

# --- 3. PROSES PREDIKSI ---
# Load Model Klasifikasi
try:
    # PASTIKAN FILE INI ADALAH HASIL TRAINING BARU (KLASIFIKASI)
    model = load_model('direction_model.h5') 
except:
    st.error("Model 'direction_model.h5' tidak ditemukan. Jalankan script training (train_direction.py) terlebih dahulu!")
    st.stop()

# Preprocessing Data Terakhir
# Kita hanya butuh data Close untuk input model
df_close = data[['Close']]
dataset = df_close.values

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Ambil 60 hari terakhir buat nebak besok
look_back = 60
if len(scaled_data) < look_back:
    st.error("Data tidak cukup untuk melakukan prediksi. Butuh minimal 60 hari.")
    st.stop()

last_60_days = scaled_data[-look_back:]
X_input = last_60_days.reshape(1, look_back, 1)

# EKSEKUSI PREDIKSI (Outputnya probabilitas 0.0 s/d 1.0)
prediction_prob = model.predict(X_input)[0][0]

# Logic Sinyal (Threshold 0.5)
# > 0.5 = AI yakin Naik
# < 0.5 = AI yakin Turun
threshold = 0.5
if prediction_prob > threshold:
    signal = "NAIK (BULLISH) 🚀"
    color = "green"
    # Confidence: Seberapa jauh dari 0.5?
    confidence = prediction_prob * 100
else:
    signal = "TURUN (BEARISH) 🔻"
    color = "red"
    # Kalau prob 0.2, berarti confidence turunnya 80% (1 - 0.2)
    confidence = (1 - prediction_prob) * 100

# --- 4. HITUNG VOLATILITAS (RISK MANAGEMENT) ---
# Menjawab request temen lu: "predict volatilitynya kek kira2 bakal fluctuate sebesar berapa"
# Kita hitung standar deviasi return harian selama 30 hari terakhir
data['Return'] = data['Close'].pct_change()
# Dikali 100 biar jadi persen
volatility_30d = data['Return'].tail(30).std() * 100 

if volatility_30d > 4.0: # Batas risiko tinggi (Crypto emang volatil)
    vol_status = "EXTREME RISK ⚡"
    advice = "Pasar sangat liar. Hindari leverage tinggi."
elif volatility_30d > 2.0:
    vol_status = "MEDIUM RISK ⚠️"
    advice = "Pasar cukup aktif. Gunakan Stop Loss yang wajar."
else:
    vol_status = "LOW RISK ✅"
    advice = "Pasar relatif tenang (Sideways/Stabil)."

# --- 5. TAMPILAN DASHBOARD ---
st.divider()

# Kolom Dashboard Utama
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Sinyal AI Besok")
    if color == "green":
        st.success(f"### {signal}")
    else:
        st.error(f"### {signal}")
    st.caption(f"Prediksi untuk perdagangan besok.")

with col2:
    st.subheader("AI Confidence")
    st.metric(label="Tingkat Keyakinan", value=f"{confidence:.2f}%")
    st.progress(int(confidence))
    st.caption("Semakin tinggi %, semakin yakin AI dengan sinyalnya.")

with col3:
    st.subheader("Analisis Risiko")
    # Delta color inverse: Merah kalau angkanya gede (High Risk)
    st.metric(label="Volatilitas (30 Hari)", value=f"{volatility_30d:.2f}%", delta=vol_status, delta_color="inverse")
    st.info(advice)

st.divider()

# Visualisasi Harga Terakhir (Full History dari 2020)
st.subheader(f"Tren Harga {ticker} (Full History)")

# Mengambil data full untuk plot (bukan cuma tail)
chart_data = data.set_index('Date')['Close']
st.line_chart(chart_data)

st.success("✅ Analisis Selesai. Gunakan sebagai referensi pendukung keputusan trading Anda.")