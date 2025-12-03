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

# --- 2. SIDEBAR & LOAD DATA ---
st.sidebar.header("Konfigurasi")
ticker = st.sidebar.text_input("Ticker Symbol", value="BTC-USD")

# Fungsi Load Data (Download Full History)
@st.cache_data
def load_data(ticker):
    max_retries = 3
    for i in range(max_retries):
        try:
            # period="max" artinya ambil data dari awal koin itu lahir sampai sekarang
            data = yf.download(ticker, period="max", progress=False)
            
            if len(data) > 0:
                data.reset_index(inplace=True)
                return data
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

# --- 3. FILTER TANGGAL (Sesuai Request) ---
# Kita ambil tanggal paling awal dan paling akhir dari data yang ditarik
min_date = data['Date'].min().date()
max_date = data['Date'].max().date()

# Sidebar Input Tanggal
# Default value = min_date (Data paling lama)
start_user_date = st.sidebar.date_input(
    "Tampilkan Grafik Sejak", 
    value=min_date, 
    min_value=min_date, 
    max_value=max_date
)

# Filter data untuk VISUALISASI GRAFIK saja
# (Prediksi tetep pake data terbaru, tapi grafik ngikutin mau user liat dari kapan)
mask = (data['Date'].dt.date >= start_user_date)
filtered_data = data.loc[mask]

# --- 4. PROSES PREDIKSI (AI) ---
# Load Model Klasifikasi
try:
    model = load_model('direction_model.h5') 
except:
    st.error("Model 'direction_model.h5' tidak ditemukan. Jalankan script training (train_direction.py) terlebih dahulu!")
    st.stop()

# Preprocessing Data (Pake data full original biar akurat 60 hari terakhirnya)
df_close = data[['Close']]
dataset = df_close.values

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Ambil 60 hari terakhir buat nebak besok
look_back = 60
if len(scaled_data) < look_back:
    st.error("Data tidak cukup untuk melakukan prediksi.")
    st.stop()

last_60_days = scaled_data[-look_back:]
X_input = last_60_days.reshape(1, look_back, 1)

# EKSEKUSI PREDIKSI
prediction_prob = model.predict(X_input)[0][0]

# Logic Sinyal (Threshold 0.5)
threshold = 0.5
if prediction_prob > threshold:
    signal = "NAIK (BULLISH) 🚀"
    color = "green"
    confidence = prediction_prob * 100
else:
    signal = "TURUN (BEARISH) 🔻"
    color = "red"
    confidence = (1 - prediction_prob) * 100

# --- 5. HITUNG VOLATILITAS ---
# Hitung volatilitas 30 hari terakhir (tetap dari data terbaru)
data['Return'] = data['Close'].pct_change()
volatility_30d = data['Return'].tail(30).std() * 100 

if volatility_30d > 4.0:
    vol_status = "EXTREME RISK ⚡"
    advice = "Pasar sangat liar. Hindari leverage tinggi."
elif volatility_30d > 2.0:
    vol_status = "MEDIUM RISK ⚠️"
    advice = "Pasar cukup aktif. Gunakan Stop Loss yang wajar."
else:
    vol_status = "LOW RISK ✅"
    advice = "Pasar relatif tenang (Sideways/Stabil)."

# --- 6. TAMPILAN DASHBOARD ---
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
    st.metric(label="Volatilitas (30 Hari)", value=f"{volatility_30d:.2f}%", delta=vol_status, delta_color="inverse")
    st.info(advice)

st.divider()

# Visualisasi Harga (Menggunakan Filter Tanggal User)
st.subheader(f"Tren Harga {ticker}")
st.caption(f"Menampilkan data dari {start_user_date} sampai {max_date}")

# Plotting data yang sudah difilter tanggalnya
chart_data = filtered_data.set_index('Date')['Close']
st.line_chart(chart_data)

st.success("✅ Analisis Selesai. Gunakan sebagai referensi pendukung keputusan trading Anda.")