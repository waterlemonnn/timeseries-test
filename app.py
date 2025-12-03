import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# --- 1. KONFIGURASI HALAMAN WEB ---
st.set_page_config(page_title="Crypto Predictor", layout="wide")

st.title("💰 Cryptocurrency Price Prediction App")
st.write("Aplikasi prediksi harga berbasis **LSTM (Deep Learning)**. Dibuat untuk memenuhi Tugas Final Project.")

# Sidebar untuk Input User
st.sidebar.header("Konfigurasi")
ticker = st.sidebar.text_input("Ticker Symbol (Yahoo Finance)", value="BTC-USD")
start_date = st.sidebar.date_input("Mulai Tanggal", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("Sampai Tanggal", value=pd.to_datetime("today"))

# --- 2. FUNGSI LOAD DATA & MODEL ---
@st.cache_data
def load_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        if len(data) == 0:
            return None
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        return None

data_load_state = st.text('Sedang memuat data...')
data = load_data(ticker, start_date, end_date)

if data is None or data.empty:
    st.error(f"Data {ticker} tidak ditemukan atau kosong. Coba ganti ticker symbol (misal: BTC-USD, ETH-USD).")
    st.stop()

data_load_state.text(f'Data {ticker} berhasil dimuat!')

# Tampilkan Data Mentah (Opsional, buat laporan)
st.subheader('Grafik Harga Historis')
# Pastikan kolom Date ada
if 'Date' in data.columns:
    st.line_chart(data.set_index('Date')['Close'])
else:
    st.line_chart(data['Close'])

# --- 3. PERSIAPAN PREDIKSI (PREPROCESSING) ---
st.subheader(f'Analisis Prediksi {ticker}')

# Ambil data Close saja
df_close = data[['Close']]
dataset = df_close.values

# Scaling (0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Load Model yang sudah dilatih
try:
    model = load_model('crypto_model.h5')
except:
    st.error("File 'crypto_model.h5' tidak ditemukan. Pastikan file model ada di folder yang sama!")
    st.stop()

# Siapkan Data Test
look_back = 60

if len(scaled_data) < look_back:
    st.error(f"Data tidak cukup! Butuh minimal {look_back} hari data.")
else:
    # --- A. VALIDASI MODEL (MASA LALU) ---
    test_start_idx = int(len(scaled_data) * 0.8)
    test_data = scaled_data[test_start_idx - look_back: , :]

    x_test = []
    
    for i in range(look_back, len(test_data)):
        x_test.append(test_data[i-look_back:i, 0])
    
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Eksekusi Prediksi Validasi
    prediction = model.predict(x_test)
    prediction = scaler.inverse_transform(prediction)

    # Visualisasi Validasi
    train = data[:test_start_idx]
    valid = data[test_start_idx:].copy() # Pakai copy biar aman
    valid.loc[:, 'Predictions'] = prediction

    fig2, ax2 = plt.subplots(figsize=(16,8))
    ax2.set_title('Validasi Model: Seberapa Akurat AI?', fontsize=16)
    ax2.set_xlabel('Tanggal')
    ax2.set_ylabel('Harga (USD)')
    
    # Plotting
    if 'Date' in train.columns:
        ax2.plot(train['Date'], train['Close'], label='Data Training')
        ax2.plot(valid['Date'], valid['Close'], label='Harga Asli', color='blue')
        ax2.plot(valid['Date'], valid['Predictions'], label='Prediksi AI', color='red')
    else:
        ax2.plot(train.index, train['Close'], label='Data Training')
        ax2.plot(valid.index, valid['Close'], label='Harga Asli', color='blue')
        ax2.plot(valid.index, valid['Predictions'], label='Prediksi AI', color='red')
        
    ax2.legend()
    st.pyplot(fig2)

    # --- B. FORECASTING MASA DEPAN (REAL PREDICTION) ---
    st.divider()
    st.subheader(f"🔮 Prediksi Real-Time: 7 Hari Ke Depan")
    
    # Ambil 60 hari data terakhir real
    last_60_days = scaled_data[-look_back:]
    curr_input = last_60_days.reshape(1, look_back, 1)
    
    future_predictions = []
    
    # Loop 7 hari
    for i in range(7):
        pred = model.predict(curr_input)
        future_predictions.append(pred[0][0])
        # Reshape pred to (1, 1, 1) to match curr_input dimension
        curr_input = np.append(curr_input[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

    # Kembalikan ke USD
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)
    
    # Bikin Tanggal
    if 'Date' in data.columns:
        last_date = data['Date'].iloc[-1]
    else:
        last_date = pd.to_datetime('today')

    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)
    
    # DataFrame Masa Depan
    df_future = pd.DataFrame({
        'Tanggal': future_dates,
        'Prediksi (USD)': future_predictions.flatten()
    })
    
    # Format Currency
    df_future['Formatted'] = df_future['Prediksi (USD)'].apply(lambda x: f"${x:,.2f}")

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("### Tabel Angka")
        st.dataframe(df_future[['Tanggal', 'Formatted']], hide_index=True)
    
    with col2:
        st.write("### Visualisasi Tren")
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        
        # Ambil data terakhir buat konteks
        last_30_days = data.iloc[-30:]
        
        # --- PERBAIKAN ERROR (Make sure it's a scalar) ---
        # Kita ambil nilai terakhir dan paksa jadi float
        # .values.flatten()[-1] memastikan kita ambil 1 angka terakhir meskipun formatnya Series/DataFrame
        last_real_price = float(last_30_days['Close'].values.flatten()[-1])
        
        if 'Date' in last_30_days.columns:
            last_real_date = last_30_days['Date'].values.flatten()[-1]
            dates_for_plot = last_30_days['Date']
        else:
            last_real_date = pd.to_datetime('today')
            dates_for_plot = last_30_days.index

        # Plot Data Asli
        ax3.plot(dates_for_plot, last_30_days['Close'], label='History Terakhir', color='blue')
        
        # Plot Prediksi (Sambungin titik terakhir biar gak putus)
        plot_dates = [last_real_date] + list(future_dates)
        plot_prices = [last_real_price] + list(future_predictions.flatten())
        
        ax3.plot(plot_dates, plot_prices, label='Forecasting Masa Depan', color='green', linestyle='--', marker='o')
        
        ax3.set_title(f'Forecasting {ticker} Minggu Depan')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)

    st.success("✅ Analisis Selesai! Lihat grafik putus-putus hijau di atas untuk prediksi masa depan.")