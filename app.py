import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Sistem Peringatan Dini Banjir - XGBoost", 
    page_icon="⛈️", 
    layout="wide"
)

# --- CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #060D1A !important;
    color: #E2E8F0 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stHeader"] { background: #060D1A !important; }
section.main > div { padding: 2rem 2rem 3rem !important; }

h1 { color: #E2E8F0 !important; font-size: 26px !important; font-weight: 700 !important; }
p  { color: #64748B !important; font-size: 13px !important; font-family: 'JetBrains Mono', monospace !important; }
h3 { color: #1E56A0 !important; font-size: 11px !important; font-weight: 600 !important;
     text-transform: uppercase !important; letter-spacing: .12em !important; }
hr { border-color: #0F2040 !important; }

[data-testid="stSelectbox"] label { color: #64748B !important; font-size: 12px !important; }
[data-testid="stSelectbox"] > div > div {
    background: #0C1829 !important; border: 1px solid #0F2A50 !important;
    border-radius: 10px !important; color: #E2E8F0 !important;
}

[data-testid="stInfo"] {
    background: #0C1829 !important; border: 1px solid #0F2A50 !important;
    border-radius: 10px !important; color: #7DD3FC !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important;
}

[data-testid="stButton"] > button {
    background: #1C3F7A !important; color: #E2E8F0 !important;
    border: 1px solid #2563EB !important; border-radius: 10px !important;
    font-weight: 600 !important; font-size: 14px !important;
    padding: 12px 0 !important; transition: all .2s !important;
}
[data-testid="stButton"] > button:hover {
    background: #2563EB !important;
    box-shadow: 0 0 20px rgba(37,99,235,.35) !important;
    transform: translateY(-1px) !important;
}

[data-testid="stMetric"] {
    background: #0C1829; border: 1px solid #0F2A50;
    border-radius: 12px; padding: 16px 20px; text-align: center;
}
[data-testid="stMetricLabel"] p { color: #64748B !important; font-size: 11px !important; }
[data-testid="stMetricValue"]   {
    color: #7DD3FC !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 36px !important; font-weight: 700 !important;
}

[data-testid="stNotification"] { border-radius: 12px !important; }
div[data-baseweb="notification"][kind="positive"] {
    background: rgba(5,150,105,.1) !important;
    border: 1px solid rgba(5,150,105,.35) !important;
    color: #34D399 !important; font-weight: 600 !important;
}
div[data-baseweb="notification"][kind="negative"] {
    background: rgba(220,38,38,.1) !important;
    border: 1px solid rgba(220,38,38,.45) !important;
    color: #F87171 !important; font-weight: 600 !important;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100% { border-color: rgba(220,38,38,.45); }
    50%      { border-color: rgba(220,38,38,.9); box-shadow: 0 0 14px rgba(220,38,38,.2); }
}

.footer {
    text-align: center; font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: #1C3057;
    padding-top: 18px; border-top: 1px solid #0F2040;
    margin-top: 28px; line-height: 1.8;
}
</style>
""", unsafe_allow_html=True)

# --- Konstanta & Database ---
BEST_THRESH = 0.531 
DATA_WILAYAH = {
    "Jakarta Pusat":   {"lat": -6.18, "lon": 106.83, "elevasi": 15.0},
    "Jakarta Barat":   {"lat": -6.16, "lon": 106.75, "elevasi": 8.0},
    "Jakarta Selatan": {"lat": -6.26, "lon": 106.81, "elevasi": 45.0},
    "Jakarta Timur":   {"lat": -6.22, "lon": 106.89, "elevasi": 30.0},
    "Jakarta Utara":   {"lat": -6.13, "lon": 106.89, "elevasi": 2.0}
}

# --- Filter kecamatan per kota ---
KECAMATAN_PER_KOTA = {
    "Jakarta Pusat":   ["Gambir","Sawah Besar","Kemayoran","Senen","Cempaka Putih","Menteng","Tanah Abang","Johar Baru"],
    "Jakarta Barat":   ["Tamansari","Tambora","Palmerah","Grogol Petamburan","Cengkareng","Kalideres","Kembangan","Kebon Jeruk"],
    "Jakarta Selatan": ["Tebet","Setiabudi","Mampang Prapatan","Pancoran","Pasar Minggu","Jagakarsa","Pesanggrahan","Cilandak","Kebayoran Baru","Kebayoran Lama"],
    "Jakarta Timur":   ["Matraman","Pulo Gadung","Jatinegara","Duren Sawit","Kramat Jati","Pasar Rebo","Ciracas","Cipayung","Cakung","Makasar"],
    "Jakarta Utara":   ["Penjaringan","Pademangan","Tanjung Priok","Koja","Kelapa Gading","Cilincing"],
}

# --- Load Model & Encoder ---
@st.cache_resource
def load_xgboost_resources():
    with open('models/xgboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/le_kota.pkl', 'rb') as f:
        le_kota = pickle.load(f)
    with open('models/le_kecamatan.pkl', 'rb') as f:
        le_kecamatan = pickle.load(f)
    return model, le_kota, le_kecamatan

# --- Fungsi Ambil Data Cuaca Real-Time ---
def get_live_weather(lat, lon):
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)

    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,rain,wind_speed_10m"
        f"&timezone=Asia%2FJakarta"
    )
    response = session.get(url, timeout=15, verify=True)
    response.raise_for_status()
    current = response.json()['current']
    return {
        'suhu': current['temperature_2m'],
        'kelembapan': current['relative_humidity_2m'],
        'curah_hujan': current['rain'],
        'kecepatan_angin': current['wind_speed_10m']
    }

# --- Main Logic ---
model, le_kota, le_kecamatan = load_xgboost_resources()

st.title("⛈️ Sistem Informasi & Peringatan Dini Banjir")
st.markdown("Model: **XGBoost Classifier** | Threshold Optimasi: **0.531**")
st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📍 Pilih Wilayah")
    pilihan_kota = st.selectbox("Pilih Kota/Wilayah:", list(DATA_WILAYAH.keys()))
    pilihan_kecamatan = st.selectbox("Pilih Kecamatan:", KECAMATAN_PER_KOTA[pilihan_kota])
    
    lat, lon = DATA_WILAYAH[pilihan_kota]["lat"], DATA_WILAYAH[pilihan_kota]["lon"]
    elevasi = DATA_WILAYAH[pilihan_kota]["elevasi"]
    st.info(f"🌐 **Koordinat:** {lat}, {lon} | **Elevasi:** {elevasi} mdpl")

with col2:
    st.subheader("📊 Prediksi Risiko")
    if st.button("🚀 Jalankan Prediksi", use_container_width=True):
        with st.spinner('Menghubungi stasiun cuaca dan menganalisis parameter...'):
            try:
                # 1. Ambil Data Real-Time
                cuaca = get_live_weather(lat, lon)
                
                # 2. Tampilkan Transparansi Data
                st.info(f"✨ **Data Kondisi Cuaca Terkini Berhasil Diterima:**\n"
                        f"* 🌡️ Suhu Udara: {cuaca['suhu']} °C\n"
                        f"* 💧 Kelembapan Udara: {cuaca['kelembapan']} %\n"
                        f"* 🌧️ Intensitas Hujan: {cuaca['curah_hujan']} mm/jam\n"
                        f"* 💨 Kecepatan Angin: {cuaca['kecepatan_angin']} km/jam")

                # 3. Persiapan Input Model
                input_df = pd.DataFrame([{
                    'kota_encoded': le_kota.transform([pilihan_kota])[0],
                    'kecamatan_encoded': le_kecamatan.transform([pilihan_kecamatan])[0],
                    'garis_lintang': lat,
                    'garis_bujur': lon,
                    'durasi_hujan_harian': cuaca['curah_hujan'] * 24,
                    'evapotranspirasi_potensial_standar': 4.5,
                    'kecepatan_angin_maksimum': cuaca['kecepatan_angin'] * 1.2,
                    'kelembapan_tanah_lapisan_atas': cuaca['kelembapan'],
                    'kelembapan_tanah_lapisan__dalam': cuaca['kelembapan'] - 5,
                    'elevasi_topografi': elevasi,
                    'bulan': pd.Timestamp.now().month, 
                    'hari': pd.Timestamp.now().day,
                    'debit_sungai': 15.5
                }])

                # 4. Prediksi dengan Threshold Optimal (0.531)
                prob = model.predict_proba(input_df)[0][1]
                prediksi = 1 if prob >= BEST_THRESH else 0
                
                st.divider()
                st.metric(label="Probabilitas Risiko Banjir Saat Ini", value=f"{prob * 100:.2f} %")
                
                if prediksi == 1:
                    st.error(f"🚨 **PERINGATAN: Wilayah {pilihan_kecamatan} SIAGA BANJIR!**")
                else:
                    st.success(f"✅ **KONDISI AMAN: Wilayah {pilihan_kecamatan} terpantau kondusif.**")
                
            except Exception as e:
                st.error(f"Gagal memproses prediksi: {e}")

# --- Footer ---
st.markdown("""
<div class="footer">
  SIPD BANJIR JAKARTA &nbsp;·&nbsp; Project Based Learning Data Analyst &nbsp;·&nbsp; PPKD Jakarta Selatan &nbsp;·&nbsp; 2026<br>
  XGBoost Classifier &nbsp;·&nbsp; Open-Meteo API &nbsp;·&nbsp; 42 Kecamatan &nbsp;·&nbsp; 98.574 baris &nbsp;·&nbsp; 2020–2026
</div>
""", unsafe_allow_html=True)
