import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="TFT Forecast App", layout="wide")

st.title("📈 TFT Forecast Dashboard")
st.write("Upload data, lihat hasil prediksi, dan tampilkan grafik.")

# Load price data
st.subheader("Data Harga Saham (BBCA)")
try:
    df_price = pd.read_csv("bbca.csv")
    st.dataframe(df_price.head())
except:
    st.warning("File bbca.csv tidak ditemukan. Pastikan file ada di repository.")

# Load fundamental data
st.subheader("Data Fundamental")
try:
    df_fund = pd.read_csv("bbca_fundamentals_quarterly_2021_2023.csv")
    st.dataframe(df_fund.head())
except:
    st.warning("File bbca_fundamentals_quarterly_2021_2023.csv tidak ditemukan.")

# Load prediction result (if exists)
st.subheader("Hasil Prediksi TFT (Jika Sudah Ada)")
try:
    df_pred = pd.read_csv("predicted_tft.csv")

    st.success("Prediksi berhasil dimuat!")
    st.dataframe(df_pred.head())

    fig, ax = plt.subplots()
    ax.plot(df_pred["date"], df_pred["prediction"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Prediction")
    ax.set_title("TFT Prediction Graph")

    st.pyplot(fig)

except:
    st.info("Belum ada file 'predicted_tft.csv'. Jalankan notebook di lokal untuk membuatnya.")

st.divider()

st.markdown("""
### Cara Menggunakan
1. Jalankan **START TFT.ipynb di komputer kamu**, bukan di Streamlit.
2. Notebook akan menghasilkan file:  
   **predicted_tft.csv**
3. Upload file itu ke GitHub repository ini.
4. Streamlit Cloud akan otomatis menampilkan grafik dan tabel prediksi.
""")
