import streamlit as st
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

st.set_page_config(page_title="TFT Forecast App", layout="wide")
st.title("📈 TFT Real-Time Forecast Dashboard")

# -----------------------------
# 1. LOAD DATA
# -----------------------------

@st.cache_data
def load_price():
    return pd.read_csv("bbca.csv")

@st.cache_data
def load_fundamental():
    return pd.read_csv("bbca_fundamentals_quarterly_2021_2023.csv")

df_price = load_price()
df_fund  = load_fundamental()

st.subheader("📊 Data Harga BBCA")
st.dataframe(df_price.head())

st.subheader("🏦 Data Fundamental BBCA")
st.dataframe(df_fund.head())

# -----------------------------
# 2. LOAD TRAINED TFT MODEL
# -----------------------------

class SimpleTFT(torch.nn.Module):
    def __init__(self, input_size, hidden_size=32, output_size=1):
        super().__init__()
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.l2(self.relu(self.l1(x)))

@st.cache_resource
def load_model():
    model = SimpleTFT(input_size=5)
    model.load_state_dict(torch.load("tft_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# -----------------------------
# 3. USER SELECT FORECAST DATE
# -----------------------------

st.subheader("📅 Pilih Tanggal untuk Prediksi")

last_date = pd.to_datetime(df_price["Date"]).max()
default_pred_date = last_date + timedelta(days=1)

pred_date = st.date_input(
    "Tanggal prediksi:",
    value=default_pred_date,
    min_value=default_pred_date,
)

# -----------------------------
# 4. GENERATE FORECAST
# -----------------------------

def prepare_input(df):
    """Ambil 5 fitur terakhir untuk prediksi"""
    last_row = df[["Close", "High", "Low", "Open", "Volume"]].tail(1)
    return torch.tensor(last_row.values, dtype=torch.float32)

if st.button("🔮 Generate Forecast"):

    x = prepare_input(df_price)
    with torch.no_grad():
        y_pred = model(x).item()

    st.success(f"Prediksi harga BBCA untuk {pred_date}: **{y_pred:,.2f}**")

    # plot
    fig, ax = plt.subplots()
    ax.plot(df_price["Date"].tail(50), df_price["Close"].tail(50), label="History")
    ax.scatter(pred_date, y_pred, color="red", label="Forecast")
    ax.legend()
    st.pyplot(fig)

