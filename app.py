import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open("flat_price_model.pkl", "rb"))

# Load feature names
feature_names = model.feature_names_in_
locations = [f for f in feature_names if f not in ['total_sqft', 'bath', 'bhk']]

# Load original data for alternatives
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_bangalore_data.csv")  # Upload this file too

df = load_data()

# App settings
st.set_page_config(page_title="Flat Price Predictor", page_icon="🏠", layout="centered")

# Title
st.markdown("<h1 style='text-align:center;'>🏠 Flat Price Predictor & Smart Recommendation</h1>", unsafe_allow_html=True)
st.markdown("### 💡 Estimate your flat price, get recommendations & alternatives.")

st.sidebar.header("📋 Enter Flat Details")
location = st.sidebar.selectbox("📍 Location", locations)
sqft = st.sidebar.slider("📏 Total Area (sqft)", 500, 5000, step=50)
bhk = st.sidebar.selectbox("🛏️ Bedrooms (BHK)", [1, 2, 3, 4, 5])
bath = st.sidebar.selectbox("🚿 Bathrooms", [1, 2, 3, 4])
budget = st.sidebar.number_input("💰 Your Budget (₹ Lakhs)", 10, 1000, step=5)

if st.sidebar.button("🔍 Predict & Recommend"):
    # Input Vector
    input_data = np.zeros(len(feature_names))
    input_data[0] = sqft
    input_data[1] = bath
    input_data[2] = bhk

    if location in feature_names:
        loc_index = np.where(feature_names == location)[0][0]
        input_data[loc_index] = 1

    # Prediction
    predicted_price = model.predict([input_data])[0]

    st.markdown("---")
    st.markdown("## 🧮 Prediction Result")
    st.success(f"📌 Predicted Price: **₹ {predicted_price:.2f} Lakhs**")

    # Buy/Not Buy Recommendation
    st.markdown("## 🧠 Buy Recommendation")
    if predicted_price <= budget:
        st.markdown("### ✅ Within budget! Great deal.")
        st.balloons()
    elif predicted_price <= budget * 1.1:
        st.markdown("### ⚠️ Slightly above budget. Negotiate!")
    else:
        st.markdown("### ❌ Overpriced. Look for alternatives.")

    # Price Comparison Chart
    st.markdown("### 📊 Price Comparison")
    fig, ax = plt.subplots()
    bars = ax.bar(["Budget", "Predicted"], [budget, predicted_price], color=["#4CAF50", "#FF5733"])
    ax.bar_label(bars)
    st.pyplot(fig)

    # Nearby Alternatives Feature
    st.markdown("## 🔍 Cheaper Alternatives")
    similar = df[(df['bhk'] == bhk) & (df['total_sqft'] >= sqft*0.8) & (df['total_sqft'] <= sqft*1.2)]
    if not similar.empty:
        cheaper = similar.groupby('location')['price'].mean().sort_values().head(3)
        st.table(cheaper.reset_index().rename(columns={'location': 'Location', 'price': 'Avg Price (Lakhs)'}))
    else:
        st.warning("No alternatives found for this configuration.")
