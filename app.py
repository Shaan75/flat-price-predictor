import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open("flat_price_model.pkl", "rb"))

# Load location names from model input features
feature_names = model.feature_names_in_
locations = [f for f in feature_names if f not in ['total_sqft', 'bath', 'bhk']]

# App settings
st.set_page_config(page_title="Flat Price Predictor", page_icon="🏠", layout="centered")

# Title
st.markdown("<h1 style='text-align:center;'>🏠 Flat Price Predictor & Smart Recommendation</h1>", unsafe_allow_html=True)
st.markdown("### 💡 Estimate your flat price and know if it's worth buying.")

st.markdown("---")

# Sidebar inputs
st.sidebar.header("📋 Enter Flat Details")
location = st.sidebar.selectbox("📍 Location", locations)
sqft = st.sidebar.slider("📏 Total Area (sqft)", 500, 5000, step=50)
bhk = st.sidebar.selectbox("🛏️ Bedrooms (BHK)", [1, 2, 3, 4, 5])
bath = st.sidebar.selectbox("🚿 Bathrooms", [1, 2, 3, 4])
budget = st.sidebar.number_input("💰 Your Budget (₹ Lakhs)", 10, 1000, step=5)

# Predict Button
if st.sidebar.button("🔍 Predict & Recommend"):
    # Create input vector
    input_data = np.zeros(len(feature_names))
    input_data[0] = sqft
    input_data[1] = bath
    input_data[2] = bhk

    # Set location = 1 if selected
    if location in feature_names:
        loc_index = np.where(feature_names == location)[0][0]
        input_data[loc_index] = 1

    # Prediction
    predicted_price = model.predict([input_data])[0]

    # Display Results
    st.markdown("---")
    st.markdown("## 🧮 Prediction Result")
    st.success(f"📌 Predicted Price: **₹ {predicted_price:.2f} Lakhs**")

    # Buy/Not Buy Recommendation
    st.markdown("## 🧠 Buy Recommendation")
    if predicted_price <= budget:
        st.markdown("### ✅ Go for it! Within your budget.")
        st.balloons()
    elif predicted_price <= budget * 1.1:
        st.markdown("### ⚠️ Slightly above budget. Consider negotiating.")
    else:
        st.markdown("### ❌ Too expensive! Overpriced for your budget.")

    # Show Price Difference
    price_diff = predicted_price - budget
    st.info(f"Difference: ₹ {price_diff:.2f} Lakhs")

    # Price Comparison Chart
    st.markdown("### 📊 Price Comparison")
    fig, ax = plt.subplots()
    bars = ax.bar(["Your Budget", "Predicted Price"], [budget, predicted_price], color=["#4CAF50", "#FF5733"])
    ax.bar_label(bars)
    st.pyplot(fig)
