import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("flat_price_model.pkl", "rb"))

# App settings
st.set_page_config(page_title="Flat Price Predictor", page_icon="🏢", layout="centered")

# Title
st.markdown("<h1 style='text-align: center;'>🏠 Flat Price Predictor & Recommendation</h1>", unsafe_allow_html=True)
st.markdown("### 💡 Get a price estimate and smart recommendation whether to buy or skip.")

st.markdown("---")

# Sidebar for input
st.sidebar.header("📋 Input Flat Details")

# Inputs
sqft = st.sidebar.slider("📏 Total Area (sqft)", 500, 5000, step=50)
bhk = st.sidebar.selectbox("🛏️ Bedrooms (BHK)", [1, 2, 3, 4, 5])
bath = st.sidebar.selectbox("🚿 Bathrooms", [1, 2, 3, 4])
budget = st.sidebar.number_input("💰 Your Budget (in ₹ Lakhs)", 10, 1000, step=5)

# Predict button
if st.sidebar.button("🔍 Predict Price & Recommend"):
    # Dummy location input (as we're using only sqft, bath, bhk)
    features = np.array([sqft, bath, bhk] + [0]*(model.n_features_in_ - 3)).reshape(1, -1)

    predicted_price = model.predict(features)[0]

    # Result Card
    st.markdown("---")
    st.markdown("## 🧮 Prediction Result")
    st.success(f"📌 **Predicted Price:** ₹ {predicted_price:.2f} Lakhs")

    # Recommendation Logic
    st.markdown("## 🧠 Buy Recommendation")

    if predicted_price <= budget:
        st.markdown("### ✅ **Go for it!** The flat is within your budget.")
        st.balloons()
    elif predicted_price <= budget * 1.1:
        st.markdown("### ⚠️ **Slightly above budget.** Consider negotiating.")
    else:
        st.markdown("### ❌ **Too expensive!** Overpriced for your budget.")

    # Price comparison
    price_diff = predicted_price - budget
    st.markdown(f"**Price difference:** ₹ {price_diff:.2f} Lakhs")

    # (Optional Future Features)
    st.markdown("🔧 Want location input, nearby suggestions, or charts? Add that in future upgrades.")
