import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open('flat_price_model.pkl', 'rb'))

st.title("🏢 Flat Price Predictor & Buy Recommendation")

# Inputs from user
sqft = st.number_input("Enter Total Square Feet", 500, 5000)
bhk = st.slider("Number of BHK", 1, 10)
bath = st.slider("Number of Bathrooms", 1, 10)
budget = st.number_input("Your Budget (in ₹ Lakhs)", 10, 1000)

# For now, we won't use location encoding manually (you can improve later)
# We'll just take example input vector: sqft, bath, bhk + zero for locations
# In real deployment, we need the full dummy variable list from training

if st.button("Predict & Recommend"):
    # IMPORTANT: Adjust input size as per your model's features
    # For simplicity, let's use only sqft, bath, bhk
    input_data = np.array([sqft, bath, bhk] + [0]*(model.n_features_in_-3)).reshape(1, -1)

    predicted_price = model.predict(input_data)[0]
    st.subheader(f"Predicted Price: ₹ {predicted_price:.2f} Lakhs")

    # Buy/Not Buy Logic
    if predicted_price <= budget:
        st.success("✅ Yes! This flat is within your budget.")
    elif predicted_price <= budget * 1.1:
        st.warning("⚠️ Slightly above budget, consider negotiation.")
    else:
        st.error("❌ No! Overpriced for your budget.")
