import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import shap

# Load model
model = pickle.load(open("flat_price_model.pkl", "rb"))

# Load feature names
feature_names = model.feature_names_in_
locations = [f for f in feature_names if f not in ['total_sqft', 'bath', 'bhk']]

# Load cleaned data for alternatives
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_bangalore_data.csv")  # Upload this file

df = load_data()

# App settings
st.set_page_config(page_title="Flat Price Predictor", page_icon="🏠", layout="centered")

st.markdown("<h1 style='text-align:center;'>🏠 Flat Price Predictor & Explainable AI</h1>", unsafe_allow_html=True)
st.markdown("### 💡 Predict, Compare, and Understand Why!")

st.sidebar.header("📋 Enter Flat Details")
location = st.sidebar.selectbox("📍 Location", locations)
sqft = st.sidebar.slider("📏 Total Area (sqft)", 500, 5000, step=50)
bhk = st.sidebar.selectbox("🛏️ Bedrooms (BHK)", [1, 2, 3, 4, 5])
bath = st.sidebar.selectbox("🚿 Bathrooms", [1, 2, 3, 4])
budget = st.sidebar.number_input("💰 Your Budget (₹ Lakhs)", 10, 1000, step=5)

if st.sidebar.button("🔍 Predict & Explain"):
    # Create input vector
    input_data = np.zeros(len(feature_names))
    input_data[0] = sqft
    input_data[1] = bath
    input_data[2] = bhk
    if location in feature_names:
        loc_index = np.where(feature_names == location)[0][0]
        input_data[loc_index] = 1

    # Prediction
    predicted_price = model.predict([input_data])[0]

    st.markdown("## 🧮 Predicted Price")
    st.success(f"📌 ₹ {predicted_price:.2f} Lakhs")

    # Recommendation
    st.markdown("## 🧠 Buy Recommendation")
    if predicted_price <= budget:
        st.markdown("✅ Within budget! Great deal.")
    elif predicted_price <= budget * 1.1:
        st.markdown("⚠️ Slightly above budget. Consider negotiating.")
    else:
        st.markdown("❌ Overpriced. Check alternatives below.")

    # Price Comparison Chart
    st.markdown("### 📊 Price Comparison")
    fig, ax = plt.subplots()
    ax.bar(["Budget", "Predicted"], [budget, predicted_price], color=["#4CAF50", "#FF5733"])
    st.pyplot(fig)

    # Alternatives
    st.markdown("## 🔍 Cheaper Alternatives")
    similar = df[(df['bhk'] == bhk) & (df['total_sqft'] >= sqft*0.8) & (df['total_sqft'] <= sqft*1.2)]
    if not similar.empty:
        cheaper = similar.groupby('location')['price'].mean().sort_values().head(3)
        st.table(cheaper.reset_index().rename(columns={'location': 'Location', 'price': 'Avg Price (Lakhs)'}))
    else:
        st.warning("No alternatives found.")

    # SHAP Explanation
    st.markdown("## 🤖 Why this price? (Explainable AI)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values([input_data])
    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.initjs()
    st.pyplot(shap.force_plot(explainer.expected_value, shap_values[0], feature_names))
# ---- Price Trend Dashboard ----
st.markdown("---")
st.markdown("## 📊 Price Trend Dashboard")

# Option to select visualization
viz_option = st.selectbox("Choose a Visualization", ["Avg Price by Location", "Price Distribution by BHK", "Location-wise Price vs Sqft"])

if viz_option == "Avg Price by Location":
    avg_prices = df.groupby('location')['price'].mean().sort_values(ascending=False).head(10)
    st.markdown("### 🔝 Top 10 Locations by Avg Price")
    fig, ax = plt.subplots(figsize=(8, 4))
    avg_prices.plot(kind='bar', color="#FF5733", ax=ax)
    ax.set_ylabel("Price (Lakhs)")
    st.pyplot(fig)

elif viz_option == "Price Distribution by BHK":
    st.markdown("### 🛏️ Price Distribution by BHK")
    fig, ax = plt.subplots()
    df.boxplot(column='price', by='bhk', ax=ax, grid=False)
    ax.set_ylabel("Price (Lakhs)")
    st.pyplot(fig)

elif viz_option == "Location-wise Price vs Sqft":
    loc = st.selectbox("Select Location", df['location'].unique())
    subset = df[df['location'] == loc]
    fig, ax = plt.subplots()
    ax.scatter(subset['total_sqft'], subset['price'], color="green")
    ax.set_xlabel("Total Sqft")
    ax.set_ylabel("Price (Lakhs)")
    ax.set_title(f"Price vs Sqft in {loc}")
    st.pyplot(fig)
