import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="House Price Predictor")

# Load trained model
model = pickle.load(open("house_model.pkl", "rb"))

st.title("🏠 House Price Prediction System")
st.write("Enter property details below to predict sale price")

st.markdown("---")

# Input fields
area = st.number_input("Above Ground Living Area (sq ft)", 500, 5000, 1500)
bedrooms = st.number_input("Number of Bedrooms", 1, 10, 3)
bathrooms = st.number_input("Number of Bathrooms", 1, 5, 2)
quality = st.slider("Overall Quality (1 - 10)", 1, 10, 5)

if st.button("Predict Price"):
    input_data = np.array([[area, bedrooms, bathrooms, quality]])
    prediction = model.predict(input_data)
    st.success(f"Predicted House Price: ${int(prediction[0]):,}")
