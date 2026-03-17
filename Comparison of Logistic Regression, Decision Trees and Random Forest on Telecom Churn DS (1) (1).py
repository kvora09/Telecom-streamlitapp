import streamlit as st
import pickle
import pandas as pd

# Load model and features
model = pickle.load(open("churn_model.pkl", "rb"))
features = pickle.load(open("model_features.pkl", "rb"))

st.title("📊 Telecom Churn Prediction App")

st.write("Enter customer details below:")

# Create input fields dynamically
input_data = {}

for feature in features:
    input_data[feature] = st.number_input(f"{feature}", value=0.0)

# Predict button
if st.button("Predict"):

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to churn (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Customer is likely to stay (Probability: {probability:.2f})")
