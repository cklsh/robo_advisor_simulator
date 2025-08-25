import streamlit as st
import pandas as pd
import joblib
import sys
import os

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '../utils')))

from preprocessing import clean_and_encode_data

model_path = os.path.join(BASE_DIR, 'model.pkl')
encoders_path = os.path.join(BASE_DIR, 'encoders.pkl')
data_path = os.path.join(BASE_DIR, '..', 'data', 'bank.csv')

model = joblib.load(model_path)
encoders = joblib.load(encoders_path)

st.title("Robo-Advisor Risk Profiler")
st.markdown("Fill in your information to get a simulated investment risk profile.")
st.markdown("by Github: @cklsh")

age = st.slider("Age", 18, 80, 30)
job = st.selectbox("Job", [
    'admin', 'technician', 'services', 'management', 'entrepreneur',
    'self-employed', 'blue-collar', 'retired', 'unemployed', 'student', 'housemaid'
])
education = st.selectbox("Education Level", ['tertiary', 'secondary', 'primary'])
housing = st.radio("Do you have a housing loan?", ('yes', 'no'))
loan = st.radio("Do you have a personal loan?", ('yes', 'no'))

if st.button("Predict Risk Profile"):
    user_input_df = pd.DataFrame([{
        'age': age,
        'job': job,
        'education': education,
        'housing': housing,
        'loan': loan
    }])

    # Load original data for consistent columns
    original_df = pd.read_csv(data_path, sep=',')

    combined_df = pd.concat([original_df, user_input_df], ignore_index=True)

    # Encode with loaded encoders (no save here)
    processed_df, _ = clean_and_encode_data(combined_df, encoders=encoders)

    processed_input = processed_df.tail(1)

    prediction_encoded = model.predict(processed_input)[0]

    # Decode prediction label
    risk_profile_decoder = encoders['risk_profile']
    prediction_label = risk_profile_decoder.inverse_transform([prediction_encoded])[0]

    if prediction_label == 'Conservative':
        allocation = "20% Stocks, 70% Bonds, 10% Gold"
    elif prediction_label == 'Moderate':
        allocation = "50% Stocks, 40% Bonds, 10% Gold"
    else:
        allocation = "80% Stocks, 10% Bonds, 10% Crypto/Gold"

    st.success(f"Predicted Risk Profile: **{prediction_label}**")
    st.info(f"Suggested Asset Allocation: {allocation}")

    
