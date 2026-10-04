import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. सेव किए गए मॉडल्स, स्केलर और फीचर्स को लोड करें
model = joblib.load('telecom_churn_xgb_model.pkl')
scaler = joblib.load('scaler.pkl')
model_features = joblib.load('model_features.pkl')

st.title("📱 Telecom Customer Churn Prediction Web App")
st.write("Enter the customer details below to predict if they will leave the company.")

gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Is Senior Citizen?", ["No", "Yes"])
partner = st.selectbox("Has Partner?", ["No", "Yes"])
dependents = st.selectbox("Has Dependents?", ["No", "Yes"])

st.subheader("Subscription Details")
tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=12)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing?", ["No", "Yes"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

st.subheader("Charges")
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)

if st.button("Predict Churn Status"):
    # एक खाली डेटाफ्रेम बनाएं जो मॉडल के ट्रेनिंग फीचर्स जैसा हो
    input_data = pd.DataFrame(0, index=[0], columns=model_features)
    
    input_data['tenure'] = tenure
    input_data['MonthlyCharges'] = monthly_charges
    input_data['TotalCharges'] = total_charges
    
    if senior_citizen == "Yes": input_data['SeniorCitizen'] = 1
    if gender == "Male": input_data['gender_Male'] = 1
    if partner == "Yes": input_data['Partner_Yes'] = 1
    if dependents == "Yes": input_data['Dependents_Yes'] = 1
    if paperless_billing == "Yes": input_data['PaperlessBilling_Yes'] = 1
    
    if contract == "One year": input_data['Contract_One year'] = 1
    elif contract == "Two year": input_data['Contract_Two year'] = 1
    
    if payment_method == "Credit card": input_data['PaymentMethod_Credit card (automatic)'] = 1
    elif payment_method == "Electronic check": input_data['PaymentMethod_Electronic check'] = 1
    elif payment_method == "Mailed check": input_data['PaymentMethod_Mailed check'] = 1
    
    # न्यूमेरिकल डेटा को स्केल करें (जो हमने Step 4 में सीखा था)
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])
    
    # मॉडल से प्रेडिक्शन करवाएं
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1] 
    
    st.subheader("Result:")
    if prediction == 1:
        st.error(f"🚨 High Risk! This customer is likely to CHURN. Probability: {prediction_proba*100:.2f}%")
    else:
        st.success(f"✅ Safe! This customer is likely to STAY. Probability of leaving: {prediction_proba*100:.2f}%")
