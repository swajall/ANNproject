import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle

model =  tf.keras.models.load_model('model.h5')

with open ('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f) 

with open ('label_encoder_gender.pkl', 'rb') as f:
    label_encoder = pickle.load(f)  

with open ('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder = pickle.load(f)

#stremlit app
st.title("Customer Churn Prediction")

#User input
geography = st.selectbox("Geography", onehot_encoder.categories_[0])
gender = st.selectbox("Gender",[0,1])
age = st.slider("Age", 18,92)
balance = st.number_input("Balance", 0.0, 250898.09)
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0,10)
num_of_products = st.number_input("Number of Products", 1,4)
has_cr_card = st.selectbox("Has Credit Card", [0,1])
is_active_member = st.selectbox("Is Active Member", [0,1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
})

geo_encoded = onehot_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder.get_feature_names_out(['Geography'])) 

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

input_data_scaled = scaler.transform(input_data)
prediction_prob = model.predict(input_data_scaled)
if st.button("Predict"):
    if prediction_prob[0][0] > 0.5:
        st.write("The customer is likely to churn.")
        st.write(f"Churn Probability: {prediction_prob[0][0]:.2f}") 
    else:
        st.write("The customer is unlikely to churn.")
        st.write(f"Churn Probability: {prediction_prob[0][0]:.2f}") 