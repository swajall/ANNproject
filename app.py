import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle
import json
import h5py


def _replace_batch_shape(config):
    if isinstance(config, dict):
        updated = {}
        for key, value in config.items():
            if key == "batch_shape":
                updated["batch_input_shape"] = value
            else:
                updated[key] = _replace_batch_shape(value)
        return updated
    if isinstance(config, list):
        return [_replace_batch_shape(item) for item in config]
    return config


def load_model_compat(model_path):
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except TypeError as exc:
        if "batch_shape" not in str(exc):
            raise

        with h5py.File(model_path, "r") as h5_file:
            raw_config = h5_file.attrs["model_config"]
            if isinstance(raw_config, bytes):
                raw_config = raw_config.decode("utf-8")

        model_config = json.loads(raw_config)
        model_config = _replace_batch_shape(model_config)
        model = tf.keras.models.model_from_json(json.dumps(model_config))
        model.load_weights(model_path)
        return model


model = load_model_compat("model.h5")

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
