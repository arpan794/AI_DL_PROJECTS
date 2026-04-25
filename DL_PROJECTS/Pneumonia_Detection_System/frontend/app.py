import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"

st.title("Pneumonia Detection System")

uploaded_file = st.file_uploader("Upload Chest X-Ray", type=["jpg", "png"])

if uploaded_file is not None:
    files = {"file": uploaded_file.getvalue()}
    response = requests.post(API_URL, files=files)

    if response.status_code == 200:
        result = response.json()

        st.write("Prediction:", result["prediction"])
        st.write("Probability:", result["probability"])