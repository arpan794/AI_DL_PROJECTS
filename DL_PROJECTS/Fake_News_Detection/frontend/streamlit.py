import streamlit as st
import requests

API_URL = "http://backend:8000/predict"

st.title("Fake News Detection")

text = st.text_area("Enter News Article")

if st.button("Check News"):

    response = requests.post(
        API_URL,
        json={"text": text}
    )

    result = response.json()

    st.success(f"Prediction: {result['prediction']}")