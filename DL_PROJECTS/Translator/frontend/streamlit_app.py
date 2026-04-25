import streamlit as st
import requests

API_URL = "http://backend:8000/translate"

st.title("English → Hindi Translator")

text = st.text_area("Enter English Sentence")

if st.button("Translate"):

    response = requests.post(
        API_URL,
        json={"text": text}
    )

    result = response.json()

    st.success(result["translation"])