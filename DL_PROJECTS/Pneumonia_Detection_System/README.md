Train:

python training/transfer_training.py

Backend:

uvicorn app.main:app --reload

Frontend:

streamlit run frontend/app.py