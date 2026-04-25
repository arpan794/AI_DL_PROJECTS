Train LSTM:

python training/train_lstm.py

Train BERT:

python training/train_bert.py

Backend:

uvicorn app.main:app --reload

Frontend:

streamlit run frontend/app.py