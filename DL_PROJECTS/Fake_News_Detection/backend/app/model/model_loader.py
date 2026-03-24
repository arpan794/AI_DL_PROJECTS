import pickle
from tensorflow.keras.models import load_model

MODEL_PATH = "saved_models/fake_news_model.h5"
TOKENIZER_PATH = "saved_models/tokenizer.pkl"

model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)
