from tensorflow.keras.preprocessing.sequence import pad_sequences
from .model_loader import model, tokenizer
from ..utils.preprocessing import clean_text

MAX_LEN = 200

def predict_news(text):

    text = clean_text(text)

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)

    prediction = model.predict(padded)[0][0]

    if prediction > 0.5:
        return "FAKE"
    else:
        return "REAL"