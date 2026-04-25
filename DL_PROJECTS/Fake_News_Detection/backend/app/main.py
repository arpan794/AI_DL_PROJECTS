from fastapi import FastAPI
from .schemas.request import NewsRequest
from .schemas.response import NewsResponse
from .model.predictor import predict_news

app = FastAPI(title="Fake News Detection API")

@app.get("/")
def home():
    return {"message": "Fake News Detection API running"}

@app.post("/predict", response_model=NewsResponse)
def predict(request: NewsRequest):

    result = predict_news(request.text)

    return {"prediction": result}