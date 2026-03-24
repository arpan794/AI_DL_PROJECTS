from fastapi import FastAPI, UploadFile, File
from app.model_loader import load_model
from app.utils import preprocess_image

app = FastAPI(title="Medical Image Classification API")

model = load_model()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()
    img = preprocess_image(image_bytes)

    prob = model.predict(img)[0][0]
    prediction = "PNEUMONIA" if prob > 0.5 else "NORMAL"

    return {
        "prediction": prediction,
        "probability": float(prob)
    }