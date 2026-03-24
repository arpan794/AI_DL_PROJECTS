from fastapi import FastAPI

from .schemas.request import TranslateRequest
from .inference.translator import translate_text

app = FastAPI(title="English Hindi Translator API")

@app.get("/")
def health():

    return {"status": "running"}


@app.post("/translate")
def translate(req: TranslateRequest):

    result = translate_text(req.text)

    return {"translation": result}