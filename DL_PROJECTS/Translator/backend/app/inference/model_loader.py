from transformers import pipeline

MODEL_PATH = "model"

translator = pipeline(
    "translation_en_to_hi",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH
)