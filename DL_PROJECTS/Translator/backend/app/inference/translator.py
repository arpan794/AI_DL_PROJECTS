from .model_loader import translator

def translate_text(text):

    result = translator(text)[0]

    return result["translation_text"]