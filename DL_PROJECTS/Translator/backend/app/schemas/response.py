from pydantic import BaseModel

class TranslateResponse(BaseModel):

    translation: str