from pydantic import BaseModel

class NewsResponse(BaseModel):
    prediction: str