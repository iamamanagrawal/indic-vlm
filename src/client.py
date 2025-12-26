from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()

class Message(BaseModel):
    role: str
    content: str
    image: str | None

class Request(BaseModel):
    messages: list[Message]

@app.get("/")
async def read_root():
    return {"Hello": "World"}