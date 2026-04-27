
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("sentiment_model.pkl")

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "AI API is running 🚀"}

@app.post("/predict")
def predict(data: TextInput):
    prediction = model.predict([data.text])
    return {"prediction": prediction[0]}
