
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# 👇 ADD THIS
from fastapi.middleware.cors import CORSMiddleware

# load model
model = joblib.load("svm_sentiment_model.pkl")

app = FastAPI()

# 👇 ADD CORS HERE (IMPORTANT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# request schema
class TextInput(BaseModel):
    text: str

# routes
@app.get("/")
def home():
    return {"message": "API running 🚀"}

@app.post("/predict")
def predict(data: TextInput):
    prediction = model.predict([data.text])[0]
    return {"prediction": prediction}
