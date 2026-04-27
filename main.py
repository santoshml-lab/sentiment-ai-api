from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware

# load model
model = joblib.load("svm_sentiment_model.pkl")

app = FastAPI()

# CORS (important for frontend)
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

# home route
@app.get("/")
def home():
    return {"message": "API running 🚀"}

# prediction route
@app.post("/predict")
def predict(data: TextInput):
    try:
        prediction = model.predict([data.text])[0]

        # confidence (optional but powerful)
        try:
            score = model.decision_function([data.text])[0]
            confidence = round(abs(score), 2)
        except:
            confidence = None

        return {
            "input": data.text,
            "prediction": prediction,
            "confidence": confidence
        }

    except Exception as e:
        return {"error": str(e)}
