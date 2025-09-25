from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import pickle
import tensorflow as tf
import numpy as np
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI(title='Sentiment Model Server')

MODEL_PATH = os.getenv("SENTIMENT_MODEL_PATH","model/reviews_sentiment.h5")
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "model/tokenizer.pkl")
MAX_LEN = int(os.getenv("MAX_LEN", "150"))

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    score: float

@app.on_event("startup")
def startup_load():
    global model,tokenizer 
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        raise

    if os.path.exists(TOKENIZER_PATH):
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
    else:
        return("Tokenizer not found")
    
def preprocess_texts(text):
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'[^A-Za-z]+', ' ', text)
    text = re.sub(r' +', ' ' , text)
    text = text.lower()

    input_text = tokenizer.texts_to_sequences([text])
    return pad_sequences(input_text,maxlen=MAX_LEN,padding='post')

    
@app.post("/predict",response_model = PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503,detail="Model not loaded")
    try:
  
        x = preprocess_texts(req.text)

        score = model.predict(x)[0]
        label = "positive" if score >= 0.5 else "negative"
        return {"label": label, "score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction failed")
        
