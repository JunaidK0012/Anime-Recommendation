from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import List, Optional
import pickle
import pandas as pd
from tensorflow import keras
from utils import user_anime_recommendations, selected_anime,user_embedding,personal_recommendation,reviews_pred
import requests
import os
import sqlite3
import logging
import mlflow.pyfunc
from fastapi.encoders import jsonable_encoder

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# -----------------------------
# Configs (env variables preferred)
# -----------------------------
DB_PATH = os.getenv("DB_PATH", "data/anime_recommendation.db")
MLFLOW_URI = os.getenv("MLFLOW_URI", "http://127.0.0.1:8080")
MODEL_NAME = os.getenv("MODEL_NAME", "hybrid_anime_recommendation")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
SENTIMENT_API_URL = os.getenv("SENTIMENT_API_URL", "http://127.0.0.1:8001/predict")


# -----------------------------
# Database Connection
# -----------------------------
try:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    logger.info("Connected to database successfully")
except Exception as e:
    logger.error(f"DB connection failed: {e}")
    raise

# -----------------------------
# MLflow Model Loading
# -----------------------------
try:
    mlflow.set_tracking_uri(MLFLOW_URI)
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}" # Get the model version using a model URI
    model = mlflow.pyfunc.load_model(model_uri)
    logger.info(f"Loaded MLflow model: {model_uri}")
except Exception as e:
    logger.error(f"Error loading MLflow model: {e}")
    raise



# -----------------------------
# Similarity Matrix
# -----------------------------
try:
    similarity = pickle.load(open("model/similarity_matrix.pkl", "rb"))
    logger.info("Loaded similarity matrix")
except Exception as e:
    logger.error(f"Error loading similarity matrix: {e}")
    raise


# -----------------------------
# FastAPI Setup
# -----------------------------
app = FastAPI(
    title="Anime recommendation API",
    description="Recommendation system + sentiment analysis using FastAPI",
    version="1.0.0",
)


# Allowing CORS for calling this API from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], #frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache storage (in-memory for now)
user_data_storage = {}
# Load anime data
anime_df = pd.read_sql_query("SELECT * FROM anime", conn)
user_df  = pd.read_sql_query("SELECT * FROM users", conn)
# -----------------------------
# Pydantic Models
# -----------------------------
class UserRequest(BaseModel):
    user_id: int

class AnimeRequest(BaseModel):
    anime: str

class AnimeRecommendationResponse(BaseModel):
    selected: Optional[dict]
    recommendations: List[dict]

class ReviewResponse(BaseModel):
    reviews: list[str]
    sentiments: list[str]

# -----------------------------
# Error Handler
# -----------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error"},
    )


# -----------------------------
# Routes
# -----------------------------
@app.get("/",response_class=HTMLResponse)
def root():
    return "<h1>Anime Recommendation API is running 🚀</h1>"

@app.post("/start")
def start(req: UserRequest):
    user_id = req.user_id
    try:
        # Check if the user exists in user_id_encoded column
        if user_df.empty or user_id not in user_df["user_id_encoded"].values:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found in DB")
        
        predictions, watched_anime = user_embedding(user_df , user_id, model)
        user_data_storage["predictions"] = predictions
        user_data_storage["watched_anime"] = watched_anime

        # Load anime table for recommendations
        anime_df  = pd.read_sql_query("SELECT * FROM anime", conn)

        rec_for_you = personal_recommendation(predictions, anime_df , watched_anime)
        user_data_storage["rec"] = rec_for_you

        return {"message": "User initialized", "user_id": user_id}
    except Exception as e:
        logger.error(f"Error initializing user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error initializing user: {e}")


@app.get("/recommendations")
def get_recommendations():
    if "rec" not in user_data_storage:
        raise HTTPException(status_code=400, detail="User not initialized")
    
    recs = user_data_storage["rec"]
    
    return {"recommendations": jsonable_encoder(recs)}



@app.get("/anime", response_model=AnimeRecommendationResponse)
def anime_recommendations(watched: str = Query(..., description="Name of watched anime")):
    try:

        sel = selected_anime(watched, anime_df)

        predictions = user_data_storage.get("predictions")
        
        watched_anime = user_data_storage.get("watched_anime")

        ra = user_anime_recommendations(watched, similarity, anime_df, watched_anime, predictions)
  
        return {"selected": sel, "recommendations": ra}
    except Exception:
        logger.error(f"Error fetching recommendations for {watched}: {e}")

        raise HTTPException(status_code=404, detail="Anime not found")

@app.get("/reviews/{anime_id}")
def get_reviews(anime_id: int):
    api_url = f"https://api.jikan.moe/v4/anime/{anime_id}/reviews"
    response = requests.get(api_url)

    if response.status_code != 200:
        raise HTTPException(status_code=502, detail="Failed to fetch reviews from Jikan API")

    data = response.json()
    reviews, sentiments = [], []

    if data.get("data"):
        for review_data in data["data"]:
            review = review_data.get("review", "")
            reviews.append(review)

            try:
                senti_resp = requests.post(
                    SENTIMENT_API_URL,
                    json={"text": review},
                )
                if senti_resp.status_code == 200:
                    result = senti_resp.json()
                    sentiments.append(result["label"])
                else:
                    sentiments.append("error")
            except Exception:
                sentiments.append("error")
    else:
        reviews.append("No reviews found")
        sentiments.append("")

    return {"reviews": reviews, "sentiments": sentiments}
@app.get("/genre/{genre_name}")
def get_animes_by_genre(
    genre_name: str,
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(10, ge=1, le=100, description="Number of items to return")
):
    try:
        query = """
        SELECT * FROM anime 
        WHERE genres LIKE ? 
        ORDER BY popularity
        LIMIT ? OFFSET ?
        """
        # Pass 3 params: genre_name, limit, skip
        filtered_anime = pd.read_sql_query(query, conn, params=(f"%{genre_name}%", limit, skip))

        # Ensure JSON serializable
        filtered_anime = filtered_anime.where(pd.notnull(filtered_anime), None)

        result = filtered_anime.to_dict(orient="records")
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Error filtering genre {genre_name}: {e}")

        raise HTTPException(status_code=400, detail=f"Error filtering genre: {e}")
