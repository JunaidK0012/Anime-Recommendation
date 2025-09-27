# 🎬 Anime Recommendation System

## 📌 Description
This project builds an **Anime Recommendation System** using a **hybrid approach** that combines:

- **Matrix Factorization (Collaborative Filtering)**  
- **Content-Based Filtering (TF-IDF + PCA + cosine similarity)**  
- **Hybrid Recommender** blending both approaches  
 
Additionally, **sentiment analysis** is performed on user reviews using an **LSTM model** to enhance the recommendations
The backend is served via **FastAPI** (for recommendation and sentiment analysis).

https://github.com/user-attachments/assets/dc0258b3-66b2-48b0-b50e-3ddd0ae7fc90

---

## 🛠️ Tech Stack  
**Languages & Libraries**  
- Pandas, MLflow, Sqlite, FastAPI, Numpy, Pydantic
- Scikit-learn, Spacy, TensorFlow/Keras (LSTM)  


## 📂 Project Structure
```
anime-recommendation-system/
├── data/                        # Raw and cleaned data (anime.csv, user.csv, processed files)
├── notebook/                    # Jupyter notebooks for experimentation
│   ├── matrix_factorization.ipynb
│   ├── content_based.ipynb
│   └── hybrid_recommendation.ipynb
│
├── models/                      # Saved models (Matrix Factorization, Similarity Matrix, Sentiment LSTM)
│
├── app.py                       # Main Flask application for recommendations
├── model_server.py              # FastAPI sentiment analysis service
├── utils.py                     # Helper functions
├── requirements.txt             # Dependencies
├── setup.py                     # Setup script
└── README.md                    # Project README
```

---

## 🚀 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/JunaidK0012/Anime-Recommendation.git
   cd anime-recommendation-system
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   conda create -p venv python=3.8 -y
   conda activate venv/
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Steps to Run the Project

### 1. Download Data
Download the **anime** and **user** CSV datasets from the provided link.  
Place them inside a new `data/` folder in the project root.

```
anime-recommendation-system/
├── data/
│   ├── jikan_final.csv
│   └── userratings.csv
```

---

### 2. Start MLflow Tracking Server
Run MLflow to track experiments and save models:
```bash
mlflow ui
```
This will start MLflow at `http://127.0.0.1:5000`.

---

### 3. Run the Hybrid Notebook
Execute all cells in `notebook/hybrid_recommendation.ipynb`.  
This will:
- Clean and preprocess datasets  
- Train the **Matrix Factorization model**  
- Build the **Content-Based similarity matrix**  
- Save outputs:
  - `models/model.keras` → trained collaborative filtering model
  - `models/similarity_matrix.pkl` → similarity matrix for content-based filtering
  - `data/cleaned_data.csv` and `data/cleaned_user_data.csv`

---

### 4. Start the Sentiment Analysis Service
Run the FastAPI service to analyze user reviews:
```bash
python model_server.py
```

---

### 5. Start the Recommendation Backend
Run the Flask server for anime recommendations:
```bash
python app.py
```

---

## ✨ Features
- ✅ Data preprocessing & cleaning  
- 📊 Exploratory Data Analysis (EDA)  
- 🔗 Matrix Factorization for collaborative filtering  
- 🎭 Content-Based Filtering with TF-IDF & PCA  
- ⚡ Hybrid Recommendation (blend of CF + CBF)  
- 📝 Sentiment Analysis of reviews (LSTM)  
- 🌐 Flask + FastAPI backend services  

---

## 📌 Usage
Once both services are running:
- **Flask (app.py)** → provides anime recommendations  
- **FastAPI (model_server.py)** → provides sentiment analysis for reviews  

Integrate them in the frontend or API client to deliver personalized anime experiences.  
