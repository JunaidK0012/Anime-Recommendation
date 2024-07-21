# Anime Recommendation System

## Description
This project aims to build an anime recommendation system using a hybrid approach that combines matrix factorization and content-based filtering. Additionally, sentiment analysis is performed on user reviews using an LSTM model to enhance the recommendations. The backend is built using Flask.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Features](#features)

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/YourUsername/anime-recommendation-system.git
    ```
2. Navigate to the project directory:
    ```bash
    cd anime-recommendation-system
    ```
3. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
4. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Run the setup script to prepare the environment:
    ```bash
    python setup.py
    ```
2. Train the models:
    - Matrix Factorization Model:
        ```bash
        python src/train_matrix_factorization.py
        ```
    - Content-Based Filtering Model:
        ```bash
        python src/train_content_based.py
        ```
    - Sentiment Analysis Model:
        ```bash
        python src/train_sentiment_analysis.py
        ```
3. Start the Flask server:
    ```bash
    python app.py
    ```
4. Access the web application by navigating to `http://127.0.0.1:5000/` in your web browser.

## Project Structure
```
anime-recommendation-system/
│
├── data/ # Contains the dataset
│ ├── raw/ # Raw data
│ └── processed/ # Processed data for training
│
├── models/ # Trained models and model-related scripts
│ ├── matrix_factorization/ # Matrix Factorization model and scripts
│ ├── content_based/ # Content-Based Filtering model and scripts
│ └── sentiment_analysis/ # Sentiment Analysis model and scripts
│
├── notebook/ # Jupyter notebooks for analysis and experimentation
│ ├── EDA.ipynb # Exploratory Data Analysis notebook
│ ├── matrix_factorization.ipynb # Matrix Factorization experiments
│ ├── content_based.ipynb # Content-Based Filtering experiments
│ └── sentiment_analysis.ipynb # Sentiment Analysis experiments
│
├── src/ # Source code for the project
│ ├── data_preprocessing.py # Scripts for data preprocessing
│ ├── train_matrix_factorization.py # Script to train the matrix factorization model
│ ├── train_content_based.py # Script to train the content-based filtering model
│ ├── train_sentiment_analysis.py # Script to train the sentiment analysis model
│ └── recommend.py # Script to generate recommendations
│
├── templates/ # HTML templates for the Flask web application
│
├── static/ # Static files for the Flask web application (CSS, JS, images)
│
├── tests/ # Unit tests
│
├── .gitignore # Git ignore file
├── Dockerfile # Dockerfile for containerization
├── README.md # Project README file
├── app.py # Main Flask application script
├── requirements.txt # Python dependencies
├── setup.py # Setup script
└── config.py # Configuration file
```


## Features
- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Matrix Factorization model for collaborative filtering
- Content-Based Filtering model
- Sentiment Analysis using LSTM
- Flask backend for serving recommendations
- Web interface for user interaction
- Docker support for containerization
