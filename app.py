from flask import Flask,render_template,request,redirect,url_for,jsonify
import pickle
import pandas
from tensorflow import keras
from utils import user_anime_recommendations,selected_anime,user_embedding,personal_recommendation,reviews_pred
import numpy as np
import requests




animes = pickle.load(open('model/anime.pkl','rb'))
similarity = pickle.load(open('model/similarity.pkl','rb'))
model = keras.models.load_model('model/model.h5')
user_data = pickle.load(open('model/user.pkl','rb'))
user_id = 1583

genres = set()
for row in animes['genres']:
    genres.update(row.split(','))


w = animes.sort_values(by = 'score',ascending=False).head(14)
x = animes[animes.type == 'TV'].sort_values(by = 'popularity').head(14)
y = animes[animes.type == 'Movie'].sort_values(by = 'popularity').head(14)
z = animes[(animes.rating == "PG")].sort_values(by = 'popularity').head(14)

best = w.to_dict('records')
popular = x.to_dict('records')
popular_movie = y.to_dict('records')
children = z.to_dict('records')

user_data_storage = {} 


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def start():
    if request.method == 'GET':
        return render_template('landing.html')
    else: 
        user_id = int(request.form.get('user_id'))  # Get user ID from landing page form
        if user_id:
            predictions, watched_anime = user_embedding(user_data, user_id, model)
            user_data_storage['predictions'] = predictions
            user_data_storage['watched_anime'] = watched_anime
            rec_for_you = personal_recommendation(predictions, animes, watched_anime)
            user_data_storage['rec'] = rec_for_you
            return redirect(url_for('predict'))
        else:
            # Handle case where no user ID is submitted
            return render_template('landing.html', error="Please enter a valid user ID")

@app.route('/home')
def predict():
    rec_for_you = user_data_storage['rec']
    # If no user ID was submitted on landing page, redirect
    return render_template('home.html' ,genres = genres, popular=popular, popular_movie=popular_movie,
                                children=children, best=best, mf_rating=rec_for_you)

@app.route('/search', methods=['GET'])
def search():
    try:
        query = request.args.get('term', '')
        if not query:
            return jsonify([])

        # Log the type of 'animes'
        print(f"Type of animes: {type(animes)}")

        # Assuming 'animes' is a DataFrame
        matching_animes = animes[animes['title'].str.contains(query, case=False, na=False)]['title'].tolist()
        return jsonify(matching_animes)
    except Exception as e:
        print(f"Error occurred: {e}")  # Log the error to the console
        return jsonify([]), 500  # Return an empty list and a 500 status code

@app.route("/animes" , methods = ['GET'])
def anime_recommendations():
    anime = request.args.get("watched") 
    try:
        if anime:
            sel = selected_anime(anime, animes)
            predictions = user_data_storage['predictions'] 
            watched_anime = user_data_storage['watched_anime']
            ra = user_anime_recommendations(anime, similarity, animes, watched_anime, predictions)
            return render_template('page.html', selected=sel, recommendations=ra)
        else:
            # Handle case where no anime is selected in the POST request
            return render_template('page.html', selected=None, recommendations=[])
    except:
        return render_template('error.html',error_message="No such anime available")

@app.route("/get_reviews",methods=['GET'])
def get_reviews():
    anime_id = int(request.args.get("anime-id"))
    api_url = f"https://api.jikan.moe/v4/anime/{anime_id}/reviews"
    response = requests.get(api_url)
    data = response.json()
    reviews = []
    sentiments = []
    if data['data']:  # Check if there are reviews
        for review_data in data['data']:
            review = review_data['review']
            senti = reviews_pred(review)  # Assuming reviews_pred is your sentiment analysis function
            reviews.append(review)
            sentiments.append(senti)
    else:
        reviews.append("No reviews found")
        sentiments.append("")
    return jsonify({"reviews": reviews, "sentiments": sentiments})  # Return JSON data








@app.route("/get_animes_by_genre", methods=["GET"])
def get_animes_by_genre():
    if request.method == "GET":
        selected_genre = request.args.get("genre")
        if selected_genre:
            try:
                filtered_anime = animes[animes.genres.str.contains(selected_genre)].sort_values(by='popularity')
                filtered_anime = filtered_anime.to_dict("records")
                return render_template('genre.html',genre = selected_genre,animes = filtered_anime)
            except (KeyError, ValueError) as e:
            # Handle errors gracefully, e.g., log the error and return an informative message
                return render_template('error.html', error_message="An error occurred while processing your request.")
        else:
            # Handle invalid or missing genre selection
            return render_template('error.html', error_message="Invalid genre selection.")



if __name__ == "__main__":
    app.run(debug=True)