from flask import Flask,render_template,request,session
import pickle
import pandas
from tensorflow import keras
from utils import user_anime_recommendations,selected_anime,user_embedding,personal_recommendation
import numpy as np



animes = pickle.load(open('model/anime.pkl','rb'))
similarity = pickle.load(open('model/similarity.pkl','rb'))
model = keras.models.load_model('model/model.h5')
user_data = pickle.load(open('model/user.pkl','rb'))
user_id = 1583

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
            return render_template('index.html', popular=popular, popular_movie=popular_movie,
                                   children=children, best=best, mf_rating=rec_for_you)
        else:
            # Handle case where no user ID is submitted
            return render_template('landing.html', error="Please enter a valid user ID")

@app.route('/home', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        # If no user ID was submitted on landing page, redirect
        return render_template('landing.html', error="Please submit a user ID first")

    anime = request.form.get("watched")
    if anime:
        sel = selected_anime(anime, animes)
        predictions = user_data_storage['predictions'] 
        watched_anime = user_data_storage['watched_anime']
        ra = user_anime_recommendations(anime, similarity, animes, watched_anime, predictions)
        return render_template('page.html', selected=sel, recommendations=ra)
    else:
        # Handle case where no anime is selected in the POST request
        return render_template('page.html', selected=None, recommendations=[])


if __name__ == "__main__":
    app.run(debug=True)