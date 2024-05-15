from flask import Flask,render_template,request,redirect,url_for
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

@app.route('/home', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        rec_for_you = user_data_storage['rec']
        # If no user ID was submitted on landing page, redirect
        return render_template('home.html' ,genres = genres, popular=popular, popular_movie=popular_movie,
                                   children=children, best=best, mf_rating=rec_for_you)

    else:
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


@app.route("/get_animes_by_genre", methods=["GET","POST"])
def get_animes_by_genre():
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