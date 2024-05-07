from flask import Flask,render_template,request
import pickle
import pandas


app = Flask(__name__)

animes = pickle.load(open('anime.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))


def recommend(anime):
    index = animes[animes.title == anime].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_animes = []
    for i in distances[1:11]:
        anime_data = animes.iloc[i[0]]
        recommended_animes.append({'Name': anime_data.title, 'Image': anime_data.image , 'Synopsis': anime_data.synopsis})
    return recommended_animes

def selected(anime):
    sel = animes[animes.title == anime]
    return sel.to_dict('records')[0]

x = animes.sort_values(by = 'popularity').head(10)
y = animes[animes.type == 'Movie'].sort_values(by = 'popularity').head(10)

popular = x.to_dict('records')
popular_movie = y.to_dict('records')


@app.route('/', methods=['GET', 'POST'])
def predict():
  if request.method == "GET":
    return render_template('index.html', popular=popular,popular_movie = popular_movie)

  else:
    anime = request.form.get("watched")
    if anime:  # Check if 'watched' parameter exists in POST request
      sel = selected(anime)
      ra = recommend(anime)
      return render_template('page.html', selected=sel, recommendations=ra)
    else:
      # Handle case where no anime is selected in the POST request
      return render_template('page.html', selected=None, recommendations=[])




if __name__ == "__main__":
    app.run(debug=True)