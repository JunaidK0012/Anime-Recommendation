from flask import Flask,render_template,request
import pickle
import pandas


app = Flask(__name__)

animes = pickle.load(open('anime_list.pkl','rb'))

anime_list = animes['Name'].values
x = animes.sort_values(by = 'Popularity').head()

recommendations_list = x.to_dict('records')


@app.route('/', methods = ['GET','POST'] )
def predict():
    if request.method == "GET":
        return render_template('index.html',recommendations = recommendations_list)
    




if __name__ == "__main__":
    app.run(debug=True)