import pickle
import pandas
from tensorflow import keras
import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = pickle.load(open('model/tokenizer.pkl','rb'))
model = keras.models.load_model('model/reviews_sentiment.h5')

def user_embedding(user_data,user_id,model):
    anime_ids = np.array(list(set(user_data.anime_id_encoded)))
    anime_size = anime_ids.shape[0]
    user_ids = np.array([user_id]*anime_size)
    predictions = model.predict([user_ids,anime_ids]).flatten()
    watched_anime_data = user_data[user_data.user_id_encoded == user_id]['anime_id_encoded']
    watched_anime = watched_anime_data.unique()

    return predictions,watched_anime




def personal_recommendation(predictions,data,watched_anime):
    mf_rating_index = predictions.argsort()[-30:][::-1]
    mask = np.isin(mf_rating_index, watched_anime)
    top_unwatched_rec = mf_rating_index[~mask]
    personal_rec = data.iloc[top_unwatched_rec][:14]
    rec_for_you = personal_rec.to_dict('records')
    return rec_for_you







def user_anime_recommendations(anime_name,similarity_matrix,data,watched_anime,prediction):
    
    anime_index = data[data.title == anime_name].index[0]
    cb_similarity = similarity_matrix[anime_index]
    
    ratings = 0.4*cb_similarity + 0.6*prediction
    top_anime_index = ratings.argsort()[-50:][::-1]
    mask = np.isin(top_anime_index, watched_anime) | (top_anime_index == anime_index)
    top_unwatched_anime_index = top_anime_index[~mask]

    recommended_animes = []
    for i in top_unwatched_anime_index[:21]:
        anime_data = data.iloc[i]
        recommended_animes.append({'Name': anime_data.title, 'Image': anime_data.image})
        
    return recommended_animes
    
    


def selected_anime(anime_name,data):
    sel = data[data.title == anime_name]
    return sel.to_dict('records')[0]


def preclean(text):
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'[^A-Za-z]+', ' ', text)
    text = re.sub(r' +', ' ' , text)
    text = text.lower()
    return text

import spacy
nlp = spacy.load('en_core_web_lg')
def clean(text):
    doc = nlp(text)
    return ' '.join(token.lemma_ for token in doc if not token.is_stop and not token.is_oov)

def reviews_pred(text):
    text = preclean(text)
    text = clean(text)
    input_text = tokenizer.texts_to_sequences([text])
    padded_input = pad_sequences(input_text,maxlen=150,padding='post')
    x = int(np.round(model.predict(padded_input)[0]))
    if x==0:
        return 'negative'
    else:
        return 'positive'