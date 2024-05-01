import pandas

import pickle
animes = pickle.load(open('anime_list.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))


def recommend(anime):
    index = animes[animes['Name'] == anime].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key= lambda x:x[1])
    for i in distances[1:6]:
        print(animes.iloc[i[0]].Name)