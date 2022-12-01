import numpy as np
import pandas as pd

# reads the csv file
movies=pd.read_csv("tmdb_5000_movies.csv")
credits=pd.read_csv("tmdb_5000_credits.csv")

# both movies and credits gets merged and stored in movies
movies=movies.merge(credits,on="title")

# just the columns required for recommendation are used and all other columns get dropped
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

# ast library is used to convert the string to list of dictionaries by traversing
import ast
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 

# drops rows containing null values
movies.dropna(inplace=True)

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# convert3 function gives tyhe top three cast in every movie
def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L 

movies['cast'] = movies['cast'].apply(convert3)

# fetch dorector fetches the director through traversing whole dictionary for every movie
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 

movies['crew'] = movies['crew'].apply(fetch_director)

# collapse function returns the names of each cast,crew,genres which dont have space in between in each word of respective names
def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1

movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)


movies['overview'] = movies['overview'].apply(lambda x:x.split()) #split function used to return the list of words in string

# tags do contain union of overview, genres,keywords, cast and crew
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new = movies.drop(columns=['overview','genres','keywords','cast','crew'])
new['tags'] = new['tags'].apply(lambda x: " ".join(x))#it gives a  string formed from each word in list before

# porterstemmer from nltk library used to remove duplicate words in different verb forms like dance,dancing,dancer,danced.. and replaces with a single word danc 
import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

# function to use portstemmer for tag of each movie
def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)
new["tags"]=new["tags"].apply(stem)

# We are converting string to vector such that we can find similarity between movies through by distances between each vector
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')#max_features gives maximum number of repeated words and stop words ignores in,of,the,.. from sentences in tags

vector = cv.fit_transform(new['tags']).toarray()

# we are calculating the distances between vectors through the cosine angle between them
from sklearn.metrics.pairwise import cosine_similarity
# similarity do consists of similarness of a movie with each other movie
similarity = cosine_similarity(vector)

# recommend function prints the top five movies which has more simmilarit with given movie
def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])#enumerate function gives list of tuples which do stores index of each movie similarness with other which uses after sorting
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)

# pickle used to import data frames to .pkl file to local systems
import pickle
pickle.dump(new,open('movie_list.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))
    

