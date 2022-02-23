# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 20:34:34 2022

@author: khans
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

#"C:\Users\khans\OneDrive\Desktop\ml-latest-small\ml-latest-small\movies.csv"
movies = pd.read_csv("C:/Users/khans/OneDrive/Desktop/ml-latest-small/ml-latest-small/movies.csv")
ratings = pd.read_csv("C:/Users/khans/OneDrive/Desktop/ml-latest-small/ml-latest-small/ratings.csv")

#print out the first few rows of the dataset
print(movies['title'])
print(ratings.head())

#those two datasets are hard to understand
#we will combine them together to see which user has rated what
fixed_dataset = ratings.pivot(index = 'movieId', columns = 'userId', values = 'rating')
#print(fixed_dataset.head())

#replace the NaN with 0 to make it easier to read
fixed_dataset.fillna(0, inplace = True)
#print(fixed_dataset.head())

#removing unreliable data (noise)

#to qualify a movie, a min of 10 users should have voted it
#to qualify a user, they should vote a min of 50 movies

#put together the number of users who voted and the number of movies that were voted
no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
no_movies_voted = ratings.groupby('userId')['rating'].agg('count')

"""
#visualizing (putting it into a graph)
f,ax = plt.subplots(1,1,figsize=(16,4))
# ratings['rating'].plot(kind='hist')
plt.scatter(no_user_voted.index,no_user_voted,color='red')
plt.axhline(y=10,color='r')
plt.xlabel('MovieId')
plt.ylabel('No. of users voted')
plt.show()

#any movie less than 10 we will disregard
fixed_dataset = fixed_dataset.loc[no_user_voted[no_user_voted > 10].index,:]

#doing this part myself (until print(fixed_dataset))
#visualizing (putting it into a graph)
f,ax = plt.subplots(1,1,figsize=(16,4))
# ratings['rating'].plot(kind='hist')
plt.scatter(no_movies_voted.index,no_movies_voted,color='mediumseagreen')
plt.axhline(y=10,color='r')
plt.xlabel('userId')
plt.ylabel('No. of movies voted')
plt.show()
"""
#any user with less than 50 rated movies we will disregard
fixed_dataset = fixed_dataset.loc[:,no_movies_voted[no_movies_voted > 50].index]

#print(fixed_dataset)

#getting rid of the zeros (sparsity)
csr_data = csr_matrix(fixed_dataset.values)
fixed_dataset.reset_index(inplace = True)

#creating the model itself (knn algorithm)
knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute', n_neighbors = 20, n_jobs = -1)
knn.fit(csr_data)
#knn.kneighbors_graph()

present = 2

#the recommendation function
def get_movie_recommendation(movie_name):
    
    #present = False
    
    n_movies_to_reccomend = 10
    movie_list = movies[movies['title'].str.contains(movie_name)]  
    #print(movie_list)
    if len(movie_list):        
        movie_idx= movie_list.iloc[0]['movieId']
        movie_idx = fixed_dataset[fixed_dataset['movieId'] == movie_idx].index[0]
        distances , indices = knn.kneighbors(csr_data[movie_idx],n_neighbors=n_movies_to_reccomend+1)    
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        recommend_frame = []
        
        present = 2
        print(present)
        
        for val in rec_movie_indices:
            movie_idx = fixed_dataset.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommend_frame.append({'Title':movies.iloc[idx]['title'].values[0],'Distance':val[1]})
        df = pd.DataFrame(recommend_frame,index=range(1,n_movies_to_reccomend+1))
        return df
    else:
        present = 0
        return "No movies found. Please check your input"
    

#creating the website
#import libraries
import numpy as np
from flask import Flask, render_template,request
import flask
import pickle#Initialize the flask App
app = Flask(__name__)
#model = pickle.load(open('model', 'rb'))

@app.route("/", methods = ['GET', 'POST'])
def home():
    if flask.request.method == 'GET':   
        return(flask.render_template("home.html"))
    
    if flask.request.method == 'POST':
        m_name = flask.request.form['moviename']
        m_name = m_name.title()
        movie_list = movies[movies['title'].str.contains(m_name)]  
        #get_movie_recommendation(m_name)
        """
        if present == 2:
            
            return flask.render_template('positive.html')
        else:
            return flask.render_template('negative.html')
        
        """
        if len(movie_list):#m_name not in movie_list:#movies['title']:
        #get_movie_recommendation(m_name)
        #if present:
            print(movie_list)
            print(present)
            #print("NOT FOUND NOT FOUND NOT FOUND")
            
            #returning a dataframe
            final_result = get_movie_recommendation(m_name)
            names = []
            for i in range(len(final_result)):
                #print(get_movie_recommendation(m_name))
                names.append(final_result.iloc[i][0])
                #dates.append(result_final.iloc[i][1])
            
            return flask.render_template('positive.html', movie_names = names, search_name = m_name) #movie_names=get_movie_recommendation(m_name), movie_name = m_name))
       
        else: 
            return flask.render_template('negative.html')#,movie_names=names, search_name = m_name)#movie_date=dates,search_name=m_name)

            #result_final = get_movie_recommendation(m_name)
            #names = []
            #dates = []
            #for i in range(len(result_final)):
                #print(get_movie_recommendation(m_name))
                #names.append(result_final.iloc[i][0])
                #dates.append(result_final.iloc[i][1])
            

if __name__ == '__main__':
    app.run(debug = True)
# @app.route("/predict", methods = ['POST'])
# def predict(): 
#     #For rendering results on HTML GUI
#     str_features = [str(x) for x in request.form.values()]
#     final_features = [np.array(str_features)]
#    # prediction = model.predict(final_features)
    
#     prediction = 
    
#     #output = round(prediction[0], 2) 


    

