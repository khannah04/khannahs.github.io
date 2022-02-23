# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 20:22:11 2022

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


#the recommendation function
def get_movie_recommendation(movie_name):
    n_movies_to_reccomend = 10
    movie_list = movies[movies['title'].str.contains(movie_name)]  
    if len(movie_list):        
        movie_idx= movie_list.iloc[0]['movieId']
        movie_idx = fixed_dataset[fixed_dataset['movieId'] == movie_idx].index[0]
        distances , indices = knn.kneighbors(csr_data[movie_idx],n_neighbors=n_movies_to_reccomend+1)    
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        recommend_frame = []
        for val in rec_movie_indices:
            movie_idx = fixed_dataset.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommend_frame.append({'Title':movies.iloc[idx]['title'].values[0],'Distance':val[1]})
        df = pd.DataFrame(recommend_frame,index=range(1,n_movies_to_reccomend+1))
        return df
    else:
        return "No movies found. Please check your input"
    
def getMovies(): 
    return movies
    

print(get_movie_recommendation("Spider-Man"))

"""    
import pickle

filename = 'model'
outfile = open(filename, 'wb')

pickle.dump(knn, outfile)
outfile.close()
        

#model = pickle.load(open(filename, 'rb'))"""
    
